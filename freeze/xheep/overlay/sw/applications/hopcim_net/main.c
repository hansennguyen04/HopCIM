// HopCIM R6a: the MLPerf Tiny int8 DS-CNN on the CIM endpoint, CPU-orchestrated (baseline B1 for
// the network). The firmware builds the 61 network tiles (8..68) from compact weights, then per
// inference runs 7,501 CIM passes: the CPU gathers each input vector (activation - zero point,
// padding taps 0), starts the pass and requantises the results with TFLite's integer arithmetic
// (model/dscnn_int.py). Checks the int8 logits of test clips bit-exactly against the golden model
// and reports cycles per layer type. Simulation runs 1 clip, the board all NV_N.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "x-heep.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"
#include "netvec.h"

#ifndef N_RUN                                     // a variant app may fix it (hopcim_net1)
#ifdef TARGET_SIM
#define N_RUN 1
#else
#define N_RUN NV_N
#endif
#endif

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }

// ---------------------------------------------------------------- TFLite requantisation
static int32_t srdhm(int32_t a, int32_t b) {          // SaturatingRoundingDoublingHighMul
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t r = (int64_t)a * b;
    r += r >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)(r >= 0 ? r >> 31 : -((-r) >> 31));   // division by 2^31 toward zero
}

static int32_t rdbp(int32_t x, int e) {                // RoundingDivideByPOT
    int32_t mask = (int32_t)((1u << e) - 1u);
    int32_t rem = x & mask, thr = (mask >> 1) + (x < 0);
    return (x >> e) + (rem > thr);
}

// bias, MultiplyByQuantizedMultiplier (double rounding), output zero point, int8 clamp
static inline int8_t requant(int32_t acc, int L, int c, int32_t zout) {
    int sh = DS_SH[L][c];
    acc += DS_BIAS[L][c];
    int32_t v = sh > 0 ? srdhm(acc * (1 << sh), DS_QM[L][c]) : rdbp(srdhm(acc, DS_QM[L][c]), -sh);
    v += zout;
    return (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
}

// ---------------------------------------------------------------- tiles
static int8_t tbuf[64][16] __attribute__((aligned(4)));

static void put_tile(unsigned t) { cim_load_tile(t, &tbuf[0][0]); }

static void load_net_tiles(void) {
    for (int j = 0; j < 4; j++) {                       // conv1: rows 4 fy + fx, column = channel 16 j + n
        for (int r = 0; r < 64; r++)
            for (int n = 0; n < 16; n++) tbuf[r][n] = r < 40 ? DS_W_CONV1[16 * j + n][r] : 0;
        put_tile(DS_T_CONV1 + j);
    }
    for (int l = 0; l < 4; l++) {
        for (int g = 0; g < 10; g++) {                  // depthwise: block diagonal, 7 channels per tile, tap-major
            for (int r = 0; r < 64; r++)
                for (int n = 0; n < 16; n++) tbuf[r][n] = 0;
            for (int n = 0; n < 7 && 7 * g + n < 64; n++)
                for (int k = 0; k < 9; k++) tbuf[7 * k + n][n] = DS_W_DW[l][7 * g + n][k];
            put_tile(DS_T_DW(l) + g);
        }
        for (int j = 0; j < 4; j++) {                   // pointwise: rows = input channel
            for (int r = 0; r < 64; r++)
                for (int n = 0; n < 16; n++) tbuf[r][n] = DS_W_PW[l][16 * j + n][r];
            put_tile(DS_T_PW(l) + j);
        }
    }
    for (int r = 0; r < 64; r++)                        // fully connected: rows = input, 12 columns
        for (int n = 0; n < 16; n++) tbuf[r][n] = n < 12 ? DS_W_FC[n][r] : 0;
    put_tile(DS_T_FC);
}

// ---------------------------------------------------------------- one inference
static volatile uint32_t *const X = &CIM_REG(CIM_X(0));
static int8_t act_a[25][5][64], act_b[25][5][64];
enum { C_CONV1, C_DW, C_PW, C_POOL, C_FC, C_N };
static uint32_t ccyc[C_N];

static inline int32_t res(int n) { return (int32_t)CIM_REG(CIM_RES_LO(n)); }   // |acc| < 2^22

static void infer(const int8_t feat[49][10], int8_t logits[12]) {
    uint32_t t = cyc(), t1;
    for (int oy = 0; oy < 25; oy++)                     // conv1 10 x 4, stride 2, SAME (top 4, left 1)
        for (int ox = 0; ox < 5; ox++) {
            for (int fy = 0; fy < 10; fy++)
                for (int fx = 0; fx < 4; fx++) {
                    int iy = 2 * oy - 4 + fy, ix = 2 * ox - 1 + fx;
                    X[4 * fy + fx] = (iy >= 0 && iy < 49 && ix >= 0 && ix < 10) ? (uint32_t)(feat[iy][ix] - DS_ZIN0) : 0u;
                }
            for (int j = 0; j < 4; j++) {
                cim_go(DS_T_CONV1 + j, DS_XB, 0);
                cim_sync();
                for (int n = 0; n < 16; n++) act_a[oy][ox][16 * j + n] = requant(res(n), 0, 16 * j + n, DS_ZACT);
            }
        }
    t1 = cyc(); ccyc[C_CONV1] += t1 - t; t = t1;
    for (int l = 0; l < 4; l++) {
        for (int oy = 0; oy < 25; oy++)                 // depthwise 3 x 3, SAME
            for (int ox = 0; ox < 5; ox++)
                for (int g = 0; g < 10; g++) {
                    int nc = 64 - 7 * g < 7 ? 64 - 7 * g : 7;
                    for (int n = 0; n < nc; n++)
                        for (int fy = 0; fy < 3; fy++)
                            for (int fx = 0; fx < 3; fx++) {
                                int iy = oy - 1 + fy, ix = ox - 1 + fx;
                                X[7 * (3 * fy + fx) + n] = (iy >= 0 && iy < 25 && ix >= 0 && ix < 5)
                                                             ? (uint32_t)(act_a[iy][ix][7 * g + n] - DS_ZACT) : 0u;
                            }
                    cim_go(DS_T_DW(l) + g, DS_XB, 0);
                    cim_sync();
                    for (int n = 0; n < nc; n++)
                        act_b[oy][ox][7 * g + n] = requant(res(n), 1 + 2 * l, 7 * g + n, DS_ZACT);
                }
        t1 = cyc(); ccyc[C_DW] += t1 - t; t = t1;
        for (int oy = 0; oy < 25; oy++)                 // pointwise 1 x 1
            for (int ox = 0; ox < 5; ox++) {
                for (int i = 0; i < 64; i++) X[i] = (uint32_t)(act_b[oy][ox][i] - DS_ZACT);
                for (int j = 0; j < 4; j++) {
                    cim_go(DS_T_PW(l) + j, DS_XB, 0);
                    cim_sync();
                    for (int n = 0; n < 16; n++) act_a[oy][ox][16 * j + n] = requant(res(n), 2 + 2 * l, 16 * j + n, DS_ZACT);
                }
            }
        t1 = cyc(); ccyc[C_PW] += t1 - t; t = t1;
    }
    int8_t pool[64];                                    // average pool 25 x 5, TFLite rounding
    for (int c = 0; c < 64; c++) {
        int32_t acc = 0;
        for (int y = 0; y < 25; y++)
            for (int x = 0; x < 5; x++) acc += act_a[y][x][c];
        int32_t q = acc > 0 ? (acc + 62) / 125 : (acc - 62) / 125;
        pool[c] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
    t1 = cyc(); ccyc[C_POOL] += t1 - t; t = t1;
    for (int i = 0; i < 64; i++) X[i] = (uint32_t)(pool[i] - DS_ZACT);
    cim_go(DS_T_FC, DS_XB, 0);
    cim_sync();
    for (int n = 0; n < 12; n++) logits[n] = requant(res(n), 9, n, DS_ZFC);
    ccyc[C_FC] += cyc() - t;
}

int main(void) {
    static const char *const CN[C_N] = {"conv1", "depthwise x4", "pointwise x4", "avg pool", "fc"};
    int bad = 0, correct = 0, agree = 0;
    cim_init();
    uint32_t t = cyc();
    load_net_tiles();
    uint32_t c_load = cyc() - t;
    for (int k = 0; k < N_RUN; k++) {
        int8_t y[12];
        uint32_t t0 = cyc();
        infer(NV_FEAT[k], y);
        uint32_t dt = cyc() - t0;
        int am = 0;
        for (int n = 0; n < 12; n++) {
            bad += y[n] != NV_LOGITS[k][n];
            if (y[n] > y[am]) am = n;
        }
        int ref = 0;
        for (int n = 1; n < 12; n++) if (NV_LOGITS[k][n] > NV_LOGITS[k][ref]) ref = n;
        agree += am == ref;
        correct += am == NV_LABEL[k];
        printf("clip %d: label %d predicted %d, %lu cycles\n", k, NV_LABEL[k], am, (unsigned long)dt);
    }
    uint32_t tot = 0;
    for (int c = 0; c < C_N; c++) tot += ccyc[c];
    printf("tiles: 61 network tiles built and loaded in %lu cycles\n", (unsigned long)c_load);
    for (int c = 0; c < C_N; c++) printf("%-13s %9lu cyc/inference\n", CN[c], (unsigned long)(ccyc[c] / N_RUN));
    printf("total         %9lu cyc/inference, 7501 CIM passes\n", (unsigned long)(tot / N_RUN));
    printf("HopCIM net: %d clips, logit mismatches %d, prediction = golden %d, correct %d -> %s\n", N_RUN, bad, agree,
           correct, bad ? "FAIL" : "PASS");
    printf("done\n");
    return bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
