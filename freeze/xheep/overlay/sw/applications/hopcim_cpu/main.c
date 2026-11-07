// HopCIM evaluation baseline: the MLPerf Tiny int8 DS-CNN on the CPU only (no CIM), plain C with
// the integer semantics of the TFLite reference kernels (model/dscnn_int.py): SAME padding by
// skipping taps, per-channel MultiplyByQuantizedMultiplier, the int8 average pool. Checks the
// logits bit-exactly against the golden model and reports cycles per layer type and per inference.
// N_CLIP clips (default 2).

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "x-heep.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"
#include "../hopcim_net/netvec.h"

#ifndef N_CLIP
#define N_CLIP 2
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


static int8_t a0[25][5][64], a1[25][5][64];
static int8_t pooled[64];

int main(void) {
    int bad = 0;
    uint32_t t_conv1 = 0, t_dw = 0, t_pw = 0, t_pool = 0, t_fc = 0;
    for (int k = 0; k < N_CLIP; k++) {
        const int8_t (*f)[10] = NV_FEAT[k];
        uint32_t c0 = cyc();
        for (int oy = 0; oy < 25; oy++)                 // conv1 10x4 / 2, SAME: top 4, left 1
            for (int ox = 0; ox < 5; ox++)
                for (int oc = 0; oc < 64; oc++) {
                    int32_t acc = 0;
                    for (int fy = 0; fy < 10; fy++) {
                        int iy = 2 * oy - 4 + fy;
                        if (iy < 0 || iy >= 49) continue;
                        for (int fx = 0; fx < 4; fx++) {
                            int ix = 2 * ox - 1 + fx;
                            if (ix < 0 || ix >= 10) continue;
                            acc += (int32_t)DS_W_CONV1[oc][4 * fy + fx] * (f[iy][ix] - DS_ZIN0);
                        }
                    }
                    a0[oy][ox][oc] = requant(acc, 0, oc, DS_ZACT);
                }
        uint32_t c1 = cyc();
        t_conv1 += c1 - c0;
        for (int l = 0; l < 4; l++) {
            uint32_t d0 = cyc();
            for (int oy = 0; oy < 25; oy++)             // depthwise 3x3, SAME
                for (int ox = 0; ox < 5; ox++)
                    for (int c = 0; c < 64; c++) {
                        int32_t acc = 0;
                        for (int fy = 0; fy < 3; fy++) {
                            int iy = oy - 1 + fy;
                            if (iy < 0 || iy >= 25) continue;
                            for (int fx = 0; fx < 3; fx++) {
                                int ix = ox - 1 + fx;
                                if (ix < 0 || ix >= 5) continue;
                                acc += (int32_t)DS_W_DW[l][c][3 * fy + fx] * (a0[iy][ix][c] - DS_ZACT);
                            }
                        }
                        a1[oy][ox][c] = requant(acc, 1 + 2 * l, c, DS_ZACT);
                    }
            uint32_t d1 = cyc();
            for (int oy = 0; oy < 25; oy++)             // pointwise
                for (int ox = 0; ox < 5; ox++)
                    for (int oc = 0; oc < 64; oc++) {
                        int32_t acc = 0;
                        for (int ic = 0; ic < 64; ic++) acc += (int32_t)DS_W_PW[l][oc][ic] * (a1[oy][ox][ic] - DS_ZACT);
                        a0[oy][ox][oc] = requant(acc, 2 + 2 * l, oc, DS_ZACT);
                    }
            uint32_t d2 = cyc();
            t_dw += d1 - d0;
            t_pw += d2 - d1;
        }
        uint32_t p0 = cyc();
        for (int c = 0; c < 64; c++) {                  // average pool over 25 x 5 (reference rounding)
            int32_t s = 0;
            for (int p = 0; p < 125; p++) s += a0[p / 5][p % 5][c];
            int32_t q = s > 0 ? (s + 62) / 125 : (s - 62) / 125;
            pooled[c] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
        uint32_t p1 = cyc();
        int ok = 1;
        for (int n = 0; n < 12; n++) {                  // fully connected
            int32_t acc = 0;
            for (int c = 0; c < 64; c++) acc += (int32_t)DS_W_FC[n][c] * (pooled[c] - DS_ZACT);
            ok &= requant(acc, 9, n, DS_ZFC) == NV_LOGITS[k][n];
        }
        uint32_t p2 = cyc();
        t_pool += p1 - p0;
        t_fc += p2 - p1;
        bad += !ok;
        printf("clip %d: logits %s, %u cycles\n", k, ok ? "exact" : "WRONG", (unsigned)(p2 - c0));
    }
    uint32_t tot = t_conv1 + t_dw + t_pw + t_pool + t_fc;
    printf("per inference (mean of %d): conv1 %u, depthwise %u, pointwise %u, pool %u, fc %u, total %u cycles\n", N_CLIP,
           (unsigned)(t_conv1 / N_CLIP), (unsigned)(t_dw / N_CLIP), (unsigned)(t_pw / N_CLIP), (unsigned)(t_pool / N_CLIP),
           (unsigned)(t_fc / N_CLIP), (unsigned)(tot / N_CLIP));
    printf("HopCIM CPU-only network: %d clips, %d wrong -> %s\n", N_CLIP, bad, bad ? "FAIL" : "PASS");
    return bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
