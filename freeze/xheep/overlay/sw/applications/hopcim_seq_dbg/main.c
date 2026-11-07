// HopCIM R6b board debug: localises a sequencer mismatch. Runs netvec clip 0 with stop-after-layer
// (conv1, then depthwise 0), reads every output back through the ACT window and compares it with
// the golden layers (dbgvec.h); recomputes conv1 at two positions through the CPU path (CPU-gathered
// X, CPU passes, CPU requantisation) to separate gather, engine and requantisation.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"
#include "../hopcim_net/netvec.h"
#include "dbgvec.h"

#define NFR 49
#define NCH 10

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


// ---------------------------------------------------------------- sequencer parameters
static int load_params(void) {
    for (int L = 0; L < 10; L++)
        for (int c = 0; c < 64; c++) {
            if (DS_SH[L][c] > 0) return -1;            // the sequencer implements right shifts only
            CIM_REG(CIM_PARAM(64 * L + c, 0)) = (uint32_t)DS_BIAS[L][c];
            CIM_REG(CIM_PARAM(64 * L + c, 1)) = (uint32_t)DS_QM[L][c];
            CIM_REG(CIM_PARAM(64 * L + c, 2)) = (uint32_t)(-DS_SH[L][c]);
        }
    CIM_REG(CIM_SEQ_ZP) = (uint32_t)(uint8_t)DS_ZIN0 | (uint32_t)(uint8_t)DS_ZACT << 8 | (uint32_t)(uint8_t)DS_ZFC << 16;
    return 0;
}

// FEAT ring row r, words 0..2: int8 features 4w .. 4w+3 of one frame
static void write_frame(unsigned row, const int8_t *f) {
    for (int w = 0; w < 3; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4 && 4 * w + b < NCH; b++) v |= (uint32_t)(uint8_t)f[4 * w + b] << (8 * b);
        CIM_REG(CIM_FEAT(row, w)) = v;
    }
}

static void run_stop(unsigned stop) {
    cim_seq_go(stop, 0, DS_T_CONV1);
    while (!(CIM_REG(CIM_SEQ_STATUS) & CIM_SEQ_DONE)) {
    }
    CIM_REG(CIM_SEQ_STATUS) = CIM_SEQ_DONE;
}

static void compare_layer(int L) {
    int bad = 0, shown = 0, badpos = 0, lane[7] = {0}, edge = 0;
    for (int p = 0; p < 125; p++) {
        int pb = 0;
        for (int c = 0; c < 64; c++) {
            int8_t v = (int8_t)CIM_REG(CIM_ACT(L & 1, p, c)), e = DBG_ACT[L][64 * p + c];
            if (v != e) {
                bad++; pb = 1; lane[c % 7]++;
                if (shown < 10) { printf("  L%d p%d c%d: got %d exp %d\n", L, p, c, v, e); shown++; }
            }
        }
        badpos += pb;
        if (pb && (p / 5 == 0 || p / 5 == 24 || p % 5 == 0 || p % 5 == 4)) edge++;
    }
    printf("L%d: %d/8000 wrong, %d/125 positions (%d on the border), per lane %d %d %d %d %d %d %d\n", L, bad,
           badpos, edge, lane[0], lane[1], lane[2], lane[3], lane[4], lane[5], lane[6]);
}

// conv1 at position (oy, ox) through the CPU path
static void cpu_conv1(int oy, int ox) {
    volatile uint32_t *X = &CIM_REG(CIM_X(0));
    int32_t acc[64];
    for (int r = 0; r < 64; r++) {
        int fy = r / 4, fx = r % 4, iy = 2 * oy - 4 + fy, ix = 2 * ox - 1 + fx;
        X[r] = (uint32_t)((r < 40 && iy >= 0 && iy < NFR && ix >= 0 && ix < NCH) ? NV_FEAT[0][iy][ix] - DS_ZIN0 : 0);
    }
    for (int j = 0; j < 4; j++) {
        cim_go(DS_T_CONV1 + j, DS_XB, 0);
        cim_sync();
        for (int n = 0; n < 16; n++) acc[16 * j + n] = (int32_t)cim_result(n);
    }
    int p = 5 * oy + ox, bad_cpu = 0, bad_seq = 0;
    for (int c = 0; c < 64; c++) {
        int8_t y = requant(acc[c], 0, c, DS_ZACT), e = DBG_ACT[0][64 * p + c];
        int8_t s = (int8_t)CIM_REG(CIM_ACT(0, p, c));
        bad_cpu += y != e; bad_seq += s != e;
        if (c < 3) printf("  p%d c%d: acc %ld, cpu %d, seq %d, golden %d\n", p, c, (long)acc[c], y, s, e);
    }
    printf("conv1 p%d via CPU path: %d/64 wrong; sequencer: %d/64 wrong\n", p, bad_cpu, bad_seq);
}

int main(void) {
    cim_init();
    load_net_tiles();
    if (load_params()) { printf("left shift\n"); return EXIT_FAILURE; }
    for (int r = 0; r < NFR; r++) write_frame((unsigned)r, NV_FEAT[0][r]);
    printf("SEQ_ZP 0x%08lx, status 0x%08lx\n", (unsigned long)CIM_REG(CIM_SEQ_ZP), (unsigned long)CIM_REG(CIM_SEQ_STATUS));
    run_stop(1);
    printf("after conv1: status 0x%08lx\n", (unsigned long)CIM_REG(CIM_SEQ_STATUS));
    compare_layer(0);
    cpu_conv1(12, 2);
    cpu_conv1(0, 0);
    run_stop(2);
    compare_layer(1);
    run_stop(0);
    printf("full run: status 0x%08lx, logits", (unsigned long)CIM_REG(CIM_SEQ_STATUS));
    for (int n = 0; n < 12; n++) printf(" %d", (int)(int8_t)CIM_REG(CIM_LOGIT(n)));
    printf(" | golden");
    for (int n = 0; n < 12; n++) printf(" %d", NV_LOGITS[0][n]);
    printf("\n");
    return EXIT_SUCCESS;
}
