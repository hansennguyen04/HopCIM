// HopCIM R4b: the integer CIM-MFCC front-end (hopcim/model/mfcc_int.py) with every matrix step on
// the CIM endpoint and the epilogue (window, block exponent, twiddles, magnitude, log, normalisation,
// requantisation) on the CPU. Tile layout: hopcim/model/cim_map.py. Checks lraw of every frame and
// the int8 features of a test clip bit-exactly against the golden model; reports cycles per stage.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/mfcc_tables.h"
#include "testvec.h"

#define WIN   480
#define HOP   320
#define NFR   49
#define NMEL  40
#define NMFCC 10
#define LRAW_ZERO INT32_MIN                         // exact zero (mel == 0 or a silent frame)

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }

// round-half-up arithmetic right shifts (s > 0), as rshift() in the model
static inline int32_t rsh32(int32_t x, int s) { return (x + (1 << (s - 1))) >> s; }
static inline int64_t rsh64(int64_t x, int s) { return (x + ((int64_t)1 << (s - 1))) >> s; }

static uint32_t isqrt64(uint64_t v) {
    uint64_t r = 0, bit = (uint64_t)1 << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) { v -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return (uint32_t)r;
}

static int32_t log2_q8(uint64_t v) {               // v >= 1
    int e = 63 - __builtin_clzll(v);
    uint32_t frac = (uint32_t)((e >= 6 ? v >> (e - 6) : v << (6 - e)) & 63u);
    return e * 256 + LOG2_LUT_Q8[frac];
}

static int32_t log2fix16(uint32_t v) {              // floor(log2(v) * 2^16), 1 <= v < 2^31
    int e = 31 - __builtin_clz(v);
    uint64_t m = (uint64_t)v << (30 - e);
    int32_t f = 0;
    for (int i = 0; i < 16; i++) {
        m = (m * m) >> 30;
        f <<= 1;
        if (m >= ((uint64_t)1 << 31)) { m >>= 1; f |= 1; }
    }
    return e * 65536 + f;
}

// ---------------------------------------------------------------- frame path
enum { S_WIN, S_F1, S_TW, S_F3, S_MAG, S_MEL, S_LOG, S_FIN, S_N };
static const char *const SNAME[S_N] = {"window+exp", "FFT step 1", "twiddles", "FFT step 3", "magnitude", "mel", "log",
                                       "norm+DCT+q"};
static uint32_t scyc[S_N];

static int32_t xb[512];
static int32_t Ar[16][32], Ai[16][32], Br[16][32], Bi[16][32];
static int32_t sre[128], sim_[128];
static uint32_t mag[128];
static int32_t lraw[NFR][NMEL];
static int8_t feat[NFR][NMFCC];

static volatile uint32_t *const X = &CIM_REG(CIM_X(0));

static void frame_lraw(const int16_t *x, int32_t *out) {
    uint32_t t = cyc(), t1;
    int32_t peak = 0;
    for (int i = 0; i < WIN; i++) {
        int32_t v = (int32_t)x[i] * HANN_Q15[i];
        xb[i] = v;
        int32_t a = v < 0 ? -v : v;
        if (a > peak) peak = a;
    }
    for (int i = WIN; i < 512; i++) xb[i] = 0;
    if (peak == 0) {
        for (int m = 0; m < NMEL; m++) out[m] = LRAW_ZERO;
        scyc[S_WIN] += cyc() - t;
        return;
    }
    int sh = 31 - __builtin_clz((uint32_t)peak) - 14;
    if (sh > 0) for (int i = 0; i < WIN; i++) xb[i] = rsh32(xb[i], sh);
    else if (sh < 0) for (int i = 0; i < WIN; i++) xb[i] = xb[i] << -sh;
    t1 = cyc(); scyc[S_WIN] += t1 - t; t = t1;

    // FFT step 1: one pass per n2 (x[n1] = xb[32 n1 + n2]); mirrored outputs negated before rounding
    for (int n2 = 0; n2 < 32; n2++) {
        for (int r = 0; r < 16; r++) X[r] = (uint32_t)xb[32 * r + n2];
        cim_go(T_F1, B_F1, 0);
        cim_sync();
        int32_t s[16];
        for (int n = 0; n < 16; n++) s[n] = (int32_t)CIM_REG(CIM_RES_LO(n));   // |s| < 2^31
        for (int k = 0; k <= 8; k++) Ar[k][n2] = rsh32(s[k], 11);
        Ai[0][n2] = 0;
        Ai[8][n2] = 0;
        for (int k = 1; k <= 7; k++) {
            Ai[k][n2] = rsh32(s[8 + k], 11);
            Ar[16 - k][n2] = Ar[k][n2];
            Ai[16 - k][n2] = rsh32(-s[8 + k], 11);
        }
    }
    t1 = cyc(); scyc[S_F1] += t1 - t; t = t1;

    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 32; n++) {
            int32_t ar = Ar[k][n], ai = Ai[k][n], wr = TWR_Q14[k][n], wi = TWI_Q14[k][n];
            Br[k][n] = rsh32(ar * wr - ai * wi, 14);
            Bi[k][n] = rsh32(ar * wi + ai * wr, 14);
        }
    t1 = cyc(); scyc[S_TW] += t1 - t; t = t1;

    // FFT step 3: one pass per k1, bins k1 + 16 k2 (k2 0..7)
    for (int k = 0; k < 16; k++) {
        for (int r = 0; r < 32; r++) X[r] = (uint32_t)Br[k][r];
        for (int r = 0; r < 32; r++) X[32 + r] = (uint32_t)Bi[k][r];
        cim_go(T_F3, B_F3, 0);
        cim_sync();
        for (int j = 0; j < 8; j++) {
            sre[k + 16 * j] = rsh32((int32_t)CIM_REG(CIM_RES_LO(j)), 9);
            sim_[k + 16 * j] = rsh32((int32_t)CIM_REG(CIM_RES_LO(8 + j)), 9);
        }
    }
    t1 = cyc(); scyc[S_F3] += t1 - t; t = t1;

    for (int b = 0; b < 128; b++)
        mag[b] = isqrt64((uint64_t)((int64_t)sre[b] * sre[b]) + (uint64_t)((int64_t)sim_[b] * sim_[b]));
    t1 = cyc(); scyc[S_MAG] += t1 - t; t = t1;

    int64_t mel[NMEL];
    for (int r = 0; r < 64; r++) X[r] = mag[r];
    cim_go(T_MEL00, B_MEL, 0);
    cim_sync();
    for (int n = 0; n < 16; n++) mel[n] = cim_result(n);
    cim_go(T_MEL01, B_MEL, 0);
    for (int r = 0; r < 64; r++) X[r] = mag[64 + r];   // held until the pass has read X
    cim_go(T_MEL11, B_MEL, 1);                         // accumulate onto T_MEL01
    cim_sync();
    for (int n = 0; n < 16; n++) mel[16 + n] = cim_result(n);
    cim_go(T_MEL12, B_MEL, 0);
    cim_sync();
    for (int n = 0; n < 8; n++) mel[32 + n] = cim_result(n);
    t1 = cyc(); scyc[S_MEL] += t1 - t; t = t1;

    for (int m = 0; m < NMEL; m++)
        out[m] = mel[m] > 0 ? log2_q8((uint64_t)mel[m]) * 256 + 65536 * (sh + 5) - MFCC_K16 : LRAW_ZERO;
    scyc[S_LOG] += cyc() - t;
}

// ---------------------------------------------------------------- window path
static void finish(const int32_t *lr, int32_t lmx16, int8_t *q) {
    uint32_t t = cyc();
    for (int n = 0; n < NMEL; n++) {
        int32_t v = MFCC_FLOOR_Q8;
        if (lr[n] != LRAW_ZERO) {
            int64_t l = rsh64((int64_t)(lr[n] - lmx16) * MFCC_LN2_Q16, 24);
            if (l > v) v = (int32_t)l;
        }
        X[n] = (uint32_t)v;
    }
    int64_t y1[16], acc;
    cim_go(T_D1, B_DCT, 0);
    cim_sync();
    for (int n = 0; n < 16; n++) y1[n] = cim_result(n);
    cim_go(T_D2, B_DCT, 0);
    cim_sync();
    for (int k = 0; k < NMFCC; k++) {
        acc = 256 * y1[k] + (k < 6 ? y1[10 + k] : cim_result(k - 6));
        int64_t v = acc * MFCC_RQ_M + ((int64_t)MFCC_IN_ZP << MFCC_RQ_F);
        int64_t r = v >= 0 ? v >> MFCC_RQ_F : -((-v) >> MFCC_RQ_F);
        q[k] = (int8_t)(r > 127 ? 127 : (r < -128 ? -128 : r));
    }
    scyc[S_FIN] += cyc() - t;
}

int main(void) {
    int bad_l = 0, bad_q = 0;
    cim_init();
    uint32_t t = cyc();
    for (int i = 0; i < MFCC_NTILES; i++) cim_load_tile(i, &MFCC_TILES[i][0][0]);
    uint32_t c_load = cyc() - t;

    for (int f = 0; f < NFR; f++) {
        frame_lraw(&TV_CLIP[f * HOP], lraw[f]);
        for (int m = 0; m < NMEL; m++) bad_l += lraw[f][m] != TV_LRAW[f][m];
    }
    int32_t mx = 0;
    for (int i = 0; i < TV_NS; i++) if (TV_CLIP[i] > mx) mx = TV_CLIP[i];
    int32_t lmx = log2fix16((uint32_t)mx);
    if (lmx != TV_LMX16) { printf("log2fix16 %ld != %d\n", (long)lmx, TV_LMX16); bad_q++; }
    for (int f = 0; f < NFR; f++) {
        finish(lraw[f], lmx, feat[f]);
        for (int k = 0; k < NMFCC; k++) bad_q += feat[f][k] != TV_FEAT[f][k];
    }

    uint32_t tot = 0;
    for (int s = 0; s < S_N; s++) tot += scyc[s];
    printf("tiles: %d loaded in %lu cycles; passes per frame 54 + 2 (DCT)\n", MFCC_NTILES, (unsigned long)c_load);
    for (int s = 0; s < S_N; s++)
        printf("%-11s %7lu cyc/frame\n", SNAME[s], (unsigned long)(scyc[s] / NFR));
    printf("total       %7lu cyc/frame\n", (unsigned long)(tot / NFR));
    printf("HopCIM MFCC: %d frames, lraw mismatches %d, feature mismatches %d -> %s\n", NFR, bad_l, bad_q,
           (bad_l || bad_q) ? "FAIL" : "PASS");
    printf("done\n");
    return (bad_l || bad_q) ? EXIT_FAILURE : EXIT_SUCCESS;
}
