// HopCIM R5a: the integer CIM-MFCC front-end with the frame path in hardware (cim_fe). Per hop the
// CPU writes the 320 new samples into the endpoint's audio ring, starts the front-end and reads
// lraw[40]; per window it finishes on the CPU (normalisation, DCT as 2 CIM passes, requantisation).
// Checks lraw and the int8 features of the test clip bit-exactly against the golden model
// (hopcim/model/mfcc_int.py) and reports CPU-active cycles and front-end latency per frame.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/mfcc_tables.h"
#ifdef HOPCIM_DED                                  // baseline B3 build: its own golden values
#include "../hopcim_mfcc/testvec_ded.h"
#else
#include "../hopcim_mfcc/testvec.h"
#endif

#define WIN   480
#define HOP   320
#define NFR   49
#define NMEL  40
#define NMFCC 10

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }
static inline int64_t rsh64(int64_t x, int s) { return (x + ((int64_t)1 << (s - 1))) >> s; }

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

static volatile uint32_t *const X = &CIM_REG(CIM_X(0));
static volatile uint32_t *const RING = &CIM_REG(CIM_SAMPLES(0));
static volatile uint32_t *const LRAW = &CIM_REG(CIM_LRAW(0));

static int32_t lraw[NFR][NMEL];
static int8_t feat[NFR][NMFCC];

// window path (as hopcim_mfcc): lmq -> x, DCT on tiles D1/D2, requantisation
static void finish(const int32_t *lr, int32_t lmx16, int8_t *q) {
    for (int n = 0; n < NMEL; n++) {
        int32_t v = MFCC_FLOOR_Q8;
        if ((uint32_t)lr[n] != CIM_LRAW_ZERO) {
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
}

int main(void) {
    int bad_l = 0, bad_q = 0;
    uint32_t t, c_samp = 0, c_fe = 0, c_fe_max = 0, c_rd = 0, c_fin = 0;
    cim_init();
    for (int i = 0; i < MFCC_NTILES; i++) cim_load_tile(i, &MFCC_TILES[i][0][0]);

    int next = 0;                                       // next clip sample to write into the ring
    for (int f = 0; f < NFR; f++) {
        // per hop: the new samples of this frame (480 for the first, then 320)
        t = cyc();
        for (; next < f * HOP + WIN; next++) RING[next & (CIM_RING - 1)] = (uint16_t)TV_CLIP[next];
        c_samp += cyc() - t;
        t = cyc();
        cim_fe_go((uint32_t)(f * HOP) & (CIM_RING - 1));
        cim_fe_wait();
        uint32_t d = cyc() - t;
        c_fe += d;
        if (d > c_fe_max) c_fe_max = d;
        t = cyc();
        for (int m = 0; m < NMEL; m++) lraw[f][m] = (int32_t)LRAW[m];
        c_rd += cyc() - t;
        for (int m = 0; m < NMEL; m++) bad_l += lraw[f][m] != TV_LRAW[f][m];
    }
    int32_t mx = 0;
    for (int i = 0; i < TV_NS; i++) if (TV_CLIP[i] > mx) mx = TV_CLIP[i];
    int32_t lmx = log2fix16((uint32_t)mx);
    t = cyc();
    for (int f = 0; f < NFR; f++) finish(lraw[f], lmx, feat[f]);
    c_fin = cyc() - t;
    for (int f = 0; f < NFR; f++)
        for (int k = 0; k < NMFCC; k++) bad_q += feat[f][k] != TV_FEAT[f][k];

    printf("per frame (mean over %d): samples %lu, lraw readback %lu, finish %lu CPU cycles\n", NFR,
           (unsigned long)(c_samp / NFR), (unsigned long)(c_rd / NFR), (unsigned long)(c_fin / NFR));
    printf("front-end go->done %lu cycles (max %lu); CPU-active excl. front-end %lu cycles/frame\n",
           (unsigned long)(c_fe / NFR), (unsigned long)c_fe_max, (unsigned long)((c_samp + c_rd + c_fin) / NFR));
    printf("HopCIM FE: %d frames, lraw mismatches %d, feature mismatches %d -> %s\n", NFR, bad_l, bad_q,
           (bad_l || bad_q) ? "FAIL" : "PASS");
    printf("done\n");
    return (bad_l || bad_q) ? EXIT_FAILURE : EXIT_SUCCESS;
}
