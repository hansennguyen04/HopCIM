// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// HopCIM endpoint driver (CPU passes R3, front-end R5a, layer sequencer R6b). Register map:
// hopcim/rtl/cim_endpoint.sv.
// CIM_BASE is 256 KiB-aligned. In both builds (testharness and FPGA) the endpoint is the CPU data
// port's external slave and decodes addr[17:0].

#ifndef HOPCIM_CIM_H_
#define HOPCIM_CIM_H_

#include <stdint.h>

#define CIM_BASE 0xC0040000u

#define CIM_CTRL       (CIM_BASE + 0x0000u)
#define CIM_STATUS     (CIM_BASE + 0x0004u)
#define CIM_IRQ_EN     (CIM_BASE + 0x0008u)
#define CIM_DONE_CNT   (CIM_BASE + 0x000Cu)
#define CIM_RES_LO(n)  (CIM_BASE + 0x0040u + 4u * (n))
#define CIM_RES_HI(n)  (CIM_BASE + 0x0080u + 4u * (n))
#define CIM_X(r)       (CIM_BASE + 0x1000u + 4u * (r))
#define CIM_W(t, r, q) (CIM_BASE + 0x20000u + ((uint32_t)(t) << 10) + ((uint32_t)(r) << 4) + ((uint32_t)(q) << 2))
#define CIM_CYCLES     (CIM_BASE + 0x0018u)   // free-running clock counter (wall time)
#define CIM_TILES      128

#define CIM_STATUS_READY 0x1u
#define CIM_STATUS_DONE  0x2u   // sticky; cleared by a go or by writing 1 (W1C)
#define CIM_STATUS_IRQ   0x4u

// hardware MFCC front-end (R5a): audio ring, frame start, lraw of the frame
#define CIM_FE_CTRL    (CIM_BASE + 0x0010u)
#define CIM_FE_STATUS  (CIM_BASE + 0x0014u)
#define CIM_SAMPLES(i) (CIM_BASE + 0x2000u + 4u * (i))
#define CIM_LRAW(m)    (CIM_BASE + 0x3000u + 4u * (m))
#define CIM_RING       1024
#define CIM_FE_BUSY    0x1u
#define CIM_FE_DONE    0x2u     // sticky; cleared by a front-end go or by writing 1 (W1C)
#define CIM_LRAW_ZERO  0x80000000u

// layer sequencer (R6b): runs the DS-CNN from FEAT to a class without the CPU
#define CIM_SEQ_CTRL    (CIM_BASE + 0x0020u)
#define CIM_SEQ_STATUS  (CIM_BASE + 0x0024u)
#define CIM_SEQ_ZP      (CIM_BASE + 0x0028u)    // [7:0] conv1 input zp, [15:8] activation zp, [23:16] logits zp
#define CIM_FEAT(r, w)  (CIM_BASE + 0x4000u + ((uint32_t)(r) << 4) + ((uint32_t)(w) << 2))   // bytes 4w..4w+3 of row r
#define CIM_LOGIT(n)    (CIM_BASE + 0x5000u + 4u * (n))
#define CIM_PARAM(e, f) (CIM_BASE + 0x8000u + ((uint32_t)(e) << 4) + ((uint32_t)(f) << 2))   // entry 64 L + c; f: 0 bias, 1 qm, 2 e
#define CIM_ACT(b, p, c) (CIM_BASE + 0x10000u + ((uint32_t)(b) << 15) + ((uint32_t)(p) << 8) + ((uint32_t)(c) << 2))
#define CIM_SEQ_BUSY    0x1u
#define CIM_SEQ_DONE    0x2u     // sticky; cleared by a sequencer go or by writing 1 (W1C)
#define CIM_SEQ_ERR     0x4u     // a result was lost (never expected)
#define CIM_SEQ_CLASS(st) (((st) >> 4) & 0xFu)
#define CIM_SEQ_ALL     0x1000u  // status: the last window path recomputed all 49 frames
#define CIM_SEQ_LMX     (CIM_BASE + 0x002Cu)  // log2fix16 of the window maximum (Q16)
#define CIM_SEQ_WMAX    (CIM_BASE + 0x0030u)  // window maximum (signed)
// SEQ_CTRL window-path bits (R5b)
#define CIM_SEQ_WP       0x40000000u   // run the window path before the network
#define CIM_SEQ_WP_ONLY  0x20000000u   //   and stop after it (no full window yet)
#define CIM_SEQ_WP_FRAME 0x10000000u   //   a new frame: copy the front-end's lraw
#define CIM_SEQ_RESET    0x00800000u   //   start of a stream: forget the block maxima
#define CIM_IRQ_PASS    0x1u     // IRQ_EN bits
#define CIM_IRQ_FE      0x2u
#define CIM_IRQ_SEQ     0x4u

#define CIM_ROWS 64
#define CIM_COLS 16
#define CIM_REG(a) (*(volatile uint32_t *)(uintptr_t)(a))

static uint32_t cim_issued;                     // passes started by this program

// call once before the first pass (DONE_CNT is not cleared when a program is reloaded)
static inline void cim_init(void) { cim_issued = CIM_REG(CIM_DONE_CNT); }

// A go is held on the bus until the engine can accept it, so back-to-back passes need no polling.
static inline void cim_go(unsigned tile, unsigned bits, int accum) {
    CIM_REG(CIM_CTRL) = 0x80000000u | ((accum ? 1u : 0u) << 16) | ((bits & 31u) << 8) | (tile & 127u);
    cim_issued++;
}

// wait until every started pass has completed; the results are then those of the last pass.
// (STATUS.done alone is not enough after back-to-back passes: the previous pass can set it.)
static inline void cim_sync(void) {
    while (CIM_REG(CIM_DONE_CNT) != cim_issued) {
    }
}

static inline void cim_wait_done(void) {
    while (!(CIM_REG(CIM_STATUS) & CIM_STATUS_DONE)) {
    }
}

// wait until DONE_CNT reaches target (use for back-to-back passes)
static inline void cim_wait_count(uint32_t target) {
    while ((int32_t)(CIM_REG(CIM_DONE_CNT) - target) < 0) {
    }
}

// a 64 x 16 int8 tile, row-major w[r][n]: word q of row r holds columns 4q..4q+3 (byte k = 4q+k)
static inline void cim_load_tile(unsigned tile, const int8_t *w) {
    const uint32_t *src = (const uint32_t *)w;
    volatile uint32_t *dst = &CIM_REG(CIM_W(tile, 0, 0));
    for (int i = 0; i < CIM_ROWS * CIM_COLS / 4; i++) dst[i] = src[i];
}

// inputs: the low b bits of each word are used (b given with the go)
static inline void cim_load_x(const int32_t *x) {
    volatile uint32_t *dst = &CIM_REG(CIM_X(0));
    for (int r = 0; r < CIM_ROWS; r++) dst[r] = (uint32_t)x[r];
}

// start the front-end on the frame of 480 ring samples beginning at ring index base. The write
// is held until the engine has no pass in flight; the front-end uses tiles 0..5.
static inline void cim_fe_go(unsigned base) { CIM_REG(CIM_FE_CTRL) = 0x80000000u | (base & (CIM_RING - 1u)); }

static inline void cim_fe_wait(void) {
    while (!(CIM_REG(CIM_FE_STATUS) & CIM_FE_DONE)) {
    }
}

// run the network on FEAT (frame 0 in ring row frow; network tiles from tbase): stop = 0 runs to
// the class, s stops after layer s - 1. The write is held until no CPU pass is in flight and cim_fe
// is idle. The sequencer overwrites both X banks.
static inline void cim_seq_go(unsigned stop, unsigned frow, unsigned tbase) {
    CIM_REG(CIM_SEQ_CTRL) = 0x80000000u | ((stop & 15u) << 24) | ((frow & 63u) << 16) | (tbase & 127u);
}

// one hop of a stream (R5b): the window path (flags: CIM_SEQ_WP_*, CIM_SEQ_RESET), then unless
// CIM_SEQ_WP_ONLY the network on the window whose frame 0 is in FEAT row frow. With
// CIM_SEQ_WP_FRAME the write is held until no front-end runs.
static inline void cim_seq_hop(uint32_t flags, unsigned frow, unsigned tbase) {
    CIM_REG(CIM_SEQ_CTRL) = 0x80000000u | CIM_SEQ_WP | flags | ((frow & 63u) << 16) | (tbase & 127u);
}

// 48-bit signed result of column n of the last completed pass
static inline int64_t cim_result(unsigned n) {
    uint32_t lo = CIM_REG(CIM_RES_LO(n));
    uint32_t hi = CIM_REG(CIM_RES_HI(n));
    return (int64_t)(((uint64_t)hi << 32) | lo);
}

#endif  // HOPCIM_CIM_H_
