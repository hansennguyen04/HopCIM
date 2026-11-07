// HopCIM: the integer CIM-MFCC front-end with the frame path in hardware, the CPU asleep while it
// runs. Per hop the CPU writes the 320 new samples into the endpoint's audio ring and starts the
// front-end, then sleeps in WFI until the front-end interrupt (PLIC, external line 3) wakes it;
// it reads lraw[40]. Per window it finishes on the CPU (normalisation, DCT as 2 CIM passes,
// requantisation). Checks lraw and the int8 features of the test clip bit-exactly against the
// golden model. Works with either front-end build (cim_fe, or the dedicated baseline B3).
//
// CPU-active cycles: mcycle, stopped (mcountinhibit) only around WFI, since on the FPGA the
// clock gate is a pass-through and mcycle would otherwise count the sleep. Wall cycles: the
// endpoint's free-running CYCLES counter.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "csr.h"
#include "rv_plic.h"
#include "core_v_mini_mcu.h"
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
static inline uint32_t wall(void) { return CIM_REG(CIM_CYCLES); }
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
static volatile int fe_irq;
static volatile uint32_t t_irq;                     // wall time at which the interrupt was taken

static void cim_irq_handler(uint32_t id) {
    (void)id;
    t_irq = wall();
    CIM_REG(CIM_FE_STATUS) = CIM_FE_DONE;            // W1C: drop the level interrupt before complete
    fe_irq = 1;
}

// sleep until the front-end interrupt; race-free: WFI wakes on a pending enabled interrupt even
// with mstatus.MIE = 0, and the interrupt is taken once MIE is set again
static void sleep_until_fe_done(void) {
    CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    while (!fe_irq) {
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 1);         // stop mcycle while asleep
        asm volatile("wfi");
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);
        CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
        CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    }
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
}

// window path (as hopcim_fe): lmq -> x, DCT on tiles D1/D2, requantisation
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
    uint32_t c_cpu = 0, c_wall = 0, c_fe = 0, c_fe_max = 0, c_cpu_max = 0;
    cim_init();
    for (int i = 0; i < MFCC_NTILES; i++) cim_load_tile(i, &MFCC_TILES[i][0][0]);

    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);              // machine interrupts on
    CSR_SET_BITS(CSR_REG_MIE, 1u << 11);             // machine external interrupt (PLIC)
    if (plic_Init() || plic_irq_set_priority(EXT_INTR_3, 1) || plic_irq_set_enabled(EXT_INTR_3, kPlicToggleEnabled)) {
        printf("PLIC setup failed\n");
        return EXIT_FAILURE;
    }
    plic_assign_external_irq_handler(EXT_INTR_3, &cim_irq_handler);
    CIM_REG(CIM_IRQ_EN) = 2;                         // interrupt when the front-end is done

    int next = 0;
    for (int f = 0; f < NFR; f++) {
        uint32_t w0 = wall(), c0 = cyc();
        for (; next < f * HOP + WIN; next++) RING[next & (CIM_RING - 1)] = (uint16_t)TV_CLIP[next];
        fe_irq = 0;
        uint32_t w_go = wall();
        cim_fe_go((uint32_t)(f * HOP) & (CIM_RING - 1));
        sleep_until_fe_done();
        for (int m = 0; m < NMEL; m++) lraw[f][m] = (int32_t)LRAW[m];
        uint32_t dc = cyc() - c0, dw = wall() - w0, dfe = t_irq - w_go;
        c_cpu += dc; c_wall += dw; c_fe += dfe;
        if (dfe > c_fe_max) c_fe_max = dfe;
        if (dc > c_cpu_max) c_cpu_max = dc;
        for (int m = 0; m < NMEL; m++) bad_l += lraw[f][m] != TV_LRAW[f][m];
    }
    int32_t mx = 0;
    for (int i = 0; i < TV_NS; i++) if (TV_CLIP[i] > mx) mx = TV_CLIP[i];
    int32_t lmx = log2fix16((uint32_t)mx);
    uint32_t c0 = cyc();
    for (int f = 0; f < NFR; f++) finish(lraw[f], lmx, feat[f]);
    uint32_t c_fin = cyc() - c0;
    for (int f = 0; f < NFR; f++)
        for (int k = 0; k < NMFCC; k++) bad_q += feat[f][k] != TV_FEAT[f][k];

    printf("per frame (mean over %d): CPU-active %lu (max %lu), wall %lu, front-end go->irq %lu (max %lu) cycles\n",
           NFR, (unsigned long)(c_cpu / NFR), (unsigned long)c_cpu_max, (unsigned long)(c_wall / NFR),
           (unsigned long)(c_fe / NFR), (unsigned long)c_fe_max);
    printf("window path (finish) %lu CPU cycles/frame; CPU-active per hop incl. finish %lu cycles\n",
           (unsigned long)(c_fin / NFR), (unsigned long)((c_cpu + c_fin) / NFR));
    printf("HopCIM FE irq: %d frames, lraw mismatches %d, feature mismatches %d -> %s\n", NFR, bad_l, bad_q,
           (bad_l || bad_q) ? "FAIL" : "PASS");
    printf("done\n");
    return (bad_l || bad_q) ? EXIT_FAILURE : EXIT_SUCCESS;
}
