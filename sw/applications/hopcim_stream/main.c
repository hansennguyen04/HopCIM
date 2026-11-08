// HopCIM R5b: streaming keyword spotting with all per-hop computation in hardware. Per 20 ms hop
// the CPU writes the 320 new samples into the endpoint's audio ring (which tracks their maximum),
// starts the front-end on the new frame and sleeps; on its interrupt it starts the sequencer with
// the window path (and, once a full 1-s window exists, the network) and sleeps again; on that
// interrupt it reads the class. The window path normalises, DCTs and requantises the window's
// features in hardware and recomputes all 49 frames only when the window maximum's log2fix16
// changes (model/stream_int.py). Checks lmx16 and the recompute flag of every hop and the logits
// and class of every decision bit-exactly (streamvec.h: 2 s of GSC speech, 100 hops, 51
// decisions). HOPCIM_DED selects the golden of the B3 front-end (hopcim_stream_ded).
//
// Cycles per hop: CPU-active = mcycle, stopped by mcountinhibit around WFI (on the FPGA the clock
// gate is a pass-through); wall = the endpoint's CYCLES counter.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "csr.h"
#include "rv_plic.h"
#include "core_v_mini_mcu.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/mfcc_tables.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"
#include "../hopcim_stream/streamvec.h"
#ifdef HOPCIM_DED
#define GOLD_LOGITS SV_LOGITS_DED
#define GOLD_CLASS  SV_CLASS_DED
#else
#define GOLD_LOGITS SV_LOGITS
#define GOLD_CLASS  SV_CLASS
#endif

#define HOP 320
#define NFR 49

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }
static inline uint32_t wall(void) { return CIM_REG(CIM_CYCLES); }

// ---------------------------------------------------------------- tiles (as hopcim_net)
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

// ---------------------------------------------------------------- interrupts and sleep
static volatile int fe_irq, seq_irq;

static void cim_irq_handler(uint32_t id) {
    (void)id;
    if (CIM_REG(CIM_FE_STATUS) & CIM_FE_DONE) {      // W1C drops the level interrupt
        CIM_REG(CIM_FE_STATUS) = CIM_FE_DONE;
        fe_irq = 1;
    }
    if (CIM_REG(CIM_SEQ_STATUS) & CIM_SEQ_DONE) {
        CIM_REG(CIM_SEQ_STATUS) = CIM_SEQ_DONE;
        seq_irq = 1;
    }
}

// race-free: WFI wakes on a pending enabled interrupt even with mstatus.MIE = 0
static void sleep_until(volatile int *flag) {
    CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    while (!*flag) {
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 1);         // stop mcycle while asleep
        asm volatile("wfi");
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);
        CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
        CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    }
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
}

static volatile uint32_t *const RING = &CIM_REG(CIM_SAMPLES(0));

int main(void) {
    cim_init();
    for (int t = 0; t < MFCC_NTILES; t++) cim_load_tile(t, &MFCC_TILES[t][0][0]);   // front-end, DCT
    load_net_tiles();
    if (load_params()) {
        printf("FAIL: a layer needs a left shift\n");
        return EXIT_FAILURE;
    }
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
    CSR_SET_BITS(CSR_REG_MIE, 1u << 11);
    if (plic_Init() || plic_irq_set_priority(EXT_INTR_3, 1) || plic_irq_set_enabled(EXT_INTR_3, kPlicToggleEnabled)) {
        printf("PLIC setup failed\n");
        return EXIT_FAILURE;
    }
    plic_assign_external_irq_handler(EXT_INTR_3, &cim_irq_handler);
    CIM_REG(CIM_IRQ_EN) = CIM_IRQ_FE | CIM_IRQ_SEQ;

    int bad_hop = 0, bad_dec = 0, n_all = 0;
    uint32_t a_samp = 0, a_rest_min = ~0u, a_rest_max = 0, a_sum = 0;   // CPU-active per decision hop
    uint32_t w_one_min = ~0u, w_one_max = 0, w_all_min = ~0u, w_all_max = 0;  // wall per decision hop
    static uint8_t cls[SV_DEC];
    for (int h = 0; h < SV_HOPS; h++) {
        uint32_t c0 = cyc(), w0 = wall();
        for (int i = 0; i < HOP; i++) RING[(HOP * h + i) & (CIM_RING - 1)] = (uint16_t)SV_SAMPLES[HOP * h + i];
        uint32_t c_samp = cyc() - c0;
        if (h >= 1) {                                // the new frame starts at sample 320 (h - 1)
            fe_irq = 0;
            cim_fe_go((uint32_t)(HOP * (h - 1)) & (CIM_RING - 1));
            sleep_until(&fe_irq);
        }
        seq_irq = 0;
        cim_seq_hop((h < NFR ? CIM_SEQ_WP_ONLY : 0u) | (h >= 1 ? CIM_SEQ_WP_FRAME : 0u) | (h == 0 ? CIM_SEQ_RESET : 0u),
                    (unsigned)(h % NFR), DS_T_CONV1);
        sleep_until(&seq_irq);
        uint32_t st = CIM_REG(CIM_SEQ_STATUS), lmx = CIM_REG(CIM_SEQ_LMX);
        int all = (st & CIM_SEQ_ALL) != 0;
        bad_hop += (lmx != SV_LMX[h]) || (all != SV_ALL[h]) || ((st & CIM_SEQ_ERR) != 0);
        if (h >= NFR) {
            int k = h - NFR, ok = 1;
            for (int n = 0; n < 12; n++) ok &= (int8_t)CIM_REG(CIM_LOGIT(n)) == GOLD_LOGITS[k][n];
            cls[k] = (uint8_t)CIM_SEQ_CLASS(st);
            ok &= cls[k] == GOLD_CLASS[k];
            bad_dec += !ok;
            uint32_t ca = cyc() - c0, cw = wall() - w0;
            a_samp += c_samp;
            a_sum += ca - c_samp;
            if (ca - c_samp < a_rest_min) a_rest_min = ca - c_samp;
            if (ca - c_samp > a_rest_max) a_rest_max = ca - c_samp;
            if (all) {
                n_all++;
                if (cw < w_all_min) w_all_min = cw;
                if (cw > w_all_max) w_all_max = cw;
            } else {
                if (cw < w_one_min) w_one_min = cw;
                if (cw > w_one_max) w_one_max = cw;
            }
        }
    }
    printf("classes of the %d decisions:", SV_DEC);
    for (int k = 0; k < SV_DEC; k++) printf(" %d", cls[k]);
    printf("\n");
    printf("per decision hop, CPU-active: copying 320 samples %u cycles (mean); everything else %u..%u (mean %u)\n",
           (unsigned)(a_samp / SV_DEC), (unsigned)a_rest_min, (unsigned)a_rest_max, (unsigned)(a_sum / SV_DEC));
    printf("per decision hop, wall: %u..%u cycles (new frame only), %u..%u (all 49 frames, %d hops)\n",
           (unsigned)w_one_min, (unsigned)w_one_max, (unsigned)w_all_min, (unsigned)w_all_max, n_all);
    printf("HopCIM stream: %d hops, %d decisions, %d hop and %d decision mismatches -> %s\n", SV_HOPS, SV_DEC,
           bad_hop, bad_dec, (bad_hop == 0 && bad_dec == 0) ? "PASS" : "FAIL");
    return (bad_hop == 0 && bad_dec == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
