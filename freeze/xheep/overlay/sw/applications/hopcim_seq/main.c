// HopCIM R6b: the MLPerf Tiny int8 DS-CNN run by the endpoint's layer sequencer while the CPU
// sleeps. Once, the firmware builds the 61 network tiles from the compact weights (as hopcim_net)
// and loads the sequencer's parameter memory (bias, multiplier and shift of every output channel)
// and the zero points. Per decision the CPU writes the 49 x 10 int8 features (147 words) into
// FEAT, starts the sequencer and sleeps in WFI until its interrupt (PLIC, external line 3), then
// reads the class and the 12 logits. Checks logits and class of the 20 netvec clips bit-exactly
// against the golden model (model/dscnn_int.py = TFLite reference kernels), each clip three times:
// FEAT in order; as a ring with frame 0 at row 7 k + 3; and a streaming step, which writes only
// the newest frame (3 words, same values) into that ring, i.e. the CPU work of one decision when
// one frame arrives per hop (the per-window feature step is not part of this program).
//
// Cycles: sequencer = go -> interrupt and wall time = the endpoint's CYCLES counter; CPU-active =
// mcycle, stopped by mcountinhibit around WFI (on the FPGA the clock gate is a pass-through).

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "csr.h"
#include "rv_plic.h"
#include "core_v_mini_mcu.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"
#include "../hopcim_net/netvec.h"

#define NFR 49
#define NCH 10

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

// ---------------------------------------------------------------- interrupt and sleep
static volatile int seq_irq;
static volatile uint32_t t_irq;                     // wall time at which the interrupt was taken

static void cim_irq_handler(uint32_t id) {
    (void)id;
    t_irq = wall();
    CIM_REG(CIM_SEQ_STATUS) = CIM_SEQ_DONE;          // W1C: drop the level interrupt before complete
    seq_irq = 1;
}

// race-free: WFI wakes on a pending enabled interrupt even with mstatus.MIE = 0
static void sleep_until_seq_done(void) {
    CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    while (!seq_irq) {
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 1);         // stop mcycle while asleep
        asm volatile("wfi");
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);
        CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
        CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    }
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
}

// FEAT ring row r, words 0..2: int8 features 4w .. 4w+3 of one frame
static void write_frame(unsigned row, const int8_t *f) {
    for (int w = 0; w < 3; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4 && 4 * w + b < NCH; b++) v |= (uint32_t)(uint8_t)f[4 * w + b] << (8 * b);
        CIM_REG(CIM_FEAT(row, w)) = v;
    }
}

int main(void) {
    uint32_t c0 = cyc();
    cim_init();
    load_net_tiles();
    if (load_params()) {
        printf("FAIL: a layer needs a left shift\n");
        return EXIT_FAILURE;
    }
    uint32_t c_setup = cyc() - c0;

    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);              // machine interrupts on
    CSR_SET_BITS(CSR_REG_MIE, 1u << 11);             // machine external interrupt (PLIC)
    if (plic_Init() || plic_irq_set_priority(EXT_INTR_3, 1) || plic_irq_set_enabled(EXT_INTR_3, kPlicToggleEnabled)) {
        printf("PLIC setup failed\n");
        return EXIT_FAILURE;
    }
    plic_assign_external_irq_handler(EXT_INTR_3, &cim_irq_handler);
    CIM_REG(CIM_IRQ_EN) = CIM_IRQ_SEQ;               // interrupt when the sequencer is done

    int bad = 0, n_runs = 0, n_label = 0, n_golden = 0;
    uint32_t s_min = ~0u, s_max = 0, a_min = ~0u, a_max = 0, a_sum = 0, w_max = 0, n_full = 0;
    uint32_t t_min = ~0u, t_max = 0;                 // CPU-active of a streaming step
    static uint32_t s_clip[NV_N], a_clip[NV_N];
    static int cls_clip[NV_N], ok_clip[NV_N];
    for (int k = 0; k < NV_N; k++) {
        for (int ring = 0; ring < 3; ring++) {
            unsigned frow = ring ? (7u * (unsigned)k + 3u) % NFR : 0u;
            uint32_t w0 = wall(), c1 = cyc();
            if (ring < 2)
                for (int r = 0; r < NFR; r++) write_frame(((unsigned)r + frow) % NFR, NV_FEAT[k][r]);
            else
                write_frame((NFR - 1u + frow) % NFR, NV_FEAT[k][NFR - 1]);
            seq_irq = 0;
            uint32_t w_go = wall();
            cim_seq_go(0, frow, DS_T_CONV1);
            sleep_until_seq_done();
            uint32_t st = CIM_REG(CIM_SEQ_STATUS);
            int8_t lg[12];
            for (int n = 0; n < 12; n++) lg[n] = (int8_t)CIM_REG(CIM_LOGIT(n));
            uint32_t ca = cyc() - c1, sq = t_irq - w_go, dw = wall() - w0;
            int g = 0, ok = !(st & CIM_SEQ_ERR);
            for (int n = 0; n < 12; n++) {
                ok &= lg[n] == NV_LOGITS[k][n];
                if (NV_LOGITS[k][n] > NV_LOGITS[k][g]) g = n;
            }
            ok &= (int)CIM_SEQ_CLASS(st) == g;
            bad += !ok; n_runs++;
            if (sq < s_min) s_min = sq;
            if (sq > s_max) s_max = sq;
            if (ring < 2) {
                if (ca < a_min) a_min = ca;
                if (ca > a_max) a_max = ca;
                if (dw > w_max) w_max = dw;
                a_sum += ca; n_full++;
            } else {
                if (ca < t_min) t_min = ca;
                if (ca > t_max) t_max = ca;
            }
            if (!ring) {
                s_clip[k] = sq; a_clip[k] = ca; cls_clip[k] = (int)CIM_SEQ_CLASS(st); ok_clip[k] = ok;
                n_label += cls_clip[k] == NV_LABEL[k];
                n_golden += cls_clip[k] == g;
            } else ok_clip[k] &= ok;
        }
    }
    for (int k = 0; k < NV_N; k++)
        printf("clip %d: class %d (label %d), logits %s, sequencer %u cycles, CPU-active %u\n", k, cls_clip[k],
               NV_LABEL[k], ok_clip[k] ? "exact" : "WRONG", (unsigned)s_clip[k], (unsigned)a_clip[k]);
    printf("setup (61 tiles built and loaded, 640 parameter entries): %u cycles, once\n", (unsigned)c_setup);
    printf("sequencer per inference (go -> interrupt): %u..%u cycles\n", (unsigned)s_min, (unsigned)s_max);
    printf("CPU-active per decision, whole window (147 FEAT words, go, wake, class and logits): %u..%u cycles, mean %u\n",
           (unsigned)a_min, (unsigned)a_max, (unsigned)(a_sum / n_full));
    printf("CPU-active per decision, streaming step (3 FEAT words, go, wake, class and logits): %u..%u cycles\n",
           (unsigned)t_min, (unsigned)t_max);
    printf("wall per decision, whole window: max %u cycles\n", (unsigned)w_max);
    printf("%d runs (%d clips: FEAT in order, as a ring, streaming step): %d with wrong logits or class; "
           "predictions = golden %d/%d, = label %d/%d\n", n_runs, NV_N, bad, n_golden, NV_N, n_label, NV_N);
    if (bad == 0 && n_golden == NV_N) {
        printf("HopCIM seq: %d runs bit-exact -> PASS\n", n_runs);
        return EXIT_SUCCESS;
    }
    printf("HopCIM seq: %d of %d runs wrong -> FAIL\n", bad, n_runs);
    return EXIT_FAILURE;
}
