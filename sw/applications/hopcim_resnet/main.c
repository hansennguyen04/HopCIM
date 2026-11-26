// HopCIM: MLPerf Tiny's published int8 ResNet8 (image classification, unchanged) on the CIM with cim_gseq (docs/NET2_GATES.md).
// Once, the firmware builds the 86 network tiles from the weights (77,360 bytes, row x output matrices), loads the parameter
// memory and the 14 layer records. Per image it writes the 32 x 32 x 3 int8 pixels, starts the network and sleeps in WFI until the
// interrupt (PLIC, external line 3), then compares the 10 logits and the class with the golden model (model/resnet_int.py = TFLite's
// reference kernels). Needs a bitstream with use_cim_net2 and a 256-KiB build (weights in .rodata, images in .data).
// Cycles: go -> interrupt and wall time are the endpoint's CYCLES counter; CPU-active is mcycle, stopped around WFI.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "csr.h"
#include "rv_plic.h"
#include "core_v_mini_mcu.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/resnet_tables.h"
#include "resvec.h"

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }
static inline uint32_t wall(void) { return CIM_REG(CIM_CYCLES); }

static int8_t tbuf[64][16] __attribute__((aligned(4)));

// tile (r, c) of a layer: rows 64 r .., outputs 16 c ..
static void load_tiles(void) {
    for (int L = 0; L < 10; L++) {
        const rn_layer_t *p = &RN_LAYER[L];
        for (int r = 0; r < p->R; r++)
            for (int c = 0; c < p->C; c++) {
                for (int i = 0; i < 64; i++)
                    for (int n = 0; n < 16; n++) {
                        int row = 64 * r + i, col = 16 * c + n;
                        tbuf[i][n] = (row < p->rows && col < p->outs) ? p->w[row * p->outs + col] : 0;
                    }
                cim_load_tile(p->tile + r * p->C + c, &tbuf[0][0]);
            }
    }
}

static volatile int net_irq;
static volatile uint32_t t_irq;

static void net_irq_handler(uint32_t id) {
    (void)id;
    t_irq = wall();
    CIM_REG(CIM_NET_STATUS) = CIM_NET_DONE;          // W1C: drop the level interrupt before complete
    net_irq = 1;
}

static void sleep_until_done(void) {                 // race-free: WFI wakes on a pending enabled interrupt even with MIE = 0
    CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    while (!net_irq) {
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 1);
        asm volatile("wfi");
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);
        CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
        CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    }
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
}

int main(void) {
    uint32_t c0 = cyc();
    cim_init();
    load_tiles();
    for (int L = 0; L < 10; L++)
        for (int c = 0; c < 64; c++) {
            int k = 64 * L + c;
            CIM_REG(CIM_NET_PARAM(k, 0)) = (uint32_t)RN_BIAS[k];
            CIM_REG(CIM_NET_PARAM(k, 1)) = (uint32_t)RN_QM[k];
            CIM_REG(CIM_NET_PARAM(k, 2)) = (uint32_t)RN_E[k];
        }
    for (int o = 0; o < RN_NOPS; o++)
        for (int w = 0; w < 8; w++) CIM_REG(CIM_NET_DESC(o, w)) = RN_DESC[o][w];
    uint32_t c_setup = cyc() - c0;

    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
    CSR_SET_BITS(CSR_REG_MIE, 1u << 11);
    if (plic_Init() || plic_irq_set_priority(EXT_INTR_3, 1) || plic_irq_set_enabled(EXT_INTR_3, kPlicToggleEnabled)) {
        printf("PLIC setup failed\n");
        return EXIT_FAILURE;
    }
    plic_assign_external_irq_handler(EXT_INTR_3, &net_irq_handler);
    CIM_REG(CIM_IRQ_EN) = CIM_IRQ_SEQ;

    int bad = 0, n_label = 0;
    uint32_t s_min = ~0u, s_max = 0, a_sum = 0;
    for (int k = 0; k < RV_N; k++) {
        uint32_t w0 = wall(), c1 = cyc();
        const uint8_t *p = RV_IMG[k];
        for (int i = 0; i < 1024; i++, p += 3)
            CIM_REG(CIM_NET_IMG(i)) = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16;
        net_irq = 0;
        uint32_t w_go = wall();
        CIM_REG(CIM_NET_CTRL) = 0x80000000u | RN_NOPS;
        sleep_until_done();
        uint32_t st = CIM_REG(CIM_NET_STATUS);
        int8_t lg[10];
        for (int n = 0; n < 10; n++) lg[n] = (int8_t)CIM_REG(CIM_NET_LOGIT(n));
        uint32_t ca = cyc() - c1, sq = t_irq - w_go;
        int g = 0, ok = !(st & 0x4u);
        for (int n = 0; n < 10; n++) {
            ok &= lg[n] == RV_LOGITS[k][n];
            if (RV_LOGITS[k][n] > RV_LOGITS[k][g]) g = n;
        }
        ok &= (int)CIM_NET_CLASS(st) == g;
        bad += !ok;
        n_label += g == RV_LABEL[k];
        if (sq < s_min) s_min = sq;
        if (sq > s_max) s_max = sq;
        a_sum += ca;
        (void)w0;
        printf("image %d: class %d (label %d), logits %s, network %u cycles, CPU-active %u\n", k, (int)CIM_NET_CLASS(st), RV_LABEL[k],
               ok ? "exact" : "WRONG", (unsigned)sq, (unsigned)ca);
    }
    printf("setup (86 tiles built and loaded, 640 parameter entries, 14 records): %u cycles, once\n", (unsigned)c_setup);
    printf("network per inference (go -> interrupt): %u..%u cycles; CPU-active mean %u\n", (unsigned)s_min, (unsigned)s_max, (unsigned)(a_sum / RV_N));
    printf("%d images: %d with wrong logits or class; golden predictions = label %d/%d\n", RV_N, bad, n_label, RV_N);
    if (bad == 0) {
        printf("HopCIM resnet: %d images bit-exact -> PASS\n", RV_N);
        return EXIT_SUCCESS;
    }
    printf("HopCIM resnet: %d of %d wrong -> FAIL\n", bad, RV_N);
    return EXIT_FAILURE;
}
