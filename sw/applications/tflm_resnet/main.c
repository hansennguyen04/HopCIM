// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// The CPU-only baseline of the second network (docs/NET2_GATES.md, N6; unverified): TensorFlow Lite Micro, the runtime of
// MLPerf Tiny's reference submission, runs MLPerf Tiny's ResNet8 (pretrainedResnet_quant.tflite, unchanged) on X-HEEP's CV32E20
// with its reference kernels, on the same board, bitstream and clock as HopCIM's run of the network (hopcim_resnet on
// xheep_nexys_bscan_cim128knet2.bit, 15.36 MHz; the CIM endpoint is in the bitstream and never used here).
//
// Per image (tvec.h, CIFAR-10 test images 0 ..): the 3,072 int8 pixels into the model's input tensor (timed), one Invoke (timed;
// the model ends in its softmax), then the 10 int8 outputs compared with TensorFlow Lite Micro's on the host at the same commit
// (mlperf/gen_tflm_resnet.py), and the class with the class of HopCIM's golden logits (model/resnet_int.py). The input is copied
// before every Invoke: TFLM's memory planner lets later layers reuse the input tensor's bytes.
// Cycles: mcycle/mcycleh (the CPU never sleeps here, and on the FPGA the clock gate is a pass-through).
//
// Build: mlperf/build_tflm_resnet.sh <X-HEEP tree made with configs/kws_nexys_256k_c180.py> (code and the 98,496-byte model
// need more than the 256-KiB layout's 144-KiB code section: patch 0008 gives 180 KiB of code and 76 KiB of data; the images
// are in the data section).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "csr.h"
#include "tvec.h"

int tflm_setup(void);
int8_t *tflm_input(void);
const int8_t *tflm_output(void);
int tflm_invoke(void);
size_t tflm_arena_used(void);
size_t tflm_arena_size(void);

static uint64_t cyc64(void) {
    uint32_t hi, lo, hi2;
    do {
        asm volatile("csrr %0, mcycleh" : "=r"(hi));
        asm volatile("csrr %0, mcycle" : "=r"(lo));
        asm volatile("csrr %0, mcycleh" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

// a 64-bit count as decimal (the printf here has no %llu)
static void put_u64(uint64_t v) {
    char b[21];
    int i = 20;
    b[i] = 0;
    do { b[--i] = (char)('0' + v % 10); v /= 10; } while (v);
    printf("%s", &b[i]);
}

static int argmax10(const int8_t *v) {                // the first maximum, as the golden class
    int g = 0;
    for (int n = 1; n < 10; n++)
        if (v[n] > v[g]) g = n;
    return g;
}

int main(void) {
    uint64_t c0 = cyc64();
    int rc = tflm_setup();
    uint64_t c_setup = cyc64() - c0;
    if (rc != 0) {
        printf("TFLM setup failed at step %d\n", -rc);
        printf("done\n");
        return EXIT_FAILURE;
    }
    printf("TFLM ResNet8: setup ");
    put_u64(c_setup);
    printf(" cycles, arena %u of %u bytes used\n", (unsigned)tflm_arena_used(), (unsigned)tflm_arena_size());

    int bad = 0, fail = 0, n_label = 0, n_hop = 0;
    uint64_t inv_min = ~0ull, inv_max = 0, inv_sum = 0, cp_min = ~0ull, cp_max = 0;
    for (int k = 0; k < TV_N; k++) {
        uint64_t t0 = cyc64();
        int8_t *t = tflm_input();
        const int8_t *p = TV_IMG[k];
        for (int i = 0; i < 3072; i++) t[i] = p[i];
        uint64_t t1 = cyc64();
        int r = tflm_invoke();
        uint64_t t2 = cyc64();
        uint64_t inv = t2 - t1, cp = t1 - t0;
        const int8_t *o = tflm_output();
        int ok = r == 0;
        for (int n = 0; n < 10; n++) ok &= o[n] == TV_OUT[k][n];
        int cls = argmax10(o), hop = argmax10(TV_LOGITS[k]);
        bad += !ok;
        fail += r != 0;
        n_label += cls == TV_LABEL[k];
        n_hop += cls == hop;
        if (inv < inv_min) inv_min = inv;
        if (inv > inv_max) inv_max = inv;
        if (cp < cp_min) cp_min = cp;
        if (cp > cp_max) cp_max = cp;
        inv_sum += inv;
        printf("image %d: class %d (label %d, HopCIM's class %d), outputs %s, Invoke ", k, cls, TV_LABEL[k], hop, ok ? "exact" : "WRONG");
        put_u64(inv);
        printf(" cycles, input copy %u\n", (unsigned)cp);
    }
    printf("Invoke per inference: ");
    put_u64(inv_min);
    printf("..");
    put_u64(inv_max);
    printf(" cycles, mean ");
    put_u64(inv_sum / TV_N);
    printf("; input copy %u..%u cycles\n", (unsigned)cp_min, (unsigned)cp_max);
    printf("%d images: %d with outputs different from TFLite Micro on the host, %d failed invokes; class = label %d/%d, = HopCIM's %d/%d\n",
           TV_N, bad, fail, n_label, TV_N, n_hop, TV_N);
    if (bad == 0) printf("TFLM resnet: %d images exact -> PASS\n", TV_N);
    else printf("TFLM resnet: %d of %d wrong -> FAIL\n", bad, TV_N);
    printf("done\n");
    return bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
