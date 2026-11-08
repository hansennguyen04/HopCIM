// HopCIM R3 smoke test: the CPU programs the CIM endpoint over the bus, runs passes over several
// tiles, input widths and accumulate chains, and checks every result against a C reference.
// Also reports the CPU cycles (mcycle) of each MMIO step: weight load, X load, pass, readback.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../../hw/ip/hopcim/sw/cim.h"

#define NT 8

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }

static int8_t W[NT][CIM_ROWS][CIM_COLS] __attribute__((aligned(4)));
static int32_t X[CIM_ROWS];
static int64_t ref[CIM_COLS];
static int64_t got[CIM_COLS];
static uint32_t xs = 2463534242u;
static inline uint32_t rnd(void) { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }

static int32_t sext(int32_t v, int b) { return (int32_t)((uint32_t)v << (32 - b)) >> (32 - b); }

static void ref_pass(int t, int b, int acc) {
    for (int n = 0; n < CIM_COLS; n++) {
        int64_t s = 0;
        for (int r = 0; r < CIM_ROWS; r++) s += (int64_t)W[t][r][n] * sext(X[r], b);
        ref[n] = (acc ? ref[n] : 0) + s;
    }
}

static void random_x(int b) {
    for (int r = 0; r < CIM_ROWS; r++) X[r] = (int32_t)(rnd() & ((1u << b) - 1u));
}

static uint32_t readback(void) {
    uint32_t t0 = cyc();
    for (int n = 0; n < CIM_COLS; n++) got[n] = cim_result(n);
    return cyc() - t0;
}

static int compare(const char *tag) {
    int e = 0;
    for (int n = 0; n < CIM_COLS; n++) {
        if (got[n] != ref[n]) {
            if (e < 4) printf("%s: col %d got %08x%08x exp %08x%08x\n", tag, n, (unsigned)(got[n] >> 32),
                              (unsigned)got[n], (unsigned)(ref[n] >> 32), (unsigned)ref[n]);
            e++;
        }
    }
    return e;
}

int main(void) {
    static const struct { uint8_t tile, bits, acc; } tests[] = {
        {0, 8, 0}, {1, 16, 0}, {2, 24, 0}, {3, 1, 0}, {4, 12, 0}, {5, 8, 1}, {6, 20, 1}, {7, 24, 1}};
    int errors = 0, passes = 0;
    uint32_t t0, c_w = 0, c_x = 0, c_rb = 0, c_pass[25] = {0};

    for (int t = 0; t < NT; t++)
        for (int r = 0; r < CIM_ROWS; r++)
            for (int n = 0; n < CIM_COLS; n++) W[t][r][n] = (int8_t)rnd();
    W[0][0][0] = -128; W[0][1][0] = 127;              // extremes

    for (int t = 0; t < NT; t++) {
        t0 = cyc();
        cim_load_tile(t, &W[t][0][0]);
        c_w = cyc() - t0;
    }

    // single passes, each checked
    for (unsigned i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        int b = tests[i].bits, t = tests[i].tile, acc = tests[i].acc;
        random_x(b);
        if (b == 24) X[0] = 0x800000;                  // -2^23
        t0 = cyc();
        cim_load_x(X);
        c_x = cyc() - t0;
        ref_pass(t, b, acc);
        t0 = cyc();
        cim_go(t, b, acc);
        cim_wait_done();
        c_pass[b] = cyc() - t0;
        passes++;
        c_rb = readback();
        char tag[24];
        snprintf(tag, sizeof(tag), "pass t%d b%d a%d", t, b, acc);
        errors += compare(tag);
    }

    // back-to-back chain: a 512-long dot product over 8 tiles (go is held while the engine is busy)
    random_x(8);
    cim_load_x(X);
    for (int t = 0; t < NT; t++) ref_pass(t, 8, t != 0);
    uint32_t cnt0 = CIM_REG(CIM_DONE_CNT);
    t0 = cyc();
    for (int t = 0; t < NT; t++) cim_go(t, 8, t != 0);
    cim_wait_count(cnt0 + NT);
    uint32_t c_chain = cyc() - t0;
    passes += NT;
    readback();
    errors += compare("chain");

    // DONE_CNT and the STATUS irq bit (IRQ_EN only; the PLIC is not used here)
    if (CIM_REG(CIM_DONE_CNT) != (uint32_t)passes) {
        printf("DONE_CNT %u != %d\n", (unsigned)CIM_REG(CIM_DONE_CNT), passes);
        errors++;
    }
    CIM_REG(CIM_IRQ_EN) = 1;
    if (!(CIM_REG(CIM_STATUS) & CIM_STATUS_IRQ)) { printf("irq bit not set\n"); errors++; }
    CIM_REG(CIM_STATUS) = CIM_STATUS_DONE;              // W1C
    if (CIM_REG(CIM_STATUS) & (CIM_STATUS_DONE | CIM_STATUS_IRQ)) { printf("done not cleared\n"); errors++; }
    CIM_REG(CIM_IRQ_EN) = 0;

    printf("cycles: tile load %u (256 words), X load %u (64 words), readback %u (32 words)\n",
           (unsigned)c_w, (unsigned)c_x, (unsigned)c_rb);
    printf("cycles: go->done b1 %u, b8 %u, b16 %u, b24 %u; chain of %d b8 passes %u\n", (unsigned)c_pass[1],
           (unsigned)c_pass[8], (unsigned)c_pass[16], (unsigned)c_pass[24], NT, (unsigned)c_chain);
    printf("CIM smoke: %d passes, %d errors -> %s\n", passes, errors, errors ? "FAIL" : "PASS");
    printf("done\n");
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
