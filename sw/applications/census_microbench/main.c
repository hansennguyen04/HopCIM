// Census microbenchmarks for an X-HEEP (CV32E20, rv32imc) host.
// Measures the CPU-side costs that remain around a CIM accelerator:
// int8 kernels, requantisation, copies (CPU vs DMA), MMIO access and a
// KWS-style audio front-end (512-pt fixed-point FFT, power, mel, log, DCT).
// All timings are mcycle deltas; each kernel runs once after a warm-up call.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "core_v_mini_mcu.h"
#include "x-heep.h"
#include "dma_sdk.h"
#include "dma_regs.h"
#include "tables.h"
#include "timer_sdk.h"

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }

#define REPORT(name, cycles, work, unit) \
    printf("%-18s %8u cyc  %7u %-5s  %u.%02u cyc/%s\n", name, (unsigned)(cycles), (unsigned)(work), unit, \
           (unsigned)((cycles) / (work)), (unsigned)(((cycles) % (work)) * 100 / (work)), unit)

// ---------------------------------------------------------------- buffers
#define NV 1024
static int8_t va[NV] __attribute__((aligned(4)));
static int8_t vb[NV] __attribute__((aligned(4)));
static int32_t acc32[NV] __attribute__((aligned(4)));
int8_t q8[NV] __attribute__((aligned(4)));

// conv3x3: 8x8 output, Cin=Cout=16, HWC, input pre-padded to 10x10
#define C3_H 8
#define C3_C 16
static int8_t c3_in[(C3_H + 2) * (C3_H + 2) * C3_C] __attribute__((aligned(4)));
static int8_t c3_w[C3_C * 9 * C3_C] __attribute__((aligned(4)));   // [cout][ky][kx][cin]
int8_t c3_out[C3_H * C3_H * C3_C] __attribute__((aligned(4)));

// KWS DS-CNN shapes: 25x5 spatial, 64 channels
#define KH 25
#define KW 5
#define KC 64
static int8_t pw_in[KH * KW * KC] __attribute__((aligned(4)));
static int8_t pw_w[KC * KC] __attribute__((aligned(4)));            // [cout][cin]
int8_t pw_out[KH * KW * KC] __attribute__((aligned(4)));
static int8_t dw_in[(KH + 2) * (KW + 2) * KC] __attribute__((aligned(4)));
static int8_t dw_w[9 * KC] __attribute__((aligned(4)));             // [ky][kx][c]

// copies
#define CPY_BYTES 4096
static uint32_t src_buf[CPY_BYTES / 4] __attribute__((aligned(4)));
uint32_t dst_buf[CPY_BYTES / 4] __attribute__((aligned(4)));

// front-end
static int16_t fft_re[FFT_N] __attribute__((aligned(4)));
static int16_t fft_im[FFT_N] __attribute__((aligned(4)));
static int32_t pwr[FFT_N / 2 + 1];
static int32_t mel[N_MEL];
static int32_t mfcc[N_MFCC];

static volatile int32_t sink;
static int32_t csum(const int8_t *p, int n) { int32_t s = 0; for (int i = 0; i < n; i++) s += p[i]; return s; }
static inline uint32_t wall(void) { return timer_get_cycles(); }
static uint32_t xs = 2463534242u;
static inline int8_t rnd8(void) { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return (int8_t)xs; }

// ---------------------------------------------------------------- kernels
__attribute__((noinline)) int32_t dot8(const int8_t *x, const int8_t *w, int n) {
    int32_t a = 0;
    for (int i = 0; i < n; i++) a += x[i] * w[i];
    return a;
}

// word loads, 4 MACs per iteration (CMSIS-NN style without packed SIMD)
__attribute__((noinline)) int32_t dot8_w(const int8_t *x, const int8_t *w, int n) {
    int32_t a = 0;
    const uint32_t *xp = (const uint32_t *)x, *wp = (const uint32_t *)w;
    for (int i = 0; i < n / 4; i++) {
        uint32_t xv = xp[i], wv = wp[i];
        a += (int8_t)(xv) * (int8_t)(wv);
        a += (int8_t)(xv >> 8) * (int8_t)(wv >> 8);
        a += (int8_t)(xv >> 16) * (int8_t)(wv >> 16);
        a += (int8_t)(xv >> 24) * (int8_t)(wv >> 24);
    }
    return a;
}

static inline int8_t requant_simple(int32_t a, int32_t mult, int shift, int32_t zp) {
    int32_t v = ((a * mult) >> shift) + zp;
    return (int8_t)(v < 0 ? 0 : (v > 127 ? 127 : v));          // fused ReLU clamp
}

// TFLite-Micro MultiplyByQuantizedMultiplier (64-bit rounding doubling high mul)
static inline int8_t requant_tflm(int32_t a, int32_t mult, int shift, int32_t zp) {
    int64_t p = (int64_t)a * (int64_t)mult;
    int32_t hi = (int32_t)((p + (1ll << 30)) >> 31);
    int32_t mask = (1 << shift) - 1, rem = hi & mask, thr = (mask >> 1) + (hi < 0);
    int32_t v = (hi >> shift) + (rem > thr) + zp;
    return (int8_t)(v < 0 ? 0 : (v > 127 ? 127 : v));
}

__attribute__((noinline)) void requant_vec_simple(const int32_t *a, int8_t *o, int n) {
    for (int i = 0; i < n; i++) o[i] = requant_simple(a[i], 1234, 16, 3);
}
__attribute__((noinline)) void requant_vec_tflm(const int32_t *a, int8_t *o, int n) {
    for (int i = 0; i < n; i++) o[i] = requant_tflm(a[i], 1518500250, 9, 3);
}

__attribute__((noinline)) void conv3x3(void) {
    for (int y = 0; y < C3_H; y++)
        for (int x = 0; x < C3_H; x++)
            for (int co = 0; co < C3_C; co++) {
                int32_t a = 0;
                const int8_t *w = &c3_w[co * 9 * C3_C];
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++)
                        a += dot8_w(&c3_in[((y + ky) * (C3_H + 2) + (x + kx)) * C3_C], &w[(ky * 3 + kx) * C3_C], C3_C);
                c3_out[(y * C3_H + x) * C3_C + co] = requant_simple(a, 1234, 16, 3);
            }
}

__attribute__((noinline)) void pw1x1(void) {
    for (int p = 0; p < KH * KW; p++)
        for (int co = 0; co < KC; co++) {
            int32_t a = dot8_w(&pw_in[p * KC], &pw_w[co * KC], KC);
            pw_out[p * KC + co] = requant_simple(a, 1234, 16, 3);
        }
}

__attribute__((noinline)) void dw3x3(void) {
    for (int y = 0; y < KH; y++)
        for (int x = 0; x < KW; x++)
            for (int c = 0; c < KC; c++) {
                int32_t a = 0;
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++)
                        a += dw_in[((y + ky) * (KW + 2) + (x + kx)) * KC + c] * dw_w[(ky * 3 + kx) * KC + c];
                pw_out[(y * KW + x) * KC + c] = requant_simple(a, 1234, 16, 3);
            }
}

__attribute__((noinline)) void cpu_copy(uint32_t *d, const uint32_t *s, int words) {
    for (int i = 0; i < words; i++) d[i] = s[i];
}

// 512-pt radix-2 DIT complex FFT, Q15 with 1-bit scaling per stage
__attribute__((noinline)) void fft512(int16_t *re, int16_t *im) {
    const int n = FFT_N;
    for (int i = 1, j = 0; i < n; i++) {                      // bit reversal
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { int16_t t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < half; k++) {
                int32_t wr = tw_cos[k * step], wi = tw_sin[k * step];
                int a = i + k, b = a + half;
                int32_t tr = (re[b] * wr - im[b] * wi) >> 15;
                int32_t ti = (re[b] * wi + im[b] * wr) >> 15;
                int32_t ar = re[a], ai = im[a];
                re[a] = (int16_t)((ar + tr) >> 1); im[a] = (int16_t)((ai + ti) >> 1);
                re[b] = (int16_t)((ar - tr) >> 1); im[b] = (int16_t)((ai - ti) >> 1);
            }
    }
}

static inline int32_t log2_q8(uint32_t v) {                    // log2 in Q8 via CLZ + linear mantissa
    if (!v) return 0;
    int msb = 31 - __builtin_clz(v);
    uint32_t frac = msb >= 8 ? (v >> (msb - 8)) & 0xff : (v << (8 - msb)) & 0xff;
    return (msb << 8) | frac;
}

__attribute__((noinline)) void power_mel_log_dct(void) {
    for (int k = 0; k <= FFT_N / 2; k++) pwr[k] = fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k];
    const int16_t *w = mel_w;
    for (int m = 0; m < N_MEL; m++) {
        int32_t a = 0;
        for (int j = 0; j < mel_len[m]; j++) a += (pwr[mel_start[m] + j] >> 8) * w[j];
        w += mel_len[m];
        mel[m] = log2_q8((uint32_t)a);
    }
    for (int c = 0; c < N_MFCC; c++) {
        int32_t a = 0;
        for (int m = 0; m < N_MEL; m++) a += mel[m] * dct_m[c][m];
        mfcc[c] = a >> 15;
    }
}

// ---------------------------------------------------------------- main
int main(void) {
    printf("census_microbench start\n");
    for (int i = 0; i < NV; i++) { va[i] = rnd8(); vb[i] = rnd8(); acc32[i] = (int32_t)(xs * 2654435761u) >> 8; }
    for (unsigned i = 0; i < sizeof(c3_in); i++) c3_in[i] = rnd8();
    for (unsigned i = 0; i < sizeof(c3_w); i++) c3_w[i] = rnd8();
    for (unsigned i = 0; i < sizeof(pw_in); i++) pw_in[i] = rnd8();
    for (unsigned i = 0; i < sizeof(pw_w); i++) pw_w[i] = rnd8();
    for (unsigned i = 0; i < sizeof(dw_in); i++) dw_in[i] = rnd8();
    for (unsigned i = 0; i < sizeof(dw_w); i++) dw_w[i] = rnd8();
    for (int i = 0; i < CPY_BYTES / 4; i++) src_buf[i] = xs += 0x9e3779b9u;
    for (int i = 0; i < FFT_N; i++) { fft_re[i] = (int16_t)(rnd8() * 64); fft_im[i] = 0; }

    uint32_t t0, t1;
    printf("census_microbench: CV32E20 rv32imc, mcycle deltas\n");

    sink = dot8(va, vb, NV);
    t0 = cyc(); sink = dot8(va, vb, NV); t1 = cyc();             REPORT("dot8_naive", t1 - t0, NV, "MAC");
    t0 = cyc(); sink = dot8_w(va, vb, NV); t1 = cyc();           REPORT("dot8_wordload", t1 - t0, NV, "MAC");
    t0 = cyc(); conv3x3(); t1 = cyc(); sink = csum(c3_out, sizeof(c3_out)); REPORT("conv3x3_16x16", t1 - t0, C3_H * C3_H * C3_C * 9 * C3_C, "MAC");
    t0 = cyc(); pw1x1(); t1 = cyc(); sink = csum(pw_out, sizeof(pw_out)); REPORT("pw1x1_kws", t1 - t0, KH * KW * KC * KC, "MAC");
    t0 = cyc(); dw3x3(); t1 = cyc(); sink = csum(pw_out, sizeof(pw_out)); REPORT("dw3x3_kws", t1 - t0, KH * KW * KC * 9, "MAC");
    t0 = cyc(); requant_vec_simple(acc32, q8, NV); t1 = cyc();   REPORT("requant_simple", t1 - t0, NV, "elem");
    t0 = cyc(); requant_vec_tflm(acc32, q8, NV); t1 = cyc();     REPORT("requant_tflm", t1 - t0, NV, "elem");
    t0 = cyc(); cpu_copy(dst_buf, src_buf, CPY_BYTES / 4); t1 = cyc(); REPORT("cpu_copy_4KiB", t1 - t0, CPY_BYTES, "B");

    timer_cycles_init();
    timer_start();
    uint32_t w0, w1, wcal;
    w0 = wall(); w1 = wall(); wcal = w1 - w0;
    printf("wall_read_overhead %u cyc\n", (unsigned)wcal);
    dma_sdk_init();
    dma_copy((uint32_t)dst_buf, (uint32_t)src_buf, CPY_BYTES / 4, 0, DMA_DATA_TYPE_WORD, DMA_DATA_TYPE_WORD, 0);
    w0 = wall(); t0 = cyc();
    dma_copy((uint32_t)dst_buf, (uint32_t)src_buf, CPY_BYTES / 4, 0, DMA_DATA_TYPE_WORD, DMA_DATA_TYPE_WORD, 0);
    t1 = cyc(); w1 = wall();
    REPORT("dma4KiB_cpu_active", t1 - t0, CPY_BYTES, "B");
    REPORT("dma4KiB_wall", w1 - w0 - wcal, CPY_BYTES, "B");
    w0 = wall(); t0 = cyc();
    for (int r = 0; r < 16; r++)
        dma_copy((uint32_t)dst_buf, (uint32_t)src_buf, 64 / 4, 0, DMA_DATA_TYPE_WORD, DMA_DATA_TYPE_WORD, 0);
    t1 = cyc(); w1 = wall();
    REPORT("dma64B_cpu_active", t1 - t0, 16, "xfer");
    REPORT("dma64B_wall", w1 - w0 - wcal, 16, "xfer");
    w0 = wall(); t0 = cyc();
    for (int r = 0; r < 16; r++)
        dma_copy((uint32_t)dst_buf, (uint32_t)src_buf, 4 / 4, 0, DMA_DATA_TYPE_WORD, DMA_DATA_TYPE_WORD, 0);
    t1 = cyc(); w1 = wall();
    REPORT("dma4B_cpu_active", t1 - t0, 16, "xfer");
    REPORT("dma4B_wall", w1 - w0 - wcal, 16, "xfer");
    w0 = wall(); t0 = cyc(); fft512(fft_re, fft_im); t1 = cyc(); w1 = wall();
    printf("fft512 mcycle %u wall %u (check mcycle==wall when no WFI)\n", (unsigned)(t1 - t0), (unsigned)(w1 - w0 - wcal));

    volatile uint32_t *mmio = (volatile uint32_t *)(DMA_START_ADDRESS + 3 * DMA_CH_SIZE + DMA_SRC_PTR_REG_OFFSET);
    t0 = cyc();
    for (int r = 0; r < 64; r++) *mmio = r;
    t1 = cyc();                                                  REPORT("mmio_write", t1 - t0, 64, "wr");
    uint32_t s = 0;
    t0 = cyc();
    for (int r = 0; r < 64; r++) s += *mmio;
    t1 = cyc(); sink = s;                                        REPORT("mmio_read", t1 - t0, 64, "rd");

    t0 = cyc(); fft512(fft_re, fft_im); t1 = cyc();             REPORT("fft512_cplx_q15", t1 - t0, 1, "frame");
    t0 = cyc(); power_mel_log_dct(); t1 = cyc();                REPORT("pwr_mel_log_dct", t1 - t0, 1, "frame");
    sink = mfcc[0] + csum(q8, NV) + (int32_t)dst_buf[7];

    printf("done\n");
    return EXIT_SUCCESS;
}
