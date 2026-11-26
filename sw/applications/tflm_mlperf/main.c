// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// The CPU-only baseline under MLPerf Tiny's own benchmark runner: TensorFlow Lite Micro (the runtime of
// MLPerf Tiny's reference submission) runs MLPerf Tiny's KWS model on X-HEEP's CV32E20, kws01, closed
// division, on the same board, bitstream family and clock as HopCIM (the baseline S with 256 KiB of RAM,
// xheep_nexys_bscan_cim128ks256k.bit, 15.36 MHz; the CIM endpoint is in the bitstream and never used here).
// The runner talks to this firmware exactly as to hopcim_mlperf: MLCommons' unchanged API code
// (internally_implemented.cpp) over the USB UART at 115200 baud; this file is the submitter's part.
//
//   th_final_initialize  the interpreter: the model, the six operators, the arena (tflm_glue.cpp)
//   th_load_tensor       the runner's 490 int8 features from the db buffer into a CPU array
//   th_infer             timed: the 490 bytes into the model's input tensor, then one interpreter Invoke (the
//                        model ends in its softmax). The reference copies the input in th_load_tensor and only
//                        invokes in th_infer; but TFLM's memory planner lets later layers reuse the input
//                        tensor's arena bytes, so a second Invoke without a new copy (the runner's "infer N W"
//                        runs N on one input) computes on overwritten input. Copying in th_infer keeps every
//                        output exact and times the input write, as hopcim_mlperf times its FEAT write.
//   th_results           the output as the reference prints it, with the reference's own float formula
//   th_timestamp         microseconds from mcycle/mcycleh (the CPU never sleeps here, and on the FPGA the
//                        clock gate is a pass-through, so mcycle is wall time)
//
// An extra command, "tflm%", prints this firmware's own counts (not used by the runner): setup status, the
// arena used, inferences, failed invokes, and the cycles of the input copy and of Invoke (minimum and maximum);
// "tflm-reset%" clears them.
//
// Build: mlperf/build_tflm.sh <X-HEEP tree made with configs/kws_nexys_256k.py> (TFLM needs C++17 and about
// 116 KiB of code and read-only data: patch 0005).

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csr.h"
#include "core_v_mini_mcu.h"
#include "x-heep.h"
#include "mmio.h"
#include "uart.h"
#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"

#define NIN 490
#define NCLS 12
#ifndef BAUD
#define BAUD 115200u
#endif

// api_glue.cpp: C entry points to the C++ API code; tflm_glue.cpp: the interpreter
void hop_ee_benchmark_initialize(void);
void hop_ee_serial_callback(char c);
void hop_ee_command(char *command);
size_t hop_ee_get_buffer(uint8_t *buffer, size_t max_len);
int tflm_setup(void);
int8_t *tflm_input(void);
const int8_t *tflm_output(void);
float tflm_output_scale(void);
int tflm_output_zero_point(void);
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

// ---------------------------------------------------------------- benchmark state
static uart_t uart;
static int setup_status = 1;                         // 1: not run; 0: ready; < 0: tflm_setup's failing step
static int8_t input_copy[NIN];
static uint32_t n_infer, n_fail, n_bad_input;
static uint64_t inv_min, inv_max, cp_min, cp_max;

static void stats_reset(void) {
    n_infer = n_fail = n_bad_input = 0;
    inv_min = cp_min = ~0ull;
    inv_max = cp_max = 0;
}

// ---------------------------------------------------------------- required core API
void th_load_tensor(void) {                          // the buffer, as the reference; the tensor in th_infer
    size_t bytes = hop_ee_get_buffer((uint8_t *)input_copy, NIN * sizeof(int8_t));
    if (bytes / sizeof(int8_t) != NIN) {
        th_printf("Input db has %d elemented, expected %d\n", (int)(bytes / sizeof(int8_t)), NIN);   // the reference's message
        n_bad_input++;
    }
}

void th_infer(void) {
    if (setup_status != 0) return;
    uint64_t c0 = cyc64();
    int8_t *t = tflm_input();                        // the reference's SetInput loop
    for (int i = 0; i < NIN; i++) t[i] = input_copy[i];
    uint64_t c1 = cyc64();
    int rc = tflm_invoke();
    uint64_t d = cyc64() - c1, dc = c1 - c0;
    n_infer++;
    n_fail += rc != 0;
    if (d < inv_min) inv_min = d;
    if (d > inv_max) inv_max = d;
    if (dc < cp_min) cp_min = dc;
    if (dc > cp_max) cp_max = dc;
}

// the reference's th_results: DequantizeInt8ToFloat, then three truncated decimals ("0.ddd")
void th_results(void) {
    th_printf("m-results-[");
    const int8_t *out = setup_status == 0 ? tflm_output() : 0;
    for (int i = 0; i < NCLS; i++) {
        float converted = out ? (float)(out[i] - tflm_output_zero_point()) * tflm_output_scale() : 0.0f;
        th_printf("0.%d", (int)(converted * 10));
        th_printf("%d", (int)(converted * 100) % 10);
        th_printf("%d", (int)(converted * 1000) % 10);
        if (i < NCLS - 1) th_printf(",");
    }
    th_printf("]\r\n");
}

// microseconds since th_timestamp_initialize
static uint64_t ts_epoch;

void th_timestamp(void) {
    unsigned long us = (unsigned long)((cyc64() - ts_epoch) * 1000000ull / REFERENCE_CLOCK_Hz);
    th_printf(EE_MSG_TIMESTAMP, us);
}

int th_vprintf(const char *format, va_list ap) {
    static char buf[192];
    int n = vsnprintf(buf, sizeof buf, format, ap);
    if (n > (int)sizeof buf - 1) n = (int)sizeof buf - 1;
    if (n > 0) uart_write(&uart, (const uint8_t *)buf, (size_t)n);
    return n;
}

void th_printf(const char *p_fmt, ...) {
    va_list args;
    va_start(args, p_fmt);
    (void)th_vprintf(p_fmt, args);
    va_end(args);
}

char th_getchar(void) {
    uint8_t c;
    uart_getchar(&uart, &c);
    return (char)c;
}

// ---------------------------------------------------------------- optional API
// X-HEEP's printf re-initialises the UART at 9600 baud on every call, so this firmware never uses it
void th_serialport_initialize(void) {
    uart.base_addr = mmio_region_from_addr((uintptr_t)UART_START_ADDRESS);
    uart.baudrate = BAUD;
    uart.clk_freq_hz = REFERENCE_CLOCK_Hz;
    uart.nco = (((uint64_t)BAUD * 16u) << 16) / REFERENCE_CLOCK_Hz;
    (void)uart_init(&uart);
}

void th_timestamp_initialize(void) {
    ts_epoch = cyc64();
    th_printf(EE_MSG_TIMESTAMP_MODE);               // "This message must NOT be changed."
    th_timestamp();
}

void th_final_initialize(void) {
    stats_reset();
    setup_status = tflm_setup();
    if (setup_status != 0) th_printf("e-[TFLM setup failed at step %d]\r\n", -setup_status);
}

void th_pre(void) {}
void th_post(void) {}

void th_command_ready(char volatile *p_command) {
    char *cmd = (char *)p_command;
    if (strcmp(cmd, "tflm") == 0) {
        th_printf("m-tflm-clock-%lu-Hz\r\n", (unsigned long)REFERENCE_CLOCK_Hz);
        th_printf("m-tflm-setup-%s-%d\r\n", setup_status == 0 ? "ok" : "FAILED", setup_status);
        if (setup_status == 0) {
            int s1000 = (int)(tflm_output_scale() * 256000.0f + 0.5f);   // 1000 for the expected 1/256
            th_printf("m-tflm-arena-used-%lu-of-%lu-bytes\r\n", (unsigned long)tflm_arena_used(), (unsigned long)tflm_arena_size());
            th_printf("m-tflm-output-zero-point-%d-scale-x256000-%d\r\n", tflm_output_zero_point(), s1000);
        }
        th_printf("m-tflm-infer-%lu-failed-%lu-bad-inputs-%lu\r\n", (unsigned long)n_infer, (unsigned long)n_fail,
                  (unsigned long)n_bad_input);
        if (n_infer) {
            th_printf("m-tflm-input-copy-%lu-%lu-cycles\r\n", (unsigned long)cp_min, (unsigned long)cp_max);
            th_printf("m-tflm-invoke-%lu-%lu-cycles\r\n", (unsigned long)inv_min, (unsigned long)inv_max);
        }
        th_printf(EE_MSG_READY);
        return;
    }
    if (strcmp(cmd, "tflm-reset") == 0) {
        stats_reset();
        th_printf(EE_MSG_READY);
        return;
    }
    hop_ee_command(cmd);
}

// ---------------------------------------------------------------- libc hooks (as the reference)
int th_strncmp(const char *str1, const char *str2, size_t n) { return strncmp(str1, str2, n); }
char *th_strncpy(char *dest, const char *src, size_t n) { return strncpy(dest, src, n); }
size_t th_strnlen(const char *str, size_t maxlen) { return strnlen(str, maxlen); }
char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }
char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }
int th_atoi(const char *str) { return atoi(str); }
void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }
void *th_memcpy(void *dst, const void *src, size_t n) { return memcpy(dst, src, n); }

int main(void) {
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);             // mcycle counts (th_timestamp, the Invoke cycles)
    hop_ee_benchmark_initialize();                   // serial port, timestamp, the interpreter; m-init-done, m-ready
    for (;;) hop_ee_serial_callback(th_getchar());
    return 0;
}
