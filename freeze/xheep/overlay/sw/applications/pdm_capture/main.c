// PDM microphone bring-up for X-HEEP on the Nexys A7 (onboard ADMP421, J5/H5).
//
// PDM2PCM (CIC-only) configuration, 15 MHz system clock:
//   CLKDIVIDX 12 -> div 6 -> 2.5 MHz -> PDM clock 1.25 MHz (mic range 1.0-3.3 MHz)
//   4 CIC stages, decimation 16 (DECIMCIC 15), comb delay 1 -> 78.125 kHz, 18-bit two's complement
// FPGA: the CPU drains the 4-entry FIFO and applies a 20-tap Q14 FIR with decimation 5
//   -> 15.625 kHz int16, 1 s capture, then prints base64 between PCM_BEGIN/PCM_END.
// Simulation: the testharness pdm2pcm_dummy replays hw/ip/pdm2pcm/tb/signals/pdm.txt; the raw
//   CIC samples are printed in hex for a bit-exact comparison with kws_bringup/cic_model.py.

#include <stdint.h>
#include <stdio.h>
#include "core_v_mini_mcu.h"
#include "x-heep.h"
#include "mmio.h"
#include "pdm2pcm_regs.h"
#include "fir_taps.h"
#if !TARGET_SIM
#include "gpio.h"
#define BTNC 0           // gpio_io[0], centre button (N17), active high
#define LED_READY 5      // gpio_io[5] -> LD10 (U14): waiting for the button
#define LED_REC 10       // gpio_io[10] -> LD0 (H17): recording
#endif

#ifndef PDM2PCM_IS_INCLUDED
#error "PDM2PCM is not included: build with X_HEEP_CFG=configs/kws_nexys.py"
#endif

#define CLKDIVIDX 12
#define CIC_STAGES 0xF
#define CIC_DECIM 15
#define CIC_DELAY 1

#if TARGET_SIM
#define NRAW 2048
static int32_t raw[NRAW];
#else
#define FS_OUT 15625
#define NOUT FS_OUT
static int16_t pcm[NOUT];
#endif

static mmio_region_t pdm;

static inline int pdm_empty(void) {
    return mmio_region_read32(pdm, PDM2PCM_STATUS_REG_OFFSET) & (1u << PDM2PCM_STATUS_EMPTY_BIT);
}

static inline int32_t pdm_pop(void) {
    int32_t v = (int32_t)(mmio_region_read32(pdm, PDM2PCM_RXDATA_REG_OFFSET) & 0x3FFFF);
    return (v & 0x20000) ? v - 0x40000 : v;                 // sign-extend 18 bits
}

static void pdm_start(void) {
    mmio_region_write32(pdm, PDM2PCM_CONTROL_REG_OFFSET, 1u << PDM2PCM_CONTROL_CLEAR_BIT);
    mmio_region_write32(pdm, PDM2PCM_CLKDIVIDX_REG_OFFSET, CLKDIVIDX);
    mmio_region_write32(pdm, PDM2PCM_CIC_ACTIVATED_STAGES_REG_OFFSET, CIC_STAGES);
    mmio_region_write32(pdm, PDM2PCM_DECIMCIC_REG_OFFSET, CIC_DECIM);
    mmio_region_write32(pdm, PDM2PCM_CIC_DELAY_COMB_REG_OFFSET, CIC_DELAY);
    mmio_region_write32(pdm, PDM2PCM_CONTROL_REG_OFFSET, 1u << PDM2PCM_CONTROL_ENABL_BIT);
}

static void pdm_stop(void) { mmio_region_write32(pdm, PDM2PCM_CONTROL_REG_OFFSET, 0); }

#if !TARGET_SIM
static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void print_base64(const uint8_t *p, int n) {
    char line[77];
    int k = 0;
    for (int i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        line[k++] = b64[(v >> 18) & 63];
        line[k++] = b64[(v >> 12) & 63];
        line[k++] = i + 1 < n ? b64[(v >> 6) & 63] : '=';
        line[k++] = i + 2 < n ? b64[v & 63] : '=';
        if (k == 76) { line[k] = 0; printf("%s\n", line); k = 0; }
    }
    if (k) { line[k] = 0; printf("%s\n", line); }
}
#endif

int main(void) {
    pdm = mmio_region_from_addr((uintptr_t)PDM2PCM_START_ADDRESS);
    printf("pdm_capture: clkdiv %d, CIC %d stages, R %d, D %d\n", CLKDIVIDX, 4, CIC_DECIM + 1, CIC_DELAY);
    pdm_start();
#if TARGET_SIM
    for (int i = 0; i < NRAW; i++) {
        while (pdm_empty()) {}
        raw[i] = pdm_pop();
    }
    pdm_stop();
    printf("RAW_BEGIN %d\n", NRAW);
    for (int i = 0; i < NRAW; i += 8)
        printf("%05x %05x %05x %05x %05x %05x %05x %05x\n", raw[i] & 0xFFFFF, raw[i + 1] & 0xFFFFF, raw[i + 2] & 0xFFFFF,
               raw[i + 3] & 0xFFFFF, raw[i + 4] & 0xFFFFF, raw[i + 5] & 0xFFFFF, raw[i + 6] & 0xFFFFF, raw[i + 7] & 0xFFFFF);
    printf("RAW_END\n");
#else
    gpio_cfg_t led1 = {.pin = LED_READY, .mode = GpioModeOutPushPull};
    gpio_cfg_t led2 = {.pin = LED_REC, .mode = GpioModeOutPushPull};
    gpio_cfg_t btn = {.pin = BTNC, .mode = GpioModeIn, .en_input_sampling = true};
    gpio_config(led1); gpio_config(led2); gpio_config(btn);
    gpio_write(LED_READY, false); gpio_write(LED_REC, false);
    // let the microphone settle (0.5 s), keep draining the FIFO
    for (int32_t k = 0; k < 39062; k++) { while (pdm_empty()) {} (void)pdm_pop(); }
    printf("READY: press BTNC (centre button) and speak; LD10 = waiting, LD0 = recording\n");
    gpio_write(LED_READY, true);
    bool pressed = false;
    for (int32_t k = 0; k < 78125 * 60 && !pressed; k++) {   // wait up to 60 s, then record anyway
        while (pdm_empty()) {}
        (void)pdm_pop();
        if ((k & 63) == 0) gpio_read(BTNC, &pressed);
    }
    gpio_write(LED_READY, false);
    gpio_write(LED_REC, true);
    int32_t ring[FIR_TAPS] = {0};
    int head = 0, phase = 0, n = 0, skip = FIR_TAPS;          // refill the FIR before the first output
    uint32_t overrun = 0;
    while (n < NOUT) {
        if (pdm_empty()) continue;
        if (mmio_region_read32(pdm, PDM2PCM_STATUS_REG_OFFSET) & (1u << PDM2PCM_STATUS_FULLL_BIT)) overrun++;
        int32_t x = pdm_pop() >> 2;                           // 16-bit range
        if (skip) { skip--; continue; }
        ring[head] = x;
        head = head == FIR_TAPS - 1 ? 0 : head + 1;
        if (++phase == FIR_DECIM) {
            phase = 0;
            int32_t acc = 0;
            int idx = head;
            for (int k = 0; k < FIR_TAPS; k++) {             // oldest .. newest
                acc += ring[idx] * fir_q14[k];
                idx = idx == FIR_TAPS - 1 ? 0 : idx + 1;
            }
            acc >>= 14;
            pcm[n++] = (int16_t)(acc > 32767 ? 32767 : (acc < -32768 ? -32768 : acc));
        }
    }
    pdm_stop();
    gpio_write(LED_REC, false);
    printf("recorded (button %s)\n", pressed ? "pressed" : "timeout");
    int32_t mn = 32767, mx = -32768;
    int64_t s = 0, s2 = 0;
    for (int i = 0; i < NOUT; i++) {
        mn = pcm[i] < mn ? pcm[i] : mn;
        mx = pcm[i] > mx ? pcm[i] : mx;
        s += pcm[i];
        s2 += (int64_t)pcm[i] * pcm[i];
    }
    printf("stats: n %d min %d max %d mean %d meansq %u fifo_full_seen %u\n", NOUT, (int)mn, (int)mx, (int)(s / NOUT),
           (unsigned)(s2 / NOUT), (unsigned)overrun);
    printf("PCM_BEGIN fs %d n %d int16le base64\n", FS_OUT, NOUT);
    print_base64((const uint8_t *)pcm, NOUT * 2);
    printf("PCM_END\n");
#endif
    printf("done\n");
    return 0;
}
