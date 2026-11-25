// HopCIM R7: live keyword spotting on the Nexys A7's on-board microphone, the CPU asleep.
// The endpoint's audio path (rtl/cim_audio.sv) clocks the PDM microphone (1.024 MHz), decimates
// to 16 kHz, and every 20 ms hop runs the front-end, the window path and the DS-CNN by itself.
// The CPU loads the tiles and parameters, enables the audio path, checks the microphone level once
// (1 s of audio read back from the ring), and then only wakes on the decision interrupt: it reads
// the class and prints a keyword when it has been the top class for KW_HOLD decisions in a row
// (silence and unknown are never printed). Every 10 s it prints statistics and how often each class
// was the decision.
// LEDs, through X-HEEP's Nexys pin map: GPIO 5..8 (LD10..LD13) show the last keyword for 1 s, as
// 1 (down) .. 10 (yes) in binary with LD10 the least significant bit; GPIO 9 (LD14) is lit while the
// decision is not _silence_; GPIO 10 (LD0) blinks at 1 Hz while decisions arrive.
// Build: make app PROJECT=hopcim_live TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
// and run it on a bitstream built with --flag use_cim_mic (15.36 MHz).
//
// CPU-active cycles per decision: mcycle (stopped by mcountinhibit around WFI) from one loop
// iteration to the next, i.e. everything the CPU does per decision: interrupt entry, PLIC claim and
// complete, the handler, reading and clearing the decision, the keyword and LED logic, the sleep
// loop. Iterations that print are counted apart.
//
// KW_TEST (app hopcim_kwtest): a scripted keyword test instead of the free-running demo. Each of
// KW_TEST rounds prompts the 10 keywords, two words of the _unknown_ class and one quiet trial in a
// shuffled order, one every TRIAL_DEC decisions (4 s). Prompts and keywords carry the decision
// number (20 ms each); the statistics come once, at the end, followed by "done".
// hopcim/model/score_kwtest.py scores the log.
//
// NET_K (default 1, range 1..16; COMPILER_FLAGS=-DNET_K=<k>): the hardware runs the network and makes
// a decision every k hops (AUD_CTRL[19:16] = k - 1), on a bitstream with that field. Everything timed
// in decisions keeps its duration: the keyword hold spans at least the same 80 ms (1 + ceil(4 / k)
// decisions: 5, 3, 3, 2, 2 for k = 1 .. 5, then 2), the statistics, the LED times and the test's prompts
// stay in seconds, and the test's @n stays in 20-ms units (hops).
//
// KW_SUM (default 0; COMPILER_FLAGS=-DKW_SUM=<n>): instead of the class of each decision, the rule uses the top
// class of the last n decisions' logits summed (hopcim/sw/kw_rule.h), and reports a keyword as soon as it is that
// top class (a hold of 1). The low-power mode U of docs/WHOLEWORD_STUDY.md is -DNET_K=16 -DKW_SUM=2: the network
// every 320 ms, the sum of the last 2 decisions. LD14 still shows each decision's own class.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "csr.h"
#include "rv_plic.h"
#include "gpio.h"
#include "gpio_structs.h"
#include "core_v_mini_mcu.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/mfcc_tables.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"

#ifndef NET_K
#define NET_K 1                                      // a decision every NET_K hops
#endif
#if (NET_K) < 1 || (NET_K) > 16
#error "NET_K must be 1..16"
#endif
#ifndef KW_SUM
#define KW_SUM 0                                     // > 0: decide on the last KW_SUM decisions' summed logits
#endif
#if (KW_SUM) > 0
#include "../../../hw/ip/hopcim/sw/kw_rule.h"
#define KW_HOLD   1                                  // the summed class reports a keyword at once
#else
#define KW_HOLD   (1 + (3 + (NET_K)) / (NET_K))      // decisions in a row (spanning >= 80 ms) to report a keyword
#endif
#define STAT_DEC  (500 / (NET_K))                    // statistics every 10 s
#define LED_HOLD  (50 / (NET_K))                     // decisions a keyword stays on the LEDs (1 s)
#define LED_BLINK (25 / (NET_K))                     // decisions per half period of LD0
#define GPIO_AO   ((volatile gpio *)GPIO_AO_START_ADDRESS)   // GPIO 0..7
#define GPIO_PER  ((volatile gpio *)GPIO_START_ADDRESS)      // GPIO 8..31
#define CIM_AUD_CTRL   (CIM_BASE + 0x0034u)
#define CIM_AUD_STATUS (CIM_BASE + 0x0038u)
#define CIM_AUD_HOPS   (CIM_BASE + 0x003Cu)
#define CIM_IRQ_AUD    0x8u

static const char *const LABEL[12] = {"down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes",
                                      "_silence_", "_unknown_"};

static inline uint32_t cyc(void) { uint32_t x; asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }

#ifdef KW_TEST
#define TRIAL_DEC 200                                // 20-ms units per trial (4 s)
#define LEAD_DEC  250                                // 20-ms units before the first prompt (5 s)
#define N_TRIAL   (13u * (KW_TEST))
static const char *const OTHER[10] = {"bird", "house", "happy", "seven", "wow",   // GSC words of the
                                      "dog", "cat", "tree", "marvin", "zero"};    // _unknown_ class
static uint32_t lcg = 0x4B575453u;                   // fixed seed: every run prompts the same order

static uint32_t rnd(uint32_t n) {
    lcg = lcg * 1664525u + 1013904223u;
    return (lcg >> 16) % n;
}
#endif

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
#if CIM_DWU
            if (CIM_DWU == 4 || l == 0) continue;      // on the digital depthwise unit: from its store, below
#endif
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
#if CIM_DWU
    cim_load_dwstore(DS_W_DW);
#endif
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

// ---------------------------------------------------------------- LEDs
static uint32_t led_now;

static void leds(uint32_t v) {                       // bit n = GPIO n, n = 5..10
    if (v == led_now) return;
    led_now = v;
    GPIO_AO->GPIO_SET0 = v & 0x0E0u;
    GPIO_AO->GPIO_CLEAR0 = ~v & 0x0E0u;
    GPIO_PER->GPIO_SET0 = v & 0x700u;
    GPIO_PER->GPIO_CLEAR0 = ~v & 0x700u;
}

static int leds_init(void) {
    for (int p = 5; p <= 10; p++) {
        gpio_cfg_t cfg = {.pin = p, .mode = GpioModeOutPushPull};
        if (gpio_config(cfg) != GpioOk) return -1;
    }
    led_now = 0x7E0u;
    leds(0);
    return 0;
}

static volatile int dec_irq;

static void cim_irq_handler(uint32_t id) {
    (void)id;
    dec_irq = 1;                                     // the main loop reads and clears the decision
    CIM_REG(CIM_IRQ_EN) = 0;                         // level interrupt: off until the decision is read
}

static void sleep_until(volatile int *flag) {
    CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    while (!*flag) {
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 1);
        asm volatile("wfi");
        CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);
        CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
        CSR_CLEAR_BITS(CSR_REG_MSTATUS, 0x8);
    }
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
}

int main(void) {
    cim_init();
    for (int t = 0; t < MFCC_NTILES; t++) cim_load_tile(t, &MFCC_TILES[t][0][0]);
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
    if (leds_init()) {
        printf("GPIO setup failed\n");
        return EXIT_FAILURE;
    }

    CIM_REG(CIM_AUD_CTRL) = 0x3u | ((uint32_t)DS_T_CONV1 << 8) | ((uint32_t)((NET_K) - 1) << 16);   // microphone on, network on
    printf("HopCIM live: listening at 16 kHz (system clock %u Hz)\n", (unsigned)REFERENCE_CLOCK_Hz);
#if (KW_SUM) > 0
    printf("network every %u hops (%u ms); a keyword is the top class of the last %u decisions' summed logits\n",
           (unsigned)(NET_K), 20u * (NET_K), (unsigned)(KW_SUM));
#elif (NET_K) > 1
    printf("network every %u hops (%u ms); a keyword needs %u decisions in a row\n", (unsigned)(NET_K), 20u * (NET_K), (unsigned)KW_HOLD);
#endif
    printf("LEDs: LD10-LD13 keyword 1 (down) .. 10 (yes), LD14 not silence, LD0 blinks at 1 Hz\n");

    // microphone check: after 1 s (50 hops), the last 1024 samples of the ring
    while (CIM_REG(CIM_AUD_HOPS) < 50) {
    }
    int32_t mn = 32767, mx = -32768;
    int64_t sum = 0, sq = 0;
    for (int i = 0; i < CIM_RING; i++) {
        int32_t v = (int32_t)CIM_REG(CIM_SAMPLES(i));
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sq += (int64_t)v * v;
    }
    int32_t mean = (int32_t)(sum / CIM_RING);
    uint32_t ms = (uint32_t)(sq / CIM_RING - (int64_t)mean * mean);
    uint32_t rms = 0;
    while ((rms + 1) * (rms + 1) <= ms) rms++;       // isqrt (once)
    printf("microphone: 1024 samples, min %d max %d mean %d rms %u\n", (int)mn, (int)mx, (int)mean, (unsigned)rms);
#ifdef KW_TEST
    printf("KWTEST rounds %u trials %u period %u lead %u\n", (unsigned)(KW_TEST), (unsigned)N_TRIAL, TRIAL_DEC, LEAD_DEC);
    printf("Say each word once when it appears, 20-50 cm from the board; say nothing at (stay quiet).\n");
    printf("The first word comes in %u s.\n", LEAD_DEC / 50);
    uint32_t trial = 0, next = LEAD_DEC;
    uint8_t ord[13];
#endif

    int last = -1, run = 0, said = -1, printed = 1;
#if (KW_SUM) > 0
    static kw_rule_t rule;
    kw_rule_init(&rule);
#endif
    uint32_t n_dec = 0, n_kw = 0, c_sum = 0, c_max = 0, c_n = 0, missed = 0, t_prev = 0;
    uint32_t hist[12] = {0}, kw_led = 0, kw_left = 0, blink = LED_BLINK, hb = 0;
    uint16_t prev_cnt = (uint16_t)(CIM_REG(CIM_AUD_STATUS) >> 16);
    CIM_REG(CIM_AUD_STATUS) = 0x6u;                  // clear pending, overrun
    for (;;) {
        uint32_t t = cyc();                          // CPU-active since the previous iteration
        if (!printed) {
            uint32_t c = t - t_prev;
            c_sum += c;
            c_n++;
            if (c > c_max) c_max = c;
        }
        t_prev = t;
        printed = 0;
        dec_irq = 0;
        CIM_REG(CIM_IRQ_EN) = CIM_IRQ_AUD;
        sleep_until(&dec_irq);
        uint32_t st = CIM_REG(CIM_AUD_STATUS);
        CIM_REG(CIM_AUD_STATUS) = 0x2u;              // decision read
        uint16_t cnt = (uint16_t)(st >> 16);
        missed += (uint16_t)(cnt - prev_cnt) - 1u;
        prev_cnt = cnt;
        int cls = (int)((st >> 4) & 0xFu);
        if (cls < 12) hist[cls]++;
#if (KW_SUM) > 0
        int8_t lg[12];
        for (int c = 0; c < 12; c++) lg[c] = (int8_t)CIM_REG(CIM_LOGIT(c));
        int kw = kw_rule_step(&rule, lg, KW_HOLD);   // the keyword to report, or -1
        int say = kw >= 0;
        (void)last;
        (void)run;
        (void)said;
#else
        run = (cls == last) ? run + 1 : 1;
        last = cls;
        int say = (cls < 10) && (run == KW_HOLD) && (cls != said);
        if (cls >= 10) said = -1;
        int kw = cls;
#endif
        n_dec++;
        if (kw_left && --kw_left == 0) kw_led = 0;
        if (--blink == 0) {
            blink = LED_BLINK;
            hb ^= 1u;
        }
        if (say) {
#if (KW_SUM) == 0
            said = cls;
#endif
            n_kw++;
            kw_led = (uint32_t)kw + 1u;
            kw_left = LED_HOLD;
        }
        leds(kw_led << 5 | (uint32_t)(cls != 10) << 9 | hb << 10);
#ifdef KW_TEST
        uint32_t now = (n_dec + missed) * (NET_K);   // decisions since the start, in 20-ms units
        if (say) {
            printf(">> %s @%u\n", LABEL[kw], (unsigned)now);
            printed = 1;
        }
        if (now >= next) {
            if (trial == N_TRIAL) break;
            uint32_t pos = trial % 13u, r = trial / 13u;
            if (pos == 0) {                          // a new round: shuffle (Fisher-Yates)
                for (int i = 0; i < 13; i++) ord[i] = (uint8_t)i;
                for (int i = 12; i > 0; i--) {
                    int j = (int)rnd((uint32_t)i + 1u);
                    uint8_t x = ord[i];
                    ord[i] = ord[j];
                    ord[j] = x;
                }
            }
            int it = ord[pos];
            trial++;
            if (it < 10)
                printf("[%u/%u] say: %s @%u\n", (unsigned)trial, (unsigned)N_TRIAL, LABEL[it], (unsigned)now);
            else if (it < 12)
                printf("[%u/%u] say: %s (not a keyword) @%u\n", (unsigned)trial, (unsigned)N_TRIAL,
                       OTHER[(2u * r + (uint32_t)it - 10u) % 10u], (unsigned)now);
            else
                printf("[%u/%u] (stay quiet) @%u\n", (unsigned)trial, (unsigned)N_TRIAL, (unsigned)now);
            next += TRIAL_DEC;
            printed = 1;
        }
#else
        if (say) {
            printf(">> %s\n", LABEL[kw]);
            printed = 1;
        }
        if (n_dec % STAT_DEC == 0) {
            printed = 1;
            printf("[%u s] %u decisions (%u missed while printing), %u keywords, overrun %u; CPU-active per decision %u cycles (max %u)\n",
                   (unsigned)(n_dec * (NET_K) / 50u), (unsigned)n_dec, (unsigned)missed, (unsigned)n_kw, (unsigned)((st >> 2) & 1u),
                   (unsigned)(c_n ? c_sum / c_n : 0), (unsigned)c_max);
            printf("    classes:");
            for (int c = 0; c < 12; c++)
                if (hist[c]) printf(" %s %u", LABEL[c], (unsigned)hist[c]);
            printf("\n");
            for (int c = 0; c < 12; c++) hist[c] = 0;
        }
#endif
    }
#ifdef KW_TEST
    printf("end @%u: %u decisions (%u missed while printing), %u keywords, overrun %u; CPU-active per decision %u cycles (max %u)\n",
           (unsigned)((n_dec + missed) * (NET_K)), (unsigned)n_dec, (unsigned)missed, (unsigned)n_kw,
           (unsigned)((CIM_REG(CIM_AUD_STATUS) >> 2) & 1u), (unsigned)(c_n ? c_sum / c_n : 0), (unsigned)c_max);
    printf("    classes:");
    for (int c = 0; c < 12; c++)
        if (hist[c]) printf(" %s %u", LABEL[c], (unsigned)hist[c]);
    printf("\ndone\n");
#endif
    return EXIT_SUCCESS;
}
