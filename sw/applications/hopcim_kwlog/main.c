// HopCIM: the same-audio logger (docs/SAME_AUDIO.md). The audio path runs exactly as in hopcim_live,
// with a decision every 20 ms hop; the CPU polls and streams what the hardware does over the UART
// at 921,600 baud (the bitstream's UART is software configured; 9600 baud cannot carry the audio):
//   KWLOG_AUDIO = 1 (app hopcim_kwlog): every hop's 320 samples and every decision's 12 logits, so one
//     spoken run can be scored offline at every rate, phase and confirmation rule on identical audio,
//     and every decision can be checked against the golden model;
//   KWLOG_AUDIO = 0 (app hopcim_kwlog_noaudio, -DKWLOG_AUDIO=0): the decisions only, for long
//     false-alarm runs. No audio leaves the board, only each hop's level.
// KW_TEST = rounds of hopcim_kwtest's prompts (same order, same seed, same 4-s period); 0: no prompts,
// runs until the host stops it.
// This is a measurement tool: it polls, so its CPU-active cycles say nothing about the live path.
//
// Binary frames, little endian: A5 <type> <payload> <Fletcher-16 of type + payload, 2 bytes>
//   'H': protocol version (u8), audio (u8), rounds (u8), system clock (u32)    once, at the start
//   'A': hop (u32), 320 x int16                                                that hop's samples
//   'D': hop (u32), count (u16), class | overrun << 4 (u8), 12 x int8          a decision
//   'R': hop (u32), class (u8)                  hopcim_live's rule (5 decisions in a row), checks the offline scorer
//   'P': hop (u32), trial (u16), kind (u8: 0 keyword, 1 other word, 2 quiet), word (u8)   a prompt
//   'L': hop (u32), min (i16), max (i16), mean (i16), rms (u16)                level of the ring, every 10 s
//   'M': hop (u32), max (i16), min (i16), rms (u16)                            a hop's level (KWLOG_AUDIO = 0)
//   'E': hop (u32), frames lost (u32), overrun (u8)                            the test has ended
// Build: make app PROJECT=hopcim_kwlog TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
// Run on a bitstream built with --flag use_cim_mic: planenest/hw/board_kwlog.sh.

#include <stdint.h>
#include <stdlib.h>
#include "core_v_mini_mcu.h"
#include "mmio.h"
#include "uart.h"
#include "uart_regs.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/mfcc_tables.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"

#ifndef KWLOG_AUDIO
#define KWLOG_AUDIO 1
#endif
#ifndef KW_TEST
#define KW_TEST 5
#endif
#define KWLOG_BAUD 921600u
#define KW_HOLD    5                                 // hopcim_live's rule at a decision every hop
#define HOP_SAMPLES 320u
#define TRIAL_DEC  200                               // as hopcim_kwtest: 4 s per trial
#define LEAD_DEC   250                               // 5 s before the first prompt
#define N_TRIAL    (13u * (KW_TEST))
#define LEVEL_DEC  500u                              // a level frame every 10 s
#define CIM_AUD_CTRL   (CIM_BASE + 0x0034u)
#define CIM_AUD_STATUS (CIM_BASE + 0x0038u)
#define CIM_AUD_HOPS   (CIM_BASE + 0x003Cu)

// ---------------------------------------------------------------- transmit without blocking
#define TXB 8192u                                    // power of two
static uint8_t txb[TXB];
static volatile uint32_t tx_w, tx_r;
static mmio_region_t uart_base;
static uint32_t lost;
static uint32_t f1, f2;

static inline void tx_poll(void) {
    while (tx_r != tx_w) {
        if (mmio_region_read32(uart_base, UART_STATUS_REG_OFFSET) & (1u << UART_STATUS_TXFULL_BIT)) return;
        mmio_region_write32(uart_base, UART_WDATA_REG_OFFSET, txb[tx_r & (TXB - 1u)]);
        tx_r++;
    }
}

static inline uint32_t tx_room(void) { return TXB - (tx_w - tx_r); }

static inline void put8(uint32_t b) {
    txb[tx_w & (TXB - 1u)] = (uint8_t)b;
    tx_w++;
    f1 = (f1 + (b & 0xFFu)) % 255u;
    f2 = (f2 + f1) % 255u;
}

static inline void put16(uint32_t v) { put8(v); put8(v >> 8); }
static inline void put32(uint32_t v) { put16(v); put16(v >> 16); }

static void begin(uint32_t type) {
    txb[tx_w & (TXB - 1u)] = 0xA5u;
    tx_w++;
    f1 = f2 = 0;
    put8(type);
}

static void finish(void) {
    uint32_t a = f1, b = f2;
    txb[tx_w & (TXB - 1u)] = (uint8_t)a;
    tx_w++;
    txb[tx_w & (TXB - 1u)] = (uint8_t)b;
    tx_w++;
}

// ---------------------------------------------------------------- tiles (as hopcim_net)
static int8_t tbuf[64][16] __attribute__((aligned(4)));

static void put_tile(unsigned t) { cim_load_tile(t, &tbuf[0][0]); }

static void load_net_tiles(void) {
    for (int j = 0; j < 4; j++) {
        for (int r = 0; r < 64; r++)
            for (int n = 0; n < 16; n++) tbuf[r][n] = r < 40 ? DS_W_CONV1[16 * j + n][r] : 0;
        put_tile(DS_T_CONV1 + j);
    }
    for (int l = 0; l < 4; l++) {
        for (int g = 0; g < 10; g++) {
            for (int r = 0; r < 64; r++)
                for (int n = 0; n < 16; n++) tbuf[r][n] = 0;
            for (int n = 0; n < 7 && 7 * g + n < 64; n++)
                for (int k = 0; k < 9; k++) tbuf[7 * k + n][n] = DS_W_DW[l][7 * g + n][k];
            put_tile(DS_T_DW(l) + g);
        }
        for (int j = 0; j < 4; j++) {
            for (int r = 0; r < 64; r++)
                for (int n = 0; n < 16; n++) tbuf[r][n] = DS_W_PW[l][16 * j + n][r];
            put_tile(DS_T_PW(l) + j);
        }
    }
    for (int r = 0; r < 64; r++)
        for (int n = 0; n < 16; n++) tbuf[r][n] = n < 12 ? DS_W_FC[n][r] : 0;
    put_tile(DS_T_FC);
}

static int load_params(void) {
    for (int L = 0; L < 10; L++)
        for (int c = 0; c < 64; c++) {
            if (DS_SH[L][c] > 0) return -1;
            CIM_REG(CIM_PARAM(64 * L + c, 0)) = (uint32_t)DS_BIAS[L][c];
            CIM_REG(CIM_PARAM(64 * L + c, 1)) = (uint32_t)DS_QM[L][c];
            CIM_REG(CIM_PARAM(64 * L + c, 2)) = (uint32_t)(-DS_SH[L][c]);
        }
    CIM_REG(CIM_SEQ_ZP) = (uint32_t)(uint8_t)DS_ZIN0 | (uint32_t)(uint8_t)DS_ZACT << 8 | (uint32_t)(uint8_t)DS_ZFC << 16;
    return 0;
}

#if (KW_TEST) > 0
static uint32_t lcg = 0x4B575453u;                   // the seed of hopcim_kwtest: the same order

static uint32_t rnd(uint32_t n) {
    lcg = lcg * 1664525u + 1013904223u;
    return (lcg >> 16) % n;
}
#endif

static void level_frame(uint32_t hop) {              // statistics only: no sample leaves the board
    int32_t mn = 32767, mx = -32768;
    int64_t sum = 0, sq = 0;
    for (int i = 0; i < CIM_RING; i++) {
        int32_t v = (int32_t)CIM_REG(CIM_SAMPLES(i));
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sq += (int64_t)v * v;
        if ((i & 63) == 0) tx_poll();
    }
    int32_t mean = (int32_t)(sum / CIM_RING);
    uint32_t ms = (uint32_t)(sq / CIM_RING - (int64_t)mean * mean);
    uint32_t rms = 0;
    while ((rms + 1) * (rms + 1) <= ms) rms++;
    if (tx_room() < 32u) {
        lost++;
        return;
    }
    begin('L');
    put32(hop);
    put16((uint32_t)mn);
    put16((uint32_t)mx);
    put16((uint32_t)mean);
    put16(rms);
    finish();
}

int main(void) {
    uart_t uart;
    uart.base_addr = mmio_region_from_addr((uintptr_t)UART_START_ADDRESS);
    uart.baudrate = KWLOG_BAUD;
    uart.clk_freq_hz = REFERENCE_CLOCK_Hz;
    uart.nco = (uint32_t)(((uint64_t)KWLOG_BAUD << (NCO_WIDTH + 4)) / REFERENCE_CLOCK_Hz);
    if (uart_init(&uart) != kErrorOk) return EXIT_FAILURE;
    uart_base = uart.base_addr;

    cim_init();
    for (int t = 0; t < MFCC_NTILES; t++) cim_load_tile(t, &MFCC_TILES[t][0][0]);
    load_net_tiles();
    if (load_params()) return EXIT_FAILURE;
    CIM_REG(CIM_IRQ_EN) = 0;

    begin('H');
    put8(1u);
    put8(KWLOG_AUDIO);
    put8(KW_TEST);
    put32(REFERENCE_CLOCK_Hz);
    finish();
    tx_poll();

    CIM_REG(CIM_AUD_CTRL) = 0x3u | ((uint32_t)DS_T_CONV1 << 8);   // microphone on, network on, every hop
    CIM_REG(CIM_AUD_STATUS) = 0x6u;                  // clear pending, overrun

    uint32_t h_done = 0;
#if (KW_TEST) > 0
    uint32_t trial = 0, next = LEAD_DEC;
    uint8_t ord[13];
#endif
    int last = -1, run = 0, said = -1;
    uint32_t n_dec = 0, n_level = 0;
    for (;;) {
        tx_poll();
        uint32_t hops = CIM_REG(CIM_AUD_HOPS);
#if KWLOG_AUDIO
        if (h_done < hops) {                         // hop h_done has ended: its samples are in the ring
            if (hops > h_done + 2u) {                // the ring holds 1024 samples: too late for these
                lost += hops - 1u - h_done;
                h_done = hops - 1u;
            }
            if (tx_room() >= 2u * HOP_SAMPLES + 16u) {
                begin('A');
                put32(h_done);
                for (uint32_t n = HOP_SAMPLES * h_done; n < HOP_SAMPLES * h_done + HOP_SAMPLES; n++) {
                    put16(CIM_REG(CIM_SAMPLES(n % CIM_RING)));
                    if ((n & 15u) == 0u) tx_poll();
                }
                finish();
                h_done++;
            } else if (hops >= h_done + 2u) {        // no room, and the next poll may be too late
                lost++;
                h_done++;
            }
        }
#else
        if (h_done < hops) {                         // the hop's level, no samples
            if (hops > h_done + 2u) {
                lost += hops - 1u - h_done;
                h_done = hops - 1u;
            }
            int32_t mn = 32767, mx = -32768;
            int64_t sum = 0, sq = 0;
            for (uint32_t n = HOP_SAMPLES * h_done; n < HOP_SAMPLES * h_done + HOP_SAMPLES; n++) {
                int32_t v = (int32_t)CIM_REG(CIM_SAMPLES(n % CIM_RING));
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
                sq += (int64_t)v * v;
                if ((n & 15u) == 0u) tx_poll();
            }
            int32_t mean = (int32_t)(sum / (int32_t)HOP_SAMPLES);
            uint32_t ms = (uint32_t)(sq / (int32_t)HOP_SAMPLES - (int64_t)mean * mean);
            uint32_t rms = 0;
            while ((rms + 1) * (rms + 1) <= ms) rms++;
            if (tx_room() >= 16u) {
                begin('M');
                put32(h_done);
                put16((uint32_t)mx);
                put16((uint32_t)mn);
                put16(rms);
                finish();
            } else {
                lost++;
            }
            h_done++;
        }
#endif
        uint32_t st = CIM_REG(CIM_AUD_STATUS);
        if (!(st & 0x2u)) continue;
        CIM_REG(CIM_AUD_STATUS) = 0x2u;              // decision read
        uint32_t hop = CIM_REG(CIM_AUD_HOPS) - 1u;   // the decision of the last hop that ended
        uint32_t cls = (st >> 4) & 0xFu;
        int8_t lg[12];
        for (int k = 0; k < 12; k++) lg[k] = (int8_t)CIM_REG(CIM_LOGIT(k));
        n_dec++;
        if (tx_room() >= 64u) {
            begin('D');
            put32(hop);
            put16(st >> 16);
            put8(cls | ((st >> 2) & 1u) << 4);
            for (int k = 0; k < 12; k++) put8((uint32_t)(uint8_t)lg[k]);
            finish();
        } else {
            lost++;
        }
        // hopcim_live's rule
        run = ((int)cls == last) ? run + 1 : 1;
        last = (int)cls;
        int say = (cls < 10u) && (run == KW_HOLD) && ((int)cls != said);
        if (cls >= 10u) said = -1;
        if (say) {
            said = (int)cls;
            if (tx_room() >= 16u) {
                begin('R');
                put32(hop);
                put8(cls);
                finish();
            } else {
                lost++;
            }
        }
        if (++n_level == LEVEL_DEC) {
            n_level = 0;
            level_frame(hop);
        }
#if (KW_TEST) > 0
        if (n_dec >= next) {
            if (trial == N_TRIAL) break;
            uint32_t pos = trial % 13u, r = trial / 13u;
            if (pos == 0u) {                         // a new round: shuffle (Fisher-Yates), as hopcim_kwtest
                for (int i = 0; i < 13; i++) ord[i] = (uint8_t)i;
                for (int i = 12; i > 0; i--) {
                    int j = (int)rnd((uint32_t)i + 1u);
                    uint8_t x = ord[i];
                    ord[i] = ord[j];
                    ord[j] = x;
                }
            }
            uint32_t it = ord[pos];
            trial++;
            uint32_t kind = it < 10u ? 0u : (it < 12u ? 1u : 2u);
            uint32_t word = it < 10u ? it : (it < 12u ? (2u * r + it - 10u) % 10u : 0u);
            if (tx_room() >= 16u) {
                begin('P');
                put32(hop);
                put16(trial);
                put8(kind);
                put8(word);
                finish();
            } else {
                lost++;
            }
            next += TRIAL_DEC;
        }
#endif
    }
#if (KW_TEST) > 0
    begin('E');
    put32(CIM_REG(CIM_AUD_HOPS));
    put32(lost);
    put8((CIM_REG(CIM_AUD_STATUS) >> 2) & 1u);
    finish();
    while (tx_r != tx_w) tx_poll();
    CIM_REG(CIM_AUD_CTRL) = 0;
#endif
    return EXIT_SUCCESS;
}
