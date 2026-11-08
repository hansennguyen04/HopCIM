// HopCIM R7: record the on-board microphone through the live audio path, with the decisions the
// hardware makes on it, so that both can be checked offline (hopcim/model/check_micrec.py).
// The audio path runs as in hopcim_live (microphone, decimator, front-end, window path, DS-CNN,
// all in the endpoint). The CPU polls: from hop REC_H0 on it copies every hop's 320 new samples
// out of the audio ring (REC_HOPS hops, 1.5 s) and, for every decision, its count, class and 12
// logits. Afterwards it prints:
//   REC <first hop> <hops> <system clock>
//   S <32 samples>      the recording, 3 characters per sample (16 bits, base 64, A-Za-z0-9+/)
//   D <hop> <decision count> <class> <12 logits>
//   done
// A decision at hop h >= REC_H0 + 49 sees a window that lies inside the recording, so the golden
// model must reproduce its logits exactly from the recorded samples.
// Build: make app PROJECT=hopcim_micrec TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
// With -DNET_K=<k> (1..16) the network runs every k hops (AUD_CTRL[19:16] = k - 1); check the log with
// check_micrec.py --k <k>. The default build is unchanged.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "core_v_mini_mcu.h"
#include "../../../hw/ip/hopcim/sw/cim.h"
#include "../../../hw/ip/hopcim/sw/mfcc_tables.h"
#include "../../../hw/ip/hopcim/sw/dscnn_tables.h"

#ifndef NET_K
#define NET_K 1                                      // a decision every NET_K hops
#endif
#define REC_H0    100                                // first recorded hop (2 s after the start)
#define REC_HOPS  75                                 // 1.5 s
#define CIM_AUD_CTRL   (CIM_BASE + 0x0034u)
#define CIM_AUD_STATUS (CIM_BASE + 0x0038u)
#define CIM_AUD_HOPS   (CIM_BASE + 0x003Cu)

static int16_t rec[REC_HOPS * 320];
static struct { uint32_t hop, cnt, cls; int8_t lg[12]; } dec[REC_HOPS + 2];

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

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int main(void) {
    cim_init();
    for (int t = 0; t < MFCC_NTILES; t++) cim_load_tile(t, &MFCC_TILES[t][0][0]);
    load_net_tiles();
    if (load_params()) {
        printf("FAIL: a layer needs a left shift\n");
        return EXIT_FAILURE;
    }
    CIM_REG(CIM_IRQ_EN) = 0;
    CIM_REG(CIM_AUD_CTRL) = 0x3u | ((uint32_t)DS_T_CONV1 << 8) | ((uint32_t)((NET_K) - 1) << 16);   // microphone on, network on
    uint32_t h_done = REC_H0, nd = 0;
    while (h_done < REC_H0 + REC_HOPS) {
        uint32_t hops = CIM_REG(CIM_AUD_HOPS);
        if (hops > h_done + 1) {                     // a hop was not copied in time
            printf("FAIL: hop %u missed\n", (unsigned)h_done);
            return EXIT_FAILURE;
        }
        if (hops == h_done + 1) {                    // hop h_done has ended: its samples are in the ring
            int16_t *p = &rec[(h_done - REC_H0) * 320];
            for (uint32_t n = 320 * h_done; n < 320 * h_done + 320; n++)
                *p++ = (int16_t)CIM_REG(CIM_SAMPLES(n % CIM_RING));
            h_done++;
        }
        uint32_t st = CIM_REG(CIM_AUD_STATUS);
        if ((st & 0x2u) && hops > REC_H0 && nd < REC_HOPS + 2) {
            CIM_REG(CIM_AUD_STATUS) = 0x2u;
            dec[nd].hop = hops - 1;                  // the decision of the last hop that ended
            dec[nd].cnt = st >> 16;
            dec[nd].cls = (st >> 4) & 0xFu;
            for (int k = 0; k < 12; k++) dec[nd].lg[k] = (int8_t)CIM_REG(CIM_LOGIT(k));
            nd++;
        } else if (st & 0x2u) {
            CIM_REG(CIM_AUD_STATUS) = 0x2u;
        }
    }
    uint32_t st = CIM_REG(CIM_AUD_STATUS);
    CIM_REG(CIM_AUD_CTRL) = 0;
    printf("REC %u %u %u overrun %u\n", (unsigned)REC_H0, (unsigned)REC_HOPS, (unsigned)REFERENCE_CLOCK_Hz,
           (unsigned)((st >> 2) & 1u));
    char line[3 * 32 + 1];
    for (int i = 0; i < REC_HOPS * 320; i += 32) {
        for (int j = 0; j < 32; j++) {
            uint16_t u = (uint16_t)rec[i + j];
            line[3 * j] = B64[u >> 12];
            line[3 * j + 1] = B64[(u >> 6) & 63u];
            line[3 * j + 2] = B64[u & 63u];
        }
        line[96] = 0;
        printf("S %s\n", line);
    }
    for (uint32_t k = 0; k < nd; k++) {
        printf("D %u %u %u", (unsigned)dec[k].hop, (unsigned)dec[k].cnt, (unsigned)dec[k].cls);
        for (int j = 0; j < 12; j++) printf(" %d", (int)dec[k].lg[j]);
        printf("\n");
    }
    printf("done\n");
    return EXIT_SUCCESS;
}
