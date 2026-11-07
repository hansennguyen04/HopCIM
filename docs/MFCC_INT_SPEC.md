# R4a — integer CIM-MFCC front-end (golden model)

`model/mfcc_int.py` is the bit-exact specification of the front-end that HopCIM runs: every
value is an integer, from the int16 waveform to the int8 input of the **unchanged** MLPerf Tiny
DS-CNN. RTL and firmware are verified against it. Tables: `model/mfcc_tables.npz`
(`model/make_tables.py`, from the frozen gate-2 datapath).

## 1. Pipeline

Per frame (480 samples, hop 320, 49 frames per 1 s window):

| # | Step | Where | Arithmetic |
|---|---|---|---|
| 1 | Hann window | epilogue | int16 × Q15 periodic Hann |
| 2 | block exponent | epilogue | `sh = msb(peak) − 14`; shift (round half up) to [−2¹⁵, 2¹⁵] |
| 3 | FFT step 1: 32 × 16-point DFT | **CIM** | int8 DFT tile, `>> 11` |
| 4 | twiddles | epilogue | Q14 complex multiply, `>> 14` |
| 5 | FFT step 3: 16 × 32-point DFT | **CIM** | int8 DFT tiles, `>> 9` |
| 6 | magnitude | epilogue | exact `isqrt(re² + im²)` |
| 7 | mel (40 filters) | **CIM** | int8 mel tile |
| 8 | log2 | epilogue | leading one + 64-entry log2(1+f) table, Q8; frame exponent and constant scales added in Q16 → `lraw` |
| 9 | normalisation | firmware | `lmq = max(round((lraw − log2fix16(max x)) · ln2), floor)`, Q8 natural log |
| 10 | DCT (10 of 40) | **CIM** | int16 coefficients as two int8 weights (`c = 256·hi + lo`) |
| 11 | requantisation | epilogue | `trunc(acc · M / 2⁴⁴ + 83)`, saturated to int8 |

Constants: `K16 = 1374529` (Q16 log2 of 127² · mel scale), `ln2 = 45426 / 2¹⁶`,
`floor = −3537` (ln 10⁻⁶ in Q8), `M = 808325`, DCT scale 127·256 / max|c|.
`log2fix16` is a 16-step repeated-squaring integer log2 (exact floor at the checked points).
Step 9 is the only per-clip (per-window) step: it needs `max x` over the 1 s window, so a
streaming implementation keeps `lraw` (40 × Q16) for the 49 frames in the window.

## 2. CIM mapping (per frame)

Exact reductions used, all bit-exact with the model:

- **Step 3 computes only bins 0–127.** The mel matrix is zero outside bins 1–127, so higher
  bins never reach the features: 16 × 8 bins → one 16-column pass per k1.
- **Step 1 uses conjugate symmetry.** The int8 DFT tile is exactly conjugate-symmetric (checked), and the
  imaginary parts at k1 = 0 and 8 are zero. Each 16-point DFT of real data therefore has 16
  real outputs (re k1 = 0..8, im k1 = 1..7): one pass per n2. The twiddle unit restores k1 = 9..15.
- **Mel uses only its 4 non-zero (64-row × 16-column) tiles of 15.**

| Step | Tiles | Passes | Input width b (worst case) | Engine cycles (D = max(8, b)) |
|---|---:|---:|---:|---:|
| FFT step 1 | 1 | 32 | 17 | 544 |
| FFT step 3 | 1 | 16 | 17 | 272 |
| mel | 4 | 4 | 22 | 88 |
| DCT | 2 | 2 | 14 | 28 |
| **total** | **8** | **54** | | **932** |

Input widths are worst-case bounds, not observed ranges. Observed on 489 test clips:
|step-1 x| ≤ 32,767, |step-3 x| ≤ 15,102, |X| ≤ 118,897, lmq ∈ [−3537, 1369].

Epilogue work per frame: 480 window products, 512 complex twiddle products, 127 square roots,
40 log2 lookups, 10 requantisations.

## 3. Gate (same test set and criterion as gate 2)

`model/gate2_int.py` on the 4,890 GSC test clips, unchanged int8 DS-CNN:

| Features | int8 accuracy | float32 model |
|---|---:|---:|
| float reference MFCC, reference int8 quantisation | 91.66 % | 91.86 % |
| **integer CIM-MFCC (this model)** | **91.78 % (Δ +0.12)** | 91.76 % (dequantised) |

McNemar (int8): 18 vs 24 discordant clips, p = 0.44. The int8 features equal the reference in 96.7 %
of values and are within ±1 step in 99.94 %. The largest differences (255) come from 28 values (0.0012 %,
5 clips, all c0): the reference value is ≥ 128 and the reference's numpy int8 cast wraps it to
negative, while this model saturates. **Gate: pass.** On 3 spot-checked clips the int8
features are identical to those of the gate-2 datapath.

SHA-256: `mfcc_tables.npz` 298861f6…c35f, `mfcc_int.py` df76d26e…aad8.

## 4. Mapping and firmware verification (R4b)

**Mapping model** (`model/cim_map.py`): the front-end computed only through engine passes on the
8 tiles plus the epilogue rules is bit-exact against `mfcc_int.py` on **all 4,890 test clips
(239,610 frames): 0 lraw and 0 feature mismatches**. The test is sensitive to the one subtle rule:
negating the mirrored step-1 outputs *after* the round-half-up shift instead of before gives 42
lraw and 2 feature mismatches on 2,940 frames.

**Firmware on the SoC** (`sw/applications/hopcim_mfcc`, Verilator X-HEEP + CIM endpoint, CPU
orchestrates every pass and runs the epilogue): all 49 frames of the test clip (GSC test
`yes/022cd682_nohash_0.wav`) match the golden model bit-exactly, lraw and int8 features.
7.56 M cycles in total. Tables and test vector: `model/gen_c.py` → `sw/mfcc_tables.h`.

| Stage (CPU-orchestrated) | Cycles / frame | Share |
|---|---:|---:|
| window + block exponent (CPU) | 15,955 | 10.9 % |
| FFT step 1, 32 passes (MMIO) | 18,681 | 12.8 % |
| twiddles (CPU) | 21,726 | 14.9 % |
| FFT step 3, 16 passes (MMIO) | 12,202 | 8.3 % |
| magnitude, 128 × isqrt (CPU) | 70,881 | 48.5 % |
| mel, 4 passes (MMIO) | 1,801 | 1.2 % |
| log2 (CPU) | 2,926 | 2.0 % |
| normalisation + DCT + requantisation | 2,104 | 1.4 % |
| **total** | **146,278** | |

For comparison, the CPU-only front-end measured on the board (census: radix-2 FFT512 145,142
+ power/mel/log/DCT 15,632) costs 160,774 cycles/frame. Offloading only the matrix steps gains
1.1×: the CPU epilogue (window, twiddles, magnitude) is 74 % of the remaining time and MMIO most
of the rest, while the engine computes for 932 cycles. This motivates R5: hardware epilogue
units and a hop sequencer that feeds the engine without the CPU.
