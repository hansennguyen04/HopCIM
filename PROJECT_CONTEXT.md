# HopCIM — project context for AI agents

Read this first. It is a self-contained briefing on the HopCIM project: what it is, why it exists, what has been
built and measured, how to build and check it, the rules the owner has set, and what is still open. It was written
on 2026-11-14 from the repository's documents and the project history, and updated on 2026-11-26.

**Source of truth.** If a number here disagrees with `freeze/RESULTS.md`, `docs/HOPCIM_PROPOSAL.md` or
`docs/EVALUATION.md`, those files win (they are checked by `freeze/freeze.py check`). Update this file when they change.

---

## 0. Start here (the 2-minute version)

- **Who:** an NYCU master's student (electrical engineering) researching digital SRAM compute-in-memory (CIM) ×
  RISC-V. Thesis due **June 2027**; a **conference paper** is also a goal (venue not chosen yet). The advisor has
  asked for more architecture novelty for the paper. Refer to the student as "they/them".
- **What HopCIM is:** a digital SRAM compute-in-memory accelerator (the "endpoint") inside an X-HEEP RISC-V
  microcontroller (CV32E20 core). The endpoint runs the **standard MFCC front-end** and the
  **unchanged, pretrained MLPerf Tiny int8 DS-CNN** in hardware, **bit-exact** to TensorFlow Lite's reference kernels,
  while the CPU sleeps. It runs live on a Digilent **Nexys A7-100T** FPGA with the on-board PDM microphone.
- **Two designs.** The **default design is S+DWU** (since 2026-11-25): the baseline S with a digital depthwise unit
  that runs the four depthwise layers beside the CIM. **S** is kept unchanged as the comparison baseline
  (`docs/DESIGNS.md`). Numbers quoted before 2026-11-25 are S's and say so.
- **Headline numbers (measured on the board unless [E]):**
  - 264 CPU-active cycles per 20 ms decision with S+DWU (S: 263); the CPU is asleep otherwise.
  - Network: 67,856 cycles per inference with S+DWU (S: 67,909), against 47,974,801 on the CPU alone.
  - Real time down to a 4.096 MHz system clock (S on the board; S+DWU in RTL, board run pending).
  - Accuracy 91.80 % on the 4,890 Google Speech Commands test clips, the reference model's own accuracy reproduced
    exactly, by both designs.
  - [E] power at mode U (network every 16 hops, sum of 2 decisions): 27.96 µW with S+DWU against 44.11 µW with S.
  - Under MLCommons' own MLPerf Tiny runner (S): 90.1 % top-1 and 197.9 inferences/s, both **unverified**.
- **HopCIM is a keyword-spotting system only** (the student's decision, 2026-11-26). A second MLPerf Tiny network
  (ResNet8) runs bit-exact on the same CIM engine through a second sequencer, as **evidence that the engine and the
  exactness generalise**, nothing more (§3.8).
- **State on 2026-11-26:** building is done. The phase now is evidence, claims, comparison with published work and
  writing. The MLPerf runner work is still **uncommitted** (§12).
- **Read next, in order:**
  1. `README.md`, then `docs/README.md`: the repository map and the index of documents.
  2. `docs/HOPCIM_PROPOSAL.md` §1, §13, §15: the living proposal, claims and status.
  3. `docs/DESIGNS.md`: the default design against the baseline.
  4. `docs/EVALUATION.md`: all result tables.
  5. `freeze/RESULTS.md`: every frozen claim with its evidence file (as of the 2026-11-13 freeze).
  6. `docs/NOVELTY_SYSTEM.md`, `docs/LITERATURE_REVIEW.md`: the novelty boundary and related work.
- **Before you change anything,** read §10 (rules). Several are strict and non-obvious.

---

## 1. Background and motivation

### 1.1 The task

Always-on KWS listens for a small vocabulary on a battery-powered device. The standard setting is **MLPerf Tiny KWS**:

| Item | Setting |
|---|---|
| Dataset | Google Speech Commands v2, 12 classes (10 words + silence + unknown); 4,890 test clips |
| Window | 1 s of 16 kHz audio |
| Features | 49 frames × 10 MFCCs; 30 ms frames (480 samples), 20 ms hop (320 samples), 40 mel bands 20–4000 Hz, 512-point FFT; each 1-s window normalised by its peak |
| Model | DS-CNN: 10×4 conv (stride 2) → 4 × (3×3 depthwise + 1×1 pointwise, 64 ch) → avg-pool → FC 12. int8, 22,016 weights (21.5 KiB), 2.66 M MACs per inference. File `kws_ref_model.tflite` (sha256 `aeea4368…`) |

### 1.2 The problem HopCIM attacks

- A CIM makes the network's MACs cheap. The bottleneck then moves to everything else: the MFCC front-end, data
  movement and requantisation.
- A census model of streaming hops gave the front-end 78–92 % of the remaining per-hop work once a CIM runs the
  network.
- Measured on this MCU, putting only the front-end's MACs on the CIM gives just **1.1×** (146,364 vs 160,774 cycles
  per frame). The CPU epilogue (window, twiddles, square root) and CPU-driven data movement dominate. This Amdahl's-law
  result is the core motivation: **the whole pipeline has to move into hardware.**
- Existing CIM KWS chips avoid the MFCC in one of two ways. Either they leave it outside the CIM, or they replace it
  with a learned, model-specific front-end and retrain the network. Analog and binary CIMs cannot compute a 16-bit FFT
  exactly; a **digital bit-serial CIM** can, at b cycles for b-bit inputs.

### 1.3 The idea

- Compute the **standard MFCC exactly in integers on the digital CIM**:
  - a four-step FFT (512 = 16 × 32) as small int8 DFT matrices with a per-frame block exponent;
  - mel and DCT as int8 weight tiles;
  - a small hardware epilogue for the non-matrix steps.
- Run the **unchanged pretrained network on the same CIM** with a hardware layer sequencer.
- Move the per-window normalisation and the microphone path into hardware too. The CPU then only wakes to read
  decisions.

### 1.4 How the direction was chosen (short)

HopCIM became the direction on 2026-10-26, after earlier directions were closed. Those directions tried to optimise
bottlenecks that turned out not to bind. **Lesson carried over: measure which resource binds before optimising it.**
The earlier material was deleted on 2026-11-26 at the student's request. What HopCIM still uses was kept:
`recim-rv/accelforge_pilot/` (the energy tool) and `research_discovery/20261026_application_driven/` (the report
that chose HopCIM). The `planenest/` folder keeps its old name because it holds HopCIM's X-HEEP trees and board files.

---

## 2. Architecture

```
            Nexys A7-100T FPGA (xc7a100t), system clock 15.36 MHz (real time down to 4.096 MHz)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │  X-HEEP RISC-V MCU (upstream commit b3eb3c8 + patch series in xheep/)                         │
 │   CV32E20 core, 4 × 32 KiB SRAM (8 × 32 KiB in the 256 KiB variant), OBI bus, PLIC, UART,     │
 │   debug over the on-board USB (BSCANE2)                                                       │
 │        │ CPU data port, direct tap (no crossbar), window 0xC004_0000, 256 KiB; IRQ ext. line 3 │
 │  ┌─────▼──────────────────────────── HopCIM endpoint (rtl/cim_endpoint.sv) ────────────────┐  │
 │  │ PDM mic ─► pdm_dec (CIC5/64 + 15-tap FIR + DC blocker, exactly 16 kHz)                   │  │
 │  │          ─► cim_audio: hop controller, ring of 1024 samples, every 320 samples = 1 hop  │  │
 │  │ cim_fe: front-end epilogue (Hann window, block exponent, twiddles, isqrt, log2)         │  │
 │  │          drives the CIM for FFT step 1 (32 passes), step 3 (16), mel (4), DCT (2)       │  │
 │  │          (or mfcc_ded.sv = B3, a dedicated radix-2 MFCC baseline, by parameter)          │  │
 │  │ cim_engine: 128 resident tiles of 64 rows × 16 int8 columns (1 KiB each), bit-serial    │  │
 │  │          inputs of 1–24 bits, 48-bit accumulators; pass issue interval D = max(8, b)    │  │
 │  │ cim_seq: layer sequencer — gathers inputs, issues all 7,501 network passes,             │  │
 │  │          requantises TFLite-exactly, keeps activations; also the window path (R5b):     │  │
 │  │          per-window peak normalisation, lraw ring 49×40, DCT + requant of features      │  │
│  │ cim_seq + DWU (default S+DWU): a digital unit runs the 4 depthwise layers,               │  │
│  │          2,501 passes on the CIM instead of 7,501; 29 tiles instead of 69                │  │
│  │ cim_gseq: second network (ResNet8), descriptor-driven, behind SEQ_NET2 = 0/1             │  │
 │  └─────────────────────────────────────────────────────────────────────────────────────────┘  │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
 Per 20 ms hop: mic → front-end → window path → network (every k hops) → IRQ → CPU reads class (263 cycles)
```

### 2.1 Blocks (all RTL in `rtl/`)

| File | Role | Built in |
|---|---|---|
| `cim_engine.sv` | CIM emulation: weights in 32 BRAM banks, 128 DSPs compute 8 rows × 16 columns per cycle; cycle behaviour of a 64-row bit-serial digital macro. A pass accepted at t0 delivers at t0 + D + 5 and the next may start at t0 + D | R1 |
| `cim_endpoint.sv` | 32-bit OBI slave, register map (§2.3), arbitration between CPU, front-end and sequencer | R2 |
| (X-HEEP integration) | FuseSoC flag `use_cim`; direct CPU-port tap so simulation equals the board cycle for cycle | R3 |
| `cim_fe.sv` | hardware front-end, frame path: the CPU's front-end work per frame falls from 146,364 to 4,743 CPU-active cycles | R5a |
| `mfcc_ded.sv` | baseline B3: dedicated radix-2 18-bit MFCC; 91.60 %; costs the same logic as `cim_fe` | B3 |
| `cim_seq.sv` | layer sequencer + requantisation (R6b); window path (R5b); baseline S (`SEQ_S`); one StreamFuse-B block (`SEQ_B1`/`FUSE1`); the digital depthwise unit (`SEQ_DWU = 4`, the default design) | R6b, R5b, S, B1, DWU |
| `cim_gseq.sv` | the second network's descriptor-driven sequencer (ResNet8: 3×3 convolutions, residual ADD, pool, FC), endpoint parameter `SEQ_NET2` (default 0) | net2 |
| `cim_log2fix16.sv` | 16-step integer log2 for the window normalisation | R5b |
| `pdm_dec.sv`, `cim_audio.sv` | microphone decimator and hop controller; `AUD_CTRL[19:16]` = k − 1 runs the network every k hops | R7, k-hop |
| `*_tables.svh` | generated coefficient tables (from `model/`) | — |

FPGA cost, Vivado out of context:
- R7 endpoint: 7,635 LUT, 5,819 FF, 46 BRAM tiles, 145 DSP.
- Endpoint with S: 8,116 LUT, 5,977 FF, 147 DSP, +45.8 ns slack.
- Whole SoC with S: 42,758 LUT (67 %), +15.3 ns setup slack at 15.36 MHz.
- Whole SoC with B1: 43,418 LUT.
- Endpoint with S+DWU: 8,743 LUT, 6,795 FF (+1,254 / +815 against S; 12,414 gate equivalents as priced); whole SoC
  44,022 LUT.
- Whole SoC with S+DWU and the second network, 256 KiB RAM: 50,946 LUT (80.4 %), 131.5 of 135 BRAM tiles.

### 2.2 The integer MFCC (golden model `model/mfcc_int.py`, spec `docs/MFCC_INT_SPEC.md`)

| # | Step | Where | Arithmetic |
|---|---|---|---|
| 1 | Hann window | epilogue | int16 × Q15 |
| 2 | Block exponent | epilogue | shift so the peak lies in [2¹⁴, 2¹⁵), round half up |
| 3 | FFT step 1: 32 × 16-point DFT | **CIM**, 1 tile, 32 passes, b = 17 | int8 DFT tile, `>> 11` |
| 4 | Twiddles | epilogue | Q14 complex multiply |
| 5 | FFT step 3: 16 × 32-point DFT, bins 0–127 only | **CIM**, 1 tile, 16 passes | `>> 9` |
| 6 | Magnitude | epilogue | exact `isqrt(re² + im²)` |
| 7 | Mel, 40 bands | **CIM**, 4 tiles, 4 passes, b = 22 | int8 |
| 8 | Log2 | epilogue | leading one + 64-entry table (Q8 → Q16) |
| 9 | Window normalisation | window path | subtract log2(window max), × ln 2, floor at ln 10⁻⁶ |
| 10 | DCT, 10 of 40 | **CIM**, 2 tiles, 2 passes | int16 coefficients as two int8 tiles |
| 11 | Requantisation | epilogue | `trunc(acc·M/2⁴⁴ + 83)`, saturated to int8 (the model's input quantisation) |

Totals: 8 tiles, 54 passes per frame (FFT step 1: 32, step 3: 16, mel: 4, DCT: 2), 932 engine cycles per frame.

Exactness rules:
- Mirrored step-1 outputs are negated **before** round-half-up; negating after gives 42 tie errors in 2,940 frames.
- Input widths are worst-case bounds, so the hardware is exact for any input.

### 2.3 Network mapping (sequencer, `docs/SEQUENCER.md`, `docs/NETWORK.md`)

| Layer | Tiles | Passes per inference |
|---|---:|---:|
| L0: conv 10×4, stride 2 | 4 | 500 |
| L1, L3, L5, L7: depthwise 3×3, block-diagonal, 7 channels per tile, weights tap-major (rows 7k + c′) | 4 × 10 | 4 × 1,250 |
| L2, L4, L6, L8: pointwise 1×1 | 4 × 4 | 4 × 500 |
| L9: avg-pool + FC 64 → 12 | 1 | 1 |
| **Total** | **61** (+8 front-end = 69 of 128) | **7,501** |

- The sequencer stops at the FC logits; the argmax equals the model's.
- For MLPerf, the CPU also computes TFLite's int8 softmax (`softmax_s8.c`), ported line by line.

### 2.4 Endpoint register map (offsets in the 256 KiB window; full table in `docs/INTEGRATION.md` §1)

| Offset | Register | Content |
|---|---|---|
| `0x00` | CTRL | CPU pass: go, accumulate, width b, tile |
| `0x04` / `0x08` / `0x0C` | STATUS / IRQ_EN / DONE_CNT | |
| `0x10` / `0x14` | FE_CTRL / FE_STATUS | front-end go, ring index |
| `0x18` | CYCLES | free-running wall-clock counter; used for sleep accounting and MLPerf timestamps |
| `0x20` / `0x24` / `0x28` | SEQ_CTRL / SEQ_STATUS / SEQ_ZP | sequencer go/stop/FEAT row/first tile; class, layer, error; zero points |
| `0x2C` / `0x30` | SEQ_LMX / SEQ_WMAX | window path |
| `0x34` / `0x38` / `0x3C` | AUD_CTRL / AUD_STATUS / AUD_HOPS | `[0]` enable, `[1]` network, `[14:8]` first tile, `[19:16]` k − 1; decision pending, overrun, class |
| `0x1000` | X[64] | pass inputs |
| `0x2000` | SAMPLES[1024] | audio ring |
| `0x3000` | LRAW[40] | log-mel of the last frame |
| `0x4000` | FEAT | int8 features, ring of 49 rows |
| `0x5000` | LOGITS[12] | |
| `0x8000` | PARAMS | per-channel bias, multiplier, shift |
| `0x1_0000` | ACT | debug readback |
| `0x2_0000` | WEIGHTS | 128 KiB |

Added later (`docs/INTEGRATION.md`): the depthwise unit's weight store at `0x6000`; the second network's records at
`0x7000`, `NET_CTRL`/`NET_STATUS`/`NET_DBG` at `0x7200`–`0x7208`, logits at `0x7300`, parameters at `0xC000`, image at
`0xF000`.

Driver: `sw/cim.h`.

### 2.5 The design steps (names used everywhere)

| Name | What it added | Effect (measured unless [E]) |
|---|---|---|
| R1–R3 | engine, endpoint, X-HEEP integration | CPU-driven pass costs about 25× its compute (MMIO about 9 cycles per word) |
| R4a/R4b | integer MFCC golden model; tile mapping; CPU-orchestrated front-end on the CIM | 91.78 % re-gated; **1.1×** only |
| R5a | `cim_fe` hardware front-end | 21.8–31× fewer CPU-active cycles per frame (6,710, later 4,743 with sleep accounting, vs 146,364) |
| B3 | dedicated MFCC baseline | same logic cost (1,447 vs 1,530 LUT): **H4 not supported** — reusing the CIM saves no hardware |
| R6a | network on the CIM, CPU-orchestrated | 13.02 M cycles per inference (depthwise data movement about 75 %) |
| R6b | hardware layer sequencer | **85,420** cycles per inference (152×), CPU asleep, 5.69 ms at 15 MHz |
| R5b | window path in hardware (recompute features only when the window peak changes, 11.6 % of hops) | worst CPU-active per hop about 109k → 4,447 |
| R7 | live microphone path | **263** CPU-active cycles per decision; f_min 5.12 MHz |
| k-hop | network every k hops (18 LUT, 8 FF) | passes per hop 7,501 / 3,750 / 1,875 at 20 / 40 / 80 ms |
| **S+DWU** (default since 2026-11-25) | digital unit for the four depthwise layers beside the CIM; CIM 69 → 29 tiles; `SEQ_DWU = 4`, firmware `-DCIM_DWU=4` | [E] 27.96 µW at mode U against S's 44.11; 67,789 cycles; `docs/DWU_RESULTS.md`, `docs/DESIGNS.md` |
| **S** (comparison baseline) | sign-gated requantisation (50.5 % of requantisations give exactly −128); second requantisation lane (even/odd channels); no L8 ACT write | network **67,909** (−20.5 %), worst hop 80,072, **f_min 4.096 MHz**, [E] 1,121 nJ per inference |
| B1 | one block of StreamFuse-B: depthwise L1 → line buffer → fusion vector → pointwise L2 | 67,885 on board; RTL equals the schedule model exactly; [E] −2.4 % network energy only; +635 LUT, +202 FF |
| net2 | MLPerf Tiny's int8 ResNet8 on the same engine through `cim_gseq` (evidence of generality only) | bit-exact on 10,000 CIFAR-10 images (RTL) and 20 on the board; 155,882 cycles go → IRQ; `docs/NET2_RESULTS.md` |

---

## 3. Key results (labels: [M] measured, [E] estimate/model, [P] published by others, [H] hypothesis)

### 3.1 End to end per 20 ms hop (`docs/EVALUATION.md` §2; configurations A–E)

| Config | Front-end / window / network / audio | CPU-active per hop | Lowest real-time clock |
|---|---|---:|---:|
| A | all on the CPU (plain C) | ≈ 48.1 M | ≈ 2.4 GHz |
| B | CPU drives CIM passes | ≈ 13.17 M | ≈ 660 MHz |
| C | HW front-end + sequencer, CPU window path | ≈ 7.3k typ., ≈ 109k worst | 5.0 / 10.1 MHz |
| D | + window path in HW | 4,447 | 5.1 MHz |
| **E** | + microphone path in HW | **263** | **5.12 MHz** (R6b), **4.096 MHz** (S) |

### 3.2 Network alone (cycles per inference, board)

| Implementation | Cycles | Relative |
|---|---:|---:|
| CPU, plain C `-O2` (`hopcim_cpu`) | 47,974,801 | 1× |
| CPU, TensorFlow Lite Micro reference kernels (`tflm_mlperf`, 256 KiB bitstream) | 75.24–75.26 M | 0.64× |
| CPU drives CIM passes (`hopcim_net1`) | 13,020,998 | 3.7× |
| Sequencer R6b (`hopcim_seq`, go → IRQ) | 85,420 | 562× |
| **Sequencer S** | **67,909** | 706× |
| B1 | 67,885 | — |
| **S+DWU (default)** | **67,856** | 707× |

Two details:
- Cycle counts depend on the GCC build of the CPU code (for example 12.87 M vs 12.01 M for the same clip). Compare
  simulation and board only with identical C code.
- With S, the bottleneck is the depthwise layers (engine-bound, 45,092 of 67,841 cycles) and, in the energy model,
  leakage.

### 3.3 Accuracy (exactness is the point; the accuracy is the reference model's own)

| Item | Result |
|---|---|
| Integer CIM-MFCC + unchanged int8 DS-CNN (TFLite default kernels) | 91.78 % vs 91.66 % with float MFCC (McNemar p = 0.44) |
| Integer network vs TFLite **reference** kernels | equal in every layer on all 4,890 clips: **91.80 %** |
| B3 dedicated front-end | 91.60 % |
| Tile mapping vs golden model | 239,610 frames, 0 mismatches |
| Real microphone on the board | logits = golden model (25/25, 13/13, 7/7 at k = 1, 2, 4); logger 2,850/2,850 |
| One speaker, scripted test | 48 / 47 / 42 of 50 keywords at 20 / 40 / 80 ms → **40 ms is the operating point** |
| False alarms in a room (8.38 h, no audio kept) | 3.6 / 4.1 / 2.6 per hour at 20 / 40 / 80 ms, all in the first 1.38 h while a person was in the room, none in the 7 h night; no confirmation rule chosen yet |

### 3.4 Energy — [E] only, never measured, never compared with chips (`docs/ENERGY_ESTIMATE.md`)

The model:
- 28 nm, exact activity counters from the RTL (`tb/energy_counters.sv`);
- CACTI/Aladdin via HWComponents/AccelForge, and ZigZag-IMC's digital equations;
- ZigZag-IMC itself was run: ratio 1.000 on the pointwise layers, 1.029 overall.

| | R6b | S | B1 | Four-block B (model only) |
|---|---:|---:|---:|---:|
| Network energy per inference | 1,389.3 nJ | 1,120.8 nJ | 1,094.1 nJ | 0.837 × S |
| Average power at 20 / 40 / 80 ms | 110 / 75 / 58 µW | 97.0 / 68.8 / 54.7 µW | — / 68.7 / — | about −7 % at 40 ms |

The default design S+DWU [E]: 1,013.2 nJ per inference (−9.6 % against S); average power 27.96 µW at mode U (S
44.11, −36.6 %) and 50.28 µW at 40 ms (S 68.78). About 98 % of the saving at U is leakage, from 29 tiles in 2 macros
instead of 69 in 5 (`docs/DWU_RESULTS.md`).

Where the energy goes:
- The CIM array is only about 10 % of an inference; requantisation and activation memory cost more.
- Leakage is 33–62 % depending on the rate.

What is not included or not done:
- The microphone and the X-HEEP host are not included.
- The digital logic is coarse: an open ASIC flow was not allowed to be installed.

### 3.5 Hypotheses (frozen falsification lines; never reword them, only update status)

| ID | Hypothesis | Outcome |
|---|---|---|
| H1 | Front-end ≥ 70 % of per-hop CPU cycles once the network is on the CIM | Holds only once the network's data movement is also off the CPU. With the CPU orchestrating the network: 1.1 %, falsified there. With the sequencer: 91–98 % |
| H2 | Integer CIM-MFCC within 0.3 pp of float MFCC | **Passed** (+0.12 pp) |
| H3 | CPU-active ≥ 20× lower **and** f_min < 1 MHz | **Not supported as written.** The CPU half: about 50,000×. The f_min half falsified: 5.12 MHz, then 4.096 MHz with S (failure line 2 MHz) |
| H4 | CIM-reuse front-end costs no more than a dedicated block | **Not supported** (B3 costs the same) |
| H5 | End to end bit-exact and real time at 15 MHz | **Met** |
| H6 | A shared CIM-MFCC beats per-model learned front-ends | Optional; not done |

### 3.6 Verification

The RTL regression `sim/run_all.sh` has **54 benches** (46 for R6b/S/B1, 7 for the depthwise unit, 1 for the
second network), all passing in the main tree and from a clean clone; logs in `sim/logs/`. One bench is expected to
fail: R6b must overrun at 4.096 MHz.

Further checks:
- 57 assertions, all exercised, none failing (not extended to the depthwise unit or `cim_gseq`).
- 33 of 33 mutants caught (3 more shown equivalent); the depthwise unit's 3 of 3; the second network's 4 of 4, the
  last one only after the bench gained a check of the rewrite margin (`docs/NET2_RESULTS.md` N7).
- About 99 % line coverage.
- Gate-level netlist simulations are exact (endpoint, `cim_audio`).

Simulation equals the board cycle for cycle; there is one fixed 4-cycle offset in a one-time setup. Details are in
`docs/VERIFICATION.md`.

### 3.7 MLPerf Tiny runner and public comparisons (2026-11-14; `docs/MLPERF_TINY.md`, `EVALUATION.md` §14–15)

The run used MLCommons' runner unchanged (commit `4addd0fa`), the EEMBC `kws01` inputs (1,000 clips), the S bitstream
at 15.36 MHz and firmware `hopcim_mlperf`.

| | Closed (MLPerf's features in) | Open (raw audio in, HopCIM's MFCC) |
|---|---|---|
| Top-1 | 90.1 % (unverified) | 90.5 % (unverified) |
| Outputs vs reference | 1,000/1,000 = TFLite Micro | 1,000/1,000 = golden model |
| Throughput | 197.902 inf/s (5.053 ms) | 21.798 inf/s |

- **Not submitted.** Only MLCommons members can submit. Every score must say "unverified" with the footnote
  *Result not verified by MLCommons Association.* Comparing with verified results needs the second footnote too
  (text in `MLPERF_TINY.md` §8).
- **Published KWS results** (75 rows from MLCommons' result repositories, `mlperf/public_kws.py`), as cycles per
  inference = latency × stated clock:
  - HopCIM 77.6 k;
  - fpgaConvNet ZedBoard 32–46 k;
  - Syntiant 9120 134 k;
  - fpgaConvNet Zybo / Cora 269 k / 602 k;
  - K1 RISC-V 857 k;
  - best Cortex-M about 1.6 M;
  - reference TFLite Micro on Cortex-M4 21.8 M;
  - Cortex-M0+ 30.4 M.

  Cycles favour a slow clock: they are a normalisation, never "faster than". Energy is never compared.
- **TFLite Micro on the same CV32E20, same runner, board and clock:**
  - 75.26 M cycles per timed inference (4.90 s), 90.1 %, all outputs exact.
  - HopCIM uses 970× fewer cycles (network alone 1,108×).
  - TFLite Micro's reference kernels are 1.57× slower than the plain-C baseline, so the plain-C baseline was the
    stronger one.
- **Literature table:** `NOVELTY_SYSTEM.md` §3a, the six closest works plus CIMR-V.
- **NM-Caesar / NM-Carus** (near-memory units on X-HEEP): no public RTL, so no equal-platform accelerator comparison
  is possible.

### 3.8 Since 2026-11-14: whole-word decisions, the depthwise layers, the default design, a second network

Each study had pre-registered gates frozen in `docs/*_GATES.md` before its model or RTL existed.

| Study | Outcome |
|---|---|
| Whole-word decisions (`WHOLEWORD_GATES.md`, frozen `8e12400`; `WHOLEWORD_STUDY.md`) | Choosing whole-word windows (A, B) is not worth new hardware. **Mode U**: the existing k-hop field at k = 16 with a sum of 2 decisions, firmware only (`-DNET_K=16 -DKW_SUM=2`); a spoken run on the board is still needed |
| Depthwise swap-in (`DWSWAP_GATES.md`; `DWSWAP_STUDY.md`) | V (depthwise layers swapped onto two working tiles) 28.13 µW at U, 0.638 × S; but G5 failed: a digital depthwise unit priced 27.72 µW |
| Digital depthwise unit (`DWU_GATES.md`, frozen `95b8cf8`; `DWU_RESULTS.md`) | RTL equals the model in every layer and counter (67,789 cycles); 4,890 clips exact; board 60/60 exact, go → IRQ 67,856 (1 below the gate's figure: firmware-dependent offset, explained and accepted); 27.96 µW at U [E] |
| Default design (`DESIGNS.md`) | S+DWU is the default since 2026-11-25; S kept unchanged; code defaults stay off, so old commands reproduce |
| Second network (`NET2_GATES.md`, frozen `b610c36`; `NET2_RESULTS.md`) | MLPerf Tiny's int8 ResNet8 unchanged: N1 exact on all 10,000 CIFAR-10 test images in RTL; N2 155,814 cycles against 155,811 predicted; N3 old designs unchanged; N4 fits (97.4 % BRAM); N5 board 20/20 exact, 155,882 cycles; N6 TensorFlow Lite Micro on the same CPU 227.39 M cycles, 1,459 × more (unverified); N7 4/4 broken variants caught. No power, speed or accuracy claim |

---

## 4. Contributions and the novelty boundary (be precise; this is where agents most often overclaim)

### 4.1 The claim that holds (wording proposed for the advisor, `HOPCIM_PROPOSAL.md` §13b)

> The published, pretrained int8 MLPerf Tiny KWS model, unchanged and bit-exact to TFLite's reference kernels,
> together with a standard MFCC computed exactly, both executed on one digital CIM inside a RISC-V microcontroller
> whose CPU sleeps while a hardware sequencer runs the network; shown live on an FPGA, with the bottleneck measured as
> it moves.

### 4.2 What is new, with honest novelty levels

1. **The exact integer formulation of the standard MFCC on a digital bit-serial CIM, with a pretrained model kept
   unchanged.** This is the core novelty. Its limit: it gives exactness, not speed; alone it is 1.1×.
2. **Measured "what must move into hardware".** The bottleneck was measured at every step:
   - 47.97 M cycles on the CPU alone;
   - 13.02 M with the CPU driving the CIM;
   - 85,420 with the sequencer;
   - 67,909 with S;
   - 263 CPU cycles per decision for the whole pipeline.
3. **Exact, data-dependent optimisations chosen from measurement (S).** Sign-gated requantisation is a known idea
   (cf. SnaPEA); the second lane is standard parallelisation. The value is the measured −20.5 % and the f_min of
   4.096 MHz.
4. **A cycle-exact schedule model** (`model/streamfuse_sched.py`) that equals the RTL of R6b, S and B1 in every segment
   and access count. Its predictions for designs not built (four-block B) are therefore credible.
5. **Exact verification against external references:** TFLite reference kernels, and MLCommons' runner.

The sequencer, streaming, rate control (k-hop) and layer fusion are **engineering built on known ideas**, not
inventions.

### 4.3 Never claim

- FFT-on-CIM; mel/DCT on a memory macro; streaming CIM; learned front-ends; autonomous CIM controllers.
- Sharing the datapath between front-end and network as new or as a saving. Fu et al. share an FFT core in silicon,
  and B3 found no saving here.
- Exactness as such: Fu et al. state that their chip equals their own model. HopCIM's point is equality with an
  **external, unchanged** reference.
- RISC-V + CIM for KWS (CIMR-V, RV-Flash); a CIM in a KWS SoC (Dong et al.).
- **Any energy advantage** over measured chips. HopCIM's energies are 28 nm model estimates from an FPGA prototype.
- "Faster than" any MLPerf submission.
- That other work is unreliable or wrong. The student dislikes that framing: present trade-offs instead.
- **The depthwise split as an idea:** offloading depthwise layers to a digital unit beside a CIM is published
  (Garofalo et al., JETCAS 2022; DIANA runs it in silicon, with HTVM running the DS-CNN). Claim only the measured,
  priced and exact implementation on this network.
- For the second network: "HopCIM is a general-purpose accelerator", "one sequencer runs two networks", or any power,
  speed or accuracy claim. Claim only that the engine and the exact arithmetic carry over.

### 4.4 Closest works (full texts read; details in `docs/NOVELTY_SYSTEM.md`)

| Work | What it is | Why HopCIM differs |
|---|---|---|
| Dong et al., JSSC 2026 | measured 55 nm SoC: MLPerf Tiny DS-CNN *architecture* retrained for 35 keywords on a non-CIM DNN accelerator, MFCC on a DSP; CIM only for voice-activity detection; Arm M0/M4 | closest system; network not on the CIM; exactness not discussed |
| Jia et al., ISCAS 2022 | simulated 28 nm; log moved before mel; mel+DCT merged as a ternary matrix on the memory-boundary array; FFT in logic; jointly retrained | closest front-end; not the standard MFCC; simulation only |
| Fu et al., TVLSI 2026 | measured 65 nm; shared FFT core runs the network; log-mel without DCT; distilled model; chip = its own quantised model | sharing and self-exactness taken |
| Shan et al., ISSCC 2020 | measured 28 nm; serial-FFT MFCC circuit; binarised DS-CNN; no CPU | dedicated front-end, custom model |
| Lee et al., TCAS-I 2026 (KWS-CIM) | eDRAM charge-domain macro; KWS accuracy from software with measured macro statistics | cite its accuracies as such |
| Kang et al., ASP-DAC 2025 (RV-Flash) | RISC-V + binary eFlash CIM + FP16 vector unit; CIM alone −7.7 to −11.4 % latency | supports the Amdahl argument |
| CIMR-V (Guo, Chang et al., NYCU/ITRI, ISCAS 2024, arXiv 2503.22072) | the student's own group's work: 28 nm synthesis, 1-bit 10T SRAM CIM, Ibex host, raw-audio binary 1-D CNN, 94.02 % (12 classes) | the advisor will compare against it; learned front-end, custom model |
| iMCU (Lin et al., JSSC 2024) | measured 28 nm digital-IMC RISC-V MCU running a published MLPerf Tiny model (ResNet) | the reason the claim must say *keyword spotting*: a published MLPerf Tiny model on a digital CIM in a RISC-V MCU exists |
| Garofalo et al., JETCAS 2022; DIANA (JSSC 2023) with HTVM/MATCH | heterogeneous IMC clusters/SoCs with digital depthwise; HTVM runs the DS-CNN on DIANA | the depthwise split is theirs (§4.3) |

The manual, logged-in IEEE Xplore search is **still the student's to do**, so the thesis keeps "to our knowledge".

---

## 5. Repository map (`~/APlus_Project/hopcim`, a git repo)

The short map is `README.md`; documents are indexed in `docs/README.md`, models in `model/README.md`.

| Path | Content |
|---|---|
| `rtl/` | all RTL (§2.1); `hopcim.core` is the FuseSoC core `hopcim:ip:cim_endpoint` |
| `tb/` | testbenches (`tb_cim_*.sv`), `cim_assertions.sv`, `energy_counters.sv`, `hop_monitor.sv`, `cim_engine_tm.sv` (timing-parametric engine), vectors (`*_vec*`; `seq_vec_all/` is gitignored; `net2_vec/` holds 4 images) |
| `sim/` | run scripts (`run_*_tb.sh`, `run_all.sh` = 54 benches, `run_seq_allclips.sh`, `run_net2_*.sh`, `run_net2_mutants.py`, `run_cov.sh`, `run_*_nl.sh` gate-level, `run_soc.sh`); **`sim/logs/`** all bench logs (moved there 2026-11-26); `dwu_jobs/`, `net2_jobs/` long job outputs; `sim/bin/` Verilator SoC models (gitignored); `obj_*` build folders are regenerated on demand |
| `syn/` | Vivado out-of-context script `ooc.tcl` and reports; `vivado_miscompile/` (5-line repro of a Vivado 2024.1 bug) |
| `model/` | Python golden models, schedule/energy models, generators and their results, side by side (several are cited by frozen gates, so they stay where they are; index in `model/README.md`) |
| `sw/` | `cim.h` (driver), generated tables; `sw/applications/` = 23 board apps (§6.4) |
| `board/` | **the board scripts** (`board_run.sh`, `board_suite.sh`, `board_khop.sh`, `board_kwlog.sh`) and `xheep_env.sh`, versioned here since 2026-11-26; run them through the links in `planenest/hw/` (they locate the X-HEEP tree from the path they are started from) |
| `xheep/` | X-HEEP as a patch series: `base.env`, `patches/0001–0008`, `setup.sh`, `check.sh`, `build.sh`, `lib.sh`, `generated*.sha256` (one per configuration), `README.md` |
| `mlperf/` | committed: the TensorFlow Lite Micro baselines (`tflm_vendor.py`, `build_tflm.sh`, `build_tflm_resnet.sh`, `gen_tflm_resnet.py`); uncommitted: the MLPerf runner integration (`board_mlperf.sh`, `probe.py`, `check_session.py`, YAMLs, `reference*.py`, `public_kws.py`, `runs/`, …) |
| `third_party/` | TensorFlow Lite Micro headers (vendored at commit `c0ecf6ff`, `VENDORED.md`), `tflm_xheep_assert.h` |
| `figures/` | `figdata.py` reads every plotted number from evidence; `make_figures.py` draws `out/*.pdf|png` (needs `~/venvs/defines`); `make_lit_figures.py` the literature figures |
| `freeze/` | `freeze.py` (snapshot / manifest / check), `selftest.py`, `MANIFEST.json`, `RESULTS.md`, `RESULTS_FREEZE.md`, snapshots of board logs, board scripts, gate and novelty files, as of the 2026-11-13 freeze (paths in its texts are as of then) |
| `docs/` | indexed in `docs/README.md` |

**Pinned paths.** The frozen gate documents cite files by path, and their `.sha256` files record their own paths;
`freeze/freeze.py` hard-codes about 70 repository paths and `HopCIM_Illustrations/` reads more. Never move a file
that a `docs/*_GATES.md` cites; search for any other file's path before moving it.

### 5.1 Outside the repository (`~/APlus_Project/`)

| Path | Content | Note |
|---|---|---|
| `planenest/hw/` | links to `hopcim/board/*`, board ELFs, UART logs, bitstreams | the name is historical |
| `planenest/hw/x-heep/` | the **working X-HEEP tree** (default configuration); `hw/ip/hopcim` links to this repo | never edit by hand; does not yet have patches 0007/0008 |
| `planenest/hw/x-heep-dwu/` | tree that built the S+DWU bitstreams | lacks 0007/0008 |
| `planenest/hw/x-heep-net2/`, `x-heep-net2def/` | trees with the second network (256 KiB / default) | patched to 0008 |
| `planenest/hw/x-heep-256k-c180/` | 180-KiB-code configuration, for `tflm_resnet` | |
| `planenest/hw/x-heep-256k*/` (`-fw`, `-fw2`) | older 256 KiB trees (TFLite Micro KWS baseline) | `-fw` is stale |
| `planenest/hw/bitstreams/` | all bitstreams: `…cim128kdwu.bit` (S+DWU), `…cim128mic4dwu.bit` (S+DWU, 4.096 MHz), `…cim128knet2.bit` (S+DWU + ResNet8, 256 KiB), `…cim128ks.bit` (S), `…cim128kb1.bit` (B1), `…cim128mic4s.bit`, `…cim128ks256k.bit`, … | older ones pinned in `freeze/MANIFEST.json` |
| `planenest/data/cifar-10-batches-py/` | CIFAR-10 test set for ResNet8 | |
| `kws_gate/` | accuracy-gate scripts, pretrained MLPerf Tiny KWS models, GSC test set (`data/test`) | read by `freeze.py`; dataset not in git |
| `mlperf_tiny/` | MLCommons `tiny` (models, runner), `energyrunner`, `datasets_open`, `results/`, `src/tflite-micro` (c0ecf6ff) | |
| `recim-rv/accelforge_pilot/` | AccelForge + CACTI environment for `model/energy_tool_run.py` | the only part of `recim-rv` kept |
| `research_discovery/20261026_application_driven/` | the round-5 report that chose HopCIM | read by `freeze.py snapshot` |
| `HopCIM_Illustrations/` | verified illustrations; `run_all.sh` reads repository files by path | |
| `hopcim_backup/`, `HopCIM_Backup_2026-11-20/` | release tarballs; a full local backup | |
| `hopcim_recordings/` | ambient/voice recordings | **private: never upload** |
| `Literature review/` | paper PDFs | **never upload** |

---

## 6. How to build, run and check

### 6.1 Environment

| Tool | Where |
|---|---|
| RISC-V GCC 10.1 (SiFive), Verilator 5.040, Vivado 2024.1, verible | `source ~/APlus_Project/planenest/hw/xheep_env.sh` (a link to `hopcim/board/xheep_env.sh`) |
| X-HEEP Python venv | `~/APlus_Project/planenest/hw/x-heep/.venv` (pass as `XHEEP_VENV`; `make venv` needs the network) |
| Models, generators | `~/venvs/planenest/bin/python` |
| TensorFlow (accuracy gate, network check, softmax params) | `~/venvs/kwsgate` |
| matplotlib (figures), DeFiNES | `~/venvs/defines` |
| ZigZag-IMC | `~/venvs/zigzag`, `~/tools/energy/zigzag-imc` |
| MLPerf runner, tflite-micro Python package | `~/venvs/mlperf_tiny` |
| Energy tool (AccelForge, CACTI) | `~/APlus_Project/recim-rv/accelforge_pilot/.venv312` |
| Freeze tools, most scripts | `python3` (standard library only) |

Node.js is not installed. Neither is an ASIC flow: Yosys/OpenSTA downloads were denied, so don't work around that.

### 6.2 Common commands (from `~/APlus_Project` unless noted)

```bash
# frozen claims, the self-test of that check, the RTL regression
python3 hopcim/freeze/freeze.py check && python3 hopcim/freeze/selftest.py
hopcim/sim/run_all.sh                               # 54 benches, ~15 min (longer after the obj_* folders were deleted); logs in sim/logs/

# accuracy gate and mapping self-test
~/venvs/kwsgate/bin/python hopcim/model/gate2_int.py
~/venvs/planenest/bin/python -c "import sys; sys.path.insert(0,'hopcim/model'); import cim_map; cim_map.self_test(4890)"

# an X-HEEP tree from upstream + the patch series; bitstreams; firmware
XHEEP_VENV=~/APlus_Project/planenest/hw/x-heep/.venv hopcim/xheep/setup.sh ~/xheep-hopcim [configs/kws_nexys_256k.py]
hopcim/xheep/build.sh ~/xheep-hopcim dwu            # variants: dwu, dwu-4, s, s4, b1, b1-4, s256, net2-256, r6b, r6b5; sim, sim-ded, sim-s, sim-b1, sim-dwu, sim-net2-256
hopcim/xheep/build.sh ~/xheep-hopcim app hopcim_seq_dwu_nexys hopcim_seq "-DREFERENCE_CLOCK_Hz=15360000 -DCIM_DWU=4"   # default design
hopcim/mlperf/build_tflm_resnet.sh <tree made with configs/kws_nexys_256k_c180.py>       # ResNet8 CPU baseline (SIM=1: Verilator hex)
hopcim/xheep/check.sh                               # checks planenest/hw/x-heep against the series

# board (Nexys A7 on USB; SW0 must be up)
cd planenest/hw && ./board_run.sh <elf> bitstreams/<bit> [seconds]   # FOLLOW=1 prints the UART live

# figures
~/venvs/defines/bin/python hopcim/figures/make_figures.py

# MLPerf runner (see docs/MLPERF_TINY.md §10 for every step)
hopcim/mlperf/board_mlperf.sh <elf> smoke|p|a       # OPEN=1 for the open division, TFLM=1 for the CPU baseline
```

### 6.3 Bitstream variants (`xheep/build.sh`; FuseSoC flags)

| Variant | Flags beyond `use_bscane_xilinx use_cim use_cim_mic` | Bitstream | What |
|---|---|---|---|
| `dwu` | `use_cim_s use_cim_dwu` | `xheep_nexys_bscan_cim128kdwu.bit` | **default S+DWU**, 15.36 MHz (firmware `-DCIM_DWU=4`) |
| `dwu-4` | `use_cim_s use_cim_dwu use_cim_mic4` | `…cim128mic4dwu.bit` | S+DWU at 4.096 MHz |
| `net2-256` | `use_cim_s use_cim_dwu use_cim_net2` (256 KiB tree) | `…cim128knet2.bit` | S+DWU plus the second network (ResNet8) |
| `s` | `use_cim_s` | `xheep_nexys_bscan_cim128ks.bit` | **baseline S**, 15.36 MHz |
| `s4` | `use_cim_s use_cim_mic4` | `…cim128mic4s.bit` | S at 4.096 MHz (MMCM makes 8.192 MHz, BUFGCE halves it) |
| `b1` | `use_cim_s use_cim_b1` | `…cim128kb1.bit` | S + one StreamFuse-B block |
| `s256` | `use_cim_s` (tree with `configs/kws_nexys_256k.py`) | `…cim128ks256k.bit` | S with 256 KiB RAM (TFLite Micro baseline) |
| `r6b` / `r6b5` | — / `use_cim_mic5` | `…cim128k.bit` / `…cim128mic5.bit` | R6b at 15.36 / 5.12 MHz |

Other flag: `use_cim_ded` selects the B3 front-end.

### 6.4 Firmware apps (`sw/applications/`, linked into the X-HEEP tree file by file)

| App | What it does |
|---|---|
| `census_microbench` | CPU cost census |
| `cim_smoke` | SoC smoke test |
| `hopcim_cpu` | network on the CPU alone (baseline A) |
| `hopcim_mfcc` | R4b, front-end CPU-orchestrated |
| `hopcim_fe`, `hopcim_fe_irq` (+ `_ded`) | R5a front-end |
| `hopcim_net` (20-clip build), `hopcim_net1` (one clip) | CPU-orchestrated network |
| `hopcim_seq` | sequencer; 60-run exactness check |
| `hopcim_seq_dbg` | layer readback on the board |
| `hopcim_stream` (+ `_ded`) | window path |
| `hopcim_live` | live microphone, LEDs (`-DNET_K=<k>`) |
| `hopcim_micrec` | records 1.5 s plus decisions, checked by `model/check_micrec.py` |
| `hopcim_kwtest` | scripted spoken test, scored by `model/score_kwtest.py` |
| `hopcim_kwlog` | streams audio + logits at 921,600 baud |
| `pdm_capture` | raw microphone capture |
| `hopcim_mlperf` | MLPerf runner DUT (uncommitted) |
| `tflm_mlperf` | TFLite Micro CPU baseline, KWS |
| `hopcim_resnet` | the second network on the CIM: 20 images, exactness and cycles (needs the `net2-256` bitstream) |
| `tflm_resnet` | TFLite Micro running ResNet8 on the CPU (2 images; 180-KiB code configuration) |

Build with `-DREFERENCE_CLOCK_Hz=<Hz>` matching the bitstream's clock, and with `-DCIM_DWU=4` for the S+DWU bitstreams:
**the firmware must match the bitstream** (S firmware on S+DWU, or the reverse, gives wrong logits).

### 6.5 Measurement rules

- **CPU-active cycles.** On the FPGA the X-HEEP clock gate is a pass-through, so `mcycle` keeps counting in WFI.
  CPU-active time therefore uses `mcountinhibit` around WFI plus the endpoint's `CYCLES` register.
- **Cycles are the portable figure;** 15.36 MHz is an FPGA clock.
- **f_min** = the smallest integer multiple of the 1.024 MHz PDM clock whose hop (20,480 × N cycles) holds the worst
  hop.

---

## 7. The freeze system (how claims stay honest)

- `freeze/freeze.py check` recomputes every claim in `RESULTS.md` from its evidence file, checks that the documents
  quote the same numbers, and checks hashes in `MANIFEST.json`.
- `freeze/selftest.py` proves the check can fail. It clones HEAD and breaks things deliberately; every break must be
  caught (`freeze/selftest.log`). To test uncommitted work, commit it in a throw-away copy and set `HOPCIM_PROJECT`.
- **After any change to a claim or its evidence**, in this order:
  1. `freeze.py snapshot` if the board logs changed;
  2. `freeze.py manifest`;
  3. `freeze.py check`;
  4. `selftest.py`;
  5. commit;
  6. tag `freeze-YYYY-MM-DD`;
  7. the **user** pushes with `cd ~/APlus_Project/hopcim && git push origin main --tags`;
  8. the user attaches the release files from `~/APlus_Project/hopcim_backup/` in GitHub's web page (no `gh` CLI
     here).
- Tags so far: `freeze-2026-11-07`, `-07b`, `-08`, `-08s`, `-08b1`, `-13` (latest). Everything since 2026-11-13 is
  committed but not yet frozen: `freeze.py check` fails on 7 known lines (new bitstreams, ELFs, patches and firmware;
  `RESULTS_FREEZE.md`; the working tree lacking patches 0007/0008; board scripts edited since). A re-freeze is the
  student's decision.

---

## 8. Glossary and name collisions (read carefully)

| Term | Meaning |
|---|---|
| hop | 20 ms = 320 samples at 16 kHz; one new frame per hop |
| frame | 30 ms = 480 samples |
| window | 49 frames = 1 s, the network's input |
| decision | one network inference; every hop, or every k hops (k = 2 → 40 ms) |
| pass | one CIM operation: one input vector against one 64 × 16 tile |
| b, D | input bit width; issue interval D = max(8, b) cycles (the macro assumption) |
| lraw | log-mel values (40 per frame) before window normalisation |
| FEAT | the 49 × 10 int8 feature rows the network reads |
| go → IRQ | network time from the CPU's start store to the interrupt (board figure) |
| f_min | lowest system clock with no overrun on the live path |
| [M] / [E] / [P] / [H] | measured / estimate-model / published / hypothesis — every number carries one |
| S+DWU, DWU | the default design: S plus the digital depthwise unit (`SEQ_DWU = 4`) |
| mode U | the low-power firmware mode: network every 16 hops (`NET_K = 16`), decision = sum of 2 (`KW_SUM = 2`) |
| net2 | the second network, MLPerf Tiny's ResNet8, on `cim_gseq` (gates N1–N7) |

**"B1" means three different things:**
1. In the proposal's baseline list (§9.2): "CIM for MACs only, CPU epilogue" (R4b).
2. In the StreamFuse work, and **most often now**: one block of StreamFuse-B in RTL (`STREAMFUSE_B1.md`, flag
   `use_cim_b1`).
3. In `freeze/RESULTS.md`, claim B1 is the CPU-only network (47,974,801 cycles).

Other collisions:
- **"A–E"** are end-to-end configurations in `EVALUATION.md`. StreamFuse also has variants named A, B, C plus B0 (R6b)
  and S.
- **"S"** is the baseline S (sign-gated requantisation + second lane + no L8 write), the comparison baseline (the default design is S+DWU since 2026-11-25, `docs/DESIGNS.md`). Claim IDs
  S1–S8 in `RESULTS.md` refer to it.

---

## 9. Pitfalls already paid for (don't rediscover them)

**Vivado and RTL**
- **Vivado 2024.1 miscompiles signed `int` address arithmetic** (`syn/vivado_miscompile/`):
  - Write address arithmetic as unsigned bit-level logic, with no negative intermediates.
  - Before a bitstream, simulate risky blocks' Vivado netlists against the RTL (`xsim -L unisims_ver`; define
    `CIM_ZERO_INIT` for 4-state netlist sims; start gate-level benches after the 100 ns GSR).
  - The user decided **not** to report this bug to AMD; don't ask again.
- An unsigned term in a signed sum zero-extends: make every operand `$signed`.
- **BRAM inference:**
  - Any register between a RAM and a DSP, or a read-data register absorbed into DSP BREG, turns the weights into
    LUTRAM. Read one cycle later straight into the DSP, or register the address.
  - 32-bit partial writes into 128-bit words also break inference.
  - Buffers written inside an async-reset FSM block become flip-flops.

**X-HEEP and FuseSoC**
- **Never run two FuseSoC builds in the same X-HEEP tree at once.** The boot-ROM generator runs in place and races.
- A FuseSoC `--build-root` must keep the default 3-level depth.
- A FuseSoC rebuild wipes `sim-verilator`: keep model binaries in `hopcim/sim/bin`.
- The Verilator testharness taps the CIM directly, like the FPGA. A crossbar in simulation made cycle counts differ
  from the board.
- X-HEEP's `printf` re-initialises the UART at 9,600 baud on every call. Apps that need speed drive the UART
  themselves.
- Every Nexys build reports negative slack on X-HEEP's unused `spi_slave_clk_pin` CDC. Ignore it and quote the system
  clock's slack.
- BSCANE2 debug needs TCK on a BUFG plus `bscan.xdc` (in patch 0002). SW0 must be up, or OpenOCD reports
  `dtmcontrol=0`.
- The X-HEEP pad ring owns the microphone IOBUFs, so the microphone goes through new `x_heep_system` ports in the
  template.
- The 4.096 MHz clock:
  - `clk_wiz` ignores MMCM settings unless `CONFIG.OVERRIDE_MMCM` is true.
  - A BUFG→BUFGCE cascade fails placement.
- **TFLite Micro on X-HEEP:**
  - X-HEEP's CMake puts every header directory under an app on `-I`, so vendored headers live in `third_party/tflm`.
  - CMake appends `-std` after `COMPILER_FLAGS`, so patch 0005 sets C++17.
  - X-HEEP's freestanding `assert.h` breaks C++ (fixed by `third_party/tflm_xheep_assert.h` plus its guard define).
  - The X-HEEP runtime already defines `operator delete` and `__cxa_pure_virtual`.
  - TFLite Micro reuses the input tensor's arena bytes, so copy the input inside the timed `th_infer`.

**Board, shell and measurement**
- Board runs: Ctrl-C used to orphan `openocd` and `cat /dev/ttyUSB1` (`board_run.sh` now traps it and names any
  leftovers). The CPU reset button leaves X-HEEP in the boot ROM; rerun `board_run.sh`.
- Shell: `pkill -f` and `pgrep -f` match the invoking shell when the pattern's text is in the same command. Use
  `[x]pattern` and keep kills in their own command.
- Never kill the user's processes.
- **Disk:** scratch clean clones once filled the disk. Delete scratch trees after use, and run `df -h` before long
  runs.
- **A self-test must include deliberately broken variants.** Early "mutants" were equivalent and proved nothing.
- **Cycle counts of CPU code depend on the GCC build:** compare simulation and board only with identical C code. A
  TFLite Micro rebuild moved its cycles by 1.4 %. The go → IRQ offset is firmware-dependent too (68 with S's firmware,
  67 with the depthwise unit's: the compiler hoisted one instruction out of the WFI loop).

**Toolchain and verification, learned since 2026-11-14**
- GNU ld 2.35 relaxes accesses to small read-only constants to global-pointer-relative ones when they are within
  ±2 KiB *plus a page* of the global pointer (`DATA_SEGMENT_ALIGN` in X-HEEP's linker script), then fails with
  "relocation truncated to fit: R_RISCV_GPREL_I". Keep the end of the code section several KB away from the data
  section (patch 0008 uses 180 KiB of code for that reason).
- A broken variant that changes no output cannot be caught by comparing outputs: check the property itself (the
  second network's bench checks its rewrite margin directly).
- `xheep/check.sh --record` on a partly regenerated tree records too few files: record only from a fresh `setup.sh`
  tree.
- Verilator rejects a conditional on unpacked-array ports: use an `always_comb` mux loop.
- `board_run.sh` prints "timeout" when a firmware does not end with a `done` line, even if all output arrived.

---

## 10. Rules set by the owner (follow them; they override defaults)

### 10.1 Research method

- **Exactness first.** Golden models in `model/` are the specification. Every hardware or firmware change must be
  bit-exact against them, on all 4,890 clips where applicable.
- **The model is never retrained or changed** for novelty.
- **Digital only:** RTL, firmware, Verilator, FPGA. No analog or device-level CIM, no SPICE, no tape-out. **No energy
  claims from FPGA power.** Energy appears only as labelled [E] model estimates, with their limits stated.
- **Label every number** [M], [E], [P] or [H]. Separate verified facts from hypotheses.
- **Frozen things stay frozen:**
  - Hypotheses H1–H6 and their falsification lines: only the status column changes, and H1/H3 are reported as not
    coming out as written.
  - The gate documents `docs/STREAMFUSE_GATES.md`, `WHOLEWORD_GATES.md`, `DWSWAP_GATES.md`, `DWU_GATES.md`,
    `NET2_GATES.md` (sha256-pinned, never edited; corrections go into the results documents; no threshold moved).
  - Gate protocols.
- **HopCIM is keyword spotting only** (2026-11-26). No multi-network work: no KWS on `cim_gseq`, no ResNet8 power
  model, no "one sequencer for both".
- **RTL only with the student's go-ahead** (given for the depthwise unit and for ResNet8).
- **Work order** (the student's plan of 2026-11-08): evidence quality > claim quality > simple exact optimisation >
  new complex RTL.
  - Every new optimisation must beat **S**, not only R6b.
  - **Do not build** StreamFuse A or C, ExactReuse RTL, or large rewrites.
  - **All four blocks of StreamFuse-B only on the user's explicit go-ahead.** It is the open advisor decision.
- **Honesty about novelty:** give honest novelty levels; present trade-offs, never that other work is unreliable.
- **Strong baselines:** require a headroom check against the strongest static baseline before proposing a new
  optimiser or scheduler.

### 10.2 Repository, X-HEEP and git

- **Do not move files a frozen document cites;** check `freeze.py`, `HopCIM_Illustrations/` and the docs for any
  other path before moving it (`README.md`, "Pinned paths").
- **X-HEEP changes only through `hopcim/xheep/patches/`.** Edit in a tree made by `setup.sh`, regenerate the series
  (`xheep/README.md` "Changing X-HEEP"), and run `xheep/check.sh`. Never hand-edit `planenest/hw/x-heep`.
- **Commit or push only when the user asks** (or as part of a freeze they asked for). Freezes are committed on
  `main`.
  - Repo-local identity: `hansen0411 <hansennguyen04@gmail.com>`.
  - Commit messages end with the co-author trailer the session specifies.
  - The agent's `git push` has been blocked by permissions: give the user the push command.
- **GitHub backup:** private repo `git@github.com:hansennguyen04/HopCIM.git` (origin, branch `main` + freeze tags).
  SSH auth is set in the repo's `core.sshCommand`. **Never print or share the private key.**
- **Never upload to GitHub (repo or releases):**
  - voice and ambient recordings (`hopcim_recordings/`, `pdm_capture` `.wav`);
  - the `Literature review/` PDFs;
  - earlier research directions (deleted locally on 2026-11-26).
- Do not send the student's university email address to any external service.

### 10.3 People, tools and the machine

- **Recordings of people:** consent, coded speaker names, stored outside the repository.
- **Downloads and installs:** the OSS CAD Suite / ASIC-flow download was denied. Don't work around it; ask.
  ZigZag-IMC was approved and run.
- **Machine:** check `df -h` before long runs, delete scratch trees, never kill the user's processes, never two
  FuseSoC builds in one tree.

### 10.4 Reporting

- **MLPerf wording:** every HopCIM MLPerf number says "unverified" plus *Result not verified by MLCommons
  Association.* Comparisons with verified results also carry the second footnote. Write MLPerf®.
  Cycles-per-inference comparisons are a normalisation, not speed. No energy comparison.
- **Communication style the student prefers:**
  - plain, precise language; tables; numbers with sources;
  - what was measured vs estimated;
  - honest limits;
  - a recommendation rather than an exhaustive survey;
  - web-research answers end with a "Sources:" list.

---

## 11. Timeline (condensed)

| Date | Milestone |
|---|---|
| 2026-10-26 | HopCIM chosen; novelty gate passed with a narrowed claim; accuracy gate 91.80 % vs 91.66 %; board bring-up (BSCANE2 debug over USB, PDM microphone) |
| 2026-10-27 | R1–R4b, the 1.1× Amdahl result; R5a hardware front-end; B3 (H4 not supported); CIM grown to 128 tiles; DS-CNN bit-exact on the CIM; R6b sequencer 85,420 cycles; Vivado miscompile found and fixed |
| 2026-10-28 – 10-30 | R5b window path; R7 live microphone path at 15.36 and 5.12 MHz; verification pass |
| 2026-11-05 | k-hop rate control; exact activation reuse evaluated in software (not built) |
| 2026-11-07 | One-speaker test (48/47/42 of 50); energy estimate [E]; first freeze; same-audio logger; 8.38-h ambient run; timing sweep; ZigZag-IMC run; verification |
| 2026-11-08 | StreamFuse study with pre-registered gates; **baseline S** in RTL and on the board (4.096 MHz); X-HEEP as a patch series (clean clone = same bitstream); **B1**; six closest works read in full |
| 2026-11-13 | GitHub backup; paper figures from the freeze; tag `freeze-2026-11-13` |
| 2026-11-14 | MLPerf Tiny runner (closed + open); public KWS results comparison; TFLite Micro CPU baseline on a 256 KiB bitstream (patch 0005); literature table |
| 2026-11-20 | full local backup `HopCIM_Backup_2026-11-20/`; verified illustrations `HopCIM_Illustrations/` |
| 2026-11-22 | whole-word study (mode U); depthwise swap-in study (G5 failed); depthwise unit gates frozen; DWU RTL |
| 2026-11-23 – 11-25 | DWU board run (R6) accepted; **S+DWU becomes the default design**; literature review; ResNet8 gates frozen, ResNet8 RTL |
| 2026-11-26 | ResNet8 board run (N5) and TFLite Micro baseline (N6, patch 0008); N7 4/4; commits `cf60772`, `1440e42`; earlier directions deleted; repository tidied (`sim/logs/`, `board/`, README indexes) |

---

## 12. Current state and open items (as of 2026-11-26)

### 12.1 Git

- Committed on `main`, not yet pushed (the student pushes: `cd ~/APlus_Project/hopcim && git push origin main`):
  `b610c36` (ResNet8 gates), `cf60772` (ResNet8 RTL, board, CPU baseline, patches 0007/0008, TFLite Micro files),
  `1440e42` (default design, literature review), and the repository tidy-up with this file.
- **Uncommitted on purpose:** the MLPerf runner work: `docs/MLPERF_TINY.md`, the runner files in `mlperf/` (and
  `mlperf/runs/`), `sw/applications/hopcim_mlperf/`, `EVALUATION.md` §14–15 and its TFLite Micro row, the proposal's
  MLPerf rows, and `xheep/README.md`'s app count and MLPerf link. Committed documents already cite `MLPERF_TINY.md`.

### 12.2 For the student

| Item | Detail |
|---|---|
| Board runs of the default design | mode U at 15.36 MHz and the four 4.096-MHz runs; commands and expected results in `docs/DESIGNS.md` §5 |
| X-HEEP trees | apply patches 0007/0008 to `planenest/hw/x-heep` and `x-heep-dwu` (`git am`, then `make mcu-gen`), so `xheep/check.sh` passes there again |
| Re-freeze | add the new claims, `freeze.py manifest`, check, self-test, tag; the student decides when |
| Commit the MLPerf work | when wanted; it also resolves the dangling `MLPERF_TINY.md` references |
| Energy tool location | offered: move `recim-rv/accelforge_pilot` under a HopCIM name (reinstall its two editable packages, update four paths) |
| Advisor meeting | the claim (§4.1) and its relation to CIMR-V; the default design; whether the [E] energy form is acceptable; how to present H1/H3; venue and deadline |
| IEEE Xplore search | the manual, logged-in search with the strings in `HOPCIM_PROPOSAL.md` §15.2 |
| Spoken runs | several speakers on the board with the same-audio logger; mode U needs one |
| Papers | the JSSC 2023 0.61-µW KWS ASIC (DOI 10.1109/JSSC.2023.3339528) is still wanted in full text |
| Writing | thesis chapters and the paper; figures come from the freeze and from `figures/` |

### 12.3 Not planned unless the student decides otherwise

- StreamFuse A/C, all four StreamFuse-B blocks, ExactReuse RTL.
- Retraining; multi-network features.
- Energy measurement on the FPGA; energy comparison with chips.
- Reporting the Vivado bug.
