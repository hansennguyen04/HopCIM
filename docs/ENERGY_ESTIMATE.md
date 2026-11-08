# Energy estimate at 28 nm, from exact activity and published tools

Status (2026-11-07): **a model-based estimate, not a measurement.** No number here comes from FPGA
power, and no silicon exists. Its value is the breakdown and the comparisons between design options;
treat the absolute figures as good to about a factor of two (§8). Reproduce with §10.

## 1. Result

| Average power, baseline | Decision every 20 ms | 40 ms | 80 ms |
|---|---:|---:|---:|
| Network | 69.5 µW | 34.7 µW | 17.4 µW |
| Front-end and window path | 1.3 | 1.3 | 1.3 |
| CPU (263 cycles per decision, upper bound) | 0.4 | 0.2 | 0.1 |
| Audio input | 2.9 | 2.9 | 2.9 |
| Leakage | 36.2 | 36.2 | 36.2 |
| **Total** | **110** | **75** | **58** |
| Leakage share | 33 % | 48 % | 62 % |
| Total across the sensitivity variants (§8) | 94–192 | 60–117 | 42–84 |

The microphone itself and the X-HEEP host (its SRAM, bus and peripherals) are not included.

One inference costs about **1.39 µJ** (1.09–3.01 µJ across the variants), or 523 fJ per MAC with
memory, requantisation and control included.

What the estimate shows:
1. **The CIM array is not where most of the energy goes.** With the measured bit activity it is 10 % of
   an inference. Requantisation (42 %), the activation memory (21 %) and tile selection (12 %) cost more.
   This depends on the macro style: on a Colonnade-style array, where every bit cell has its own adder,
   the array is 64 % (§4).
2. **The front-end is not the energy problem.** Both front-ends use about 1 µW at 50 frames per second.
   The CIM front-end uses less dynamic energy per frame than the dedicated one (20.8 against 44.5 nJ)
   but needs 6 more tiles. The choice moves the total by less than 4 % (§5).
3. **The 69 resident tiles need a real organisation.** One macro per tile would leak about 320 µW, and
   sharing the logic among 16 tiles costs extra energy per pass instead (§6). The engine model charged
   for neither.
4. **Lower decision rates hit leakage.** At 80 ms, leakage is 62 % of the total, so going slower gains
   less and less. Leakage reduction would be the next lever, and it is not modelled here.
5. **Activation reuse now shows a net saving, with leakage included** (§7). At 40 ms: 13.3 µW, or 18 %.
   That is the second condition for building it.

**The decision rate.** With a real speaker, 40 ms showed no loss against 20 ms (47 and 48 of 50 keywords)
while 80 ms lost six words (42 of 50; `LIVE_AUDIO.md` §9). So 40 ms is the supported operating point: an
estimated 32 % below 20 ms.

## 2. Method

Every number is one of three kinds, kept apart in `model/energy_est.py`.

**Counted** (exact, from this design):
- every memory access, pass, multiplication and busy cycle, from counters bound into the RTL bench
  (`tb/energy_counters.sv`, `sim/run_energy_counts.sh`), on a run that still passes all its checks;
- the bit activity of every pass on 40 GSC test clips (`model/energy_activity.py`);
- the work activation reuse skips (`REUSE_EVALUATION.md`).

**Published** (tool or paper):

| Part | Source | How it was used |
|---|---|---|
| CIM array, primary | ZigZag-IMC digital model and its 28 nm fit (Houshmand, Sun, Verhelst, arXiv 2305.18335): 6T cells, ripple-carry adder tree, 0.9 V | **Equations transcribed from its repository into `energy_est.py`; the tool itself has not been run** (§9) |
| CIM array, cross-check | AccelForge (commit 7312a7d) with its Colonnade model (Kim et al., JSSC 2021: digital, bit-serial, adder in every cell, 65 nm) | Run on HopCIM's pass shapes; at 28 nm by the library's scaling, 0.8 V |
| SRAM energy, leakage, area | CACTI through hwcomponents-cacti 1.0.40: low-standby-power (ITRS LSTP) cells and periphery, leakage-optimised organisation, 300 K | Run on each HopCIM memory; 28 nm is interpolated between 32 and 22 nm |
| Multipliers, adders | Aladdin models in hwcomponents-library 1.0.58; Horowitz (ISSCC 2014) as the alternative | Run at 28 nm |
| CPU | HEEPocrates, X-HEEP in TSMC 65 nm LP: 28 µW/MHz at 1.2 V, whole chip | Unscaled, so an upper bound |

**Assumed** (no tool here gives these; §8 varies each one):

| Assumption | Baseline |
|---|---|
| Tiles sharing one macro's logic | 16, the largest depth among ZigZag's validated chips |
| Gate equivalents | 6 per FPGA LUT, 6 per flip-flop, 1,500 per DSP |
| Logic activity | 10 % of gates per active cycle; blocks are clock-gated when idle |
| Logic leakage | 50 pW per gate equivalent, low-power 28 nm |

Scope of the macro model: 64 rows, 16 columns of int8, one input bit per cycle, as in `ENGINE_SPEC.md`.

## 3. Checks

| Check | Result |
|---|---|
| Counters against known figures | exact: 7,501 passes, 72,012 requantisations and 72,000 ACT writes per inference; 2,656,768 MACs; 67,509 engine bit-cycles; 512 twiddles, 128 magnitudes and 960 samples per frame (`cim_fe`); 4,608 butterfly reads and 5,120 writes per frame (B3) |
| Counter runs still correct | `tb_cim_mic` passes with both front-ends, 101 decisions exact, and ends at the same cycle as without counters (15,492,295 with `cim_fe`) |
| k = 4 run | per-inference counts unchanged, front-end still every hop |
| Transcribed ZigZag equations against the two chips with energy in its main validation script | 36.29 against 36.63 TOP/s/W (ISSCC 2022 15.5), 19.99 against 19.50 (ISSCC 2023 7.2): ratios 0.99, 1.03 |
| AccelForge's Colonnade against the paper, 0.8 V | 146.7 against 144.4, 68.3 against 68.3, 254.1 against 243.2, 1,173.5 against 1,169.7 fJ per MAC: ratios 1.00–1.04. **Not usable below 0.8 V:** 1.3× at 0.7 V, 2.8× at 0.6 V |
| Two sources for one full adder at 28 nm | 3.40 fJ (ZigZag) against 3.87 fJ (Aladdin) |
| CACTI's SRAM leakage against silicon | 10.4–25 pW per bit against a measured 7.4 pW per bit (28 nm FD-SOI, 8T, in retention) |

These show that the models were applied correctly and agree with what they were fitted to. They do
not validate the estimate for a chip that does not exist.

**Measured activity.** ZigZag's validation assumes input toggle rate × weight density = 0.25. HopCIM's
real values are much lower, because many inputs are small and the depthwise tiles are sparse:

| Pass group | conv1 | depthwise | pointwise | FC | FFT 1 | FFT 3 | mel | DCT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Toggle rate × weight density | 0.047 | 0.009 | 0.077 | 0.147 | 0.040 | 0.155 | 0.012 | 0.133 |

The model is not validated this far below 0.25, so §4 shows both.

## 4. One inference

| Part, nJ | Measured activity | Activity 0.25 | Colonnade array |
|---|---:|---:|---:|
| CIM array | 134 | 409 | 1,927 |
| Tile selection (16 tiles per macro) | 169 | 169 | 0 |
| Activation memory | 298 | 298 | 298 |
| FEAT memory | 0.4 | 0.4 | 0.4 |
| Requantisation parameters | 152 | 152 | 152 |
| Requantisation arithmetic | 436 | 436 | 436 |
| Input and result registers | 4 | 4 | 4 |
| Sequencer logic | 196 | 196 | 196 |
| **Total** | **1,389** | **1,664** | **3,013** |
| fJ per MAC, everything included | 523 | 626 | 1,134 |

Published KWS chips report from 2.91 nJ to 14 µJ per decision, but with other models and on real
silicon. A per-decision comparison is not meaningful unless it accounts for network size.

## 5. Front-end: on the CIM or dedicated

| One frame, nJ | `cim_fe`, measured | `cim_fe`, 0.25 | `cim_fe`, Colonnade | B3 `mfcc_ded` |
|---|---:|---:|---:|---:|
| CIM array and tile selection | 5.5 | 10.7 | 22.6 | 0 |
| Ring memory | 1.05 | 1.05 | 1.05 | 1.09 |
| Buffers | 2.0 | 2.0 | 2.0 | 11.7 |
| Multipliers | 6.4 | 6.4 | 6.4 | 24.7 |
| Logic and registers | 5.9 | 5.9 | 5.9 | 7.0 |
| **Total** | **20.8** | **26.0** | **37.9** | **44.5** |

The dedicated unit does all 2,304 butterflies with four 18×18 products each, which is where its
energy goes. The four-step FFT on the CIM needs 2,048 smaller products.

Leakage goes the other way, since the CIM front-end needs 69 tiles rather than 63:
- without crossing a macro boundary: about +0.7 µW, against −1.2 µW of dynamic power, so the CIM
  front-end is about 0.5 µW lower overall;
- at 16 tiles per macro, 69 tiles need a fifth macro: +5.0 µW, so B3 is 3.9 µW lower (106.4 against
  110.3 µW).

Either way the difference is under 4 % of the total, which agrees with `HOPCIM_PROPOSAL.md` §13a: the CIM
MFCC is one evaluated option, not an energy claim.

## 6. Leakage and how the tiles are organised

Leakage at the baseline, 36.2 µW: weight cells 5.9 (565 kbit at 10.4 pW per bit), other SRAM 4.1, CIM
macro logic 22.2, digital logic 4.0. Everything stays powered, at full voltage.

The 69 tiles are active one at a time, so they need not each have their own adder trees:

| Tiles per macro | Macros | Energy per inference | Total at 20 / 40 / 80 ms |
|---:|---:|---:|---:|
| 1 | 69 | 1,220 nJ | 386 / 355 / 340 µW |
| 4 | 18 | 1,270 nJ | 162 / 130 / 114 µW |
| 8 | 9 | 1,310 nJ | 124 / 91 / 75 µW |
| **16** | **5** | **1,389 nJ** | **110 / 75 / 58 µW** |
| 32 | 3 | 1,548 nJ | 109 / 71 / 51 µW |

Sharing raises the cost of selecting a tile (ZigZag's local bitline term, proportional to the depth),
but removes far more logic leakage. Depths above 16 are beyond ZigZag's validated chips. This is the
capacity question left open in `ENGINE_SPEC.md`, and it changes the total by more than any other choice.
The estimated area at 16 tiles per macro is 0.47 mm²: weight cells 0.10, macro logic 0.27, other SRAM
0.05, digital logic 0.05. That is not a layout.

## 7. Activation reuse, with leakage

Simple reuse against the same decision rate without it, using the measured skipped work of
`REUSE_EVALUATION.md`:

| Decision every, cache | Dynamic saved | Leakage added | **Net saving** | Break-even leakage |
|---|---:|---:|---:|---:|
| 20 ms, two caches, +72 KB | 31.6–33.8 µW | 6.2 µW | **+25.4 to +27.5 µW** | 55–59 pW per bit |
| 40 ms, one cache, +32 KB | 15.8–16.9 µW | 2.8 µW | **+13.0 to +14.1 µW** | 62–66 pW per bit |
| 80 ms, one cache, +32 KB | 6.5–7.3 µW | 2.8 µW | **+3.7 to +4.5 µW** | 26–29 pW per bit |

Ranges are over the five stream groups.

- **Net positive in all 18 variants of §8.** At 40 ms, the lowest is +9.7 µW and the highest +32.2 µW.
  Break-even is 49–137 pW per bit, against 11 from CACTI for these arrays and a measured 7.4.
- **This replaces the earlier thresholds** of 0.9–10 pW per bit (`REUSE_EVALUATION.md` §7). Those counted
  only MACs and 10 fJ per ACT byte, both assumed. With tool numbers, an ACT byte costs about 620 fJ on a
  read, and a skipped position also skips its requantisation, its parameter reads and its control cycles.
- **Not counted:** the added control logic. At 5,000 gate equivalents that is about 0.3 µW of leakage.
- **Against plain 80 ms:** 40 ms with reuse is 62.0 µW, and plain 80 ms is 57.9 µW with no memory and no
  RTL. Reuse is worth building only if 80 ms loses on real speech and reuse still wins after the stronger
  modelling (`HOPCIM_PROPOSAL.md` §15.2).

The savings come from synthetic streams, not real speech. The worst case stays a full recompute.

## 8. Sensitivity

Each row changes one thing against the baseline.

| Variant | Inference, nJ | Total at 20 / 40 / 80 ms, µW | Reuse net at 20 / 40 ms, µW | Reuse break-even at 40 ms, pW per bit |
|---|---:|---:|---:|---:|
| Baseline | 1,389 | 110 / 75 / 58 | +26.0 / +13.3 | 63 |
| Activity 0.25 | 1,664 | 124 / 82 / 62 | +32.4 / +16.5 | 75 |
| Colonnade array, 0.8 V | 3,013 | 192 / 117 / 79 | +63.7 / +32.2 | 137 |
| Multipliers: Horowitz | 1,086 | 94 / 67 / 53 | +18.9 / +9.7 | 49 |
| Logic activity 5 % | 1,339 | 107 / 74 / 57 | +24.9 / +12.7 | 61 |
| Logic activity 20 % | 1,489 | 116 / 79 / 60 | +28.3 / +14.5 | 68 |
| 4 gate equivalents per LUT | 1,366 | 108 / 74 / 57 | +25.5 / +13.0 | 62 |
| 10 gate equivalents per LUT | 1,437 | 114 / 78 / 60 | +27.1 / +13.9 | 65 |
| Logic leakage 20 pW per gate | 1,389 | 95 / 60 / 42 | +26.0 / +13.3 | 63 |
| Logic leakage 100 pW per gate | 1,389 | 136 / 102 / 84 | +26.0 / +13.3 | 63 |
| SRAM 7.4 pW per bit (silicon) | 1,389 | 107 / 72 / 54 | +28.0 / +14.2 | 63 |
| SRAM 25 pW per bit | 1,389 | 122 / 87 / 69 | +17.9 / +9.7 | 63 |
| Cache as one flat array | 1,389 | 110 / 75 / 58 | +15.3 / +9.7 | 49 |
| 5.12 MHz clock | 1,389 | 109 / 74 / 57 | +26.0 / +13.3 | 63 |

The four tile organisations of §6 complete the 18 variants; their reuse net is +22.1 to +29.7 µW at
20 ms and +11.4 to +15.2 µW at 40 ms.

The clock barely matters here, because idle blocks are assumed clock-gated. Its real effect would be a
lower supply voltage, which is not modelled.

## 9. Limitations

- **ZigZag-IMC models full switching and its own mapping.** The tool has now been run (§11). It agrees with the transcribed
  equations where the array shape matches, and gives an upper bound for the array.
- **The digital logic is the weakest part:** FPGA sizes converted to gate equivalents, with an assumed
  activity. An open ASIC flow (`HOPCIM_PROPOSAL.md` §9.5) would replace these with synthesised numbers; it has not been run. It would not validate a 28-nm macro.
- **Mixed supply voltages:** 0.9 V for ZigZag, 0.8 V for Colonnade, CACTI's own for SRAM. Nothing is
  estimated at low voltage, although a 5–15 MHz design would run there.
- **Not modelled:** power gating, retention voltage, temperature above 300 K, clock tree, wires between
  blocks, loading weights, process corners.
- **Not included:** the microphone and the X-HEEP host. A published X-HEEP configuration leaks 29 µW at
  65 nm, 3 µW with everything possible powered down.
- CACTI's 28 nm is interpolated, and its ITRS data are not a foundry's. The toggle rates come from GSC
  clips, not from long quiet periods.

## 10. Reproduce

```
cd hopcim
sim/run_energy_counts.sh cim 5 1      # counters, about 40 s: model/energy_counts_cim.log
sim/run_energy_counts.sh ded 5 1      # model/energy_counts_ded.log
cd model
~/venvs/planenest/bin/python energy_activity.py 40       # bit activity, about 30 s: energy_activity.json
../../recim-rv/accelforge_pilot/.venv312/bin/python energy_tool_run.py    # tools, about 30 s: energy_components.json
python3 energy_est.py                                    # the tables, energy_results.json (frozen run: energy_run.log)
```

## 11. ZigZag-IMC itself (2026-11-07)

Until now its equations were transcribed into `energy_est.py`. Here the tool itself runs (commit df34a6f, its own
`mlperf_tiny/ds_cnn.onnx`, the same network), with HopCIM's macro as its hardware: 64 rows, 16 int8 columns, 1 bit per cycle, 16 tiles per macro
(`model/zigzag_run/`, log `run.log`). Two of its limits matter here: it requires 8-bit inputs (HopCIM's are 9 bits), and it has no activity input,
because its adders switch fully on the rows it maps (its README: "Sparsity impact is not included").

**The same settings: 8-bit input, full switching, pJ per inference.**

| Layers | Tool | Transcribed | Ratio | Why |
|---|---:|---:|---:|---|
| Pointwise, all 64 rows used | 633,443 | 633,443 | **1.000** | the same array and tile switching |
| Fully connected | 242 | 238 | 1.020 | the tool also charges the 4 unused columns' multipliers |
| Conv1, 40 of 64 rows | 103,385 | 150,958 | 0.685 | it charges used rows only; the transcription leaves that to the activity |
| Depthwise | 152,253 | 632,210 | 0.241 | the same, more strongly: 9 active rows per column, and it does not reselect the tile each pass |

**At the measured activity** (the tool's switching terms times the activity on the rows that hold weights, against `energy_est.py`'s equations at 8 bits):

| Layers | Activity on used rows | Tool's structure, nJ | Transcribed, nJ | Ratio |
|---|---:|---:|---:|---:|
| Conv1 | 0.075 | 13.5 | 13.4 | 1.010 |
| Depthwise | 0.066 | 36.9 | 33.6 | 1.097 |
| Pointwise | 0.077 | 69.3 | 69.3 | 1.000 |
| Fully connected | 0.147 | 0.04 | 0.04 | 1.017 |
| **Total** | | **119.8** | **116.4** | **1.029** |

So the transcription is exact where the structure matches, and within 3 % overall once the same activity is applied.

**As an upper bound.** Using the tool's raw figure (full switching, its mapping) for the array and tile select changes those from 303 to 889 nJ:

| | Baseline | With the tool's raw array |
|---|---:|---:|
| One inference, network | 1,389 nJ | 1,975 nJ |
| A decision every 20 / 40 / 80 ms | 110 / 75 / 58 µW | 140 / 90 / 65 µW |

These stay inside the ranges of §1 (94–192, 60–117, 42–84 µW), so the stated uncertainty holds.

**Other cross-checks.**
- Passes: the tool's own mapping gives exactly HopCIM's 500 (conv1), 2,000 (pointwise) and 1 (fully connected). For depthwise it needs 6,400 against HopCIM's 5,000: its
  search found a less dense placement than the block-diagonal one.
- Clock: 3.53 ns on this macro at 0.9 V, the input to `TIMING_SENSITIVITY.md`.
- Area: 0.099 mm² per macro, so 0.50 mm² for 5 macros, against 0.37 mm² here (weight cells 0.097, macro logic 0.273). The difference is mostly the cell array. Area is not a claim of this work.
- Memory is not compared: the tool's template (256 KB SRAM, DRAM) is not HopCIM's memory.

Reproduce: `~/venvs/zigzag/bin/python model/zigzag_run/run_zigzag.py`, after `python3 -m venv ~/venvs/zigzag` and `pip install -r requirements.txt` in the tool's folder.

## 12. The baseline S from its RTL counters (2026-11-08)

S (sign-gated requantisation, a second requantisation lane, no L8 write; `BASELINE_S.md`) is now built. Its activity
comes from the counters of `tb/energy_counters.sv` in the live-path bench (`sim/run_energy_counts.sh cim 5 1 s`,
101 decisions), which now also count the second lane, the gated results and the multiplies executed:
72,012 requantisations per inference, 36,176 gated (50.24 %), 35,836 multiplies, 64,000 ACT byte writes,
and 17,511 fewer sequencer cycles (85,352 → 67,841, equal to the sequencer bench). `python3 model/energy_est.py s_rtl`
prices them with the StreamFuse study's rules (`model/energy_s_rtl.log`, `energy_s_rtl.json`):

| [E] | R6b | S (RTL counts) |
|---|---:|---:|
| Network energy per inference | 1,389.3 nJ | **1,120.8 nJ** (−19.3 %) |
| of which requantisation arithmetic | 435.9 nJ | 220.0 nJ |
| of which sequencer logic | 195.6 nJ | 155.5 nJ |
| of which ACT memory | 298.2 nJ | 285.8 nJ |
| Total power at 20 / 40 / 80 ms | 110.3 / 75.3 / 57.9 µW | **97.0 / 68.8 / 54.7 µW** |
| Across the 18 variants of §8, saving per inference | | 8.9–22.0 % |

- **The RTL agrees with the model** of the study (1,119.8 nJ): ratio 1.0009, only because the bench's stream has
  fewer gated results (50.24 %) than the test-set mean (50.46 %).
- **Measured size:** S adds 403 LUT and 158 FF to `cim_seq` (out of context). The study charged the second lane as
  3,000 gate equivalents of leakage plus its operations. Charging the measured extra logic every cycle as well, which
  double-counts the lane's datapath, gives an upper bound of 1,138.0 nJ (−18.1 %) and 69.2 µW at 40 ms.
- **At S's lowest real-time clock, 4.096 MHz**, the always-clocked part of the audio path costs a little less:
  67.6 µW at 40 ms.
- `python3 model/energy_est.py` without arguments still reproduces `energy_run.log` byte for byte.

## References

- P. Houshmand, J. Sun, M. Verhelst, "Benchmarking and modeling of analog and digital SRAM in-memory computing architectures," arXiv 2305.18335; ZigZag-IMC: github.com/KULeuven-MICAS/zigzag-imc
- H. Kim, T. Yoo, T. T.-H. Kim, B. Kim, "Colonnade: A Reconfigurable SRAM-Based Digital Bit-Serial Compute-In-Memory Macro for Processing Neural Networks," IEEE JSSC 56(7), 2021, DOI 10.1109/JSSC.2021.3061508
- AccelForge: arXiv 2609.11906, github.com/Accelergy-Project/accelforge; HWComponents: github.com/Accelergy-Project/hwcomponents
- M. Horowitz, "Computing's energy problem (and what we can do about it)," ISSCC 2014
- HEEPocrates: epfl.ch/labs/esl/research/systems-on-chip/heepocrates; X-HEEP leakage: arXiv 2508.16959
- "An 8-T ULV SRAM macro in 28nm FDSOI with 7.4 pW/bit retention power and back-biased-scalable speed/energy trade-off," IEEE S3S 2018
