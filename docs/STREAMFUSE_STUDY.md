# StreamFuse model study: where the network's energy goes once the CPU is gone, and which exact dataflow changes pay

Status (2026-11-08): **model study done, no RTL.** The gates were frozen before any result (`STREAMFUSE_GATES.md`,
commit a80834c, baseline `freeze-2026-11-07b`). Energy is a model-based estimate [E]; cycles, exactness, hazards
and buffer sizes come from a schedule model that reproduces the RTL exactly (V1).

**Result: B passes all four gates, C passes all four, A fails two.** B passes the latency gate only with a
refinement found by the model after its first result (§5). Building B is the student's decision (§7).

## 1. Answer

| One inference, nJ [E] | B0 today | **S** = B0 + sign gating + 2nd lane | A | **B** | C |
|---|---:|---:|---:|---:|---:|
| CIM array | 134.2 | 134.2 | 134.2 | 134.2 | 134.2 |
| Tile selection | 169.1 | 169.1 | 169.1 | 169.1 | 169.1 |
| Activation memories | 298.2 | 285.8 | 170.2 | 85.8 | 42.4 |
| FEAT memory | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| Requantisation parameters | 152.0 | 152.0 | 152.0 | 152.0 | 152.0 |
| Requantisation arithmetic | 435.9 | 219.0 | 219.0 | 219.0 | 219.0 |
| Input and result registers | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 |
| Sequencer logic (with added control) | 195.6 | 155.5 | 162.7 | 173.2 | 198.0 |
| **Total** | **1,389.3** | **1,119.8** | 1,011.4 | **937.6** | 918.9 |
| Against S | | 1 | 0.903 | **0.837** | 0.821 |
| Total power at 20 / 40 / 80 ms, µW | 110.3 / 75.3 / 57.9 | 96.9 / 68.8 / 54.7 | 91.7 / 66.2 / 53.5 | 87.5 / 63.9 / 52.1 | 87.3 / 64.1 / 52.5 |
| Cycles per inference | 85,352 | 67,841 | 67,841 | 67,745 | 67,625 |
| Activation storage, bits | 140,000 | 140,000 | 146,720 | 80,544 | 37,696 |

What this shows:
1. **Once the CPU is gone, requantisation is the largest item, not the activation memory.** It is 588 nJ of 1,389
   (arithmetic and parameters); the activation memory is 298 nJ (21 %).
2. **The cheapest exact changes come first and are large.**
   - **Sign gating:** in layers L0–L8 the output zero point is −128, so a result with `acc + bias ≤ 0` is exactly −128. That holds for **50.5 %** of all requantisations (36,338 of 72,000 per inference, over the 4,890 test clips), and skipping their multiply saves 217 nJ.
   - **Second requantisation lane:** cycles fall 85,352 → 67,841, which also cuts the sequencer's logic energy by 40 nJ.
   - **Not writing L8 to ACT:** another 12 nJ.
   - **Together (S):** −19.4 % per inference and −8.6 % of total power at 40 ms, with almost no new logic.
3. **Beyond that, the cross-layer dataflow saves another 16 % (B) to 18 % (C) of the network's energy** [E], and 7 %
   of total power at 40 ms (68.8 → 63.9 µW). Over today's design, B is −32.5 % per inference and −15 % at 40 ms.
4. **Latency does not move** (B −0.14 %, C −0.32 % against S): requantisation and gathering bind, as
   `TIMING_SENSITIVITY.md` found. StreamFuse is a memory and energy change, not a speed change.
5. **Leakage is untouched.** At 40 ms it is still 36 µW of 64–69 µW in every design; the activation memory is under 2 µW of it.

## 2. The gates

| Gate | A | B | C |
|---|---|---|---|
| G1 exactness: L0–L8 and logits, 4,890 clips | 0 mismatches | 0 mismatches | 0 mismatches |
| G2 network energy ≤ 0.90 × S | **0.903, fails** | 0.837 | 0.821 |
| G2 guard: total power at 40 ms ≤ S (68.76 µW) | 66.23 | 63.90 | 64.13 |
| G3 activation storage ≤ 84,000 bits (0.60 × 140,000) | **146,720, fails** | 80,544 (0.575) | 37,696 (0.269) |
| G4 cycles ≤ S (67,841) | 67,841 | 67,745 | 67,625 |
| **Result** | **fails** | **passes** | **passes** |

A cannot pass G3 by construction: it adds a line buffer to both maps.

## 3. Validity of the models

| ID | Check | Result |
|---|---|---|
| V1 | Schedule model of B0 against the RTL: cycles per layer, and access counts against the RTL counters | **Exact**: 8,034 / 11,273 / 8,032 … / 98, total 85,352 (all 10 layers equal); 37,960 + 5,000 ACT reads, 72,000 byte writes, 1,175 FEAT reads, 7,501 passes, 72,012 requantisations, 325,064 X rows |
| V2 | Schedule model of B0 against `dscnn_int`, 4,890 clips | **0 mismatches** in every output of L0–L8 and the logits |
| V3 | B0 from the schedule model's counts against today's energy model | **Equal in all 18 variants** (1,389.3 nJ); `energy_est.py` without arguments still reproduces `energy_run.log` byte for byte; CACTI's existing entries unchanged |
| V4 | DeFiNES, layer by layer and depth-first | **Agrees within 0–11 % on the one block it computes at HopCIM's size**, the difference explained (§4); it cannot check the network-level reduction |

**The checks can fail** (`streamfuse_sched.py selftest`, 50 clips):

| Broken design | Caught by |
|---|---|
| Sign gate too wide (gates `x ≤ 63`) | 634,239 wrong values |
| Depthwise tap rows shifted in the X bank | 2,094,085 wrong values |
| C: pointwise reads the other fusion vector | version check: deadlock |
| B: line-buffer slot off by one | version check: deadlock |
| B with its source interlock removed | 38,940 hazards, 2,057,589 wrong values: **the interlock is required hardware** |

## 4. V4: DeFiNES

DeFiNES (HPCA 2023, built on ZigZag, validated against the DepFiN chip; `~/tools/energy/DeFiNES` at 7097d60) was run on
the DS-CNN's layers L0–L8. Its accelerator is described like HopCIM (`model/defines_run/hopcim_hw.py`): a 1,024-MAC
array with resident weights at no access cost, the X banks as one register level, column accumulators, and ACT and
line buffers at CACTI's 28-nm costs. DeFiNES chooses its own loop orders, so it checks the traffic, not the schedule.

**It agrees where it computes the same thing.** This version of DeFiNES has no SAME padding: to produce the final 25 × 5 map it
computes every earlier layer larger by the receptive field of the layers after it (conv1 at 33 × 13). Only the last
block, layers L7 and L8, has HopCIM's geometry:

| Last block, layer by layer | DeFiNES | Schedule model | Ratio |
|---|---:|---:|---:|
| Pointwise L8, ACT word reads | 1,250 | 1,250 | **1.000** |
| Depthwise L7, ACT word reads | 10,500 | 9,490 valid taps (11,250 with the padded taps) | 1.106 (0.933) |
| Depthwise L7, output written, 56-bit words | 1,250 | 8,000 bytes = 1,143 words (1,250 words touched) | 1.094 |

The depthwise difference is the zero border: DeFiNES stores and reads it, while HopCIM never reads a padded tap. The
writes agree once DeFiNES's whole-word rounding (10 words per position, as HopCIM's ACT) is taken into account.

**It cannot check the size of the depth-first reduction.** Over the whole network DeFiNES spends 2.1 times the
activation-memory energy of the schedule model (613.6 against 285.8 nJ), because of the enlarged maps, and finds depth-first execution 55 % cheaper in
activation-memory energy (613.6 → 278.2 nJ) against the schedule model's 85 % (S 285.8 → C 42.4 nJ). The direction
agrees; the size does not, for a reason that is understood: with its enlarged maps DeFiNES also spills 23,397 reads to
ACT. So V4 confirms the counts layer by layer, and the gain of B and C rests on the schedule model alone.

Two adaptations were needed and are in the input file: the X banks as one unshared register level (an input register
shared along the channel dimension cannot serve a depthwise layer and made DeFiNES retry forever), and memory sizes that
hold its padded maps. Output: `model/defines_run/run.log`, `model/streamfuse_defines.json`.

## 5. What the model found, and the refinements (reported as post hoc)

- **The fusion vector must be read in 8-byte chunks.** As first modelled (ten 7-byte words, like ACT), each pointwise
  gather after a depthwise pass took one cycle more than the engine allows: **B 68,241 cycles (+400 against S) and C
  68,025 (+184), both failing G4.** A register has no 7-byte word, so it was changed to eight 8-byte chunks after
  seeing this: B 67,745 and C 67,625. The 7-byte versions would pass G2 and G3 (B: 938.9 nJ, 80,640 bits).
- **Line-buffer fill in the order of first use.** Filling in raster order cost 54 cycles per depthwise layer (the first window needs word 60).
- **Smallest hazard-free line buffers:** 120 words of 56 bits for A and B; **150 for C** (3 rows), and C deadlocks below that with
  this wave order. Larger buffers change nothing.
- **B needs no spare row once it has the line buffer:** a 25-row map is hazard-free with the same cycles (storage 77,744 bits).
  The study keeps the 26-row map as designed.
- **B and C need two interlocks** that B0 does not: a gather waits until its source word is written (B stalls 24 times),
  and a vector waits until the words it overwrites have been read. Without the first, B is wrong (§3).
- **Side result of S's second lane** [model]: the worst hop falls from 97,583 to about 80,072 cycles, under the
  81,920 of a 4.096-MHz clock. That would lower the lowest real-time clock from 5.12 to 4.10 MHz with 2.3 % margin;
  this is not built or measured.

## 6. Sensitivity

Network energy against S, in all 18 variants of `ENERGY_ESTIMATE.md` §8, with added control as frozen and doubled:

- **B passes in 17 of 18** (0.808–0.869; doubled 0.827–0.882). It fails with the Colonnade-style array (0.934), where the
  array is two thirds of the energy.
- **C passes in 17 of 18**, with the same failure (0.927); with doubled control it also fails at logic activity 0.2 (0.905).
- **A fails in most** (0.886–0.960).

Full table: `model/streamfuse_energy.log`.

## 7. Decision and next steps

**Since 2026-11-08 B is built in one block (B1, `STREAMFUSE_B1.md`), and the RTL equals this model exactly**: 67,817
cycles, every segment and every access count. One block alone saves 2.4 % of the network energy [E] and nothing at
40 ms; the gain of the table above needs all four blocks and B's one map.

The frozen rule allows **RTL for B in one block (depthwise 1 → pointwise 1), behind a parameter, if the student decides
to go ahead.** Points for that decision:
- B's gain is real but modest at system level: 7 % of total power at 40 ms beyond S, all of it estimated.
- S (sign gating and a second lane) gives 8.6 % at almost no design risk, and probably a lower clock. It is the obvious first build if any is made.
- B needs new sequencer control (interleaved layers, two interlocks, fusion vectors, line-buffer fill), estimated
  here at 25 % of the network sequencer. The model cannot verify that share; an RTL prototype would.
- The novelty is in the evaluation, not the mechanism (`STREAMFUSE_NOVELTY.md`).

## 8. Reproduce

```
cd hopcim/model
~/venvs/planenest/bin/python streamfuse_sched.py all            # cycles, self-test, 4,890-clip replay, ~7 min
../../recim-rv/accelforge_pilot/.venv312/bin/python energy_tool_run.py   # CACTI, adds the new buffer sizes
python3 energy_est.py streamfuse                                # V3, G2, variants: streamfuse_energy.log
~/venvs/defines/bin/python defines_run/run_defines.py           # V4 (DeFiNES, ~/tools/energy/DeFiNES @ 7097d60)
```
Results: `model/streamfuse_results.json`, `streamfuse_run.log`, `streamfuse_energy.json`, `streamfuse_energy.log`,
`streamfuse_defines.json`.

## 9. Limits

- A model, not hardware: the added control logic is an assumed share, the interlocks and fusion vectors are not built,
  and FPGA resources are not measured.
- Energy is the same estimate as `ENERGY_ESTIMATE.md` [E], with its limits; CACTI is weak for buffers as small as 120 words.
- The pass order of B and C is one choice (B: depthwise of p + 1 before pointwise of p; C: waves of rows). Others may do better.
- Sign gating's 50.5 % is measured on the GSC test clips; a quiet room may differ.
