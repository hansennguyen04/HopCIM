# The baseline S: sign-gated requantisation, a second requantisation lane, no L8 write

**2026-11-25: S is the comparison baseline.** The default HopCIM design is S with the digital depthwise unit
(`DWU_RESULTS.md`); `DESIGNS.md` has both side by side. Everything below is S and unchanged.

Status (2026-11-08): **built in RTL behind a parameter, verified in simulation and on the board at 15.36 and 4.096 MHz (§8).** S is the
mandatory baseline for every later optimisation (`HopCIM_Specific_Next_Steps_and_Professor_Advice.md` §5, §17). Its
three changes were found and priced by the StreamFuse model study (`STREAMFUSE_STUDY.md`), before any RTL.
Energy figures are estimates [E] (`ENERGY_ESTIMATE.md`); everything else is measured in simulation or on the board.

| | R6b (today) | **S** | Change |
|---|---:|---:|---:|
| Network cycles per inference (go → done, sequencer bench) | 85,352 | **67,841** | −20.5 % |
| Worst hop of the live path (hop monitor) | 97,583 | **80,072** | −17.9 % |
| Lowest real-time clock with a 1.024-MHz PDM clock (RTL and board) | 5.12 MHz | **4.096 MHz** | −20 % |
| Exactness: 4,890 test clips, logits and class (RTL) | 0 errors | **0 errors** | |
| Network energy per inference [E], from the RTL's counters | 1,389.3 nJ | **1,120.8 nJ** | −19.3 % |
| Total power at 20 / 40 / 80 ms [E] | 110.3 / 75.3 / 57.9 µW | **97.0 / 68.8 / 54.7 µW** | −8.6 % at 40 ms |
| Endpoint, out of context (Vivado 2024.1, xc7a100t) | 7,635 LUT, 5,819 FF, 145 DSP | 8,116 LUT, 5,977 FF, 147 DSP | +481 LUT, +158 FF, +2 DSP, +0.5 BRAM |
| Worst setup slack at 15 MHz (endpoint, out of context) | +46.9 ns | +45.8 ns | |

The schedule model of the study predicted S exactly: all ten layers' cycles, the access counts and the worst hop (§5).

## 1. What changed

`rtl/cim_seq.sv` has three parameters, off by default (then it is R6b, cycle for cycle); `cim_endpoint`'s `SEQ_S`
sets all three, and FuseSoC's `--flag use_cim_s` sets `SEQ_S` on the FPGA and in the SoC testharness.

- **`RQ_SG`, sign-gated requantisation.** In L0..L8 the output zero point is −128. When `acc + bias ≤ 0` the result
  is exactly −128 (§2), so the multiplier and the rounding shift hold their registers (clock enables off) and the
  last stage writes −128. The gate needs the activation zero point to be −128 and is never used in L9 (the logits
  have their own zero point). On the 4,890 test clips 50.5 % of all requantisations are gated; on the live-path
  bench's speech stream, 50.2 %.
- **`RQ_2L`, a second requantisation lane.** Results drain two per cycle: lane 0 takes the even channel of each pair,
  lane 1 the odd one. The parameter memory is split into an even and an odd bank (bank k holds channels with
  c mod 2 = k), so it needs no extra capacity. Two results of a pair have consecutive channels, so they always go to
  different ACT byte lanes and different pool-sum banks, and the first maximum of the logits is kept (lane 0 holds
  the lower channel). Passes are paced by max(9, ceil(outputs / 2)) instead of max(9, outputs).
- **`NO_L8A`, no L8 write to ACT.** L8's only consumer is the pool sums, so in a full run L8 is not written to ACT.
  A run that stops after L8 still writes it, so that it can be read back (tests, the debug app).

## 2. Why the sign gate is exact

TFLite's requantisation is `y = clamp(rdbp(srdhm(x, qm), e) + zout, −128, 127)` with `x = acc + bias`,
`qm ∈ [2^30, 2^31)` and `e ≥ 0` (`SEQUENCER.md` §3). For `x ≤ 0`:

- `srdhm(x, qm) = (x·qm + 2^30) >> 31`: `x·qm ≤ 0`, so the sum is at most 2^30 and the floor is at most 0;
- `rdbp(r, e)` for `r ≤ 0`: `r = 0` gives `2^(e−1) >> e = 0`; `r ≤ −1` gives `(r + 2^(e−1) − 1) >> e ≤ 0`; `e = 0` gives `r`;
- so `rdbp(…) + zout ≤ −128`, and the clamp gives −128.

Small positive x can also give −128, but the gate does not use that: it needs only the sign and a zero test.

## 3. Verification

| Check | Result |
|---|---|
| Lint, `verilator -Wall`, both `SEQ_S` and both front-ends | clean |
| R6b unchanged (`SEQ_S = 0`): sequencer bench, every layer, 20 clips, ring, interplay, both synthetic sets | 0 errors, 85,352 cycles, as before |
| S, sequencer bench (cim_fe and B3): every output of L0..L8 of clip 0, 20 clips' logits, FEAT ring, interplay | 0 errors, 67,841 cycles |
| S, synthetic set 1 (zero point 5: the gate must stay off; shifts 0..15, power-of-two multipliers) | 0 mismatches in every layer and the logits |
| S, synthetic set 2 (zero point −128, shifts 0..15, power-of-two multipliers, and acc + bias = −1, 0, +1 at one position of every channel of L0..L8: the gate's boundary, 21 to 1,007 cases of each per layer) | 0 mismatches in every layer and the logits; R6b likewise |
| S, all 4,890 test clips through the RTL: logits and class against the golden model | **0 errors**; 91.80 % (§3.1) |
| S, window path and network streaming (`tb_cim_stream`, cim_fe, B3, edge vectors) | pass |
| S, live audio path end to end from PDM bits (`tb_cim_mic`): 101 decisions' logits and class, no overrun, at 15.36, 5.12 and 4.096 MHz; k = 1, 2, 4 | pass |
| S, Verilator SoC (X-HEEP with `--flag use_cim_s`, `sim/run_soc.sh` with `SIMVAR=cim_s`): `hopcim_seq` and `hopcim_stream` | 67,909 cycles go → interrupt, 60 runs bit-exact; 100 hops, 51 decisions exact; **every figure equals the board's** (§8) |
| R6b at 4.096 MHz (`tb_cim_mic`, divider 4) | **overruns from the first decision** (expected; it is in the regression as a bench that must fail) |
| Assertions: 4 new for S, 2 extended to two lanes (§3.2) | none failing |
| Mutation test: 11 broken S designs (§3.3), with the 16 earlier ones | **27 of 27 caught**; the 2 equivalent changes change nothing |

### 3.1 All test clips in RTL

`sim/run_seq_allclips.sh [s]`: the 4,890 GSC test clips' features (integer CIM-MFCC, `model/gen_seq_allclips.py`)
go through the endpoint in Verilator one by one, and every clip's 12 logits and its class are compared with
`dscnn_int` (= TFLite's reference kernels). The vectors are pinned by hash in the log.

**S: 4,890 clips, 0 with a logit or class error, 67,841 cycles on every clip** (`sim/logs/seq_allclips_s.log`). The RTL's
classes are therefore the golden ones: 4,489 of 4,890 correct, **91.80 %**, the accuracy of claim A2. The cycle count
does not depend on the data.

### 3.2 Assertions (`tb/cim_assertions.sv`, module `sva_seq_s`)

- `sign_gate_exact`: for every gated result it recomputes the whole requantisation from x, qm, e and the zero point
  and requires −128, and requires a layer other than L9. It checks the claim the gate rests on, not the gate's own
  condition.
- `pair_writes_distinct_lanes`: when both lanes write ACT in one cycle, they write different byte lanes of one position.
- `lane_channel_parity`: lane 0 only holds even channels, lane 1 only odd ones.
- `l8_act_only_when_stopped_after_l8`: with `NO_L8A`, L8 reaches ACT only in a run that stops after L8.
- `results_drained_before_pop` allows two outstanding results with two lanes; `requant_keeps_count` counts both lanes.

### 3.3 Mutation test

`python3 model/verif_mutants.py 4` (`model/verif_mutants.json`): each mutant breaks S in one place, and the bench must
then trip the named assertion, or (marked *bench*) fail its own comparison with the golden model. All 11 are caught,
and the 16 mutants of the freeze still are (27 of 27). "L8 written to ACT in full runs" is caught only by its
assertion: it costs energy but changes no result, so no bench can see it.

| Mutant | Caught by |
|---|---|
| sign gate too wide (x < 64) | `sign_gate_exact` |
| sign gate ignores the zero point | `sign_gate_exact` |
| sign gate also in L9 | `sign_gate_exact` |
| a gated result is −127 instead of −128 | bench |
| lane 1 reads the even parameter bank | bench |
| lanes take the wrong parity | `lane_channel_parity` |
| the second result of a pair goes to the first one's byte lane | `pair_writes_distinct_lanes` |
| lane 1's ACT write dropped at one position | bench |
| L8 written to ACT in full runs | `l8_act_only_when_stopped_after_l8` |
| L8 not written even when the run stops after it | bench |
| two results drained, three counted | bench |

One change is **equivalent at today's timing**: removing the drain pacing of the two lanes. A pass drains in at most
8 cycles with two lanes, and the engine spaces passes 9 cycles apart, so the pacing never binds. It is kept (it
matters if the engine gets faster, `TIMING_SENSITIVITY.md`), and the mutation test checks that removing it changes nothing.

## 4. Cycles

| Layer | R6b | S | Model (study) |
|---|---:|---:|---:|
| L0 conv1 | 8,034 | 4,533 | 4,533 |
| L1, L3, L5, L7 depthwise, each | 11,273 | 11,273 | 11,273 |
| L2, L4, L6, L8 pointwise, each | 8,032 | 4,531 | 4,531 |
| pool + L9 | 98 | 92 | 92 |
| **Total** | **85,352** | **67,841** | **67,841** |

The depthwise layers do not move: they are engine-bound (one pass every 9 cycles, 7 results each), as
`TIMING_SENSITIVITY.md` found. conv1 and the pointwise layers were requantisation-bound (16 results per pass, one per
cycle) and become engine-bound.

## 5. The lowest real-time clock

The worst hop of the live path (hop monitor, `HOPMON=1 sim/run_mic_tb.sh cim 4 1 s`) is a hop with all 49 feature
rows recomputed and the network: **80,072 cycles** with S against 97,583 with R6b. The model's estimate was the same
80,072 (= 97,583 − 17,511). A 20-ms hop has 81,920 cycles at 4.096 MHz, a margin of 1,848 cycles (2.3 %); 4.096 MHz is
the lowest system clock that gives an exact 1.024-MHz PDM clock below 5.12 MHz (divider 4).

In simulation the whole live path at divider 4 is bit-exact with no overrun (101 decisions; k = 1, 2, 4), and R6b at
divider 4 overruns from its first decision.

## 6. Cost

Vivado 2024.1 out of context, xc7a100tcsg324-1, 15 MHz (`syn/ooc.tcl`):

| | LUT | FF | BRAM tiles | DSP | WNS |
|---|---:|---:|---:|---:|---:|
| cim_seq, R6b (current source, with the window path) | 2,535 | 1,982 | 11.5 | 9 | +50.9 ns |
| cim_seq, S | 2,938 | 2,140 | 12 | 11 | +50.0 ns |
| cim_endpoint, R6b (claim D2) | 7,635 | 5,819 | 46 | 145 | +46.9 ns |
| cim_endpoint, S | 8,116 | 5,977 | 46.5 | 147 | +45.8 ns |

The parameter banks stay in block RAM (512 entries each; +0.5 tile from the split), and no memory falls back to LUT RAM.

Whole SoC as placed on the board (X-HEEP with the endpoint and the live audio path):

| Bitstream | LUT | FF | BRAM tiles | DSP |
|---|---:|---:|---:|---:|
| `xheep_nexys_bscan_cim128k.bit`, R6b, 15.36 MHz | 42,306 | 40,221 | 82 | 146 |
| `xheep_nexys_bscan_cim128ks.bit`, S, 15.36 MHz | 42,758 (+1.1 %) | 40,363 | 82.5 | 148 |
| `xheep_nexys_bscan_cim128mic4s.bit`, S, 4.096 MHz | 42,761 | 40,361 | 82.5 | 148 |

## 7. Energy from the RTL's counters [E]

`sim/run_energy_counts.sh cim 5 1 s` binds the activity counters into the live-path bench with S, now including the
second lane, the gated results and the multiplies executed. Per inference (101 decisions): 72,012 requantisations,
36,176 gated (50.24 %), 35,836 multiplies, 64,000 ACT byte writes (no L8), the same passes and gathers as R6b, and
17,511 fewer sequencer cycles, which equals the sequencer bench's 85,352 − 67,841. `python3 model/energy_est.py s_rtl`
prices these with the study's costing rules (`model/energy_s_rtl.log`):

| [E] | R6b | S, RTL counts | S, model |
|---|---:|---:|---:|
| Network energy per inference | 1,389.3 nJ | **1,120.8 nJ** | 1,119.8 nJ |
| Total power at 20 / 40 / 80 ms | 110.3 / 75.3 / 57.9 µW | **97.0 / 68.8 / 54.7 µW** | 96.9 / 68.8 / 54.7 µW |

The RTL's figure is 0.09 % above the model's, only because this stream has fewer gated results (50.24 %) than the
test-set mean (50.46 %). S saves 19.3 % per inference in the baseline variant, 8.9–22.0 % over the 18 variants of
`ENERGY_ESTIMATE.md` §8 (least with a Colonnade-style array). With the measured size (+403 LUT, +158 FF) charged as
logic that switches every cycle, on top of the per-operation cost of the second lane, which double-counts the lane,
the upper bound is 1,138.0 nJ (−18.1 %) and 69.2 µW at 40 ms. At 4.096 MHz the always-clocked part of the audio path
costs a little less: 67.6 µW at 40 ms.

## 8. Board

**At 15.36 MHz** (`xheep_nexys_bscan_cim128ks.bit`: `--flag use_cim_s` with the live audio path, FuseSoC build root
`build_micks/core`; system clock setup slack +15.31 ns, hold +0.229 ns; the only negative slack is X-HEEP's stock
unused SPI-slave clock crossing, as in every Nexys build):

| Run (2026-11-08) | Result |
|---|---|
| `hopcim_seq` (firmware for 15.36 MHz), 20 clips × 3 ways | **67,909 cycles go → interrupt** on every run (the RTL's 67,841 plus the fixed 68-cycle offset that R6b also shows, 85,420 against 85,352); **60 runs bit-exact**, 20/20 = golden = label; CPU-active 9,677 cycles (whole window) or 459 (streaming step) per decision, as with R6b |
| `hopcim_stream`, window path in hardware | **100 hops, 51 decisions, 0 mismatches**; wall per decision hop 80,001 cycles (new frame) or 84,513 (all 49 frames) |
| `board_khop.sh`, live path, 40 s per k | k = 1, 2, 4: **2,000 / 1,000 / 500 decisions, overrun 0**; CPU-active 263 / 263 / 266 cycles per decision; the same as R6b's run of 2026-11-05, line for line |
| `board_khop.sh`, real microphone audio (1.5 s) against the golden model | **25 of 25, 13 of 13, 7 of 7 decisions exact** at k = 1, 2, 4 |

**At 4.096 MHz** (`xheep_nexys_bscan_cim128mic4s.bit`: `--flag use_cim_s --flag use_cim_mic4` with `use_cim_mic`,
build root `build_mic4s/core`; firmware with `REFERENCE_CLOCK_Hz = 4096000`):

- **Clock.** No MMCM output divider makes 4.096 MHz from the board's 100 MHz: only CLKOUT0 divides fractionally, and
  it would need 156.25 (at most 128); the integer outputs and the CLKOUT4 cascade need a multiplier the MMCM does not
  have. So the MMCM makes exactly 8.192 MHz (VCO 100 MHz × 32 / 5 = 640 MHz, divided by 78.125, the wizard in
  override mode) without an output buffer, and a BUFGCE passes every second rising edge (`CIM_MIC_4M096` in the
  FPGA wrapper). Two first attempts failed and are worth knowing: the wizard ignores manual MMCM settings unless
  override mode is on (it chose 8.191994 MHz), and an extra BUFG before the BUFGCE makes X-HEEP's own BUFGMUX
  cascades on the system clock one level too deep to place. Timing is analysed at 8.192 MHz, stricter than the real
  clock: setup slack +15.68 ns, hold +0.259 ns (the SPI-slave crossing again excepted).
- **Live path, 130 s** (`hopcim_live`): **6,000 decisions in 120 s, 500 per 10 s, overrun 0**, 263 CPU-active cycles
  per decision (max 289). The microphone works with the 50 % PDM clock (RMS 35). The UART at 9,600 baud is readable,
  so the system clock is within a few per cent of 4.096 MHz, and the 1.024-MHz PDM clock is its exact quarter.
- **Real microphone audio** (`hopcim_micrec`, 1.5 s, `model/micrec_4m096_check.log`): **25 of 25 decisions whose
  window lies in the recording equal the golden model**; the recording's spectrum is the room's (94 % below 300 Hz,
  mains harmonics), as in the 5.12-MHz run.

So the plan's P4 holds on the board: the minimum real-time clock moves from 5.12 MHz to **4.096 MHz**, measured,
with 2.3 % margin in the worst hop.

## 9. Reproduce

```
cd hopcim
python model/gen_seq_vectors.py; python model/gen_seq_allclips.py     # vectors (~/venvs/planenest/bin/python)
sim/run_seq_tb.sh cim s; sim/run_seq_tb.sh ded s                      # sequencer bench with S
sim/run_seq_allclips.sh s                                             # 4,890 clips in RTL, about 15 min
sim/run_mic_tb.sh cim 4 1 s                                           # live path at 4.096 MHz; HOPMON=1 for the worst hop
sim/run_all.sh                                                        # the whole regression, R6b and S
python3 model/verif_mutants.py 4                                      # mutation test
sim/run_energy_counts.sh cim 5 1 s; python3 model/energy_est.py s_rtl # energy from the RTL counters
vivado -mode batch -source syn/ooc.tcl -tclargs cim_endpoint FE_DED=0 PDM_CLKDIV=15 SEQ_S=1
```
FPGA: `--flag use_cim_s` (and `--flag use_cim_mic4` with `use_cim_mic` for 4.096 MHz), `INTEGRATION.md`.

## 10. Limits

- Energy is the estimate of `ENERGY_ESTIMATE.md` [E]; the gated multiplies are counted, not measured in silicon.
- The gate's share depends on the audio: 50.5 % on the GSC test clips, 50.2 % on the bench's stream; a quiet room may differ.
- 4.096 MHz leaves 2.3 % margin at today's macro timing (D = 9); a slower macro raises the lowest clock as in
  `TIMING_SENSITIVITY.md`.
