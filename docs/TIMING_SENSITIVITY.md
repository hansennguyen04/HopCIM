# CIM timing sensitivity

Status (2026-11-07): done in simulation, with macro assumptions taken from the published tools' validation data. **The
timing of no real macro has been confirmed here**: the figures from Chih et al. and Colonnade still need checking against the papers (§7).

## 1. Question and answer

Every cycle count in this project rests on an assumed macro: a pass of b input bits has an issue interval D = max(8, b) and
a result latency L = 5 (`ENGINE_SPEC.md`). How much of the result depends on that?

**The lowest real-time clock stays at 5.12 MHz for every assumption with D ≤ 9, including a macro on its own fast clock.** A faster macro
gains almost nothing, because the macro is not what limits this design: requantisation and gathering are. A slower macro does
matter, about linearly once D exceeds 9.

| Macro assumption | D | L | Mean hop with the network | Worst hop | Lowest clock | Below it |
|---|---:|---:|---:|---:|---:|---:|
| Own fast clock: 9 bits in under one system cycle, 2-cycle synchroniser | 1 | 2 | 92,572 | 95,044 | **5.12 MHz** | overruns |
| 2 bits per cycle, no adder-tree pipeline | 5 | 2 | 93,168 | 96,096 | **5.12 MHz** | overruns |
| 1 bit per cycle, no adder-tree pipeline | 9 | 2 | 93,790 | 97,250 | **5.12 MHz** | overruns |
| **HopCIM's assumption** | 9 | 5 | 94,009 | 97,583 | **5.12 MHz** | overruns |
| 1 bit per cycle, pipelined adder tree (log2 64 + 2 stages) | 9 | 8 | 94,228 | 97,916 | **5.12 MHz** | overruns |
| 1 bit per cycle, bit-pipelined columns (weight + log2 64 + 2 stages) | 9 | 16 | 94,811 | 98,804 | **5.12 MHz** | overruns |
| 2 system cycles per bit, or a tile split into 2 time steps | 18 | 5 | 145,264 | 149,903 | 8.19 MHz | overruns |
| A tile split into 4 time steps | 36 | 5 | 282,705 | 289,473 | 15.36 MHz | overruns |
| Bit-serial columns without pipelining: 17 cycles per bit | 153 | 2 | 1,175,852 | 1,196,345 | 60.42 MHz | overruns |

D is for a network pass, b = 9. Each row comes from the complete live-path bench (`tb_cim_mic`: 3 s of speech as PDM, the real
front-end, window path and sequencer), run at that clock and at one divider below. At the lowest clock all 101 decisions are
bit-exact with no overrun; one divider below, the bench fails with overruns.

## 2. Method

`tb/cim_engine_tm.sv` replaces `rtl/cim_engine.sv` at compile time, in simulation only: same module name and ports, the same results,
with the timing from run-time arguments. **The synthesised RTL is not changed.** It can go below the RTL's 8-cycle floor, which
exists only because the FPGA emulation computes 8 rows per cycle. It samples its input at the end of the interval, as a bit-serial
macro reads it, and counts every write to that bank in the meantime as a hazard.

| Validation at the RTL's own timing | Result |
|---|---|
| `tb_cim_engine`, whose checks include the exact timing contract | log identical to the RTL's |
| `tb_cim_fe`, `tb_cim_seq`, `tb_cim_stream` (`cim_fe`) | logs identical, cycle counts included |
| `tb_cim_mic` at divider 5 | log identical; worst hop 97,583 gives 5.12 MHz, the value measured on the board |

Outside that point the model is checked by bit-exactness of results, not against an RTL of the same timing.

## 3. Sources of the assumptions

| Macro, as the tool's files describe it | Bits per cycle | Rows × int8 columns, per bank | Clock | Adder pipeline |
|---|---:|---:|---:|---|
| ISSCC 2022 15.5 (ZigZag-IMC validation) | 2 (Booth) | 32 × 6 | 195 MHz | no |
| ISSCC 2022 11.7 (ZigZag-IMC) | 1 | 32 × 1, 16 cells per group | 333 MHz | yes |
| ISSCC 2023 7.2 (ZigZag-IMC) | 2 | 128 × 8 | 182 MHz | no |
| ISSCC 2023 16.3 (ZigZag-IMC) | 1 | 128 × 8 | 400 MHz | no |
| Colonnade, JSSC 2021 (AccelForge's model) | 1 | 128 rows, 16 registers per column | from a critical path of 0.4 ns per adder | bit-pipelined |
| HopCIM's macro in ZigZag-IMC itself (`ENERGY_ESTIMATE.md` §11) | 1 | 64 × 16 | 283 MHz, 3.53 ns at 0.9 V | no |

- **D** follows from the bits per cycle when the macro shares the system clock. At those clocks, a 9-bit pass takes 32 ns on HopCIM's macro,
  half of one 15.36-MHz cycle, so a macro with its own clock reaches D = 1.
- **L is inferred** from the pipeline descriptions, not quoted from a paper.
- Macros narrower than a tile (8 columns or 32 rows) need 2 in parallel, or 2 time steps; 8 columns and 32 rows together need 4. Parallel use
  costs area and no time; time steps are the D = 18 and D = 36 rows. The cost of merging results is not modelled.

## 4. What limits the network

| D (L = 5) | 1 | 3 | 5 | 9 | 10 | 12 | 14 | 16 | 18 | 20 | 24 | 27 | 32 | 36 | 48 | 72 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cycles per inference, thousands | 85.27 | 85.29 | 85.31 | **85.35** | 90.36 | 100.37 | 110.38 | 120.39 | 135.40 | 150.40 | 180.40 | 202.91 | 240.41 | 270.41 | 360.43 | 540.45 |

| L (D = 9) | 1 | 2 | 3 | 5 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cycles per inference, thousands | 85.31 | 85.32 | 85.33 | **85.35** | 85.38 | 85.42 | 85.46 | 85.54 | 93.11 | 113.24 | 139.57 | 259.76 |

All of it fits one line, within 1.39 % over these 40 timings and within 0.1 % below L = 32:

```
cycles ≈ 2,500 max(D, 16, (D + L) / 4) + 5,000 max(D, 9, (D + L) / 4) + 10 (L + 30)
```

- **2,500 passes** (conv1, pointwise) produce 16 results each, and requantisation takes one result per cycle: 16 cycles per pass.
- **5,000 depthwise passes** need 9 cycles to gather their input vector.
- **At most 4 passes are in flight** (the sequencer's tag queue), which starts to limit at L ≈ 27 for D = 9.
- **10 layers** each add about 30 cycles plus the latency.

So the macro's interval only binds above 9 cycles, and fully above 16.

## 5. A design limit this found

The sequencer releases an input bank a fixed 7 cycles after its last pass starts. That constant fits today's timing and no other:

| D, RTL's constant | Hazards | Result |
|---|---:|---|
| ≤ 10 | 0 | bit-exact |
| 11 | 1,360,488 | bit-exact in this bench, but writes land inside the read window |
| 12 | 2,720,976 | **78,492 errors** |
| 18 | 10,883,904 | **98,712 errors** |
| 18, 36, 153 with the hold scaled to D − 2 | 0 | bit-exact |

Scaling the hold is one constant and a wider counter, and it costs no cycles, because a slower macro also spaces the passes further apart.
The sweeps above apply it to a simulation-only copy of `cim_seq.sv` (`sim/run_timing.sh`), so the rows with D > 10 describe the design with that
change. A macro needing more than 10 cycles per pass would require it in the RTL, along with wider `gap`, `dm8`, `wait_cnt` and `pace` counters,
and a longer CPU write lock in the endpoint. Nothing at today's timing is affected.

## 6. What could lower the clock (model, not built)

Since requantisation and gathering bind, these are the levers rather than the macro, for H3 and for any later extension. With R results per
cycle, chunks gathered G times faster, D = 1 and the measured 9.8k cycles of front-end and window path:

| R, G | Network, cycles | Worst hop | Lowest clock |
|---|---:|---:|---:|
| 1, 1 (built) | 85.4k | 95.0k, measured | 5.12 MHz |
| 2, 1 | 65.4k | 75.2k | 4.10 MHz |
| 2, 2 | 42.9k | 52.7k | 3.07 MHz |
| 4, 4 | 21.6k | 31.4k | 2.05 MHz |
| 8, 9 | 10.4k | 20.2k | 2.05 MHz (1.02 MHz leaves about 300 cycles, and see below) |

**Built since (2026-11-08):** R = 2, with sign-gated requantisation, is the baseline S (`BASELINE_S.md`): network
67,841 cycles, worst hop 80,072 (measured with the hop monitor), lowest real-time clock 4.096 MHz in RTL. The one-line
estimate of the second row was 3.6 % and 6.1 % low on these two counts; the clock is the same.

- **H3's target (below 1 MHz) looks structurally out of reach** in this architecture, whatever the macro: the microphone's PDM clock is 1.024 MHz and is derived
  from the system clock by a divider of at least 2 (a design with one clock).
- A network spread over several hops at k > 1 would lower the requirement further: on average 2.6 and 1.5 dividers at k = 2 and 4. The hop controller
  does not allow it today (a decision hop must finish within its hop), and the engine would have to be shared with the front-end.

**Energy [E].** The array's energy per pass does not depend on D. The sequencer's logic does scale with its busy time: at the estimate's 2.3 pJ per cycle, D = 18
adds at most 115 nJ per inference (8 %), and D = 36 at most 424 nJ (31 %). D below 9 saves nothing.

## 6a. The baseline S (2026-11-08)

The plan's sweep D = max(8, αb + β) with L = 3, 5, 8, 16, run on S (`BASELINE_S.md`) with the same behavioural
engine (`timing_sweep.py seq_s`, `hop_s`; `run_timing.sh` with `S=1`). For every pass here αb ≥ 8, so the model's
max(8, αb) + β is the plan's formula exactly. α = 1.5 has no integer form; ⌈b/2⌉·3 (15 for b = 9) stands in.

**Validation:** at the RTL's timing the S bench with the behavioural engine gives a log identical to the RTL's
(67,841 cycles, 0 errors), and the live path the worst hop measured on the RTL, 80,072.

| Network passes (b = 9) | D | L = 3 | L = 5 | L = 8 | L = 16 | Engine busy (L = 5) |
|---|---:|---:|---:|---:|---:|---:|
| own fast clock (L = 2) | 1 | | 65,236 (L = 2) | | | 11.5 % |
| **max(8, b), HopCIM's assumption** | 9 | 67,821 | **67,841** | 67,871 | 67,951 | 99.5 % |
| max(8, b + 4) | 13 | 97,825 | 97,845 | 97,875 | 97,955 | 99.7 % |
| ≈ max(8, 1.5b) | 15 | | 112,847 | | | 99.7 % |
| max(8, b + 8) | 17 | 127,829 | 127,849 | 127,879 | 127,959 | 99.7 % |
| **max(8, 2b)** | 18 | 135,330 | **135,350** | 135,380 | 135,460 | 99.8 % |
| ≈ max(8, 1.5b + 4) | 19 | | 142,851 | | | 99.8 % |
| max(8, 2b + 4) | 22 | 165,334 | 165,354 | 165,384 | 165,464 | 99.8 % |
| ≈ max(8, 1.5b + 8) | 23 | | 172,855 | | | 99.8 % |
| **max(8, 2b + 8)** | 26 | 195,338 | **195,358** | 195,388 | 195,468 | 99.8 % |

Cycles per inference; engine busy = 7,501 passes × D / cycles. All 28 points are bit-exact (layers, synthetic sets,
logits) with no bank hazard. One line fits them within 0.13 %:

```
S: cycles ≈ 2,500 max(D, 8, (D + L) / 4) + 5,000 max(D, 9, (D + L) / 4) + 10 (L + 30)
```

| Live path (`tb_cim_mic`, 101 decisions) | D | L | Worst hop | Lowest real-time clock | One divider below |
|---|---:|---:|---:|---:|---|
| own fast clock | 1 | 2 | 75,038 | **4.096 MHz** | 101 overruns |
| **max(8, b)** | 9 | 5 | 80,072 | **4.096 MHz** | 101 overruns |
| **max(8, 2b)** | 18 | 5 | 149,857 | **8.192 MHz** | 101 overruns |
| **max(8, 2b + 8)** | 26 | 5 | 211,065 | **11.264 MHz** | 76 overruns |

At each lowest clock all 101 decisions are bit-exact with no overrun.

What this adds to §4:

1. **S moved the bottleneck onto the macro.** With two requantisation lanes a pointwise pass needs 8 cycles to drain,
   so at D = 9 the macro's interval binds every pass: the engine is busy 99.5 % of the network (R6b: 7,501 × 9 /
   85,352 = 79.1 %). Every cycle of D above 9 costs about 7,500 cycles per inference, with no slack in between.
2. **S gains nothing over R6b once D ≥ 16.** At D = 18, S takes 135,350 cycles against R6b's 135,40x (§4), and the
   lowest clock is 8.19 MHz for both; at D = 26 R6b's line gives the same 195.4k. S's 20 % and its 4.096 MHz depend on
   a macro that issues a 9-bit pass in about 9 cycles or less.
3. **A faster macro does not lower the clock.** D = 1 saves 3.8 % of the network (the depthwise gather still needs 9
   cycles per pass) and the lowest clock stays 4.096 MHz: 75,038 cycles do not fit the 61,440 of 3.072 MHz.
4. **L hardly matters:** 3 → 16 adds 130 cycles (0.2 %) at every D, since at most 4 passes are in flight.
5. The plan's very conservative point, max(8, 2b + 8), gives 11.26 MHz, still real time on this FPGA (its clock is
   15.36 MHz), but 2.75 × the lowest clock of HopCIM's assumption.

The thesis wording stays: a parameterised cycle-faithful CIM architectural model, not the timing of a fabricated macro.

## 7. Limits

- The macro figures come from the tools' files. **Please check bits per cycle, pipeline stages, clock and array size against Chih et al. (ISSCC 2021) and
  Colonnade (JSSC 2021):** public sources give neither paper's cycle counts, and this study could not read them.
- L is inferred, the synchroniser is a 2-cycle assumption, and tile splitting is modelled only as a longer D.
- One point of the sweep did not finish: D = 144 stopped at the bench's 400-ms watchdog, after showing no data errors. The live-path run at D = 153 did complete.
- The sweep used `cim_fe`; B3 was not swept.
- It says nothing about whether a macro at these timings can be built at low voltage. Energy stays an estimate.

## 8. Reproduce

```
cd hopcim
python3 model/timing_sweep.py seq      # cycles and exactness over the timings, about 5 min on 8 cores
python3 model/timing_sweep.py hop      # the live path per assumption, about 15 min on 16 cores
JOBS=6 python3 model/timing_sweep.py seq_s   # S over the plan's grid, about 2.5 min (§6a)
JOBS=4 python3 model/timing_sweep.py hop_s   # S's live path at 4 points, about 40 min
python3 model/timing_sweep.py report   # results in model/timing_results.json
```
