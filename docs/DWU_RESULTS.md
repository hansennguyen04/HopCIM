# Digital depthwise unit: the RTL against its frozen gates

Built 2026-11-22, board runs 2026-11-23. Gates: [`DWU_GATES.md`](DWU_GATES.md), frozen at commit `95b8cf8` before any RTL, SHA-256
`0e060c59…19e0e461` (`DWU_GATES.sha256`), unchanged. Predictions: `model/dwu.py` and `model/dwu_prediction.json`,
frozen with the gates, whose default output is still byte-identical.

Power is a model estimate [E] from the RTL's own counters (`ENERGY_ESTIMATE.md`), not an FPGA measurement. The
FPGA engine keeps its 128 tile slots; the energy model's CIM has the 29 tiles the network still uses.

## Verdict

| Gate | Result | |
|---|---|---|
| R1 exact | D1 and D4: every bench exact, 4,890 clips with 0 errors, live path exact, Vivado's netlist exact, regression 53 of 53, the 3 broken variants caught | pass |
| R2 cycles | D1 67,828 and D4 67,789 cycles, equal to the model in every layer; all 9 counters equal | pass |
| R3 real time | worst hop 80,020 cycles (model 80,020) ≤ 81,920 at 4.096 MHz; no overrun at k = 1, 2, 4 | pass |
| R4 size | 12,414 gate equivalents (Vivado), 12,170 by construction ≤ 15,900 | pass |
| R5 power | **27.96 µW at U** [E] ≤ the swap-in's 28.13; 0.634 × S's 44.11 | pass |
| R6 board | timing met (system clock +16.2 ns setup, +0.23 ns hold); 60 runs exact; live path without overrun at k = 1, 2, 4; microphone logits 45 of 45 equal to the golden model. Cycles **67,856: one below the gate's 67,857**, traced to the test firmware (below) | accepted by the student with the one-cycle difference explained (2026-11-25) |

**What this means:**
- **The four depthwise layers now run off the CIM, exactly, at no time cost.** HopCIM gives the same logits as
  TFLite on every test clip, 52 cycles faster per inference than S (the unit's result is ready 1 cycle after a pass
  instead of 14).
- **At U's rate the system drops from 44.1 to 28.0 µW [E] (−36.6 %), and at 40 ms from 68.8 to 50.3 µW (−26.9 %).**
  The CIM shrinks from 69 tiles in 5 macros to 29 in 2; the unit costs about 12,400 gate equivalents and a 2.5-KiB
  weight store.
- **R1 to R5 pass. R6 misses its cycle check by one cycle by the gate's wording; the student accepted it with the cause
  explained.** The RTL equals the frozen model cycle for cycle and count for count. One synthesis finding (the store's
  memory type) was fixed, and everything rerun on the fixed RTL (below).

## The design as built

`rtl/cim_seq.sv`, option `DWU` (endpoint `SEQ_DWU`, built with S):
- In a depthwise layer on the unit the gather runs group by group (for g: for p).
- Each chunk (one tap, 7 channels) is multiplied in stage G1 by 7 weights and added into the 7 accumulators of the
  chunk's X bank. The engine's X bank is not written, and the pass goes to the unit, not the engine.
- The weights of the current group are in a 63-byte register. When chunk k of a group's last vector is multiplied,
  tap k of the next group is read from the 360 × 56-bit store (two RAMB18) and written the cycle after. When the
  network starts, group 0 of L1 is loaded.
- The CPU writes the store at `0x0_6000` (`cim_load_dwstore` in `sw/cim.h`; apps built with `-DCIM_DWU=4` fill
  it and skip the depthwise tiles).
- X-HEEP: patch 0006 (flag `use_cim_dwu`), build variants `dwu`, `dwu-4`, `sim-dwu` (`xheep/build.sh`).

**Found by synthesis and fixed before any gate was scored:** the first version read the store through an address
register loaded only when loading. Vivado cannot map such a read to block RAM (the read must follow later writes),
so it built the store from 608 distributed-RAM LUTs. The store now has a read register with an enable, the standard
block-RAM pattern, with the same timing; every bench, the 4,890 clips, the live path, the counters and the mutants
were rerun on the fixed RTL. The size gate is unaffected (the store is priced as memory either way).

## R1: exactness

| Check | D1 (L1 on the unit) | D4 (all four) |
|---|---|---|
| `tb_cim_seq`: every output of L0–L8 of clip 0, 20 clips' logits and class, FEAT ring, synthetic sets 1 and 2 | 0 errors | 0 errors |
| The same with the B3 front-end (`ded`) | — | 0 errors |
| All 4,890 GSC test clips (`run_seq_allclips.sh`) | 0 logit or class errors | 0 logit or class errors |
| Live path from PDM (`tb_cim_mic`, 101 decisions) at 15.36 MHz; at 4.096 MHz with k = 1, 2, 4 | — | exact, no overrun |
| Streaming (`tb_cim_stream`) | — | exact |
| Vivado functional netlist of the endpoint (`run_endpoint_nl.sh`, every layer of one clip and one inference) | — | 0 errors, the RTL's cycles in every layer |
| Regression `sim/run_all.sh` (53 benches: the 46 before and the unit's 7) | 53 of 53 pass | |

A first regression run, started after a reboot had killed one mid-build, gave 50 of 53: three benches that the unit
does not touch (endpoint, both front-ends) failed to compile on stale precompiled headers left in their object
folders. With those folders removed, a full rerun passed 53 of 53 (logs `sim/dwu_jobs/`).

In every unit run the bench overwrites the unit's depthwise tiles in the engine with a garbage pattern, so a layer
on the unit is right only if it never uses the engine. The engine got 0 passes on depthwise tiles.

Broken variants (`model/verif_mutants.py`, prefix "DWU:"), each caught by the bench's own comparison:

| Variant | Result |
|---|---|
| The next group's taps loaded during the group's second-to-last vector (p = 123) | caught |
| The two banks' accumulators swapped at the issue | caught |
| After group 9 of a layer, group 0 of the same layer loaded instead of the next | caught |

## R2: cycles and counts

| Layer | S | D1 RTL = model | D4 RTL = model |
|---|---:|---:|---:|
| L0 conv1 | 4,533 | 4,533 | 4,533 |
| L1 depthwise | 11,273 | **11,260** | **11,260** |
| L3, L5, L7 depthwise, each | 11,273 | 11,273 | **11,260** |
| L2, L4, L6, L8 pointwise, each | 4,531 | 4,531 | 4,531 |
| pool + L9 | 92 | 92 | 92 |
| **Total** | **67,841** | **67,828** | **67,789** |

Counters per inference (live path, 101 decisions), RTL = model in every row:

| Unit passes | Chunks | MACs | Store reads | Engine passes | ACT word reads | ACT byte writes | X rows | Requantisations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 45,000 | 288,000 | 369 | 2,501 | 42,960 | 64,000 | 37,064 | 72,012 |

The live path's run cycles fall by 52 per inference against S, as the benches say.

## R4: size

Vivado 2024.1, out of context, xc7a100t at 15 MHz, the endpoint with `SEQ_DWU = 4` against S on the same source:

| | S | S + unit | Added |
|---|---:|---:|---:|
| Logic LUTs | 7,489 | 8,743 | +1,254 |
| Distributed-RAM LUTs | 604 | 604 | 0 |
| Flip-flops | 5,980 | 6,795 | +815 (280 accumulator bits, 504 weight-register bits, 31 control) |
| RAMB18 | 11 | 13 | +2 (the store) |
| DSPs | 147 | 147 | 0 (multipliers in LUTs) |

- At 6 gate equivalents per LUT and per flip-flop (the energy model's rule): **12,414**.
- By construction (multipliers, adders, muxes) with the synthesised flip-flops: 12,170.
- The larger one, 12,414, is used. The study had assumed 7,756 for its 9-multiplier unit without a weight
  register; the gate's prediction for this design was 12,224.
- Timing slack +41.1 ns at 15 MHz.

## R5: power [E]

From the RTL's counters (`sim/run_energy_counts.sh cim 5 1 d4`) against S's, with the study's costing rules:

| | S | S + unit |
|---|---:|---:|
| Energy per inference | 1,120.8 nJ | 1,013.2 nJ |
| … CIM array | 134.2 | 94.9 |
| … tile select | 169.1 | 91.3 |
| … the unit's MACs, store reads, weight register | — | 10.9 + 0.8 + 0.02 |
| … input registers | 3.9 | 1.7 |
| Leakage | 36.33 µW | 20.52 µW |
| … CIM macro logic (5 → 2 macros) | 22.20 | 8.88 |
| … weight cells (69 → 29 tiles) | 5.88 | 2.47 |
| … other SRAM (+ the store) | 4.07 | 4.37 |
| … digital logic (+ 12,414 GE) | 4.18 | 4.80 |
| **Power at U (k = 16)** | **44.11 µW** | **27.96 µW** (0.634 ×) |
| Power at 40 ms | 68.78 µW | 50.28 µW (0.731 ×) |

- Lower than S in all 19 variants: 19.75–147.49 µW, 0.450 × (1 tile per macro) to 0.699 × (20 pW per gate equivalent).
- Against the study's predictions: the swap-in 28.13 µW, the study's DIG 27.72, this design's frozen prediction 27.95.

## R6: board

Bitstream `planenest/hw/bitstreams/xheep_nexys_bscan_cim128kdwu.bit` (`xheep/build.sh <tree> dwu`, 39 min; sha256
`87954077de759372…`): 44,022 LUTs (69.4 %), 41,178 registers, 83.5 block-RAM tiles, 148 DSPs. Timing met: system clock
(15.36 MHz) WNS +16.18 ns, WHS +0.229 ns, no failing endpoint. Firmware in `planenest/hw/`, all built with
`-DREFERENCE_CLOCK_Hz=15360000 -DCIM_DWU=4`: `hopcim_seq_dwu_nexys.elf`, `hopcim_live{,_k2,_k4}_dwu_nexys.elf`,
`hopcim_micrec{,_k2,_k4}_dwu_nexys.elf`. The default builds (without `-DCIM_DWU`) still disassemble identically to the
ELFs used before. The student ran the board on 2026-11-23.

| Check | Gate | Board | |
|---|---|---|---|
| System clock timing | met | WNS +16.18 ns, WHS +0.229 ns | pass |
| 60 sequencer runs (20 clips, 3 ways) | exact | 60 of 60; classes 20 of 20 = golden = label | pass |
| go→IRQ cycles | 67,789 + S's offset 68 = 67,857 | **67,856 in all 60 runs** | 1 below |
| Live path, k = 1, 2, 4 (four 10-s reports each) | no overrun | 500 / 250 / 125 decisions per 10 s, overrun 0 | pass |
| Real microphone, k = 1, 2, 4 | logits = golden model | 25 of 25, 13 of 13, 7 of 7 decisions | pass |

**The one cycle.** The offset is the CPU's time from the go write to the interrupt handler, so it depends on the
firmware, and the gate assumed it did not. In the sleep loop S's build has `li a5,0` between `wfi` and the
`csrw mcountinhibit`; in the unit's build the compiler moved it ahead of the loop, so the interrupt is taken one
instruction sooner. This was predicted (all four numbers below) and then tested in the X-HEEP Verilator system
(`sim-s` and `sim-dwu`, `hopcim_seq` built both ways):

| go→IRQ cycles | S's firmware | The unit's firmware (`-DCIM_DWU=4`) |
|---|---:|---:|
| S hardware | 67,909 (60 exact) | 67,908 |
| Unit hardware | 67,857 | **67,856 (60 exact)** |

- The hardware difference is 52 cycles with either firmware, the 67,841 − 67,789 of R2. The firmware's is 1 cycle on
  either hardware. (The two mixed runs give wrong logits, as they must: the firmware loads the weights where the other
  hardware looks. The sequencer's time does not depend on the data.)
- The system simulation of the unit gives the board's log line for line, including 67,856 cycles and 9,669 to 9,671
  CPU-active cycles; only the one-off setup count differs (314,960 against 314,968; S's differs by the same 8).
- By the gate's wording the cycle check is missed by one cycle. No threshold was moved, and the gate is unchanged. The
  student accepted R6 with this difference explained on 2026-11-25; the firmware was not changed and the run not repeated.
  Logs: `sim/dwu_jobs/offset_2x2/`.

**Scope of the microphone check.** All 130 decisions recorded across the three runs were class `_unknown_`, so it covers
that class only, as S's runs did. The live k = 2 run reported one keyword ("up", 13 of its first 250 decisions); no cause
was recorded, and the gate does not score it. CPU-active cycles per decision were 264 / 266 / 267 (S: 263 / 263 / 266).

## Reproduce

```
sim/run_seq_tb.sh cim d1 ; sim/run_seq_tb.sh cim d4          # benches (also: ded d4)
sim/run_seq_allclips.sh d1 ; sim/run_seq_allclips.sh d4      # 4,890 clips each, ~15 min
sim/run_mic_tb.sh cim 15 1 d4 ; HOPMON=1 sim/run_mic_tb.sh cim 4 1 d4    # live path; worst hop
sim/run_energy_counts.sh cim 5 1 d4                          # counters -> model/energy_counts_cim_d4.log
vivado -mode batch -source syn/ooc.tcl -tclargs cim_endpoint FE_DED=0 PDM_CLKDIV=15 SEQ_S=1 SEQ_DWU=4   # and SEQ_DWU=0
cd model && ~/venvs/planenest/bin/python dwu.py rtl          # R2, R4, R5 -> dwu_rtl.json, dwu_rtl.log
~/venvs/planenest/bin/python verif_mutants.py 3 "DWU:"       # the broken variants
```
