# Depthwise swap-in: results against the frozen gates

Model study, 2026-11-22. The gates are in [`DWSWAP_GATES.md`](DWSWAP_GATES.md):
- frozen at commit `ce3710d`, before the schedule model, the replay or the pricing of these designs existed;
- SHA-256 `ea605b2f…d893d9a8` (`DWSWAP_GATES.sha256`), unchanged.

Nothing here was built in RTL. Power is a model estimate [E] (`ENERGY_ESTIMATE.md`), not an FPGA measurement.

## Verdict

| Gate | V (the swap-in) | Result |
|---|---|---|
| G1 exact, stateful tiles | 0 wrong values on 4,890 clips; the 3 broken variants caught on every clip | pass |
| G2 ≤ 0.80 × S at U, below S in every variant | 28.13 against 44.11 µW = **0.638 ×**; 0.477–0.717 × over the 19 variants | pass |
| G3 ≤ 0.90 × N7 | 28.13 against 34.70 µW = 0.811 × | pass |
| G4 ≤ FS | 28.13 against 28.48 µW (−0.35) | pass, narrowly |
| G5 ≤ DIG | 28.13 against **27.72** µW (+0.41) | **fail** |
| G6 timing | 67,841 cycles (S's, layer for layer); worst hop 80,072 ≤ 81,920 | pass |
| G7 storage ≤ 0.50 × | 272,384 against 565,248 bits = 0.482 × | pass |

**Decision (frozen rule):** G1–G3, G6 and G7 pass, G5 fails. So removing the resident depthwise tiles is worth it, but the
swap-in is not the best way to remove them. A 9-multiplier digital depthwise unit does it for 0.41 µW less. The
comparison is reported below. **No RTL by default; the student chooses.**

**What this means:**
- **The lever is the 40 resident depthwise tiles.** At U's rate they cost about 16 µW of the 44.1. Every way of removing
  them lands at 27.7–28.5 µW (−35 to −37 %). The static alternative, narrow macros (N7), only reaches 34.7.
- **How they are removed matters little, and the swap-in is not the cheapest.** V, FS and DIG leak almost the same
  (20.2–20.3 µW). So the ranking is set by the energy per inference: V 1,157.8 nJ, FS 1,275.1, DIG 1,016.5 (S 1,119.8).
- **The swap-in removes the tiles' leakage, but keeps the dense array's cost of computing a sparse layer.** Every
  depthwise pass still reads 63 rows × 7 columns, 441 cells for 63 non-zero weights. That costs 117 nJ per inference:
  - tile select 77.7 nJ;
  - accumulators 33.8 nJ;
  - array 5.5 nJ.

  The digital unit does the same 288,000 MACs for 8.9 nJ.

## Validity (V1–V4): all pass

| ID | Result |
|---|---|
| V1 | `streamfuse_sched.py`'s S: 67,841 cycles, 4,533 / 11,273 / 4,531 / … / 92 by layer, equal to the RTL (`BASELINE_S.md` §4) |
| V2 | S through this study's replay (persistent input banks, below) equals `dscnn_int` on all 4,890 clips: 0 wrong values, also with adversarial unused rows |
| V3 | S at U 44.106 µW, at 40 ms 68.785 µW (gate: 44.11, 68.78) |
| V4 | Each of V, N7, FS and DIG with its change off gives S's power exactly (largest difference 0.0 µW, 19 variants, both rates). V's program with its order and tiles off gives S's cycle-model events exactly |

## Exactness (G1)

The replay (`model/dwswap.py exact`) applies the cycle model's gathers, passes, rewrites and result writes in cycle order
on real clips. Three things carry state, as the hardware would:
- **The input banks persist.** `cim_seq.sv`'s wide port is masked to the chunk's lanes: 7, or 1 for group 9. So the
  rows a vector does not write keep an earlier vector's data, random at the start.
- **The working tiles persist** from group to group and across inferences. A row write writes the whole 16-column row.
- **Group 9 runs on stale cells.** It writes only column 0's 9 cells. Rows 7k + 1 … 7k + 6 keep group 7's weights and
  meet stale inputs. This is harmless only because column 0 is zero in those rows and columns 1–6 are discarded.

| Case | Clips | Wrong values | Wrong logits (clips) | Working-tile hazards |
|---|---:|---:|---:|---:|
| S | 4,890 | 0 | 0 | — |
| V | 4,890 | **0** | 0 | 0 |
| V, every unwritten input row replaced by fresh random values before each pass | 4,890 | **0** | 0 | 0 |
| broken: a group written into the tile still in use | 4,890 | 200,583,187 | 4,890 | 5,096 |
| broken: one group's write skipped (L3, group 4) | 4,890 | 98,962,592 | 4,890 | 125 |
| broken: writes issued at the last pass's issue, not its done | 4,890 | 2,662,419 | 2,955 | 76 |

In the last broken variant, a write lands during a pass's bit cycles. The replay then computes that pass bit plane by
bit plane, least significant bit first, with the tile as it is in each cycle (38 such passes per inference).

## Timing (G6)

- **Cycles:** V's group-by-group order takes exactly S's cycles in every layer: 67,841 per inference, with the same
  access counts.
- **No stalls:** every group's rewrites land at least 1,056 cycles before its first pass. One row write per cycle
  starts after the tile's previous group is done. So the rewrites never delay a pass.
- **Worst hop:** 80,072 cycles, unchanged, against a 4.096-MHz hop of 81,920.

## Power (G2–G5)

Baseline variant. Leakage in µW; energy of one inference in nJ:

| Part | S | V | N7 | FS | DIG |
|---|---:|---:|---:|---:|---:|
| Weight cells | 5.88 | 2.81 | 3.96 | 2.81 | 2.47 |
| Other SRAM (incl. the 2,304-byte store) | 4.07 | 4.34 | 4.07 | 4.34 | 4.34 |
| CIM macro logic | 22.20 | 8.88 | 14.71 | 8.88 | 8.88 |
| Digital logic | 4.18 | 4.20 | 4.18 | 4.18 | 4.57 |
| **Leakage** | **36.33** | **20.23** | **26.92** | **20.22** | **20.26** |
| CIM array | 134.2 | 134.2 | 134.2 | 129.5 | 94.9 |
| Tile select | 169.1 | 169.1 | 169.1 | 102.4 | 91.3 |
| ACT memory | 285.8 | 285.8 | 285.8 | 121.4 | 285.8 |
| Sequencer logic | 155.5 | 155.5 | 155.5 | 94.6 | 155.5 |
| FS gathers | — | — | — | 150.3 | — |
| Rewrites | — | 33.1 | — | 264.8 | — |
| Store reads | — | 4.9 | — | 38.9 | 4.9 |
| Digital unit | — | — | — | — | 8.9 |
| Rest (requantisation, FEAT, registers) | 375.3 | 375.3 | 375.3 | 373.2 | 375.3 |
| **Energy per inference** | **1,119.8** | **1,157.8** | **1,119.8** | **1,275.1** | **1,016.5** |
| **Power at U (k = 16), µW** | **44.11** | **28.13** | **34.70** | **28.48** | **27.72** |
| Power at 40 ms, µW (not gated) | 68.78 | 53.63 | 59.37 | 56.55 | 50.14 |

**Why DIG is below V:**
- **Dynamic energy:** DIG avoids V's depthwise passes and rewrites, 141 nJ per inference (0.44 µW at U).
- **Leakage, nearly equal:** DIG's 7,756 gate equivalents leak 0.39 µW. V's two working tiles leak 0.34 µW (two tiles of
  cells, charged at 2 ×) and its control 0.015 µW. DIG leaks 0.03 µW more.
- **Net:** 0.41 µW.
- **Break-even size:** DIG stays below V up to about 15,900 gate equivalents, 2.05 × the frozen size, in the baseline
  variant. At 100 pW per gate equivalent the margin is only 0.04 µW.

**Why FS is above V:** its 18,432 row writes per inference (265 nJ) and gathers (150 nJ) cost more than it saves:
- the depthwise passes' select and array energy;
- S's depthwise gathers;
- 26,568 sequencer cycles.

**In every variant** (`model/dwswap_run.log` has the tables at both rates):
- **G2:** V is below S in all 19 variants: 0.477 × with 1 tile per macro, 0.717 × at 20 pW per gate equivalent.
- **G5:** DIG is below V in all 19, by 0.04 µW (logic leakage 100 pW per gate equivalent) to 9.05 µW (1 tile per macro).
- **G4:** FS is below V in 2 variants: by 0.03 µW (alpha 0.25) and by 2.76 µW (Colonnade, where the whole depthwise array
  term is scaled to 9/63).

### Readings of the costing rules (baseline, at U; not gated)

Where the frozen rules did not say, the reading kinder to the baseline, and so harder for V, was taken. The other
reading is reported here.

| Reading | µW | Against V (28.13) |
|---|---:|---:|
| V, working-tile cells at 1 × / 3 × | 27.95 / 28.30 | — |
| V, rewrites and store reads free | 28.01 | — |
| V, 1 × cells and free rewrites together | 27.84 | still 0.12 above DIG, 0.64 below FS |
| FS, tile select not scaled (narrow reading) | 28.69 | V lower by 0.56 |
| FS, 8-bit passes | 28.46 | V lower by 0.34 |
| FS, free gathers (lower bound) | 28.01 | **FS lower by 0.12** |
| FS, S's cycles (sequencer not credited) | 28.67 | V lower by 0.54 |
| FS, 8-bit passes and free gathers together | 27.99 | **FS lower by 0.13** |
| DIG, AND gates at 1.33 GE | 27.73 | DIG lower by 0.40 |
| DIG, a 63-byte weight register (+3,024 GE) | 27.87 | DIG lower by 0.26 |
| DIG, weights read from the store every cycle (32,000 reads) | 27.91 | DIG lower by 0.21 |
| DIG, all three unkind readings together | 28.07 | DIG lower by 0.05 |

**No single reading flips G5.** Only all of V's kind readings (27.84) against all of DIG's unkind ones (28.07) put V
first, by 0.23 µW. G4 flips with free gathers.

### Interpretations made in this study

The frozen text left these open; each was settled before scoring, by the rule above:
- **FS:**
  - Tile select is scaled by 9/63 with the array term ("active rows"; ZigZag's select term is per row).
  - The sequencer is charged FS's own cycles: 4,631 per depthwise layer, 9 a pass, with gathers, writes and issue
    fully overlapped.
  - Its input registers hold 9 rows a pass. It has no control logic (its rule names only the working tiles and the
    store).
  - Under Colonnade, which gives no split, the whole depthwise array term is scaled.
  - The frozen × 9/63 itself is kind to FS. The measured depthwise alpha already includes the 7.3 % weight density.
- **DIG:**
  - An AND gate is 1 gate equivalent.
  - The weights come from the frozen 2,304 store reads.
  - The MAC activity is the depthwise inputs' toggle rate, 0.117: 0.5 under "alpha 0.25", Horowitz's 8 × 8 figure under
    "multipliers: Horowitz".
  - DIG takes S's cycles. It is gather-bound: 9 ACT words per 7 channels.
- **The replay:**
  - A group's first write lands strictly after its tile's previous group is done.
  - Bit planes run least significant first; this only matters in the broken variant.

## Storage (G7) and area

- **Bits:** V holds 31 tiles plus the store, 272,384 bits, against S's 69 tiles, 565,248 bits (0.482 ×).
- **Storage area** (cells and store, 28 nm, with 1R1W cells at 2 × area): S 0.0965 mm², V 0.0493, N7 0.0651, FS 0.0493,
  DIG 0.0437 (plus the unit's logic).

## The choice

| Option | U, µW | 40 ms, µW | Needs | Against |
|---|---:|---:|---|---|
| **DIG**: a digital depthwise unit | 27.72 (−37 %) | 50.14 | 9 multipliers of 8 × 8 and a tree (~7,800 GE), fed by today's gathers; the 40 depthwise tiles removed; no new cell | The usual workaround in KWS chips; the depthwise layers leave the CIM |
| **V**: the swap-in | 28.13 (−36 %) | 53.63 | 2 working tiles with a 1R1W cell, a 2,304-byte store, a rewrite path; exact, no cycles added | 0.41 µW above DIG under the frozen rules, 3.5 µW at 40 ms; needs a 12T cell |
| **N7**: narrow static macros | 34.70 (−21 %) | 59.37 | Narrow macros only | Far behind |
| Keep S with mode U | 44.11 | 68.78 | Nothing | — |

What holds whichever way is chosen: at a low duty cycle, the sparse depthwise layers' resident tiles are the largest
removable cost in HopCIM. Removing them is worth about 16 µW, 36 % of the system at U, and costs no cycles. Any RTL needs
its own frozen gate first; for DIG, the size measured out of context against the 15,900-GE break-even.

## Reproduce

```
cd model
~/venvs/planenest/bin/python dwswap.py all > dwswap_run.log    # about 7 minutes; writes dwswap_results.json
```

`dwswap.py` imports `streamfuse_sched.py`, `energy_est.py` and `cim_net_map.py` unchanged.
