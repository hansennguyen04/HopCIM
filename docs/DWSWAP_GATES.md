# Depthwise swap-in gates (frozen before the model study)

Frozen: 2026-11-22, before the study's schedule model, data replay or pricing of these designs existed. Baseline:
commit `3b73d7a`. The student asked to freeze a gate for "Idea 1" of 2026-11-22: swap the depthwise weights into two
working tiles instead of keeping 40 resident depthwise tiles.

**Not blind.** One estimate was made the same day, with the energy model (`model/energy_est.py`) and tile count
alone, and only for S and the candidate:
- at U's rate (network every 16 hops), 31 tiles in 2 macros against S's 69 tiles in 5 macros gave 28.0 against 44.1 µW
  (0.634 ×), and 0.58–0.71 × over 8 variants;
- at 40 ms it gave 53.5 against 68.8 µW;
- that estimate charged 2,520 row writes per inference; the correct count is 2,304 (below).

The alternatives FS and DIG below were not computed. Reasoning alone suggested both could come close to V, so G4 and
G5 are genuinely open. The thresholds are frozen so that the study cannot move them; any later change is reported as
post hoc.

## Question

At U's rate (44.1 µW, baseline variant) the network is 8 % of the power. The CIM's leakage is 64 %: its compute logic
in 5 macros 50 %, and its weight cells 13 %. 40 of the 69 tiles hold the depthwise layers at 63 non-zero cells of 1,024 each. So:
1. Is removing the resident depthwise tiles worth it, against S and against the best static organisation?
2. Is the swap-in the best way to remove them, against feature-stationary depthwise (Wang et al., ISSCC 2023, made
   digital and exact) and against a small digital depthwise unit, the common workaround?

## Designs

All keep the int8 model, TFLite's arithmetic and the 7,501 network passes of S, except where a design says otherwise.

| Name | Definition |
|---|---|
| **S** | Today's sequencer and tile map (`docs/BASELINE_S.md`): 69 tiles, 5 macros of 16 tiles |
| **V** (candidate) | The 40 depthwise tiles become **2 working tiles**. The depthwise weights live in a dense store of 2,304 bytes (18,432 bits). A depthwise layer runs **group by group** (`for g: for p`, against S's `for p: for g`). Before group g runs, its non-zero cells are written into the working tile g mod 2: 63 cells for groups 0–8, 9 for group 9 (channel 63 only). Each non-zero cell is in its own row (tap-major rows 7k + c′), so that is one row write per cell. The zero pattern of a block-diagonal tile never changes and is written once at start-up. Writes go through a second write port while the other tile computes (simultaneous MAC and write, Fujiwara et al., ISSCC 2022). A group may not be written before the last pass of the group that used that tile is done. The columns of group 9 other than column 0 keep the previous group's weights; their results are discarded. Per inference: **2,304 row writes and 2,304 store reads**. 31 tiles in 2 macros |
| **N7** (static) | S with the 40 depthwise tiles in 3 macros that are only 7 columns wide (the depthwise tiles use columns 0–6). No new mechanism. The other 29 tiles are in 2 full macros |
| **FS** (feature-stationary, digital) | The depthwise layers on 2 working tiles, feature-stationary as in Wang et al. (ISSCC 2023), but digital and exact. Cells hold activations x; the 9 weights of one channel are the input vector; columns are 16 output positions. The zero point is folded into the bias (Σ w (x − zp) = Σ w x − zp Σ w). Per depthwise layer: 64 channels × 8 position groups = **512 passes** of 9 active rows, each preceded by **9 row writes** (one tap of 16 positions per row) on the second write port. 31 tiles in 2 macros, and the same 2,304-byte weight store |
| **DIG** (digital unit) | The depthwise layers on a small digital unit: **9 multipliers of 8 × 8 bits** and a 9-input adder tree, one channel-position per cycle: 288,000 MACs = 32,000 cycles per inference, within S's depthwise time. The same weight store. The CIM keeps 29 tiles in 2 macros. Requantisation and ACT as in S |

## Costing rules (fixed now)

**Shared by every design:**
- The basis is `energy_est.py`'s model: CACTI memories, ZigZag-IMC gate energies, measured input activity.
- The network energy of S comes from its RTL counters (`energy_s_rtl.json`).
- Leakage comes from `Model.leak` with S's second lane (`sf_leak`): weight cells scale with tiles, macro compute logic with macros.
- Power is S's power with no inference, plus network runs per second × energy per inference, plus CPU wakes × 263 × 28 pJ (`energy_est.ww_power`).
- The primary point is U (k = 16); 40 ms (k = 2) is reported.
- The pass decision uses the baseline variant. All 19 variants are reported.

**V:**
- Row writes at the `weights` memory's write energy; store reads at the `params` memory's read energy.
- The store's leakage is at the `params` memory's pW per bit.
- The 2 working tiles use a 1R1W cell: their cell leakage is charged at **2 ×** (12T against 6T; reported at 1 × and 3 ×).
- Swap control is 300 gate equivalents of logic leakage.
- Passes are priced as S's depthwise passes.

**N7:**
- The compute logic of a depthwise macro is 7/16 of a full macro.
- Depthwise cells are 40 × 64 × 7 × 8 bits.
- Everything else is as S.

**FS:**
- A pass is priced as an S depthwise pass, with the array term scaled by 9/63 (active rows) and the accumulator term unchanged.
- Row writes and store reads are priced as for V (9 per pass for each).
- Its gather (144 activation bytes per pass) is charged **per byte needed**, at 1/7 of an `act_word` read, as if ACT were laid out for it. The lower bound with free gathers is reported.
- Working tiles and the store are charged as for V.

**DIG:**
- A MAC costs an 8 × 8 array multiplier (64 AND gates, 56 full adders) at ZigZag's gate energies and the depthwise layers' measured input activity, plus its share of the adder tree (8 adders of 18 bits).
- Size: per multiplier 64 AND + 56 full adders (7.8 gate equivalents each, the model's full-adder size), the tree, +20 % registers, +1,000 gate equivalents of control, all leaking at the logic leakage per gate equivalent.
- The 5,000 CIM depthwise passes (array and tile select) are removed.
- The store is charged as for V.

## Validity (before any gate)

| ID | Condition |
|---|---|
| V1 | The schedule model reproduces S: 67,841 cycles and its per-layer cycles (`streamfuse_sched.py`, as in `STREAMFUSE_STUDY.md`) |
| V2 | The data replay of S equals `dscnn_int` on all 4,890 test clips |
| V3 | The pricing gives S's 44.11 µW at U and 68.78 µW at 40 ms (baseline variant) |
| V4 | Each of V, N7, FS and DIG, priced with its change switched off, gives S's numbers exactly |

If V1–V4 fail, the tool is fixed before any gate is scored.

## Gates

| ID | Gate | Pass if |
|---|---|---|
| G1 | Exactness of V | 0 mismatches against `dscnn_int` on all 4,890 clips: every output of L0–L8 and the 12 logits. The replay carries the working tiles' state from group to group (stale cells included). Three broken variants must each be caught: a group written into the tile still in use, a group's write skipped, and a write issued before the last pass of the tile's previous group is done |
| G2 | Worth it against S | power at U ≤ **0.80 ×** S's (baseline variant), and below S's in every variant |
| G3 | Worth it against the best static organisation | power at U ≤ **0.90 ×** N7's (baseline variant) |
| G4 | Better than feature-stationary | power at U ≤ FS's (baseline variant) |
| G5 | Better than a digital depthwise unit | power at U ≤ DIG's (baseline variant) |
| G6 | Timing | V's network ≤ **1.02 ×** 67,841 cycles, and its worst hop ≤ **81,920** cycles (the 4.096-MHz live path), from the cycle model with the rewrites |
| G7 | Storage | V's CIM cell bits plus its store bits ≤ **0.50 ×** S's CIM cell bits |

## Decision

- **All of G1–G7 pass:** an RTL proposal follows. First one depthwise layer, then all four; checked by the regression,
  the 4,890-clip replay, FPGA utilisation and a board run. It is built only if the student decides to go ahead.
- **G1–G3, G6 and G7 pass, but G4 or G5 fails:** removing the resident depthwise tiles is worth it, but the swap-in is
  not the best way. The comparison is reported and the student chooses. No RTL by default.
- **G2 or G3 fails:** no RTL. Reported as a measured negative result.

Reported beside the gates, not gated:
- every design at 40 ms;
- cycles of N7, FS and DIG;
- every variant;
- the area of the CIM storage.

## Literature read for this gate

- **Wang et al. (ISSCC 2023, 7.5):** a 28-nm analog CIM unit-macro that runs depthwise feature-stationary, with weight
  and feature shifts that avoid reloading overlapping features. Its aims are utilisation and buffer-to-macro traffic,
  and its outputs are approximate (accuracy below software). It removes resident depthwise weights as a side effect, so
  FS is built from it.
- **Fujiwara et al. (ISSCC 2022, 11.6):** a 5-nm digital CIM with a 1R1W 12T cell. Simultaneous MAC and write work as
  long as the write address differs from the MAC location. This is V's assumption for overlapping the rewrites.
  Weights are multiplexed 4 rows per adder tree, shallower than the model's 16 tiles per macro, so the variants with 4,
  8 and 32 tiles per macro matter and are reported.
- The file `Literature review/Seo et al., JSSC 2024.pdf` is MACC-SRAM (Zhang et al., JSSC 59(6), 2024), an analog CIM
  macro, not the 1.5-µW KWS SoC (JSSC 59(1), 2024, pp. 29–39) meant for the power-gating idea. It plays no part in this
  gate.
