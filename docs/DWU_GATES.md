# Digital depthwise unit: gates (frozen before the RTL)

Frozen: 2026-11-22, before any RTL of the unit existed. Baseline: commit `f3b8b56` (the swap-in study). The student
chose the digital depthwise unit with group-by-group order (`DWSWAP_STUDY.md`, "The choice"). They also asked for it
to be built in steps, one depthwise layer first, with a size check, a board run and a re-pricing from the RTL's
counters.

**Not blind.** The predictions below come from `model/dwu.py` (`model/dwu_prediction.json`), run before this freeze
on the design fixed here. They are what the RTL must reproduce. Energy is a model estimate [E], never FPGA power.

## The design (fixed now)

**Where:** in `cim_seq`, beside the gather, built on S (`SEQ_S`):
- parameter `DWU`: 1 = L1 only (step D1), 4 = all four depthwise layers (step D4);
- endpoint parameter `SEQ_DWU`;
- FuseSoC flag `use_cim_dwu` (D4), added to X-HEEP as a new patch of `hopcim/xheep/patches`.

**Order:** each depthwise layer on the unit runs group by group (for g = 0..9: for p = 0..124; S: for p: for g). A
vector still has 9 chunks: tap k = 0..8, with the group's 7 channels, or 1 for group 9.

**Datapath (per chunk, in the gather's stage G1):**
- The inputs are x = act − zp, 0 for padding taps, and lanes beyond the group's channels forced to 0.
- 7 products x[i] × w[k][i] (9-bit × 8-bit, signed) are added into the 7 accumulators of the chunk's X bank:
  2 banks × 7 × 20 bits (|sum| ≤ 9 · 256 · 128 < 2^19). Chunk 0 replaces instead of adding.
- In the unit's layers the gather does not write the engine's X bank.

**Passes:**
- The sequencer's issue logic is unchanged: bank full, drain pacing, tag queue.
- A depthwise pass's go goes to the unit, not the engine. The unit is always ready and gives done one cycle after
  the issue.
- The results latched are the 7 accumulators of the pass's bank.

**Weights:**
- A 63-byte register wreg[k][i] in flip-flops.
- A store of 360 words × 56 bits: word (10 l + g) · 9 + k is tap k of group g of depthwise layer l, byte i is
  channel 7g + i, and group 9 uses byte 0.
- The CPU writes the store at `0x0_6000` (word w: bits 31..0 at `0x6000 + 8w`, bits 55..32 at `0x6004 + 8w`), only
  while the sequencer is idle, like PARAMS.
- **Loading:**
  - When the network starts, all 9 taps of group 0 of the first layer on the unit are loaded (9 store reads).
  - When chunk k of a vector at p = 124 has been multiplied, tap k of the next group is read and written into
    wreg[k] in the following cycle.
  - The next group follows group 9 of a layer with group 0 of the next layer on the unit, and after the last layer
    with the first one's group 0.
  - Chunk k of the next group's first vector comes at least 9 cycles later.

**Everything else is as S:**
- conv1, pointwise and FC run on the engine;
- requantisation (two lanes, sign gate), ACT and the window path are unchanged;
- in D4 the firmware does not load the 40 depthwise tiles. The FPGA engine keeps its 128 tile slots; the energy
  model's CIM has 29 tiles in 2 macros.

**FPGA:** multipliers in LUTs (`use_dsp = "no"`), so the size measurement counts them as logic.

**Departures from the study's DIG, disclosed:**
- **Tap-serial:** 7 multipliers and 7 accumulators per bank, instead of 9 multipliers and a 9-input tree. The gather
  delivers one tap of 7 channels per cycle, so this needs no input buffer. It does the same 288,000 MACs in the same
  9 cycles a vector.
- **Weight register:** the study's "63-byte weight register" reading, with 369 store reads per inference instead of
  2,304.

## Predictions (`model/dwu.py`)

| | S | D1 | D4 |
|---|---:|---:|---:|
| Network cycles per inference | 67,841 | **67,828** | **67,789** |
| By layer | 4,533 / 11,273 / 4,531 / … / 92 | L1 11,260, rest as S | each depthwise layer 11,260, rest as S |
| Worst hop | 80,072 | 80,059 | 80,020 |
| Unit passes / MACs / store reads | — | 1,250 / 72,000 / 99 | 5,000 / 288,000 / 369 |
| Engine passes | 7,501 | 6,251 | 2,501 |
| ACT word reads / byte writes | 42,960 / 64,000 | as S | as S |
| Power at U [E] | 44.11 µW | 39.64 µW | **27.95 µW** (19.75–147.48 over the 19 variants) |
| Power at 40 ms [E] | 68.78 µW | 63.73 µW | 50.28 µW |

**Size by construction: 12,224 gate equivalents.** It is made up of:
- 7 multipliers of 9 × 8 bits: 3,998;
- 7 adders of 20 bits: 1,092;
- accumulators: 2,426;
- the weight register and its read mux: 4,068;
- the loader and control: 640.

With this size, D4 breaks even with the swap-in (28.13 µW) at about 15,860 gate equivalents.

## Gates

| ID | Gate | Pass if |
|---|---|---|
| R1 | Exact | D1 and D4 pass four checks. (1) `tb_cim_seq`: stop-after-layer runs (every output of L0–L8 of clip 0), 20 clips' logits and class, the FEAT ring and synthetic sets 1 and 2, with 0 errors. (2) The RTL on all 4,890 GSC test clips (`run_seq_allclips.sh`): 0 logit or class mismatches against `dscnn_int`. (3) For D4, the live path (`tb_cim_mic`, 101 decisions from PDM) is bit-exact. (4) The existing regression (`sim/run_all.sh`) still passes. Three broken variants must each be caught by the benches: (a) the next group's taps loaded during the group's second-to-last vector (p = 123); (b) the two banks' accumulators swapped at the issue; (c) after group 9 of a layer, group 0 of the same layer loaded instead of the next |
| R2 | Cycles | Network cycles ≤ 1.02 × 67,841 = 69,198. The RTL equals the prediction in every layer (D1 67,828; D4 67,789), and its counters (unit passes, MACs, store reads, engine passes, ACT reads and writes) equal the predicted counts. A difference is a defect in the RTL or the model, found and reported before the gate is scored |
| R3 | Real time | D4's worst hop (hop monitor, `tb_cim_mic` at the 4.096-MHz divider) ≤ 81,920 cycles, with no overrun |
| R4 | Size | The unit's added logic is ≤ 15,900 gate equivalents by both counts. (a) Vivado out of context: the endpoint with `SEQ_DWU` = 4 against S on the same source, at 6 per LUT and 6 per FF (the energy model's rule), 1,500 per DSP; the store's block RAM is priced as memory. (b) By construction, as `dwu.py`, with the register count taken from synthesis. The larger one is used |
| R5 | Power | Re-priced from the RTL's own counters (`sim/run_energy_counts.sh`, as for S and B1) with the size from R4: ≤ 28.13 µW at U (the swap-in, baseline variant). Reported against S and in every variant |
| R6 | Board | The D4 bitstream meets timing (system clock). On the board, three checks: (1) 60 sequencer runs exact, go→IRQ cycles = the simulation plus S's fixed setup offset; (2) the live path at k = 1, 2, 4 has no overrun; (3) the real-microphone logits equal the golden model. The student runs the board commands |

**Order:**
1. D1: R1 (bench, 4,890 clips) and R2.
2. D4: R1–R5.
3. The bitstream, then R6.

## Decision

- **All pass:** the unit becomes a built HopCIM option ("S + DWU"). The student decides whether it replaces S in the
  default bitstream and the documents.
- **R1, R2, R3 or R6 fails:** it is a defect; it is fixed and rerun, with no threshold moved.
- **R4 or R5 fails:** the unit costs more than the swap-in would. This is reported, and the student chooses; S stays
  the default.
