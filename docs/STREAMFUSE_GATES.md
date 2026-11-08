# StreamFuse gates (frozen before the model study)

Frozen: 2026-11-08, before the schedule model or the extended energy model existed. Baseline: tag `freeze-2026-11-07b`.
The proposal under test is `HopCIM_StreamFuse_Research_Proposal.md`. Its own gates are replaced here (last section).

**Not blind.** A rough estimate was made the same day, before these gates, with hand counts and the energy model's
28-nm parts: B saves about 193 nJ and C about 249 nJ per inference, sign gating about 217 nJ; storage −48 % (B) and
−68 % (C); B's latency about 72k cycles against 67.9k for the second lane. The thresholds below were proposed in the
same message. They are frozen so that the model study cannot move them. Any later change is reported as post hoc.

## Question

Once the CPU is gone, does an exact cross-layer dataflow on HopCIM's resident-weight CIM save enough energy and
storage to justify building it, against the cheapest exact alternatives?

## Designs

All keep the engine, the 61 network tiles, the 7,501 passes, the 72,012 requantisations and TFLite's arithmetic
unchanged. Only the order of passes and where activations are kept may change. The order must be fixed (independent
of the data), hazard-free and exact.

| Name | Definition |
|---|---|
| **B0** | Today's sequencer, `rtl/cim_seq.sv` at `freeze-2026-11-07b`: 85,352 cycles, two 25 × 5 × 64 maps (2,500 words of 56 bits) |
| **SG** | Sign-gated requantisation: in L0–L8 (output zero point −128) a result with `acc + bias ≤ 0` is exactly −128, so its multiply and rounding are skipped. The parameter read stays |
| **R2** | A second requantisation lane (2 more DSPs), as in `SEQUENCER.md` §5 |
| **S** (strong baseline) | B0 + SG + R2 |
| **A** | A line buffer of input rows for each depthwise layer, filled from ACT once per word; the depthwise gather reads the line buffer |
| **B** | A, plus direct forwarding: a position's depthwise outputs go to a fusion vector and the pointwise gather reads it; pointwise outputs are written in place into one map of 25 + r rows (circular rows) |
| **C** | Depth-first: no full map; each depthwise layer's line buffer is fed by the previous layer's outputs; L8 goes to the pool sums only; conv1 reads FEAT as today |

Each option is evaluated as option + SG + R2, against S. **Fairness rule:** a change that does not depend on the
option (for example, not writing L8 to ACT) is applied to S as well.

## Validity of the models (preconditions)

| ID | Condition |
|---|---|
| V1 | The schedule model of B0 reproduces the RTL's cycles: 8,034 (conv1), 11,273 (each depthwise), 8,032 (each pointwise), 98 (pool and FC), 85,352 in total, each within 0.5 %. Its access counts equal the RTL's counters exactly: 37,960 depthwise and 5,000 pointwise ACT word reads, 72,000 ACT byte writes per inference (`model/energy_counts_cim.log`) |
| V2 | The schedule model of B0 gives the same outputs as `dscnn_int` on all 4,890 test clips: every output of L0–L8 and the 12 logits |
| V3 | The extended energy model reproduces today's 1,389 nJ for B0 with every variant of `ENERGY_ESTIMATE.md` §8 unchanged |
| V4 | DeFiNES (cross-check, not a gate): its activation-memory traffic for the layer-by-layer and the depth-first schedule agrees with the schedule model within 10 %, or the difference is explained before any conclusion is drawn |

If V1–V3 fail, the model is fixed before any option is evaluated.

## Gates (each option; all four must pass)

| ID | Gate | Pass if |
|---|---|---|
| G1 | Exactness | 0 mismatches against `dscnn_int` on all 4,890 test clips: every output of L0–L8 that the option computes, and the 12 logits |
| G2 | Energy | Network energy per inference of option + SG + R2 ≤ **0.90 ×** that of S. Network energy is the total of `ENERGY_ESTIMATE.md` §4 (baseline variant, measured activity), with the option's buffers and added control included. Guard: average total power at 40 ms, leakage included, is not higher than S's |
| G3 | Storage | Activation storage (maps, line buffers, fusion vectors, any other activation storage), in bits at the schedule model's hazard-free size, ≤ **0.60 ×** 140,000 bits |
| G4 | Latency | Cycles per inference of option + R2 ≤ those of S, both from the schedule model |

**Costing rules, fixed now:**
- Memories: CACTI through hwcomponents-cacti 1.0.40, low-standby-power, 28 nm, at their size in 56-bit words; a byte write
  is 0.4 of a word write, as today. Fusion vectors are flip-flops at ZigZag's gate energies and the model's logic activity.
- SG: the skipped fraction is counted by the schedule model over the 4,890 clips. A skipped result saves the 25 × 31
  multiply and 88 of the 113 adder bits; the 25-bit bias add stays.
- R2: 2 more multipliers at 1,500 gate equivalents each, leaking as logic; the sequencer's logic energy scales with its busy cycles.
- Added control logic, as a share of `cim_seq`'s gate equivalents (1,176 LUT, 894 FF): A +10 %, B +25 %, C +60 %,
  charged for dynamic energy (logic activity per active cycle) and leakage. The results are also reported at twice these shares.
- The pass decision uses the baseline variant. All 18 variants of `ENERGY_ESTIMATE.md` §8 and the doubled overheads are reported beside it.

## Decision

- **RTL for B in one block** (depthwise 1 → pointwise 1, behind a parameter) only if B passes G1–G4, and only after the
  student decides to go ahead. Its checks: the 28-bench regression, the logits of 20 clips, new assertions and mutants,
  FPGA utilisation, a board run.
- **B matched the model** means: the RTL's cycles within 2 % of the schedule model, its access counters exactly equal,
  0 mismatches.
- **All four blocks** only after one block matched the model. **C** only if B matched the model and time remains before April 2027.
- **SG alone** is a separate, optional hardware change; the gates do not decide it.

## Replaced from the proposal

| Proposal's gate | Here | Why |
|---|---|---|
| Modelled total energy ≥ 15 % | G2, network energy ≥ 10 % beyond S | Even an activation memory that costs nothing saves 12.4 % of the total at 40 ms, so the original gate cannot be met by any activation dataflow |
| Large ACT accesses ≥ 60 % fewer | Reported, not gated | Energy (G2) is what the accesses are for |
| ACT storage ≥ 50 % less | G3, ≥ 40 % less | As proposed by the student's plan of 2026-11-08 |
| Network latency: no regression, ideally ≥ 10 % better | G4, no regression against S | S already includes the second lane |
| Small added LUT/DSP | Reported, not gated | Part of the costing (added control) |
| A second DS-CNN-like model | Dropped | No second model in this project fits the resident-weight CIM; finding one is its own task |
| FPGA dynamic power (SAIF) | Not used | Energy stays a model-based estimate [E]; no energy claim from FPGA power |
