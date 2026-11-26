# Documents

An index by kind. Every number in these documents traces to a file in the repository (most to `model/`, `sim/logs/`,
`syn/` or a board log). Start with `HOPCIM_PROPOSAL.md` for the research question and status, `DESIGNS.md` for the
two designs (the default S+DWU and the baseline S), and `EVALUATION.md` for the end-to-end results.

**Frozen documents.** The `*_GATES.md` files were written and committed before the work they judge, and are never
edited; each has a `.sha256` file (`sha256sum -c docs/<NAME>_GATES.sha256` from the repository root). Corrections go
into the matching results document.

## Proposal, plan and status

| Document | What it holds |
|---|---|
| `HOPCIM_PROPOSAL.md` | the research proposal: background, gap, questions, hypotheses, plan to June 2027, status table |
| `DESIGNS.md` | the default design S+DWU against the comparison baseline S: how to select each, side-by-side results |
| `EVALUATION.md` | end-to-end evaluation: what each step that moved work into hardware buys |
| `HopCIM_Specific_Next_Steps_and_Professor_Advice.md` | the 2026-11-08 execution plan (supplied) |
| `HopCIM_StreamFuse_Research_Proposal.md`, `ExactReuse_CIM_Research_Proposal.md` | proposals for directions that were studied and closed (supplied) |

## Specifications and build steps (R1 to R7)

| Document | What it holds |
|---|---|
| `ENGINE_SPEC.md` | R1: `cim_engine`, the FPGA emulation of a bit-serial digital CIM |
| `INTEGRATION.md` | R2/R3: `cim_endpoint` and its X-HEEP integration (FuseSoC flags, address map) |
| `MFCC_INT_SPEC.md` | R4a: the integer MFCC front-end (golden model) |
| `R5_FRONTEND_PLAN.md`, `WINDOW_PATH.md` | R5: the hardware front-end and the window path |
| `NETWORK.md` | R6: the DS-CNN on the CIM |
| `SEQUENCER.md` | R6b: the hardware layer sequencer |
| `LIVE_AUDIO.md` | R7: the live microphone path without the CPU |
| `B3_BASELINE.md` | B3: the dedicated MFCC front-end baseline |
| `BASELINE_S.md` | the baseline S (sign-gated requantisation, second lane, no L8 write) |

## Studies: gates (frozen) and their results

| Gates | Results | Subject |
|---|---|---|
| `STREAMFUSE_GATES.md` | `STREAMFUSE_STUDY.md`, `STREAMFUSE_B1.md`, `STREAMFUSE_NOVELTY.md` | fused dataflows; B1 in RTL |
| `WHOLEWORD_GATES.md` | `WHOLEWORD_STUDY.md` | whole-word decisions, mode U |
| `DWSWAP_GATES.md` | `DWSWAP_STUDY.md` | ways to run the depthwise layers |
| `DWU_GATES.md` | `DWU_RESULTS.md` | the digital depthwise unit in RTL |
| `NET2_GATES.md` | `NET2_FEASIBILITY.md` (before), `NET2_RESULTS.md` | the second network, ResNet8 |

Other studies: `REUSE_EVALUATION.md` (exact activation reuse), `SAME_AUDIO.md` (rate, phase and confirmation on
identical speech), `TIMING_SENSITIVITY.md` (CIM timing).

## Cross-cutting

| Document | What it holds |
|---|---|
| `VERIFICATION.md` | assertions, coverage, mutation, extreme input, gate level |
| `ENERGY_ESTIMATE.md` | the 28-nm energy model [E]: activity, published tools, how to rerun |
| `LITERATURE_REVIEW.md` | related work by category, gaps, comparison figures |
| `NOVELTY_SYSTEM.md` | the system-level novelty boundary: the closest works, read in full |
| `MLPERF_TINY.md` | HopCIM under MLPerf Tiny's own runner (unverified results); in the working tree until the MLPerf work is committed |
