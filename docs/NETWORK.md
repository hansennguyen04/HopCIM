# R6 — the DS-CNN on the same CIM

Status (2026-10-27):
- integer reference verified against TFLite;
- mapping verified;
- CIM grown to 128 tiles;
- CPU-orchestrated network (baseline B1 for the network) bit-exact on the SoC and on the board (20 clips);
- **hardware layer sequencer (R6b) done: 85,420 cycles (5.69 ms) per inference with the CPU asleep,
  bit-exact on the board** (`SEQUENCER.md`).

Next: the per-window feature step in hardware (R5b), the largest remaining CPU term
(`SEQUENCER.md` §9).

## 1. The model

The MLPerf Tiny int8 DS-CNN (`kws_ref_model.tflite`):
- conv 10 × 4, stride 2, on the 49 × 10 int8 features → 25 × 5 × 64;
- 4 × (depthwise 3 × 3 + pointwise 1 × 1), 64 channels;
- average pool and a 12-way fully connected layer.

It has 22,016 int8 weights (21.5 KiB) and 588 int32 biases, and needs 2,656,768 MACs per inference.
Weights are per-channel symmetric. Activations have zero point −128 (the ReLU is the int8 floor);
the input has zero point 83, the logits 14.

## 2. Integer reference, bit-exact against TFLite

`model/dscnn_int.py` implements the TFLite reference-kernel semantics:
- SAME padding by skipping taps;
- `MultiplyByQuantizedMultiplier` with double rounding, multipliers from `QuantizeMultiplier`;
- int8 average-pool rounding (half away from zero);
- the prediction is the argmax of the fully connected int8 output.

Checked against the TFLite `BUILTIN_REF` interpreter on **all 4,890 test clips**, fed with the
integer CIM-MFCC features (`dscnn_int.py check`):

| Check | Result |
|---|---|
| Every layer output (conv1, 4 × dw, 4 × pw, pool, fc) | **0 mismatching clips** |
| argmax(fc) vs TFLite's int8 softmax argmax | 0 differences |
| Accuracy (integer CIM-MFCC features) | **91.80 %**, ours = TFLite reference kernels |
| Accuracy with TFLite's default interpreter (XNNPACK) | 91.78 %; its predictions differ from the reference kernels on 24 clips |

The earlier gates used the default interpreter. The hardware is bit-exact with the reference
kernels, which give 91.80 %.

## 3. Mapping onto the CIM (`model/cim_net_map.py`)

| Layer | Tiles (64 × 16) | Mapping | Passes/inference |
|---|---:|---|---:|
| conv1 | 4 (8–11) | rows 4 fy + fx (40), columns 16 output channels | 500 |
| depthwise ×4 | 4 × 10 | block diagonal, 7 channels per tile, tap-major (rows 7 (3 fy + fx) + c') | 5,000 |
| pointwise ×4 | 4 × 4 | rows = 64 input channels | 2,000 |
| fully connected | 1 (68) | rows = 64 inputs, 12 columns | 1 |
| **Total** | **61 (8–68)** | | **7,501** |

**Tap-major depthwise rows (R6b).** The 7 channels of one tap are 7 consecutive rows, so the
layer sequencer fills a depthwise input vector with 9 activation-word reads (one per tap), one per
cycle. The first mapping (rows 9 c' + 3 fy + fx, channel-major) needed 63 single-row writes.
Model, generator assertion and firmware use the new layout; mapping self-test: 20 clips, 0 logit
mismatches.

**Zero points.** Each input carries its zero point (x = act − zp; conv1 gives −211..44, the
others 0..255). Each pass then gives Σ W·(x − zp) exactly, SAME-padding taps are simply 0, and
the inputs are 9 bits wide (9 cycles per pass).

**Exactness.** Mapping model against the integer reference: 40 clips, 0 logit mismatches. The
firmware's tile construction is asserted equal to the mapping model's tiles.

**Capacity decision: 128 tiles.** The front-end (8 tiles) plus the network (61) need 69. Options:

| Option | Assessment |
|---|---|
| **Grow the CIM to 128 tiles** | Chosen. +32 RAMB18 on the FPGA; keeps the whole network on one CIM; 59 tiles left for a second model |
| Depthwise on a small vector unit | Rejected for now. Breaks "everything on one CIM"; RV-Flash already does this |

Block-diagonal depthwise is the most capacity-efficient CIM mapping, but it still inflates
2.25 KiB of depthwise weights into 40 KiB of tiles. Reported as a known cost.

Endpoint address map change: the window is now 256 KiB, weights at offset `0x2_0000`, tile field 7 bits.

## 4. CPU-orchestrated network on the SoC (`sw/applications/hopcim_net`)

The firmware builds the 61 tiles from the compact weights (it stores 22 KB, not 61 KB) and runs
all 7,501 passes. For each pass the CPU gathers the inputs, starts the pass, reads the results,
and requantises them with TFLite's integer arithmetic.

The measurements below were made with the first, channel-major depthwise layout. The tap-major
layout re-measured on the R6b bitstreams is at the end of this section.

**Board** (Nexys A7-100T, `xheep_nexys_bscan_pdm_cim128_cim.bit`): 20 GSC test clips covering all
12 classes. The int8 logits are bit-exact against the golden model for all 20, all 20
predictions are correct, and all equal the golden predictions. The suite was run twice with
identical output.

**Simulation equals the board.** The same C code, run on one clip (`hopcim_net1`), gives
**12,874,346 cycles on both the board and Verilator**, with every layer total identical:

| Part | Cycles/inference (clip 0, `hopcim_net1`) | Share |
|---|---:|---:|
| conv1 | 646,646 | 5.0 % |
| depthwise ×4 | 9,612,127 | **74.7 %** |
| pointwise ×4 | 2,578,408 | 20.0 % |
| average pool | 35,631 | 0.3 % |
| fully connected | 1,524 | 0.0 % |
| **Total** | **12,874,336 (0.86 s at 15 MHz)** | |

Tile construction and loading take 442k cycles, once.

**Sensitivity to code generation.** The cycle count of this software baseline depends on how GCC
compiles its hot loops:
- In the 20-clip build (`hopcim_net`), every clip takes 12.006–12.022 M cycles, so the data
  dependence is only ±0.07 %.
- The same clip 0 takes 12.87 M cycles in the 1-clip build and 12.01 M in the 20-clip build.
  The difference is in the depthwise loops (9.61 M vs 8.74 M), which compile differently.
- The first simulation run (13.07 M) used an earlier revision of the source.

CPU-orchestrated numbers are therefore quoted per build. None of the conclusions depend on this
±7 %.

**Tap-major layout (R6b firmware, `xheep_nexys_bscan_pdm_cim128s_cim.bit`).** The C gather for the
new layout is slightly slower:
- `hopcim_net1` gives **13,020,998 cycles on both the board and Verilator**, every layer total
  identical: conv1 660,243; depthwise 9,728,330; pointwise 2,596,756; pool 34,218; FC 1,441.
- Only the one-time tile setup differs, by a fixed 4 cycles (459,035 vs 459,031). `hopcim_seq`
  shows the same offset.
- The 20-clip build takes 11.79–11.81 M cycles per clip, all 20 bit-exact and correct.

**Comparison.**
- The CPU-only network is about 33 M cycles. That is an estimate from the measured per-MAC costs:
  conv1 12.6, pointwise 11.5, depthwise 18.8 cycles/MAC.
- The CPU-orchestrated CIM network (12.0–12.9 M cycles) is therefore about 2.6× faster, but the
  CPU is busy for the whole 0.8–0.86 s, and every hop is 0.02 s.
- Depthwise on the CIM (8.7–9.6 M) is slower than depthwise on the CPU (about 5.4 M). Each pass
  needs 63 CPU-written inputs for 7 outputs.
- Requantisation dominates the pointwise layers (64 outputs per position, about 40 cycles each).

**Implication.** As with the front-end (R4b), offloading only the MACs does not make the system
CPU-light; data movement and requantisation dominate. A hardware layer sequencer (R6b) has to
take over three jobs:
1. gathering inputs from an activation buffer, including zero point and padding;
2. requantisation (per-channel multiplier and shift);
3. pass sequencing.

The engine bound is 7,501 × 9 ≈ 67.5k cycles per inference. With gathers of 4–8 values per cycle,
an exact inference every hop (300k cycles) is in reach [E].

**Outcome (R6b, `SEQUENCER.md`).** The sequencer does all three jobs in hardware:
- 85,420 cycles per inference on the board, bit-exact, with the CPU asleep;
- the depthwise layers run at the engine's own rate (9 cycles per pass);
- CPU-active per decision: 9,678 cycles with a whole window, 459 per streaming step.
