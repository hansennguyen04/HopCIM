# A second MLPerf Tiny network on HopCIM: feasibility (no RTL yet)

Question: is the exactness special to the KWS DS-CNN, or does the same CIM run another published MLPerf Tiny model unchanged?
Started 2026-11-25 on the student's request. This is the model-side step; the RTL waits for a frozen gate.

## 1. Which network

| Model (MLCommons reference, int8) | Weights | Tiles of 128 | New layers | Data here | |
|---|---:|---:|---|---|---|
| **ResNet8, image classification** (`pretrainedResnet_quant.tflite`) | 77,360 | **86** | 3 × 3 convolution with stride and SAME padding, residual add, 8 × 8 pool | CIFAR-10 test set | **chosen** |
| Streaming wakeword | 46,040 | 45 + 1-D depthwise | 1-D depthwise (kernels 3 to 15, 128 channels) | not here | fits; smaller step |
| Anomaly autoencoder | 264,192 | 266 | none | not here | does not fit |
| Visual wake words | 208,112 | > 128 | 1 × 1 and depthwise only | not here | does not fit |

ResNet8 is the most different network (2-D images, no depthwise, residuals), its data is here, and iMCU (JSSC 2024) ran the
same MLPerf Tiny ResNetv1, so it gives a published point of reference (86.96 % on CIFAR-10, 60.9 ms, 102 µJ in silicon).
It is the model's own accuracy that is reproduced, not a better one.

## 2. What is done, with evidence

| Check | Result |
|---|---|
| `model/resnet_int.py`: integer golden model in TFLite reference-kernel semantics (convolutions, residual add, average pool, FC) against the TFLite reference interpreter, every layer, 10,000 CIFAR-10 test images | see §3 |
| Accumulators (with bias, worst case over inputs) | at most 24 bits (5,655,833 < 2²³): HopCIM's 24-bit latched accumulators suffice |
| Shifts | all right shifts (0 to 11, 4-bit field; add 17, 18): HopCIM's requantisation already has what the convolutions need |
| Wiring of the file | equals the model's, asserted at export |
| Inputs x − zp | in −255..255: 9-bit passes, as in the DS-CNN |
| `model/resnet_map.py exact`: every convolution and the FC executed pass by pass on 64 × 16 tiles | 0 mismatches in every layer (4 images); 15,617 passes, 86 tiles |
| `model/resnet_sched.py`: estimated cycles of a descriptor-driven sequencer | about 156,000 (140,544 engine-bound), 10.2 ms at 15.36 MHz |

## 3. Integer model on the 10,000 CIFAR-10 test images

All 10,000 test images, every layer (13 tensors, from the first convolution to the FC logits) against TFLite's
reference-kernel interpreter: **0 mismatching layers on any image** (log `model/resnet_int_check.log`). Argmax equal to TFLite's softmax on
10,000 of 10,000. Accuracy **87.07 %** (8,707 of 10,000), the published model's own. iMCU reports 86.96 % for its ResNetv1
(not necessarily the same file); the difference is not a result of this work.

## 4. What the RTL needs

Reused unchanged: the CIM engine (128 tiles, accumulating passes), the requantisation arithmetic (right shifts, 24-bit
accumulator), the parameter memory format (64 entries × 10 layers: ResNet8 has exactly 10 weighted layers of at most 64 channels).

New, behind a parameter, with `cim_seq` (S, S+DWU) left untouched:

1. A **descriptor-driven sequencer** (`cim_gseq`): per layer a small register record (type, kernel, stride, channels, map
   size, tile base, row tiles R, column tiles C, buffers, zero points, clamp).
   One loop nest serves convolutions (K = 1, 3), FC and pool.
   X vectors are gathered a word (8 channels) per cycle; row passes are chained in the engine's accumulators.
2. **Activation memory of 48 KiB** (three 16-KiB banks, 64-bit words; residual tensors live in different banks) and a 3-KiB image memory.
3. A **residual-add unit**, 2 elements per cycle: two 9 × 31-bit and one 30 × 31-bit multiplier per lane with TFLite's exact rounding
   (`srdhm(a, b) = (a b + 2^30) >>> 31` for all signs, as the DS-CNN's).
4. Average pool over 8 × 8 (64 × 64 elements).

Cost, estimated: 12 to 14 RAMB36 (83.5 of 135 used now), a few thousand LUTs (44,022 of 63,400 used now), and about 16 DSPs.
Weights, parameters and images are loaded by the firmware (77 KB), which needs the 256-KiB configuration (patch 0005) or a loader.

What this will and will not show:
- It shows that HopCIM's **CIM engine, arithmetic and exactness method** are not special to the DS-CNN.
- The **sequencer** is new for this network. If `cim_gseq` also runs the DS-CNN (an optional later step), one sequencer
  would run both; otherwise the generality claim is for the engine, the arithmetic and the method, not the sequencer.
- It is an image-classification network; it does not test the audio path.

## 5. Proposed gate (to be frozen before the RTL; not frozen yet)

| | Gate | Pass |
|---|---|---|
| N1 | Exactness in RTL | every layer of 20 images bit-exact against `resnet_int.py`; all 10,000 CIFAR-10 test images, logits and argmax, 0 errors |
| N2 | Cycles | equal to a frozen schedule model, or ≤ 1.02 × it, per inference |
| N3 | S and S+DWU unaffected | regression 53 of 53; `cim_seq.sv` and the engine unchanged; their cycles unchanged |
| N4 | Size and timing | fits the XC7A100T (LUT, BRAM, DSP), system clock slack ≥ 0 at 15.36 MHz |
| N5 | Board | logits of MLPerf Tiny's 200 performance images equal the model's; go → interrupt cycles equal simulation plus a measured offset |
| N6 | CPU baseline | TensorFlow Lite Micro of the same model on the CV32E20, same board, for the cycle ratio (unverified) |

A failure of N1, N2, N3 or N5 is a defect and is fixed, with no threshold moved. N4 failing means the layer set is reduced or the memory resized
and reported.
