# Second network on HopCIM (ResNet8): gates, frozen before the RTL

Frozen 2026-11-25, before any RTL. Evidence so far: `NET2_FEASIBILITY.md` (the integer model equals TFLite on every layer of all
10,000 CIFAR-10 test images; the pass-level mapping is exact). This file is never edited; errata go elsewhere.

## 1. What is built

A descriptor-driven layer sequencer, `rtl/cim_gseq.sv`, in the endpoint behind a parameter `SEQ_NET2` (default 0). `cim_engine.sv` and
`cim_seq.sv` are not changed. With `SEQ_NET2 = 0` every existing bench, cycle count and bitstream-affecting behaviour is unchanged.

Model: MLCommons' published `pretrainedResnet_quant.tflite` (sha256 `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`), unchanged; parameters from
`model/resnet_int.py export`.

## 2. Design, fixed here

- **Descriptors**: up to 16 layer records in registers: type (conv, add, pool, fc), kernel 1 or 3, stride 1 or 2, input channels (3, 16, 32, 64),
  output channels, map size, engine tile base, row tiles R, column tiles C, input and output ACT buffers, zero points, clamp, parameter base.
- **Memories**: ACT, three 16-KiB banks of 64-bit words (8 channels per word, words in raster order, channel groups of 8); IMG 32 × 32 pixels of 3 bytes,
  written by the CPU; parameters, 10 layers of 64 entries (bias, multiplier, shift), the format of the DS-CNN's.
- **Gather**: one ACT word (or IMG pixel) per cycle into the input bank of the engine (2 banks). The pass starts 3 cycles after the last address, when
  the engine's 9-cycle interval has passed. The gather may start rewriting a bank 1 cycle after the pass that uses it started (the
  engine reads row group g at edge t0 + 2 + g; the gather writes groups in ascending order).
- **Row passes** of a column tile run in order and accumulate in the engine; only the last one's 16 results are requantised (the DS-CNN's
  srdhm and rdbp arithmetic, 2 lanes) and written to ACT as 8-channel words. Layers are separated by a barrier.
- **Residual add**: 2 elements per cycle in TFLite's exact integer arithmetic (both inputs rescaled with a left shift of 20, summed, rescaled).
- **Pool**: 8 channels per cycle; **FC**: one pass. The class is the index of the first maximum.
- Addresses (endpoint window): ACT debug reads, IMG writes and descriptors use free regions; the map of existing registers is unchanged.

## 3. Predictions (from `model/resnet_sched.py`, `model/resnet_prediction.json`)

Total 155,811 cycles per inference (go to done). Passes 15,617. Per operation:

| c0 | c1 | c2 | a3 | c4 | c5 | c6 | a7 | c8 | c9 | c10 | a11 | pool | fc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9,252 | 27,683 | 27,683 | 8,208 | 13,859 | 23,075 | 4,637 | 4,112 | 11,555 | 20,771 | 2,335 | 2,064 | 526 | 51 |

Tiles 86. Mapped ADD elements 28,672. Memory added by construction: 3 × 16 KiB ACT, 3 KiB IMG, parameter words.

## 4. Gates

| | Gate | Pass |
|---|---|---|
| N1 | Exactness, RTL | a bench compares every layer output of 20 images with `resnet_int.py`: 0 errors; the replay of all 10,000 CIFAR-10 test images gives the model's logits and argmax in every image: 0 errors |
| N2 | Cycles | the bench's go-to-done cycles ≤ 1.02 × 155,811 = 158,927, and within 2 % for every operation of 1,000 cycles or more; any difference from the prediction is reported |
| N3 | Existing designs | regression 53 of 53; with `SEQ_NET2 = 0` the endpoint's S and S+DWU cycles unchanged; `cim_engine.sv` and `cim_seq.sv` identical to commit `38e5d7d` |
| N4 | Size and timing | the board bitstream with S, the unit and `SEQ_NET2 = 1` fits the XC7A100T and meets timing of the system clock at 15.36 MHz (setup and hold ≥ 0) |
| N5 | Board | 20 images in the firmware: logits equal the model's, `go → interrupt` cycles equal the RTL's count plus an offset measured in system simulation of the same firmware |
| N6 | Baseline | TensorFlow Lite Micro of the same model on the CV32E20 of the same board (unverified result): the cycle ratio is reported, whatever it is |
| N7 | Broken variants | 4 mutants caught by the bench: residual operands swapped in scaling, the bank rewritten one cycle early, stride ignored in the gather, a row tile's accumulate flag dropped |

**If a gate fails:** N1, N2 (exactness), N3, N5 or N7 is a defect: it is fixed and rerun, with no threshold moved. N4 failing is reported, and the student
decides what to reduce. A cycle count above N2's limit is reported as a miss.

**Not claimed:** power (no estimate is made for this network), accuracy above the model's own, real-time use, or a comparison of speed with other chips. iMCU's
ResNetv1 numbers are a context, not a race.
