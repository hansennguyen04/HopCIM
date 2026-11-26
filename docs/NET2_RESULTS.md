# The second network (ResNet8): the RTL against its frozen gates

Built 2026-11-25. Gates: [`NET2_GATES.md`](NET2_GATES.md), frozen at commit `b610c36` before any RTL, SHA-256 `bc5de064…d4c757`
(`NET2_GATES.sha256`), unchanged. Prediction: `model/resnet_sched.py`, `model/resnet_prediction.json`.

What it shows: MLPerf Tiny's published int8 ResNet8, unchanged, runs bit-exact on the same CIM engine as the DS-CNN, through a second,
descriptor-driven sequencer. It is a generality result for the engine, the arithmetic and the method; it makes **no power, speed or accuracy claim**
(the accuracy, 87.07 % on the 10,000 test images, is the published model's own).

## Verdict

| Gate | Result | |
|---|---|---|
| N1 exact in RTL | every operation of 20 images equal to `model/resnet_int.py`, 0 errors; all **10,000** CIFAR-10 test images through the RTL, 0 wrong logits, classes or lost results | pass |
| N2 cycles | **155,814** against 155,811 predicted; limit 158,927; each operation within 0.1 % except the two smallest (see below) | pass |
| N3 existing designs | regression 54 of 54 (53 existing and the new bench); `cim_engine.sv` and `cim_seq.sv` byte-identical to `38e5d7d`; S+DWU firmware exact on the same chip with the second network present (60 runs) | pass |
| N4 size and timing | board bitstream (S, the unit, the second network, 256 KiB of CPU RAM): 50,946 LUT (80.4 %), 131.5 of 135 block-RAM tiles (97.4 %); system clock at 15.36 MHz: setup slack +15.95 ns, hold +0.229 ns, no failing path | pass |
| N5 board | run 2026-11-26: 20 of 20 images with the model's logits and class; go → interrupt **155,882** cycles in every image = RTL 155,814 + the 68 measured in system simulation | pass |
| N6 CPU baseline | run 2026-11-26 (unverified): TensorFlow Lite Micro's `Invoke` **227,387,324** cycles per inference on the same board, bitstream and clock; HopCIM's network 155,882: **1,459 ×** fewer cycles (about 1,303 × with the input write) | reported |
| N7 broken variants | 4 of 4 caught, after the bench was extended (2026-11-26) with a check of the gate's rewrite margin; the first run caught 3 (below) | pass, on the second run |

## N2: cycles, operation by operation

| | c0 | c1 | c2 | a3 | c4 | c5 | c6 | a7 | c8 | c9 | c10 | a11 | pool | fc | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| predicted | 9,252 | 27,683 | 27,683 | 8,208 | 13,859 | 23,075 | 4,637 | 4,112 | 11,555 | 20,771 | 2,335 | 2,064 | 526 | 51 | 155,811 |
| RTL | 9,262 | 27,684 | 27,684 | 8,206 | 13,860 | 23,076 | 4,638 | 4,110 | 11,556 | 20,772 | 2,336 | 2,062 | 526 | 42 | 155,814 |

The RTL values are increments of successive runs; each run adds 10 cycles for fetching the first layer record, which the first operation
shows. The FC's 42 against 51 is a conservative allowance in the prediction.

## N4: size and timing

Out of context (Vivado 2024.1, xc7a100t, 15 MHz), the endpoint with the unit: +4,992 LUT, +3,800 FF, +16 block-RAM tiles and +20 DSPs
against the unit alone (`SEQ_NET2 = 0`); `cim_gseq` itself 4,888 LUT, 3,806 FF, 13 RAMB36 + 6 RAMB18, 20 DSP; slack +42.0 ns.

Board bitstream `xheep_nexys_bscan_cim128knet2.bit` (`xheep/build.sh <tree> net2-256`, 35 min, sha256 `b72d12421c64ecbe…`):
50,946 LUT (80.4 %), 45,075 registers, **131.5 of 135 block-RAM tiles (97.4 %)** with the 256-KiB CPU memory (the 128-KiB configuration would use
about 99.5). System clock (15.36 MHz): setup slack +15.95 ns, hold +0.229 ns, no failing endpoint. Vivado's design-wide summary is not met: −12.322 ns
setup (64 endpoints) and −0.868 ns hold (9 endpoints), all on the crossing to X-HEEP's stock, unused SPI-slave clock
(S and the unit's bitstreams have the setup part of it; the hold part is new here, on the same crossing).

## N7: broken variants (`sim/run_net2_mutants.py`)

| Variant | First run (2026-11-25) | Second run (2026-11-26), bench with the margin check |
|---|---|---|
| Residual operands' multipliers swapped | caught (41,192 errors) | caught (41,192) |
| Input bank rewritten one cycle early | **not caught** | **caught**: 0 output errors, 7,016,176 margin violations |
| Stride ignored in the gather | caught | caught (38,640) |
| Accumulate flag dropped | caught | caught (85,169) |
| *(extra)* Bank released when its vector is complete, before its pass starts | caught (the bench hangs) | caught (hangs) |
| *(extra)* Zero point not subtracted | caught | caught (87,413) |
| *(extra)* RELU clamp ignored | not caught | not caught |

**Why the first run missed one, and what changed.** The frozen design lets the gather rewrite a bank 1 cycle after the pass that uses
it started (`NET2_GATES.md`, Gather): row group g is written one edge after the engine reads it (edge t0 + 2 + g). The variant moves
the write to the same edge as the read. The engine's input register file is LUT-RAM with an asynchronous read, registered at that edge,
so it still gets the old value, and no output changes: the variant is equivalent at the outputs, and the bench, which compared only outputs,
could not see it. The bench now also checks the margin itself (`tb/tb_cim_net2.sv`): at every edge, no lane may write the
entry {bank, row group} that the engine's stage 1 reads at that edge. The frozen design passes it: 20 images, every operation, 0 errors
and 0 violations (`sim/net2_jobs/margin_tb20.out`). In the second run that check alone catches the variant (its outputs are still all
correct), and it reports 0 violations in the variants caught by their outputs. The first run's output is kept
(`sim/net2_jobs/mutants_run1_20261125.out`).

This is a bench fix after a failing run, as the gate's rule for a failing N7 allows ("fixed and rerun"). No threshold moved, and the design was
not changed. What the check guards is the documented margin. In this RTL, losing it changes no output, so the check is a guard against a later
change (for example a registered-read input memory), not evidence of a fault that would show today.

The RELU-clamp variant (not in the gate) stays equivalent: in this network every RELU output zero point is −128, so the clamp is the int8 floor.

## N5: what the board run should show

Firmware `hopcim_resnet` (20 images embedded, 256-KiB configuration; board ELF `hopcim_resnet_nexys_15m36.elf`, sha256 `516701d952c5489f…`). In the full-system simulation (Verilator, same firmware source, 100 MHz UART settings):
20 of 20 images exact, go → interrupt **155,882** cycles (155,814 + 68), CPU-active 18,688 cycles per image
(1,024 pixel writes), setup 2,147,154 cycles once. The board's ELF is built with `-DREFERENCE_CLOCK_Hz=15360000`; its wake-up code is instruction for instruction the simulated one's. **Expected on the board: 20 images bit-exact, 155,882 cycles.**

**Board result (2026-11-26, `hopcim_resnet_nexys_15m36_board_20261126_182659.uart.log`, run by the student):** 20 of 20 images `logits exact`,
go → interrupt 155,882..155,882 cycles, CPU-active 18,688 per image, golden predictions = label in 18 of 20; `HopCIM resnet: 20 images bit-exact -> PASS`.
The log is line for line the full-system simulation's (`sim/net2_jobs/soc_resnet.uart.log`) except the one-time setup: 2,147,162 cycles on the board
against 2,147,154 in simulation (+8; setup is in no gate, cause not investigated). `board_run.sh` printed "timeout" because this firmware ends with the
PASS line, not with the `done` line the script looks for; all output had arrived.

## N6: the CPU-only baseline (TensorFlow Lite Micro on the CV32E20; unverified)

The gate asks for the cycle ratio of TensorFlow Lite Micro running the same model on the CV32E20 of the same board, "whatever it is".

| Item | Value |
|---|---|
| Runtime | TensorFlow Lite Micro at commit `c0ecf6ff`, the one the KWS baseline uses (`MLPERF_TINY.md` §9); its reference kernels (no optimised set exists for this core); 34 sources vendored unchanged by `mlperf/tflm_vendor.py`: the KWS baseline's without the depthwise kernel, with the ADD kernel (3 more headers; the KWS baseline's 37 files came out byte-identical) |
| Model, operators | `pretrainedResnet_quant.tflite` byte for byte (`sw/applications/tflm_resnet/resnet_model_data.cpp`, sha256 `3c002613…`); the reference submission's operators in its order (`benchmark/reference_submissions/image_classification/submitter_implemented.cpp`) without `DepthwiseConv2D`, which the model does not use and whose code would not fit |
| Build | the KWS baseline's flags (release build, `-O2`); `mlperf/build_tflm_resnet.sh` |
| Memory | X-HEEP's 256 KiB with the code section enlarged to 180 KiB (data 76 KiB): patch 0008, `configs/kws_nexys_256k_c180.py`. The generated files differ from `kws_nexys_256k.py`'s only in the two linker scripts (all 95 others, the hardware among them, identical), so **the run uses HopCIM's own ResNet8 bitstream**, `xheep_nexys_bscan_cim128knet2.bit` (the endpoint is in it, unused). Code and read-only data 182,020 of 184,320 bytes; data 75,152 of 77,824 |
| Arena | 54,364 bytes used of 57,344 (the reference reserves 100 KiB) |
| Images | CIFAR-10 test images 0 and 1 (the first two of `hopcim_resnet`'s 20), in the data section; only two fit. The outputs are compared with TFLite Micro's on the host at the same commit (`mlperf/gen_tflm_resnet.py`), and the class with HopCIM's golden class. The softmax saturates (most outputs −128, the winner 127), so the output check is a weak one; the cycle count is the result |
| Firmware | `sw/applications/tflm_resnet`; board ELF `tflm_resnet_nexys_15m36.elf` (sha256 `02453015ebd11e77…`); per image: the 3,072 input bytes copied into the input tensor (timed), one `Invoke` (timed), mcycle |

**A link problem met on the way, and its fix.** With a 184-KiB code section the firmware did not link: GNU ld 2.35 relaxed accesses to small
read-only constants at the end of the code section to global-pointer-relative ones and then found them out of range ("relocation truncated
to fit: R_RISCV_GPREL_I"). The constants were 6.6 KB below the global pointer, outside its ±2 KiB reach; this ld widens its test by the data
segment's page alignment (`DATA_SEGMENT_ALIGN` in X-HEEP's linker script). The KWS baseline never met it: its constants are 16 KB away. The 180-KiB
split with the images in the data section puts them about 11 KB away and links. Relaxation stays on, as in every HopCIM build.

**The simulated firmware is the board's.** The Verilator build (`SIM=1 mlperf/build_tflm_resnet.sh`, X-HEEP's simulation clock and UART settings) and
the board ELF have the same instruction sequence in every function except the start-up and UART-write routines (one instruction each); with numbers
masked, the 26,190 disassembled lines are identical. Code addresses differ by 4 bytes, so the word alignment of every instruction, which the
fetch depends on, is the same.

**Expected on the board (written before the run, 2026-11-26):** setup with "arena 54364 of 57344 bytes used" (as in simulation; setup
689,326 cycles there); image 0 class 3 and image 1 class 8, each equal to the label and to HopCIM's class, outputs exact;
`TFLM resnet: 2 images exact -> PASS`. No Invoke cycle count is predicted: the gate reports the ratio whatever it is (a rough
estimate from 12.5 M multiply-accumulates at the KWS baseline's 20 to 30 cycles each: 250 M to 375 M cycles, 16 to 25 s per image).
The full-system simulation of image 0 (about 18,000 cycles/s, so about 4 hours) is the cross-check: same instruction stream, so the same count is expected.

**Board result (2026-11-26, `tflm_resnet_nexys_15m36_board_20261126_192214.uart.log`, run by the student; unverified):**

| | TensorFlow Lite Micro on the CV32E20 | HopCIM (N5, same bitstream) | HopCIM is |
|---|---:|---:|---:|
| Network: `Invoke` / go → interrupt | **227,387,324** cycles (mean of 227,382,204 and 227,392,444) | **155,882** | **1,459 × fewer** |
| Per inference with the input write | 227,418,113 (input copy 30,789) | about 174,570 (CPU-active 18,688, which holds the 1,024 pixel writes, plus the network) | about 1,303 × fewer |
| At 15.36 MHz | 14.80 s | 10.15 ms | |

- Both images: outputs equal to TFLite Micro's on the host, class = label = HopCIM's class; 0 failed invokes; setup 689,326 cycles and arena
  54,364 bytes, both as in simulation. The two images differ by 10,240 cycles (0.0045 %).
- For comparison, the KWS network's ratio is 1,108 × (75.24 M against 67,908, `MLPERF_TINY.md` §9). ResNet8's is larger: it is mostly 3 × 3
  convolutions, which HopCIM maps onto the CIM while TFLite Micro's reference kernel runs them multiply by multiply on the CPU.

**How to read it.** The CPU side is TensorFlow Lite Micro's *reference* kernels, the only ones it has for this core (its optimised sets are CMSIS-NN,
Xtensa, CEVA, ARC); an optimised RISC-V kernel would narrow the ratio. This firmware is not under MLPerf Tiny's runner, so it is not an MLPerf
performance result; both numbers are unverified (*Result not verified by MLCommons Association*). As the gate says, there is no power, accuracy or
cross-chip claim: this is a cycle ratio on one board.

## Notes
- The bench runs the network up to each operation and compares what it wrote (`tb/tb_cim_net2.sv`); all 14 operations are checked, the pool and the
  FC through registers.
- With `SEQ_NET2 = 0` nothing is instantiated; the endpoint's existing behaviour is unchanged (N3).
- The whole system on the FPGA keeps all of S's and the unit's logic; the second network adds memories for activations, images and
  parameters of its own. ACT of ResNet8 is 48 KiB against the DS-CNN's 16 KiB.

## Reproduce

```
model/resnet_int.py check 10000         # integer model against TFLite on all images
python model/gen_net2_vectors.py 20     # vectors (bench gate uses 20; tb/net2_vec in the repository holds 4)
sim/run_net2_tb.sh 20 0 ; sim/run_net2_all.sh 10000 16 ; python3 sim/run_net2_mutants.py
sim/run_all.sh                           # 54 benches
vivado -mode batch -source syn/ooc.tcl -tclargs cim_endpoint FE_DED=0 PDM_CLKDIV=15 SEQ_S=1 SEQ_DWU=4 SEQ_NET2=1
```
