# System-level novelty boundary: the closest KWS works, read in full

Status (2026-11-08, night): **all six works of the student's plan (`HopCIM_Specific_Next_Steps_and_Professor_Advice.md`
§2, §23) read in full.** Works 1, 2, 4, 5, 6 from the PDFs in `Literature review/` (downloaded by the student through
NYCU's IEEE access); work 3 from the student's copy, read on 2026-10-26 (`NOVELTY_ASSESSMENT.md`). Work 1 was read in its
ISSCC 2020 version (3 pages), not the JSSC 2021 journal version. An earlier version of this file had only the published
abstracts of works 1, 4, 5, 6; §2 replaces it and corrects one point it got wrong (§4). The plan's search was an
IEEE-Xplore-targeted web search, not a logged-in manual Xplore search, so the thesis keeps "to our knowledge".

## 1. The six works (bibliographic data verified in Crossref)

| # | Work | Evaluated on |
|---|---|---|
| 1 | W. Shan, M. Yang, J. Xu, Y. Lu, S. Zhang, T. Wang, J. Yang, L. Shi, M. Seok, "A 510nW 0.41V low-memory low-computation keyword-spotting chip using serial FFT-based MFCC and binarized depthwise separable convolutional neural network in 28nm CMOS", ISSCC 2020, pp. 230-232, DOI 10.1109/ISSCC19947.2020.9063000 (journal version JSSC 56(1), 151-164, DOI 10.1109/JSSC.2020.3029097) | measured silicon, 28 nm |
| 2 | X. Jia, H. Zhu, Y. Wang, J. Zhang, F. Lin, X. Xiong, D. Xu, C. Chen, Q. Liu, "A 11.6μW computing-on-memory-boundary keyword spotting processor with joint MFCC-CNN ternary quantization", ISCAS 2022, DOI 10.1109/ISCAS48785.2022.9937206 | **simulation** of a 28-nm layout (power by Synopsys PTPX) |
| 3 | G. Kang, D. Park, H. Lee, S. Jung, J. Park, J. G. Min, Y. Lee, J. Kung, "RISC-V driven orchestration of vector processing units and eFlash compute-in-memory arrays for fast and accurate keyword spotting", ASP-DAC 2025, DOI 10.1145/3658617.3697697 | 65-nm place-and-route, RTL cycle counts |
| 4 | Z. Fu, K. Lin, X. Zhao, L. Ge, W. Ye, "A 95.3% 12-class, 108-nJ/inference keyword spotting chip with hybrid FFT-BFNet architecture and exponent-aware nonuniform quantization in 65-nm CMOS", IEEE TVLSI 34(2), 568-578, 2026, DOI 10.1109/TVLSI.2025.3632791 | measured silicon, 65 nm |
| 5 | D.-H. Lee, S.-U. Kang, J. Jang, S.-M. Yoon, H. Kim, M. Seok, M.-S. Choo, "KWS-CIM: a charge-domain eDRAM-based compute-in-memory macro for ternary DFSMN keyword spotting", IEEE TCAS-I 73(11), 7610-7623, 2026, DOI 10.1109/TCSI.2026.3717749 | measured macro, 65 nm; KWS accuracy partly in software (§2.5) |
| 6 | Y. Dong, X. Liu, K. Bai, G. Li, M. Wu, Y. Jing, Y. Zhang, P. Zhan, Y. Zhang, Y. Ma, R. Huang, T. Jia, L. Ye, "A heterogeneous TinyML SoC with systematic minimum energy searching and management for keyword spotting", IEEE JSSC 61(6), 3098-3112, 2026, DOI 10.1109/JSSC.2025.3619427 | measured silicon, 55 nm |

## 2. What each work builds (from the full texts)

### 2.1 Shan et al., ISSCC 2020 (work 1)

- **Chip:** 28 nm, 0.41 V, 40 kHz, 0.51 µW (MFCC 0.34 µW, network 0.17 µW), core 0.44 × 0.52 mm², 2 KB of memory,
  64-ms latency; measured (Fig. 14.1.5, 14.1.7).
- **Front-end (MFCC circuit, Fig. 14.1.2):** pre-emphasis, framing (32-ms frames, 16-ms step, 8-kHz sampling, clips cut
  to 500 ms), windowing, a 16-bit 256-point serial-parallel FFT (radix-2² single-path delay feedback), mel filters
  reduced from 20 MACs to 2 by exploiting their overlap, log2 by a look-up table, DCT, "hybrid precision quantization";
  10 coefficients of 8 bits per frame. A dedicated circuit, not a CIM; no claim of equality with a software MFCC.
- **Network:** a 4-layer binarized DSCNN (conv, depthwise-separable conv, pooling, FC), activations binarized to 0/1
  except the 8-bit input layer; 3,456 weights; 150,496 operations, reduced to 8,736 per frame by computing only the
  newest frame and storing the rest ("eliminate 71.4 % of the data storage and 94.3 % of computations without losing
  detection accuracy"). Dedicated FSM and two PEs, custom register-file memories, near-threshold design.
- **Task:** one or two keywords against fillers (ratio 6:1): about 98 % one-word, 94.6 % two-word on GSC.
- **No CPU on chip**; an FPGA drives the test board.

### 2.2 Jia et al., ISCAS 2022 (work 2)

- **Not measured silicon:** "a 28nm silicon prototype designed and simulated"; layout (Fig. 6), power from Synopsys
  PTPX. 11.6 µW at 0.72 V and 250 kHz, 64-ms latency.
- **Front-end:** framing, Hamming window, 256-point R2²SDF FFT, CORDIC logarithm. **The log is moved before the mel
  filter** ("swaps the logarithm module and the Mel filter") so that mel (129 × 64) and DCT (64 × 40) merge into one
  129 × 40 "MF-DCT" matrix. That matrix runs **on the memory-boundary array** with ternary coefficients and 8-bit
  inputs; FFT and log do not. Since log and mel do not commute, these are not standard MFCCs.
- **Network:** a custom CNN (CONV 8 × 6, CONV 7 × 6, max-pooling, FC 512 × 64, FC 64 × 10), ternary weights, 2-bit
  activations, trained jointly with the front-end by learned-step-size quantization; 10 words, 90.54 % (float 93.50 %).
- **"CIM":** the COMB macro, 96 b × 288 8T SRAM, selects **one column per cycle** into 48 ternary multipliers and an
  adder tree (16 macros); digital.
- **Control:** instruction memory, decoder and controller running pre-compiled instructions; no general CPU.

### 2.3 Kang et al., ASP-DAC 2025, "RV-Flash" (work 3; read 2026-10-26)

- 2 KB ADC-less binary eFlash CIM, an RV32I core with 11 custom instructions, an FP16 vector unit (16 multipliers) for
  depthwise and element-wise operations; 65-nm place-and-route and RTL cycle counts, not fabricated.
- A binarized, reparameterised FSMN (HRepFSMN), 96.5 % on GSC v2-12; the front-end is not addressed (the model starts
  from features). The CIM alone gives −7.7 to −11.4 % latency; with the vector unit −89.5 to −92.2 %.

### 2.4 Fu et al., TVLSI 2026 (work 4)

- **Chip:** TSMC 65 nm, 0.58-mm² core, 0.65 V, 600 kHz; measured 7.3 µW dynamic + 0.55 µW static, 108.04 nJ per
  inference, 14.82-ms latency (inside a 16-ms frame); 7.5 KB weight SRAM, 2.4 KB feature-map memory.
- **Front-end (Fig. 2):** pre-emphasis, framing (256 samples, 164-sample shift at 8 kHz), windowing, a 256-point FFT
  (an FFT128 core plus an FFT128-to-256 stage), the absolute values of the real and imaginary parts (a two's-complement
  converter), mel filtering as a sparse matrix-vector product, log2 by a look-up table; 16 features per frame, 48
  frames. **The datapath has no DCT stage** (Fig. 2; the text lists pre-emphasis through logarithm, "yielding 16-D
  MFCCs"), so the features are log-mel energies, computed from |Re| and |Im| rather than the squared magnitude.
- **Network:** BFNet: four blocks of 1-D depthwise conv (5 × 1) and **butterfly-structured pointwise conv** (N² → 2N
  log₂N weights), three max-pools, FC; distilled from a DSCNN teacher (96.5 % → BFNet 95.8 %), quantization-aware
  training, 6-bit exponent-aware nonuniform weights (95.29 %); 12 classes.
- **Sharing (Fig. 4, 6, Table I):** one dual-channel FFT128 core runs the FFT, the 1-D depthwise conv (as FIR filters)
  and the butterfly pointwise conv by switch settings; the mel filter and the FC layer share multipliers (two 16 × 16
  decomposed into eight 8 × 8); the FFT output buffer doubles as feature-map / residual memory. The shared part is about
  45 % of the chip area (a synthesis estimate).
- **Exactness:** "our quantization method on PyTorch is exactly the same as the actual inference of the chip, so the
  accuracy is consistent"; the chip's timing "matches the simulation exactly". Equality with **its own** quantized model.
- **I/O:** a Zynq-7020 FPGA feeds GSC clips over UART, or a PC microphone's audio through a Python interface. No CPU.

### 2.5 Lee et al., KWS-CIM, TCAS-I 2026 (work 5)

- **Macro:** 65 nm, 128 × 128 3T2C eDRAM, ternary inputs (+1, 0, −1), binary weights, charge-domain MAC, **1-bit
  sense-amplifier readout** (a conventional SA mode, or a two-step SA mode for accuracy); 16 calibration rows (112 × 128
  effective) and a column-offset table extracted before the run and kept on an **off-chip host**.
- **Model:** a CIM-aware ternary DFSMN (from DFSMN/BiFSMNv2), trained with the macro's row tiling and a MAC-output
  quantizer that models the 1-bit readout. The front-end (1-s audio at 16 kHz → 32-bin mel spectrogram), the embedding,
  Conv2D and keyword-spotter layers run outside the array; the CIM runs the backbone's convolutions (76 % of the MACs).
- **KWS accuracy is not end to end on the chip:** "the CIFAR-10 and KWS accuracies are evaluated by applying
  output-probability statistics measured from the fabricated CIM macro to the corresponding software inference flow";
  only MNIST is measured at the full-chip level (Table I, II). 91.66 / 89.87 / 87.30 % (12 / 20 / 35 classes) against
  92.48 / 90.98 / 88.11 % in software: 0.8 to 1.1 points lost.
- **Power:** 308 µW (SA) or 966 µW (two-step SA) at the operating point; 12.3 or 48.3 pJ per CIM access;
  930–1,335 / 82–336 TOPS/W.

### 2.6 Dong et al., JSSC 2026 (work 6), the closest system

- **SoC:** TSMC 55 nm, 9 mm², measured. An always-on Cortex-M0 (control), a Cortex-M4 (optional NLMS echo
  cancellation), an in-house DSP (FFT of 512 points, 40 mel filters, log, 20 DCT outputs, FIR; 16 kHz, 32-ms frames,
  16-ms shift), an in-house **DNN accelerator** (16-MAC PE array, DMA, SRAM; conv, depthwise conv, FC, add, pooling,
  ReLU), a **digital CIM used only for voice-activity detection** (256 × 128 6T SRAM, 16 1-bit inputs × 8-bit weights
  per cycle, 4.4 TOPS/W INT8; runs conv2 and FC1 of the VAD CNN), an amplitude/zero-crossing event detector, 96 KB
  global + 16 KB ultra-low-leakage SRAM, switched-capacitor regulators, in-block power switches, and synthesizable
  energy, event and latency monitors that drive a system-level minimum-energy-point search.
- **KWS:** MFCC on the DSP; "the neural network (NN) engine performs inference operations once 60 frames are
  accumulated"; per layer, weights and activations move by DMA between the global memory and the accelerator (about
  160 kB in total). **"The workload for KWS inference mode is the DS-CNN from the MLPerf Tiny benchmark"**, which "achieves
  over 93 % accuracy on a dataset based on the Google Speech Command Dataset v2, which contains 35 keywords" (also
  MicroNets' DSCNN-M and a custom DSCNN-S). MLPerf Tiny's reference model has 12 classes, so this is its architecture
  trained for 35 keywords; the paper says nothing of its quantisation or of equality with a reference. It runs on the
  DNN accelerator, not on the CIM.
- **Measured:** 3.5 µW standby, 86.4 µW in VAD mode, 66.8 mW in KWS mode at 1.2 V and 96 MHz; the system-level search
  saves 51 % of energy on average (28.8 % more than tuning the DNN accelerator alone); 1.7 mW at a 200-ms latency
  constraint; 31.1-ms minimum end-to-end latency.

## 3. HopCIM's candidate claims against the six (full texts)

✓ = the work does it; (✓) = in part, as noted.

| Property | 1 Shan | 2 Jia | 3 Kang | 4 Fu | 5 Lee | 6 Dong | HopCIM |
|---|---|---|---|---|---|---|---|
| Network of the KWS model on a CIM | | ✓ (column-serial digital) | ✓ (binary eFlash) | | ✓ (backbone, analog) | (CIM only for VAD) | ✓ (digital, all 7,501 passes) |
| Part of the MFCC on a CIM | | ✓ (mel + DCT merged, ternary) | | | | | ✓ |
| Standard MFCC including the FFT, computed exactly, on the CIM | | | | | | | ✓ |
| Front-end and network share compute hardware | | ✓ (same array) | | ✓ (FFT core, mel/FC multipliers, memory) | | | ✓ |
| A published network architecture unchanged | | | | | | ✓ (MLPerf Tiny DS-CNN, 35 keywords) | ✓ |
| The published **pretrained int8** model unchanged, bit-exact to its reference kernels | | | | | | | ✓ (TFLite reference, 4,890 clips) |
| Chip equal to its own quantized model | | | | ✓ (stated) | (no: 0.8–1.1 points lost) | | ✓ |
| RISC-V host | | | ✓ | | | (Arm Cortex-M0/M4) | ✓ |
| Accelerator sequences the network while the CPU waits | ✓ (no CPU) | ✓ (own controller) | (CPU orchestrates by custom instructions) | ✓ (no CPU) | | ✓ (DMA, accelerator) | ✓ (CPU asleep) |
| Real-time processing of an audio stream | (✓: 16-ms frames; clips from the test platform) | | | ✓ (GSC over UART, or a PC microphone via an FPGA) | | ✓ (audio over I2S from an FPGA) | ✓ (on-board PDM microphone, FPGA) |
| Measured energy | ✓ | (simulated) | (RTL) | ✓ | ✓ (macro) | ✓ | (estimate [E]; FPGA prototype) |

## 4. What this means for the claim

**The safer claim stands, with precise wording:** *the published, pretrained int8 MLPerf Tiny KWS model, unchanged and
bit-exact to TFLite's reference kernels, together with a standard MFCC computed exactly, both executed on one digital
CIM inside a RISC-V microcontroller whose CPU sleeps while a hardware sequencer runs the network; shown live on an
FPGA, with the bottleneck measured as it moves.* No work of the six computes a standard MFCC exactly on a CIM, and none
runs a published pretrained int8 model unchanged and bit-exact. Dong et al. (6) come closest as a system (MLPerf Tiny's
DS-CNN architecture in a measured KWS SoC with hardware MFCC and a CIM), but their network runs on a non-CIM accelerator
and nothing is said about exactness; Jia et al. (2) come closest for the front-end (mel and DCT on the array), but with a
reordered, ternary, jointly trained approximation and only in simulation.

**Correction of the abstract-based reading:** it said that none of the six keeps the model unchanged. Dong et al. do
run an unchanged published architecture (MLPerf Tiny's DS-CNN, retrained for 35 keywords). The distinction that holds is
the *pretrained int8 model* unchanged and *bit-exact*, *on the CIM*.

**The thesis must not claim:**
- sharing the datapath between front-end and network as new or as a saving: Fu et al. share an FFT core, multipliers
  and memory in measured silicon, Jia et al. run front-end matrices and network on one array, and HopCIM's own B3 found
  no hardware saving (`B3_BASELINE.md`). Fu et al. report about 45 % of the area shared; if both are cited, explain the
  difference: their network accelerator *is* the FFT core, so the saving is a whole second datapath, whereas HopCIM's
  CIM front-end costs about what a dedicated MFCC block costs;
- exactness as such: Fu et al. state that the chip equals its PyTorch quantized model. HopCIM's specific point is
  equality with an *external, unchanged* reference (TFLite's reference kernels on the published model);
- streaming or real-time KWS (1, 4, 6), RISC-V with CIM for KWS (3), a heterogeneous energy-managed KWS SoC or a CIM in
  a KWS SoC (6);
- anything about energy against these chips: 1, 4, 5, 6 are measured silicon; HopCIM's energies are model estimates
  [E] at 28 nm with an FPGA prototype.

**Citing work 5:** its KWS accuracies are software inference with the macro's measured output statistics, not
end-to-end chip runs; cite them as such.

## 5. Still to do

- The manual, logged-in IEEE Xplore search (`HOPCIM_PROPOSAL.md` §15.2) is still the student's.
- Work 1's JSSC 2021 version was not read; nothing above depends on what it adds.

## Sources

- The full texts in `Literature review/` (works 1, 2, 4, 5, 6, downloaded by the student 2026-11-08) and the student's
  copy of work 3
- Crossref records of the six DOIs (api.crossref.org/v1/works/<DOI>), retrieved 2026-11-08
- [EPFL record of Shan et al.](https://infoscience.epfl.ch/record/280400?ln=en)
