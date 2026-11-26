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

## 3a. Comparison table for the paper: platform, front-end, model, accuracy (2026-11-14)

Every entry re-checked against the full texts on 2026-11-14 (pdftotext of the PDFs in `Literature review/`; work 3 from
`research_discovery/20261026_application_driven/Novelty affection/`). Added as row 7: CIMR-V, the group's own RISC-V +
SRAM-CIM KWS work (Guo, Chang et al., arXiv 2503.22072), which the advisor will compare against (proposal §15.3); it is
not one of the plan's six.

| # | Work | Evaluated as | Front-end, and where it runs | Network and precision | What runs the network | Host | Published model kept unchanged? | Task and accuracy (how obtained) |
|---|---|---|---|---|---|---|---|---|
| 1 | Shan et al., ISSCC 2020 | measured chip, 28 nm, 0.41 V, 40 kHz | MFCC on a dedicated circuit: serial 256-point FFT, mel reduced to 2 MACs, log LUT, DCT; 10 × 8-bit coefficients per frame | custom 4-layer binarized DS-CNN, 3,456 weights; 1-bit activations, 8-bit input layer | two dedicated PEs and an FSM; no CIM | none (an FPGA drives the test board) | no: a model designed and trained for the chip | 1 or 2 keywords against fillers (6 : 1), GSC: about 98 % one-word, 94.6 % two-word (chip, Fig. 14.1.5) |
| 2 | Jia et al., ISCAS 2022 | 28-nm layout, **simulated** (PTPX), 0.72 V, 250 kHz | 256-point FFT and CORDIC log in logic; **log moved before mel**, mel and DCT merged into one ternary 129 × 40 matrix **on the array**: not the standard MFCC | custom CNN (2 conv, max-pool, 2 FC); ternary weights, 2-bit activations | digital memory-boundary macro (COMB), one column per cycle | own controller, pre-compiled instructions | no: front-end and network trained jointly | 10 keywords: 90.54 % (float 93.50 %) (simulation) |
| 3 | Kang et al., ASP-DAC 2025 (RV-Flash) | 65-nm place-and-route, RTL cycle counts; 150 MHz | not addressed (the model starts from features) | HRepFSMN (reparameterised FSMN): binary convolutions, FP16 element-wise | 2-KB binary eFlash CIM + FP16 vector unit | RV32I, 11 custom instructions, orchestrates | no: a new model | GSC v2-12: 96.52 % (v1-12: 96.46 %) (the model in software; no chip) |
| 4 | Fu et al., TVLSI 2026 | measured chip, 65 nm, 0.65 V, 600 kHz | dedicated: 256-point FFT, \|Re\| + \|Im\|, mel as a sparse product, log2 LUT, **no DCT**: 16 log-mel features | BFNet (1-D depthwise + butterfly pointwise), 6-bit exponent-aware nonuniform weights | the FFT core, reused for the depthwise and butterfly layers; no CIM | none (an FPGA feeds the clips) | no: distilled from a DS-CNN teacher (96.5 → 95.8 → 95.29 %); the chip equals its own quantised model | 12 classes: 95.3 % (chip) |
| 5 | Lee et al., TCAS-I 2026 (KWS-CIM) | measured macro, 65 nm, up to 250 MHz | mel spectrogram outside the array (software) | CIM-aware ternary DFSMN; embedding, Conv2D and classifier outside the array | charge-domain eDRAM CIM: ternary inputs, binary weights, 1-bit sense-amplifier readout | off-chip host (calibration table) | no: trained for the macro | 12 / 20 / 35 classes: 91.66 / 89.87 / 87.30 % (software 92.48 / 90.98 / 88.11 %) (**software inference with the macro's measured output statistics**, not end to end) |
| 6 | Dong et al., JSSC 2026 | measured SoC, 55 nm; 96 MHz in KWS mode | MFCC on an in-house DSP (512-point FFT, 40 mel, log, 20 DCT), 12 MHz | **MLPerf Tiny's DS-CNN architecture** (also MicroNets' DSCNN-M, a custom DSCNN-S); quantisation not stated | in-house 16-MAC DNN accelerator, per-layer DMA; the digital CIM runs only the voice-activity detector | Cortex-M0 (control), Cortex-M4 | the architecture, retrained for 35 keywords; no equality with a reference stated | 35 keywords, GSC v2: "over 93 %" (how measured not stated per model) |
| 7 | Guo, Chang et al., CIMR-V (NYCU/ITRI), arXiv 2503.22072 | synthesized (Design Compiler), simulated, 28 nm, 0.9 V, 50 MHz | raw audio: high-pass filter, batch norm, quantise; no MFCC | 1-D binary CNN (5 × conv/max-pool on the CIM, weight fusion) | 10T SRAM CIM, 1-bit inputs, weights and outputs | Ibex RISC-V (PULPissimo) with CIM instructions | no: a model designed for the CIM | GSC, 12 classes: 94.02 % (the paper's flow; no chip) |
| | **HopCIM** | **FPGA prototype** (Nexys A7-100T), 15.36 MHz (real time down to 4.096 MHz); board = RTL cycle for cycle | **the standard MFCC** (512-point FFT in four steps, power, mel, log, DCT) in exact integer arithmetic **on the CIM** plus a hardware epilogue; equal to its golden model on all 239,610 test frames | **MLPerf Tiny's DS-CNN, int8** | digital bit-serial CIM, 128 tiles (emulated), and a hardware layer sequencer | X-HEEP CV32E20 (RISC-V), asleep while the network runs | **yes: the published pretrained int8 model, unchanged, bit-exact to TFLite's reference kernels** (4,890 clips; 1,000 of 1,000 under MLPerf's runner) | 12 classes, GSC v2 test set: 91.80 % (the same model on float MFCC: 91.66 %); MLPerf Tiny's runner: 90.1 % closed, 90.5 % open (**unverified**) (board) |

How to read it:
- **The accuracies are not a ranking.** The tasks differ (1–2 keywords against fillers, 10, 12, 35 classes), and so do
  the models and the ways the numbers were obtained (chip, simulation, software with measured statistics). HopCIM's
  accuracy is not a design result at all: it is the reference model's own, reproduced exactly; a better model would
  run on the same hardware.
- **"Model kept unchanged" is the column only HopCIM fills with a published pretrained model.** Works 1–5 and 7 design
  or retrain their model for their hardware (which is how they reach 1-bit, ternary or 6-bit arithmetic); work 6 keeps
  MLPerf Tiny's architecture but retrains it for 35 keywords.
- **Energy is left out on purpose.** Works 1, 4, 5 and 6 report measured silicon; HopCIM's energies are model estimates
  [E] for 28 nm (`ENERGY_ESTIMATE.md`), and a column would invite a comparison the evidence does not support.

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

## 4a. Works close to the digital depthwise unit, and one system work the list missed (2026-11-25)

Read status: **F** full text at hand, searched for depthwise, KWS, MFCC and exactness and read at those places; **A** abstract or
record page only. Numbers are the papers' own.

| Work | Read | What it does with depthwise layers | Against HopCIM (S+DWU) |
|---|---|---|---|
| Garofalo et al., "A Heterogeneous In-Memory Computing Cluster for Flexible End-to-End Inference of Real-World Deep Neural Networks", IEEE JETCAS 12(2), 422–435, 2022 (arXiv 2201.01089) | F | **The same split**: pointwise layers on an analog PCM array (256 × 256), depthwise on a dedicated digital accelerator (36 multipliers, weights from memory, 29.7 MAC/cycle average), 8 RISC-V cores. 22-nm place and route, array model from 14-nm prototypes; MobileNetV2's bottleneck layer: 11.5 × the cores' performance, 2.6 × their software depthwise | **The idea is theirs** (their earlier work, Ottavi et al., arXiv 2109.01404, ran the depthwise layers on the cores). HopCIM: exact digital CIM, a KWS network, 12.4 kGE priced against three alternatives, 4,890 clips exact, FPGA |
| Lin et al., "iMCU: A 28-nm Digital In-Memory Computing-Based Microcontroller Unit for TinyML", IEEE JSSC 59(8), 2684–2693, 2024 (accepted manuscript read) | F | Depthwise is not mentioned | **Measured silicon**: RISC-V, 32 KB digital IMC, TFLite-micro; ResNetv1 from MLPerf-Tiny (86.96 % on CIFAR-10, 60.9 ms, 102.18 µJ at 0.7 V) and tiny-conv for **4 keywords** (91.34 %). No DS-CNN, no MFCC, no bit-exactness statement |
| Zhou et al., "AnalogNets: ML-HW Co-Design of Noise-Robust TinyML Models and Always-On Analog Compute-in-Memory Accelerator" (arXiv 2111.06503; IEEE Micro 2022) | F | **Replaces** depthwise-separable blocks by regular 3 × 3 convolutions and retrains: analog CIM wastes cells on depthwise (local array utilisation as low as 0.9 %) | Keeps the published model instead of redesigning it. AnalogNets is better on accuracy (95.6 % on 12 keywords, PCM, 8-bit) and efficiency (8.2 µJ per inference, its own estimate); not exact |
| Song and Jeong, "Computing-In-Memory Dataflow for Minimal Buffer Traffic" (ConvDK; arXiv 2508.14375, 2025) | F | Keeps depthwise on an analog CIM by duplicating kernels: 77.4–87.0 % less buffer traffic, 15.6–27.8 % less latency on the depthwise layers; simulated, 28 nm | Different answer to the same problem. No accuracy result, depthwise layers only, no system |
| Wang et al., ISSCC 2023, 7.5 | F | Feature-stationary analog macro for depthwise (FS in `DWSWAP_STUDY.md`) | Priced as FS: 28.48 µW at U against 27.96 |
| Qiao et al., "MixCIM: A Hybrid Computing-in-Memory Macro With Less Data-Movement and Better Memory-Reuse for Depthwise Separable Neural Networks", IEEE JSSC 61(7), 3801–, 2026, DOI 10.1109/JSSC.2026.3651775 | F | Both halves **inside one macro**: depthwise on an eDRAM array (activation-stationary, sliding in place, 6-bit ADC), pointwise on an SRAM array (8-bit ADC); 28-nm measured macro, 15.31 TOPS/W depthwise, 40.99 pointwise, 17.2 for both, 8 bits; MobileNet-V2, 92.47 % on CIFAR-10, 69.53 % on ImageNet | Analog, retrained (QAT with the circuit's quantisation) and 2.37 % below the floating-point software model; a macro, with a Zynq supplying data: no processor, no KWS, no MFCC. HopCIM: exact, full system. MixCIM: measured silicon |
| Houshmand et al., "DIANA: An End-to-End Hybrid DIgital and ANAlog Neural Network SoC for the Edge", IEEE JSSC 58(1), 203–215, 2023, run with HTVM (Van Delm et al., arXiv 2406.07453) and MATCH (Hamdi et al., TCAD 2025) | F | **The same split, in silicon**: 22-nm RISC-V SoC with a digital core ("pointwise layers, depthwise layers, and FC layers") and an analog core. HTVM **runs MLPerf Tiny's KWS DS-CNN** on it (1.75 ms at 260 MHz on the digital core, first filter changed to 7 × 5; depthwise uses one PE row, at most 3.75 MAC/cycle); MATCH reports HTVM at 7.26 ms for it | KWS on a RISC-V chip with a CIM is therefore **not HopCIM's claim**. HopCIM: the pretrained model unchanged and bit-exact, with the MFCC on the CIM, CPU asleep; no accuracy is stated for DIANA's DS-CNN. DIANA: measured |
| Garofalo et al., "DARKSIDE: A Heterogeneous RISC-V Compute Cluster for Extreme-Edge On-Chip DNN Inference and Training", arXiv 2303.17954 (journal version to check) | F | Digital depthwise engine (up to 30 MAC/cycle) in an 8-core RISC-V cluster, measured 65-nm silicon | No CIM and no KWS; the same kind of engine as HopCIM's unit, with wider throughput |
| Zheng et al., "MobiLattice: A Depth-wise DCNN Accelerator with Hybrid Digital/Analog Nonvolatile Processing-In-Memory Block", ICCAD 2020 (DOI 10.1145/3400302.3415666) | F | Depthwise on the ReRAM crossbars in a digital mode; simulated, Design Compiler 40 nm scaled to 32 nm; 2.2–3.2 × ISAAC on MobileNet, MobileNetV2, Xception | Layers and models without accuracy or a system; HopCIM is exact, KWS, and built. MobiLattice has far more compute |

**What follows:**
- **The unit's idea must not be claimed.** Offloading depthwise layers to a digital unit beside a CIM is published
  (Garofalo et al.), and DIANA does it in silicon with a RISC-V host; a dedicated digital depthwise engine in a RISC-V
  cluster is too (DARKSIDE). MixCIM keeps both convolutions in memory instead.
- **What can be claimed:** on this exact, published KWS network, four ways of running the depthwise layers (swap-in,
  feature-stationary, large digital, this unit) priced with one model and frozen gates, the chosen one built, and its
  RTL equal to the model in cycles and counts and exact on 4,890 clips.
- **Correction (2026-11-25).** An earlier note said none of the hybrid papers does KWS; HTVM runs the DS-CNN on DIANA.
- **§4 is too wide on one point.** It did not know iMCU, which runs a published MLPerf-Tiny model on a digital CIM in a
  RISC-V microcontroller, in silicon. The claim that holds is the *keyword-spotting* model: the pretrained int8 DS-CNN,
  bit-exact, with a standard MFCC on the CIM. Say "KWS" in it.
- **Not better, and not to be implied:** accuracy (it is the reference model's), power (an estimate against measured
  silicon), speed.


## 5. Still to do

- The manual, logged-in IEEE Xplore search (`HOPCIM_PROPOSAL.md` §15.2) is still the student's.
- Work 1's JSSC 2021 version was not read; nothing above depends on what it adds.

## Sources

- The full texts in `Literature review/` (works 1, 2, 4, 5, 6, downloaded by the student 2026-11-08) and the student's
  copy of work 3
- Crossref records of the six DOIs (api.crossref.org/v1/works/<DOI>), retrieved 2026-11-08
- [EPFL record of Shan et al.](https://infoscience.epfl.ch/record/280400?ln=en)
