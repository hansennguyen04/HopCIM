# Literature review: the works around HopCIM, what they leave open, and how HopCIM compares

Draft for the thesis (2026-11-25). Sources and their read status are in §6. "HopCIM" is the default design S+DWU
(`DESIGNS.md`); where a result exists only for the baseline S, it says so. Statements that a work lacks something mean
*in the text read*, not that its authors could not do it.

## 1. Categories at a glance

| | Category | What these works achieve | What is missing in the works read | How HopCIM covers it |
|---|---|---|---|---|
| C1 | Keyword-spotting (KWS) chips with dedicated logic | sub-µW to µW power in measured silicon; dedicated MFCC circuits | the model is designed or retrained for the chip; nobody shows equality with a published model | runs the published, pretrained int8 DS-CNN unchanged and bit-exact to TensorFlow Lite's reference kernels |
| C2 | KWS on compute-in-memory (CIM) | 1-bit, ternary or analog arrays run the network | accuracy changes with the array; the front-end is learned, moved outside, or approximated | exact digital bit-serial CIM; the standard MFCC computed on it |
| C3 | Front-end: MFCC hardware and FFT on CIM | compact MFCC circuits; FFT on memory arrays | FFT-on-CIM works do not feed a KWS network; mel and DCT on an array appear only as a changed, ternary MFCC | the whole standard MFCC, in exact integers, on the CIM plus a small epilogue; equal to the golden model on 239,610 test frames |
| C4 | RISC-V systems with CIM | microcontrollers with CIM accelerators, TensorFlow Lite Micro flows, compilers | the CPU sets up or runs layers; no audio front-end; exactness against an outside reference not shown | a hardware sequencer runs all layers while the CPU sleeps (264 active cycles per decision); live audio on an FPGA |
| C5 | Depthwise convolution next to CIM | dedicated digital engines, feature-stationary or duplicated-kernel mappings, mixed macros | shown on MobileNet-type layers; alternatives not priced on one network under fixed rules | five options (the CIM itself, narrow macros, feature-stationary, swap-in, a digital unit) priced with one model and gates frozen in advance; the best one built and checked against that model |
| C6 | Mapping, scheduling and compilers for CIM | cross-layer scheduling, compilers, design-space exploration | results are mostly from simulators or a cost model; whole-system equality with RTL is rarely shown | the schedule model predicts the RTL exactly: every layer's cycles and every access count |
| C7 | Benchmarks and verification | MLPerf Tiny results in many classes | CIM KWS papers report their own datasets and flows; few use the public runner | MLCommons' own runner, unchanged (unverified); 4,890-clip bit-exact replay of the RTL; 53 regression benches; a live path from microphone to class |

Figure 1 (`figures/out/lit1_capability.*`) shows the same as a matrix of works and capabilities.

## 2. The categories in detail

### C1. KWS chips with dedicated logic
- Shan et al. (ISSCC 2020): 28 nm, 0.51 µW, serial FFT MFCC, binarised DSCNN, about 98 % (one word) and 94.6 % (two words) on GSC.
- Fu et al. (TVLSI 2026): 65 nm, 95.3 % on 12 classes, 108 nJ per inference, FFT core reused for the network; its
  accuracy equals its own quantised model.
- Dong et al. (JSSC 2026): 55 nm, MFCC on a DSP, MLPerf Tiny's DS-CNN architecture retrained for 35 keywords on a dedicated
  accelerator; the CIM is only for voice-activity detection.
- **Gap.** Their networks are designed or retrained for the hardware, and none claims equality with a published,
  pretrained model. *HopCIM keeps the model fixed, so its accuracy is the reference's own.*

### C2. KWS on CIM
- Jia et al. (ISCAS 2022, simulated): ternary network and a merged, reordered mel-DCT on the array; 90.54 % against 93.50 % float.
- Lee et al., KWS-CIM (TCAS-I 2026): analog eDRAM macro; 91.66 % against 92.48 % in software for 12 classes, KWS accuracy from
  software with the macro's measured statistics.
- AnalogNets (arXiv 2111.06503): PCM; **replaces depthwise layers by regular convolutions** and retrains; 95.6 % against 96.4 %.
- CIMR-V (ISCAS 2024, NYCU/ITRI), PSCNN, Chiang et al. (TVLSI 2022, NYCU): binary networks with learned raw-audio front-ends.
  RV-Flash (ASP-DAC 2025): binary eFlash CIM; the non-matrix operations dominate (the CIM alone gives −7.7 to −11.4 % latency).
- **Gap.** A CIM that changes the network or the front-end cannot show whether the *published* system works on it.
  *HopCIM's digital array computes the model's int8 arithmetic exactly.*

### C3. Front-ends: MFCC hardware and FFT on CIM
- Dedicated MFCC: serial-FFT circuits (Shan et al.; a 0.61-µW KWS ASIC, JSSC 2023, abstract only).
- FFT on CIM is established (CSIFA, SHIFFT, ReFFT, FourierPIM, analog FFT; proposal §4.2); Wang et al. (Science Advances 2025)
  put a 128-point DFT on a memristor SoC, with the magnitude on a PC and no mel, log or DCT.
- **Gap.** In the works read none computes the standard MFCC (FFT, power, mel, log, DCT) exactly on a CIM and feeds an
  unchanged network. *HopCIM does; FFT on a CIM alone is not its claim.*

### C4. RISC-V systems with CIM
- iMCU (Lin et al., JSSC 2024): 28-nm silicon, digital CIM, TensorFlow Lite Micro; ResNetv1 from MLPerf Tiny, and a 4-keyword model.
- DIANA (JSSC 2023) with HTVM (2024): RISC-V, digital and analog cores, 22 nm; HTVM **runs the MLPerf Tiny DS-CNN on it**, with
  the first filter changed to 7 × 5; depthwise on the digital core, which reaches 3.75 MAC per cycle on them.
- Garofalo et al. (JETCAS 2022), CIMR-V, RV-Flash: RISC-V plus CIM.
- **Gap.** The host drives layers or data movement; no front-end; no bit-exactness statement. *HopCIM's sequencer runs the
  network alone, the CPU sleeping, with the MFCC on the CIM.*

### C5. Depthwise convolution next to CIM
- On a digital unit: Garofalo et al. (29.7 MAC/cycle); DIANA; DARKSIDE (up to 30 MAC/cycle, 65 nm). **The split itself is theirs.**
- In memory: Wang et al. (ISSCC 2023), ConvDK (Song and Jeong 2025, analog, −77 to −87 % buffer traffic), MixCIM (JSSC 2026, eDRAM
  and SRAM arrays in one analog macro; 17.2 TOPS/W, 2.37 % below the floating-point software), MobiLattice (ICCAD 2020).
- Avoided: AnalogNets.
- **Gap.** Results are for MobileNet-type layers or macros; alternatives are not compared on one KWS network under common
  rules, and not exactly. *HopCIM prices five options with one model and frozen gates (Figure 3) and builds the chosen one;
  the RTL equals the model in every layer and counter.* **HopCIM does not claim the split, and its unit is slower than theirs
  (about 6.4 MAC per cycle, limited by feeding data).**

### C6. Mapping, scheduling and compilers
- CLSA-CIM (DATE 2024), CIM-MLC (ASPLOS 2024), CIMFlow, AccelCIM, CIM-Tuner, CMSwitch, CINM, HARP: mapping or design-space tools for
  CIM. HTVM and MATCH: compilers measured on chips. Sun, Houshmand and Verhelst (ICCAD 2023): IMC performance model.
- **Gap.** They optimise mapping for accelerators and mostly evaluate by model or simulation. *HopCIM is a small complete system
  whose cycle and access model is verified against its RTL.*

### C7. Benchmarks and verification
- MLPerf Tiny: 75 public KWS results (`MLPERF_TINY.md` §8). Of the CIM systems read, iMCU (ResNetv1), DIANA with HTVM (all four tasks) and Dong et al.
  (the DS-CNN architecture, retrained) use MLPerf Tiny models.
- **Gap.** CIM KWS works read report their own models and flows. *HopCIM runs MLCommons' public runner unchanged (unverified) and
  publishes exactness over the whole test set.*

## 3. Capability comparison

**Figure 1.** `figures/out/lit1_capability.pdf`. Filled: yes; half: partly; empty: no; dash: not stated. HopCIM is the
only row with a published pretrained model unchanged **and** bit-exact **and** a standard MFCC on the CIM. It is
**not** a measured chip and has **no** measured energy; several others do (last two columns).

Per-entry notes are in `figures/lit_data.py`. The columns were chosen to show HopCIM's question, so the figure must be read
with the last two columns, where HopCIM has nothing.

## 4. Results compared with previous works

| Work | Platform | Task and model | Accuracy | Against its own reference | Model kept as published? |
|---|---|---|---:|---|---|
| Shan 2020 | 28-nm chip | 1–2 keywords; custom binarised DSCNN | 94.6–98 % | not stated | no |
| Jia 2022 | 28-nm simulation | 10 words; custom ternary CNN | 90.54 % | −2.96 points (float 93.50 %) | no |
| Fu 2026 | 65-nm chip | 12 classes; BFNet | 95.3 % | 0 vs its own quantised model; −0.5 vs its float network (95.8 %) | no (distilled) |
| KWS-CIM 2026 | 65-nm macro | 12 classes; ternary DFSMN | 91.66 % † | −0.82 (software 92.48 %) | no |
| CIMR-V 2024 | 28-nm synthesis | 12 classes; binary 1-D CNN | 94.02 % | not stated | no |
| AnalogNets 2022 | PCM hardware, 14-nm model | 12 classes; redesigned CNN | 95.6 % | −0.8 (float 96.4 %) | no (depthwise replaced) |
| Dong 2026 | 55-nm chip | 35 words; MLPerf DS-CNN architecture | "over 93 %" | not stated | retrained |
| iMCU 2024 | 28-nm chip | 4 keywords; tiny-conv | 91.34 % | not stated | TFLM model |
| DIANA + HTVM 2023/24 | 22-nm chip | KWS DS-CNN, first filter 7 × 5 | not stated for KWS | ResNet: about −2 (ResNet20) and −5 (ResNet18) points in mixed mode | changed filter |
| MixCIM 2026 | 28-nm macro | CIFAR-10, MobileNet-V2 | 92.47 % | −2.37 (float) | retrained |
| **HopCIM** | **FPGA prototype (Artix-7)** | **12 classes; MLPerf Tiny DS-CNN int8** | **91.80 %** (4,890 clips, GSC v2) | **0.00: every logit equal to TFLite's reference kernels, in RTL** (MLCommons runner 90.1 %, unverified, every output equal to TFLM) | **yes** |

† software inference with the macro's measured statistics, not the chip end to end.

**How to read it.** The accuracies are **not** a ranking: tasks, class counts and reference points differ (floating point,
own quantised model, TFLite int8). HopCIM's reference is the published int8 model, so its zero is zero *loss of the
hardware*, not a better model: several works classify more accurately (95 % and above).

**Cycles per inference on the MLPerf Tiny KWS task.** `figures/out/lit2_cycles.pdf` (**Figure 2**). HopCIM's network runs in
67,856 cycles (S+DWU, board) and 77,600 under MLPerf's runner (S, unverified); DIANA's digital core needs 455,000 in
HTVM's measurement (1.75 ms at 260 MHz, changed first filter) and 1.89 M in MATCH's own HTVM run (7.26 ms), and TensorFlow Lite Micro
on the CV32E20 75.3 M (measured on HopCIM's board).
- **Cycles are not speed and not energy.** They favour a slow clock and wide hardware (HopCIM's endpoint uses 147 DSPs; the CPUs
  have one or two multipliers); the other clocks are 2 to 40 times HopCIM's. No measurement says what HopCIM would clock at
  in silicon.
- HopCIM's 67,856 cycles are from go to interrupt with the weights already in the array; DIANA's number includes the weight
  transfer. GAP9's is an MLPerf submission that HTVM's authors normalised to 260 MHz.
- \* unverified MLPerf result. † two published measurements of the same chip, 1.75 ms (changed first filter) and 7.26 ms.
- fpgaConvNet's design on a larger FPGA needs fewer cycles (32,000 to 46,000); HopCIM is programmable and runs the model unchanged.

**Depthwise options, one model [E].** `figures/out/lit3_depthwise.pdf` (**Figure 3**): five ways of running the depthwise layers
priced on the same network, plus the one built. All are HopCIM's own model, so the bars compare *ideas*, not chips:
keeping them on the CIM (44.11 µW at mode U), narrow macros (34.70), feature-stationary (28.48), the swap-in (28.13), the
built unit (27.96, RTL counters) and the 9-multiplier unit (27.72, study).
The unit is 0.6 % from the swap-in and 1.8 % from feature-stationary: they cannot be told apart by this model. The difference
that holds is against S (−36.6 %). S's and the unit's numbers come from RTL counters; the others from a model.

## 5. Where HopCIM is stronger, and where it is not

**Stronger, in the works read:**
1. A published pretrained model, unchanged and bit-exact to an outside reference, with the standard MFCC, on one digital CIM.
2. CPU-free network with live audio on a RISC-V microcontroller, with the CPU-active cycles measured.
3. A schedule and energy model that equals the RTL in cycles and counts, with rules frozen before the RTL.
4. Fewer cycles per inference on the KWS DS-CNN than DIANA (the CIM system with a published number) and than every CPU and MCU
   listed; a design generated for a larger FPGA (fpgaConvNet) needs fewer.

**Not stronger:**
- Power and energy are estimates [E], not measurements; measured chips exist and are lower (Shan: 0.51 µW).
- Accuracy is the reference model's (91.80 %); several works report more on other tasks or models.
- The depthwise split is Garofalo's and DIANA's idea, and DARKSIDE's and Garofalo's engines reach about 30 MAC per cycle.
- No silicon, no process of its own; the clock is an FPGA choice (15.36 MHz, real time down to 4.096 MHz in RTL).

## 6. Sources, read status, and limits

Read in full text at the relevant places: Shan, Jia, Kang (RV-Flash), Fu, Lee (KWS-CIM), Dong, CIMR-V, Garofalo, iMCU, DIANA, MixCIM,
MobiLattice, ConvDK (text), AnalogNets, DARKSIDE, HTVM, MATCH, Ottavi, Wang (ISSCC 2023). C1 to C6 use notes in
`NOVELTY_SYSTEM.md` (the six works, 2026-11-13), `HOPCIM_PROPOSAL.md` §4 (earlier screening, some abstract-only and marked
there) and `STREAMFUSE_NOVELTY.md`. Compilers of C6: title and abstract only, except HTVM and MATCH.
- The IEEE Xplore manual search is still open (proposal §15.2).
- Not read yet: "A 0.61-µW Fully Integrated Keyword-Spotting ASIC With Real-Point Serial FFT-Based MFCC and Temporal Depthwise Separable
  CNN" (IEEE JSSC 2023, DOI 10.1109/JSSC.2023.3339528); Chih et al. (ISSCC 2021) and Colonnade (JSSC 2021) (`TIMING_SENSITIVITY.md`).
- MLPerf: HopCIM's results are *not verified by MLCommons Association*; unverified results have not been through an MLPerf
  review and may use methodologies or implementations that are inconsistent with the MLPerf specification for verified results.
