# HopCIM novelty assessment against the overlapping papers

Date: 2026-10-26 · Inputs: the seven PDFs in `Novelty affection/`, plus targeted searches · Evidence labels as in `RESEARCH_DISCOVERY.md` ([M] measured or computed here, [P] published, [A] assumption, [I] inference, [H] hypothesis).

## 0. Bottom line

- **HopCIM as originally framed is not novel.** Each of its pillars already has at least one published instance: a learned front-end on the CIM, streaming CIM inference, an autonomous controller, and a RISC-V host. Two of the seven papers (CIMR-V, PSCNN) come from the same NYCU group.
- **Real gaps remain.** Every paper either keeps MFCC outside the CIM or *replaces* it with a model-specific front-end (a binary or sinc convolution on raw audio, or a distilled encoder). None computes the **standard MFCC on the CIM**. None uses an **exact, digital, multi-bit bit-serial CIM**. None runs **inside a general-purpose MCU** with CPU involvement and real time measured on hardware.
- **Recommended re-scope.** Make the core claim *model-preserving*: the unchanged standard MFCC runs exactly on the same digital CIM as the network, using factored FFT matrices, a small twiddle/square unit and a log epilogue.
  - A fixed-point screen [M] gives ≈ **0.10 dB mean / ≤ 0.5 dB p99 log-mel error** from 0 to −30 dBFS with 8-bit coefficients and per-frame block floating point.
  - Cost: ≈ **1.5 k CIM-side cycles per frame** and **13 KiB** of resident coefficients (≈ 8 KiB if the mel matrix is limited to 0–4 kHz).
  - Keep streaming, autonomous execution and the MCU/FPGA demonstration as the *system context*, not as the novelty claim.

> **Gate update (2026-10-26, later).** Full results are in [`kws_gate/GATE_RESULTS.md`](../../kws_gate/GATE_RESULTS.md).
>
> **Step 1 passes, with a narrowed claim.** The Fudan paper keeps the FFT in a dedicated serial block and changes the front-end. USC's *Science Advances* 2025 runs a direct DFT plus a CNN on an analog memristor SoC, with no mel, log or DCT. FFT-on-CIM itself is established, including on SRAM, so the claim is now limited to the complete unchanged MFCC on a digital CIM with no retraining.
>
> **Step 2 passes.** With the unchanged MLPerf Tiny DS-CNN, int8 accuracy is 91.80 % with CIM-MFCC vs 91.66 % with float MFCC: Δ +0.14 pp, McNemar p = 0.35.

## 1. What each paper actually does

| Paper | CIM type | Front-end | Streaming | Control / host | Evaluation | KWS result |
|---|---|---|---|---|---|---|
| **CIMR-V** (ISCAS 2024; NYCU + ITRI) | 512 Kb SRAM CIM, 1-bit in/w/out, sense-amp output with ReLU | **Raw audio**: HPF + BN + quantise on RISC-V → 1-D **binary conv** ×5 in CIM | No (window) | Ibex on PULPissimo; **single-cycle CIM instructions issued by the core**; layer fusion, conv/pool pipeline, weight fusion (DMA from DRAM) | Design Compiler 28 nm; latency simulated with Ramulator DDR4 | 94.02 % GSC-12; 3707.84 TOPS/W (1-bit) |
| **PSCNN** (2022; NYCU) | 1 Mb SRAM CIM, binary activation / ternary weight, 128 SAs | **Raw audio** (1 s, 8-bit) → binary 1-D CNN | No | **Own 32-bit ISA** (MAC / weight replacement / pointer / halt) run by a system controller; ping-pong feature SRAM; pooling write-back (−35.9 % latency) | TSMC 28 nm ("fabricated"); 10 MHz | 92.53 % GSC-12; 885.86 TOPS/W |
| **RV-Flash** (ASP-DAC 2025) | 2 KB ADC-less **binary** eFlash CIM | Not addressed (model starts with Conv2D layers) | No | RV32I + 11 custom instructions; 16-multiplier FP16 VPU for DW/element-wise | 65 nm PnR; RTL cycle counts | CIM alone −7.7 to −11.4 % latency (FP16 ops dominate); with VPU −89.5 to −92.2 %. 96.5 % V2-12 |
| **Multi-stationary** (JSTS, Oct 2026) | 3T2C **eDRAM analog** CIM, 16 cores of 144×144, SP-ADC | **Raw audio sinc conv** with frequency-selective stride (IFM −99.2 %, −0.8 % acc), kernel fusion | No (1 s window) | Top controller + SIMD; GDCONV weight sharing + IA-stationary depthwise (16× DW throughput); layer fusion | 28 nm post-layout | 95.1 % GSC-10+2; 0.62 µJ/decision; −2.3 % accuracy overall vs baseline |
| **Song et al.** (Interspeech 2022) | Algorithm only | **KD "IMC encoder"**: Conv1d (128 × 150, stride 62) + a·\|x\|/(1+b·\|x\|) + avg-pool, replacing MFCC/SincConv | No | — | Accuracy | 96.57 % vs MFCC 95.85 % / SincConv 96.32 % (GSC v2-12, ResNet-8) |
| **FeFET streaming** (TETC 2024; Georgia Tech + Google) | FeFET **analog** CIM, 80 PEs × 4 arrays (128×128), 3-bit ADC | **External MFCC module** (40 × INT16, 30 ms / 10 ms) | **Yes**: per-layer circular buffers, all layers concurrent | On-chip controller protocol (SET_PATH / START / TRANSFER); all weights resident across PEs | NeuroSim 28 nm | 7.7–17.6× lower latency than sequential; 1.65 µJ/decision |
| **Multibit ReRAM** (TCAS-I 2026) | Charge-mode multi-bit **ReRAM analog**, APoT ADC | **External MFCC** | No | Top controller, NoC; DSCONV layer-fusion clusters, pipelined | 45 nm post-layout | 92.7 % GSC; 0.74 µJ/decision |

**Additional prior found by searching (not in the folder):**
- A Fudan **computing-on-memory-boundary** KWS processor merges the **mel filterbank and DCT into one matrix multiply**, with ternary coefficients in the macro (28 nm simulation, 11.6 µW). Its FFT handling is unknown; read the full text. [P: [IEEE CAS listing](https://catavault.ieee-cas.org/presentation/technical-lectures-and-demos/116%C2%B5w-computingonmemoryboundary-keyword-spotting)]
- FFT as small DFT GEMMs (four-step) on tensor cores [P: [tcFFT](https://arxiv.org/abs/2104.11471), [theory](https://arxiv.org/abs/1908.06649)].
- DFT on memristor IMC for 5G [P: [arXiv 2609.21537](https://arxiv.org/abs/2609.21537)].
- IIR-based KWS front-end hardware (ASAP-FE, ISLPED 2025) [P: [arXiv 2506.14657](https://arxiv.org/abs/2506.14657)].

## 2. Overlap matrix

| HopCIM pillar | CIMR-V | PSCNN | RV-Flash | Multi-stat. | Song | FeFET | ReRAM | Status |
|---|---|---|---|---|---|---|---|---|
| P1 RISC-V host + CIM for KWS | ✓ | — | ✓ | — | — | — | — | **Taken** |
| P2 Front-end moved into the CIM (learned or raw-audio conv) | ✓ | ✓ | — | ✓ | ✓ (alg.) | — | — | **Taken ×4** |
| P3 Streaming CIM inference with ring buffers | — | — | — | — | — | ✓ | — | **Taken** |
| P4 Autonomous execution (no per-op CPU) | ✗ (CPU issues) | ✓ | ✗ (CPU issues) | ✓ | — | ✓ | ✓ | **Taken** |
| P5 Depthwise handling | — | — | ✓ (VPU) | ✓ (GDCONV) | — | ✓ | ✓ | **Taken** |
| P6 Layer fusion / pipelining / fused epilogue | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | **Taken** |
| P7 "After CIM, non-MVM work dominates" evidence | — | — | ✓ (in-network ops) | — | ✓ (cites MFCC = 97.8 % of CPU time) | — | — | **Partly taken** (front-end case not quantified in a CIM MCU) |
| P8 Exact digital multi-bit bit-serial CIM | ✗ | ✗ | ✗ | ✗ | — | ✗ | ✗ | **Open** |
| P9 *Standard* MFCC computed on the CIM (model unchanged) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (external) | ✗ (external) | **Open** (Fudan: mel+DCT only) |
| P10 Inside a general-purpose MCU, real time and CPU involvement measured on hardware | ✗ | ✗ | ✗ | ✗ | — | ✗ | ✗ | **Open** |
| P11 Equal-platform comparison of front-end strategies | ✗ | ✗ | ✗ | ✗ | partial (accuracy only) | ✗ | ✗ | **Open** |
| P12 Streaming when the CIM is smaller than the model | ✗ | ✗ | ✗ | ✗ | — | ✗ (all resident) | ✗ | Open, but likely reduces to fixed batching |
| P13 One front-end shared by several always-on models | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **Open** |

## 3. What to drop or demote

- **Drop as contributions:**
  - a learned or raw-audio front-end on the CIM (P2);
  - streaming CIM (P3);
  - an autonomous instruction-driven controller (P4);
  - depthwise tricks (P5);
  - layer fusion (P6).

  These can still be *used*; they are no longer claims.
- The original H2 ("learned front-end within 1 pp of MFCC") is now a *baseline* reproduced from the literature, not a hypothesis.

## 4. Ways to make the work different

| # | Option | What is new | Evidence so far | Main risk | Verdict |
|---|---|---|---|---|---|
| **D1** | **Model-preserving exact front-end on a digital CIM.** Four-step FFT (512 = 16 × 32) as two resident int8 DFT matrices, a 4-multiplier twiddle/square unit, mel and DCT matrices, a log2 epilogue and a per-frame block exponent; all on the same bit-serial DCIM as the network | First standard-MFCC-on-CIM for KWS found. It is enabled by digital bit-serial precision (16–24-bit inputs at proportional cycles, exact integer MACs), which the analog/binary CIMs above lack — they had to change the model | [M] fidelity screen (table below); ~1.5 k CIM cycles per frame; 13.1 KiB coefficients; per-hop model **11.8 k cycles (0.59 MHz real time), ~500 CPU cycles** vs 104 k for the strong software baseline | A dedicated FFT unit may cost similar area → measure it (B3) and report the trade-off; Fudan's full text must be checked; the method lineage (tensor-core FFT) must be cited | **Core of the re-scope** |
| **D2** | **One shared front-end for several always-on audio models** (KWS + VAD + acoustic event or speaker), all on one DCIM | Model-specific CIM front-ends (P2) cannot be shared; a standard front-end is computed once per hop | [I] Capacity already favours D1 for one model (13 vs ~19 KiB for a 128 × 150 learned encoder). On cycles, a learned encoder is ~0.6 k/hop per model vs ~4.6 k once for MFCC → break-even ≈ 7 models (to be measured) | Multi-model evaluation work; datasets | **Extension that sharpens D1's value** |
| **D3** | **MCU-system evidence**: X-HEEP host, autonomous hop program, CPU-active cycles and real time on FPGA with a live mic; *equal-platform* comparison of CPU MFCC, dedicated FFT unit, learned CIM front-end (CIMR-V/PSCNN/Song style) and exact CIM MFCC | No paper measures a CIM KWS pipeline inside a host MCU on hardware, or compares front-end strategies on one platform | [M] census constants; PDM IP and mic pins verified | Reads as integration if presented alone | **Keep as system context and evaluation, not as the claim** |
| D4 | Signal-adaptive bit-serial precision: reuse the per-frame block exponent to skip high bit planes in quiet frames (exact) | Small twist on dynamic precision | — | Gain limited to the front-end; VAD frame skipping is the baseline | Optional ablation |
| D5 | PDM-to-decision on one DCIM: the PDM decimation filter as a 1-bit-input MVM | Neat (1-bit inputs are the cheapest DCIM pass) | [I] ~0.4 % load at 15 MHz | Saves only a tiny CIC/FIR block | Optional demo feature |
| D6 | Streaming on a CIM smaller than the model (hop batching + residency) | Not covered (FeFET keeps all weights resident) | — | Likely collapses to fixed batching (N4R lesson) | Operating-range sweep only |

**[M] Fidelity screen** (`analysis/dcim_mfcc_fidelity.py`): synthetic speech-like frames, 200 per level; 8-bit DFT/mel/DCT coefficients; int16 inter-stage values; log2 by leading one plus 4-bit mantissa.

| Input level (dBFS) | 0 | −10 | −20 | −30 | −40 | −50 | −60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Log-mel error, mean / p99 (dB), fixed shifts | 2.44 / 14.2 | 0.13 / 1.0 | 0.10 / 0.46 | 0.11 / 0.59 | 0.12 / 0.75 | 0.25 / 1.9 | 0.88 / 12.9 |
| **Log-mel error, mean / p99 (dB), block floating point** | **0.10 / 0.40** | **0.10 / 0.48** | **0.10 / 0.47** | **0.10 / 0.49** | 0.12 / 0.72 | 0.20 / 1.6 | 0.49 / 4.1 |
| MFCC c1–c9 relative error, block floating point | 1.0 % | 1.1 % | 1.0 % | 1.1 % | 1.2 % | 2.1 % | 5.0 % |

Limits: this is synthetic audio, not speech data. Per-frame gain is removed before comparison (the exponent is added back exactly in the log domain). **KWS accuracy has not been measured yet**; see 6, step 2.

## 5. Recommended re-scope

**Claim (one sentence) [H].** On a digital bit-serial CIM inside a RISC-V MCU, the complete always-on KWS pipeline can run as one autonomous per-hop program — *including the unchanged standard MFCC front-end*. Existing MFCC-trained models then deploy without retraining, at < 1 MHz real time and a few hundred CPU cycles per hop. We quantify when this beats a CPU front-end, a dedicated FFT unit, and model-specific learned CIM front-ends.

**Hypotheses:**
- **H1 (unchanged):** with the network on the DCIM, the CPU front-end is ≥ 70 % of per-hop cycles, measured on the board.
- **H2′:** CIM-MFCC log-mel error is ≤ 0.5 dB p99 for speech levels (−40 to 0 dBFS). The pretrained, *unchanged* MLPerf Tiny DS-CNN loses ≤ 0.3 pp accuracy on GSC with CIM-MFCC features versus float MFCC.
- **H3 (unchanged):** per-hop CPU-active cycles are ≥ 20× lower than the strong software baseline (S3), and f_min < 1 MHz.
- **H4′:** the added datapath (twiddle/square unit, log2, sequencer) is ≤ the LUT+DSP cost of the dedicated FFT/MFCC baseline. Otherwise, report the capacity-versus-logic trade-off honestly.
- **H5 (D2):** find the number of co-resident models at which a shared CIM-MFCC beats per-model learned front-ends in CIM capacity and per-hop cycles.

**Baselines:**
- B0: CPU only.
- S3: strong software (hardware epilogue, lean driver, real-FFT MFCC on the CPU).
- **B3:** dedicated FFT/mel unit.
- **B5: literature-style learned front-ends on the same DCIM** — a CIMR-V/PSCNN-style raw-audio conv and a Song-style KD encoder — giving an equal-platform comparison.
- Proposed: exact CIM-MFCC + streaming + hop program.

**What stays from the original plan:** the X-HEEP endpoint, streaming DS-CNN, hop sequencer, PDM microphone path and FPGA demo. These are the platform, not the claim.

**Framing.** Present the approaches as complementary. Analog and binary CIMs moved the front-end into the network because they cannot do 16-bit DSP exactly; a digital bit-serial CIM can. The open question is what that costs and when it pays. Do not describe earlier work as wrong.

If your group is the one behind CIMR-V and PSCNN, this reads naturally as the next step of that line: from binary analog raw-audio front-ends to a multi-bit digital CIM that runs the standard front-end inside an open MCU.

## 6. First steps (a ~2-week gate, mostly CPU-only)

1. **Novelty check.**
   - Read the full Fudan computing-on-memory-boundary KWS paper.
   - Search IEEE Xplore for "FFT" or "MFCC" with "compute-in-memory" / "in-memory computing" in JSSC, TCAS-I/II and ISSCC from 2020 to 2026.
   - **Stop D1** if a paper already computes the full standard MFCC, including the FFT, on a CIM for KWS.
2. **Accuracy without retraining.**
   - Extend `dcim_mfcc_fidelity.py` to the MLPerf Tiny KWS feature settings: 40 mel bands, 10 MFCCs, 30 ms window, 20 ms stride, 20–4000 Hz, natural-log offset, orthonormal DCT.
   - Download the GSC v2 test split and the pretrained MLPerf Tiny DS-CNN.
   - Compare accuracy with float features against CIM-MFCC features. No GPU is needed.
   - **Pass rule (H2′):** Δaccuracy ≤ 0.3 pp.
3. **In parallel, the board steps from the main report:** re-measure the census on the Nexys, and bring up the PDM microphone.

Start RTL (the endpoint plus a twiddle/square unit) only after steps 1 and 2 pass.

## 7. Sources

**Papers in the folder**

| Paper | Source |
|---|---|
| CIMR-V | [arXiv 2503.22072](https://arxiv.org/abs/2503.22072) / ISCAS 2024 |
| PSCNN | [arXiv 2205.01569](https://arxiv.org/abs/2205.01569) |
| RV-Flash | [DOI 10.1145/3658617.3697697](https://doi.org/10.1145/3658617.3697697) |
| Multi-stationary | [DOI 10.5573/JSTS.2026.26.5.407](https://doi.org/10.5573/JSTS.2026.26.5.407) |
| Song et al. | [ISCA archive](https://www.isca-archive.org/interspeech_2022/song22c_interspeech.html) |
| FeFET streaming | [DOI 10.1109/TETC.2023.3345346](https://doi.org/10.1109/TETC.2023.3345346) |
| Multibit ReRAM | [DOI 10.1109/TCSI.2025.3603076](https://doi.org/10.1109/TCSI.2025.3603076) |

**Additional prior (not in the folder):** Fudan computing-on-memory-boundary KWS ([IEEE CAS listing](https://catavault.ieee-cas.org/presentation/technical-lectures-and-demos/116%C2%B5w-computingonmemoryboundary-keyword-spotting)), [tcFFT](https://arxiv.org/abs/2104.11471), [memristor DFT DSE](https://arxiv.org/abs/2609.21537), [ASAP-FE](https://arxiv.org/abs/2506.14657).

**Local:** `analysis/dcim_mfcc_fidelity.py` (+ `_output.txt`); `analysis/post_cim_census.py` (scenario S5x, output in `post_cim_census_kws_with_mfcc_variant.txt`).
