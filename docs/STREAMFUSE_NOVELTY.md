# StreamFuse: novelty boundary and citation check

Status (2026-11-08): the two papers the student supplied were read in full, and CLSA-CIM from the same folder; the
proposal's other citations were checked by Crossref and web search. **This is not the IEEE Xplore search**, which
remains the student's (`HOPCIM_PROPOSAL.md` §15.2).

## 1. Bottom line

Every mechanism in the proposal is published: depth-first execution with line buffers, depthwise→pointwise
forwarding, cross-layer scheduling on a CIM whose weights stay resident, and input reuse for depthwise layers on a
CIM. What this project can add is narrower:

- the mechanisms combined on **one time-shared digital CIM engine inside an MCU**, with all weights resident and
  the CPU asleep, running an **unchanged int8 TFLite model bit-exactly** (checked on all 4,890 test clips);
- a **cycle-exact, RTL-validated model** of the hardware sequencer that prices each option, with exactness,
  hazards and buffer sizes proven by replay;
- the **evaluation result**: once the CPU is gone, what an exact activation dataflow is worth beyond the cheapest exact
  alternatives, and that latency does not move because requantisation and gathering bind (`STREAMFUSE_STUDY.md`).

That is an evaluation and design-space contribution, not a new mechanism. It fits the repositioning of
`HOPCIM_PROPOSAL.md` §13a ("what has to move into hardware"), and should be presented that way.

## 2. The closest prior art

| Work | What it does | Where HopCIM differs |
|---|---|---|
| **CIMR-V** (Guo, T.-S. Chang et al., NYCU and ITRI; arXiv 2503.22072, 2025) | SRAM CIM (512 Kb, 1-bit) with an Ibex RISC-V on PULPissimo; **CIM layer fusion** by row-wise blocks whose overlap stays in the CIM input buffer; conv/max-pool pipeline; weights streamed from DRAM ("weight fusion"). A binary 1-D CNN for KWS, 94.02 % on GSC. Latency −85.14 % in simulation, of which layer fusion −33.16 %, all against DRAM traffic. 28 nm, 3,707.84 TOPS/W at 1 bit | Fusion there removes **off-chip** traffic; HopCIM has none. Binary, no depthwise layers, own model rather than an unchanged int8 reference, weights not resident, and the CPU issues every CIM instruction. Same institute as this thesis: worth citing prominently |
| **Yildirim, Ozturk** (arXiv 2511.21232, 2025) | Fused pixel-wise dataflow for MobileNetV2's Ex→Dw→Pr as a RISC-V CFU (VexRiscv) on an **Artix-7 XC7A100T**: dedicated engines (9 expansion, 1 depthwise, 56 projection), **no intermediate buffer**, padding on the fly. Data movement −87 % (derived from L-by-L byte counts), 59.3× against software on one layer, 20,922 LUT, 97 BRAM, 178 DSP; ASIC by Genus and CACTI | Not a CIM. "Zero buffer" is bought by **recomputing** the expansion for each 3×3 window, not by line buffers. Fusion stays inside one block; each block's output goes back to the CPU. Evaluated per layer; bit-exactness against TFLite is not reported |
| **CLSA-CIM** (Pelke et al., RWTH, DATE 2024) | Cross-layer scheduling for **tiled RRAM CIM with all weights resident**: partial output maps are forwarded to the next layer before it completes; up to 29.2× speedup through PE utilisation. States that CIM's weight-stationary dataflow suits cross-layer inference | Its gain is parallelism across many tiles; HopCIM has one engine, so the gain is memory only. **The proposal's §7 argument (resident weights make layer switching free) is already made here** |
| Xuan et al. (TCAS-II 2022); J. Li et al. (A-SSCC 2021); **EDEA** (SOCC 2024, arXiv 2503.11707) | Depthwise→pointwise fusion on FPGA/ASIC accelerators: partial fusion (−80.5 % BRAM traffic), a double-layer MAC (−41 %), dedicated DW and PW engines with **direct transfer and the requantisation in between** | Option B's forwarding is published for non-CIM accelerators |
| Alwani et al. (MICRO 2016); **DepFiN** (JSSC 2023); **DeFiNES** (HPCA 2023); Stream (ISPASS 2023) | Fused-layer and depth-first execution with line buffers; DeFiNES models it on ZigZag and is validated against DepFiN | Option C is the classic depth-first schedule |
| Wang, Tsai, Liu (ASP-DAC 2026, NTHU); ConvDK (Song, Jeong; arXiv 2508.14375) | Input reuse and mapping for depthwise layers on CIM | Intra-layer; not used here. HopCIM's depthwise layers are already engine-bound at today's timing |

## 3. The proposal's citations

| Citation as given | Check | Result |
|---|---|---|
| Wang, Tsai, Liu, ASP-DAC 2026, DOI 10.1109/ASP-DAC66049.2026.11420629 | Crossref | **Verified**: "Input Reuse, Weight-Stationary Dataflow and Mapping Strategy for Depthwise Convolution in Computing-in-Memory Neural Network Accelerators", NTHU. Its figures (5.19×, 2.87×, 6.3 %) not checked |
| MIREDO, ASP-DAC 2026 | arXiv 2510.26463 | **Verified** (Beihang); up to 3.2× (whether speed or EDP: check the paper) |
| FuseCU, "DAC 2025", IEEE 11132765 | web | **Exists**: "Principle-based Dataflow Optimization for Communication Lower Bound in Operator-Fused Tensor Accelerator", 38.7–63.6 % less data movement, 1.14–1.33×. Venue not confirmed |
| Thanos, DATE 2025 | Crossref | **Verified**, DOI 10.23919/DATE64628.2025.10993014 |
| EDEA, "2025" | arXiv | **Exists, but SOCC 2024** (arXiv 2025) |
| ROZK, "JSSC 2026" | web | **Exists** ("…Reconfigurable NoC and Local Zero-Skipping"); venue, year and KWS focus not confirmed |
| PIPECIM | web | A TVLSI paper "PIPECIM: … Pipelined Computing-in-Memory Computation Engine With Sparsity-Aware Technique" is listed; the fine-grained ping-pong detail is not confirmed |
| DepFiN, TileFlow, CiMLoop, DeFiNES | known | Real; DeFiNES was run here (`STREAMFUSE_STUDY.md` V4) |
| GPU DW/PW fusion (arXiv 2404.19331), GAP8 fusion (arXiv 2406.12478) | web | Both found; their figures not checked |
| TCAD 2024 (IEEE 10318181), 2025 dataflow study (IEEE 11096377), dual-sided sparsity (IEEE 10562243) | — | **Not checked** |

The proposal's links carry `utm_source=chatgpt.com`. Every citation above should be read in the original before it
goes into the thesis.

## 4. What not to claim

- Fusion, depth-first execution, line buffers or depthwise→pointwise forwarding as new.
- That resident weights make layer interleaving cheap (CLSA-CIM).
- Sign-gated requantisation as new: skipping work for outputs that a ReLU clamps is old (for example, predicting
  negative outputs to skip MACs). Here it is an exact, cheap baseline.

## Sources

- CIMR-V: [arXiv 2503.22072](https://arxiv.org/html/2503.22072v1) (and the student's copy)
- Yildirim, Ozturk: [arXiv 2511.21232](https://arxiv.org/abs/2511.21232v1) (and the student's copy)
- CLSA-CIM, DATE 2024 (student's copy)
- [MIREDO, arXiv 2510.26463](https://arxiv.org/abs/2510.26463)
- [FuseCU, IEEE 11132765](https://ieeexplore.ieee.org/document/11132765/)
- [EDEA, arXiv 2503.11707](https://arxiv.org/html/2503.11707v1)
- [ConvDK, arXiv 2508.14375](https://arxiv.org/html/2508.14375v1)
- [DeFiNES, arXiv 2212.05344](https://arxiv.org/pdf/2212.05344), [code](https://github.com/ZigZag-Project/DeFiNES)
- [Thanos, Sogang record](https://scholarworks.sogang.ac.kr/item/e9f6afe1-82e8-4251-828c-891d2acdddd3)
- [PSCNN, arXiv 2205.01569](https://arxiv.org/abs/2205.01569v1)
- Crossref records for the ASP-DAC 2026 and DATE 2025 DOIs (api.crossref.org)
