# System-level novelty boundary: the closest KWS works, citations checked

Status (2026-11-08): the six works of the student's plan (`HopCIM_Specific_Next_Steps_and_Professor_Advice.md` §2,
§23), checked against Crossref. **Bibliographic data verified for all six; what the plan says about their contents is
not yet checked against the full texts**, except where noted. The plan's search was an IEEE-Xplore-targeted web
search, not a logged-in manual Xplore search (the plan says so itself), so the thesis keeps "to our knowledge".
Later the same day (§1a): the published abstracts of works 1, 4, 5, 6 were read (their full texts cannot be fetched
from here), and work 4's abstract retires the plan's "first-style" sharing claim (§2).

## 1. The six works

| # | Work (Crossref) | Verified | The plan's reading (not checked unless noted) |
|---|---|---|---|
| 1 | W. Shan, M. Yang, J. Xu, Y. Lu, S. Zhang, T. Wang, J. Yang, L. Shi, M. Seok, "A 510nW 0.41V low-memory low-computation keyword-spotting chip using serial FFT-based MFCC and binarized depthwise separable convolutional neural network in 28nm CMOS", ISSCC 2020, pp. 230-232, DOI 10.1109/ISSCC19947.2020.9063000. Journal version: JSSC 56(1), 151-164, DOI 10.1109/JSSC.2020.3029097 (from the EPFL record, not Crossref) | yes | serial-FFT MFCC and a binarized DS-CNN in one chip; MFCC not on a CIM; network co-designed |
| 2 | X. Jia, H. Zhu, Y. Wang, J. Zhang, F. Lin, X. Xiong, D. Xu, C. Chen, Q. Liu, "A 11.6μW computing-on-memory-boundary keyword spotting processor with joint MFCC-CNN ternary quantization", ISCAS 2022, DOI 10.1109/ISCAS48785.2022.9937206 | yes | read earlier (as "Fudan ISCAS'22", `NOVELTY_ASSESSMENT.md`): FFT in a serial R2²SDF block, log before mel, ternary mel and DCT on the memory macros; model changed jointly |
| 3 | G. Kang, D. Park, H. Lee, S. Jung, J. Park, J. G. Min, Y. Lee, J. Kung, "RISC-V driven orchestration of vector processing units and eFlash compute-in-memory arrays for fast and accurate keyword spotting", ASP-DAC 2025, DOI 10.1145/3658617.3697697 | yes | RISC-V + eFlash CIM + vector units for KWS with a changed (reparameterised) model; supplied by the student earlier as RV-Flash |
| 4 | Z. Fu, K. Lin, X. Zhao, L. Ge, W. Ye, "A 95.3% 12-class, 108-nJ/inference keyword spotting chip with hybrid FFT-BFNet architecture and exponent-aware nonuniform quantization in 65-nm CMOS", IEEE TVLSI 34(2), 568-578, 2026, DOI 10.1109/TVLSI.2025.3632791 | yes | FFT and network share hardware; network and quantisation changed |
| 5 | D.-H. Lee, S.-U. Kang, J. Jang, S.-M. Yoon, H. Kim, M. Seok, M.-S. Choo, "KWS-CIM: a charge-domain eDRAM-based compute-in-memory macro for ternary DFSMN keyword spotting", IEEE TCAS-I 73(11), 7610-7623, 2026, DOI 10.1109/TCSI.2026.3717749 | yes | a fabricated analog-domain CIM macro for a ternary DFSMN |
| 6 | Y. Dong, X. Liu, K. Bai, G. Li, M. Wu, Y. Jing, Y. Zhang, P. Zhan, Y. Zhang, Y. Ma, R. Huang, T. Jia, L. Ye, "A heterogeneous TinyML SoC with systematic minimum energy searching and management for keyword spotting", IEEE JSSC 61(6), 3098-3112, 2026, DOI 10.1109/JSSC.2025.3619427 | yes | system-level energy management for KWS, CIM-assisted wake-up |

With CIMR-V, PSCNN, the FeFET and ReRAM KWS papers (`NOVELTY_ASSESSMENT.md`) and the dataflow works
(`STREAMFUSE_NOVELTY.md`), these mark the boundary: KWS on CIM, RISC-V with CIM for KWS, hardware MFCC, serial-FFT
MFCC, mel/DCT near memory, FFT/network sharing and system energy management are all published.

## 1a. Works 1, 4, 5, 6: access and what their abstracts state (2026-11-08, later)

The plan asks for these four to be read in full. **None could be read in full from here**: OpenAlex lists 1, 4 and 6 as
closed access with no repository copy (`any_repository_has_fulltext: false`); 5 is open access (CC-BY, hybrid), but its
only copy is on IEEE Xplore, which refuses automated downloads (HTTP 418). What follows is from the **published
abstracts** (Semantic Scholar's copy of the publisher's abstract; Crossref holds none for these DOIs), so it is firmer
than the plan's reading but still short of the full text.

| # | Access | What the abstract states | Consequence for HopCIM's claims |
|---|---|---|---|
| 1 Shan et al. (JSSC 56(1) 2021, journal version of the ISSCC 2020 paper) | closed | serial-FFT MFCC circuit for the front-end; a small **binarized** DS-CNN; **framewise incremental computation** instead of whole-word processing; near-threshold 0.41 V at 40 kHz, 0.51 µW, 0.23 mm², 28 nm; Google Speech Commands, 97.3 % on a one-word and 94.6 % on a two-word task | streaming, frame-by-frame KWS with MFCC in hardware is published; HopCIM must not claim incremental/streaming processing or a low always-on clock as new. Model changed (binarized), not 12-class |
| 4 Fu et al. (TVLSI 34(2) 2026) | closed | FFT-butterfly-structured network (BFNet) replacing dense pointwise convolutions (6.3× fewer MACs, half the parameters); 6-bit exponent-aware nonuniform weight quantisation; **"Mel-filter-banks reuse fully-connected (FC) layer multipliers through decomposed 8-bit arithmetic, and FFT output buffers double as convolutional neural network (CNN) feature map memory"**; 95.3 % 12-class, 108.04 nJ/inference, 0.65 V/600 kHz, 14.82 ms, 65 nm | **the front-end and the network already share digital multipliers and memory in a published KWS chip.** The plan's "first-style" sharing claim (§2) is not safe in any wording that rests on sharing alone. Model and quantisation changed |
| 5 Lee et al., KWS-CIM (TCAS-I 73(11) 2026) | open (CC-BY), Xplore only | a 3T2C eDRAM **charge-domain** CIM **macro** for a DFSMN with ternary activations and binary weights; dual-mode sensing, in-array offset calibration, 1-bit readout; 65 nm, 250 MHz, 930–1335 / 82–336 TOPS/W by mode; measured 91.66 / 89.87 / 87.30 % for 12 / 20 / 35 classes | a macro, not a system: no front-end or host in the abstract. Analog domain, changed model. (Its 91.66 % equals HopCIM's float-MFCC figure by coincidence; do not mix them up) |
| 6 Dong et al. (JSSC 61(6) 2026; VLSI Symposium 2024 version) | closed | heterogeneous TinyML SoC for KWS, a microprocessor plus domain-specific accelerators; synthesizable energy/event/performance monitors per block guide a system-level minimum-energy-point search; in-block power switches and on-chip switched-capacitor regulators; a **two-stage CIM-based event-driven wake-up**; 55 nm, 51 % average energy saving, 3.5 µW minimum, 30,000× peak-to-idle | a measured, energy-managed MCU+accelerator KWS SoC with CIM exists. HopCIM's "CPU-light MCU, CPU asleep" is a system property others have, and HopCIM's energy figures are estimates [E], not silicon: no energy-leadership claim |

What the abstracts do **not** settle, and the full texts must: whether any of the four keeps a standard pretrained
model unchanged (none says so; 1, 4 and 5 state binarized, restructured or ternary networks; 6 says nothing about its
network); whether 1's or 4's front-end is the standard MFCC computed exactly; what 6's CIM computes besides the
wake-up stage; and whether any claims bit-exactness to a software reference.

## 2. What this means for the claim

The plan's safer claim stays defensible: an **unchanged** pretrained int8 KWS model, deployed end to end on a digital
CIM-equipped RISC-V MCU **without retraining**, bit-exact to TFLite's reference kernels, with a live real-time
FPGA demonstration, and the measured migration of the bottleneck as more moves into hardware. None of the six keeps
the model unchanged (2, 3, 4 change it; 1 and 5 use binarized or ternary networks; 6 is silent in its abstract,
which the full text must settle before the claim says "none of the six").

The plan's "first-style" claim ("MFCC and network share one exact multi-bit digital CIM datapath") should be
**dropped as a contribution**, for two reasons that are now both on record:

1. `B3_BASELINE.md`: sharing the CIM **saves no hardware** for the front-end (the dedicated front-end costs the same
   logic; H4 not supported).
2. Fu et al. (work 4) share the network's FC multipliers with the mel filter banks and the FFT buffers with the
   feature-map memory, in measured silicon. Sharing as such is published; what remains specific to HopCIM is *exact*
   (bit-exact to the reference) and *unchanged model*, which the safer claim already states.

HopCIM may still describe the shared CIM as a design fact; it must not present it as new or as a saving.

## 3. Still to do

- **The student:** download work 5 (free, CC-BY; it needs only a browser, not a subscription) and works 1, 4, 6
  (NYCU's IEEE subscription) into `Literature review/`. With the PDFs there, the reading can be finished and this
  table updated from the full texts. Questions to answer from each:
  - 1 Shan: is the front-end the standard MFCC (window, mel, log, DCT) or a reduced one? Is "framewise incremental"
    exact, i.e. identical to whole-word processing?
  - 4 Fu: what does the decomposed 8-bit arithmetic share, exactly, and is the mel/FFT path exact against a float or
    integer reference? Is the 95.3 % model trained for BFNet (it must be, as the pointwise layers are replaced)?
  - 5 Lee: is there any host, front-end or real-time evaluation outside the macro? How was the DFSMN trained?
  - 6 Dong: which KWS network, trained how? What runs on the CIM besides the wake-up stage? Is any part bit-exact?
- The manual, logged-in IEEE Xplore search (`HOPCIM_PROPOSAL.md` §15.2) is still the student's.

## Sources

- Crossref records of the six DOIs (api.crossref.org/v1/works/<DOI>), retrieved 2026-11-08
- [EPFL record of Shan et al.](https://infoscience.epfl.ch/record/280400?ln=en)
- Abstracts: Semantic Scholar API records of DOIs 10.1109/JSSC.2020.3029097, 10.1109/TVLSI.2025.3632791,
  10.1109/TCSI.2026.3717749, 10.1109/JSSC.2025.3619427 (api.semanticscholar.org/graph/v1/paper/DOI:<DOI>), retrieved 2026-11-08
- Access status: OpenAlex records of the same DOIs (api.openalex.org/works/doi:<DOI>), retrieved 2026-11-08
- [IEEE Xplore page of work 5](https://ieeexplore.ieee.org/document/11654553/) (refuses automated access)
