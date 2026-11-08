# System-level novelty boundary: the closest KWS works, citations checked

Status (2026-11-08): the six works of the student's plan (`HopCIM_Specific_Next_Steps_and_Professor_Advice.md` §2,
§23), checked against Crossref. **Bibliographic data verified for all six; what the plan says about their contents is
not yet checked against the full texts**, except where noted. The plan's search was an IEEE-Xplore-targeted web
search, not a logged-in manual Xplore search (the plan says so itself), so the thesis keeps "to our knowledge".

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

## 2. What this means for the claim

The plan's safer claim stays defensible: an **unchanged** pretrained int8 KWS model, deployed end to end on a digital
CIM-equipped RISC-V MCU **without retraining**, bit-exact to TFLite's reference kernels, with a live real-time
FPGA demonstration, and the measured migration of the bottleneck as more moves into hardware. None of the six keeps
the model unchanged (2, 3, 4 change it; 1 and 5 use binarized or ternary networks).

One caution on the plan's "first-style" claim ("MFCC and network share one exact multi-bit digital CIM datapath"):
it is true of HopCIM, but `B3_BASELINE.md` found that sharing the CIM **saves no hardware** for the front-end (the
dedicated front-end costs the same logic, H4 not supported). The thesis must not imply that sharing is an advantage
in cost; it is an exactness and integration property. That is a further reason to prefer the safer claim.

## 3. Still to do

- Read works 1, 4, 5 and 6 in full before citing their contents (the student's IEEE Xplore access).
- The manual, logged-in IEEE Xplore search (`HOPCIM_PROPOSAL.md` §15.2) is still the student's.

## Sources

- Crossref records of the six DOIs (api.crossref.org/v1/works/<DOI>), retrieved 2026-11-08
- [EPFL record of Shan et al.](https://infoscience.epfl.ch/record/280400?ln=en)
