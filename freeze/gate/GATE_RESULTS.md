# Gate results: model-preserving MFCC on a digital CIM (2026-10-26)

Decision: **Step 1 passes (the stop condition was not met), with a narrowed claim. Step 2 passes. Board steps done:** the CPU costs are re-measured on the Nexys (identical to simulation) and the onboard microphone captures speech (section 4).

## 1. Step 1 — prior art for FFT/MFCC on CIM

**Stop rule.** Stop if any work already runs the full *standard* MFCC, including the FFT, on a CIM.

**Method:**
- Read the Fudan paper in full.
- Run 19 query families through OpenAlex: 1,444 unique works screened automatically, 80 survivors reviewed by hand, 33 of them with IEEE DOIs.
- Run 8 arXiv query families: 107 entries.
- Read the full text of the closest open-access hit.

IEEE Xplore's own search endpoint refuses scripted requests (HTTP 418). The equivalent Command Search strings are in `novelty/lit_search_output.txt`, so you can re-run them in the Xplore web UI.

| Work | What runs on the CIM | Front-end | Verdict |
|---|---|---|---|
| **Jia et al., ISCAS 2022 (Fudan)**, full text read | Mel + DCT fused into one 129 × 40 ternary matrix on digital computing-on-memory-boundary macros | 256-pt FFT in a **dedicated serial R2²SDF FFT block**; log by **CORDIC**, moved **before** the mel filter; coefficients ternary, model retrained (< 3 % loss, 90.54 % on 10 keywords) | Not the standard MFCC; FFT not on the macro |
| Wang et al., *Science Advances* 2025 (USC), full text read | **Direct 128-point DFT** on a 128 × 128 analog memristor crossbar (8-bit inputs, PSNR 33.5 dB) + CNN on the same SoC (has a RISC-V CPU) | Magnitude computed **on a PC**; no mel, log or DCT; AudioMNIST spoken digits, 94.72 %; live-microphone demo | Closest system; not MFCC; analog |
| Xiao et al., arXiv 2409.19071 (analog FFT) | FFT mapped to analog charge-trapping arrays; demonstrated on audio and images | No MFCC or KWS pipeline | FFT-on-CIM exists (analog) |
| CSIFA (ASAP 2024); SRAM-CIM small-point FFT (*Microelectronics J.* 2026); SHIFFT (ISVLSI 2024); ReFFT (TCAD 2025); FourierPIM (2023); memristor DFT papers | FFT or DFT on SRAM, RRAM or PIM | No KWS front-end | **FFT on CIM is not new**, including on SRAM |
| AM-CIM (TCAS-I 2025) | Analog-memory near-sensor CIM for KWS with Mel-energy input | Simplified features; GRU; 180 nm | Not standard MFCC |
| Earlier folder papers (CIMR-V, PSCNN, RV-Flash, multi-stationary, Song, FeFET, ReRAM) | — | MFCC external, or replaced by a learned / raw-audio conv | Not standard MFCC |

**Conclusion:** no work runs the complete standard MFCC (FFT + mel + log + DCT) on a CIM — and none feeds such features to an *unchanged* pretrained KWS model.

**What remains claimable** (narrower than in the earlier novelty assessment):

> The complete, unchanged MFCC front-end and the KWS network on one **digital** bit-serial CIM, with **no retraining**, inside a RISC-V MCU, with measured real-time behaviour.

What can no longer be claimed:
- FFT-on-CIM;
- mel/DCT on a memory macro;
- "spectral front-end + classifier on one CIM SoC" (USC did this in analog).

## 2. Step 2 — unchanged MLPerf Tiny DS-CNN with CIM-computed MFCC

The protocol, datapath and pass rule were frozen in `GATE2_MANIFEST.md` before the run. SHA-256 values of the manifest and script are in `gate2_frozen.sha256`. The script is `gate2_eval.py`; raw results are in `gate2_results.json`.

| Model | Reference float MFCC | CIM-MFCC | Δ |
|---|---:|---:|---:|
| **int8** (`kws_ref_model.tflite`, reference input quantisation) — gate metric | 91.66 % | 91.80 % | **+0.14 pp** |
| int8, round-to-nearest input quantisation | 91.86 % | 91.94 % | +0.08 pp |
| float32 | 91.86 % | 91.88 % | +0.02 pp |

**Pass rule:** Δ ≥ −0.30 pp → **pass.**

**Supporting checks:**
- McNemar, int8: 17 vs 24 discordant clips, p = 0.35. Float32: 5 vs 6, p = 1.0. There is no significant difference.
- The reference replication matches the published ≈ 92 %.

**Feature fidelity**, in int8 input steps (step = 0.585):
- mean 0.034, median 0.012, p99 0.35;
- 99.46 % of values within half a step.

**Outliers** (post-hoc observation; the frozen datapath is unchanged): the worst value is 9.2 steps off, and 389 of 4,890 clips have at least one value > 1 step off. All of these are in **c0** of very quiet frames. The hard floor `max(ln v, ln 1e-6)` and the integer rounding of tiny magnitudes depart from TensorFlow's `ln(v + 1e-6)`. A soft floor (a small `ln(1 + e^-d)` LUT) would remove most of this. It did not affect accuracy.

**Datapath tested** (all exact integer operations; CIM weights 8-bit, the DCT 16-bit as two passes):
- Hann window in Q15 with a per-frame block exponent;
- four-step FFT (16 × 32) using int8 DFT matrices;
- Q14 twiddles;
- exact integer square root;
- int8 mel matrix;
- log2 from a leading-one detector plus a 64-entry LUT;
- int16 DCT.

## 3. Is the direction purely digital and evaluable with simulation and FPGA tools?

**Yes.** Every element is digital logic or software:

| Element | Implementation | How it is evaluated |
|---|---|---|
| CIM macro | Digital SRAM CIM: bit-serial inputs, exact integer adder tree. On the FPGA, BRAM storage plus a DSP/LUT compute engine, cycle-faithful to a 64 × 64 macro | Verilator unit tests vs the integer reference; FPGA cycles |
| Front-end (FFT matrices, twiddle multiplier, integer sqrt, log LUT, DCT) | Digital RTL plus resident CIM coefficients | Bit-exact vs `gate2_eval.py`; accuracy in software (done above) |
| Streaming DS-CNN, epilogue, hop sequencer, DMA | Digital RTL plus firmware on X-HEEP | Verilator SoC simulation; FPGA real-time runs |
| Microphone path | Digital PDM interface (X-HEEP `pdm2pcm`, CIC) | Verilator (PDM replay) and FPGA (onboard mic) |
| Cost | — | Vivado LUT/FF/BRAM/DSP and timing. Optionally post-synthesis digital power or area from an open ASIC flow (Yosys/OpenROAD with switching activity from simulation) |

**Out of scope, and not needed:**
- analog or mixed-signal CIM, ADCs, memristor/eFlash/FeFET/eDRAM devices;
- SPICE;
- tape-out;
- energy claims from FPGA power.

Consequently, the comparison with the literature must use cycles, CPU-active cycles, real-time headroom, accuracy, bytes moved and FPGA resources — **not TOPS/W**. Most competing papers' advantages are in analog energy, which this work deliberately does not claim.

## 4. Board steps (status)

| Item | Status |
|---|---|
| CPU-cost re-measurement on the Nexys | **Done (2026-10-26 23:08 and 23:09; two runs identical).** The board matches the Verilator RTL simulation **cycle for cycle** on every kernel: int8 MACs 11.0–18.8 cyc/MAC, requantisation 20/35 cyc/elem, MMIO 8/9 cyc, DMA wall time, FFT512 145,142 cyc and mel/log/DCT 15,632 cyc. The single difference: `dma4KiB_cpu_active` is 2,663 (board) vs 695 (simulation). The FPGA clock gate is a pass-through (`xilinx_clk_gating: assign clk_o = clk_i`), so on the board `mcycle` keeps counting during WFI. **Methodology consequence:** on the FPGA, CPU-active cycles must come from a sleep-cycle counter or from instruction counts, not from `mcycle`. Logs: `research_discovery/20261026_application_driven/evidence/census_microbench_nexys_board_*.uart.log` |
| Microphone bring-up | **Done (2026-10-26 23:23).** The PDM-enabled one-cable bitstream (`bitstreams/xheep_nexys_bscan_pdm.bit`, sha256 `a967f74d…4d56`) adds 2,096 LUTs. Hold timing is met (WHS +0.051 ns); the remaining setup failures are the pre-existing debug SPI-slave clock-crossing paths only. Simulation: 2,048/2,048 CIC samples bit-exact vs the Python RTL model. Board: button-triggered 1 s capture at 15.625 kHz (1.25 MHz PDM, CIC ÷16, CPU FIR ÷5), no FIFO overflow. Noise floor −72 to −76 dBFS; a spoken word at 600–900 ms reaches −47 dBFS (≈ 27 dB above the floor), with 78–92 % of its energy below 1 kHz and a periodic pitch of 125–131 Hz (autocorrelation 0.51–0.89). Informal check: the unchanged MLPerf Tiny DS-CNN gives the same top-1 class for float MFCC and CIM-MFCC on this recording (max feature difference 0.077). Notes for the real system: exact 16 kHz needs a 16 MHz system clock (or resampling); the mic DC offset is ≈ 600 LSB. Logs and WAV are in `research_discovery/20261026_application_driven/evidence/` |
| RTL of the CIM endpoint | **Not started.** It is allowed now that steps 1 and 2 have passed; it waits for your go-ahead |
