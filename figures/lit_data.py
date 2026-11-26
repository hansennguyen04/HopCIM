#!/usr/bin/env python3
"""Data of the literature figures and tables (docs/LITERATURE_REVIEW.md): every value with its source.

Values of other works are typed in from the source named beside them (the full text of the paper, or MLCommons' published
results through docs/MLPERF_TINY.md section 8); HopCIM's come from model/, freeze/ and docs/DWU_RESULTS.md.
"""

# ---------------------------------------------------------------- figure 1: capability matrix
# y yes, p partly, n no, - not stated or not applicable
COLS = [
    ("KWS on the\nhardware", "kws"), ("Standard MFCC,\nexact, on the CIM", "mfcc"),
    ("Published pretrained\nmodel unchanged", "model"), ("Bit-exact to an\noutside reference", "exact"),
    ("Runs depthwise\nlayers", "dw"), ("Digital (exact-\narithmetic) CIM", "dcim"),
    ("RISC-V host", "riscv"), ("CPU free during\nthe network", "cpufree"),
    ("Live audio\ninput", "live"), ("Measured\nsilicon", "si"), ("Measured\nenergy", "en"),
]
WORKS = [
    # name, note, values in the order of COLS
    ("Shan, ISSCC 2020", "ynnnynnypyy"),
    ("Jia, ISCAS 2022", "ypnnnynynnn"),
    ("RV-Flash, ASP-DAC 2025", "ynnnynynnnn"),
    ("Fu, TVLSI 2026", "ynnpynnypyy"),
    ("KWS-CIM, TCAS-I 2026", "pnnnnnn-nyy"),
    ("Dong, JSSC 2026", "ynp-ypnypyy"),
    ("CIMR-V, ISCAS 2024", "ynnnnpynnnn"),
    ("AnalogNets, 2022", "ynnnnnn-npn"),
    ("iMCU, JSSC 2024", "pnp--yypnyy"),
    ("Garofalo, JETCAS 2022", "nn--ynynnnn"),
    ("DIANA + HTVM, 2023/24", "ynp-ynynnyy"),
    ("MixCIM, JSSC 2026", "nnnnynn-nyy"),
]
HOPCIM = "yyyyyyyyynn"
NOTES = {
    "Shan": "dedicated MFCC circuit, no CIM; clips from a test platform",
    "Jia": "mel and DCT on the array as one ternary matrix, log before mel; simulated",
    "RV-Flash": "binary eFlash CIM; depthwise on an FP16 vector unit; CPU orchestrates; place and route",
    "Fu": "no CIM; exact to its own quantised model only; audio through an FPGA",
    "KWS-CIM": "KWS accuracy is software with the macro's measured statistics; analog eDRAM CIM",
    "Dong": "the CIM runs only voice-activity detection; DS-CNN architecture retrained for 35 keywords",
    "CIMR-V": "1-bit SRAM CIM; synthesised",
    "AnalogNets": "depthwise replaced and retrained; PCM measured for accuracy, accelerator estimated",
    "iMCU": "4-keyword tiny-conv and MLPerf-Tiny ResNetv1; the text does not mention depthwise or exactness",
    "Garofalo": "analog array model; 22-nm place and route",
    "DIANA + HTVM": "DS-CNN with the first filter changed to 7 x 5; analog CIM core and digital core",
    "MixCIM": "analog macro with ADCs, MobileNet-V2; Zynq feeds it",
}

# ---------------------------------------------------------------- figure 2: cycles per inference, the KWS DS-CNN
# low, high (equal when one value): latency x clock as published; kind: hopcim, accel, cpu
CYCLES = [
    ("fpgaConvNet, ZedBoard (FPGA)", 32_000, 46_000, "accel", "MLPerf Tiny 1.1, MLPERF_TINY.md §8"),
    ("HopCIM S+DWU, network alone (FPGA)", 67_856, 67_856, "hopcim", "board, DWU_RESULTS.md R6"),
    ("HopCIM S, under MLPerf's runner*", 77_600, 77_600, "hopcim", "unverified, MLPERF_TINY.md §8"),
    ("Syntiant 9120", 134_000, 134_000, "accel", "MLPerf Tiny 1.0-1.3, MLPERF_TINY.md §8"),
    ("GAP9 (latency normalised to 260 MHz)", 176_800, 176_800, "accel", "0.68 ms, HTVM Table II"),
    ("fpgaConvNet, Zybo (FPGA)", 269_000, 269_000, "accel", "MLPerf Tiny 1.1, MLPERF_TINY.md §8"),
    ("DIANA digital core†", 455_000, 1_890_000, "accel", "1.75 ms (HTVM Table I, changed first filter) to 7.26 ms (HTVM as run by MATCH, Table III), 260 MHz"),
    ("K1 board (RISC-V vector)", 857_000, 857_000, "cpu", "MLPerf Tiny 1.3, MLPERF_TINY.md §8"),
    ("Cortex-M85 (Renesas RA8D1)", 1_660_000, 1_660_000, "cpu", "MLPerf Tiny 1.2, MLPERF_TINY.md §8"),
    ("Cortex-M7 (ST H7A3)", 3_170_000, 3_170_000, "cpu", "MLPerf Tiny 1.2, MLPERF_TINY.md §8"),
    ("Cortex-M4 (ST L4R5)", 5_210_000, 5_210_000, "cpu", "MLPerf Tiny 1.3, MLPERF_TINY.md §8"),
    ("CV32E20, TensorFlow Lite Micro*", 75_300_000, 75_300_000, "cpu", "this board, unverified, MLPERF_TINY.md §9"),
]

# ---------------------------------------------------------------- figure 3: depthwise options, one model [E], mode U
DW = [  # label, power uW at U, kind, source
    ("S: depthwise on the CIM (69 tiles)", 44.11, "base", "model/dwu_rtl.json"),
    ("Narrow macros, 7 columns wide", 34.70, "opt", "DWSWAP_STUDY.md, N7"),
    ("Feature-stationary (after Wang, ISSCC 2023)", 28.48, "opt", "DWSWAP_STUDY.md, FS"),
    ("Swap-in: rewrite two working tiles", 28.13, "opt", "DWSWAP_STUDY.md, V"),
    ("HopCIM unit as built (RTL)", 27.96, "hopcim", "model/dwu_rtl.json"),
    ("9-multiplier digital unit (study)", 27.72, "opt", "DWSWAP_STUDY.md, DIG"),
]
