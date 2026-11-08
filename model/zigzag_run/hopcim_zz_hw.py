# HopCIM's macro as a ZigZag-IMC accelerator (docs/ENERGY_ESTIMATE.md section 11). Loaded by run_zigzag.py from the
# tool's own folder. The array is the engine's: 64 rows, 16 int8 columns, one input bit per cycle; group
# depth = tiles sharing one macro's logic. The memory levels are the tool's template (its example Dimc.py).
import importlib.util
import os

import zigzag.inputs.examples.hardware.Dimc as base
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.ImcArray import ImcArray

GROUP = int(os.environ.get("ZZ_GROUP", "16"))
MACROS = int(os.environ.get("ZZ_MACROS", "1"))
TECH = {
    "tech_node": 0.028, "vdd": 0.9, "nd2_cap": 0.7 / 1e3, "xor2_cap": 0.7 * 1.5 / 1e3, "dff_cap": 0.7 * 3 / 1e3,
    "nd2_area": 0.614 / 1e6, "xor2_area": 0.614 * 2.4 / 1e6, "dff_area": 0.614 * 6 / 1e6,
    "nd2_dly": 0.0478, "xor2_dly": 0.0478 * 2.4,
}
HD = {
    "pe_type": "in_sram_computing", "imc_type": "digital", "input_precision": 8, "weight_precision": 8,
    "input_bit_per_cycle": 1, "group_depth": GROUP, "wordline_dimension": "D1", "bitline_dimension": "D2",
    "enable_cacti": True,
}
DIMS = {"D1": 16, "D2": 64, "D3": MACROS}


def array():
    return ImcArray(TECH, HD, DIMS)


def cores():
    a = array()
    return {Core(1, a, base.memory_hierarchy_dut(a))}


accelerator = Accelerator("hopcim_zz_hw", cores())
