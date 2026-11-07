#!/usr/bin/env python3
"""Published-tool numbers for the energy estimate; writes energy_components.json (energy_est.py reads it).

Run with the AccelForge environment of recim-rv/accelforge_pilot (Python 3.12; see its README):
  ../../recim-rv/accelforge_pilot/.venv312/bin/python energy_tool_run.py

- CACTI 7 through hwcomponents-cacti (ITRS low-standby-power cells and periphery, leakage-optimised
  organisation, 300 K, each tool's own supply): access energy, leakage and area of every HopCIM memory.
- Aladdin components through hwcomponents-library: multipliers and adders.
- AccelForge with its model of the Colonnade macro (JSSC 2021, digital bit-serial, 65 nm): first the
  reproduction of the paper's measured energies, then HopCIM's pass shapes on the same macro structure,
  at 65 nm and scaled to 28 nm by the library.
"""
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
AF = os.path.join(HERE, "..", "..", "recim-rv", "accelforge_pilot", "third_party", "accelforge")
sys.path.insert(0, os.path.join(AF, "examples", "arches", "compute_in_memory"))

from hwcomponents_cacti import SRAM                                       # noqa: E402
from hwcomponents_library.library.aladdin import AladdinAdder, AladdinMultiplier  # noqa: E402
from _load_spec import get_spec                                           # noqa: E402
from accelforge.mapper.FFM.mappings import Mappings                       # noqa: E402

Mappings.render = lambda self, *a, **k: "<svg/>"                          # Graphviz is optional and absent

NODES = (65e-9, 45e-9, 32e-9, 28e-9, 22e-9)
# name: (word bits, words). Sizes are what the design needs, not the FPGA's block-RAM rounding.
MEMS = {
    "act_word":  (56, 2500),     # 2 buffers x 125 positions x 10 groups, 7 bytes a word, byte write enables
    "feat":      (32, 123),      # 49 rows x 10 bytes
    "lraw":      (32, 1960),     # 49 rows x 40 mel
    "params":    (59, 640),      # 10 layers x 64: bias 24, qm 31, e 4
    "ring":      (16, 1024),
    "bbuf":      (34, 512),      # cim_fe
    "ribuf":     (42, 128),      # cim_fe
    "magbuf":    (21, 128),      # cim_fe
    "fmem":      (36, 512),      # mfcc_ded
    "weights":   (128, 4416),    # 69 tiles of 64 rows x 16 int8, for the cell leakage per bit
    "act_plus_32k": (56, 7072),  # ACT with the reuse cache of 1 phase (+32 KB), one flat array
    "act_plus_72k": (56, 12786), # ACT with the reuse cache of 2 phases (+72 KB), one flat array
}
MULTS = {"requant_25x31": (25, 31), "window_16x15": (16, 15), "twiddle_16x16": (16, 16),
         "butterfly_18x18": (18, 18), "square_21x21": (21, 21), "mel_18x8": (18, 8), "fir_17x16": (17, 16)}


def cacti():
    out = {}
    for node in NODES:
        nm = f"{node * 1e9:.0f}"
        out[nm] = {}
        for name, (w, d) in MEMS.items():
            s = SRAM(tech_node=node, width=w, size=w * d)
            out[nm][name] = {"width": w, "depth": d, "read_pJ": s.read_energy * 1e12, "write_pJ": s.write_energy * 1e12,
                             "leak_uW": s.leak_power * 1e6, "pW_per_bit": s.leak_power / (w * d) * 1e12,
                             "area_um2": s.area * 1e12}
    return out


def aladdin():
    out = {}
    for node in NODES:
        nm = f"{node * 1e9:.0f}"
        out[nm] = {k: AladdinMultiplier(tech_node=node, width_a=a, width_b=b).read().energy * 1e12 for k, (a, b) in MULTS.items()}
        out[nm]["adder_1b"] = AladdinAdder(tech_node=node, width=1).read().energy * 1e12
    return out


def run_colonnade(wb, ib, ob=16, N=None, K=128, cols=128, regs=16, rows_per_reg=8, node=65e-9, volt=0.8):
    spec = get_spec("colonnade_jssc_2021", add_dummy_main_memory=True)
    spec.variables.voltage = volt
    spec.variables.tech_node = node
    spec.variables.supported_input_bits = 24
    spec.variables.supported_output_bits = 48
    spec.arch.variables["force_100mhz"] = True
    spec.arch.variables["n_regs_per_col"] = regs
    spec.arch.variables["n_rows_per_reg"] = rows_per_reg
    spec.arch.find_spatial("group_ARRAY_ROWS").fanout = regs
    spec.arch.find_spatial("row_ARRAY_ROWS").fanout = rows_per_reg
    spec.arch.find_spatial("column_ARRAY_COLUMNS").fanout = cols
    spec.workload.rank_sizes["N"] = N if N else cols // (wb + 7)
    spec.workload.rank_sizes["K"] = K
    for es in spec.workload.einsums:
        for ta in es.tensor_accesses:
            ta.bits_per_value = {"input": ib, "weight": wb, "output": ob}[ta.name]
    r = spec.map_workload_to_arch(print_progress=False)
    per = {k: float(v) for k, v in r.energy(per_component=True).items()}
    n = float(r.n_computes())
    return per, n


def colonnade():
    # paper values read from the figure, as stored in AccelForge's notebook (fJ per MAC at 0.8 V, 100 MHz)
    ref = {"w8_i1": (8, 1, 144.3500315), "w1_i1": (1, 1, 68.31697711), "w16_i1": (16, 1, 243.1891829), "w8_i8": (8, 8, 1169.653111)}
    check = {}
    for name, (wb, ib, paper) in ref.items():
        per, n = run_colonnade(wb, ib)
        m = sum(per.values()) / n * 1e15
        check[name] = {"model_fJ_per_MAC": m, "paper_fJ_per_MAC": paper, "ratio": m / paper}
    shapes = {   # rows active, columns active (outputs), input bits, output bits
        "pw": (64, 16, 9, 23), "conv1": (40, 16, 9, 23), "dw": (64, 7, 9, 23), "fc": (64, 12, 9, 23),
        "fft1": (16, 16, 17, 31), "fft3": (64, 16, 17, 31), "mel_16": (64, 16, 22, 36), "mel_8": (64, 8, 22, 36),
        "dct_16": (40, 16, 14, 28), "dct_4": (40, 4, 14, 28)}
    passes = {}
    for nm, node in (("65", 65e-9), ("28", 28e-9)):
        passes[nm] = {}
        for s, (K, N, b, ob) in shapes.items():
            per, _ = run_colonnade(8, b, ob=ob, N=N, K=K, cols=224, regs=8, rows_per_reg=8, node=node)
            per.pop("WeightDrivers", None)                                 # weights are resident during inference
            per.pop("MainMemory", None)
            passes[nm][s] = {"pJ_per_pass": sum(per.values()) * 1e12,
                             "by_component_pJ": {k: v * 1e12 for k, v in per.items() if v > 0}}
    return check, passes


def identity():
    def git(p):
        return subprocess.run(["git", "-C", p, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import importlib.metadata as md
    return {"accelforge_commit": git(AF), "python": sys.version.split()[0],
            **{p: md.version(p) for p in ("hwcomponents", "hwcomponents-cacti", "hwcomponents-library", "accelforge")}}


if __name__ == "__main__":
    res = {"tools": identity(), "cacti": cacti(), "aladdin": aladdin()}
    res["colonnade_check"], res["colonnade_passes"] = colonnade()
    with open(os.path.join(HERE, "energy_components.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["tools"]))
    for k, v in res["colonnade_check"].items():
        print(f"Colonnade {k}: model {v['model_fJ_per_MAC']:.1f} fJ/MAC, paper {v['paper_fJ_per_MAC']:.1f}, ratio {v['ratio']:.2f}")
    for nm in res["colonnade_passes"]:
        print(nm, {k: round(v["pJ_per_pass"], 1) for k, v in res["colonnade_passes"][nm].items()})
    for k, v in res["cacti"]["28"].items():
        print(f"28 nm {k:14s} read {v['read_pJ']:.3f} pJ write {v['write_pJ']:.3f} pJ leak {v['leak_uW']:.3f} uW ({v['pW_per_bit']:.1f} pW/bit)")
    print({k: round(v, 3) for k, v in res["aladdin"]["28"].items()})
