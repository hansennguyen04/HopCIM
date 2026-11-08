#!/usr/bin/env python3
"""Run ZigZag-IMC itself on HopCIM's macro and compare with model/energy_est.py's transcribed equations.
Run with the tool's environment:  ~/venvs/zigzag/bin/python model/zigzag_run/run_zigzag.py
Tool: ~/tools/energy/zigzag-imc (KULeuven-MICAS/zigzag-imc, commit df34a6f). Workload: its own
mlperf_tiny/ds_cnn.onnx, the same network. Writes model/zigzag_results.json.
"""
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

TOOL = os.path.expanduser("~/tools/energy/zigzag-imc")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ONNX = "zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx"
# transcribed gate energies, as in energy_est.py (fJ)
V2 = 0.9 ** 2
ND2, XOR2, DFF = 0.7, 0.7 * 1.5, 0.7 * 3
FA = (2 * XOR2 + 3 * ND2) * V2
MULT_BIT = ND2 / 2 * V2
SHAPES = {"conv1": dict(col_passes=8000, rows=40), "dw": dict(col_passes=32000, rows=63),
          "pw": dict(col_passes=32000, rows=64), "fc": dict(col_passes=12, rows=64)}


# rows with weights in one column: a depthwise column uses only its own channel's 9 taps
ACTIVE_ROWS = {"conv1": 40, "dw": 9, "pw": 64, "fc": 64}


def tree_adders(n, bits):
    depth = {64: 6}[n]
    return n * (bits + 1) - (bits + depth + 1)


def transcribed(group, b=8, alpha=1.0):
    """energy_est.Model.macro_pass_pj and select_pj, in pJ, per group of layers."""
    out = {}
    for g, s in SHAPES.items():
        col = MULT_BIT * 8 * 64 + tree_adders(64, 8) * FA
        acc = (b + 14) * (FA + DFF * V2)
        array = s["col_passes"] * b * (alpha * col + acc) / 1000
        sel = s["rows"] * s["col_passes"] * (0.35 + 0.35 * group) * V2 * 8 / 1000 if group > 1 else 0.0
        out[g] = dict(array=array, select=sel)
    return out


def run_tool(group, macros, out):
    env = dict(os.environ, ZZ_GROUP=str(group), ZZ_MACROS=str(macros),
               PYTHONPATH=os.pathsep.join([HERE, TOOL]))
    code = f"""
import logging, sys
logging.basicConfig(level=logging.WARNING)
from zigzag.classes.stages import *
m = MainStage([ONNXModelParserStage, AcceleratorParserStage, CompleteSaveStage, SearchUnusedMemoryStage, WorkloadStage,
               RemoveUnusedMemoryStage, MinimalEDPStage, SpatialMappingGeneratorStage, LomaStage, CostModelStage],
              accelerator="hopcim_zz_hw", workload={ONNX!r},
              mapping="zigzag.inputs.examples.mapping.default_imc",
              dump_filename_pattern={os.path.join(out, 'layer_?.json')!r},
              pickle_filename={os.path.join(out, 'x.pickle')!r},
              loma_lpf_limit=6, loma_show_progress_bar=False,
              enable_mix_spatial_mapping_generation=True, maximize_hardware_utilization=True,
              enable_weight_diagonal_mapping=True)
m.run()
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=TOOL, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    layers = []
    for f in glob.glob(os.path.join(out, "layer_*_complete.json")):
        d = json.load(open(f))
        idx = int(os.path.basename(f).split("LayerNode_")[1].split("_")[0])
        e = d["outputs"]["energy"]
        lat = d["outputs"]["latency"]
        layers.append(dict(idx=idx, op=e["operational_energy"], ops=e["operational_energy_breakdown"], mem=e["memory_energy"],
                           mem_by=e["memory_energy_breakdown_per_level"], cycles=lat["computation"], breakdown=lat["computation_breakdown"],
                           mac=lat["computation_breakdown"]["mac_computation"],
                           clock=d["outputs"]["clock"], area=d["outputs"]["area (mm^2)"],
                           util=d["outputs"]["spatial"]["mac_utilization"],
                           spatial=d["outputs"]["spatial"]))
    layers.sort(key=lambda x: x["idx"])
    return layers


def group_layers(layers):
    """ONNX node order: conv1, then depthwise and pointwise pairs, then the fully connected layer."""
    g = {"conv1": [], "dw": [], "pw": [], "fc": []}
    for i, l in enumerate(layers):
        if i == 0:
            g["conv1"].append(l)
        elif i == len(layers) - 1:
            g["fc"].append(l)
        elif i % 2 == 1:
            g["dw"].append(l)
        else:
            g["pw"].append(l)
    return g


def like_for_like(run):
    """The tool's structure at HopCIM's measured activity, against the transcribed equations at the same settings.
    The tool has no activity input: its adders and multipliers switch fully on the rows it maps. alpha in
    energy_activity.json is averaged over all 64 rows, so on the mapped rows it is alpha * 64 / rows."""
    act = json.load(open(os.path.join(MODEL, "energy_activity.json")))["groups"]
    est = json.load(open(os.path.join(MODEL, "energy_results.json")))
    print("\nThe tool's structure at the measured activity (input 8 bits, as the tool requires), nJ per inference:")
    print("  part    alpha on mapped rows   tool structure   transcribed   ratio")
    out = {}
    tt = tm = 0.0
    for g, s in SHAPES.items():
        r = run["rows"][g]["tool"]
        a = act[g]["alpha"]
        am = min(a * 64 / ACTIVE_ROWS[g], 1.0)
        tool = (r["mults"] + r["adders"]) * am + r["accumulators"]
        mine = transcribed(16, 8, a)[g]["array"]
        out[g] = dict(alpha=a, alpha_mapped=am, tool_pJ=tool, transcribed_pJ=mine, ratio=tool / mine)
        print(f"  {g:6s} {am:12.3f} {tool / 1000:18.1f} {mine / 1000:13.1f} {tool / mine:8.3f}")
        tt += tool
        tm += mine
    print(f"  total {'':12s} {tt / 1000:18.1f} {tm / 1000:13.1f} {tt / tm:8.3f}")
    base = est["network_pJ"]["measured alpha"]
    b_tot = sum(base.values())
    arr, sel = base["CIM array"], base["tile select"]
    raw = run["total_tool"]
    new_tot = b_tot - arr - sel + raw
    bud = est["budget_uW"]["cim"]
    upper = {}
    print(f"\nThe estimate with the tool's raw figure (full switching, its mapping, input 8 bits) in place of array and tile select:")
    print(f"  array + select {(arr + sel) / 1000:.0f} nJ -> {raw / 1000:.0f} nJ; network {b_tot / 1000:.0f} -> {new_tot / 1000:.0f} nJ per inference")
    for k in ("1", "2", "4"):
        tot = sum(bud[k].values())
        add = (raw - arr - sel) * 1e-12 / (0.02 * int(k)) * 1e6
        upper[k] = dict(baseline_uW=tot, with_tool_uW=tot + add)
        print(f"  a decision every {20 * int(k)} ms: {tot:.0f} -> {tot + add:.0f} uW")
    return dict(groups=out, total_tool_pJ=tt, total_transcribed_pJ=tm, ratio=tt / tm, network_baseline_pJ=b_tot,
                network_with_tool_pJ=new_tot, power=upper)


def main():
    results = {}
    print("ZigZag-IMC commit", subprocess.run(["git", "-C", TOOL, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
    for group, macros in ((16, 1), (1, 1), (16, 8)):
        out = tempfile.mkdtemp(prefix="zz_")
        layers = run_tool(group, macros, out)
        shutil.rmtree(out)
        g = group_layers(layers)
        tr = transcribed(group)
        print(f"\ngroup depth {group}, {macros} macro(s): {len(layers)} layers; clock {layers[0]['clock']}, area {layers[0]['area']}")
        print("  part     tool mults  adders  accum  precharge   sum pJ   | transcribed array  select   | ratio tool/transcribed   tool cycles (mac)")
        rows = {}
        tt = ta = 0.0
        for name, ls in g.items():
            br = {k: sum(l["ops"][k] for l in ls) for k in ("precharging", "mults", "adders", "accumulators")}
            tool = sum(br.values())
            mine = tr[name]["array"] + tr[name]["select"]
            cyc = sum(l["mac"] for l in ls)
            print(f"  {name:6s} {br['mults']:10.0f} {br['adders']:10.0f} {br['accumulators']:9.0f} {br['precharging']:9.0f} {tool:10.0f} |"
                  f" {tr[name]['array']:10.0f} {tr[name]['select']:9.0f} | {tool / mine:6.3f}   {cyc:9.0f}")
            rows[name] = dict(tool=br, tool_pJ=tool, transcribed=tr[name], ratio=tool / mine, tool_cycles=cyc,
                              util=[l["util"] for l in ls], mem_pJ=sum(l["mem"] for l in ls))
            tt += tool
            ta += mine
        for name, ls in g.items():
            print(f"    {name}: MAC utilisation {[round(l['util'], 3) if not isinstance(l['util'], dict) else l['util'] for l in ls][:1]}")
        print(f"  total  tool {tt:.0f} pJ, transcribed at the same settings (input 8 bits, activity 1) {ta:.0f} pJ, ratio {tt / ta:.3f}")
        print(f"  tool memory energy (its template: 256 KB SRAM, DRAM) {sum(l['mem'] for l in layers):.0f} pJ; MAC cycles {sum(l['mac'] for l in layers):.0f}")
        results[f"group{group}_macros{macros}"] = dict(rows=rows, total_tool=tt, total_transcribed=ta, ratio=tt / ta,
                                                       mem_pJ=sum(l["mem"] for l in layers), mac_cycles=sum(l["mac"] for l in layers),
                                                       clock=layers[0]["clock"], area=layers[0]["area"])
    results["like_for_like"] = like_for_like(results["group16_macros1"])
    with open(os.path.join(MODEL, "zigzag_results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
