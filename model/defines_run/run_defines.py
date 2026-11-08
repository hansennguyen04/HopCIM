#!/usr/bin/env python3
"""V4 of docs/STREAMFUSE_GATES.md: DeFiNES (Mei, Goetschalckx, Symons, Verhelst, HPCA 2023; ZigZag-Project/DeFiNES)
on HopCIM's DS-CNN, layer by layer and depth-first, as an independent check of the activation traffic that
model/streamfuse_sched.py counts. Run with the tool's environment:
    ~/venvs/defines/bin/python model/defines_run/run_defines.py
DeFiNES chooses its own loop orders (LOMA), so it is compared on bits moved at the activation memories, not on a
schedule. Writes model/defines_run/run.log and model/streamfuse_defines.json."""
import json
import logging
import os
import pickle
import subprocess
import sys

TOOL = os.path.expanduser("~/tools/energy/DeFiNES")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "streamfuse_defines.json")
sys.path.insert(0, TOOL)                  # to unpickle the tool's result objects
sys.path.insert(0, HERE)


def run_case(lb, tilex, tiley, mode, path):
    """One DeFiNES run in its own process (the accelerator module reads HOPCIM_LB at import)."""
    code = f"""
import sys, logging
sys.path.insert(0, {TOOL!r}); sys.path.insert(0, {HERE!r})
logging.basicConfig(level=logging.WARNING)
from classes.stages import *
modes = {{'1': (False, False), '2': (True, False), '3': (True, True)}}
MainStage([WorkloadAndAcceleratorParserStage, GeneralParameterIteratorStage, SkipIfDumpExistsStage, DumpStage,
           DfStackCutIfWeightsOverflowStage, DepthFirstStage, SpatialMappingConversionStage, RemoveExtraInfoStage,
           MinimalEnergyStage, LomaStage, ZigZagCostModelStage],
          accelerator_path='hopcim_hw', workload_path='hopcim_dscnn_wl', loma_lpf_limit=6,
          df_tilesize_x={tilex}, df_tilesize_y={tiley}, headname='hopcim', result_saving_path={os.path.dirname(path)!r},
          general_parameter_iterations={{('df_horizontal_caching', 'df_vertical_caching'): (modes[{mode!r}],)}},
          dump_filename_pattern={path!r}).run()
"""
    env = dict(os.environ, HOPCIM_LB="1" if lb else "0")
    r = subprocess.run([sys.executable, "-c", code], cwd=TOOL, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    with open(path, "rb") as f:
        return pickle.load(f)


def energy_by_memory(results):
    """Energy (pJ) read and written at each named memory, over every layer, tile and data copy, from DeFiNES's own
    breakdown; with the access costs this also gives the number of accesses."""
    out = {}
    for cme, extra in results:
        core = cme.accelerator.get_core(cme.core_allocation if hasattr(cme, "core_allocation") else cme.layer.core_allocation)
        hier = core.get_memory_hierarchy_dict()
        for op, levels in cme.energy_breakdown_further.items():
            mem_op = op if op in hier else cme.layer.memory_operand_links[op]
            for lv, e in enumerate(levels):
                inst = hier[mem_op][lv].memory_instance
                d = out.setdefault(inst.name, dict(read_pJ=0.0, write_pJ=0.0, r_cost=inst.r_cost, w_cost=inst.w_cost))
                d["read_pJ"] += e.rd_out_to_low + e.rd_out_to_high
                d["write_pJ"] += e.wr_in_by_low + e.wr_in_by_high
    for d in out.values():
        d["reads"] = d["read_pJ"] / d["r_cost"] if d["r_cost"] else None
        d["writes"] = d["write_pJ"] / d["w_cost"] if d["w_cost"] else None
    return out


def act_by_layer(results):
    """Words read and written at the activation memories, per layer and operand (data copies under their own name)."""
    per = {}
    for cme, extra in results:
        core = cme.accelerator.get_core(cme.core_allocation if hasattr(cme, "core_allocation") else cme.layer.core_allocation)
        hier = core.get_memory_hierarchy_dict()
        lid = str(cme.id) if type(cme).__name__ == "DataCopyLayer" else str(cme.layer.id)
        for op, levels in cme.energy_breakdown_further.items():
            mem_op = op if op in hier else cme.layer.memory_operand_links[op]
            for lv, e in enumerate(levels):
                inst = hier[mem_op][lv].memory_instance
                if inst.name in ("act", "line_buffers"):
                    d = per.setdefault(f"{lid} {inst.name} {op}", [0.0, 0.0])
                    d[0] += (e.rd_out_to_low + e.rd_out_to_high) / inst.r_cost
                    d[1] += (e.wr_in_by_low + e.wr_in_by_high) / inst.w_cost
    return per


# the schedule model, per depthwise and pointwise layer (model/streamfuse_results.json, design S)
SCHED = dict(dw_reads_valid=9490, dw_chunks_all=11250, pw_reads=1250, writes_bytes=8000)


def main():
    scratch = os.environ.get("DEFINES_SCRATCH", os.path.join(HERE, "pickles"))
    os.makedirs(scratch, exist_ok=True)
    cases = {"layer by layer": (False, 5, 25, "3"), "depth-first, one row, fully cached": (True, 5, 1, "3")}
    res = {}
    for name, (lb, tx, ty, mode) in cases.items():
        p = os.path.join(scratch, f"{'df' if lb else 'lbl'}_{tx}_{ty}_{mode}.pkl")
        if os.path.exists(p):
            os.remove(p)
        results = run_case(lb, tx, ty, mode, p)
        mem = energy_by_memory(results)
        act = {k: v for k, v in mem.items() if k in ("act", "line_buffers")}
        res[name] = dict(memories=mem, activation_nJ=sum(v["read_pJ"] + v["write_pJ"] for v in act.values()) / 1000,
                         results=len(results), per_layer=act_by_layer(results))
        print(f"{name}: activation memories {res[name]['activation_nJ']:.1f} nJ; "
              + "; ".join(f"{k} reads {v['reads']:.0f} ({v['read_pJ'] / 1000:.1f} nJ), writes {v['writes']:.0f} ({v['write_pJ'] / 1000:.1f} nJ)"
                          for k, v in act.items()), flush=True)
    # the last block (layers 7 and 8) is the only one DeFiNES computes at HopCIM's size: it has no SAME padding, so
    # every earlier layer grows by the receptive field of the layers after it (conv1 at 33 x 13 instead of 25 x 5)
    pl = res["layer by layer"]["per_layer"]
    dw7, pw8 = pl["7 act I"][0], pl["8 act I"][0]
    w7 = pl.get("7 act O", [0, 0])[1]
    cmp = dict(dw7_reads=dw7, dw_vs_valid=dw7 / SCHED["dw_reads_valid"], dw_vs_all_chunks=dw7 / SCHED["dw_chunks_all"],
               pw8_reads=pw8, pw_vs_sched=pw8 / SCHED["pw_reads"], dw7_writes_words=w7,
               writes_vs_sched=w7 / (SCHED["writes_bytes"] * 8 / 56),
               reduction_defines=1 - res["depth-first, one row, fully cached"]["activation_nJ"] / res["layer by layer"]["activation_nJ"])
    res["last block against the schedule model"] = cmp
    print(f"last block, layer by layer: depthwise reads {dw7:.0f} against {SCHED['dw_reads_valid']} valid taps "
          f"({cmp['dw_vs_valid']:.3f}) or {SCHED['dw_chunks_all']} with the padded taps ({cmp['dw_vs_all_chunks']:.3f}); "
          f"pointwise reads {pw8:.0f} against {SCHED['pw_reads']} ({cmp['pw_vs_sched']:.3f}); depthwise output writes "
          f"{w7:.0f} words against {SCHED['writes_bytes']} bytes = {SCHED['writes_bytes'] * 8 / 56:.0f} words ({cmp['writes_vs_sched']:.3f})")
    print(f"layer by layer -> depth-first, activation-memory energy: {100 * cmp['reduction_defines']:.0f} % less in DeFiNES")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
