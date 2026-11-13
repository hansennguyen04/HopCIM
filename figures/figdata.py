#!/usr/bin/env python3
"""The numbers behind the paper figures, read from the frozen evidence. Standard library only.

  python3 figures/figdata.py           write figures/figures_data.json
  python3 figures/figdata.py --check   exit status 1 unless figures_data.json equals what the evidence gives now

figures/make_figures.py draws the figures from these numbers; freeze/freeze.py check runs the check. Every value
carries the file it was read from and the pattern or key that finds it, and the model counts a figure uses are
checked here against the RTL counters they were shown to equal.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "figures_data.json"

B = "freeze/board/logs/"
CENSUS_LOG = B + "census_microbench_nexys_board_20261027_003600.uart.log"
CPU_LOG = B + "hopcim_cpu_nexys_board_20261028_092050.uart.log"
MFCC_LOG = B + "hopcim_mfcc_nexys_board_20261028_090817.uart.log"
NET1_LOG = B + "hopcim_net1_nexys_board_20261028_090921.uart.log"
FEIRQ_LOG = B + "hopcim_fe_irq_nexys_board_20261028_090841.uart.log"
SEQ_LOG = B + "hopcim_seq_nexys_board_20261028_090933.uart.log"
STREAM_LOG = B + "hopcim_stream_nexys_board_20261028_091143.uart.log"
LIVE_LOG = B + "hopcim_live_nexys_board_20261105_231010.uart.log"
SEQ_S_LOG = B + "hopcim_seq_nexys_15m36_board_20261108_151228.uart.log"
SEQ_B1_LOG = B + "hopcim_seq_nexys_15m36_board_20261108_204404.uart.log"
UTIL_S_SAME = "syn/ooc_cim_endpoint_S_same_source_as_B1/utilization.rpt"
UTIL_B1 = "syn/ooc_cim_endpoint_FE_DED0_PDM_CLKDIV15_SEQ_S1_SEQ_B11/utilization.rpt"
SF = "model/streamfuse_results.json"
SFE = "model/streamfuse_energy.json"
EB1 = "model/energy_b1_rtl.json"
ES = "model/energy_s_rtl.json"

# the energy components, in the order they are stacked, and the model's parts each one sums
ENERGY_GROUPS = [
    ("CIM array and tile select", ["CIM array", "tile select"]),
    ("activation memory", ["ACT memory", "FEAT memory"]),
    ("requantisation", ["requant parameters", "requant arithmetic"]),
    ("sequencer logic and registers", ["sequencer logic", "input and result registers"]),
]


def log(path, pattern, group=1):
    m = re.search(pattern, (ROOT / path).read_text(errors="replace"))
    if not m:
        raise ValueError(f"{path}: no match for {pattern!r}")
    return dict(value=int(m[group]), source=path, find=pattern)


def key(path, *keys):
    v = json.loads((ROOT / path).read_text())
    for k in keys:
        v = v[k]
    return dict(value=v, source=path, find="/".join(keys))


def need(cond, what):
    if not cond:
        raise ValueError(f"cross-check failed: {what}")


def cpu_per_hop():
    rows = [
        ("A", "CPU only", "board: network; front-end from the CPU census",
         [("front-end: FFT", log(CENSUS_LOG, r"fft512_cplx_q15\s+(\d+) cyc")),
          ("front-end: power, mel, log, DCT", log(CENSUS_LOG, r"pwr_mel_log_dct\s+(\d+) cyc")),
          ("network", log(CPU_LOG, r"total (\d+) cycles"))]),
        ("B", "CPU drives the CIM", "board",
         [("front-end", log(MFCC_LOG, r"total\s+(\d+) cyc/frame")),
          ("network", log(NET1_LOG, r"clip 0: label \d+ predicted \d+, (\d+) cycles"))]),
        ("C", "hardware front-end and sequencer", "composed from board measurements",
         [("front-end input and readback", log(FEIRQ_LOG, r"CPU-active (\d+) \(max")),
          ("window path on the CPU", log(FEIRQ_LOG, r"window path \(finish\) (\d+) CPU cycles/frame")),
          ("network control", log(SEQ_LOG, r"streaming step \([^)]*\): (\d+)\.\.\d+ cycles"))]),
        ("D", "+ window path in hardware", "board",
         [("copying 320 samples", log(STREAM_LOG, r"copying 320 samples (\d+) cycles")),
          ("everything else", log(STREAM_LOG, r"everything else \d+\.\.\d+ \(mean (\d+)\)"))]),
        ("E", "+ live microphone path", "board, 15.36 MHz",
         [("interrupt, decision, keyword logic", log(LIVE_LOG, r"CPU-active per decision (\d+) cycles"))]),
    ]
    out = [dict(id=i, label=l, basis=b, parts=[dict(name=n, **v) for n, v in p], total=sum(v["value"] for _, v in p))
           for i, l, b, p in rows]
    t = {r["id"]: r["total"] for r in out}
    need(t["D"] == 4447 and t["E"] == 263, "D and E equal EVALUATION.md (4,447 and 263)")
    return dict(title="CPU-active cycles per 20-ms hop, one decision every hop",
                unit="CPU-active cycles per hop", scale="log", rows=out)


def network_cycles():
    rows = [
        ("CPU only, plain C", log(CPU_LOG, r"total (\d+) cycles")),
        ("CPU drives the CIM", log(NET1_LOG, r"clip 0: label \d+ predicted \d+, (\d+) cycles")),
        ("hardware sequencer (R6b)", log(SEQ_LOG, r"sequencer (\d+) cycles")),
        ("baseline S", log(SEQ_S_LOG, r"go -> interrupt\): (\d+)\.\.\d+ cycles")),
        ("B1: StreamFuse-B in block 1", log(SEQ_B1_LOG, r"go -> interrupt\): (\d+)\.\.\d+ cycles")),
    ]
    return dict(title="Network cycles per inference, measured on the board", unit="cycles per inference", scale="log",
                rows=[dict(label=l, **v) for l, v in rows])


def energy():
    bars = [
        ("R6b", "RTL = model", key(SFE, "baseline", "B0", "network_pJ")),
        ("S", "RTL counters", key(EB1, "variants", "baseline", "parts_S_pJ")),
        ("B1", "RTL counters", key(EB1, "variants", "baseline", "parts_B1_pJ")),
        ("B, four blocks", "model only, not built", key(SFE, "baseline", "B", "network_pJ")),
    ]
    parts = sorted(p for _, ps in ENERGY_GROUPS for p in ps)
    rows = []
    for label, basis, v in bars:
        need(sorted(v["value"]) == parts, f"{label}: the energy parts are {sorted(v['value'])}")
        groups = [dict(name=g, nJ=sum(v["value"][p] for p in ps) / 1000) for g, ps in ENERGY_GROUPS]
        rows.append(dict(label=label, basis=basis, source=v["source"], find=v["find"], parts_pJ=v["value"], groups=groups,
                         total_nJ=sum(v["value"].values()) / 1000))
    t = {r["label"]: r["total_nJ"] for r in rows}
    need(abs(t["R6b"] - key(ES, "variants", "baseline", "B0_nJ")["value"]) < 1e-6, "R6b total equals energy_s_rtl.json B0_nJ")
    need(abs(t["S"] - key(EB1, "variants", "baseline", "S_nJ")["value"]) < 1e-6, "S total equals energy_b1_rtl.json S_nJ")
    need(abs(t["B1"] - key(EB1, "variants", "baseline", "B1_nJ")["value"]) < 1e-6, "B1 total equals energy_b1_rtl.json B1_nJ")
    s_model = sum(key(SFE, "baseline", "S", "network_pJ")["value"].values()) / 1000
    need(round(t["B, four blocks"] / s_model, 3) == 0.837, "B / S (model) = 0.837, as frozen in claim S2")
    return dict(title="Network energy per inference [E], 28 nm, by component", unit="nJ per inference [E]",
                note="An estimate from exact activity counts, not a measurement; B is the schedule model's, not built",
                rows=rows)


def b1_vs_s():
    n = log("model/energy_counts_cim_s.log", r"(\d+) decisions expected")["value"]
    rows = [
        ("network cycles (RTL)", key(SF, "B1", "B1", "total"), key(SF, "B1", "S", "total")),
        ("activation-memory word reads", key(SF, "B1", "B1", "reads", "act"), key(SF, "cycles", "S", "counts", "reads", "act")),
        ("activation-memory byte writes", key(SF, "B1", "B1", "byte_writes", "act"), key(SF, "cycles", "S", "counts", "byte_writes", "act")),
        ("activation storage, bits", key(SF, "B1", "B1", "storage_bits"), key(SF, "cycles", "S", "storage_bits")),
        ("network energy [E], nJ", key(EB1, "variants", "baseline", "B1_nJ"), key(EB1, "variants", "baseline", "S_nJ")),
        ("total power at 40 ms [E], µW", key(EB1, "variants", "baseline", "power_B1_uW", "2"), key(EB1, "variants", "baseline", "power_S_uW", "2")),
        ("endpoint LUT", log(UTIL_B1, r"\| Slice LUTs\s+\|\s+(\d+) \|"), log(UTIL_S_SAME, r"\| Slice LUTs\s+\|\s+(\d+) \|")),
        ("endpoint flip-flops", log(UTIL_B1, r"\| Slice Registers\s+\|\s+(\d+) \|"), log(UTIL_S_SAME, r"\| Slice Registers\s+\|\s+(\d+) \|")),
    ]
    # the schedule model's counts against the RTL counters (claims U1 and S3 say they are equal)
    r = {m: (b["value"], s["value"]) for m, b, s in rows}
    need(r["activation-memory word reads"][0] == key(EB1, "act_reads")["value"], "B1 reads: model = RTL counters")
    need(r["activation-memory byte writes"][1] == key(ES, "act_byte_writes")["value"], "S byte writes: model = RTL counters")
    for name, path in (("B1", "model/energy_counts_cim_b1.log"), ("S", "model/energy_counts_cim_s.log")):
        w = log(path, r"EC seq act_writes (\d+)")["value"]
        need(log(path, r"(\d+) decisions expected")["value"] == n and w % n == 0, f"{name}: {path} counts whole inferences")
        need(w // n == r["activation-memory byte writes"][0 if name == "B1" else 1], f"{name} byte writes: model = RTL counters")
    need(r["network cycles (RTL)"] == (67817, 67841), "network cycles equal claim U1")
    return dict(title="B1 against S, same source", unit="B1 / S",
                rows=[dict(metric=m, b1=b, s=s, ratio=b["value"] / s["value"]) for m, b, s in rows])


def build():
    return dict(generated_by="figures/figdata.py; figures/make_figures.py draws them",
                figures=dict(fig1_cpu_active_per_hop=cpu_per_hop(), fig2_network_cycles=network_cycles(),
                             fig3_energy_breakdown=energy(), fig4_b1_vs_s=b1_vs_s()))


def current():
    """True if figures_data.json equals what the evidence gives now."""
    return OUT.exists() and json.loads(OUT.read_text()) == json.loads(json.dumps(build()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        ok = current()
        print("figures_data.json equals the evidence" if ok else "figures_data.json differs from the evidence: run figures/figdata.py")
        return 0 if ok else 1
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
