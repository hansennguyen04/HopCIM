#!/usr/bin/env python3
"""CIM timing sensitivity (docs/TIMING_SENSITIVITY.md). Runs the real benches with the behavioural engine
tb/cim_engine_tm.sv at many macro timings. Standard library only.

  timing_sweep.py seq     cycles per inference over issue interval and result latency, bit-exactness
  timing_sweep.py hop     for each macro assumption: worst hop in the live path, the lowest real-time
                          divider, confirmed by running at it and one below
  timing_sweep.py report  tables from model/timing_results.json
The D in the tables is the interval of a network pass (b = 9): ceil(9 / BPC) * CPB, at least DMIN, + D0.
"""
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "model", "timing_results.json")
JOBS = int(os.environ.get("JOBS", "8"))
HOP = 320 * 64                                       # cycles per hop per unit of the PDM divider


def d_net(c):
    return max(c.get("dmin", 8), math.ceil(9 / c.get("bpc", 1)) * c.get("cpb", 1)) + c.get("d0", 0)


def plus(c):
    a = [f"+CIM_DMIN={c.get('dmin', 8)}", f"+CIM_BPC={c.get('bpc', 1)}", f"+CIM_CPB={c.get('cpb', 1)}",
         f"+CIM_D0={c.get('d0', 0)}", f"+CIM_LAT={c.get('lat', 5)}"]
    # the largest D of any pass (b = 24), so the bank hold covers it; the network's bank is what matters
    a.append(f"+CIM_HOLD={max(7, d_net(c) - 2)}")
    return a


def seq_configs():
    fast = dict(dmin=1)
    cfgs = []
    # issue interval at the RTL's latency
    for bpc in (24, 4, 2):
        cfgs.append(dict(bpc=bpc, lat=5, **fast))
    for d0 in (0, 1, 3, 5, 7, 9, 11, 15, 23, 39):
        cfgs.append(dict(d0=d0, lat=5))
    for cpb in (2, 3, 4, 8, 16):
        cfgs.append(dict(cpb=cpb, lat=5))
    # latency at the RTL's interval
    for lat in (1, 2, 3, 8, 12, 16, 24, 32, 48, 64, 128):
        cfgs.append(dict(lat=lat))
    # both
    for bpc, cpb in ((24, 1), (2, 1), (1, 2), (1, 4)):
        for lat in (1, 16, 64):
            cfgs.append(dict(bpc=bpc, cpb=cpb, dmin=1, lat=lat))
    return cfgs


def parse_seq(txt):
    m = re.search(r"full inference (\d+)\.\.(\d+) cycles", txt)
    tm = re.search(r"mean D ([\d.]+), at most (\d+) in flight, (\d+) hazards", txt)
    mism = len(re.findall(r"^MISMATCH", txt, re.M))
    layers = [int(v) for v in re.findall(r": (\d+) / 8000 mismatches", txt)]
    syn = [int(v) for v in re.findall(r"synthetic parameters, logits and class: (\d+) errors", txt)]
    return dict(cycles=int(m.group(1)) if m else None, cycles_max=int(m.group(2)) if m else None,
                data_errors=mism + sum(layers) + sum(syn), layer_checks=len(layers),
                in_flight=int(tm.group(2)) if tm else None, hazards=int(tm.group(3)) if tm else None,
                bench_lines=len(re.findall(r"^(ERROR|CONTRACT)", txt, re.M)))


def run_seq(c):
    b = os.path.join(ROOT, "sim", "obj_tm_seq_cim_hold", "Vtb_cim_seq")
    r = subprocess.run([b] + plus(c), cwd=ROOT, capture_output=True, text=True)
    return dict(cfg=c, d=d_net(c), lat=c.get("lat", 5), **parse_seq(r.stdout))


def model_cycles(d, lat):
    """2,500 passes of 16 outputs (conv1, pointwise) are paced by requantisation; 5,000 depthwise passes
    by the gather (9 chunks); at most 4 passes in flight; the rest is per-layer overhead."""
    d16 = max(d, 16, (d + lat) / 4)
    d9 = max(d, 9, (d + lat) / 4)
    return 2500 * d16 + 5000 * d9 + 10 * (lat + 30)


def do_seq():
    subprocess.run(["bash", "sim/run_timing.sh", "build", "seq", "cim"], cwd=ROOT, check=True, env=dict(os.environ, HOLD="1"))
    res = []
    with cf.ThreadPoolExecutor(JOBS) as ex:
        for x in ex.map(run_seq, seq_configs()):
            res.append(x)
            print(f"D {x['d']:3d} L {x['lat']:3d}: {x['cycles']} cycles, data errors {x['data_errors']}, hazards {x['hazards']}, "
                  f"in flight {x['in_flight']}, model {model_cycles(x['d'], x['lat']):.0f}", flush=True)
    return res


# macro assumptions, taken from ZigZag-IMC's validated table, AccelForge's Colonnade model and the spec.
# L is inferred (pipeline stages), not quoted; a macro on its own fast clock reaches D = 1.
ASSUME = [
    ("baseline", "HopCIM engine: 1 bit per cycle, 5-cycle pipeline (ENGINE_SPEC)", dict(lat=5)),
    ("2bpc_nopipe", "2 bits per cycle, no adder-tree pipeline (ISSCC'22 15.5, ISSCC'23 7.2 in ZigZag-IMC), on the system clock", dict(bpc=2, dmin=1, lat=2)),
    ("1bpc_nopipe", "1 bit per cycle, no adder-tree pipeline (ISSCC'23 16.3 in ZigZag-IMC), on the system clock", dict(dmin=1, lat=2)),
    ("1bpc_pipe", "1 bit per cycle, pipelined adder tree, log2(64) + 2 stages (ISSCC'22 11.7 in ZigZag-IMC)", dict(dmin=1, lat=8)),
    ("colonnade_bp", "1 bit per cycle, bit-pipelined columns, weight + log2(64) + 2 stages (Colonnade, as AccelForge models it)", dict(dmin=1, lat=16)),
    ("own_clock", "own fast macro clock (3-5.5 ns in ZigZag-IMC's table: 9 bits in under one system cycle), 2-cycle synchroniser", dict(bpc=24, dmin=1, lat=2)),
    ("two_steps", "2 system cycles per bit, or a tile split into 2 time steps (32 rows or 8 columns per step)", dict(cpb=2, dmin=1, lat=5)),
    ("four_steps", "a tile split into 4 time steps (32 rows and 8 columns per step)", dict(cpb=4, dmin=1, lat=5)),
    ("colonnade_np", "Colonnade-style columns without bit pipelining: 17 system cycles per bit (AccelForge's scale)", dict(cpb=17, dmin=1, lat=2)),
]


BUILT, BUILT_LOCK = {}, threading.Lock()


def mic_bin(n):
    """The bench at PDM divider n: one build per divider, whichever assumption asks first."""
    with BUILT_LOCK:
        entry = BUILT.setdefault(n, dict(lock=threading.Lock(), path=None))
    with entry["lock"]:
        if entry["path"] is None:
            env = dict(os.environ, HOLD="1", JOBS="4")
            r = subprocess.run(["bash", "sim/run_timing.sh", "build", "mic", "cim", str(n)], cwd=ROOT, env=env, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stdout[-2000:])
            entry["path"] = r.stdout.strip().splitlines()[-1]
    return entry["path"]


def run_mic(b, c):
    r = subprocess.run([b] + plus(c), cwd=ROOT, capture_output=True, text=True)
    t = r.stdout
    h = re.search(r"HOPMON (\d+) hops: worst (\d+) cycles; with the network: worst (\d+), mean (\d+); without: worst (\d+)", t)
    return dict(passed=bool(re.search(r"^PASS", t, re.M)), overrun=len(re.findall(r"overrun at decision", t)),
                wrong=[int(v) for v in re.findall(r"decisions: (\d+) of", t)],
                hazards=int(re.search(r"(\d+) hazards", t).group(1)) if "hazards" in t else None,
                worst=int(h.group(2)) if h else None, mean_net=int(h.group(4)) if h else None,
                worst_nonet=int(h.group(5)) if h else None)


def do_hop_one(item):
    name, text, c = item
    d = d_net(c)
    pred = model_cycles(d, c.get("lat", 5)) + 12500  # to choose a first divider with room
    first = max(math.ceil(pred / HOP) + 2, 6)
    a = run_mic(mic_bin(first), c)
    worst = a["worst"]
    nmin = math.ceil(worst / HOP)
    while nmin * HOP - worst < 16:                   # the hop end can fall inside the last sample
        nmin += 1
    at = run_mic(mic_bin(nmin), c)
    below = run_mic(mic_bin(nmin - 1), c) if nmin > 1 else None
    print(f"{name}: D {d}, worst hop {worst} (at divider {first}), lowest divider {nmin} = {nmin * 1.024:.2f} MHz; "
          f"at it pass={at['passed']} overrun={at['overrun']}; below pass={below and below['passed']} overrun={below and below['overrun']}", flush=True)
    return dict(name=name, text=text, cfg=c, d=d, lat=c.get("lat", 5), worst=worst, mean_net=a["mean_net"], worst_nonet=a["worst_nonet"],
                first_divider=first, first_pass=a["passed"], nmin=nmin, fmin_mhz=nmin * 1.024, at=at, below=below)


def do_hop():
    with cf.ThreadPoolExecutor(min(JOBS, 4)) as ex:
        return list(ex.map(do_hop_one, ASSUME))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if cmd == "seq":
        res["seq"] = do_seq()
    elif cmd == "hop":
        res["hop"] = do_hop()
    elif cmd != "report":
        sys.exit(__doc__)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    if res.get("seq"):
        errs = [abs(x["cycles"] - model_cycles(x["d"], x["lat"])) / x["cycles"] for x in res["seq"] if x["cycles"]]
        print(f"model against RTL benches: worst error {100 * max(errs):.2f} %, over {len(errs)} timings")
        done = [x for x in res["seq"] if x["cycles"]]
        print(f"{len(done)} of {len(res['seq'])} timings ran to the end, all checks bit-exact at {sum(x['data_errors'] == 0 for x in done)}; "
              f"hazards at {sum(bool(x['hazards']) for x in res['seq'])}")
        for x in res["seq"]:
            if not x["cycles"]:
                print(f"  D {x['d']} L {x['lat']}: stopped by the bench's 400-ms watchdog, {x['data_errors']} data errors until then")


if __name__ == "__main__":
    main()
