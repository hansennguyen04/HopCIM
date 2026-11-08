#!/usr/bin/env python3
"""Assertion and coverage report from sim/run_cov.sh (docs/VERIFICATION.md). Standard library only.
Usage: verif_report.py [sim/cov]  writes model/verif_results.json; prints assertions, per-module coverage,
the uncovered lines of the RTL, and the toggle activity of the logic per busy cycle.
"""
import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "sim", "cov")
RTL = ("cim_engine", "cim_fe", "mfcc_ded", "cim_log2fix16", "cim_seq", "pdm_dec", "cim_audio", "cim_endpoint")


def fields(key):
    """Verilator coverage keys are \\x01name\\x02value pairs."""
    return dict(p.split("\x02", 1) for p in key.split("\x01") if "\x02" in p)


def read_dat(path):
    with open(path, errors="replace") as f:
        for ln in f:
            if ln.startswith("C '"):
                i = ln.rindex("'")
                yield fields(ln[3:i]), int(ln[i + 1:])


def main():
    logs = sorted(glob.glob(os.path.join(COV, "*.log")))
    logs = [p for p in logs if not os.path.basename(p).startswith("build_")]
    active, failed, runs, bad = collections.Counter(), collections.Counter(), {}, []
    cover = collections.Counter()
    for p in logs:
        tag = os.path.basename(p)[:-4]
        txt = open(p, errors="replace").read()
        runs[tag] = bool(re.search(r"^PASS", txt, re.M))
        for m in re.finditer(r"^ASSERT (\S+) active (\d+) fail (\d+)", txt, re.M):
            active[m[1]] += int(m[2])
            failed[m[1]] += int(m[3])
        for m in re.finditer(r"^ASSERTION FAILED (.*)$", txt, re.M):
            bad.append((tag, m[1]))
        for m in re.finditer(r"^COVER (\S+) (\d+)", txt, re.M):
            cover[m[1]] += int(m[2])
    print(f"{len(runs)} benches, {sum(runs.values())} pass; {len(active)} assertions; "
          f"{sum(v > 0 for v in failed.values())} failed; {sum(v == 0 for v in active.values())} never active")
    for k in sorted(active):
        print(f"  {k:48s} active {active[k]:13d}  fail {failed[k]}")
    for t, m in bad[:20]:
        print("  FAILED", t, m)

    print("\nevents the benches must reach:")
    for k in sorted(cover):
        print(f"  {k:48s} {cover[k]:13d}")

    # coverage, merged over benches by file, line, column and kind: hit if any bench hit it
    pts = {}
    tog = collections.Counter()
    tog_n = collections.Counter()
    for p in sorted(glob.glob(os.path.join(COV, "*.dat"))):
        for f, n in read_dat(p):
            file = f.get("f", "")
            base = os.path.basename(file)
            mod = base[:-3] if base.endswith(".sv") else base
            if mod not in RTL:
                continue
            page = f.get("page", "")
            kind = page.split("/")[0]
            key = (mod, kind, f.get("l"), f.get("n", ""), f.get("o", ""), f.get("S", ""))
            pts[key] = pts.get(key, 0) + n
    summary = {}
    for (mod, kind, l, c, o, s), n in pts.items():
        d = summary.setdefault(mod, {}).setdefault(kind, [0, 0])
        d[1] += 1
        d[0] += n > 0
    print("\ncoverage points hit, all benches merged (a toggle point is one bit, one direction):")
    kinds = sorted({k for v in summary.values() for k in v})
    print("  module            " + "".join(f"{k:>18s}" for k in kinds))
    tot = collections.defaultdict(lambda: [0, 0])
    for mod in RTL:
        row = f"  {mod:18s}"
        for k in kinds:
            a, b = summary.get(mod, {}).get(k, [0, 0])
            tot[k][0] += a
            tot[k][1] += b
            row += f"{a:9d}/{b:<5d}{100 * a / max(b, 1):3.0f}%" if b else f"{'-':>18s}"
        print(row)
    print("  " + f"{'total':18s}" + "".join(f"{tot[k][0]:9d}/{tot[k][1]:<5d}{100 * tot[k][0] / max(tot[k][1], 1):3.0f}%" for k in kinds))
    miss = sorted({(m, int(l)) for (m, k, l, c, o, s), n in pts.items() if n == 0 and k in ("v_line", "v_branch")})
    print(f"\nuncovered line or branch points: {len(miss)} source lines")
    src = {}
    res_miss = []
    for m, l in miss:
        src.setdefault(m, open(os.path.join(ROOT, "rtl", m + ".sv"), errors="replace").read().split("\n"))
        text = src[m][l - 1].strip() if l - 1 < len(src[m]) else ""
        res_miss.append(dict(module=m, line=l, text=text))
        print(f"  {m}.sv:{l}: {text[:110]}")
    with open(os.path.join(ROOT, "model", "verif_results.json"), "w") as f:
        json.dump(dict(benches=runs, assertions={k: dict(active=active[k], fail=failed[k]) for k in sorted(active)},
                       cover=dict(cover), coverage=summary, totals={k: tot[k] for k in kinds}, uncovered=res_miss, failures=bad), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
