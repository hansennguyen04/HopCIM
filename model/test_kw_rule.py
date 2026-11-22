#!/usr/bin/env python3
"""The firmware's summed-logit rule (hopcim/sw/kw_rule.h, hopcim_live built with KW_SUM) against the rule the
whole-word study scored, kwlog_analyse.apply_rule(("smooth", KW_SUM, hold), ...).

kw_rule.h is compiled on the host (gcc, -Wall -Wextra -Werror) and fed the same decisions:
  - the spoken runs in hopcim_recordings/spk01_wholeword_run* (every 20-ms decision; at k = 16 every phase, and k = 1, 15);
  - the 21 test streams of model/wholeword/ at k = 16 (every phase for KW_SUM = 2);
  - 200,000 synthetic decisions drawn from a few values, so that ties, silence resets and long runs are frequent.
Every reported (decision, keyword) must be equal. Three broken copies of kw_rule.h must each be caught.

Usage: test_kw_rule.py          (needs gcc; about a minute)
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

import kwlog_analyse as ka

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "..", "sw", "kw_rule.h")
RUNS = sorted(glob.glob(os.path.join(HERE, "..", "..", "hopcim_recordings", "spk01_wholeword_run*")))
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "kw_rule.h"
int main(int argc, char **argv) {
    (void)argc;
    uint32_t hold = (uint32_t)atoi(argv[1]);
    kw_rule_t r;
    kw_rule_init(&r);
    int8_t lg[12];
    unsigned i = 0;
    while (fread(lg, 1, 12, stdin) == 12) {
        int kw = kw_rule_step(&r, lg, hold);
        if (kw >= 0) printf("%u %d\n", i, kw);
        i++;
    }
    return 0;
}
"""
MUTANTS = {                                          # each must make the test fail
    "ties go to the higher class": ("if (r->sum[c] > r->sum[top]) top = c;", "if (r->sum[c] >= r->sum[top]) top = c;"),
    "no reset by a non-keyword": ("if (top >= 10) r->said = -1;", ";"),
    "the oldest decision is never removed": ("if (r->have == (KW_SUM)) r->sum[c] -= r->ring[r->slot][c];", ";"),
}


def build(tmp, header_text, n, tag):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "kw_rule.h"), "w") as f:
        f.write(header_text)
    with open(os.path.join(d, "t.c"), "w") as f:
        f.write(HARNESS)
    exe = os.path.join(d, f"t{n}")
    subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", f"-DKW_SUM={n}", "-I", d, os.path.join(d, "t.c"), "-o", exe], check=True)
    return exe


def c_events(exe, logits, hold):
    out = subprocess.run([exe, str(hold)], input=np.ascontiguousarray(logits, np.int8).tobytes(), capture_output=True, check=True).stdout
    return [tuple(int(v) for v in line.split()) for line in out.decode().splitlines()]


def py_events(logits, n, hold):
    return [(int(h), int(c)) for h, c in ka.apply_rule(("smooth", n, hold), logits.astype(np.int64), np.arange(len(logits)))]


def cases(n):
    """(name, int8 logits [N, 12], hold) to compare."""
    out = []
    for run in RUNS:
        L = ka.load(run)["logits"].astype(np.int8)
        for k in ((1, 15, 16) if n == 2 else (16,)):
            for p in (range(k) if n == 2 and k == 16 else (0,)):
                out.append((f"{os.path.basename(run)} k={k} phase {p}", L[p::k], 1))
    for f in sorted(glob.glob(os.path.join(HERE, "wholeword", "*_s5*.npz"))):
        L = np.load(f)["logits"].astype(np.int8)
        for p in (range(16) if n == 2 else (0,)):
            out.append((f"{os.path.basename(f)[:-4]} k=16 phase {p}", L[p::16], 1))
    rng = np.random.default_rng(20261122 + n)
    vals = np.array([-128, -127, -60, 0, 0, 5, 5, 127], np.int8)
    L = vals[rng.integers(len(vals), size=(200000, 12))]
    for hold in (1, 2, 3):
        out.append((f"synthetic ties hold {hold}", L, hold))
    return out


def main():
    text = open(HEADER).read()
    total = bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for n in (1, 2, 3, 4, 16):
            exe = build(tmp, text, n, "ok")
            for name, L, hold in cases(n):
                a, b = c_events(exe, L, hold), py_events(L, n, hold)
                total += 1
                if a != b:
                    bad += 1
                    print(f"DIFFER KW_SUM={n} {name}: firmware {len(a)} reports, model {len(b)}")
            print(f"KW_SUM = {n}: {sum(1 for _ in cases(n))} cases compared", flush=True)
        print(f"{total} cases, {bad} differ -> {'pass' if bad == 0 else 'FAIL'}")
        caught = 0
        for name, (old, new) in MUTANTS.items():
            assert text.count(old) == 1, name
            exe = build(tmp, text.replace(old, new), 2, "mut")
            diff = sum(c_events(exe, L, hold) != py_events(L, 2, hold) for _, L, hold in cases(2)[-3:] + cases(2)[:2])
            caught += diff > 0
            print(f"mutant '{name}': {'caught' if diff else 'NOT caught'} ({diff} cases differ)")
        print(f"mutants caught: {caught} of {len(MUTANTS)}")
    sys.exit(0 if bad == 0 and caught == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
