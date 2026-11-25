#!/usr/bin/env python3
"""Cycle prediction of ResNet8 on the descriptor-driven sequencer (cim_gseq), frozen with docs/NET2_GATES.md before its RTL.

Design fixed here (docs/NET2_GATES.md section 2): one gather unit, one ACT word (8 channels) per cycle, two X banks used alternately;
passes in order; a pass may start when its vector is written (2 cycles after the last address), the engine's interval of 9 cycles has
passed; the gather may begin rewriting a bank as soon as the pass that used it has started (the engine reads row group g at edge t0 + 2 + g and the gather writes
groups in ascending order, one word per cycle, so group g is written at or after its read). Row passes of a column
tile follow each other (engine accumulators); only the last one's 16 results are requantised, 2 per cycle, 6 stages, and written
as 8-channel words. Layers are serialised by a barrier. ADD: 2 elements per cycle. Pool: one word (8 channels) per cycle.
  python resnet_sched.py            prediction table and model/resnet_prediction.json
"""
import json
import os

import numpy as np

from resnet_int import PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
D, GAT, REL, DONE, RQ_PER, RQ_LAT, WB, BAR, GLAT = 9, 2, 1, 14, 2, 6, 2, 4, 2
ADD_LANES, ADD_LAT, POOL_LAT = 2, 12, 10
# name: (input H = W, stride, Cin, K, kind, map of the gather)
CONV = {"c0": (32, 1, 3, 3), "c1": (32, 1, 16, 3), "c2": (32, 1, 16, 3), "c4": (32, 2, 16, 3), "c5": (16, 1, 32, 3), "c6": (32, 2, 16, 1),
        "c8": (16, 2, 32, 3), "c9": (8, 1, 64, 3), "c10": (16, 2, 32, 1)}
ORDER = ["c0", "c1", "c2", "a3", "c4", "c5", "c6", "a7", "c8", "c9", "c10", "a11", "p12", "fc"]
ADD = {"a3": 32 * 32 * 16, "a7": 16 * 16 * 32, "a11": 8 * 8 * 64}
POOL_WORDS = 64 * 8                                        # 64 channels x 64 positions, 8 channels per word


def conv_cycles(n, P):
    H, s, Cin, K = CONV[n]
    W = P[f"{n}.w"]
    Cout = W.shape[0]
    pos = (H // s) ** 2
    rows = K * K * Cin
    R, C = -(-rows // 64), -(-Cout // 16)
    nread = []
    for r in range(R):
        lo, hi = 64 * r, min(64 * (r + 1), rows)
        taps = set(x // Cin for x in range(lo, hi))
        nread.append(len(taps) * (1 if Cin == 3 else max(1, Cin // 8)))
    # last-row-tile corner for Cin = 16: rows of one tap (16) per row tile; words written per tap: 2
    t_gather_end = -10                                     # address cycle of the gather unit's last word
    p_hist = []                                            # start cycles of passes
    res_free = 0
    end = 0
    for k in range(pos * R * C):
        r = k % R
        chain_last = r == R - 1
        start_addr = max(t_gather_end + 1, (p_hist[-2] + REL) if len(p_hist) >= 2 else 0)
        t_gather_end = start_addr + nread[r] - 1
        ready = t_gather_end + GAT + 1
        p = max(ready, (p_hist[-1] + D) if p_hist else 0)
        p_hist.append(p)
        if chain_last:
            rq = max(p + DONE, res_free)
            res_free = rq + (16 + RQ_PER - 1) // RQ_PER
            end = max(end, res_free + RQ_LAT + WB)
    return end + BAR, pos * R * C, nread


def fc_cycles():
    return 1 + 8 + GAT + D + DONE + 5 + RQ_LAT + WB + BAR


def predict():
    P = np.load(PARAMS)
    out, total = [], 0
    for n in ORDER:
        if n in CONV:
            c, npass, nread = conv_cycles(n, P)
            row = dict(op=n, kind="conv", cycles=c, passes=npass, reads=sum(nread) * npass // max(1, len(nread)))
        elif n in ADD:
            c = ADD[n] // ADD_LANES + ADD_LAT + BAR
            row = dict(op=n, kind="add", cycles=c, elements=ADD[n])
        elif n == "p12":
            c = POOL_WORDS + POOL_LAT + BAR
            row = dict(op=n, kind="pool", cycles=c, words=POOL_WORDS)
        else:
            c = fc_cycles()
            row = dict(op=n, kind="fc", cycles=c, passes=1)
        total += c
        out.append(row)
    return out, total


if __name__ == "__main__":
    rows, total = predict()
    for r in rows:
        print(f"{r['op']:4s} {r['kind']:5s} cycles {r['cycles']:7d}" + "".join(f"  {k} {v}" for k, v in r.items() if k not in ("op", "kind", "cycles")))
    eng = sum(r["passes"] for r in rows if "passes" in r) * D
    print(f"total {total} cycles; engine bound {eng}; {total / 15.36e6 * 1e3:.2f} ms at 15.36 MHz")
    json.dump(dict(ops=rows, total=total, engine_cycles=eng, design=dict(D=D, GAT=GAT, REL=REL, DONE=DONE, RQ_PER=RQ_PER, RQ_LAT=RQ_LAT, WB=WB, BAR=BAR,
                   ADD_LANES=ADD_LANES)), open(os.path.join(HERE, "resnet_prediction.json"), "w"), indent=1)
