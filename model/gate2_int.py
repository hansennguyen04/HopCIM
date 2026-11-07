#!/usr/bin/env python3
"""Accuracy gate for an integer front-end: the unchanged MLPerf Tiny DS-CNN fed by the front-end's
int8 features versus the float reference MFCC with the reference int8 input quantisation. Same test
set and criterion as gate 2 (kws_gate/GATE2_MANIFEST.md): accuracy drop <= 0.3 points.
  python gate2_int.py [int|ded]   int: CIM-MFCC (mfcc_int.py, R4a); ded: B3 dedicated (mfcc_ded.py)
Run in ~/venvs/kwsgate. Results: gate2_<frontend>_results.json.
"""
import glob
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kws_gate"))
import gate2_eval as g  # noqa: E402
import mfcc_int as mi  # noqa: E402
import mfcc_ded as md  # noqa: E402

FE = sys.argv[1] if len(sys.argv) > 1 else "int"
FRONTEND = {"int": mi.cim_mfcc_int, "ded": md.mfcc}[FE]


def predict_q(model, q):
    it, inp, out = model
    it.set_tensor(inp["index"], q[None])
    it.invoke()
    return int(np.argmax(it.get_tensor(out["index"])))


def main():
    files = []
    for li, lab in enumerate(g.LABELS):
        files += [(p, li) for p in sorted(glob.glob(os.path.join(g.TEST, lab, "*.wav")))]
    models = {k: g.load_interp(v) for k, v in g.MODELS.items()}
    s, zp = models["int8"][1]["quantization"]
    assert abs(s - mi.IN_SCALE) < 1e-12 and zp == mi.IN_ZP
    keys = ("int8|ref", "int8|int", "float32|ref", "float32|int")
    pred, labels, diffs = {k: [] for k in keys}, [], []
    for i, (p, li) in enumerate(files):
        x = g.read_wav(p)
        fr = g.ref_mfcc(x)
        qi = FRONTEND(x)
        qr = np.array(fr / s + zp, dtype=np.int8)                      # reference input quantisation
        diffs.append(qi.astype(np.int64).ravel() - qr.astype(np.int64).ravel())
        labels.append(li)
        pred["int8|ref"].append(g.predict(models["int8"], fr, "reference"))
        pred["int8|int"].append(predict_q(models["int8"], qi))
        pred["float32|ref"].append(g.predict(models["float32"], fr))
        pred["float32|int"].append(g.predict(models["float32"], (qi.astype(np.float64) - zp) * s))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    labels = np.array(labels)
    acc = {k: float(np.mean(np.array(v) == labels) * 100) for k, v in pred.items()}
    d = np.abs(np.concatenate(diffs))

    def mcnemar(a, b):
        a, b = np.array(a) == labels, np.array(b) == labels
        n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
        n = n01 + n10
        return n01, n10, (min(1.0, 2 * sum(comb(n, j) for j in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0)

    res = dict(frontend=FE, n=len(labels), accuracy=acc,
               delta_int8=acc["int8|int"] - acc["int8|ref"],
               delta_float32=acc["float32|int"] - acc["float32|ref"],
               mcnemar_int8=mcnemar(pred["int8|ref"], pred["int8|int"]),
               int8_feature_diff_vs_reference=dict(equal=float(np.mean(d == 0)), within_1=float(np.mean(d <= 1)),
                                                    mean=float(d.mean()), max=int(d.max())),
               constants=dict(K16=mi.K16, LN2_Q16=mi.LN2_Q16, FLOOR_Q8=mi.FLOOR_Q8, RQ_M=mi.RQ_M, RQ_F=mi.RQ_F,
                              DCT_S=mi.DCT_S))
    res["gate_pass"] = res["delta_int8"] >= -0.30
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(HERE, f"gate2_{FE}_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
