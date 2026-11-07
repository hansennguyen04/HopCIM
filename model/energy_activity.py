#!/usr/bin/env python3
"""Bit activity of HopCIM's CIM passes on real audio, for the energy estimate (energy_est.py).

The published digital CIM model (ZigZag-IMC, Houshmand et al.) scales its multiplier and adder-tree
energy by input toggle rate x weight density, and validates at 0.5 x 0.5 = 0.25. This script measures
that product for every pass the integer golden model runs on GSC test clips:

  alpha = sum_n sum_r density[r, n] * toggles[r] / (64 * C * b)

density[r, n]: ones per weight bit of row r, column n; toggles[r]: bit changes of input row r over its
b bit planes, starting from 0; C: columns the tile uses; rows the tile leaves empty count as zero, so a
sparse tile (the depthwise block-diagonal one) gets a small alpha by itself.

Usage: energy_activity.py [clips, default 40]    writes energy_activity.json
"""
import glob
import json
import os
import sys
import wave

import numpy as np

import cim_map as fe
import cim_net_map as net
import mfcc_int as mi

HERE = os.path.dirname(os.path.abspath(__file__))


def tile_stats(tiles):
    """For each tile: density[r, n] (ones per weight bit) and the columns it uses."""
    u = tiles.astype(np.uint8)
    dens = np.unpackbits(u[..., None], axis=-1).sum(-1) / 8.0
    cols = (tiles != 0).any(axis=1)
    return dens, cols


class Meter:
    def __init__(self, tiles, group):
        self.dens, self.cols = tile_stats(tiles)
        self.group = group
        self.n, self.a, self.t, self.d = {}, {}, {}, {}

    def add(self, tile, x, b):
        xx = np.zeros(64, np.int64)
        xx[:len(x)] = x
        v = xx & ((1 << b) - 1)
        tog = v ^ ((v << 1) & ((1 << b) - 1))              # bit i against bit i-1; bit -1 is 0
        tg = np.array([bin(int(t)).count("1") for t in tog], np.float64)
        c = self.cols[tile]
        C = int(c.sum())
        g = self.group(tile)
        if C == 0:
            return
        a = float((self.dens[tile][:, c].sum(axis=1) * tg).sum()) / (64 * C * b)
        self.n[g] = self.n.get(g, 0) + 1
        self.a[g] = self.a.get(g, 0.0) + a
        self.t[g] = self.t.get(g, 0.0) + float(tg.sum()) / (64 * b)
        self.d[g] = self.d.get(g, 0.0) + float(self.dens[tile][:, c].mean())

    def result(self):
        return {g: {"passes": self.n[g], "alpha": self.a[g] / self.n[g],
                    "toggle_rate": self.t[g] / self.n[g], "weight_density": self.d[g] / self.n[g]}
                for g in sorted(self.n)}


def net_group(t):
    if t < net.t_dw(0):
        return "conv1"
    if t == net.T_FC:
        return "fc"
    return "dw" if (t - net.t_dw(0)) % 14 < 10 else "pw"


FE_NAMES = {fe.T_F1: "fft1", fe.T_F3: "fft3", fe.T_MEL00: "mel", fe.T_MEL01: "mel", fe.T_MEL11: "mel",
            fe.T_MEL12: "mel", fe.T_D1: "dct", fe.T_D2: "dct"}


def main(n_clips=40):
    test = os.path.join(HERE, "..", "..", "kws_gate", "data", "test")
    files = sorted(glob.glob(os.path.join(test, "*", "*.wav")))[::max(1, 4890 // n_clips)][:n_clips]
    mn = Meter(net.TILES, net_group)
    mf = Meter(fe.TILES, lambda t: FE_NAMES[t])
    p0, p1 = net.cim_pass, fe.cim_pass

    def net_pass(tile, x):
        mn.add(tile, x, net.XB)
        return p0(tile, x)

    def fe_pass(tile, x, b, acc=None):
        mf.add(tile, x, b)
        return p1(tile, x, b, acc)

    net.cim_pass, fe.cim_pass = net_pass, fe_pass
    bad = 0
    for p in files:
        with wave.open(p, "rb") as w:
            x16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)
        x = np.zeros(mi.NS, np.int64)
        x[:len(x16)] = x16
        q = mi.cim_mfcc_int(x16)
        bad += int(not np.array_equal(net.forward_mapped(q), net.d.forward(q)))
        lmx = mi.log2fix16(int(x16.max()))
        for f in range(mi.NFR):
            fe.finish_mapped(fe.frame_lraw_mapped(x[f * mi.HOP: f * mi.HOP + mi.WIN]), lmx)
    net.cim_pass, fe.cim_pass = p0, p1
    res = {"clips": len(files), "logit_mismatches": bad, "groups": {**mn.result(), **mf.result()},
           "columns": {"conv1": 16, "dw": 7, "pw": 16, "fc": 12, "fft1": 16, "fft3": 16,
                       "mel": [16, 16, 16, 8], "dct": [16, 4]}}
    with open(os.path.join(HERE, "energy_activity.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(files)} clips, {bad} logit mismatches")
    for g, v in res["groups"].items():
        print(f"  {g:6s} passes {v['passes']:8d}  alpha {v['alpha']:.4f}  toggle rate {v['toggle_rate']:.3f}  weight density {v['weight_density']:.3f}")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 40) else 1)
