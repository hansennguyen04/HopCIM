#!/usr/bin/env python3
"""ResNet8 on the CIM: tiles, exact pass-level execution, and a cycle model. python resnet_map.py [exact n | tiles | cycles]

Mapping: a convolution K x K x Cin -> Cout is a matrix of (K K Cin) rows x Cout columns, row = tap * Cin + c with tap = fy K + fx,
cut into R = ceil(rows / 64) row tiles and Cc = ceil(Cout / 16) column tiles. One output position takes R Cc passes: for each
column tile the R row passes add up in the engine's accumulators (accum on all but the first). Input rows hold x - zp_in
(9 bits), SAME-padding rows hold 0, so the result is sum W (x - zp_in) exactly; the requantisation adds the bias.
"""
import sys

import numpy as np

from resnet_int import PARAMS, TOPO, WIRE, add, cifar, conv, fc, pool, same_pad
from dscnn_int import mbqm

TR, TC = 64, 16                                    # tile: rows (inputs) x columns (outputs)
CONVS = ["c0", "c1", "c2", "c4", "c5", "c6", "c8", "c9", "c10"]
MAPS = {"c0": (32, 3), "c1": (32, 16), "c2": (32, 16), "c4": (32, 16), "c5": (16, 32), "c6": (32, 16), "c8": (16, 32), "c9": (8, 64), "c10": (16, 32)}


class Net:
    def __init__(self):
        self.P = np.load(PARAMS)
        self.tiles, self.base = [], {}
        for n in CONVS + ["fc"]:
            w = self.P[f"{n}.w"].astype(np.int64)
            K = w.shape[0]
            m = w.reshape(K, -1).T if n != "fc" else w.T                  # [rows, outputs]
            rows, outs = m.shape
            R, C = -(-rows // TR), -(-outs // TC)
            self.base[n] = (len(self.tiles), R, C, rows, outs)
            for r in range(R):
                for c in range(C):
                    t = np.zeros((TR, TC), np.int64)
                    blk = m[r * TR:(r + 1) * TR, c * TC:(c + 1) * TC]
                    t[:blk.shape[0], :blk.shape[1]] = blk
                    self.tiles.append(t)
        self.tiles = np.array(self.tiles)

    def passes(self, n, pos):
        b, R, C, rows, outs = self.base[n]
        return R * C * pos

    # one conv layer, executed pass by pass exactly as the CIM would (all positions)
    def layer(self, x, n):
        P = self.P
        b, R, C, rows, outs = self.base[n]
        zi, zo, s = int(P[f"{n}.zi"]), int(P[f"{n}.zo"]), int(P[f"{n}.stride"])
        kh, kw, Cin = P[f"{n}.w"].shape[1], P[f"{n}.w"].shape[2], P[f"{n}.w"].shape[3]
        Nn, H, W, _ = x.shape
        oh, ph = same_pad(H, kh, s)
        ow, pw = same_pad(W, kw, s)
        xz = np.zeros((Nn, H + kh + 1, W + kw + 1, Cin), np.int64)
        xz[:, ph:ph + H, pw:pw + W, :] = x - zi
        y = np.zeros((Nn, oh, ow, outs), np.int64)
        npass = 0
        for i in range(oh):
            for j in range(ow):
                v = xz[:, i * s:i * s + kh, j * s:j * s + kw, :].reshape(Nn, -1)          # [N, rows]
                vp = np.zeros((Nn, R * TR), np.int64)
                vp[:, :rows] = v
                for c in range(C):
                    acc = np.zeros((Nn, TC), np.int64)
                    for r in range(R):
                        assert vp[:, r * TR:(r + 1) * TR].min() >= -256 and vp.max() <= 255
                        acc = acc + vp[:, r * TR:(r + 1) * TR] @ self.tiles[b + r * C + c]
                        npass += 1
                    y[:, i, j, c * TC:(c + 1) * TC] = acc[:, :min(TC, outs - c * TC)]
        r = mbqm(y + P[f"{n}.b"].astype(np.int64), P[f"{n}.q"], P[f"{n}.sh"]) + zo
        return np.clip(r, zo if int(P[f"{n}.relu"]) else -128, 127), npass


def exact(n):
    net = Net()
    x, lab = cifar(n)
    bad, total = 0, 0
    v = {"in": x.astype(np.int64)}
    for kind, nm, _ in TOPO:
        if kind == "conv":
            got, np_ = net.layer(v[WIRE[nm][0]], nm)
            ref = conv(v[WIRE[nm][0]], net.P, nm)
            total += np_
            mism = int((got != ref).sum())
        elif kind == "add":
            src = [str(s) for s in net.P[f"{nm}.src"]]
            got = ref = add(v[src[0]], v[src[1]], net.P, nm)
            mism = 0
        elif kind == "pool":
            got = ref = pool(v[WIRE[nm][0]]).reshape(len(x), -1)
            mism = 0
        else:
            b, R, C, rows, outs = net.base["fc"]
            xv = (v[WIRE[nm][0]] - int(net.P["fc.zi"]))
            acc = np.zeros((len(x), TC), np.int64)
            for r in range(R):
                vp = np.zeros((len(x), TR), np.int64); vp[:, :rows] = xv[:, :rows]
                acc += vp @ net.tiles[b + r]
            got = np.clip(mbqm(acc[:, :outs] + net.P["fc.b"], net.P["fc.q"], net.P["fc.sh"]) + int(net.P["fc.zo"]), -128, 127)
            ref = fc(v[WIRE[nm][0]], net.P, nm)
            mism = int((got != ref).sum())
            total += R * C
        v[nm] = got
        bad += mism
        print(f"{nm:4s} {kind:4s} mismatches {mism}")
    print(f"{n} images: {bad} mismatches, {total} passes per inference, {len(net.tiles)} tiles")
    return bad == 0


def tiles():
    net = Net()
    for n in net.base:
        b, R, C, rows, outs = net.base[n]
        print(f"{n:4s} {rows:4d} x {outs:3d} -> R {R} x C {C} = {R * C} tiles (base {b})")
    print("total", len(net.tiles), "of 128;  network weights", sum(int(np.prod(net.P[f'{n}.w'].shape)) for n in net.base))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tiles"
    if cmd == "exact":
        sys.exit(0 if exact(int(sys.argv[2]) if len(sys.argv) > 2 else 4) else 1)
    tiles()
