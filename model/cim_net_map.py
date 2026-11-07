#!/usr/bin/env python3
"""Mapping model: the MLPerf Tiny int8 DS-CNN (dscnn_int.py) computed only through CIM passes on
64 x 16 int8 tiles plus the CPU/epilogue operations, exactly as firmware and the layer sequencer
do it. self_test() checks it bit-exactly against dscnn_int.forward on real clips.

Tiles (CIM indices; 0..7 hold the front-end, see cim_map.py):
  8..11    conv1   rows r = 4 fy + fx (40), column n = output channel 16 j + n   (tile 8 + j)
  12 + 14 l .. 21 + 14 l   depthwise l: 10 tiles of 7 channels, tap-major rows 7 (3 fy + fx) + (c - 7 g),
                            column c - 7 g (block diagonal); a tap's 7 channels are 7 consecutive rows
  22 + 14 l .. 25 + 14 l   pointwise l: rows = input channel, column n = output channel 16 j + n
  68       fully connected: rows = input, columns 0..11
Inputs carry their zero point: x[r] = act - zp_in (9 bits signed), and padding taps are 0, so a
pass gives sum W * (x - zp) exactly; bias, requantisation and the average pool are epilogue work.
"""
import numpy as np

import dscnn_int as d

T_CONV1 = 8
T_FC = 68
XB = 9                                              # input width of every network pass


def t_dw(l):
    return 12 + 14 * l


def t_pw(l):
    return 22 + 14 * l


def build_tiles():
    P = d.P
    t = np.zeros((128, 64, 16), np.int64)
    w = P["L0_w"]                                    # [64, 10, 4, 1]
    for j in range(4):
        for fy in range(10):
            for fx in range(4):
                t[T_CONV1 + j, 4 * fy + fx, :] = w[16 * j: 16 * j + 16, fy, fx, 0]
    for l in range(4):
        wd = P[f"L{1 + 2 * l}_w"]                    # [1, 3, 3, 64]
        for c in range(64):
            g, n = divmod(c, 7)
            for fy in range(3):
                for fx in range(3):
                    t[t_dw(l) + g, 7 * (3 * fy + fx) + n, n] = wd[0, fy, fx, c]
        wp = P[f"L{2 + 2 * l}_w"]                    # [64 out, 1, 1, 64 in]
        for j in range(4):
            t[t_pw(l) + j, :, :] = wp[16 * j: 16 * j + 16, 0, 0, :].T
    t[T_FC, :, 0:12] = P["L9_w"].T                   # [12, 64] -> rows = input
    assert t.min() >= -128 and t.max() <= 127
    return t.astype(np.int8)


TILES = build_tiles()


def cim_pass(tile, x):
    """Engine semantics: res[n] = sum_r W[tile][r][n] * sext_XB(x[r]) (x already in range)."""
    xx = np.zeros(64, np.int64)
    xx[:len(x)] = x
    assert xx.min() >= -(1 << (XB - 1)) and xx.max() < (1 << (XB - 1))
    return xx @ TILES[tile].astype(np.int64)


def requant(acc, L):
    P = d.P
    y = d.mbqm(acc + P[f"L{L}_b"], P[f"L{L}_q"], P[f"L{L}_sh"]) + int(P[f"L{L}_zout"])
    return np.clip(y, -128, 127)


def conv1_mapped(feat):
    zin = int(d.P["L0_zin"])
    x = np.asarray(feat, np.int64).reshape(49, 10) - zin
    out = np.zeros((25, 5, 64), np.int64)
    for oy in range(25):
        for ox in range(5):
            v = np.zeros(40, np.int64)               # padding taps stay 0
            for fy in range(10):
                for fx in range(4):
                    iy, ix = 2 * oy - 4 + fy, 2 * ox - 1 + fx   # SAME padding: top 4, left 1
                    if 0 <= iy < 49 and 0 <= ix < 10:
                        v[4 * fy + fx] = x[iy, ix]
            acc = np.concatenate([cim_pass(T_CONV1 + j, v) for j in range(4)])
            out[oy, ox] = requant(acc, 0)
    return out


def dw_mapped(a, l):
    L = 1 + 2 * l
    x = a - int(d.P[f"L{L}_zin"])
    out = np.zeros_like(a)
    for oy in range(25):
        for ox in range(5):
            acc = np.zeros(64, np.int64)
            for g in range(10):
                v = np.zeros(64, np.int64)
                for n in range(7):
                    c = 7 * g + n
                    if c >= 64:
                        break
                    for fy in range(3):
                        for fx in range(3):
                            iy, ix = oy - 1 + fy, ox - 1 + fx
                            if 0 <= iy < 25 and 0 <= ix < 5:
                                v[7 * (3 * fy + fx) + n] = x[iy, ix, c]
                r = cim_pass(t_dw(l) + g, v)
                nc = min(7, 64 - 7 * g)
                acc[7 * g: 7 * g + nc] = r[:nc]
            out[oy, ox] = requant(acc, L)
    return out


def pw_mapped(a, l):
    L = 2 + 2 * l
    x = a - int(d.P[f"L{L}_zin"])
    out = np.zeros_like(a)
    for oy in range(25):
        for ox in range(5):
            acc = np.concatenate([cim_pass(t_pw(l) + j, x[oy, ox]) for j in range(4)])
            out[oy, ox] = requant(acc, L)
    return out


def fc_mapped(p):
    P = d.P
    acc = cim_pass(T_FC, p - int(P["L9_zin"]))[:12]
    y = d.mbqm(acc + P["L9_b"], P["L9_q"][0], P["L9_sh"][0]) + int(P["L9_zout"])
    return np.clip(y, -128, 127)


def forward_mapped(feat):
    a = conv1_mapped(feat)
    for l in range(4):
        a = pw_mapped(dw_mapped(a, l), l)
    return fc_mapped(d.avgpool(a))


PASSES = 125 * 4 + 4 * (125 * 10 + 125 * 4) + 1      # 7,501 per inference


def self_test(n_clips=40):
    import glob
    import os
    import wave
    import mfcc_int as mi
    test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kws_gate", "data", "test")
    files = sorted(glob.glob(os.path.join(test, "*", "*.wav")))[::max(1, 4890 // n_clips)][:n_clips]
    bad = 0
    for p in files:
        with wave.open(p, "rb") as w:
            x16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)
        q = mi.cim_mfcc_int(x16)
        bad += int(not np.array_equal(forward_mapped(q), d.forward(q)))
    print(f"network mapping self-test: {len(files)} clips, {bad} logit mismatches; {PASSES} passes per inference, "
          f"tiles {T_CONV1}..{T_FC}")
    return bad == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test(int(sys.argv[1]) if len(sys.argv) > 1 else 40) else 1)
