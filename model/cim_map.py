#!/usr/bin/env python3
"""R4b mapping model: the integer front-end (mfcc_int.py) computed only through CIM passes on
eight 64 x 16 int8 tiles plus the epilogue operations, exactly as firmware and the hop
sequencer do it. self_test() checks it bit-exactly against mfcc_int on real clips.

Tile slots (CIM tile index):
  0 F1     FFT step 1: rows n1 0..15; cols 0..8 = re k1 0..8, cols 9..15 = im k1 1..7
  1 F3     FFT step 3: rows [Br n2 0..31 | Bi n2 0..31]; cols 0..7 = re k2 0..7, 8..15 = im k2 0..7
  2 MEL00  mel bins 0..63,   filters 0..15
  3 MEL01  mel bins 0..63,   filters 16..31
  4 MEL11  mel bins 64..127, filters 16..31
  5 MEL12  mel bins 64..127, filters 32..39 (cols 0..7)
  6 D1     DCT rows n 0..39; cols 0..9 = hi k 0..9, cols 10..15 = lo k 0..5
  7 D2     DCT rows n 0..39; cols 0..3 = lo k 6..9
"""
import numpy as np

import mfcc_int as mi

T_F1, T_F3, T_MEL00, T_MEL01, T_MEL11, T_MEL12, T_D1, T_D2 = range(8)
B_F1, B_F3, B_MEL, B_DCT = 17, 17, 22, 14              # input widths (worst-case bounds)


def build_tiles():
    t = np.zeros((8, 64, 16), np.int64)
    re, im = mi.M1q[:, :16], mi.M1q[:, 16:]
    t[T_F1, :16, 0:9] = re[:, 0:9]
    t[T_F1, :16, 9:16] = im[:, 1:8]
    t[T_F3, :, 0:8] = mi.M3q[:, 0:8]
    t[T_F3, :, 8:16] = mi.M3q[:, 17:25]
    M = mi.MELq
    t[T_MEL00] = M[0:64, 0:16]
    t[T_MEL01] = M[0:64, 16:32]
    t[T_MEL11] = M[64:128, 16:32]
    t[T_MEL12, :, 0:8] = M[64:128, 32:40]
    assert not M[128:].any() and not M[:, :].any(1)[0]           # only bins 1..127 are used
    lo = ((mi.DCTq + 128) & 255) - 128
    hi = (mi.DCTq - lo) >> 8
    t[T_D1, :40, 0:10] = hi
    t[T_D1, :40, 10:16] = lo[:, 0:6]
    t[T_D2, :40, 0:4] = lo[:, 6:10]
    assert t.min() >= -128 and t.max() <= 127
    return t.astype(np.int8)


TILES = build_tiles()


def sext(v, b):
    v = np.asarray(v, np.int64) & ((1 << b) - 1)
    return np.where(v >= (1 << (b - 1)), v - (1 << b), v)


def cim_pass(tile, x, b, acc=None):
    """Engine semantics (ENGINE_SPEC.md): res[n] = (acc or 0) + sum_r W[tile][r][n] * sext_b(x[r])."""
    xx = np.zeros(64, np.int64)
    xx[:len(x)] = x
    r = sext(xx, b) @ TILES[tile].astype(np.int64)
    return r if acc is None else acc + r


def frame_lraw_mapped(fr16):
    buf = np.zeros(mi.NFFT, np.int64)
    buf[:mi.WIN] = fr16.astype(np.int64) * mi.HANN_Q15
    peak = int(np.abs(buf).max())
    if peak == 0:
        return np.full(mi.NMEL, mi.ZERO, np.int64)
    sh = peak.bit_length() - 1 - 14
    xb = mi.rshift(buf, sh) if sh > 0 else np.left_shift(buf, -sh)
    # step 1: one pass per n2; the mirrored outputs are negated before rounding
    Ar, Ai = np.zeros((16, 32), np.int64), np.zeros((16, 32), np.int64)
    for n2 in range(32):
        s = cim_pass(T_F1, xb[n2::32], B_F1)                     # x[n1] = xb[32 n1 + n2]
        sre, sim = s[0:9], np.concatenate([[0], s[9:16], [0]])    # k1 0..8
        Ar[0:9, n2], Ai[0:9, n2] = mi.rshift(sre, 11), mi.rshift(sim, 11)
        Ar[9:16, n2] = mi.rshift(sre[7:0:-1], 11)                 # k1 = 16 - j, j = 7..1
        Ai[9:16, n2] = mi.rshift(-sim[7:0:-1], 11)
    Br = mi.rshift(Ar * mi.TWr - Ai * mi.TWi, 14)
    Bi = mi.rshift(Ar * mi.TWi + Ai * mi.TWr, 14)
    # step 3: one pass per k1, bins k1 + 16 k2 for k2 0..7
    re, im = np.zeros(128, np.int64), np.zeros(128, np.int64)
    for k1 in range(16):
        s = mi.rshift(cim_pass(T_F3, np.concatenate([Br[k1], Bi[k1]]), B_F3), 9)
        re[k1::16], im[k1::16] = s[0:8], s[8:16]
    mag = mi.isqrt(re * re + im * im)
    mel = np.zeros(mi.NMEL, np.int64)
    mel[0:16] = cim_pass(T_MEL00, mag[0:64], B_MEL)
    a = cim_pass(T_MEL01, mag[0:64], B_MEL)
    mel[16:32] = cim_pass(T_MEL11, mag[64:128], B_MEL, acc=a)    # accumulate pass
    mel[32:40] = cim_pass(T_MEL12, mag[64:128], B_MEL)[0:8]
    lraw = mi.log2_q8(np.maximum(mel, 1)) * 256 + 65536 * (sh + 5) - mi.K16
    return np.where(mel > 0, lraw, mi.ZERO)


def finish_mapped(lraw, lmx16):
    lmq = mi.rshift((lraw - lmx16) * mi.LN2_Q16, 24)
    lmq = np.where(lraw == mi.ZERO, mi.FLOOR_Q8, np.maximum(lmq, mi.FLOOR_Q8))
    y1 = cim_pass(T_D1, lmq, B_DCT)
    y2 = cim_pass(T_D2, lmq, B_DCT)
    acc = 256 * y1[0:10] + np.concatenate([y1[10:16], y2[0:4]])
    v = acc * mi.RQ_M + (mi.IN_ZP << mi.RQ_F)
    q = np.where(v >= 0, v >> mi.RQ_F, -((-v) >> mi.RQ_F))
    return np.clip(q, -128, 127).astype(np.int8)


def self_test(n_clips=60):
    import glob
    import os
    import wave
    test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kws_gate", "data", "test")
    files = sorted(glob.glob(os.path.join(test, "*", "*.wav")))[::max(1, 4890 // n_clips)][:n_clips]
    frames = bad_l = bad_q = 0
    for p in files:
        with wave.open(p, "rb") as w:
            x16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)
        x = np.zeros(mi.NS, np.int64)
        x[:len(x16)] = x16
        lmx = mi.log2fix16(int(x16.max()))
        for f in range(mi.NFR):
            fr = x[f * mi.HOP: f * mi.HOP + mi.WIN]
            la, lb = mi.frame_lraw(fr), frame_lraw_mapped(fr)
            bad_l += int(not np.array_equal(la, lb))
            bad_q += int(not np.array_equal(mi.finish(la, lmx), finish_mapped(lb, lmx)))
            frames += 1
    print(f"mapping self-test: {len(files)} clips, {frames} frames, lraw mismatches {bad_l}, feature mismatches {bad_q}")
    return bad_l == 0 and bad_q == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
