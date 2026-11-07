#!/usr/bin/env python3
"""R4a golden model: the CIM-MFCC front-end as a pure integer pipeline ending in the int8 input
of the unchanged MLPerf Tiny DS-CNN. RTL and firmware must match it bit-exactly.

Frame path (once per 20 ms hop; the CIM plus epilogue units):
  Hann (Q15) -> block exponent sh -> four-step FFT 16 x 32 (int8 DFT tiles, Q14 twiddles)
  -> |X| (exact isqrt) -> mel (int8 tile) -> log2 (leading one + 64-entry LUT, Q8)
  -> lraw (Q16 log2 with the frame exponent and the constant scales folded in)
Window path (firmware; per clip, or per hop when streaming):
  lmq = max(round((lraw - log2fix16(max x)) * ln2), floor)   (Q8 natural log)
  -> DCT (int16 coefficients = two int8 CIM weights) -> int8 with the model's input quantisation.

Same arithmetic as the gate-2 datapath (kws_gate/gate2_eval.py: cim_mfcc) except that the
constant log terms, ln 2 and the final requantisation are fixed point, and the DCT scale is
127 * 256 / max|c| (see make_tables.py). Tables: mfcc_tables.npz.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS, WIN, HOP, NFFT, NB, NMEL, NMFCC, NFR = 16000, 480, 320, 512, 257, 40, 10, 49
N1, N2, K2 = 16, 32, 17
IN_SCALE, IN_ZP = 0.5847029089927673, 83          # kws_ref_model.tflite input quantisation

_T = np.load(os.path.join(HERE, "mfcc_tables.npz"))
M1q, TWr, TWi, M3q = _T["M1q"], _T["TWr"], _T["TWi"], _T["M3q"]
BIN, VALID, HANN_Q15, MELq, LOG_LUT, DCTq = _T["BIN"], _T["VALID"], _T["HANN_Q15"], _T["MELq"], _T["LOG_LUT"], _T["DCTq"]
MEL_S, DCT_S = float(_T["MEL_S"]), float(_T["DCT_S"])

# Q16 log2 constant: two int8 DFT scales (127 each) and the mel scale
K16 = int(round(65536 * (np.log2(127.0 * 127.0) + np.log2(MEL_S))))
LN2_Q16 = int(round(np.log(2.0) * 65536))          # 45426
FLOOR_Q8 = int(round(np.log(1e-6) * 256))          # -3537
ZERO = -(1 << 40)                                  # lraw of an exact zero (mel == 0 or silent frame)
# requantisation: q = trunc(acc / (256 * DCT_S * IN_SCALE) + IN_ZP), saturated to int8
RQ_F = 44
RQ_M = int(round(2.0 ** RQ_F / (256.0 * DCT_S * IN_SCALE)))


def rshift(x, s):
    """Arithmetic right shift with round-half-up (numpy int64 or Python int)."""
    return (x + (1 << (s - 1))) >> s if s > 0 else x


def isqrt(v):
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64)
    r -= (r * r > v)
    r += ((r + 1) * (r + 1) <= v)
    return r


def ilog2(v):
    """floor(log2 v) for int64 v >= 1 (a leading-one detector)."""
    e = np.floor(np.log2(v.astype(np.float64))).astype(np.int64)
    e -= (np.left_shift(np.int64(1), e) > v)
    e += (np.left_shift(np.int64(1), e + 1) <= v)
    return e


def log2_q8(v):
    """Q8 log2 of int64 v >= 1: leading one + 6 mantissa bits into a 64-entry log2(1 + f) table."""
    e = ilog2(v)
    frac = np.where(e >= 6, np.right_shift(v, np.maximum(e - 6, 0)), np.left_shift(v, np.maximum(6 - e, 0))) & 63
    return e * 256 + LOG_LUT[frac]


def log2fix16(v):
    """floor(log2(v) * 2^16) for an integer 1 <= v < 2^31, by repeated squaring (64-bit integer ops)."""
    v = int(v)
    e = v.bit_length() - 1
    m = v << (30 - e)                              # mantissa in [2^30, 2^31), Q30
    f = 0
    for _ in range(16):
        m = (m * m) >> 30                          # Q30 square, in [2^30, 2^32)
        f <<= 1
        if m >= (1 << 31):
            m >>= 1
            f |= 1
    return e * 65536 + f


def frame_lraw(fr16):
    """One 480-sample int16 frame -> lraw[40] (int64, Q16 log2; ZERO for an exact zero)."""
    buf = np.zeros(NFFT, np.int64)
    buf[:WIN] = fr16.astype(np.int64) * HANN_Q15
    peak = int(np.abs(buf).max())
    if peak == 0:
        return np.full(NMEL, ZERO, np.int64)
    sh = peak.bit_length() - 1 - 14                # block exponent: peak -> [2^14, 2^15)
    xb = rshift(buf, sh) if sh > 0 else np.left_shift(buf, -sh)
    X = xb.reshape(N1, N2)                         # x[N2*n1 + n2]
    A = rshift(X.T @ M1q, 11)                      # step 1 (CIM): [n2, 16 re | 16 im]
    Ar, Ai = A[:, :N1].T, A[:, N1:].T              # [k1, n2]
    Br = rshift(Ar * TWr - Ai * TWi, 14)           # twiddle unit
    Bi = rshift(Ar * TWi + Ai * TWr, 14)
    Xs = rshift(np.concatenate([Br, Bi], 1) @ M3q, 9)   # step 3 (CIM): [k1, 17 re | 17 im]
    re, im = np.zeros(NB, np.int64), np.zeros(NB, np.int64)
    re[BIN[VALID]], im[BIN[VALID]] = Xs[:, :K2][VALID], Xs[:, K2:][VALID]
    mag = isqrt(re * re + im * im)
    mel = mag @ MELq                               # mel (CIM)
    lraw = log2_q8(np.maximum(mel, 1)) * 256 + 65536 * (sh + 5) - K16
    return np.where(mel > 0, lraw, ZERO)


def finish(lraw, lmx16):
    """lraw[..., 40] and the Q16 log2 of the clip (window) max -> int8 features [..., 10]."""
    lmq = rshift((lraw - lmx16) * LN2_Q16, 24)
    lmq = np.where(lraw == ZERO, FLOOR_Q8, np.maximum(lmq, FLOOR_Q8))
    acc = lmq @ DCTq                               # DCT (CIM, two int8 weight halves)
    v = acc * RQ_M + (IN_ZP << RQ_F)
    q = np.where(v >= 0, v >> RQ_F, -((-v) >> RQ_F))   # truncation toward zero, as the reference
    return np.clip(q, -128, 127).astype(np.int8)


def cim_mfcc_int(x16):
    """int16 clip (<= 1 s) -> int8 features [49, 10, 1], the DS-CNN input."""
    x = np.zeros(NS, np.int64)
    x[:len(x16)] = x16
    mx = int(np.max(x16))
    if mx <= 0:
        raise ValueError("clip max <= 0: reference normalisation undefined")
    lraw = np.stack([frame_lraw(x[f * HOP: f * HOP + WIN]) for f in range(NFR)])
    return finish(lraw, log2fix16(mx)).reshape(NFR, NMFCC, 1)


if __name__ == "__main__":
    print(f"K16 {K16}  LN2_Q16 {LN2_Q16}  FLOOR_Q8 {FLOOR_Q8}  RQ_M {RQ_M} (F {RQ_F})  DCT_S {DCT_S:.3f}")
    for v in (1, 2, 3, 1000, 32767):
        print(f"log2fix16({v}) = {log2fix16(v)}  exact {np.log2(v) * 65536:.2f}")
