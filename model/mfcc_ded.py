#!/usr/bin/env python3
"""B3 baseline: a dedicated MFCC front-end without the CIM; golden model of rtl/mfcc_ded.sv.

Same output contract as the CIM front-end (mfcc_int.frame_lraw): lraw[40] per frame in Q16 log2,
so the window path (mfcc_int.finish) applies unchanged. The FFT is a radix-2 FFT, not CIM passes:
  Hann (Q15) -> block exponent sh (as mfcc_int)
  -> 512-point radix-2 DIT FFT: 18-bit data, Q16 twiddles; every stage rounds the twiddle product
     (>> 16) and halves the butterfly outputs, so the output is DFT/512 and cannot overflow
  -> |X| of bins 1..127 (exact isqrt) -> mel (the same int8 weights: 247 non-zeros, <= 2 per bin)
  -> log2 (the same leading one + 64-entry table) -> lraw = log2_q8 * 256 + 65536 (sh - 6) - K16D
Scale: |X_ref| = |X| * 2^(sh - 6) / max(x), so K16D = 65536 log2(mel scale).
"""
import numpy as np

import mfcc_int as mi

NFFT, NST = 512, 9
DW = 18                                             # data width (signed), real and imaginary parts
K16D = int(round(65536 * np.log2(mi.MEL_S)))
_i = np.arange(NFFT // 2)
TWD_R = np.round(65536 * np.cos(2 * np.pi * _i / NFFT)).astype(np.int64)   # Q16, W^i = exp(-2 pi i / 512)
TWD_I = np.round(-65536 * np.sin(2 * np.pi * _i / NFFT)).astype(np.int64)
BITREV = np.array([int(f"{i:09b}"[::-1], 2) for i in range(NFFT)])


def fft_ded(xb):
    """Radix-2 DIT FFT of the real 512-sample frame xb (int64), as the RTL computes it."""
    re = np.zeros(NFFT, np.int64)
    im = np.zeros(NFFT, np.int64)
    re[BITREV] = xb                                 # bit-reversed load: x[i] goes to address bitrev(i)
    lim = 1 << (DW - 1)
    for s in range(NST):
        h = 1 << s
        k = np.arange(NFFT // 2)
        j, m = k >> s, k & (h - 1)
        i0 = j * 2 * h + m
        i1 = i0 + h
        w = m << (NST - 1 - s)
        br, bi, wr, wi = re[i1], im[i1], TWD_R[w], TWD_I[w]
        tr = mi.rshift(br * wr - bi * wi, 16)
        ti = mi.rshift(br * wi + bi * wr, 16)
        ar, ai = re[i0], im[i0]
        re[i0], im[i0] = mi.rshift(ar + tr, 1), mi.rshift(ai + ti, 1)
        re[i1], im[i1] = mi.rshift(ar - tr, 1), mi.rshift(ai - ti, 1)
        assert np.abs(re).max() < lim and np.abs(im).max() < lim, "FFT data overflow"
    return re, im


def frame_lraw(fr16):
    """One 480-sample int16 frame -> lraw[40] (int64, Q16 log2; mi.ZERO for an exact zero)."""
    buf = np.zeros(NFFT, np.int64)
    buf[:mi.WIN] = fr16.astype(np.int64) * mi.HANN_Q15
    peak = int(np.abs(buf).max())
    if peak == 0:
        return np.full(mi.NMEL, mi.ZERO, np.int64)
    sh = peak.bit_length() - 1 - 14
    xb = mi.rshift(buf, sh) if sh > 0 else np.left_shift(buf, -sh)
    re, im = fft_ded(xb)
    mag = np.zeros(128, np.int64)
    mag[1:128] = mi.isqrt(re[1:128] ** 2 + im[1:128] ** 2)
    mel = mag @ mi.MELq[:128]
    lraw = mi.log2_q8(np.maximum(mel, 1)) * 256 + 65536 * (sh - 6) - K16D
    return np.where(mel > 0, lraw, mi.ZERO)


def mfcc(x16):
    """int16 clip -> int8 features [49, 10, 1] through the dedicated front-end."""
    x = np.zeros(mi.NS, np.int64)
    x[:len(x16)] = x16
    mx = int(np.max(x16))
    if mx <= 0:
        raise ValueError("clip max <= 0: reference normalisation undefined")
    lraw = np.stack([frame_lraw(x[f * mi.HOP: f * mi.HOP + mi.WIN]) for f in range(mi.NFR)])
    return mi.finish(lraw, mi.log2fix16(mx)).reshape(mi.NFR, mi.NMFCC, 1)


if __name__ == "__main__":
    # agreement with an ideal float FFT on random full-scale frames
    rng = np.random.default_rng(1)
    err = []
    for _ in range(200):
        fr = rng.integers(-32768, 32768, mi.WIN)
        buf = np.zeros(NFFT, np.int64)
        buf[:mi.WIN] = fr * mi.HANN_Q15
        sh = int(np.abs(buf).max()).bit_length() - 15
        xb = mi.rshift(buf, sh) if sh > 0 else np.left_shift(buf, -sh)
        re, im = fft_ded(xb)
        ref = np.fft.fft(xb.astype(np.float64)) / NFFT
        err.append(np.abs((re + 1j * im) - ref)[1:128].max())
    print(f"K16D {K16D}; max |FFT error| vs float over 200 random frames: {max(err):.2f} LSB")
