#!/usr/bin/env python3
"""R7 golden model: the hardware PDM decimator of the live audio path (Nexys A7 on-board ADMP421).

System clock 15.36 MHz -> PDM clock 1.024 MHz (/15) -> 16 kHz exactly (/64):
  x = 2 b - 1 (one bit per PDM clock)
  CIC, 5 integrators at 1.024 MHz and 5 combs at 16 kHz, 32-bit wrap-around arithmetic, gain 64^5 = 2^30
  y1 = (c + 2^14) >> 15                                   (|y1| <= 2^15)
  FIR, NTAP symmetric taps at 16 kHz, Q14: y = sat16((sum h[k] y1[n-k] + 2^13) >> 14)
  DC blocker (first-order high-pass, pole 1 - 2^-8, -3 dB at about 10 Hz), accumulator in Q8:
      acc += ((y - y_prev) << 8) - (acc >> 8);  out = sat16((acc + 2^7) >> 8)
  (the microphone's PDM has a DC offset; the model's per-window normalisation divides by the
  window's maximum sample, so a DC offset would shift every feature)
The FIR flattens the CIC droop over the mel band of the model (20 Hz .. 4 kHz, MLPerf Tiny
get_dataset.py) and rolls off above it. The CIC alone rejects everything that aliases into
0..4 kHz (12..20 kHz at the PDM rate) by >= 52 dB. RTL (rtl/pdm_dec.sv) must match decimate()
bit-exactly.

Test stimuli: pdm_from_audio() turns 16 kHz int16 audio into a PDM bit stream (linear
interpolation x64, second-order sigma-delta modulator, written in C for speed).
"""
import ctypes
import os
import subprocess
import tempfile

import numpy as np

FS_OUT, R, N_CIC = 16000, 64, 5
FS_PDM = FS_OUT * R                                 # 1.024 MHz
NTAP, H_FRAC = 15, 14
MASK32 = (1 << 32) - 1


def cic_response(f):
    """|H| of the 5th-order CIC (R = 64) at frequency f (Hz), normalised to 1 at DC."""
    f = np.asarray(f, np.float64)
    num = np.sin(np.pi * f * R / FS_PDM)
    den = R * np.sin(np.pi * f / FS_PDM)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(f == 0, 1.0, num / den)
    return np.abs(h) ** N_CIC


def design_fir():
    """Least-squares symmetric FIR at 16 kHz: 1 / |H_cic| on 0..4 kHz, a smooth roll-off above."""
    f = np.linspace(0, FS_OUT / 2, 2001)
    target = np.where(f <= 4000, 1.0 / cic_response(f), 0.0)
    roll = (f > 4000) & (f < 7000)
    target[roll] = (1.0 / cic_response(f[roll])) * 0.5 * (1 + np.cos(np.pi * (f[roll] - 4000) / 3000))
    w = np.where(f <= 4000, 20.0, np.where(f >= 7000, 2.0, 0.2))   # pass band weighted, roll-off loose
    half = NTAP // 2
    # h symmetric: H(f) = h0 + 2 sum_k h_k cos(2 pi f k / fs)
    A = np.stack([np.ones_like(f)] + [2 * np.cos(2 * np.pi * f * k / FS_OUT) for k in range(1, half + 1)], 1)
    c, *_ = np.linalg.lstsq(A * w[:, None], target * w, rcond=None)
    h = np.concatenate([c[:0:-1], c])                # taps h[-half] .. h[half]
    hq = np.round(h * (1 << H_FRAC)).astype(np.int64)
    return hq


H_FIR = design_fir()


def response_db(f):
    """Overall CIC + quantised FIR response in dB."""
    k = np.arange(NTAP) - NTAP // 2
    hf = np.abs(np.sum(H_FIR[None, :] / (1 << H_FRAC) * np.exp(-2j * np.pi * np.outer(f, k) / FS_OUT), 1))
    return 20 * np.log10(np.maximum(cic_response(f) * hf, 1e-12))


def decimate(bits):
    """PDM bits (0/1, len multiple of 64) -> int16 samples at 16 kHz (the hardware's arithmetic)."""
    s = 2 * np.asarray(bits, np.int64) - 1
    for _ in range(N_CIC):                          # integrators, mod 2^32 (exact: |cumsum| < 2^53)
        s = np.cumsum(s) & MASK32
    c = s[R - 1::R]
    for _ in range(N_CIC):                          # combs at 16 kHz, initial state 0
        c = (c - np.concatenate([[0], c[:-1]])) & MASK32
    c = np.where(c >= 1 << 31, c - (1 << 32), c)
    y1 = (c + (1 << 14)) >> 15
    acc = np.convolve(y1, H_FIR)[:len(y1)]          # history starts at 0
    y = np.clip((acc + (1 << (H_FRAC - 1))) >> H_FRAC, -32768, 32767).astype(np.int64)
    return dc_block(y)


def dc_block(y):
    """First-order DC blocker on int16 samples (the hardware's arithmetic, state starts at 0)."""
    out = np.zeros(len(y), np.int64)
    acc, prev = 0, 0
    for n, v in enumerate(y.tolist()):
        acc = acc + ((v - prev) << 8) - (acc >> 8)
        prev = v
        out[n] = min(max((acc + 128) >> 8, -32768), 32767)
    return out


_LIB = None
_C_SRC = r"""
#include <stdint.h>
#include <stddef.h>
/* linear interpolation x up, second-order sigma-delta modulator (CIFB), bits 0/1 */
void sdm2(const int16_t *x, int n, int up, double gain, uint8_t *bits) {
    double v1 = 0.0, v2 = 0.0, y = -1.0, a = 0.0;
    for (int i = 0; i < n; i++) {
        double b = x[i] * gain / 32768.0;
        for (int k = 0; k < up; k++) {
            double u = a + (b - a) * (double)(k + 1) / (double)up;
            v1 += u - y;
            v2 += v1 - y;
            y = (v2 >= 0.0) ? 1.0 : -1.0;
            bits[(size_t)i * (size_t)up + (size_t)k] = (uint8_t)(y > 0.0);
        }
        a = b;
    }
}
"""


def _lib():
    global _LIB
    if _LIB is None:
        d = tempfile.mkdtemp(prefix="pdm_sdm_")
        src, so = os.path.join(d, "sdm.c"), os.path.join(d, "sdm.so")
        open(src, "w").write(_C_SRC)
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, src])
        _LIB = ctypes.CDLL(so)
        _LIB.sdm2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_void_p]
    return _LIB


def pdm_from_audio(x16, gain=0.5):
    """16 kHz int16 audio -> PDM bits at 1.024 MHz (uint8 0/1)."""
    x = np.ascontiguousarray(np.asarray(x16, np.int16))
    bits = np.zeros(len(x) * R, np.uint8)
    _lib().sdm2(x.ctypes.data, len(x), R, gain, bits.ctypes.data)
    return bits


def rtl_tables():
    """rtl/pdm_tables.svh: the decimator's constants and FIR taps."""
    here = os.path.dirname(os.path.abspath(__file__))
    taps = ", ".join(f"16'sd{v}" if v >= 0 else f"-16'sd{-v}" for v in H_FIR.tolist())
    s = ["// Generated by hopcim/model/pdm_dec.py -- do not edit.",
         "// The live audio path's decimator (rtl/pdm_dec.sv, R7): CIC 5 x /64, then this FIR at 16 kHz.", "",
         f"localparam int unsigned PD_R = {R}, PD_NCIC = {N_CIC}, PD_NTAP = {NTAP}, PD_HFRAC = {H_FRAC};",
         f"localparam logic signed [15:0] PD_H [{NTAP}] = '{{{taps}}};"]
    open(os.path.join(here, "..", "rtl", "pdm_tables.svh"), "w").write("\n".join(s) + "\n")


def write_vectors(clip_index=4):
    """tb/pdm_vec/: PDM bits of a 1-s netvec clip (bits.hex) and the decimator's samples (samples.hex)."""
    import stream_int as si
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "tb", "pdm_vec")
    os.makedirs(out, exist_ok=True)
    x = si.speech_stream(20)[16000 * clip_index: 16000 * (clip_index + 1)]
    bits = pdm_from_audio(x)
    y = decimate(bits)
    open(os.path.join(out, "bits.hex"), "w").write("\n".join(str(int(b)) for b in bits) + "\n")
    open(os.path.join(out, "samples.hex"), "w").write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in y) + "\n")
    print(f"pdm vectors: clip {clip_index}, {len(bits)} bits -> {len(y)} samples (range {y.min()}..{y.max()})")


if __name__ == "__main__":
    rtl_tables()
    write_vectors()
    f = np.array([100, 500, 1000, 2000, 3000, 3500, 4000, 5000, 6000, 7000, 7900])
    print("FIR taps (Q14):", H_FIR.tolist())
    print("response dB:", {int(a): round(float(b), 2) for a, b in zip(f, response_db(f))})
    fa = np.array([12000, 13000, 14000, 15000, 16000 - 100, 17000, 18000, 19000, 20000])
    print("CIC alias attenuation into 0..4 kHz, dB:", {int(a): round(float(20 * np.log10(cic_response(a) + 1e-30)), 1) for a in fa})
    t = np.arange(16000)
    tone = np.round(0.5 * 32767 * np.sin(2 * np.pi * 1000 * t / 16000)).astype(np.int16)
    y = decimate(pdm_from_audio(tone, gain=0.9))
    ref = np.sin(2 * np.pi * 1000 * t / 16000)
    seg = slice(2000, 15000)
    a = np.linalg.lstsq(np.stack([np.sin(2 * np.pi * 1000 * t[seg] / 16000), np.cos(2 * np.pi * 1000 * t[seg] / 16000)], 1),
                        y[seg].astype(np.float64), rcond=None)[0]
    fit = a[0] * np.sin(2 * np.pi * 1000 * t[seg] / 16000) + a[1] * np.cos(2 * np.pi * 1000 * t[seg] / 16000)
    snr = 10 * np.log10(np.sum(fit ** 2) / np.sum((y[seg] - fit) ** 2))
    print(f"1 kHz tone through PDM and the decimator: amplitude {np.hypot(*a):.0f}, SNR {snr:.1f} dB")
