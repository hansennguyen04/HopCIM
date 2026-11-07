#!/usr/bin/env python3
"""Test vectors for the hardware front-ends (tb/tb_cim_fe.sv): frames of 480 int16 samples with
their golden lraw[40], plus the 8 front-end tiles.
  python gen_fe_vectors.py        cim_fe: lraw from mfcc_int.frame_lraw, written to tb/fe_vec/
  python gen_fe_vectors.py ded    B3 mfcc_ded: lraw from mfcc_ded.frame_lraw, written to tb/ded_vec/
Frames: every frame of two GSC test clips and synthetic corner cases (silence, full scale,
impulses, tiny amplitudes, block exponents of 0 and below)."""
import glob
import os
import sys
import wave

import numpy as np

import cim_map as cm
import mfcc_ded as md
import mfcc_int as mi

HERE = os.path.dirname(os.path.abspath(__file__))
DED = len(sys.argv) > 1 and sys.argv[1] == "ded"
OUT = os.path.join(HERE, "..", "tb", "ded_vec" if DED else "fe_vec")
FRAME_LRAW = md.frame_lraw if DED else mi.frame_lraw
TEST = os.path.join(HERE, "..", "..", "kws_gate", "data", "test")


def clip_frames(path):
    with wave.open(path, "rb") as w:
        x16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)
    x = np.zeros(mi.NS, np.int64)
    x[:len(x16)] = x16
    return [x[f * mi.HOP: f * mi.HOP + mi.WIN] for f in range(mi.NFR)]


def synthetic(rng):
    n = np.arange(mi.WIN)
    fr = [np.zeros(mi.WIN, np.int64),                                   # silent
          np.full(mi.WIN, 32767), np.full(mi.WIN, -32768),              # full-scale DC
          np.where(n % 2 == 0, 32767, -32768),                          # Nyquist, full scale
          np.where(n == 240, 32767, 0), np.where(n == 1, -32768, 0),    # impulses (centre, edge)
          rng.integers(-32768, 32768, mi.WIN),                          # full-scale noise
          rng.integers(-1, 2, mi.WIN),                                  # +-1: tiny, sh < 0
          np.where(n == 100, 1, 0),                                     # one LSB
          (np.round(20000 * np.sin(2 * np.pi * 1000 * n / 16000))).astype(np.int64),
          (np.round(30000 * np.sin(2 * np.pi * (200 + 7000 * n / mi.WIN) * n / 16000))).astype(np.int64)]
    for a in (1, 2, 3, 7, 300):                                         # peaks around 2^14..2^15 (sh ~ 0)
        fr.append(np.round(a * np.cos(2 * np.pi * 440 * n / 16000)).astype(np.int64))
    for a in (16, 1000, 12000):
        fr.append(rng.integers(-a, a + 1, mi.WIN))
    for pos, a in ((1, 1), (1, -1), (3, 1), (10, 5), (479, 7), (2, -3)):   # tiny peaks: sh down to -14
        fr.append(np.where(n == pos, a, 0))
    return fr


def main():
    rng = np.random.default_rng(7)
    files = [os.path.join(TEST, "yes", "022cd682_nohash_0.wav")] + \
        [sorted(glob.glob(os.path.join(TEST, lab, "*.wav")))[3] for lab in ("stop", "go", "_unknown_", "_silence_", "left")]
    frames = [f for p in files for f in clip_frames(p)] + synthetic(rng)
    frames = [np.clip(f, -32768, 32767).astype(np.int64) for f in frames]
    lraw = [FRAME_LRAW(f) for f in frames]
    lraw32 = [np.where(l == mi.ZERO, -(1 << 31), l) & 0xFFFFFFFF for l in lraw]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "samples.hex"), "w") as h:
        for f in frames:
            h.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in f) + "\n")
    with open(os.path.join(OUT, "lraw.hex"), "w") as h:
        for l in lraw32:
            h.write("\n".join(f"{int(v):08x}" for v in l) + "\n")
    t = cm.TILES.astype(np.int64) & 0xFF                                # [8][64][16] -> words, byte k = column 4q+k
    words = (t[..., 0::4] | t[..., 1::4] << 8 | t[..., 2::4] << 16 | t[..., 3::4] << 24).reshape(-1)
    with open(os.path.join(OUT, "tiles.hex"), "w") as h:
        h.write("\n".join(f"{int(v):08x}" for v in words) + "\n")
    with open(os.path.join(OUT, "nframes.txt"), "w") as h:
        h.write(f"{len(frames)}\n")
    nz = sum(int((l == mi.ZERO).any()) for l in lraw)
    sh = [int(np.abs(f * mi.HANN_Q15).max()).bit_length() - 15 for f in frames if np.abs(f).max() > 0]
    print(f"{len(frames)} frames ({len(files)} clips + {len(frames) - 49 * len(files)} synthetic), "
          f"frames with ZERO entries {nz}, block exponent range {min(sh)}..{max(sh)}")


if __name__ == "__main__":
    main()
