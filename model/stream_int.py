#!/usr/bin/env python3
"""R5b golden model: the streaming window path, as the hardware computes it.

A decision at hop h (h >= 49) uses the 1-s window of the last 16,000 samples, i.e. hop blocks
h-49 .. h (320 samples each). Its 49 frames are stream frames m = h-49 .. h-1 (frame m covers
samples 320 m .. 320 m + 479), so every frame's lraw is computed once and shared by 49 windows.
Only the normalisation by the window maximum (mfcc_int.finish: lraw - log2fix16(max)) differs
between windows.

Per hop the hardware:
  1. pushes the maximum of the hop's 320 samples into a ring of 50 block maxima;
  2. stores the lraw of the newest frame (m = h - 1) in ring row (h - 1) mod 49;
  3. computes lmx16 = log2fix16(window max). If it changed since the previous hop, it recomputes
     the features of all 49 rows; otherwise only those of the new row (FEAT is a ring of 49 rows,
     frame 0 of the window at row h mod 49).
After every hop all rows of arrived frames are normalised with the current lmx16 (induction), so
the features equal the per-window reference whenever the window is complete. self_test checks
this against mfcc_int.cim_mfcc_int on every window of a long stream of real speech.
A window maximum <= 0 is undefined in the reference; the hardware uses 1 (log2fix16(1) = 0).
The same model serves both front-ends: mfcc_int (cim_fe) and mfcc_ded (baseline B3) differ only in
frame_lraw; the window path is shared.
"""
import numpy as np

import mfcc_ded as md
import mfcc_int as mi

HOP, NROW, NBLK = 320, 49, 50


class StreamModel:
    def __init__(self, ded=False):
        self.frame_lraw = md.frame_lraw if ded else mi.frame_lraw
        self.lraw = np.zeros((NROW, mi.NMEL), np.int64)
        self.bmax = np.full(NBLK, -32768, np.int64)
        self.lmx = None
        self.feat = np.zeros((NROW, mi.NMFCC), np.int64)
        self.n_all = self.n_one = 0

    def hop(self, x, h):
        """Process hop h of the int16 stream x; returns the window's features [49, 10] if h >= 49."""
        self.bmax[h % NBLK] = int(x[HOP * h: HOP * (h + 1)].max())
        frow = h % NROW
        new = (frow + NROW - 1) % NROW
        if h >= 1:
            m = h - 1
            self.lraw[new] = self.frame_lraw(x[HOP * m: HOP * m + mi.WIN].astype(np.int64))
        lmx = mi.log2fix16(max(int(self.bmax.max()), 1))
        if lmx != self.lmx:
            self.feat[:] = mi.finish(self.lraw, lmx)
            self.lmx = lmx
            self.n_all += 1
        elif h >= 1:
            self.feat[new] = mi.finish(self.lraw[new], lmx)
            self.n_one += 1
        if h >= NROW:
            return self.feat[(frow + np.arange(NROW)) % NROW]
        return None


def window_ref(x, h, ded=False):
    """The per-window reference at hop h >= 49: the MFCC of the last 16,000 samples."""
    f = md.mfcc if ded else mi.cim_mfcc_int
    return f(x[HOP * (h - 49): HOP * (h + 1)]).reshape(NROW, mi.NMFCC)


def speech_stream(n_clips=20):
    """The netvec clips (gen_net.clips selection) back to back, each 1 s, as one int16 stream."""
    import glob
    import os
    import wave
    from gen_net import LABELS, TEST
    files = []
    for lab in LABELS:
        files += sorted(glob.glob(os.path.join(TEST, lab, "*.wav")))
    parts = []
    for p in files[::len(files) // n_clips][:n_clips]:
        with wave.open(p, "rb") as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)
        parts.append(np.pad(a, (0, max(0, mi.NS - len(a))))[:mi.NS])
    return np.concatenate(parts)


def self_test(n_clips=20, ded=False):
    x = speech_stream(n_clips)
    sm = StreamModel(ded)
    hops = len(x) // HOP
    bad = 0
    for h in range(hops):
        f = sm.hop(x, h)
        if f is not None:
            bad += int(not np.array_equal(f, window_ref(x, h, ded)))
    n_dec = hops - NROW
    print(f"[{'B3' if ded else 'cim_fe'}] stream of {len(x) / mi.NS:.0f} s, {hops} hops, {n_dec} decisions: {bad} windows differ from the "
          f"per-window reference; window path recomputed all 49 frames at {sm.n_all} hops, one frame at {sm.n_one}")
    return bad == 0


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sys.exit(0 if (self_test(n) and self_test(n, ded=True)) else 1)
