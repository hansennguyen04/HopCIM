#!/usr/bin/env python3
"""Check a board recording of the live audio path (firmware hopcim_micrec, R7).

The firmware records 1.5 s of the on-board microphone as the endpoint's audio ring holds it,
with every decision the hardware made meanwhile. This script:
  1. decodes the recording (int16 at 16 kHz) and reports its level and spectrum;
  2. for every decision whose 1-s window lies inside the recording, recomputes the logits with
     the golden model (stream_int + dscnn_int) and compares them with the board's;
  3. optionally writes the recording as a .wav file.
Usage: check_micrec.py <uart.log> [--ded] [--wav out.wav] [--k <k>]
--k: the firmware was built with -DNET_K=<k>, one decision every k hops (the n-th at hop 49 + (n - 1) k).
"""
import argparse
import sys
import wave

import numpy as np

import dscnn_int as d
import stream_int as si

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
LABEL = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_silence_", "_unknown_"]
FS = 16000


def parse(path):
    h0 = nh = clk = ovr = None
    s, dec = [], []
    for ln in open(path, errors="replace"):
        t = ln.split()
        if not t:
            continue
        if t[0] == "REC":
            h0, nh, clk, ovr = int(t[1]), int(t[2]), int(t[3]), int(t[5])
        elif t[0] == "S" and len(t) == 2 and len(t[1]) == 96:
            for j in range(32):
                c = t[1][3 * j: 3 * j + 3]
                u = (B64.index(c[0]) << 12) | (B64.index(c[1]) << 6) | B64.index(c[2])
                s.append(u - 65536 if u >= 32768 else u)
        elif t[0] == "D" and len(t) == 16:
            dec.append((int(t[1]), int(t[2]), int(t[3]), [int(v) for v in t[4:]]))
    if h0 is None:
        sys.exit("no REC line")
    x = np.array(s, np.int16)
    if len(x) != 320 * nh:
        sys.exit(f"{len(x)} samples decoded, {320 * nh} expected")
    return h0, nh, clk, ovr, x, dec


def spectrum(x):
    """Mean power per band (dB re 1 LSB^2 per bin) from a Hann-windowed 512-point Welch estimate."""
    w = np.hanning(512)
    segs = [x[i: i + 512] * w for i in range(0, len(x) - 512 + 1, 256)]
    p = np.mean([np.abs(np.fft.rfft(sg)) ** 2 for sg in segs], axis=0) / np.sum(w ** 2)
    f = np.fft.rfftfreq(512, 1 / FS)
    return f, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ded", action="store_true", help="the bitstream uses the B3 front-end")
    ap.add_argument("--wav")
    ap.add_argument("--k", type=int, default=1, help="the network ran every k hops")
    a = ap.parse_args()
    h0, nh, clk, ovr, x, dec = parse(a.log)
    xf = x.astype(np.float64)
    print(f"recording: hops {h0} .. {h0 + nh - 1} ({len(x)} samples, {len(x) / FS:.2f} s) at {clk} Hz, "
          f"overrun {ovr}")
    print(f"  min {x.min()} max {x.max()} mean {xf.mean():.1f} rms {xf.std():.1f}")
    f, p = spectrum(xf - xf.mean())
    tot = p.sum()
    for lo, hi in ((0, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8001)):
        m = (f >= lo) & (f < hi)
        print(f"  {lo:5d} - {min(hi, 8000):4d} Hz: {100 * p[m].sum() / tot:5.1f} % of the power")
    top = np.argsort(p)[::-1][:5]
    print("  strongest bins: " + ", ".join(f"{f[k]:.0f} Hz" for k in sorted(top)))
    if a.wav:
        with wave.open(a.wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes(x.astype("<i2").tobytes())
        print(f"  written {a.wav}")

    # decisions: board hop h is recording hop h - h0; its window is inside from recording hop 49 on
    sm = si.StreamModel(a.ded)
    gold = {}
    for r in range(nh):
        feat = sm.hop(x, r)
        if feat is not None:
            gold[h0 + r] = np.asarray(d.forward(feat)).astype(int)
    n_chk = n_bad = n_align = 0
    classes = []
    for hop, cnt, cls, lg in dec:
        classes.append(cls)
        if hop != 49 + (cnt - 1) * a.k:
            n_align += 1
        if hop in gold:
            n_chk += 1
            g = gold[hop]
            if [int(v) for v in g] != lg or int(np.argmax(g)) != cls:
                n_bad += 1
                if n_bad <= 5:
                    print(f"  hop {hop}: board {cls} {lg}, model {int(np.argmax(g))} {[int(v) for v in g]}")
    print(f"decisions: {len(dec)} recorded, one every {a.k} hops ({n_align} with hop != 49 + (count - 1) k); "
          f"{n_chk} with the window inside the recording: {n_bad} differ from the golden model")
    hist = {LABEL[c]: classes.count(c) for c in sorted(set(classes))}
    print("  classes: " + ", ".join(f"{k} {v}" for k, v in hist.items()))
    return 0 if (n_chk > 0 and n_bad == 0 and n_align == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
