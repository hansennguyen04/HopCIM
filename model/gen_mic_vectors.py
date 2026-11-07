#!/usr/bin/env python3
"""Test vectors for tb_cim_mic (R7, the live audio path end to end): the 3-s speech stream of
gen_stream_vectors.py through the sigma-delta microphone model (pdm_dec.pdm_from_audio), the
hardware decimator (pdm_dec.decimate) and the streaming model (stream_int.StreamModel), in
hopcim/tb/mic_vec/:
  bits.hex           the PDM bit stream (one bit per line)
  samples.hex        the decimated 16 kHz stream (what the ring must hold)
  logits_{cim,ded}.hex, meta_{cim,ded}.hex   per decision: golden logits; number, classes
Also reports how the decisions on the decimated audio compare with those on the original audio."""
import os

import numpy as np

import dscnn_int as d
import gen_stream_vectors as gsv
import pdm_dec as pd
import stream_int as si

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tb", "mic_vec")


def decisions(x, ded):
    sm = si.StreamModel(ded)
    out = []
    for h in range(len(x) // si.HOP):
        f = sm.hop(x, h)
        if f is not None:
            out.append(d.forward(f))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    x = gsv.stream()
    bits = pd.pdm_from_audio(x)
    y = pd.decimate(bits)
    with open(os.path.join(OUT, "bits.hex"), "w") as f:
        f.write("\n".join(str(int(b)) for b in bits) + "\n")
    gsv.write(os.path.join(OUT, "samples.hex"), y, 4)
    for ded in (False, True):
        tag = "ded" if ded else "cim"
        lg = decisions(y, ded)
        cls = [int(np.argmax(v)) for v in lg]
        ref = [int(np.argmax(v)) for v in decisions(x, ded)]
        gsv.write(os.path.join(OUT, f"logits_{tag}.hex"), np.stack(lg).reshape(-1), 2)
        gsv.write(os.path.join(OUT, f"meta_{tag}.hex"), [len(lg)] + cls, 8)
        same = sum(a == b for a, b in zip(cls, ref))
        print(f"[{tag}] {len(bits)} PDM bits -> {len(y)} samples, {len(lg)} decisions; classes on the decimated "
              f"audio equal those on the original at {same}/{len(lg)}")
        print("   decimated:", cls[::5])
        print("   original: ", ref[::5])


if __name__ == "__main__":
    main()
