#!/usr/bin/env python3
"""Test vectors for tb_cim_seq (R6b layer sequencer), written to hopcim/tb/seq_vec/:
  tiles.hex   network tiles 8..68 as endpoint weight words (61 tiles x 64 rows x 4 words)
  params.hex  the parameter memory: 640 entries x (bias, qm, e = -shift)
  feat.hex    FEAT words of the 20 netvec clips (20 x 49 rows x 3 words)
  logits.hex  golden int8 logits (dscnn_int.forward, = TFLite reference kernels), 20 x 12
  acts.hex    every layer output of clip 0 (L0..L8: 9 x 125 positions x 64 channels)
  meta.hex    number of clips, the SEQ_ZP word, the golden class of each clip
  synth_*.hex the same network with synthetic requantisation parameters and zero points (shifts
              0..15, multipliers with exact powers of two, activation zero point 5 so that no ReLU
              clamp hides negative rounding): params, SEQ_ZP + class, every layer and the logits
              of clip 0. The real model only uses shifts 5..9 and zero point -128.
  synth2_*.hex the same with activation zero point -128 (the sign gate of the baseline S is active),
              shifts 0..15 and power-of-two multipliers, and biases chosen so that acc + bias is
              exactly -1, 0 or +1 at one position of every channel of L0..L8 (the gate's boundary)
Also writes the board debug app's header (sw/applications/hopcim_seq_dbg/dbgvec.h: golden conv1
and depthwise-0 outputs of clip 0; external linkage, so that it lives in ram1)."""
import os

import numpy as np

import cim_net_map as cnm
import dscnn_int as d
from gen_c import carr
from gen_net import clips

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tb", "seq_vec")
DBG = os.path.expanduser("~/APlus_Project/planenest/hw/x-heep/sw/applications/hopcim_seq_dbg")


def tile_words(t):
    """Endpoint weight words of tile t: word q of row r holds columns 4q .. 4q+3 (byte k = 4q + k)."""
    w = cnm.TILES[t].astype(np.int64) & 0xFF                      # [64, 16]
    return [int(w[r, 4 * q] | w[r, 4 * q + 1] << 8 | w[r, 4 * q + 2] << 16 | w[r, 4 * q + 3] << 24)
            for r in range(64) for q in range(4)]


def param_words():
    """Entry 64 L + c: bias (int32), qm (in [2^30, 2^31)), e = -shift (right shifts only)."""
    P = d.P
    w = np.zeros((640, 3), np.int64)
    for L in range(10):
        b = P[f"L{L}_b"].astype(np.int64)
        n = len(b)
        q = np.broadcast_to(P[f"L{L}_q"], (n,)).astype(np.int64)
        sh = np.broadcast_to(P[f"L{L}_sh"], (n,)).astype(np.int64)
        assert (sh <= 0).all() and (-sh < 16).all(), "the sequencer supports right shifts only"
        assert ((q >= 1 << 30) & (q < 1 << 31)).all() and (np.abs(b) < 1 << 23).all()
        w[64 * L: 64 * L + n] = np.stack([b & 0xFFFFFFFF, q, -sh], 1)
    return w


def feat_words(f):
    """FEAT words of int8 features [49, 10]: word w of row r holds bytes 4w .. 4w+3 of the row."""
    u = np.zeros((49, 12), np.int64)
    u[:, :10] = np.asarray(f, np.int64) & 0xFF
    return [int(u[r, 4 * k] | u[r, 4 * k + 1] << 8 | u[r, 4 * k + 2] << 16 | u[r, 4 * k + 3] << 24)
            for r in range(49) for k in range(3)]


def zp_word():
    P = d.P
    return (int(P["L0_zin"]) & 0xFF) | (int(P["L1_zin"]) & 0xFF) << 8 | (int(P["L9_zout"]) & 0xFF) << 16


def synth_params(seed=7, zact=5):
    """Requantisation parameters and zero points that cover the sequencer's whole arithmetic."""
    P, rng = d.P, np.random.default_rng(seed)
    S = {"L0_zin": np.int64(-37)}
    for L in range(10):
        n = len(P[f"L{L}_b"])
        m = len(P[f"L{L}_q"])                                       # 1 for the per-tensor FC
        e = np.clip(-P[f"L{L}_sh"].astype(np.int64) + rng.integers(-3, 5, m), 0, 15)
        e[rng.random(m) < 0.1] = 0
        e[rng.random(m) < 0.05] = 15
        q = rng.integers(1 << 30, 1 << 31, m)
        k = rng.random(m)
        q[k < 0.25] = 1 << 30
        sel = (k >= 0.25) & (k < 0.5)
        q[sel] = (1 << 30) + (1 << rng.integers(0, 30, sel.sum()))
        S[f"L{L}_q"], S[f"L{L}_sh"] = q.astype(np.int64), -e
        S[f"L{L}_b"] = P[f"L{L}_b"].astype(np.int64) + rng.integers(-(1 << 12), 1 << 12, n)
        if L >= 1:
            S[f"L{L}_zin"] = np.int64(zact)
        S[f"L{L}_zout"] = np.int64(zact if L < 9 else -3)
    return S


def layer_acc(feat, L):
    """acc + bias of every output of layer L (L0..L8) for the current d.P: the argument of mbqm."""
    got, real, n = [], d.mbqm, [0]

    def spy(a, q, sh):
        if n[0] == L:
            got.append(np.array(a))
        n[0] += 1
        return real(a, q, sh)
    d.mbqm = spy
    try:
        d.forward(feat)
    finally:
        d.mbqm = real
    return got[0]                                                    # [25, 5, 64]


def synth2_params(feat, seed=11):
    """synth_params with zero point -128, then biases that put acc + bias at -1, 0, +1 (by channel) at one position
    of each channel, layer by layer (a layer's accumulators depend on the biases before it)."""
    S = synth_params(seed, zact=-128)
    rng = np.random.default_rng(seed + 1)
    d.P.update(S)
    hits = []
    for L in range(9):
        x = layer_acc(feat, L).reshape(125, 64)
        acc = x - d.P[f"L{L}_b"]
        pos = rng.integers(0, 125, 64)
        tgt = np.array([(-1, 0, 1)[c % 3] for c in range(64)])
        d.P[f"L{L}_b"] = (tgt - acc[pos, np.arange(64)]).astype(np.int64)
        x = layer_acc(feat, L).reshape(125, 64)
        assert (x[pos, np.arange(64)] == tgt).all()
        hits.append((int((x == 0).sum()), int((x == 1).sum()), int((x == -1).sum()), float((x <= 0).mean())))
    return {k: d.P[k] for k in S}, hits


def write(name, vals, width=8):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(f"{v & ((1 << 4 * width) - 1):0{width}x}" for v in vals) + "\n")


def main(n=20):
    os.makedirs(OUT, exist_ok=True)
    P = d.P
    assert all(int(P[f"L{L}_zout"]) == int(P["L1_zin"]) for L in range(9))   # one activation zero point
    assert all(int(P[f"L{L}_zin"]) == int(P["L1_zin"]) for L in range(1, 10))
    write("tiles.hex", [v for t in range(cnm.T_CONV1, cnm.T_FC + 1) for v in tile_words(t)])
    write("params.hex", param_words().reshape(-1))
    cl = clips(n)
    feats, logits, classes = [], [], []
    for i, (q, _) in enumerate(cl):
        y, acts = d.forward(q, keep=True)
        feats += feat_words(q)
        logits += [int(v) for v in y]
        classes.append(int(np.argmax(y)))
        if i == 0:
            write("acts.hex", [int(v) for a in acts[:9] for v in np.asarray(a).reshape(-1)], 2)
            os.makedirs(DBG, exist_ok=True)
            with open(os.path.join(DBG, "dbgvec.h"), "w") as f:
                f.write("\n".join([
                    "// Generated for the R6b board debug: golden conv1 (L0) and depthwise-0 (L1) outputs of netvec clip 0",
                    "#include <stdint.h>",
                    carr("DBG_ACT", "int8_t", np.stack([np.asarray(acts[0]).reshape(-1), np.asarray(acts[1]).reshape(-1)]),
                         const=False)]) + "\n")
    write("feat.hex", feats)
    write("logits.hex", logits, 2)
    write("meta.hex", [n, zp_word()] + classes)
    print(f"seq vectors: {n} clips, classes {classes}, SEQ_ZP 0x{zp_word():06x} -> {os.path.relpath(OUT)}")

    saved = dict(d.P)
    d.P.update(synth_params())
    try:
        y, acts = d.forward(cl[0][0], keep=True)
        assert np.array_equal(y, cnm.forward_mapped(cl[0][0]))
        write("synth_params.hex", param_words().reshape(-1))
        write("synth_acts.hex", [int(v) for a in acts[:9] for v in np.asarray(a).reshape(-1)], 2)
        write("synth_logits.hex", [int(v) for v in y], 2)
        write("synth_meta.hex", [zp_word(), int(np.argmax(y))])
        A = np.stack([np.asarray(a).reshape(-1) for a in acts[:9]])
        e = np.concatenate([-d.P[f"L{L}_sh"] for L in range(10)])
        print(f"synthetic: shifts {e.min()}..{e.max()}, SEQ_ZP 0x{zp_word():06x}; share of outputs clamped "
              f"{np.mean((A == 127) | (A == -128)):.2f}, distinct values per layer {[len(np.unique(a)) for a in A]}")
    finally:
        d.P.clear()
        d.P.update(saved)

    try:
        S2, hits = synth2_params(cl[0][0])
        d.P.update(S2)
        y, acts = d.forward(cl[0][0], keep=True)
        assert np.array_equal(y, cnm.forward_mapped(cl[0][0]))
        write("synth2_params.hex", param_words().reshape(-1))
        write("synth2_acts.hex", [int(v) for a in acts[:9] for v in np.asarray(a).reshape(-1)], 2)
        write("synth2_logits.hex", [int(v) for v in y], 2)
        write("synth2_meta.hex", [zp_word(), int(np.argmax(y))])
        e = np.concatenate([-d.P[f"L{L}_sh"] for L in range(10)])
        print(f"synthetic 2: shifts {e.min()}..{e.max()}, SEQ_ZP 0x{zp_word():06x}; per layer (acc + bias = 0, +1, -1, share <= 0): {hits}")
    finally:
        d.P.clear()
        d.P.update(saved)


if __name__ == "__main__":
    main()
