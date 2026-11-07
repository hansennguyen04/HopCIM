#!/usr/bin/env python3
"""Integer reference of the MLPerf Tiny int8 DS-CNN (kws_ref_model.tflite) with TFLite
reference-kernel semantics; the golden model for running the network on the CIM.

  conv / depthwise : acc = sum W * (x - zp_in) over the taps inside the input (SAME padding taps are
                     skipped) + bias; per-channel requantisation (MultiplyByQuantizedMultiplier,
                     double rounding) + zp_out, clamped to int8
  average pool     : sum of the 125 int8 values, divided by 125 rounding half away from zero
  fully connected  : as conv, per-tensor multiplier
The prediction is the argmax of the fully connected int8 output; the softmax is monotone (checked
against TFLite's softmax argmax in check_vs_tflite). Parameters: dscnn_params.npz (export()).
  python dscnn_int.py export   write dscnn_params.npz from the .tflite (needs TensorFlow)
  python dscnn_int.py check    compare every layer with the TFLite BUILTIN_REF interpreter on all
                               4,890 test clips, fed with the integer CIM-MFCC features
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "..", "kws_gate", "mlperf_tiny_kws", "trained_models", "kws_ref_model.tflite")
PARAMS = os.path.join(HERE, "dscnn_params.npz")


# ---------------------------------------------------------------- TFLite fixed-point arithmetic
def quantize_multiplier(m):
    """tflite::QuantizeMultiplier: m (double) -> (q in [2^30, 2^31), shift)."""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    qf = int(math.floor(abs(q) * (1 << 31) + 0.5)) * (1 if q >= 0 else -1)      # std::round
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return qf, shift


def srdhm(a, b):
    """gemmlowp SaturatingRoundingDoublingHighMul for int32 a (array) and b (scalar or array)."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    r = (ab + nudge)
    r = np.where(r >= 0, r >> 31, -((-r) >> 31))          # C++ integer division by 2^31 (toward zero)
    sat = (a == -(1 << 31)) & (b == -(1 << 31))
    return np.where(sat, (1 << 31) - 1, r)


def rdbp(x, e):
    """gemmlowp RoundingDivideByPOT (round half away from zero)."""
    x = np.asarray(x, np.int64)
    e = np.asarray(e, np.int64)
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def mbqm(x, q, shift):
    """MultiplyByQuantizedMultiplier (double rounding, the TFLite default)."""
    shift = np.asarray(shift, np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return rdbp(srdhm(np.asarray(x, np.int64) * (np.int64(1) << left), q), right)


# ---------------------------------------------------------------- parameters
def export():
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=MODEL, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    out = {}
    in_q = td[ops[0]["inputs"][0]]["quantization_parameters"]
    out["in_scale"], out["in_zp"] = np.float64(in_q["scales"][0]), np.int64(in_q["zero_points"][0])
    li = 0
    for op in ops:
        name = op["op_name"]
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            continue
        x, w, b = (td[i] for i in op["inputs"][:3])
        y = td[op["outputs"][0]]
        s_in = np.float64(x["quantization_parameters"]["scales"][0])
        s_out = np.float64(y["quantization_parameters"]["scales"][0])
        s_w = w["quantization_parameters"]["scales"].astype(np.float64)
        qs = [quantize_multiplier(float(s_in * sw / s_out)) for sw in s_w]
        out[f"L{li}_kind"] = np.array(name)
        out[f"L{li}_w"] = it.get_tensor(w["index"]).astype(np.int64)
        out[f"L{li}_b"] = it.get_tensor(b["index"]).astype(np.int64)
        out[f"L{li}_q"] = np.array([q for q, _ in qs], np.int64)
        out[f"L{li}_sh"] = np.array([s for _, s in qs], np.int64)
        out[f"L{li}_zin"] = np.int64(x["quantization_parameters"]["zero_points"][0])
        out[f"L{li}_zout"] = np.int64(y["quantization_parameters"]["zero_points"][0])
        out[f"L{li}_tensor"] = np.int64(op["outputs"][0])
        li += 1
    out["nlayers"] = np.int64(li)
    out["pool_tensor"] = np.int64([op["outputs"][0] for op in ops if op["op_name"] == "AVERAGE_POOL_2D"][0])
    out["softmax_tensor"] = np.int64(ops[-1]["outputs"][0])
    np.savez(PARAMS, **out)
    print(f"exported {li} layers to {PARAMS}")


P = dict(np.load(PARAMS, allow_pickle=False)) if os.path.exists(PARAMS) else None


def same_pad(n_in, n_out, k, s):
    return max(((n_out - 1) * s + k - n_in) // 2, 0)


def conv(x, L, stride):
    """x: int8 [H, W, Cin] -> int8 [Ho, Wo, Cout] (TFLite CONV_2D, SAME, per-channel)."""
    w, b, zin, zout = P[f"L{L}_w"], P[f"L{L}_b"], int(P[f"L{L}_zin"]), int(P[f"L{L}_zout"])
    co, kh, kw, ci = w.shape
    H, W_, _ = x.shape
    Ho, Wo = -(-H // stride), -(-W_ // stride)
    ph, pw = same_pad(H, Ho, kh, stride), same_pad(W_, Wo, kw, stride)
    xp = np.zeros((H + kh + stride * Ho, W_ + kw + stride * Wo, ci), np.int64)   # zero = (x - zin) outside
    xp[ph:ph + H, pw:pw + W_] = x.astype(np.int64) - zin
    acc = np.zeros((Ho, Wo, co), np.int64)
    for fy in range(kh):
        for fx in range(kw):
            patch = xp[fy: fy + stride * Ho: stride, fx: fx + stride * Wo: stride]  # [Ho, Wo, ci]
            acc += patch @ w[:, fy, fx, :].T
    acc += b
    y = mbqm(acc, P[f"L{L}_q"], P[f"L{L}_sh"]) + zout
    return np.clip(y, -128, 127)


def dwconv(x, L):
    """TFLite DEPTHWISE_CONV_2D 3x3, stride 1, SAME, depth multiplier 1, per-channel."""
    w, b, zin, zout = P[f"L{L}_w"], P[f"L{L}_b"], int(P[f"L{L}_zin"]), int(P[f"L{L}_zout"])
    _, kh, kw, c = w.shape
    H, W_, _ = x.shape
    xp = np.zeros((H + kh - 1, W_ + kw - 1, c), np.int64)
    xp[kh // 2: kh // 2 + H, kw // 2: kw // 2 + W_] = x.astype(np.int64) - zin
    acc = np.zeros((H, W_, c), np.int64)
    for fy in range(kh):
        for fx in range(kw):
            acc += xp[fy: fy + H, fx: fx + W_] * w[0, fy, fx]
    acc += b
    y = mbqm(acc, P[f"L{L}_q"], P[f"L{L}_sh"]) + zout
    return np.clip(y, -128, 127)


def avgpool(x):
    """TFLite AVERAGE_POOL_2D over the whole 25 x 5 map (int8, in and out quantisation equal)."""
    acc = x.astype(np.int64).reshape(-1, x.shape[-1]).sum(0)
    n = x.shape[0] * x.shape[1]
    q = np.where(acc > 0, (acc + n // 2) // n, -((-(acc - n // 2)) // n))       # C++: truncate toward zero
    return np.clip(q, -128, 127)


def fc(x, L):
    w, b, zin, zout = P[f"L{L}_w"], P[f"L{L}_b"], int(P[f"L{L}_zin"]), int(P[f"L{L}_zout"])
    acc = w @ (x.astype(np.int64) - zin) + b
    y = mbqm(acc, P[f"L{L}_q"][0], P[f"L{L}_sh"][0]) + zout
    return np.clip(y, -128, 127)


def forward(feat, keep=False):
    """int8 features [49, 10] (or [49, 10, 1]) -> int8 logits [12]; keep=True also returns every layer."""
    x = np.asarray(feat, np.int64).reshape(49, 10, 1)
    acts = []
    x = conv(x, 0, 2); acts.append(x)
    for blk in range(4):
        x = dwconv(x, 1 + 2 * blk); acts.append(x)
        x = conv(x, 2 + 2 * blk, 1); acts.append(x)
    p = avgpool(x); acts.append(p)
    y = fc(p, 9); acts.append(y)
    return (y, acts) if keep else y


def check_vs_tflite():
    import glob
    import tensorflow as tf
    sys.path.insert(0, os.path.join(HERE, "..", "..", "kws_gate"))
    import gate2_eval as g
    import mfcc_int as mi
    it = tf.lite.Interpreter(model_path=MODEL, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    xnn = g.load_interp(g.MODELS["int8"])                                         # default (XNNPACK) interpreter
    inp = it.get_input_details()[0]
    tens = [int(P[f"L{l}_tensor"]) for l in range(int(P["nlayers"]))]
    tens = tens[:9] + [int(P["pool_tensor"])] + tens[9:]
    files = []
    for li, lab in enumerate(g.LABELS):
        files += [(p, li) for p in sorted(glob.glob(os.path.join(g.TEST, lab, "*.wav")))]
    bad = np.zeros(len(tens), int)
    n_ok = n_ref = n_xnn = argmax_diff = ref_vs_xnn = 0
    for i, (p, li) in enumerate(files):
        q = mi.cim_mfcc_int(g.read_wav(p))
        y, acts = forward(q, keep=True)
        it.set_tensor(inp["index"], q[None])
        it.invoke()
        for k, t in enumerate(tens):
            bad[k] += int(not np.array_equal(acts[k].ravel(), it.get_tensor(t).astype(np.int64).ravel()))
        sm = it.get_tensor(int(P["softmax_tensor"])).ravel()
        pr_ref, pr_ours = int(np.argmax(sm)), int(np.argmax(y))
        argmax_diff += int(pr_ref != pr_ours)
        n_ref += int(pr_ref == li)
        n_ok += int(pr_ours == li)
        # default interpreter on the same int8 input
        it2, inp2, out2 = xnn
        it2.set_tensor(inp2["index"], q[None])
        it2.invoke()
        pr_x = int(np.argmax(it2.get_tensor(out2["index"])))
        n_xnn += int(pr_x == li)
        ref_vs_xnn += int(pr_x != pr_ref)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    n = len(files)
    names = ["conv1"] + [f"{k}{b}" for b in range(1, 5) for k in ("dw", "pw")] + ["pool", "fc"]
    print("layer mismatches (clips):", dict(zip(names, bad.tolist())))
    print(f"argmax(fc) != argmax(TFLite softmax): {argmax_diff} clips")
    print(f"accuracy on integer CIM-MFCC features: ours {100 * n_ok / n:.2f} %, TFLite reference kernels {100 * n_ref / n:.2f} %, "
          f"TFLite default (XNNPACK) {100 * n_xnn / n:.2f} %; reference vs default predictions differ on {ref_vs_xnn} clips")
    return int(bad.sum()) == 0 and argmax_diff == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0 if check_vs_tflite() else 1)
    else:
        print(__doc__)
