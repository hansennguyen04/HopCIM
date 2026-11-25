#!/usr/bin/env python3
"""Integer reference of MLPerf Tiny's image-classification model ResNet8 (pretrainedResnet_quant.tflite) with
TensorFlow Lite reference-kernel semantics: the golden model for running it on the CIM.

  conv            acc = sum W (x - zp_in) over the taps inside the input (SAME padding skipped) + bias; per-channel
                  MultiplyByQuantizedMultiplier (double rounding) + zp_out, clamped (activation RELU: also >= zp_out)
  add             TFLite's int8 ADD: both inputs rescaled to a common scale (left shift 20), summed, rescaled
  average pool    sum of the 64 int8 values divided by 64, rounding half away from zero
  fully connected as conv, per-tensor multiplier
  python resnet_int.py export        write resnet_params.npz from the .tflite (needs TensorFlow)
  python resnet_int.py bounds        accumulator and shift ranges the hardware must support
  python resnet_int.py check [n]     every layer against TFLite's reference interpreter on n CIFAR-10 test images
"""
import os
import pickle
import sys

import numpy as np

from dscnn_int import mbqm, quantize_multiplier, rdbp, srdhm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "..", "mlperf_tiny", "tiny", "benchmark", "training", "image_classification", "trained_models", "pretrainedResnet_quant.tflite")
CIFAR = os.path.join(HERE, "..", "..", "planenest", "data", "cifar-10-batches-py", "test_batch")
PARAMS = os.path.join(HERE, "resnet_params.npz")
# (kind, name, tensor outputs of TFLite's operators): topology of the graph, operators 0..14
TOPO = [("conv", "c0", 0), ("conv", "c1", 1), ("conv", "c2", 2), ("add", "a3", 3), ("conv", "c4", 4), ("conv", "c5", 5),
        ("conv", "c6", 6), ("add", "a7", 7), ("conv", "c8", 8), ("conv", "c9", 9), ("conv", "c10", 10), ("add", "a11", 11),
        ("pool", "p12", 12), ("fc", "fc", 14)]
WIRE = {"c0": ["in"], "c1": ["c0"], "c2": ["c1"], "a3": ["c0", "c2"], "c4": ["a3"], "c5": ["c4"], "c6": ["a3"], "a7": ["c6", "c5"],
        "c8": ["a7"], "c9": ["c8"], "c10": ["a7"], "a11": ["c10", "c9"], "p12": ["a11"], "fc": ["p12"]}


def export():
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as S
    buf = open(MODEL, "rb").read()
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(buf, 0))
    g = m.subgraphs[0]
    it = tf.lite.Interpreter(model_path=MODEL)
    it.allocate_tensors()
    det = {d["index"]: d for d in it.get_tensor_details()}
    out = {}
    ops = {}
    for i, o in enumerate(g.operators):
        ops[i] = o
    # the wiring in the file must be WIRE's: follow tensor producers
    prod = {}
    for i, o in enumerate(g.operators):
        for t in o.outputs:
            prod[int(t)] = i
    name_of = {op: n for _, n, op in TOPO}
    for kind, n, opi in TOPO:
        o = ops[opi]
        feeds = [int(t) for t in o.inputs if int(t) >= 0 and int(t) not in {int(x) for x in (o.inputs[1:] if kind in ("conv", "fc") else [])}]
        srcs = []
        for t in feeds:
            p = prod.get(t)
            while p is not None and g.operators[p].opcodeIndex == ops[13].opcodeIndex:   # RESHAPE
                p = prod.get(int(g.operators[p].inputs[0]))
            srcs.append("in" if p is None else name_of[p])
        out[f"{n}.src"] = np.array(srcs)
        assert sorted(srcs) == sorted(WIRE[n]), (n, srcs, WIRE[n])
        tin = int(o.inputs[0])
        q = lambda t: (float(det[t]["quantization_parameters"]["scales"][0]), int(det[t]["quantization_parameters"]["zero_points"][0]))
        to = int(o.outputs[0])
        out[f"{n}.si"], out[f"{n}.zi"] = q(tin)
        out[f"{n}.so"], out[f"{n}.zo"] = q(to)
        if kind in ("conv", "fc"):
            wt = int(o.inputs[1])
            bt = int(o.inputs[2])
            w = np.frombuffer(m.buffers[g.tensors[wt].buffer].data.tobytes(), np.int8).reshape(g.tensors[wt].shape).copy()
            b = np.frombuffer(m.buffers[g.tensors[bt].buffer].data.tobytes(), np.int32).copy()
            sw = np.asarray(det[wt]["quantization_parameters"]["scales"], np.float64)
            zw = np.asarray(det[wt]["quantization_parameters"]["zero_points"])
            assert (zw == 0).all(), (n, "weight zero point")
            sw = np.broadcast_to(sw, (w.shape[0],))
            mul = [quantize_multiplier(float(out[f"{n}.si"]) * float(s) / float(out[f"{n}.so"])) for s in sw]
            out[f"{n}.w"], out[f"{n}.b"] = w, b
            out[f"{n}.q"] = np.array([x[0] for x in mul], np.int64)
            out[f"{n}.sh"] = np.array([x[1] for x in mul], np.int64)
            if kind == "conv":
                bo = o.builtinOptions
                out[f"{n}.stride"] = np.array(bo.strideH)
                out[f"{n}.relu"] = np.array(1 if bo.fusedActivationFunction == 1 else 0)
            else:
                out[f"{n}.relu"] = np.array(0)
        elif kind == "add":
            s1, z1 = q(int(o.inputs[0]))
            s2, z2 = q(int(o.inputs[1]))
            assert out[f"{n}.si"] == s1
            # srcs lists inputs in order: out[name.src]
            tw = 2 * max(s1, s2)
            q1 = quantize_multiplier(s1 / tw)
            q2 = quantize_multiplier(s2 / tw)
            qo = quantize_multiplier(tw / ((1 << 20) * float(out[f"{n}.so"])))
            out[f"{n}.add"] = np.array([z1, z2, q1[0], q1[1], q2[0], q2[1], qo[0], qo[1]], np.int64)
            out[f"{n}.relu"] = np.array(1 if o.builtinOptions.fusedActivationFunction == 1 else 0)
    out["zin"] = np.array(int(det[int(g.inputs[0])]["quantization_parameters"]["zero_points"][0]))
    out["out_index"] = np.array(int(g.operators[14].outputs[0]))
    np.savez(PARAMS, **out)
    print("wrote", PARAMS, {k: out[k].shape if hasattr(out[k], "shape") else out[k] for k in out if k.endswith(".w")})


def same_pad(n_in, k, s):
    n_out = -(-n_in // s)
    tot = max((n_out - 1) * s + k - n_in, 0)
    return n_out, tot // 2


def conv(x, P, n):
    """x: [N, H, W, C] int64 -> y: [N, H', W', K] int8 (as int64); P: parameters npz."""
    w, b = P[f"{n}.w"].astype(np.int64), P[f"{n}.b"].astype(np.int64)
    s, zi, zo = int(P[f"{n}.stride"]), int(P[f"{n}.zi"]), int(P[f"{n}.zo"])
    K, kh, kw, C = w.shape
    N, H, W, _ = x.shape
    oh, ph = same_pad(H, kh, s)
    ow, pw = same_pad(W, kw, s)
    xz = np.zeros((N, H + kh + 1, W + kw + 1, C), np.float64)     # SAME padding = value 0 (after zero point)
    xz[:, ph:ph + H, pw:pw + W, :] = x - zi
    W2 = w.astype(np.float64).reshape(K, -1)                      # exact in float64: |acc| < 2^53
    y = np.empty((N, oh, ow, K), np.int64)
    for i in range(oh):
        for j in range(ow):
            patch = xz[:, i * s:i * s + kh, j * s:j * s + kw, :].reshape(N, -1)
            y[:, i, j, :] = np.rint(patch @ W2.T).astype(np.int64)
    acc = y + b
    r = mbqm(acc, P[f"{n}.q"], P[f"{n}.sh"]) + zo
    lo = zo if int(P[f"{n}.relu"]) else -128
    return np.clip(r, max(lo, -128), 127)


def add(x1, x2, P, n):
    z1, z2, q1, s1, q2, s2, qo, so = [int(v) for v in P[f"{n}.add"]]
    zo = int(P[f"{n}.zo"])
    a = (x1 - z1) << 20
    b = (x2 - z2) << 20
    a = rdbp(srdhm(a, q1), -s1)
    b = rdbp(srdhm(b, q2), -s2)
    r = rdbp(srdhm(a + b, qo), -so) + zo
    lo = zo if int(P[f"{n}.relu"]) else -128
    return np.clip(r, max(lo, -128), 127)


def pool(x):
    s = x.reshape(x.shape[0], -1, x.shape[-1]).sum(1)
    n = x.shape[1] * x.shape[2]
    return np.where(s >= 0, (s + n // 2) // n, -((-s + n // 2) // n))


def fc(x, P, n):
    w, b = P[f"{n}.w"].astype(np.int64), P[f"{n}.b"].astype(np.int64)
    acc = (x - int(P[f"{n}.zi"])) @ w.T + b
    q = P[f"{n}.q"]
    return np.clip(mbqm(acc, q, P[f"{n}.sh"]) + int(P[f"{n}.zo"]), -128, 127)


def forward(x8, P, keep=False):
    """x8: [N, 32, 32, 3] int8 input of the network (uint8 - 128)."""
    v = {"in": x8.astype(np.int64)}
    for kind, n, _ in TOPO:
        if kind == "conv":
            v[n] = conv(v[WIRE[n][0]], P, n)
        elif kind == "add":
            src = [str(s) for s in P[f"{n}.src"]]
            v[n] = add(v[src[0]], v[src[1]], P, n)
        elif kind == "pool":
            v[n] = pool(v[WIRE[n][0]]).reshape(-1, v[WIRE[n][0]].shape[-1])
        else:
            v[n] = fc(v[WIRE[n][0]], P, n)
    return v if keep else v["fc"]


def cifar(n=10000):
    d = pickle.load(open(CIFAR, "rb"), encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[:n]
    return (x.astype(np.int64) - 128).astype(np.int8), np.array(d[b"labels"][:n])


def bounds():
    P = np.load(PARAMS, allow_pickle=False)
    print("layer  kind   max |acc| (bits)  shifts           |x-zp| max  relu")
    for kind, n, _ in TOPO:
        if kind in ("conv", "fc"):
            w = np.abs(P[f"{n}.w"].astype(np.int64)).reshape(P[f"{n}.w"].shape[0], -1).sum(1)
            zi = int(P[f"{n}.zi"])
            xm = max(127 - zi, 128 + zi)
            m = int((w * xm + np.abs(P[f"{n}.b"])).max())
            sh = P[f"{n}.sh"]
            print(f"{n:5s} {kind:5s} {m:12d} ({m.bit_length() + 1:2d})  [{sh.min()}, {sh.max()}]   {xm}   {int(P[f'{n}.relu'])}")
        elif kind == "add":
            print(f"{n:5s} add    {P[f'{n}.add'].tolist()} relu {int(P[f'{n}.relu'])}")


def check(n):
    import tensorflow as tf
    P = np.load(PARAMS)
    x, lab = cifar(n)
    it = tf.lite.Interpreter(model_path=MODEL, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ii = it.get_input_details()[0]["index"]
    tens = {}
    from tensorflow.lite.python import schema_py_generated as S
    g = S.ModelT.InitFromObj(S.Model.GetRootAsModel(open(MODEL, "rb").read(), 0)).subgraphs[0]
    for kind, nm, op in TOPO:
        tens[nm] = int(g.operators[op].outputs[0])
    bad = {nm: 0 for _, nm, _ in TOPO}
    fcbad = tfbad = agree = correct = 0
    B = 200
    for i in range(0, n, B):
        v = forward(x[i:i + B], P, keep=True)
        for j in range(len(x[i:i + B])):
            it.set_tensor(ii, x[i + j:i + j + 1])
            it.invoke()
            for _, nm, _ in TOPO:
                t = it.get_tensor(tens[nm]).reshape(v[nm][j].shape)
                bad[nm] += int(not np.array_equal(t, v[nm][j]))
            soft = it.get_tensor(int(it.get_output_details()[0]["index"]))[0]
            ours = v["fc"][j]
            agree += int(np.argmax(ours) == np.argmax(soft))
            correct += int(np.argmax(ours) == lab[i + j])
        print(f"{i + len(x[i:i + B])} images: mismatching layers {sum(bad.values())}, argmax = TFLite {agree}, correct {correct}", flush=True)
    print("images with a mismatch, per layer:", bad)
    print(f"accuracy {100 * correct / n:.2f} % ({correct}/{n}); argmax equal to TFLite's softmax on {agree}/{n}")
    return sum(bad.values()) == 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bounds"
    if cmd == "export":
        export()
    elif cmd == "bounds":
        bounds()
    else:
        sys.exit(0 if check(int(sys.argv[2]) if len(sys.argv) > 2 else 200) else 1)
