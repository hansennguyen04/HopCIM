#!/usr/bin/env python3
"""Gate 2: unchanged MLPerf Tiny DS-CNN, reference float MFCC vs CIM-MFCC (see GATE2_MANIFEST.md)."""
import glob, json, math, os, sys, wave
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "data", "test")
MODELS = {"int8": os.path.join(HERE, "mlperf_tiny_kws/trained_models/kws_ref_model.tflite"),
          "float32": os.path.join(HERE, "mlperf_tiny_kws/trained_models/kws_ref_model_float32.tflite")}
LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_silence_", "_unknown_"]
NS, WIN, HOP, NFFT, NB, NMEL, NMFCC, NFR = 16000, 480, 320, 512, 257, 40, 10, 49

# ------------------------------------------------------------------ reference (TF, = get_dataset.py)
MEL_TF = tf.signal.linear_to_mel_weight_matrix(NMEL, NB, NS, 20.0, 4000.0).numpy()   # [257, 40]


def ref_mfcc(x16):
    w = tf.cast(tf.constant(x16), tf.float32)
    w = w / tf.reduce_max(w)
    w = tf.pad(w, [[0, NS - tf.shape(w)[-1]]])
    w = tf.slice(tf.pad(w, [[2, 2]]), [2], [NS])
    st = tf.signal.stft(w, frame_length=WIN, frame_step=HOP, fft_length=None, window_fn=tf.signal.hann_window)
    spec = tf.abs(st)
    mel = tf.tensordot(spec, MEL_TF, 1)
    lm = tf.math.log(mel + 1e-6)
    mf = tf.signal.mfccs_from_log_mel_spectrograms(lm)[..., :NMFCC]
    return tf.reshape(mf, [NFR, NMFCC, 1]).numpy()


# ------------------------------------------------------------------ CIM-MFCC integer datapath
def rshift(x, s):
    return (x + (np.int64(1) << (s - 1))) >> s if s > 0 else x


N1, N2 = 16, 32
_n1 = np.arange(N1)
W16 = np.exp(-2j * np.pi * np.outer(_n1, _n1) / N1)
M1q = np.clip(np.round(np.concatenate([W16.real, W16.imag], 1) * 127), -127, 127).astype(np.int64)      # 16 x 32
_n2 = np.arange(N2)
TW = np.exp(-2j * np.pi * np.outer(np.arange(N1), _n2) / NFFT)
TWr, TWi = np.round(TW.real * 16384).astype(np.int64), np.round(TW.imag * 16384).astype(np.int64)        # Q14
K2 = 17
W32 = np.exp(-2j * np.pi * np.outer(_n2, np.arange(K2)) / N2)
M3q = np.clip(np.round(np.block([[W32.real, W32.imag], [-W32.imag, W32.real]]) * 127), -127, 127).astype(np.int64)  # 64 x 34
BIN = np.array([[k1 + N1 * k2 for k2 in range(K2)] for k1 in range(N1)])                              # [k1, k2] -> bin
VALID = BIN < NB
HANN_Q15 = np.round((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)) * 32767).astype(np.int64)    # periodic Hann
MEL_S = 127.0 / MEL_TF.max()
MELq = np.clip(np.round(MEL_TF * MEL_S), 0, 127).astype(np.int64)                                      # [257, 40]
LOG_LUT = np.round(np.log2(1 + np.arange(64) / 64.0) * 256).astype(np.int64)                           # Q8
_k, _n = np.meshgrid(np.arange(NMEL), np.arange(NMEL), indexing="ij")
DCT_TF = (2 * np.cos(np.pi * _k * (2 * _n + 1) / (2 * NMEL)) / np.sqrt(2 * NMEL)).T                    # [n, k]
DCT_S = 32767.0 / np.abs(DCT_TF).max()
DCTq = np.round(DCT_TF * DCT_S).astype(np.int64)                                                       # int16 (2 CIM passes)
LN2, LN_FLOOR = math.log(2), math.log(1e-6)


def isqrt(v):
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64)           # then correct to the exact integer root
    r -= (r * r > v)
    r += ((r + 1) * (r + 1) <= v)
    return r


def log2_q8(v):                                                             # v >= 1, int64
    e = np.floor(np.log2(v.astype(np.float64))).astype(np.int64)           # leading-one position (priority encoder)
    e -= (np.left_shift(np.int64(1), e) > v)                                # guard float rounding
    e += (np.left_shift(np.int64(1), e + 1) <= v)
    frac = np.where(e >= 6, np.right_shift(v, np.maximum(e - 6, 0)), np.left_shift(v, np.maximum(6 - e, 0))) & 63
    return e * 256 + LOG_LUT[frac]


def cim_mfcc(x16):
    x = np.zeros(NS, np.int64)
    x[:len(x16)] = x16
    mx = int(np.max(x16))                                                   # per-clip max (reference divides by max)
    if mx <= 0:
        raise ValueError("clip max <= 0: reference normalisation undefined")
    out = np.zeros((NFR, NMFCC))
    for f in range(NFR):
        fr = x[f * HOP: f * HOP + WIN] * HANN_Q15                           # wide product (Q15 * int16)
        buf = np.zeros(NFFT, np.int64)
        buf[:WIN] = fr
        peak = int(np.abs(buf).max())
        if peak == 0:                                                       # all-zero (padding) frame
            lmq = np.full(NMEL, int(round(LN_FLOOR * 256)), np.int64)
            out[f] = (lmq @ DCTq)[:NMFCC] / (256.0 * DCT_S)
            continue
        sh = int(math.floor(math.log2(peak))) - 14                          # block exponent: peak -> [2^14, 2^15)
        xb = rshift(buf, sh) if sh > 0 else np.left_shift(buf, -sh)
        X = xb.reshape(N1, N2)                                               # x[N2*n1 + n2]
        A = rshift(X.T @ M1q, 11)                                            # [n2, 32] step 1 (CIM)
        Ar, Ai = A[:, :N1].T, A[:, N1:].T                                    # [k1, n2]
        Br = rshift(Ar * TWr - Ai * TWi, 14)                                 # twiddle unit
        Bi = rshift(Ar * TWi + Ai * TWr, 14)
        Xs = rshift(np.concatenate([Br, Bi], 1) @ M3q, 9)                    # [k1, 34] step 3 (CIM)
        re, im = np.zeros(NB, np.int64), np.zeros(NB, np.int64)
        re[BIN[VALID]], im[BIN[VALID]] = Xs[:, :K2][VALID], Xs[:, K2:][VALID]
        mag = isqrt(re * re + im * im)                                       # exact integer magnitude
        mel = mag @ MELq                                                     # CIM mel (int8 weights)
        # value scale: |X|_float = mag * 2^(sh + 11 + 9 - 15) / (127 * 127) / max(x)   (window Q15, two int8 DFTs)
        scale_log2 = (sh + 11 + 9 - 15) - math.log2(127 * 127) - math.log2(MEL_S) - math.log2(mx)
        lm = np.where(mel > 0, (log2_q8(np.maximum(mel, 1)) / 256.0 + scale_log2) * LN2, LN_FLOOR)
        lm = np.maximum(lm, LN_FLOOR)
        lmq = np.round(lm * 256).astype(np.int64)                            # log-mel to the DCT in Q8
        out[f] = (lmq @ DCTq)[:NMFCC] / (256.0 * DCT_S)                      # CIM DCT (int16 coeffs)
    return out.reshape(NFR, NMFCC, 1)


# ------------------------------------------------------------------ evaluation
def read_wav(p):
    with wave.open(p, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == NS
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)


def load_interp(path):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


def predict(model, feat, quant_mode="reference"):
    it, inp, out = model
    if inp["dtype"] == np.int8:
        s, zp = inp["quantization"]
        if quant_mode == "reference":
            q = np.array(feat[None] / s + zp, dtype=np.int8)                # exactly as eval_quantized_model.py
        else:
            q = np.clip(np.round(feat[None] / s + zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q)
    else:
        it.set_tensor(inp["index"], feat[None].astype(np.float32))
    it.invoke()
    return int(np.argmax(it.get_tensor(out["index"])))


def main():
    files = []
    for li, lab in enumerate(LABELS):
        files += [(p, li) for p in sorted(glob.glob(os.path.join(TEST, lab, "*.wav")))]
    print(f"test clips: {len(files)}")
    models = {k: load_interp(v) for k, v in MODELS.items()}
    s8, zp8 = models["int8"][1]["quantization"]
    print(f"int8 model input quantisation: scale={s8:.6f} zero_point={zp8}")
    pred = {f"{m}|{fe}|{q}": [] for m in models for fe in ("ref", "cim") for q in (("reference", "round") if m == "int8" else ("-",))}
    labels, err_steps = [], []
    for i, (p, li) in enumerate(files):
        x = read_wav(p)
        fr, fc = ref_mfcc(x), cim_mfcc(x)
        err_steps.append(np.abs(fr - fc).ravel() / s8)
        labels.append(li)
        for m, mod in models.items():
            for fe, feat in (("ref", fr), ("cim", fc)):
                for q in (("reference", "round") if m == "int8" else ("-",)):
                    pred[f"{m}|{fe}|{q}"].append(predict(mod, feat, q if q != "-" else "reference"))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    labels = np.array(labels)
    res = {}
    for k, v in pred.items():
        res[k] = float(np.mean(np.array(v) == labels) * 100)
    err = np.concatenate(err_steps)

    def mcnemar(a, b):
        a, b = np.array(a) == labels, np.array(b) == labels
        n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
        n = n01 + n10
        from math import comb
        pval = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
        return n01, n10, pval

    summary = dict(n=len(labels), accuracy=res,
                   delta_int8_reference_quant=res["int8|cim|reference"] - res["int8|ref|reference"],
                   delta_int8_round_quant=res["int8|cim|round"] - res["int8|ref|round"],
                   delta_float32=res["float32|cim|-"] - res["float32|ref|-"],
                   mcnemar_int8=mcnemar(pred["int8|ref|reference"], pred["int8|cim|reference"]),
                   mcnemar_float32=mcnemar(pred["float32|ref|-"], pred["float32|cim|-"]),
                   feature_abs_error_in_input_steps=dict(mean=float(err.mean()), p50=float(np.percentile(err, 50)),
                                                         p99=float(np.percentile(err, 99)), max=float(err.max()),
                                                         frac_below_half_step=float(np.mean(err < 0.5))))
    summary["gate_pass"] = summary["delta_int8_reference_quant"] >= -0.30
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.join(HERE, "gate2_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
