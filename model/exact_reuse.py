#!/usr/bin/env python3
"""Software evaluation of exact activation reuse across overlapping windows (not in the RTL).

Question: how many of the DS-CNN's 7,501 CIM passes per decision can be skipped while every int8
logit stays equal to independent full inference, and what does that buy against simply running the
network less often? Results and interpretation: docs/REUSE_EVALUATION.md.

Reuse: window N + lag has the same frames as window N shifted by lag frames. For an even lag, the
stride-2 grid of conv1 is aligned, so output row r of the new window can take row r + lag / 2 of
the cached window wherever both see the same inputs. A decision every hop therefore needs two cache
sets (one per stride phase); a decision every k >= 2 even hops needs one. A position must be
recomputed if its receptive field reaches padding in one window and real data in the other, holds
new frames, or holds changed features.

ReuseModel is functional, not an estimate: every reused value comes from its cache and every
recomputed one from the layer functions applied to the reconstructed input, so a wrong dirty mask
corrupts later layers and the logits. It never reads the full run. run_stream compares it with an
independent dscnn_int.forward at every decision, layer by layer.

Modes (each includes the one before):
  simple  fixed geometry; full recompute if the window maximum changed since the cached window
  feat    fixed geometry, plus the actual difference of the aligned int8 features
  value   feat, plus: a recomputed checkpoint position equal to its cached value does not propagate
Caches: conv1 and pointwise 1..4, the 5 checkpoints. Depthwise outputs are not cached, so a recomputed
pointwise position needs all 10 depthwise groups of that position, and equality can only be tested at
the checkpoints. "Oracle" counts only the outputs that really change, with channel-group depthwise
passes: the lower bound for any reuse scheme, needing 9 cached tensors and perfect knowledge.

Commands:
  python exact_reuse.py static                  fixed geometry: passes per lag and cached subset
  python exact_reuse.py self-test [clips]       the checks below on a short stream (default 20 clips)
  python exact_reuse.py run <stream> <out.npz>  one stream: doc<n> | dense:<seed>:<clips> |
                                                on:<seed>:<seconds>:<spacing_s>:<noise_rms>
  python exact_reuse.py suite <outdir> [jobs]   the 16 streams of REUSE_EVALUATION.md
  python exact_reuse.py report <outdir>         tables from a suite, also written as results.json
"""
import glob
import json
import os
import sys
import time
import wave
from math import comb

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import dscnn_int as d  # noqa: E402
import mfcc_int as mi  # noqa: E402
import stream_int as si  # noqa: E402

R, C, CH, NF, NC = 25, 5, 64, 49, 10
FULL_PASSES = 7501
FULL_MACS = 125 * 2560 + 4 * 125 * (576 + 4096) + 768      # 2,656,768: conv1, 4 x (depthwise + pointwise), FC
MODES = ("simple", "feat", "value")
LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_silence_", "_unknown_"]
KEYWORDS = LABELS[:10]
TEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kws_gate", "data", "test")
# measured per-layer cycles (SEQUENCER.md section 5): overhead = layer cycles - passes * rate
OVERHEAD = {"conv1": 8034 - 500 * 16, "dw": 11273 - 1250 * 9, "pw": 8032 - 500 * 16, "end": 98}


# ---------------------------------------------------------------- fixed geometry (no audio)
def conv1_dirty_rows(s):
    """conv1 rows that cannot be reused for a shift of s output rows (2 s frames)."""
    bad = set()
    for r in range(R):
        if r + s >= R:
            bad.add(r)
            continue
        for t in range(2 * r - 4, 2 * r + 6):              # SAME padding: 4 rows before, 5 after
            new_pad = t < 0 or t >= NF
            old_pad = t + 2 * s < 0 or t + 2 * s >= NF
            if new_pad != old_pad:
                bad.add(r)
                break
    return bad


def dilate_rows(x):
    return {q for r in x for q in (r - 1, r, r + 1) if 0 <= q < R}


def static_plan(s, cached):
    """cached: subset of 0..4 (conv1, pointwise 1..4). Returns the rows computed per checkpoint: a cached
    tensor supplies its clean rows; anything else a consumer needs is computed, pulling in its inputs."""
    dirty = [conv1_dirty_rows(s)]
    for _ in range(4):
        dirty.append(dilate_rows(dirty[-1]) | {0} | set(range(R - s, R)))
    comp = [None] * 5
    comp[4] = set(dirty[4]) if 4 in cached else set(range(R))      # the pool needs all of checkpoint 4
    for k in range(3, -1, -1):
        need = dilate_rows(comp[k + 1])
        comp[k] = (dirty[k] & need) if k in cached else need
    return comp, dirty


def static_passes(comp):
    return len(comp[0]) * C * 4 + sum(len(comp[k]) * C * 14 for k in range(1, 5)) + 1


def report_static(out=None):
    res = {"lag": {}, "subset": {}}
    print("Fixed geometry, all 5 checkpoints cached (rows that must be recomputed per checkpoint)")
    for s in (1, 2, 3, 4, 5):
        comp, dirty = static_plan(s, {0, 1, 2, 3, 4})
        p = static_passes(comp)
        res["lag"][2 * s] = {"rows": [len(x) for x in dirty], "passes": p}
        print(f"  lag {2 * s:2d} hops: rows {[len(x) for x in dirty]}  passes {p}  ({100 * (1 - p / FULL_PASSES):.1f} % fewer)")
    print("Best subset of checkpoints for each cache size, lag 2 (8 KB per tensor per stride phase)")
    for n in range(6):
        best = None
        for mask in range(32):
            sub = {i for i in range(5) if mask >> i & 1}
            if len(sub) == n:
                p = static_passes(static_plan(1, sub)[0])
                if best is None or p < best[0]:
                    best = (p, sorted(sub))
        res["subset"][n] = {"passes": best[0], "tensors": best[1]}
        names = ["conv1", "pw1", "pw2", "pw3", "pw4"]
        print(f"  {n} tensors ({8 * n:2d} KB per phase): {best[0]} passes, {100 * (1 - best[0] / FULL_PASSES):4.1f} % fewer  "
              f"[{' '.join(names[i] for i in best[1])}]")
    return res


# ---------------------------------------------------------------- the functional reuse model
def dilate(m):
    p = np.zeros((R + 2, C + 2), bool)
    p[1:-1, 1:-1] = m
    out = np.zeros((R, C), bool)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + R, dx:dx + C]
    return out


def conv1_mask(cell, s, bad):
    """Dirty conv1 positions; cell[t, f]: the aligned input cell differs, or is a new frame; bad: rows
    with padding in one window and real data in the other, or no aligned cached row."""
    # integral image over the input for the 10 x 4 receptive field (rows 2r-4..2r+5, columns 2c-1..2c+2)
    pad = np.zeros((NF + 12, NC + 4), np.int64)
    pad[5:5 + NF, 2:2 + NC] = cell
    ii = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), np.int64)
    ii[1:, 1:] = pad.cumsum(0).cumsum(1)
    m = np.zeros((R, C), bool)
    for r in range(R):
        if r in bad:
            m[r] = True
            continue
        for c in range(C):
            t0, t1, f0, f1 = 2 * r - 4 + 5, 2 * r + 5 + 5 + 1, 2 * c - 1 + 2, 2 * c + 2 + 2 + 1
            m[r, c] = (ii[t1, f1] - ii[t0, f1] - ii[t1, f0] + ii[t0, f0]) > 0
    return m


def boundary(s):
    """Rows no block output can reuse: row 0 (a padding tap against a real row) and the last s rows."""
    m = np.zeros((R, C), bool)
    m[0] = True
    m[R - s:] = True
    return m


class ReuseModel:
    def __init__(self, mode, lag):
        assert lag % 2 == 0 and mode in MODES
        self.mode, self.lag, self.s = mode, lag, lag // 2
        self.state = {}                                    # stride phase -> (features, 5 checkpoints)
        self.boundary_fn = boundary                        # self_test replaces these three to break the model
        self.rows_fn = conv1_dirty_rows
        self.phase_fn = lambda h: h % lag

    def _shifted(self, t):
        o = np.zeros_like(t)
        o[:R - self.s] = t[self.s:]
        return o

    def step(self, h, feat, lmx_changed):
        """feat: int8 [49, 10]; lmx_changed: the window maximum changed since hop h - lag."""
        phase = self.phase_fn(h)
        st = self.state.get(phase)
        feat = np.asarray(feat, np.int64)
        s = self.s
        if st is None or (self.mode == "simple" and lmx_changed):
            y, acts = d.forward(feat, keep=True)
            self.state[phase] = (feat.copy(), [acts[i].copy() for i in (0, 2, 4, 6, 8)])
            return y, dict(full=True, conv=125, blk=[125] * 4, layers=acts[:9])
        fo, to = st
        cell = np.ones((NF, NC), bool)
        if self.mode == "simple":                          # the maximum is unchanged: only new frames differ
            cell[:NF - 2 * s] = False
        else:
            cell[:NF - 2 * s] = feat[:NF - 2 * s] != fo[2 * s:]
        x = d.conv(feat.reshape(NF, NC, 1), 0, 2)
        rec = conv1_mask(cell, s, self.rows_fn(s))
        y = np.where(rec[..., None], x, self._shifted(to[0]))
        cps, layers = [y], [y]
        n_conv = int(rec.sum())
        changed = self._changed(rec, y, to[0])
        blk = []
        for k in range(4):
            rec = dilate(changed) | self.boundary_fn(s)
            dw = d.dwconv(cps[-1], 1 + 2 * k)
            pw = d.conv(dw, 2 + 2 * k, 1)
            y = np.where(rec[..., None], pw, self._shifted(to[k + 1]))
            cps.append(y)
            layers += [dw, y]
            blk.append(int(rec.sum()))
            changed = self._changed(rec, y, to[k + 1])
        logits = d.fc(d.avgpool(y), 9)
        self.state[phase] = (feat.copy(), cps)
        return logits, dict(full=False, conv=n_conv, blk=blk, layers=layers)

    def _changed(self, rec, y, old):
        if self.mode != "value":
            return rec
        ch = rec & (y != self._shifted(old)).any(-1)
        ch[R - self.s:] = rec[R - self.s:]                 # no cached value to compare with
        return ch


def costs(st):
    """Per decision: passes, MACs, ACT bytes read and written, and estimated sequencer cycles from the
    measured rates (1 and 2 requantisation lanes), with the pool re-reading the reused positions."""
    conv, blk = st["conv"], st["blk"]
    passes = conv * 4 + sum(n * 14 for n in blk) + 1
    macs = conv * 2560 + sum(n * (576 + 4096) for n in blk) + 768
    reused = 125 - blk[3]
    rd = sum(n * (576 + 64) for n in blk) + reused * 64
    wr = conv * 64 + sum(2 * n * 64 for n in blk)
    c = {}
    for lanes, rate in ((1, 16), (2, 9)):
        c[lanes] = (conv * 4 * rate + OVERHEAD["conv1"] + OVERHEAD["end"] + reused * 10 +
                    sum(n * 90 + OVERHEAD["dw"] + n * 4 * rate + OVERHEAD["pw"] for n in blk))
    return passes, macs, rd, wr, c[1], c[2]


def oracle_passes(acts, old, s=1):
    """Passes if only the outputs that really change were computed, depthwise by channel group."""
    def diff(a, b):
        m = np.ones((R, C, CH), bool)
        m[:R - s] = a[:R - s] != b[s:]
        return m
    n = 4 * int(diff(acts[0], old[0]).any(-1).sum())
    for k in range(4):
        cdw = diff(acts[1 + 2 * k], old[1 + 2 * k])
        n += sum(int(cdw[:, :, 7 * g: min(7 * g + 7, CH)].any(-1).sum()) for g in range(10))
        n += 4 * int(cdw.any(-1).sum())
    return n + 1


# ---------------------------------------------------------------- deterministic streams
def read_wav(p):
    with wave.open(p, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)
    return np.pad(a, (0, max(0, mi.NS - len(a))))[:mi.NS]


def stream_dense(seed, n_clips):
    """GSC test clips of all 12 classes back to back in a random order."""
    rng = np.random.default_rng(seed)
    files = []
    for lab in LABELS:
        files += sorted(glob.glob(os.path.join(TEST, lab, "*.wav")))
    idx = rng.permutation(len(files))[:n_clips]
    return np.concatenate([read_wav(files[i]) for i in idx]), []


def stream_always_on(seed, seconds, spacing, noise_rms):
    """A noise bed (GSC _silence_ clips scaled to one RMS) with sparse words at random offsets, 80 %
    keywords and 20 % _unknown_. Returns the stream and the manifest (start sample, label, file)."""
    rng = np.random.default_rng(seed)
    n = seconds * mi.NS
    sil = sorted(glob.glob(os.path.join(TEST, "_silence_", "*.wav")))
    parts, have = [], 0
    while have < n:
        a = read_wav(sil[rng.integers(len(sil))]).astype(np.float64)
        rms = np.sqrt(np.mean(a * a))
        parts.append(a * (noise_rms / rms) if rms > 0 else a)
        have += len(a)
    x = np.concatenate(parts)[:n]
    man = []
    t = int(rng.integers(mi.NS, 2 * mi.NS))
    while t + mi.NS < n - mi.NS:
        lab = KEYWORDS[rng.integers(10)] if rng.random() < 0.8 else "_unknown_"
        fs = sorted(glob.glob(os.path.join(TEST, lab, "*.wav")))
        p = fs[rng.integers(len(fs))]
        x[t:t + mi.NS] += read_wav(p)
        man.append((t, lab, os.path.basename(p)))
        t += int(spacing * mi.NS * rng.uniform(0.8, 1.2))
    return np.clip(np.round(x), -32768, 32767).astype(np.int64), man


def make_stream(spec):
    k = spec.split(":")
    if k[0].startswith("doc"):
        return si.speech_stream(int(k[0][3:])), []
    if k[0] == "dense":
        return stream_dense(int(k[1]), int(k[2]))
    if k[0] == "on":
        return stream_always_on(int(k[1]), int(k[2]), float(k[3]), float(k[4]))
    raise ValueError(spec)


# ---------------------------------------------------------------- one stream
def run_stream(x, lags=(2, 4)):
    """Lag 2: a decision every hop (two phases); its even hops are also a decision every 2 hops (one
    phase). Lag 4: a decision every 4 hops (one phase)."""
    sm = si.StreamModel()
    models = {(m, lag): ReuseModel(m, lag) for lag in lags for m in MODES}
    rec = {f"{m}_{lag}": [] for m, lag in models}
    bad = {k: 0 for k in rec}
    bad_layers = {k: 0 for k in rec}
    changes, logits, chg, oracle, hist = [], [], [], [], []
    prev = None
    for h in range(len(x) // si.HOP):
        f = sm.hop(x, h)
        c = sm.lmx != prev
        prev = sm.lmx
        if c:
            changes.append(h)
        if f is None:
            continue
        f = np.asarray(f, np.int64)
        y, acts = d.forward(f, keep=True)
        logits.append(y.astype(np.int8))
        chg.append(c)
        for (m, lag), mod in models.items():
            if lag == 4 and h % 4:
                continue
            yy, st = mod.step(h, f, any(h - lag < e <= h for e in changes[-6:]))
            key = f"{m}_{lag}"
            bad[key] += int(not np.array_equal(yy, y))
            bad_layers[key] += int(any(not np.array_equal(a, b) for a, b in zip(st["layers"], acts[:9])))
            rec[key].append((h, st["full"]) + costs(st))
        hist.append(acts[:9])
        if len(hist) > 3:
            hist.pop(0)
        if len(hist) == 3:
            oracle.append(oracle_passes(acts, hist[0]))
    return dict(rec={k: np.array(v, np.int64) for k, v in rec.items()}, bad=bad, bad_layers=bad_layers,
                logits=np.array(logits), chg=np.array(chg), n_change=len(changes), hops=len(x) // si.HOP,
                oracle=np.array(oracle, np.int64))


COLS = ["h", "full", "passes", "macs", "act_rd", "act_wr", "cycles_1lane", "cycles_2lanes"]


def run_one(spec, out):
    t0 = time.time()
    x, man = make_stream(spec)
    r = run_stream(x)
    np.savez_compressed(out, logits=r["logits"], chg=r["chg"], oracle=r["oracle"], **{f"rec_{k}": v for k, v in r["rec"].items()})
    with open(out + ".json", "w") as f:
        json.dump(dict(stream=spec, bad=r["bad"], bad_layers=r["bad_layers"], n_change=r["n_change"], hops=r["hops"],
                       manifest=man, seconds=len(x) / mi.NS, wall=time.time() - t0), f)
    print(f"{spec}: {r['hops']} hops, window maximum changed at {r['n_change']}; logit mismatches {sum(r['bad'].values())}, "
          f"layer mismatches {sum(r['bad_layers'].values())}; {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- decision stride and keyword detection
def events(cls, hop_of, hold):
    """hopcim_live's rule: a keyword when it has been the top class for `hold` decisions in a row, and
    the same keyword again only after a non-keyword decision."""
    last, run, said, ev = -1, 0, -1, []
    for c, h in zip(cls, hop_of):
        run = run + 1 if c == last else 1
        last = c
        if c < 10 and run == hold and c != said:
            said = c
            ev.append((int(h), int(c)))
        if c >= 10:
            said = -1
    return ev


def hold_for(k):
    """Decisions in a row that span at least the board's 80 ms (5 decisions, 4 intervals, at k = 1)."""
    return 1 + (3 + k) // k


def detection(runs, ks=(1, 2, 3, 4, 5, 8, 10, 16)):
    """Every alignment of the stride is scored. A word can be reported from its start until 1 s after
    its end (a later window still holds it). The hold spans at least the board's 80 ms, as in the firmware."""
    res = {}
    for k in ks:
        hold = hold_for(k)
        hit = wrong = miss = unk = n_kw = n_unk = fa = 0
        secs = 0.0
        for j, z in runs:
            cls = z["logits"].argmax(1)
            n = len(cls)
            words = [(t // si.HOP, (t + mi.NS) // si.HOP + 50, lab) for t, lab, _ in j["manifest"]]
            for o in range(k):
                idx = np.arange(o, n, k)
                ev = events(cls[idx], idx + 49, hold)
                used = set()
                for a, b, lab in words:
                    inside = [e for e in ev if a <= e[0] <= b]
                    used.update(inside)
                    if lab == "_unknown_":
                        n_unk += 1
                        unk += int(bool(inside))
                    else:
                        n_kw += 1
                        li = LABELS.index(lab)
                        if any(c == li for _, c in inside):
                            hit += 1
                        elif inside:
                            wrong += 1
                        else:
                            miss += 1
                fa += sum(1 for e in ev if e not in used)
                secs += j["seconds"]
        res[k] = dict(hold=hold, recall=hit / max(n_kw, 1), wrong=wrong / max(n_kw, 1), miss=miss / max(n_kw, 1),
                      unknown_alarm=unk / max(n_unk, 1), noise_alarms=fa / k, noise_alarms_per_hour=fa / secs * 3600,
                      keyword_words=n_kw // k, unknown_words=n_unk // k)
    return res


def paired_recall(runs, k):
    """Exact McNemar test of keyword hits, stride k against every hop, same words, at the hardware's
    alignment (decisions at hops 49 + n k). Returns the words only one of the two detected, and p."""
    only_1 = only_k = n = 0
    for j, z in runs:
        cls = z["logits"].argmax(1)
        ev1 = events(cls, np.arange(len(cls)) + 49, hold_for(1))
        idx = np.arange(0, len(cls), k)
        evk = events(cls[idx], idx + 49, hold_for(k))
        for t, lab, _ in j["manifest"]:
            if lab == "_unknown_":
                continue
            a, b, li = t // si.HOP, (t + mi.NS) // si.HOP + 50, LABELS.index(lab)
            h1 = any(a <= h <= b and c == li for h, c in ev1)
            hk = any(a <= h <= b and c == li for h, c in evk)
            only_1 += int(h1 and not hk)
            only_k += int(hk and not h1)
            n += 1
    m = only_1 + only_k
    tail = sum(comb(m, i) for i in range(min(only_1, only_k) + 1)) / 2 ** m if m else 1.0
    return dict(words=n, only_every_hop=only_1, only_stride=only_k, p=min(1.0, 2 * tail))


# ---------------------------------------------------------------- what an energy decision would need
def break_even(runs, e_mac_fj=(20, 50, 100), e_byte_fj=10):
    """The retention leakage per cached bit at which simple reuse stops paying, against running the
    network at the same rate without a cache. Measured: saved MACs and activation bytes per second, the
    bits a cache must keep powered between hops. Assumed, NOT data: energy per MAC and per activation byte.
    This is a threshold to compare with a process's numbers, not an energy result."""
    a = {}
    for k, key, lag, bits in ((1, "simple_2", 2, 10 * 8000 * 8), (2, "simple_2", 2, 5 * 8000 * 8), (4, "simple_4", 4, 5 * 8000 * 8)):
        rows = np.concatenate([z[f"rec_{key}"] for j, z in runs if j["stream"].startswith(("on", "dense"))])
        if k == 2:
            rows = rows[rows[:, 0] % 2 == 0]
        per_s = 50.0 / k
        d_mac = (FULL_MACS - rows[:, 3].mean()) * per_s
        d_act = (320000 + 72000 - rows[:, 4].mean() - rows[:, 5].mean()) * per_s
        a[k] = dict(saved_macs_per_s=float(d_mac), saved_act_bytes_per_s=float(d_act), cached_bits=bits,
                    pw_per_bit={str(e): float((d_mac * e + d_act * e_byte_fj) * 1e-15 / bits * 1e12) for e in e_mac_fj})
    print(f"\nLeakage break-even of simple reuse against the same decision rate without a cache (assumed {e_byte_fj} fJ per activation byte)")
    print("   k  saved MAC/s  saved ACT B/s  cached kbit   pW per bit at " + " / ".join(f"{e} fJ/MAC" for e in e_mac_fj))
    for k, v in a.items():
        print(f"  {k:2d}  {v['saved_macs_per_s'] / 1e6:8.1f} M  {v['saved_act_bytes_per_s'] / 1e6:10.2f} M  {v['cached_bits'] / 1000:10.0f}   " +
              " / ".join(f"{p:.2f}" for p in v["pw_per_bit"].values()))
    return a


# ---------------------------------------------------------------- report
def group_of(spec):
    if spec.startswith("doc"):
        return "documented stream (20 clips)"
    if spec.startswith("dense"):
        return "dense speech"
    return f"always-on, noise RMS {spec.split(':')[4]}"


def summary(a, per_hop_k=1):
    p = a[:, COLS.index("passes")]
    return dict(passes_mean=float(p.mean()), passes_per_hop=float(p.mean() / per_hop_k),
                fewer_per_hop_pct=float(100 * (1 - p.mean() / per_hop_k / FULL_PASSES)),
                p50=float(np.percentile(p, 50)), p90=float(np.percentile(p, 90)), p99=float(np.percentile(p, 99)), max=int(p.max()),
                full_pct=float(100 * a[:, 1].mean()), macs_per_hop=float(a[:, 3].mean() / per_hop_k),
                act_rd_per_hop=float(a[:, 4].mean() / per_hop_k), act_wr_per_hop=float(a[:, 5].mean() / per_hop_k),
                cycles_1lane=float(a[:, 6].mean()), cycles_2lanes=float(a[:, 7].mean()))


def report(outdir):
    runs = []
    for p in sorted(glob.glob(os.path.join(outdir, "*.npz"))):
        runs.append((json.load(open(p[:-4] + ".json")), dict(np.load(p))))
    res = dict(streams=len(runs), decisions=int(sum(len(z["logits"]) for _, z in runs)), seconds=sum(j["seconds"] for j, _ in runs))
    res["logit_mismatches"] = sum(sum(j["bad"].values()) for j, _ in runs)
    res["layer_mismatches"] = sum(sum(j["bad_layers"].values()) for j, _ in runs)
    print(f"{res['streams']} streams, {res['seconds']:.0f} s, {res['decisions']} decisions; logit mismatches {res['logit_mismatches']}, "
          f"layer mismatches {res['layer_mismatches']} (all modes, lags 2 and 4)")
    res["static"] = report_static()
    groups = {}
    for r in runs:
        groups.setdefault(group_of(r[0]["stream"]), []).append(r)
    res["groups"] = {}
    for g, rs in sorted(groups.items()):
        chg = np.concatenate([z["chg"] for _, z in rs])
        o = np.concatenate([z["oracle"] for _, z in rs])
        e = dict(streams=len(rs), decisions=len(chg), max_change_pct=float(100 * chg.mean()),
                 oracle_mean=float(o.mean()), oracle_fewer_pct=float(100 * (1 - o.mean() / FULL_PASSES)))
        print(f"\n{g}: {len(rs)} stream(s), {len(chg)} decisions; window maximum changed at {e['max_change_pct']:.1f} % of them")
        for m in MODES:
            a = np.concatenate([z[f"rec_{m}_2"] for _, z in rs])
            b = np.concatenate([z[f"rec_{m}_4"] for _, z in rs])
            e[f"{m}_k1"] = summary(a)
            e[f"{m}_k2"] = summary(a[a[:, 0] % 2 == 0], 2)
            e[f"{m}_k4"] = summary(b, 4)
            for name, kk in (("k1", 1), ("k2", 2), ("k4", 4)):
                s = e[f"{m}_{name}"]
                print(f"  {m:6s} k={kk}: passes/hop {s['passes_per_hop']:6.0f} ({s['fewer_per_hop_pct']:5.1f} % fewer), per decision "
                      f"p50 {s['p50']:5.0f} p90 {s['p90']:5.0f} p99 {s['p99']:5.0f} max {s['max']:5d}, full {s['full_pct']:4.1f} %, "
                      f"cycles {s['cycles_1lane']:6.0f} ({s['cycles_2lanes']:6.0f} with 2 lanes)")
        print(f"  oracle k=1: {e['oracle_mean']:.0f} passes ({e['oracle_fewer_pct']:.1f} % fewer)")
        res["groups"][g] = e
    res["break_even"] = break_even(runs)
    print("\nDecision stride alone: passes per hop = 7501 / k, no cache, every kept decision bit-exact")
    res["detection"] = {}
    for rms in ("30", "100", "500"):
        rs = [r for r in runs if r[0]["stream"].startswith("on") and r[0]["stream"].split(":")[4] == rms]
        if not rs:
            continue
        det = detection(rs)
        res["detection"][rms] = det
        v = next(iter(det.values()))
        print(f"noise RMS {rms}: {v['keyword_words']} keywords, {v['unknown_words']} _unknown_ words, {sum(j['seconds'] for j, _ in rs):.0f} s")
        print("   k  hold  recall  wrong   miss  unknown-alarm  noise alarms (per alignment)")
        for k, v in det.items():
            print(f"  {k:2d}  {v['hold']:4d}  {100 * v['recall']:5.1f}  {100 * v['wrong']:5.1f}  {100 * v['miss']:5.1f}  {100 * v['unknown_alarm']:13.1f}  {v['noise_alarms']:6.1f}")
    on = [r for r in runs if r[0]["stream"].startswith("on")]
    if on:
        res["paired"] = {k: paired_recall(on, k) for k in (2, 3, 4, 5, 8)}
        print("Keyword hits against every hop, the same words, the hardware's alignment (exact McNemar test)")
        for k, v in res["paired"].items():
            print(f"  k = {k}: {v['words']} words; only every hop {v['only_every_hop']}, only stride {v['only_stride']}; p = {v['p']:.3f}")
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return res


# ---------------------------------------------------------------- self-test
def self_test(n_clips=20):
    """Fixed geometry against the hand derivation; the functional model against full inference, with two
    deliberately wrong masks that must be caught (so that 0 mismatches means something)."""
    comp, dirty = static_plan(1, {0, 1, 2, 3, 4})
    ok = [len(x) for x in dirty] == [6, 8, 10, 12, 14] and static_passes(comp) == 3201
    print(f"geometry: rows {[len(x) for x in dirty]}, {static_passes(comp)} passes: {'ok' if ok else 'WRONG'}")
    full, _ = static_plan(1, set())
    ok &= static_passes(full) == FULL_PASSES and FULL_MACS == 2656768
    x, _ = make_stream(f"doc{n_clips}")
    r = run_stream(x)
    print(f"{r['hops']} hops: window maximum changed at {r['n_change']}; logit mismatches {sum(r['bad'].values())}, layer mismatches {sum(r['bad_layers'].values())}")
    ok &= sum(r["bad"].values()) == 0 and sum(r["bad_layers"].values()) == 0

    # The test must be able to fail: each repair removed in turn, in the simple mode, which trusts the
    # geometry. (With the feature comparison, one cache for both phases stays correct: nearly every
    # aligned cell then differs, so nearly everything is recomputed. And removing only the block-level
    # boundary changes nothing while conv1's rows 0 and 1 stay dirty: dilation keeps row 0 dirty.)
    def no_left_rows(s):
        return {r for r in conv1_dirty_rows(s) if 2 * r - 4 >= 0}

    def no_left_boundary(s):
        m = np.zeros((R, C), bool)
        m[R - s:] = True
        return m

    def break_padding(m):
        m.rows_fn, m.boundary_fn = no_left_rows, no_left_boundary

    def break_phase(m):
        m.phase_fn = lambda h: 0

    res = {}
    for name, fix, see_max in (("no padding repair at the start of the window", break_padding, True),
                               ("one cache for both stride phases", break_phase, True),
                               ("window-maximum change ignored", lambda m: None, False)):
        mod = ReuseModel("simple", 2)
        fix(mod)
        sm = si.StreamModel()
        wrong = cls = n = 0
        prev, changes = None, []
        for h in range(len(x) // si.HOP):
            f = sm.hop(x, h)
            if sm.lmx != prev:
                changes.append(h)
            prev = sm.lmx
            if f is None:
                continue
            y = d.forward(np.asarray(f, np.int64))
            yy, _ = mod.step(h, f, see_max and any(h - 2 < e <= h for e in changes[-4:]))
            wrong += int(not np.array_equal(yy, y))
            cls += int(np.argmax(yy) != np.argmax(y))
            n += 1
        res[name] = dict(decisions=n, logits_wrong=wrong, class_wrong=cls)
        print(f"broken on purpose, {name}: logits wrong at {wrong} of {n} decisions, class wrong at {cls}: {'caught' if wrong else 'NOT CAUGHT'}")
        ok &= wrong > 0
    print("PASS" if ok else "FAIL")
    return ok, res


def suite_specs():
    specs = ["doc20"] + [f"dense:{s}:200" for s in (1, 2, 3)]
    for rms in (30, 100, 500):
        specs += [f"on:{s * rms}:240:3.0:{rms}" for s in (11, 12, 13, 14)]
    return specs


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 0
    if a[0] == "static":
        report_static()
    elif a[0] == "self-test":
        return 0 if self_test(int(a[1]) if len(a) > 1 else 20)[0] else 1
    elif a[0] == "run":
        run_one(a[1], a[2])
    elif a[0] == "suite":
        import concurrent.futures as cf
        os.makedirs(a[1], exist_ok=True)
        specs = suite_specs()
        jobs = int(a[2]) if len(a) > 2 else max(1, min(13, (os.cpu_count() or 2) - 2))
        with cf.ProcessPoolExecutor(jobs) as ex:
            list(ex.map(run_one, specs, [os.path.join(a[1], s.replace(":", "_").replace(".", "p")) for s in specs]))
        report(a[1])
    elif a[0] == "report":
        report(a[1])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
