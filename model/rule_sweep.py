#!/usr/bin/env python3
"""Rate, phase and confirmation rule on GSC streams (docs/SAME_AUDIO.md section 5): many speakers, with no board.

Every stream's decisions come from the golden model, which equals the hardware's bit for bit, so a rate of
one decision every k hops at phase p is again a subsequence of the 20-ms decisions. Three kinds of stream, at
three noise levels (GSC _silence_ clips scaled to one RMS, about the board's room level and above):
  words    sparse words, 80 % keywords and 20 % _unknown_, for recall (exact_reuse.stream_always_on)
  speech   only _unknown_ words: false alarms per hour on speech that holds no keyword
  noise    only the noise bed: false alarms per hour on no speech at all
These streams are synthetic: each word is one GSC clip added to noise. They choose a rule without touching
the live recordings, which then test it (kwlog_analyse.py); they do not replace them.

Usage: rule_sweep.py run [workers]      decisions into model/rule_sweep/ (about 30 core-minutes), then:
       rule_sweep.py report             tables, and model/rule_sweep_results.json
"""
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np

import dscnn_int as d
import exact_reuse as er
import kwlog_analyse as ka
import mfcc_int as mi
import stream_int as si

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rule_sweep")
RMS = (30, 100, 300)
SPECS = [("words", s, 600, r) for r in RMS for s in (11, 12)] + [("speech", 21, 600, r) for r in RMS] + [("noise", 31, 1200, r) for r in RMS]


def noise_bed(rng, n, rms):
    sil = sorted(glob.glob(os.path.join(er.TEST, "_silence_", "*.wav")))
    parts, have = [], 0
    while have < n:
        a = er.read_wav(sil[rng.integers(len(sil))]).astype(np.float64)
        r = np.sqrt(np.mean(a * a))
        parts.append(a * (rms / r) if r > 0 else a)
        have += len(a)
    return np.concatenate(parts)[:n]


def make(kind, seed, seconds, rms):
    if kind == "words":
        return er.stream_always_on(seed + rms, seconds, 4.0, float(rms))
    rng = np.random.default_rng(seed + rms)
    n = seconds * mi.NS
    x = noise_bed(rng, n, rms)
    man = []
    if kind == "speech":
        fs = sorted(glob.glob(os.path.join(er.TEST, "_unknown_", "*.wav")))
        t = int(rng.integers(mi.NS, 2 * mi.NS))
        while t + mi.NS < n - mi.NS:
            p = fs[rng.integers(len(fs))]
            x[t:t + mi.NS] += er.read_wav(p)
            man.append((t, "_unknown_", os.path.basename(p)))
            t += int(2.0 * mi.NS * rng.uniform(0.8, 1.2))
    return np.clip(np.round(x), -32768, 32767).astype(np.int64), man


def run_one(spec):
    kind, seed, seconds, rms = spec
    name = f"{kind}_s{seed}_r{rms}"
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        return name
    x, man = make(kind, seed, seconds, rms)
    sm = si.StreamModel()
    n_hops = len(x) // si.HOP
    logits = np.zeros((n_hops - 49, 12), np.int8)
    bm = np.array([x[si.HOP * h: si.HOP * (h + 1)].max() for h in range(n_hops)], np.int64)
    for h in range(n_hops):
        feat = sm.hop(x, h)
        if feat is not None:
            logits[h - 49] = np.asarray(d.forward(feat)).astype(np.int8)
    from numpy.lib.stride_tricks import sliding_window_view
    wmax = sliding_window_view(bm, 50).max(axis=1).astype(np.int32)
    np.savez_compressed(path, logits=logits, wmax=wmax, man_t=np.array([m[0] for m in man], np.int64),
                        man_lab=np.array([m[1] for m in man]), seconds=seconds)
    print("done", name, flush=True)
    return name


def rules():
    out = [("hold", n) for n in (1, 2, 3, 4, 5, 6, 8)]
    out += [("vote", 2, 3), ("vote", 3, 4), ("vote", 3, 5), ("vote", 4, 6)]
    for n in (2, 3, 5):
        out += [("margin", th, n) for th in (10, 20, 40, 80)]
    out += [("smooth", w, n) for w in (2, 3, 4, 8, 12, 16) for n in (2, 3)]
    out += [("smargin", w, th, n) for w in (4, 8, 16) for th in (5, 10, 20) for n in (2, 3)]
    out += [("gate", g, n) for g in ka.GATES for n in (2, 3, 5)]
    out += [("sgate", w, g, 2) for w in (8, 16) for g in ka.GATES]
    return out


def paired_words(rule_a, k_a, rule_b, k_b, runs):
    """Exact McNemar over identical keywords, A at phase 0 against B at each of its phases."""
    hits = []
    for p_b in range(k_b):
        only_a = only_b = n = 0
        for z in runs["words"]:
            logits, wmax = z["logits"].astype(np.int64), z["wmax"]
            hops = np.arange(len(logits)) + 49
            ia = ka.subsample(len(hops), k_a, 0)
            ib = ka.subsample(len(hops), k_b, p_b)
            ea = ka.apply_rule(rule_a, logits[ia], hops[ia], wmax[ia])
            eb = ka.apply_rule(rule_b, logits[ib], hops[ib], wmax[ib])
            for t, lab in zip(z["man_t"], z["man_lab"]):
                if lab == "_unknown_":
                    continue
                a, b, li = t // si.HOP, (t + mi.NS) // si.HOP + 50, er.LABELS.index(lab)
                ha = any(a <= h <= b and c == li for h, c in ea)
                hb = any(a <= h <= b and c == li for h, c in eb)
                only_a += int(ha and not hb)
                only_b += int(hb and not ha)
                n += 1
        hits.append(dict(words=n, only_a=only_a, only_b=only_b, p=ka.mcnemar(only_a, only_b)))
    return hits


def pareto(table, k):
    """Rules that no other rule beats on recall, speech alarms and noise alarms at once."""
    rows = [(n, v[k]) for n, v in table.items() if n != "by_noise" and k in v]
    keep = []
    for n, e in rows:
        a = (e["words"]["recall"], -e["speech"]["per_hour"], -e["noise"]["per_hour"])
        dominated = False
        for m, f in rows:
            b = (f["words"]["recall"], -f["speech"]["per_hour"], -f["noise"]["per_hour"])
            if m != n and all(x >= y for x, y in zip(b, a)) and any(x > y for x, y in zip(b, a)):
                dominated = True
                break
        if not dominated:
            keep.append((n, e))
    return sorted(keep, key=lambda r: -r[1]["words"]["recall"])


def evaluate(rule, k, runs):
    """Mean over phases: keyword recall, wrong-word rate, and reports per hour in each kind of stream."""
    res = {}
    for kind in ("words", "speech", "noise"):
        hit = wrong = n_kw = extra = 0
        secs = 0.0
        phase_recall = []
        for z in runs[kind]:
            logits, wmax = z["logits"], z["wmax"]
            hops = np.arange(len(logits)) + 49
            words = [(t // si.HOP, (t + mi.NS) // si.HOP + 50, lab) for t, lab in zip(z["man_t"], z["man_lab"])]
            for p in range(k):
                idx = ka.subsample(len(hops), k, p)
                ev = ka.apply_rule(rule, logits[idx].astype(np.int64), hops[idx], wmax[idx])
                used = set()
                ph, pn = 0, 0
                for a, b, lab in words:
                    inside = [e for e in ev if a <= e[0] <= b]
                    used.update(inside)
                    if lab != "_unknown_":
                        pn += 1
                        li = er.LABELS.index(lab)
                        if any(c == li for _, c in inside):
                            ph += 1
                        elif inside:
                            wrong += 1
                hit += ph
                n_kw += pn
                extra += sum(1 for e in ev if e not in used) if kind == "words" else len(ev)
                secs += float(z["seconds"])
                if pn:
                    phase_recall.append(ph / pn)
        res[kind] = dict(recall=hit / max(n_kw, 1), wrong=wrong / max(n_kw, 1), per_hour=extra / max(secs, 1e-9) * 3600, n_words=n_kw // k,
                         recall_min=min(phase_recall) if phase_recall else None, recall_max=max(phase_recall) if phase_recall else None)
    return res


def report():
    runs = {"words": [], "speech": [], "noise": []}
    for f in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        z = dict(np.load(f))
        z["name"] = os.path.basename(f)[:-4]
        z["rms"] = int(z["name"].split("_r")[1])
        runs[z["name"].split("_")[0]].append(z)
    hours = {k: sum(float(z["seconds"]) for z in v) / 3600 for k, v in runs.items()}
    print(f"streams: {hours['words']:.2f} h words, {hours['speech']:.2f} h non-keyword speech, {hours['noise']:.2f} h noise; "
          f"{sum(len(z['man_t']) for z in runs['words'])} words")
    table = {}
    print("\nmean over phases; recall of keywords; reports per hour: stream with words (outside any word), non-keyword speech, noise only")
    print(f"{'rule':38s}" + "".join(f"{str(20 * k) + ' ms: recall  in-words speech  noise':>44s}" for k in ka.PRIMARY_KS))
    for rule in rules():
        name = ka.rule_name(rule)
        row = f"{name:38s}"
        table[name] = {}
        for k in ka.PRIMARY_KS:
            e = evaluate(rule, k, runs)
            table[name][k] = e
            row += f"{100 * e['words']['recall']:18.1f} {e['words']['per_hour']:8.0f} {e['speech']['per_hour']:7.0f} {e['noise']['per_hour']:7.0f}  "
        print(row, flush=True)
    # by noise level, the firmware's rule against the margin rules
    print("\nby noise level, reports per hour on noise only, and recall, at 40 ms:")
    for r in RMS:
        sub = {kind: [z for z in v if z["rms"] == r] for kind, v in runs.items()}
        for rule in (("hold", 3), ("margin", 20, 3), ("margin", 40, 3), ("margin", 40, 2)):
            e = evaluate(rule, 2, sub)
            print(f"  noise RMS {r:3d}  {ka.rule_name(rule):26s} recall {100 * e['words']['recall']:5.1f} %, wrong {100 * e['words']['wrong']:4.1f} %, "
                  f"speech {e['speech']['per_hour']:6.0f}/h, noise {e['noise']['per_hour']:6.0f}/h")
            table.setdefault("by_noise", {}).setdefault(str(r), {})[ka.rule_name(rule)] = e
    print("\nrate on identical words, at the firmware's rule, against 20 ms (only 20 ms / only this rate, exact McNemar p, per phase):")
    table["paired"] = {}
    for k in (2, 4):
        ph = paired_words(("hold", 5), 1, ("hold", ka.hold_default(k)), k, runs)
        table["paired"][str(k)] = ph
        print(f"  {20 * k} ms: " + ", ".join(f"{q['only_a']}/{q['only_b']} p={q['p']:.3f}" for q in ph) + f" over {ph[0]['words']} keywords")
    print("\nthe best rules at each rate (no other rule is better on recall, speech and noise at once), recall at least 75 %:")
    for k in ka.PRIMARY_KS:
        print(f"  {20 * k} ms:")
        for n, e in pareto(table, k):
            if e["words"]["recall"] >= 0.75:
                print(f"    {n:38s} recall {100 * e['words']['recall']:5.1f} %, wrong {100 * e['words']['wrong']:4.1f} %, speech {e['speech']['per_hour']:5.0f}/h, noise {e['noise']['per_hour']:4.1f}/h")
    with open(os.path.join(HERE, "rule_sweep_results.json"), "w") as f:
        json.dump(dict(hours=hours, table=table), f, indent=1, default=float)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "run":
        os.makedirs(OUT, exist_ok=True)
        with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
            for _ in pool.imap_unordered(run_one, SPECS):
                pass
    elif sys.argv[1] == "report":
        report()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
