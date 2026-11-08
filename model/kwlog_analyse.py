#!/usr/bin/env python3
"""Analyse same-audio logger runs (firmware hopcim_kwlog, docs/SAME_AUDIO.md).

One run holds every hop's audio and every 20-ms decision's logits. Because a decision's logits do not
depend on the rate (the front-end and window path run at every hop; verified in RTL and on the
board), a rate of one decision every k hops at phase p is the subsequence of decisions at hops
49 + p + n k. So every rate, phase and confirmation rule is scored here on identical speech, and
comparisons are truly paired.

A run:
  1. integrity: checksums, hop continuity, decision count, overrun;
  2. golden model: every decision's logits and class recomputed from the recorded audio;
  3. the offline rule at k = 1 against the firmware's own reports, hop for hop;
  4. scoring with the rules of score_kwtest.py, over rates, phases and rules;
  5. the run lengths behind the 80-ms misses, against the prediction (L - 4) / 4.

Usage:
  kwlog_analyse.py <run dir> [--no-golden] [--json out.json]
  kwlog_analyse.py --pool <run dir> <run dir> ...       several speakers, paired tests over all trials
  kwlog_analyse.py --far <run dir> ...                  a run without prompts: reports per hour
  kwlog_analyse.py --self-test
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from kwlog_frames import KEYWORDS, LABEL, Parser, decode

HOP_S = 0.02
KS = (1, 2, 3, 4, 5, 8)
PRIMARY_KS = (1, 2, 4)
FIRST_DECISION_HOP = 49
GATES = (500, 1000, 2000, 4000)                      # window maximum, in samples (full scale 32,767)


# ---------------------------------------------------------------- reading a run
def load(run):
    p = Parser()
    out = dict(head=None, end=None, prompts=[], reports=[], levels=[], frames={}, env={})
    audio, dh, dc, dk, do, dl = {}, [], [], [], [], []
    with open(os.path.join(run, "frames.bin"), "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for t, pl in p.feed(chunk):
                out["frames"][t] = out["frames"].get(t, 0) + 1
                d = decode(t, pl)
                if t == "H":
                    out["head"] = d
                elif t == "A":
                    audio[d["hop"]] = np.frombuffer(d["samples"], "<i2")
                elif t == "D":
                    dh.append(d["hop"])
                    dc.append(d["cnt"])
                    dk.append(d["cls"])
                    do.append(d["ovr"])
                    dl.append(d["logits"])
                elif t == "R":
                    out["reports"].append((d["hop"], d["cls"]))
                elif t == "P":
                    out["prompts"].append(d)
                elif t == "L":
                    out["levels"].append(d)
                elif t == "M":
                    out["env"][d["hop"]] = d["max"]
                elif t == "E":
                    out["end"] = d
    out["bad_bytes"] = p.bad
    out["bad_start"] = p.bad_start
    out["trailing"] = len(p.buf)
    out["audio"] = audio
    out["hop"] = np.array(dh, np.int64)
    out["cnt"] = np.array(dc, np.int64)
    out["cls"] = np.array(dk, np.int64)
    out["ovr"] = np.array(do, np.int64)
    out["logits"] = np.array(dl, np.int64).reshape(-1, 12)
    return out


def integrity(r):
    """Everything that could make the run unusable; returns a list of problems."""
    problems = []
    if r["head"] is None:
        problems.append("no header frame")
    if r["bad_bytes"]:
        problems.append(f"{r['bad_bytes']} bytes skipped")
    h = r["hop"]
    if len(h) == 0:
        return problems + ["no decision"]
    if h[0] != FIRST_DECISION_HOP:
        problems.append(f"first decision at hop {h[0]}")
    if np.any(np.diff(h) != 1):
        problems.append(f"{int(np.sum(np.diff(h) != 1))} gaps in the decision hops")
    if np.any((r["cnt"] & 0xFFFF) != ((h - FIRST_DECISION_HOP + 1) & 0xFFFF)):
        problems.append("the decision count does not follow the hop")
    if np.any(r["ovr"]):
        problems.append("overrun flag set")
    if np.any(r["cls"] != np.argmax(r["logits"], axis=1)):
        problems.append("a class is not the first maximum of its logits")
    if r["end"] is not None and r["end"]["lost"]:
        problems.append(f"{r['end']['lost']} frames lost on the board")
    if r["head"] and r["head"]["audio"]:
        if sorted(r["audio"]) != list(range(len(r["audio"]))):
            problems.append("audio hops are not contiguous from hop 0")
    return problems


def stream(r):
    return np.concatenate([r["audio"][h] for h in range(len(r["audio"]))])


def window_max(r):
    """Per decision, the maximum sample of its 1-s window (50 hop blocks), as the hardware's window
    path sees it, from the audio or from the hop levels of a run without audio; None if neither."""
    if r["audio"]:
        bm = {h: int(v.max()) for h, v in r["audio"].items()}
    elif r["env"]:
        bm = r["env"]
    else:
        return None
    n = max(bm) + 1
    a = np.full(n, -32768, np.int64)
    for h, v in bm.items():
        a[h] = v
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(a, 50).max(axis=1)        # w[i]: blocks i .. i + 49, the decision of hop i + 49
    out = np.full(len(r["hop"]), -32768, np.int64)
    ok = (r["hop"] - 49 >= 0) & (r["hop"] - 49 < len(w))
    out[ok] = w[r["hop"][ok] - 49]
    return out


def golden(r, run=None, ded=False):
    """Recompute every decision from the recorded audio with the golden model."""
    cache = None if run is None else os.path.join(run, "golden_ded.npy" if ded else "golden.npy")
    if cache and os.path.exists(cache):
        return np.load(cache)
    import dscnn_int as d
    import stream_int as si
    x = stream(r)
    sm = si.StreamModel(ded)
    n_hops = len(r["audio"])
    out = np.zeros((max(n_hops - FIRST_DECISION_HOP, 0), 12), np.int64)
    for h in range(n_hops):
        feat = sm.hop(x, h)
        if feat is not None:
            out[h - FIRST_DECISION_HOP] = np.asarray(d.forward(feat)).astype(np.int64)
    if cache:
        np.save(cache, out)
    return out


# ---------------------------------------------------------------- confirmation rules
def margin_of(logits):
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]


def hold_default(k):
    return 1 + (3 + k) // k                          # the firmware's: spans at least 80 ms


def events_hold(cls, hops, n):
    """hopcim_live's rule: a keyword top for n decisions in a row; the same keyword again only after a
    decision that is not a keyword."""
    ev, last, run, said = [], -1, 0, -1
    for c, h in zip(cls.tolist(), hops.tolist()):
        run = run + 1 if c == last else 1
        last = c
        if c < 10 and run == n and c != said:
            said = c
            ev.append((h, c))
        if c >= 10:
            said = -1
    return ev


def events_vote(cls, hops, m, n):
    """m of the last n decisions name the same keyword (the newest among them)."""
    ev, said = [], -1
    cl = cls.tolist()
    for i, (c, h) in enumerate(zip(cl, hops.tolist())):
        if c >= 10:
            said = -1
            continue
        if c != said and cl[max(0, i - n + 1): i + 1].count(c) >= m:
            said = c
            ev.append((h, c))
    return ev


def apply_rule(rule, logits, hops, wmax=None):
    kind = rule[0]
    if kind == "gate":                               # ("gate", g, n): windows whose maximum is below g are no keyword
        if wmax is None:
            raise ValueError("a level gate needs audio or hop levels")
        cls = np.where(wmax >= rule[1], np.argmax(logits, axis=1), 11)
        return events_hold(cls, hops, rule[2])
    if kind == "hold":                               # ("hold", n)
        return events_hold(np.argmax(logits, axis=1), hops, rule[1])
    if kind == "vote":                               # ("vote", m, n)
        return events_vote(np.argmax(logits, axis=1), hops, rule[1], rule[2])
    if kind == "margin":                             # ("margin", theta, n): weak decisions count as no keyword
        cls = np.argmax(logits, axis=1)
        cls = np.where(margin_of(logits) >= rule[1], cls, 11)
        return events_hold(cls, hops, rule[2])
    if kind in ("smooth", "smargin", "sgate"):       # ("smooth", w, n): the argmax of the sum of the last w logits
        c = np.cumsum(np.vstack([np.zeros((1, 12), np.int64), logits]), axis=0)
        w = rule[1]
        s = c[1:] - c[np.maximum(np.arange(1, len(c)) - w, 0)]
        cls = np.argmax(s, axis=1)
        if kind == "smargin":                        # ("smargin", w, th, n): and a mean margin of at least th
            return events_hold(np.where(margin_of(s) >= rule[2] * w, cls, 11), hops, rule[3])
        if kind == "sgate":                          # ("sgate", w, g, n): and a window maximum of at least g
            if wmax is None:
                raise ValueError("a level gate needs audio or hop levels")
            return events_hold(np.where(wmax >= rule[2], cls, 11), hops, rule[3])
        return events_hold(cls, hops, rule[2])
    raise ValueError(rule)


def rule_name(rule):
    if rule[0] == "hold":
        return f"{rule[1]} in a row"
    if rule[0] == "vote":
        return f"{rule[1]} of last {rule[2]}"
    if rule[0] == "margin":
        return f"margin >= {rule[1]}, {rule[2]} in a row"
    if rule[0] == "gate":
        return f"level >= {rule[1]}, {rule[2]} in a row"
    if rule[0] == "smargin":
        return f"sum of {rule[1]}, margin >= {rule[2]}, {rule[3]} in a row"
    if rule[0] == "sgate":
        return f"sum of {rule[1]}, level >= {rule[2]}, {rule[3]} in a row"
    return f"sum of {rule[1]}, {rule[2]} in a row"


def subsample(n, k, p):
    return np.arange(p, n, k)


# ---------------------------------------------------------------- scoring (the rules of score_kwtest.py)
def trials_of(r):
    pr = r["prompts"]
    last_end = r["end"]["hop"] if r["end"] else (int(r["hop"][-1]) + 1 if len(r["hop"]) else 0)
    ends = [q["hop"] for q in pr[1:]] + [last_end]
    return [dict(start=q["hop"], end=e, kind=q["kind"], word=q["word"]) for q, e in zip(pr, ends)]


def score(events, trials):
    """Per trial: keyword 2 correct, 1 wrong only, 0 missed; other and quiet: 1 if something was reported."""
    res = dict(correct=0, wrong=0, missed=0, other_alarm=0, quiet_alarm=0, extra=0, n_kw=0, n_other=0, n_quiet=0)
    per_trial, i, ev = [], 0, sorted(events)
    for t in trials:
        names = []
        while i < len(ev) and ev[i][0] <= t["start"]:
            i += 1
        while i < len(ev) and ev[i][0] <= t["end"]:
            names.append(LABEL[ev[i][1]])
            i += 1
        if t["kind"] == "keyword":
            res["n_kw"] += 1
            if t["word"] in names:
                res["correct"] += 1
                res["extra"] += len(names) - 1
                per_trial.append(2)
            else:
                res["wrong" if names else "missed"] += 1
                res["extra"] += len(names)
                per_trial.append(1 if names else 0)
        else:
            res["n_other" if t["kind"] == "other" else "n_quiet"] += 1
            if names:
                res["other_alarm" if t["kind"] == "other" else "quiet_alarm"] += 1
            res["extra"] += len(names)
            per_trial.append(1 if names else 0)
    res["per_trial"] = per_trial
    return res


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(x, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def poisson_ci(x, t_h):
    """Exact 95 % interval for a rate, in events per hour (bisection on the Poisson tails)."""
    def cdf(k, lam):
        term, s = math.exp(-lam), 0.0
        for i in range(k + 1):
            s += term
            term *= lam / (i + 1)
        return s

    def bisect(f, lo, hi):
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return lo

    hi_lam = max(10.0, 4.0 * (x + 5))
    up = bisect(lambda lam: cdf(x, lam) > 0.025, 0.0, hi_lam)
    lo = 0.0 if x == 0 else bisect(lambda lam: 1 - cdf(x - 1, lam) < 0.025, 0.0, hi_lam)
    return lo / t_h, up / t_h


def trial_correct(per_trial, trials):
    return [v == 2 for v, t in zip(per_trial, trials) if t["kind"] == "keyword"]


def config_scores(logits, hops, trials, rule, k, wmax=None):
    """Scores of one rule at rate k, one per phase."""
    out = []
    for p in range(k):
        idx = subsample(len(hops), k, p)
        out.append(score(apply_rule(rule, logits[idx], hops[idx], None if wmax is None else wmax[idx]), trials))
    return out


def paired(base, other, trials):
    a = trial_correct(base["per_trial"], trials)
    b = trial_correct(other["per_trial"], trials)
    only_base = sum(x and not y for x, y in zip(a, b))
    only_other = sum(y and not x for x, y in zip(a, b))
    return only_base, only_other, mcnemar(only_base, only_other)


# ---------------------------------------------------------------- run lengths
def longest_runs(cls, hops, trials):
    """For every keyword trial, the longest run of the prompted class at 20 ms (0 if none)."""
    res = []
    for t in trials:
        if t["kind"] != "keyword":
            continue
        want = KEYWORDS.index(t["word"])
        m = (hops > t["start"]) & (hops <= t["end"])
        best = run = 0
        for c in cls[m].tolist():
            run = run + 1 if c == want else 0
            best = max(best, run)
        res.append(best)
    return res


def predicted_catch(L, k=4, hold=2):
    """Share of phases at which a run of L hops holds `hold` decisions k hops apart."""
    span = (hold - 1) * k + 1                        # hops from the first to the last needed decision
    return min(max(L - span + 1, 0), k) / k


def run_length_check(cls, hops, trials, k=4):
    """Prediction against the phase sweep, for the rule of two decisions k hops apart."""
    rows = {}
    for t, L in zip([t for t in trials if t["kind"] == "keyword"], longest_runs(cls, hops, trials)):
        want = KEYWORDS.index(t["word"])
        caught = 0
        for p in range(k):
            idx = subsample(len(hops), k, p)
            ev = events_hold(cls[idx], hops[idx], hold_default(k))
            caught += any(e[1] == want and t["start"] < e[0] <= t["end"] for e in ev)
        key = "0" if L == 0 else ("1-4" if L <= 4 else (str(L) if L <= 8 else "9+"))
        row = rows.setdefault(key, dict(trials=0, phases=0, caught=0, pred=0.0))
        row["trials"] += 1
        row["phases"] += k
        row["caught"] += caught
        row["pred"] += predicted_catch(L, k, hold_default(k)) * k
    return rows


# ---------------------------------------------------------------- report of one run
def sweep_rules():
    rules = [("hold", n) for n in (1, 2, 3, 4, 5, 6, 8)]
    rules += [("vote", 2, 3), ("vote", 3, 4), ("vote", 3, 5), ("vote", 4, 6)]
    rules += [("margin", th, 2) for th in (10, 20, 40, 80)]
    rules += [("margin", th, 3) for th in (10, 20, 40, 80)]
    rules += [("smooth", w, 2) for w in (2, 3, 4, 8, 12, 16)]
    rules += [("smargin", w, th, 2) for w in (4, 8, 16) for th in (5, 10, 20)]
    rules += [("gate", g, 2) for g in GATES] + [("gate", g, 3) for g in GATES] + [("gate", g, 5) for g in GATES]
    rules += [("sgate", w, g, 2) for w in (8, 16) for g in GATES]
    return rules


def fmt_scores(s, n_kw):
    cs = [x["correct"] for x in s]
    return f"{np.mean(cs):5.1f} ({min(cs)}-{max(cs)})"


def analyse(run, use_golden=True, ded=False, out=print):
    r = load(run)
    probs = integrity(r)
    hops, logits = r["hop"], r["logits"]
    cls = np.argmax(logits, axis=1)
    wmax = window_max(r)
    res = dict(run=os.path.basename(os.path.normpath(run)), frames=r["frames"], problems=probs,
               decisions=len(hops), seconds=len(hops) * HOP_S, overrun=int(r["ovr"].max()) if len(hops) else None)
    out(f"run {res['run']}: {len(hops)} decisions ({res['seconds']:.0f} s), frames {r['frames']}, "
        f"{r['bad_bytes']} bytes skipped between frames, {r['bad_start']} before the first")
    out("integrity: " + ("clean" if not probs else "; ".join(probs)))

    if use_golden and r["audio"]:
        g = golden(r, run, ded)
        n = min(len(g), len(logits))
        dif = np.any(g[:n] != logits[:n], axis=1)
        x = stream(r)
        res["golden"] = dict(checked=int(n), differ=int(dif.sum()))
        res["audio"] = dict(samples=int(len(x)), rms=float(x.std()), peak=int(np.abs(x.astype(np.int64)).max()))
        out(f"golden model: {n} decisions recomputed from the recorded audio, {int(dif.sum())} differ "
            f"(audio rms {x.std():.1f}, peak {res['audio']['peak']})")
        if dif.any():
            out(f"  first differences at decisions {np.nonzero(dif)[0][:10].tolist()}")

    offline = events_hold(cls, hops, 5)
    fw = [tuple(e) for e in r["reports"]]
    res["rule_check"] = dict(firmware=len(fw), offline=len(offline), same=offline == fw)
    out(f"rule check: firmware {len(fw)} reports, offline {len(offline)}, "
        + ("identical in hop and class" if offline == fw else "DIFFERENT"))
    hist = {LABEL[c]: int((cls == c).sum()) for c in range(12) if (cls == c).any()}
    res["classes"] = hist
    out("classes: " + ", ".join(f"{k} {v}" for k, v in hist.items()))
    if not r["prompts"]:
        return res, r

    trials = trials_of(r)
    res["trials"] = trials
    base = config_scores(logits, hops, trials, ("hold", 5), 1)[0]
    n_kw = base["n_kw"]
    res["n_kw"] = n_kw
    out(f"\ntrials: {n_kw} keywords, {base['n_other']} other words, {base['n_quiet']} quiet")
    out("\nrate and phase, the firmware's rule (hold spanning at least 80 ms); correct keywords of "
        f"{n_kw}: mean over phases (range)")
    out("  every   hold   correct            wrong  missed  other  quiet  extra   vs 20 ms: only 20 / only k, p (per phase)")
    res["rates"] = {}
    for k in KS:
        hold = hold_default(k)
        sc = config_scores(logits, hops, trials, ("hold", hold), k)
        pairs = [paired(base, s, trials) for s in sc]
        res["rates"][k] = dict(hold=hold, phases=[{a: b for a, b in s.items()} for s in sc],
                               vs_20=[list(q) for q in pairs])
        pp = ", ".join(f"{a}/{b} p={q:.3f}" for a, b, q in pairs)
        out(f"  {20 * k:3d} ms  {hold:4d}   {fmt_scores(sc, n_kw):16s}  {np.mean([s['wrong'] for s in sc]):5.1f}"
            f"  {np.mean([s['missed'] for s in sc]):6.1f}  {np.mean([s['other_alarm'] for s in sc]):5.1f}"
            f"  {np.mean([s['quiet_alarm'] for s in sc]):5.1f}  {np.mean([s['extra'] for s in sc]):5.1f}   {pp}")

    out("\nrules: mean correct over phases (range), and extra reports (mean)")
    res["rules"] = {}
    rules = sweep_rules()
    head = "".join(f"{20 * k:>7d} ms         " for k in PRIMARY_KS)
    out(f"  {'rule':26s}{head}")
    for rule in rules:
        row, row_res = f"  {rule_name(rule):26s}", {}
        for k in PRIMARY_KS:
            sc = config_scores(logits, hops, trials, rule, k, wmax)
            row_res[k] = dict(correct=[s["correct"] for s in sc], extra=[s["extra"] for s in sc],
                              other=[s["other_alarm"] for s in sc], quiet=[s["quiet_alarm"] for s in sc],
                              per_trial=[s["per_trial"] for s in sc])
            row += f"{np.mean(row_res[k]['correct']):5.1f} ({min(row_res[k]['correct'])}-{max(row_res[k]['correct'])}) x{np.mean(row_res[k]['extra']):4.1f}  "
        res["rules"][rule_name(rule)] = row_res
        out(row)

    lens = longest_runs(cls, hops, trials)
    res["run_lengths"] = lens
    out("\nlongest run of the prompted class at 20 ms, per keyword trial: "
        + ", ".join(f"{v}" for v in sorted(lens)))
    rows = run_length_check(cls, hops, trials, 4)
    res["run_length_check"] = rows
    out("80 ms, two decisions apart: share of phases that catch a word, against (L - 4) / 4")
    out("  run L   trials   observed   predicted")
    for key in ("0", "1-4", "5", "6", "7", "8", "9+"):
        if key in rows:
            v = rows[key]
            out(f"  {key:5s}  {v['trials']:6d}   {100 * v['caught'] / v['phases']:7.1f} %  {100 * v['pred'] / v['phases']:7.1f} %")
    return res, r


# ---------------------------------------------------------------- several runs
def pool(runs, out=print):
    per = []
    for run in runs:
        res, r = analyse(run, use_golden=False, out=lambda *_: None)
        per.append((res, r))
    keys = [(k, hold_default(k)) for k in PRIMARY_KS]
    out(f"{len(runs)} runs, paired over identical audio")
    table = {}
    all_trials = []
    for res, r in per:
        all_trials += res["trials"]
    for k, hold in keys:
        out(f"\n{20 * k} ms (hold {hold}), keywords correct by run, mean over phases:")
        total, n_kw = 0.0, 0
        only_b, only_o = 0, 0
        for res, r in per:
            logits, hops = r["logits"], r["hop"]
            tr = res["trials"]
            base = config_scores(logits, hops, tr, ("hold", 5), 1)[0]
            sc = config_scores(logits, hops, tr, ("hold", hold), k)
            mean = float(np.mean([s["correct"] for s in sc]))
            pairs = [paired(base, s, tr) for s in sc]
            only_b += float(np.mean([q[0] for q in pairs]))
            only_o += float(np.mean([q[1] for q in pairs]))
            total += mean
            n_kw += base["n_kw"]
            out(f"  {res['run']:36s} {mean:5.1f} of {base['n_kw']}")
        low, high = wilson(total, n_kw)
        out(f"  pooled: {total:.1f} of {n_kw} = {100 * total / max(n_kw, 1):.1f} % (95 % interval {low:.1f}-{high:.1f}, "
            "the trials of a speaker are not independent)")
        out(f"  against 20 ms: correct only at 20 ms {only_b:.1f}, only here {only_o:.1f} (mean over phases), "
            f"McNemar p = {mcnemar(round(only_b), round(only_o)):.3f} on the rounded counts")
        table[k] = dict(correct=total, n=n_kw, only_20=only_b, only_k=only_o)
    return table


def by_hour(r, hops, logits, wmax, out=print):
    """A long run, hour by hour of the run: whether reports are spread evenly or follow the room's activity.
    Each rule runs over the whole run (no edge effects) and its reports are binned by hop, mean over phases."""
    k = 2
    per = int(round(3600 / HOP_S))
    nbin = int(np.ceil((hops[-1] - hops[0] + 1) / per))
    rules = [("hold", hold_default(k)), ("smooth", 8, 2)] + ([("gate", 2000, 3)] if wmax is not None else [])
    counts = np.zeros((len(rules), nbin))
    for i, rule in enumerate(rules):
        for p in range(k):
            idx = subsample(len(hops), k, p)
            for h, _ in apply_rule(rule, logits[idx], hops[idx], None if wmax is None else wmax[idx]):
                counts[i, (h - hops[0]) // per] += 1 / k
    lv_h = np.array([v["hop"] for v in r["levels"]], np.int64)
    lv_r = np.array([v["rms"] for v in r["levels"]], np.int64)
    out(f"  by hour of the run, {20 * k} ms, reports (mean over phases): " + "; ".join(rule_name(q) for q in rules)
        + "; level rms median; window maximum 99.9 % and largest")
    for b in range(nbin):
        lo, hi = hops[0] + b * per, hops[0] + (b + 1) * per
        sel = (hops >= lo) & (hops < hi)
        dur = sel.sum() * HOP_S / 3600
        rms = lv_r[(lv_h >= lo) & (lv_h < hi)]
        line = f"    hour {b + 1} ({dur:.2f} h): " + ", ".join(f"{c:5.1f}" for c in counts[:, b])
        line += f"; rms {np.median(rms):.0f}" if len(rms) else "; rms -"
        if wmax is not None and sel.any():
            line += f"; {np.percentile(wmax[sel], 99.9):.0f}, {int(wmax[sel].max())}"
        out(line)


def far(runs, out=print):
    """A run without prompts: every report is a false alarm, if no keyword was said."""
    base_rules = [("hold", 5), ("hold", 3), ("hold", 2), ("margin", 10, 3), ("margin", 20, 3), ("margin", 40, 3),
                  ("smooth", 8, 2), ("smooth", 16, 2), ("smargin", 16, 5, 2)]
    table, pooled = {}, {}
    for run in runs:
        r = load(run)
        probs = integrity(r)
        hops, logits = r["hop"], r["logits"]
        hours = len(hops) * HOP_S / 3600
        out(f"run {os.path.basename(os.path.normpath(run))}: {hours:.2f} h, {len(hops)} decisions; integrity: "
            + ("clean" if not probs else "; ".join(probs)))
        if r["levels"]:
            rms = [v["rms"] for v in r["levels"]]
            out(f"  level, every 10 s: rms median {np.median(rms):.0f}, range {min(rms)}-{max(rms)}")
        cls = np.argmax(logits, axis=1)
        wmax = window_max(r)
        rules = list(base_rules)
        if wmax is not None:
            rules += [("gate", g, 2) for g in GATES] + [("gate", g, 3) for g in GATES]
            rules += [("sgate", w, g, 2) for w in (8, 16) for g in GATES]
            pct = np.percentile(wmax, [5, 50, 95, 99.9]).round().astype(int).tolist()
            out(f"  window maximum: 5 % {pct[0]}, median {pct[1]}, 95 % {pct[2]}, 99.9 % {pct[3]}")
        out("  classes: " + ", ".join(f"{LABEL[c]} {int((cls == c).sum())}" for c in range(12) if (cls == c).any()))
        for k in PRIMARY_KS:
            for rule in ([("hold", hold_default(k))] + [q for q in rules if q != ("hold", hold_default(k))]):
                counts = [len(apply_rule(rule, logits[idx], hops[idx], None if wmax is None else wmax[idx]))
                          for idx in (subsample(len(hops), k, p) for p in range(k))]
                lo, hi = poisson_ci(round(float(np.mean(counts))), hours) if hours else (0, 0)
                out(f"  {20 * k:3d} ms  {rule_name(rule):26s} reports {float(np.mean(counts)):7.1f} (phases {min(counts)}-{max(counts)}), "
                    f"{float(np.mean(counts)) / max(hours, 1e-9):8.1f} per hour (95 % {lo:.1f}-{hi:.1f})")
                table[(os.path.basename(os.path.normpath(run)), k, rule_name(rule))] = float(np.mean(counts))
                pooled.setdefault((k, rule_name(rule)), []).append((float(np.mean(counts)), hours))
        if hours >= 2:
            by_hour(r, hops, logits, wmax, out)
    if len(runs) > 1:
        total_h = sum(v[1] for v in next(iter(pooled.values())))
        out(f"\nall runs together, {total_h:.2f} h, for the rules available in every run:")
        for (k, name), v in pooled.items():
            if len(v) == len(runs):
                c = sum(x[0] for x in v)
                lo, hi = poisson_ci(round(c), total_h)
                out(f"  {20 * k:3d} ms  {name:38s} reports {c:7.1f}, {c / total_h:7.1f} per hour (95 % {lo:.1f}-{hi:.1f})")
    return table


# ---------------------------------------------------------------- self-test
def self_test():
    """The scorer against hand-made logs, and deliberately broken variants (they must disagree)."""
    errors = []

    def check(name, got, want):
        if got != want:
            errors.append(f"{name}: {got} != {want}")

    def logits_of(seq):
        lg = np.full((len(seq), 12), -100, np.int64)
        lg[np.arange(len(seq)), seq] = 100
        return lg

    # the firmware's rule: exactly at n in a row; same keyword again only after a non-keyword
    h = np.arange(100, 130)
    s = [11] * 3 + [9] * 5 + [9] * 2 + [3] * 5 + [9] * 5 + [11] + [9] * 4 + [9] + [11] * 4
    s = s[:30]
    ev = events_hold(np.array(s), h, 5)
    check("hold", ev, [(107, 9), (114, 3), (119, 9), (125, 9)])
    check("hold 6", events_hold(np.array(s), h, 6), [(108, 9)])
    check("hold on a non-keyword", events_hold(np.array([11] * 20), np.arange(20), 5), [])
    # voting, margin and smoothing
    check("vote", events_vote(np.array([9, 11, 9, 9, 11, 3, 3]), np.arange(7), 2, 3), [(2, 9), (6, 3)])
    lg = logits_of([9] * 6)
    lg[2, 9] = -90
    lg[2, 3] = -80                                   # one weak, different decision
    check("margin splits a run", apply_rule(("margin", 50, 3), lg, np.arange(6)), [(5, 9)])
    check("smooth bridges it", apply_rule(("smooth", 3, 3), lg, np.arange(6)), [(2, 9)])
    # a mean margin over 2 decisions: margins 200, then 10 and 200 sum to 210 over 2, a mean of 105
    # (the first windows reach a mean of 100 or 105, and only windows without the weak decision reach 200)
    lg3 = logits_of([9] * 5)
    lg3[1, 9] = -90
    lg3[1, 3] = -100
    check("smargin passes", apply_rule(("smargin", 2, 100, 2), lg3, np.arange(5)), [(1, 9)])
    check("smargin blocks", apply_rule(("smargin", 2, 110, 2), lg3, np.arange(5)), [(4, 9)])
    # a level gate on a smoothed decision
    check("sgate below", apply_rule(("sgate", 2, 500, 2), lg3, np.arange(5), np.array([100] * 5)), [])
    check("sgate opens", apply_rule(("sgate", 2, 500, 2), lg3, np.arange(5), np.array([100, 100, 100, 500, 500])), [(4, 9)])
    lg2 = logits_of([9] * 6)
    check("gate below", apply_rule(("gate", 500, 3), lg2, np.arange(6), np.array([100] * 6)), [])
    check("gate opens", apply_rule(("gate", 500, 3), lg2, np.arange(6), np.array([100, 100, 500, 600, 700, 800])), [(4, 9)])
    # window maximum: 50 blocks, ending at the decision's hop
    fake = dict(audio={}, env={h: (h if h != 60 else 5000) for h in range(120)}, hop=np.arange(49, 120))
    wm = window_max(fake)
    check("window max", (int(wm[0]), int(wm[11]), int(wm[60]), int(wm[61])), (49, 5000, 5000, 110))
    # subsampling
    check("phases", [subsample(10, 4, p).tolist() for p in range(4)],
          [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]])
    # scoring: a trial is (start, end]
    trials = [dict(start=10, end=20, kind="keyword", word="yes"), dict(start=20, end=30, kind="keyword", word="no"),
              dict(start=30, end=40, kind="keyword", word="up"), dict(start=40, end=50, kind="other", word="cat"),
              dict(start=50, end=60, kind="quiet", word="(quiet)")]
    ev = [(5, 9), (10, 9), (11, 9), (15, 3), (20, 3), (21, 9), (41, 0), (60, 2)]
    sc = score(ev, trials)
    check("score", {k: sc[k] for k in ("correct", "wrong", "missed", "other_alarm", "quiet_alarm", "extra")},
          dict(correct=1, wrong=1, missed=1, other_alarm=1, quiet_alarm=1, extra=5))
    # (5, 9), (10, 9) precede the first trial; (20, 3) is in trial 0; (21, 9) makes trial 1 wrong
    check("per trial", sc["per_trial"], [2, 1, 0, 1, 1])
    # statistics, against known values
    check("mcnemar 8/2", round(mcnemar(8, 2), 3), 0.109)
    check("mcnemar 3/2", round(mcnemar(3, 2), 3), 1.0)
    check("wilson", tuple(round(v, 1) for v in wilson(48, 50)), (86.5, 98.9))
    lo, hi = poisson_ci(0, 1.0)
    check("poisson 0", (round(lo, 2), round(hi, 2)), (0.0, 3.69))
    lo, hi = poisson_ci(10, 1.0)
    check("poisson 10", (round(lo, 2), round(hi, 2)), (4.8, 18.39))
    # run-length prediction: 5..8 hops give 1/4 .. 4/4 at 80 ms, shorter none, and the phase sweep agrees
    check("prediction", [predicted_catch(L) for L in (4, 5, 6, 7, 8, 9)], [0, .25, .5, .75, 1, 1])
    for L in range(1, 12):
        c = np.array([11] * 20 + [9] * L + [11] * 20)
        h2 = np.arange(len(c))
        caught = sum(bool(events_hold(c[subsample(len(c), 4, p)], h2[subsample(len(c), 4, p)], 2)) for p in range(4))
        check(f"phase sweep L = {L}", caught / 4, predicted_catch(L))

    # a frame stream, with garbage and a corrupted frame between valid ones
    from kwlog_frames import fletcher
    def frame(t, pl):
        body = bytes([ord(t)]) + pl
        return bytes([0xA5]) + body + bytes(fletcher(body))
    good = frame("R", bytes([1, 0, 0, 0, 9]))
    bad = bytearray(good)
    bad[3] ^= 1
    p = Parser()
    stream_bytes = b"\x00\xa5" + good + bytes(bad) + good
    got = []
    for i in range(0, len(stream_bytes), 3):         # in small pieces, as a UART delivers them
        got += list(p.feed(stream_bytes[i: i + 3]))
    check("frames", len(got), 2)
    check("frames skipped", p.bad > 0 and p.bad_start == 2, True)

    # broken variants must change the result
    def broken_said(cls, hops, n):                   # never re-arms
        ev, last, run, said = [], -1, 0, -1
        for c, x in zip(cls.tolist(), hops.tolist()):
            run = run + 1 if c == last else 1
            last = c
            if c < 10 and run == n and c != said:
                said = c
                ev.append((x, c))
        return ev
    if broken_said(np.array(s), h, 5) == events_hold(np.array(s), h, 5):
        errors.append("the re-arm test does not catch a rule that never re-arms")
    if score([(t, c) for t, c in [(11, 9)]], trials)["correct"] != 1:
        errors.append("a report just after the prompt must count")
    if score([(10, 9)], trials)["correct"] != 0:
        errors.append("a report on the prompt's own hop belongs to the previous trial")
    print("self-test: " + ("PASS" if not errors else "FAIL"))
    for e in errors:
        print("  " + e)
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--ded", action="store_true", help="the bitstream uses the B3 front-end")
    ap.add_argument("--json")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--far", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.runs:
        ap.error("no run directory")
    if a.pool:
        res = pool(a.runs)
    elif a.far:
        far(a.runs)
        return 0
    else:
        res, _ = analyse(a.runs[0], use_golden=not a.no_golden, ded=a.ded)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
