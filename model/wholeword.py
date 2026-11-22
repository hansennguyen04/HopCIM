#!/usr/bin/env python3
"""Whole-word decisions: the study of docs/WHOLEWORD_GATES.md (frozen at commit 8e12400).

The window path keeps the 50 hop maxima of the 1-s window for MLPerf Tiny's peak normalisation. A window is "whole"
when its first and last 3 hop maxima are at most 11/32 of the window maximum. Candidates:
  A  one decision per run of whole windows (summed logits), the network every hop;
  B  the network only on whole windows (the first of a run, then every 8th, at most 4), one decision per run;
  U  the existing k-hop field at k = 16, the summed logits of the last 2 decisions.
They are scored against the firmware rule (F20 / F40 / F80) and every firmware-only option (rules x rates the existing
hardware allows), on fresh synthetic streams (rule_sweep generators, new seeds) and on the 8.38-h ambient room run.
Every decision is the golden model's (= RTL = board). Power [E] comes from energy_est.py (ww_parts / ww_power).

Usage: wholeword.py run [workers]      the test streams into model/wholeword/ (about 50 core-minutes)
       wholeword.py report [workers]   V1-V4, the gates, model/wholeword_results.json; then energy_est.py wholeword
       wholeword.py spoken <run dir> [<run dir> ...]
                                       a spoken run of hopcim_kwlog with prompts (planenest/hw/board_kwlog.sh): every
                                       design on the same speech, per trial as score_kwtest.py, paired against F40
"""
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

import dscnn_int as d
import energy_est as ee
import exact_reuse as er
import kwlog_analyse as ka
import mfcc_int as mi
import rule_sweep as rs
import stream_int as si

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wholeword")
PILOT = os.path.join(HERE, "rule_sweep")
ROOM = [os.path.normpath(os.path.join(HERE, "..", "..", "hopcim_recordings", n))
        for n in ("far_ambient_20261107", "far_ambient_20261107b")]
SPECS = ([("words", s, 600, r) for r in rs.RMS for s in (501, 502, 503, 504)]
         + [("speech", s, 600, r) for r in rs.RMS for s in (505, 506)]
         + [("noise", 507, 1200, r) for r in rs.RMS])
NUM, DEN, M = 11, 32, 3                              # whole: 32 max(edge of 3 hops) <= 11 W, W > 0
B_STEP, B_CAP = 8, 4
U = (16, ("smooth", 2, 1))
REFS = {"F20": (1, ("hold", 5)), "F40": (2, ("hold", 3)), "F80": (4, ("hold", 2))}
KINDS = ("words", "speech", "noise")
P3 = ([("hold", n) for n in (1, 2, 3)] + [("smooth", w, n) for w in (2, 3, 4) for n in (1, 2)]
      + [("smargin", w, th, n) for w in (2, 4) for th in (5, 10) for n in (1, 2)] + [("gate", g, n) for g in (1000, 2000) for n in (1, 2)])
GATE_DOC = os.path.join(HERE, "..", "docs", "WHOLEWORD_GATES.md")


def family():
    """Firmware-only options: every rule_sweep rule at k = 1, 2, 4 and the third pilot's rules at k = 1 .. 16."""
    out = [(r, k) for r in rs.rules() for k in (1, 2, 4)] + [(r, k) for r in P3 for k in (1, 2, 4, 8, 12, 16)]
    seen, uniq = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq


# ---------------------------------------------------------------- the test streams
def used_seeds():
    """Generator seeds of the pilot streams (rule_sweep) and of exact_reuse's streams."""
    s = {seed + r for _, seed, _, r in rs.SPECS}
    for sp in er.suite_specs():
        k = sp.split(":")
        if k[0] == "on":
            s.add(int(k[1]))
    return s


def run_one(spec):
    kind, seed, seconds, rms = spec
    name = f"{kind}_s{seed}_r{rms}"
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        return name
    x, man = rs.make(kind, seed, seconds, rms)
    sm = si.StreamModel()
    n_hops = len(x) // si.HOP
    logits = np.zeros((n_hops - 49, 12), np.int8)
    for h in range(n_hops):
        feat = sm.hop(x, h)
        if feat is not None:
            logits[h - 49] = np.asarray(d.forward(feat)).astype(np.int8)
    bm = x[:n_hops * si.HOP].reshape(n_hops, si.HOP).max(axis=1)
    rng = np.random.default_rng(1000 * seed + rms)   # V2: 10 decisions against the per-window reference
    chk = rng.integers(49, n_hops, 10)
    bad = sum(int(not np.array_equal(np.asarray(d.forward(si.window_ref(x, int(h)))).astype(np.int8), logits[h - 49])) for h in chk)
    np.savez_compressed(path, logits=logits, bm=bm.astype(np.int32), man_t=np.array([m[0] for m in man], np.int64),
                        man_lab=np.array([m[1] for m in man]), man_file=np.array([m[2] for m in man]), seconds=seconds,
                        v2_hops=chk, v2_bad=bad)
    print("done", name, "V2 differ", bad, flush=True)
    return name


def load_dir(path):
    runs = {k: [] for k in KINDS}
    for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name.startswith("room"):
            continue
        z = dict(np.load(f))
        kind, _, r = name.split("_")
        bm = z["bm"].astype(np.int64) if "bm" in z else None
        q = dict(name=name, rms=int(r[1:]), logits=z["logits"].astype(np.int64), bm=bm,
                 wmax=swv(bm, 50).max(axis=1) if bm is not None else z["wmax"].astype(np.int64),
                 man_t=z["man_t"], man_lab=[str(v) for v in z["man_lab"]],
                 man_file=[str(v) for v in z["man_file"]] if "man_file" in z else None,
                 seconds=float(z["seconds"]), v2_bad=int(z["v2_bad"]) if "v2_bad" in z else None)
        if bm is not None:
            q["whole"] = whole(bm)
        runs[kind].append(q)
    return runs


def load_room():
    """The ambient runs: every decision's logits and every hop's maximum (no audio). Cached in model/wholeword/."""
    rooms = []
    for run in ROOM:
        name = os.path.basename(run)
        cache = os.path.join(OUT, f"room_{name}.npz")
        if not os.path.exists(cache):
            r = ka.load(run)
            probs = ka.integrity(r)
            L, hops = r["logits"], r["hop"]
            n_hops = len(L) + 49
            have = all(h in r["env"] for h in range(n_hops))
            bm = np.array([r["env"].get(h, -32768) for h in range(n_hops)], np.int64)
            offline = ka.events_hold(np.argmax(L, axis=1), hops, 5)
            fw = [tuple(e) for e in r["reports"]]
            np.savez_compressed(cache, logits=L.astype(np.int8), bm=bm.astype(np.int32), hops=hops,
                                problems=np.array(probs, dtype=str), have=have, fw_same=offline == fw, fw_n=len(fw),
                                contiguous=bool(np.all(np.diff(hops) == 1)) and int(hops[0]) == 49)
        z = dict(np.load(cache))
        bm = z["bm"].astype(np.int64)
        rooms.append(dict(name=name, logits=z["logits"].astype(np.int64), bm=bm, wmax=swv(bm, 50).max(axis=1), whole=whole(bm),
                          hours=len(z["logits"]) * 0.02 / 3600,
                          v3=dict(problems=[str(p) for p in z["problems"]], hop_max_every_hop=bool(z["have"]),
                                  decisions_contiguous=bool(z["contiguous"]), firmware_rule_same=bool(z["fw_same"]),
                                  board_reports=int(z["fw_n"]))))
    return rooms


# ---------------------------------------------------------------- the decisions
def whole(bm):
    blk = swv(bm, 50)                                 # [i, j] = bm[i + j]: the window of hop i + 49
    W = blk.max(axis=1)
    eL, eR = blk[:, :M].max(axis=1), blk[:, -M:].max(axis=1)
    return (W > 0) & (DEN * eL <= NUM * W) & (DEN * eR <= NUM * W)


def segments(mask):
    m = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(m == 1)[0].tolist(), np.nonzero(m == -1)[0].tolist()))


def ev_A(z):
    L, n = z["logits"], len(z["logits"])
    ev = []
    for a, b in segments(z["whole"]):
        if b < n:                                     # the run ended inside the stream: known at hop b + 49
            c = int(np.argmax(L[a:b].sum(axis=0)))
            if c < 10:
                ev.append((b + 49, c))
    return ev, n, n


def ev_B(z):
    L, n = z["logits"], len(z["logits"])
    ev, net = [], 0
    for a, b in segments(z["whole"]):
        idx = np.arange(a, b, B_STEP)[:B_CAP]
        net += len(idx)
        if b < n:
            c = int(np.argmax(L[idx].sum(axis=0)))
            if c < 10:
                ev.append((b + 49, c))
    return ev, net, net


def ev_rule(rule, k, p):
    def f(z):
        hops = np.arange(len(z["logits"])) + 49
        idx = ka.subsample(len(hops), k, p)
        return ka.apply_rule(rule, z["logits"][idx], hops[idx], z["wmax"][idx]), len(idx), len(idx)
    return f


def fns_of(name):
    if name == "A":
        return [ev_A]
    if name == "B":
        return [ev_B]
    if name == "U":
        return [ev_rule(U[1], U[0], p) for p in range(U[0])]
    k, rule = REFS[name]
    return [ev_rule(rule, k, p) for p in range(k)]


# ---------------------------------------------------------------- scoring (rule_sweep.evaluate, plus counts)
def score(fns, runs, exclude=None, rms=None, hits_out=False):
    res = {}
    for kind in KINDS:
        hit = wrong = n_kw = extra = net = cpu = hops = 0
        secs, lat, hits = 0.0, [], []
        for f in fns:
            for z in runs[kind]:
                if rms is not None and z["rms"] != rms:
                    continue
                ev, nn, nc = f(z)
                net, cpu, hops = net + nn, cpu + nc, hops + len(z["logits"])
                used = set()
                files = z["man_file"] or [None] * len(z["man_t"])
                for t, lab, fl in zip(z["man_t"].tolist(), z["man_lab"], files):
                    a, b = t // si.HOP, (t + mi.NS) // si.HOP + 50
                    inside = [e for e in ev if a <= e[0] <= b]
                    used.update(inside)
                    if lab == "_unknown_" or (exclude is not None and (lab, fl) in exclude):
                        continue
                    n_kw += 1
                    li = er.LABELS.index(lab)
                    good = [e for e in inside if e[1] == li]
                    hits.append(bool(good))
                    if good:
                        hit += 1
                        lat.append(good[0][0] - (int(round(t / si.HOP)) + 49))
                    elif inside:
                        wrong += 1
                extra += sum(1 for e in ev if e not in used) if kind == "words" else len(ev)
                secs += z["seconds"]
        q = dict(recall=hit / max(n_kw, 1), wrong=wrong / max(n_kw, 1), per_hour=extra / max(secs, 1e-9) * 3600,
                 keywords=n_kw // len(fns), reports=extra / len(fns), hours=secs / 3600 / len(fns),
                 net_per_hop=net / max(hops, 1), cpu_per_hop=cpu / max(hops, 1),
                 lat_med=float(np.median(lat)) if lat else None,
                 lat_p10=float(np.percentile(lat, 10)) if lat else None, lat_p90=float(np.percentile(lat, 90)) if lat else None)
        if hits_out:
            q["hits"] = hits
        res[kind] = q
    return res


def score_room(fns, rooms):
    rep = net = cpu = hops = 0
    hours = sum(r["hours"] for r in rooms)
    for f in fns:
        for r in rooms:
            ev, nn, nc = f(r)
            rep, net, cpu, hops = rep + len(ev), net + nn, cpu + nc, hops + len(r["logits"])
    c = rep / len(fns)
    lo, hi = ka.poisson_ci(int(round(c)), hours)
    return dict(reports=c, hours=hours, per_hour=c / hours, per_hour_ci=(lo, hi), net_per_hop=net / hops, cpu_per_hop=cpu / hops)


_RUNS = _ROOMS = None


def _family_one(o):
    rule, k = o
    fns = [ev_rule(rule, k, p) for p in range(k)]
    e = score(fns, _RUNS)
    return ka.rule_name(rule), k, {kd: {x: e[kd][x] for x in ("recall", "per_hour")} for kd in KINDS}


def mcnemar_hits(ha, hb):
    a = sum(1 for x, y in zip(ha, hb) if x and not y)
    b = sum(1 for x, y in zip(ha, hb) if y and not x)
    return dict(only_first=a, only_second=b, p=ka.mcnemar(a, b))


# ---------------------------------------------------------------- report
def report(workers):
    global _RUNS, _ROOMS
    res = dict(gate_doc_sha256=open(GATE_DOC + ".sha256").read().split()[0] if os.path.exists(GATE_DOC + ".sha256") else None)
    out = print
    # V1: the scorer on the pilot streams
    pilot = load_dir(PILOT)
    want = {"F20": (86.0, 430, 3), "F40": (87.0, 439, 8), "F80": (86.5, 362, 7)}
    v1 = {}
    for nm in REFS:
        e = score(fns_of(nm), pilot)
        got = (round(100 * e["words"]["recall"], 1), round(e["speech"]["per_hour"]), round(e["noise"]["per_hour"]))
        v1[nm] = dict(got=got, want=want[nm], same=got == want[nm])
    res["V1"] = v1
    out("V1 scorer on the pilot streams: " + "; ".join(f"{k} {v['got']} (report {v['want']})" for k, v in v1.items())
        + (" -> pass" if all(v["same"] for v in v1.values()) else " -> FAIL"))
    # the test streams, V2
    assert not ({s + r for _, s, _, r in SPECS} & used_seeds()), "a test seed was used before"
    runs = load_dir(OUT)
    assert sorted(z["name"] for kd in KINDS for z in runs[kd]) == sorted(f"{k}_s{s}_r{r}" for k, s, _, r in SPECS), "missing streams"
    v2 = sum(z["v2_bad"] for kd in KINDS for z in runs[kd])
    res["V2"] = dict(checked=10 * len(SPECS), differ=v2)
    out(f"V2 {10 * len(SPECS)} decisions of the test streams against the per-window reference: {v2} differ -> {'pass' if v2 == 0 else 'FAIL'}")
    # V3: the room
    rooms = load_room()
    res["V3"] = {r["name"]: r["v3"] for r in rooms}
    ok3 = all(not v["problems"] and v["hop_max_every_hop"] and v["decisions_contiguous"] and v["firmware_rule_same"] for v in res["V3"].values())
    for r in rooms:
        v = r["v3"]
        out(f"V3 room {r['name']}: {r['hours']:.2f} h, integrity {'clean' if not v['problems'] else v['problems']}, hop maximum for every "
            f"hop {v['hop_max_every_hop']}, decisions contiguous {v['decisions_contiguous']}, F20 offline = the board's "
            f"{v['board_reports']} reports {v['firmware_rule_same']}")
    out(f"V3 -> {'pass' if ok3 else 'FAIL'}")
    # V4: the pricing
    parts = ee.ww_parts()
    base = parts["baseline"]
    err = ee.ww_check(parts)
    res["V4"] = dict(max_err_uW=err, s_uW={k: ee.ww_power(base, 50 / k, 50 / k) for k in (1, 2, 4)})
    out(f"V4 S's power from the pricing: {' / '.join(f'{v:.1f}' for v in res['V4']['s_uW'].values())} uW at 20 / 40 / 80 ms, "
        f"largest error over the variants {err:.1e} uW -> {'pass' if err < 1e-6 else 'FAIL'}")

    # the designs
    names = ["F20", "F40", "F80", "A", "B", "U"]
    sc = {nm: score(fns_of(nm), runs) for nm in names}
    rm = {nm: score_room(fns_of(nm), rooms) for nm in names}
    pw = {}
    for nm in names:
        ge = ee.WW_LOGIC_GE if nm in ("A", "B") else 0
        pw[nm] = {kd: ee.ww_power(base, sc[nm][kd]["net_per_hop"] / ka.HOP_S, sc[nm][kd]["cpu_per_hop"] / ka.HOP_S, ge) for kd in KINDS}
        pw[nm]["room"] = ee.ww_power(base, rm[nm]["net_per_hop"] / ka.HOP_S, rm[nm]["cpu_per_hop"] / ka.HOP_S, ge)
    res["designs"] = {nm: dict(streams=sc[nm], room=rm[nm], power_uW=pw[nm]) for nm in names}
    res["rates"] = {nm: {**{kd: dict(net_per_hop=sc[nm][kd]["net_per_hop"], cpu_per_hop=sc[nm][kd]["cpu_per_hop"]) for kd in KINDS},
                         "room": dict(net_per_hop=rm[nm]["net_per_hop"], cpu_per_hop=rm[nm]["cpu_per_hop"])} for nm in names}
    hours = {kd: sc["F20"][kd]["hours"] for kd in KINDS}
    out(f"\ntest streams: {hours['words']:.2f} h words ({sc['F20']['words']['keywords']} keywords), {hours['speech']:.2f} h "
        f"non-keyword speech, {hours['noise']:.2f} h noise; room {rm['F20']['hours']:.2f} h")
    out(f"\n{'design':6s} {'recall':>7s} {'wrong':>6s} {'speech/h':>9s} {'noise/h':>8s} {'room/h':>7s} {'room 95 %':>12s}  "
        f"{'network runs per hop: words/speech/noise/room':46s} {'power [E] uW: words/speech/noise/room':40s} latency")
    for nm in names:
        s, r = sc[nm], rm[nm]
        nets = " / ".join("%5.1f" % (100 * s[kd]["net_per_hop"]) for kd in KINDS) + " / %5.2f %%" % (100 * r["net_per_hop"])
        pows = " / ".join("%5.1f" % pw[nm][kd] for kd in (*KINDS, "room"))
        out(f"{nm:6s} {100 * s['words']['recall']:6.1f}% {100 * s['words']['wrong']:5.1f}% {s['speech']['per_hour']:9.1f} "
            f"{s['noise']['per_hour']:8.2f} {r['per_hour']:7.2f} {r['per_hour_ci'][0]:5.2f}-{r['per_hour_ci'][1]:5.2f}  "
            f"{nets}       {pows}     {s['words']['lat_med']} [{s['words']['lat_p10']}, {s['words']['lat_p90']}]")

    # firmware-only options (G6)
    _RUNS, _ROOMS = runs, rooms
    fam = family()
    with mp.Pool(workers) as pool:
        frows = pool.map(_family_one, fam, chunksize=4)
    fam_res = [dict(rule=n_, k=k, power_uW=ee.ww_power(base, 50 / k, 50 / k), **{kd: e[kd] for kd in KINDS}) for n_, k, e in frows]
    res["firmware_only"] = fam_res

    # gates
    ref = {"A": "F20", "B": "F40", "U": "F40"}
    gates = {}
    for c in ("A", "B", "U"):
        R = ref[c]
        g = {}
        g["G1"] = dict(value=sc[c]["speech"]["per_hour"], limit=0.40 * sc[R]["speech"]["per_hour"])
        g["G2"] = dict(value=100 * sc[c]["words"]["recall"], limit=100 * sc[R]["words"]["recall"] - 4.0)
        g["G3"] = dict(value=sc[c]["noise"]["per_hour"], limit=sc[R]["noise"]["per_hour"])
        g["G4"] = dict(value=rm[c]["per_hour"], limit=rm[R]["per_hour"])
        for k_ in ("G1", "G3", "G4"):
            g[k_]["pass"] = g[k_]["value"] <= g[k_]["limit"]
        g["G2"]["pass"] = g["G2"]["value"] >= g["G2"]["limit"]
        if c in ("B", "U"):
            lim = 0.75 * ee.ww_power(base, 25, 25)
            g["G5"] = dict(value=max(pw[c].values()), limit=lim, by_set=pw[c], **{"pass": all(v <= lim for v in pw[c].values())})
        if c in ("A", "B"):
            qual = [o for o in fam_res if o["speech"]["per_hour"] <= sc[c]["speech"]["per_hour"]
                    and o["noise"]["per_hour"] <= sc[c]["noise"]["per_hour"] + 1.0 and o["power_uW"] <= pw[c]["words"] + 2.0]
            best = max(qual, key=lambda o: o["words"]["recall"]) if qual else None
            rstar = 100 * best["words"]["recall"] if best else None
            g["G6"] = dict(value=100 * sc[c]["words"]["recall"], r_star=rstar, limit=None if best is None else rstar + 2.0,
                           best=None if best is None else dict(rule=best["rule"], k=best["k"], recall=100 * best["words"]["recall"],
                                                               speech=best["speech"]["per_hour"], noise=best["noise"]["per_hour"],
                                                               power_uW=best["power_uW"]),
                           qualifying=len(qual))
            g["G6"]["pass"] = best is None or g["G6"]["value"] >= rstar + 2.0
        gates[c] = g
    res["gates"] = gates
    out("\ngates (value / limit):")
    for c, g in gates.items():
        row = []
        for k_, v in g.items():
            if k_ == "G6":
                b = v["best"]
                row.append(f"{k_} {v['value']:.1f} vs R* {v['r_star']:.1f} + 2 ({b['rule']}, k = {b['k']}: {b['speech']:.0f}/h, "
                           f"{b['noise']:.1f}/h, {b['power_uW']:.1f} uW; {v['qualifying']} qualify) {'PASS' if v['pass'] else 'fail'}"
                           if b else f"{k_} no firmware-only option qualifies PASS")
            else:
                row.append(f"{k_} {v['value']:.2f} / {v['limit']:.2f} {'PASS' if v['pass'] else 'fail'}")
        out(f"  {c} (against {ref[c]}): " + "; ".join(row))

    # reported, not gated: noise levels, held-out clips, paired tests
    res["by_noise"] = {}
    out("\nby noise RMS (recall %, speech/h, noise/h):")
    for r in rs.RMS:
        res["by_noise"][r] = {}
        line = []
        for nm in names:
            e = score(fns_of(nm), runs, rms=r)
            res["by_noise"][r][nm] = {kd: {x: e[kd][x] for x in ("recall", "per_hour")} for kd in KINDS}
            line.append(f"{nm} {100 * e['words']['recall']:.1f}/{e['speech']['per_hour']:.0f}/{e['noise']['per_hour']:.1f}")
        out(f"  {r:3d}: " + ", ".join(line))
    pilot_clips = set()
    for kind, s, sec, r in rs.SPECS:
        if kind != "noise":
            _, man = rs.make(kind, s, sec, r)
            pilot_clips.update((m[1], m[2]) for m in man)
    res["held_out"] = {}
    for nm in names:
        e = score(fns_of(nm), runs, exclude=pilot_clips)
        res["held_out"][nm] = dict(recall=e["words"]["recall"], keywords=e["words"]["keywords"])
    out("\nrecall on keywords whose clip is in no pilot stream: "
        + ", ".join(f"{nm} {100 * v['recall']:.1f} %" for nm, v in res["held_out"].items()) + f" (n {res['held_out']['A']['keywords']})")
    hit = {nm: [score([f], runs, hits_out=True)["words"]["hits"] for f in fns_of(nm)] for nm in names}
    res["paired"] = {"A vs F20": mcnemar_hits(hit["A"][0], hit["F20"][0]), "B vs F20": mcnemar_hits(hit["B"][0], hit["F20"][0])}
    for p in range(2):
        res["paired"][f"B vs F40 phase {p}"] = mcnemar_hits(hit["B"][0], hit["F40"][p])
    up = [mcnemar_hits(hit["U"][p], hit["F40"][0]) for p in range(U[0])]
    res["paired"]["U (each phase) vs F40 phase 0"] = dict(p_min=min(q["p"] for q in up), p_max=max(q["p"] for q in up),
                                                          only_U=[q["only_first"] for q in up], only_F40=[q["only_second"] for q in up])
    out("paired (exact McNemar, keywords found only by the first / only by the second): "
        + "; ".join(f"{k} {v['only_first']}/{v['only_second']} p={v['p']:.3g}" for k, v in res["paired"].items() if "p" in v)
        + f"; U vs F40: p {res['paired']['U (each phase) vs F40 phase 0']['p_min']:.3g}-{res['paired']['U (each phase) vs F40 phase 0']['p_max']:.3g} over its 16 phases")
    top = sorted(fam_res, key=lambda o: -o["words"]["recall"])
    out("\nfirmware-only options at <= 120 speech reports/h, best recall first (top 12):")
    for o in [o for o in top if o["speech"]["per_hour"] <= 120][:12]:
        out(f"  {o['rule']:36s} k = {o['k']:2d}  recall {100 * o['words']['recall']:5.1f} %, speech {o['speech']['per_hour']:5.0f}/h, "
            f"noise {o['noise']['per_hour']:4.1f}/h, {o['power_uW']:.1f} uW")
    with open(os.path.join(HERE, "wholeword_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


# ---------------------------------------------------------------- a spoken run (hopcim_kwlog with prompts)
DESIGNS = ("F20", "F40", "F80", "A", "B", "U")


def spoken_score(r):
    """One prompted run as kwlog_analyse.load returns it: every design scored per trial as score_kwtest.py.
    Returns the trials and, per design, one score per phase."""
    trials = ka.trials_of(r)
    L = r["logits"]
    n_hops = len(L) + 49
    if r["audio"]:
        bm = np.array([int(r["audio"][h].max()) if h in r["audio"] else -32768 for h in range(n_hops)], np.int64)
    else:
        bm = np.array([r["env"].get(h, -32768) for h in range(n_hops)], np.int64)
    z = dict(logits=L, bm=bm, wmax=swv(bm, 50).max(axis=1), whole=whole(bm))
    return trials, {nm: [ka.score(f(z)[0], trials) for f in fns_of(nm)] for nm in DESIGNS}, int((bm == -32768).sum())


def spoken(run_dirs, out=print):
    """Score F20/F40/F80/A/B/U on the same spoken prompts; exact McNemar of each against F40 over matched trials."""
    per_run, all_trials = [], []
    for run in run_dirs:
        r = ka.load(run)
        probs = ka.integrity(r)
        trials, sc, missing = spoken_score(r)
        offline = ka.events_hold(np.argmax(r["logits"], axis=1), r["hop"], 5)
        same = offline == [tuple(e) for e in r["reports"]]
        n_kw = sum(t["kind"] == "keyword" for t in trials)
        out(f"run {os.path.basename(os.path.normpath(run))}: {len(r['hop']) * 0.02 / 60:.1f} min, {len(trials)} prompts ({n_kw} keywords), "
            f"audio {'yes' if r['audio'] else 'no'}; integrity {'clean' if not probs else '; '.join(probs)}; hops without a level {missing}; "
            f"firmware rule offline = the board's {len(r['reports'])} reports: {same}")
        if not trials:
            sys.exit("no prompts in this run: use a build with KW_TEST > 0 (hopcim_kwlog_noaudio_r5_nexys.elf)")
        per_run.append(sc)
        all_trials += trials
    keys = ("correct", "wrong", "missed", "other_alarm", "quiet_alarm", "n_kw", "n_other", "n_quiet")
    tot = {}                                          # per design, one pooled score per phase
    for nm in DESIGNS:
        tot[nm] = []
        for p in range(len(per_run[0][nm])):
            agg = {x: sum(sc[nm][p][x] for sc in per_run) for x in keys}
            agg["per_trial"] = sum((sc[nm][p]["per_trial"] for sc in per_run), [])
            tot[nm].append(agg)
    res = {}
    out(f"\n{'design':6s} {'keywords correct':>22s} {'phases':>9s} {'wrong':>6s} {'missed':>7s} {'other-word alarms':>18s} "
        f"{'quiet alarms':>13s}  paired with F40 (keyword trials only F40 / only this; exact p; range over phase pairs)")
    for nm in DESIGNS:
        t0 = tot[nm][0]
        mean = {x: float(np.mean([t[x] for t in tot[nm]])) for x in keys[:5]}
        cor = [t["correct"] for t in tot[nm]]
        pv = []
        if nm != "F40":
            for t in tot[nm]:
                for f in tot["F40"]:
                    a = ka.trial_correct(f["per_trial"], all_trials)
                    b = ka.trial_correct(t["per_trial"], all_trials)
                    only_f40 = sum(x and not y for x, y in zip(a, b))
                    only_d = sum(y and not x for x, y in zip(a, b))
                    pv.append((only_f40, only_d, ka.mcnemar(only_f40, only_d)))
        res[nm] = dict(n_kw=t0["n_kw"], n_other=t0["n_other"], n_quiet=t0["n_quiet"], mean=mean, correct_by_phase=cor,
                       paired_vs_F40=[dict(only_F40=a, only_design=b, p=p) for a, b, p in pv])
        cmp_ = "" if not pv else (f"{min(v[0] for v in pv)}-{max(v[0] for v in pv)} / {min(v[1] for v in pv)}-{max(v[1] for v in pv)}; "
                                  f"p {min(v[2] for v in pv):.3g}-{max(v[2] for v in pv):.3g}")
        out(f"{nm:6s} {mean['correct']:8.1f} of {t0['n_kw']:3d} ({100 * mean['correct'] / max(t0['n_kw'], 1):5.1f} %) {min(cor):4d}-{max(cor):<4d} "
            f"{mean['wrong']:6.1f} {mean['missed']:7.1f} {mean['other_alarm']:10.1f} of {t0['n_other']:<4d} "
            f"{mean['quiet_alarm']:6.1f} of {t0['n_quiet']:<4d}  {cmp_}")
    out("\nA design's numbers are the mean over its phases (F40: 2, F80: 4, U: 16); the range is over phases. "
        "Every design is scored on the same decisions, so the comparison is paired.")
    for run in run_dirs:
        with open(os.path.join(run, "wholeword_spoken.json"), "w") as f:
            json.dump(dict(runs=run_dirs, designs=res), f, indent=1, default=float)
    return res


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "spoken":
        if len(sys.argv) < 3:
            sys.exit(__doc__)
        spoken(sys.argv[2:])
        return
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if sys.argv[1] == "run":
        assert not ({s + r for _, s, _, r in SPECS} & used_seeds()), "a test seed was used before"
        os.makedirs(OUT, exist_ok=True)
        with mp.Pool(workers) as pool:
            for _ in pool.imap_unordered(run_one, sorted(SPECS, key=lambda s: -s[2])):
                pass
    elif sys.argv[1] == "report":
        report(workers)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
