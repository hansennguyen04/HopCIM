#!/usr/bin/env python3
"""HopCIM-AP model (activity- and schedule-proportional power on S+DWU): the golden level gate, the pricing [E] of the
fixed power domains, and the scoring of the gate on streams, room runs and spoken runs.

Gate (fixed in docs/HOPCIM_AP_GATES.md). Per hop h, with bm[h] the hop's maximum sample (cim_audio's bmax):
    st = max(1, F >> S);  F = F + st if bm > F else max(1, F - DOWN * st)        (F starts at max(bm[0], 1))
    g  = max(GMIN, (A8 * F) >> 3)
The decision hop h (h >= 49, scheduled every k hops) runs the network only if W[h] = max bm[h-49..h] >= g[h]; otherwise
it is skipped: no network, no interrupt; the firmware learns the number of skips from AP_NSKIP_D and treats each skipped
decision as "no keyword". U (k = 16, sum of 2 passes) reports only when the previous scheduled decision was launched too,
so every report still rests on two network passes (a skipped decision has no logits to add). A skip restarts the hold
rule, so the gate can add a report: AP7 and AP8 allow at most 1 report more over each set.
F settles where P(bm > F) = DOWN / (1 + DOWN): the lower quartile of the hop maxima for DOWN = 3.
Hop 0 of a stream: F = max(bm[0], 1), then the update with bm[0], like every later hop.

Usage (run from hopcim/model, ~/venvs/planenest/bin/python):
  hopcim_ap.py selftest     tracker and gate invariants, broken variants that the golden vectors must catch
  hopcim_ap.py predict      power [E] of the designs at several duties and in every energy-model variant -> ap_prediction.json
  hopcim_ap.py pilot        the pilot scores (WHOLEWORD test streams 501-507, room and spoken runs) -> ap_pilot.json
Held-out evidence (docs/HOPCIM_AP_GATES.md; only after the freeze; --json FILE also writes the result):
  hopcim_ap.py heldout-streams [workers]           AP8 streams (seeds 901-907) into model/ap_heldout/
  hopcim_ap.py score-streams [dir]                 AP8 (default model/ap_heldout/)
  hopcim_ap.py score-spoken <run> [<run> ...]      AP6, AP10 (hopcim_kwlog no-audio prompted runs)
  hopcim_ap.py score-room --quiet <run> ... [--background <run> ...]   AP7, and the launch duties that price AP5
                                                   (the mean of the quiet rooms' duties: every room weighs the same)
"""
import json
import math
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

import dwswap as dw
import dwu
import energy_est as ee

HERE = os.path.dirname(os.path.abspath(__file__))
A8, S, DOWN, GMIN = 20, 6, 3, 0                 # threshold 2.5 x the lower-quartile floor of the hop maxima
W_WAKE = 64                                     # cycles from a domain's power-up request to its first use
AP_GE = 1500.0                                  # gate + tracker + domain controller, gate equivalents (estimate; AP4 measures it)
FE_WIN, NET, WP_AVG = 7476, 67789, 0.884 * 207 + 0.116 * 4719   # [M] worst front-end window, network, window path mean
# State that must stay powered, priced as flip-flops (conservative): cim_seq's logits (16 x 8), class, block-maximum ring
# (50 x 16) and its valid bits, pointer, window maximum, lmx16, flags; cim_fe's lraw output (40 x 32), read by W_COPY.
AON_SEQ_BITS = 128 + 4 + 800 + 50 + 6 + 16 + 20 + 6
AON_FE_BITS = 40 * 32
U_TODAY = None                                  # filled by prices(): D4 at k = 16, all powered, baseline variant


# ---------------------------------------------------------------- the golden gate
def tracker(bm, s=S, down=DOWN):
    out = np.empty(len(bm), np.int64)
    f = max(int(bm[0]), 1)
    for i, b in enumerate(np.asarray(bm, np.int64).tolist()):
        st = max(1, f >> s)
        f = f + st if b > f else max(1, f - down * st)
        out[i] = f
    return out


def threshold(bm, a8=A8, s=S, down=DOWN, gmin=GMIN):
    return np.maximum(gmin, (a8 * tracker(bm, s, down)) >> 3)


def gate(bm, **kw):
    """Per decision hop h = 49 .. len(bm) - 1: True if the network runs."""
    bm = np.asarray(bm, np.int64)
    W = swv(bm, 50).max(axis=1)
    return W >= threshold(bm, **kw)[49:]


# ---------------------------------------------------------------- AP2: cycles over D4 with AP_CTRL[1] = 1
# A unit start waits W after its request when a domain it needs is off; every switched domain is off at each such start
# (each is released at its unit's done). The same run of D4 on the same stimulus gives the rest of the cycles.
WAKES = {
    "live hop 0": 1,              # no front-end at hop 0: sequencer (window path only)
    "live hop": 2,                # hop >= 1 without a network (hops 1-48, not scheduled, or skipped): front-end, sequencer
    "live launched": 3,           # front-end, sequencer, network (PD_ACT, PD_DWU at the window path's end)
    "cpu front-end": 1,
    "cpu window path": 1,         # a sequencer run with the window path only
    "cpu window path + network": 2,
    "cpu network": 1,             # a network-only run: PD_SEQ, PD_CIM, PD_ACT, PD_DWU requested together at go
}


def wake_cycles(run):
    return W_WAKE * WAKES[run]


# ---------------------------------------------------------------- pricing [E]
def _setup(vname):
    pr = dw.Pricing(dict(total=dw.S_RTL_TOTAL, counts={}, words={}))
    m = pr.model(dict(pr.variants)[vname])
    q = pr.parts[vname]
    G = m.a["cim_group"]
    lS = pr.leak(m, "S")
    lD = dict(lS)
    lD["weight cells"] = 29 * m.sram_uw("weights", dw.TILE_BITS)
    lD["CIM macro logic"] = lS["CIM macro logic"] / math.ceil(69 / G) * math.ceil(29 / G)
    dws = m.sram_uw("params", dwu.STORE_WORDS * dwu.STORE_W)
    lD["other SRAM"] += dws
    a = m.a
    ge = lambda b: (ee.BLOCKS[b]["lut"] * a["ge_per_lut"] + ee.BLOCKS[b]["ff"] * a["ge_per_ff"]
                    + ee.BLOCKS[b]["dsp"] * a["ge_per_dsp"]) * a["leak_pw_per_ge"] * 1e-6
    blk = {b: ge(b) for b in ("cim_seq", "cim_audio", "glue", "cim_fe")}
    blk["dwu"] = 12414.0 * a["leak_pw_per_ge"] * 1e-6
    lD["digital logic"] += blk["dwu"]
    sram = {n: m.sram_uw(n) for n in ("act_word", "bbuf", "ribuf", "magbuf")}
    p0 = q["p0_uW"] - sum(lS.values()) + sum(lD.values())
    hop_dyn = p0 - sum(lD.values())
    e_net = sum(json.load(open(os.path.join(HERE, "dwu_rtl.json")))["parts"]["energy_D4_pJ"].values())
    alw = (ee.BLOCKS["cim_audio"]["ff"] - ee.BLOCKS["pdm_dec"]["ff"]) * m.e_ff * a["f_sys_hz"] * 1e-15 * 1e6
    lg = a["leak_pw_per_ge"] * 1e-6
    return dict(lD=lD, blk=blk, sram=sram, hop_dyn=hop_dyn, e_net=e_net, e_cpu=q["e_cpu_pJ"], alw=alw, leak_ge=lg,
                hop=a["f_sys_hz"] * ee.HOP_S,                       # cycles per hop at the variant's system clock
                aon_seq=min(blk["cim_seq"], AON_SEQ_BITS * a["ge_per_ff"] * lg),
                aon_fe=min(blk["cim_fe"], AON_FE_BITS * a["ge_per_ff"] * lg))


def power(c, k, duty, mode, r=0.05, wake_us=50.0, cg_frac=0.9):
    """Average power (uW) at one scheduled decision every k hops.
    mode "d4": D4 as built (everything powered, every scheduled decision launched; duty is ignored).
    mode "ce": D4 + clock enables on cim_audio's always-clocked flip-flops only.
    mode "ap": level gate (a fraction `duty` of the scheduled decisions launched) + clock enables + the coarse domains.
    Domain sequence per hop (docs/HOPCIM_AP_GATES.md): hop end -> PD_FE, PD_CIM wake, front-end W later; front-end done ->
    PD_SEQ wakes, sequencer W later; window path done on a launched decision -> PD_ACT, PD_DWU wake, network W later."""
    lD, blk, sram, H, W = c["lD"], c["blk"], c["sram"], c["hop"], W_WAKE
    lr = (duty if mode == "ap" else 1.0) / k             # network launches per hop
    dyn = c["hop_dyn"] + lr / ee.HOP_S * (c["e_net"] + c["e_cpu"]) * 1e-6
    if mode == "d4":
        return sum(lD.values()) + dyn
    dyn -= cg_frac * c["alw"]
    if mode == "ce":
        return sum(lD.values()) + dyn
    on = lambda f, leak: leak * (f + r * (1 - f))
    L = dict(lD)
    fe_b = sram["bbuf"] + sram["ribuf"] + sram["magbuf"]
    seq_sw = blk["cim_seq"] - c["aon_seq"]               # switched parts of cim_seq and cim_fe
    fe_sw = blk["cim_fe"] - c["aon_fe"]
    f_fe = (W + FE_WIN) / H                              # PD_FE: request at hop end, off at front-end done
    f_cim = (2 * W + FE_WIN + WP_AVG + lr * (W + NET)) / H   # PD_CIM: hop end to sequencer done
    f_seq = (W + WP_AVG + lr * (W + NET)) / H            # PD_SEQ: front-end done to sequencer done
    f_act = lr * (W + NET) / H                           # PD_ACT, PD_DWU: window path done to network done
    L["CIM macro logic"] = on(f_cim, lD["CIM macro logic"])
    L["other SRAM"] += on(f_act, sram["act_word"]) - sram["act_word"] + on(f_fe, fe_b) - fe_b
    L["digital logic"] += (on(f_fe, fe_sw) - fe_sw + on(f_seq, seq_sw) - seq_sw
                           + on(f_act, blk["dwu"]) - blk["dwu"] + AP_GE * c["leak_ge"])
    wakes_s = (lD["CIM macro logic"] + fe_b + fe_sw + seq_sw) / ee.HOP_S \
              + (sram["act_word"] + blk["dwu"]) * lr / ee.HOP_S        # leakage-weighted wakes per second
    return sum(L.values()) + dyn + wakes_s * wake_us * 1e-6


DESIGNS = (("D4-AP20", 1), ("D4-AP40", 2), ("D4-APU", 16))
DUTIES = (1.0, 0.5, 0.2, 0.069, 0.02)


def _duty_at(c, k, target):
    """The highest launch duty at which design (k) stays at or below `target` uW (bisection; 0 if never)."""
    if power(c, k, 0.0, "ap") > target:
        return 0.0
    lo, hi = 0.0, 1.0
    if power(c, k, 1.0, "ap") <= target:
        return 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power(c, k, mid, "ap") <= target else (lo, mid)
    return lo


def prices():
    global U_TODAY
    pr = dw.Pricing(dict(total=dw.S_RTL_TOTAL, counts={}, words={}))
    out, brk = {}, {}
    for vname, _ in pr.variants:
        c = _setup(vname)
        row = {f"D4 k={k}": power(c, k, 1.0, "d4") for k in (1, 2, 16)}
        row.update({f"D4+CE k={k}": power(c, k, 1.0, "ce") for k in (1, 2, 16)})
        for name, k in DESIGNS:
            for du in DUTIES:
                row[f"{name} duty {du}"] = power(c, k, du, "ap")
        out[vname] = row
        u = row["D4 k=16"]
        brk[vname] = {"AP20 below U today": _duty_at(c, 1, u), "AP20 at 0.5 x U today": _duty_at(c, 1, 0.5 * u),
                      "APU at 0.5 x U today": _duty_at(c, 16, 0.5 * u)}
    U_TODAY = out["baseline"]["D4 k=16"]
    c = _setup("baseline")
    sens = {}
    for rr in (0.02, 0.05, 0.1, 0.2):
        for w in (10.0, 50.0, 500.0):
            sens[f"r={rr} wake={w}us"] = power(c, 1, 0.069, "ap", r=rr, wake_us=w)
    return out, sens, brk


# ---------------------------------------------------------------- scoring
def events_for(design, z, rule_gate):
    """Events of a design on a stream z (logits per decision hop from 49, bm per hop); rule_gate: per-decision mask."""
    import kwlog_analyse as ka
    L = z["logits"]
    hops = np.arange(len(L)) + 49
    g = np.ones(len(L), bool) if rule_gate is None else rule_gate
    if design == "F20":
        cls = np.where(g, np.argmax(L, 1), 11)
        return [(ka.events_hold(cls, hops, 5), int(g.sum()))]
    if design == "F40":
        res = []
        for p in range(2):
            idx = ka.subsample(len(L), 2, p)
            res.append((ka.events_hold(np.where(g[idx], np.argmax(L[idx], 1), 11), hops[idx], 3), int(g[idx].sum())))
        return res
    res = []                                              # U: every 16 hops, sum of 2 consecutive launched passes
    for p in range(16):
        idx = ka.subsample(len(L), 16, p)
        Li, gi = L[idx], g[idx]
        prev_ok = np.concatenate([[True], gi[:-1]])       # the previous scheduled decision was launched (hop 49: none, 0)
        s = Li + np.vstack([np.zeros((1, 12), np.int64), Li[:-1]])
        res.append((ka.events_hold(np.where(gi & prev_ok, np.argmax(s, 1), 11), hops[idx], 1), int(gi.sum())))
    return res


def pilot():
    import kwlog_analyse as ka
    import wholeword as ww
    runs = ww.load_dir(ww.OUT)
    rooms = ww.load_room()
    res = {}
    for design in ("F20", "F40", "U"):
        for gated in (False, True):
            key = f"{design}{' + gate' if gated else ''}"
            mk = lambda z: gate(z["bm"]) if gated else None
            fns = [(lambda j: (lambda z: (events_for(design, z, mk(z))[j][0], events_for(design, z, mk(z))[j][1],
                                          events_for(design, z, mk(z))[j][1])))(j)
                   for j in range({"F20": 1, "F40": 2, "U": 16}[design])]
            r = {f"words r{rms}": 100 * ww.score(fns, runs, rms=rms)["words"]["recall"] for rms in (30, 100, 300)}
            s = ww.score(fns, runs)
            r.update({f"{kd} launches per hop": s[kd]["net_per_hop"] for kd in ("words", "speech", "noise")})
            r.update({f"{kd} reports/h": s[kd]["per_hour"] for kd in ("speech", "noise")})
            sr = ww.score_room(fns, rooms)
            r.update({"room launches per hop": sr["net_per_hop"], "room reports/h": sr["per_hour"]})
            res[key] = r
            print(key, {k: round(v, 4) for k, v in r.items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "ap_pilot.json"), "w"), indent=1)
    return res


# ---------------------------------------------------------------- held-out scoring (fixed before any held-out data)
HELDOUT = os.path.join(HERE, "ap_heldout")
HELDOUT_SPECS = ([("words", s, 600, r) for r in (30, 100, 300) for s in (901, 902, 903, 904)]
                 + [("speech", s, 600, r) for r in (30, 100, 300) for s in (905, 906)]
                 + [("noise", 907, 1200, r) for r in (30, 100, 300)])
K_OF = {"F20": 1, "F40": 2, "U": 16}                   # decision interval in hops = number of phases scored
_EV = {}


def _events(design, z, gated):
    key = (z["name"], design, gated)
    if key not in _EV:
        _EV[key] = events_for(design, z, gate(z["bm"]) if gated else None)
    return _EV[key]


def _fns(design, gated):
    """One function per phase, z -> (events, launches, launches), as wholeword.score expects."""
    def mk(j):
        def f(z):
            e, n = _events(design, z, gated)[j]
            return e, n, n
        return f
    return [mk(j) for j in range(K_OF[design])]


def heldout_streams(workers=8):
    """AP8 data: the WHOLEWORD generators with seeds never used before, into model/ap_heldout/ (after the freeze only)."""
    import multiprocessing as mp
    import wholeword as ww
    used = ww.used_seeds() | {s + r for _, s, _, r in ww.SPECS}
    assert not ({s + r for _, s, _, r in HELDOUT_SPECS} & used), "a held-out seed was used before"
    os.makedirs(HELDOUT, exist_ok=True)
    ww.OUT = HELDOUT                                     # wholeword.run_one writes into ww.OUT (inherited by the workers)
    with mp.Pool(workers) as pool:
        pool.map(ww.run_one, HELDOUT_SPECS)


def score_streams(path):
    """AP8: recall at each RMS and speech / noise reports per hour, with and without the gate, for F20, F40 and U."""
    import wholeword as ww
    runs = ww.load_dir(path)
    v2 = sum(z["v2_bad"] or 0 for kd in runs for z in runs[kd])
    res = {}
    for design in K_OF:
        for gated in (False, True):
            fns = _fns(design, gated)
            r = {f"recall r{rms}": 100 * ww.score(fns, runs, rms=rms)["words"]["recall"] for rms in (30, 100, 300)}
            sc = ww.score(fns, runs)
            r.update({f"{kd} reports/h": sc[kd]["per_hour"] for kd in ("speech", "noise")})
            r.update({f"{kd} launch duty": sc[kd]["net_per_hop"] * K_OF[design] for kd in ("words", "speech", "noise")})
            res[design + (" + gate" if gated else "")] = r
    hours = {kd: sum(z["seconds"] for z in runs[kd]) / 3600 for kd in ("speech", "noise")}
    ap8 = {}
    for design in K_OF:
        a, b = res[design], res[design + " + gate"]
        # a skip restarts the hold rule, so the gate can add a report: at most 1 report more over each set (phase mean)
        ap8[design] = all(b[f"recall r{x}"] >= a[f"recall r{x}"] - 1.0 for x in (30, 100, 300)) and \
            all(b[f"{kd} reports/h"] <= a[f"{kd} reports/h"] + 1.0 / hours[kd] for kd in ("speech", "noise"))
    return dict(path=path, n_streams=sum(len(runs[kd]) for kd in runs), v2_differ=v2, scores=res, AP8=ap8,
                AP8_pass=all(ap8.values()) and v2 == 0)


def _kwrun(path):
    """A hopcim_kwlog run (no audio): every decision's logits, every hop's maximum (missing hops: -32768)."""
    import kwlog_analyse as ka
    r = ka.load(path)
    n_hops = len(r["logits"]) + 49
    bm = np.array([r["env"].get(h, -32768) for h in range(n_hops)], np.int64)
    problems = list(ka.integrity(r))
    if not (len(r["hop"]) and int(r["hop"][0]) == 49 and np.all(np.diff(r["hop"]) == 1)):
        problems.append("decisions are not every hop from hop 49")
    z = dict(name=os.path.basename(os.path.normpath(path)), logits=np.asarray(r["logits"], np.int64), bm=bm)
    return r, z, dict(run=z["name"], hours=len(z["logits"]) * ee.HOP_S / 3600,
                      missing_hops=int(sum(1 for h in range(n_hops) if h not in r["env"])), problems=problems)


def _latency(events, trials):
    """Per keyword trial (index): seconds from the prompt to the first report of the prompted word, if any."""
    from kwlog_frames import LABEL
    out = {}
    for i, t in enumerate(trials):
        if t["kind"] == "keyword":
            hs = [h for h, c in events if t["start"] < h <= t["end"] and LABEL[c] == t["word"]]
            if hs:
                out[i] = (min(hs) - t["start"]) * ee.HOP_S
    return out


def score_spoken(paths):
    """AP6 and AP10 on prompted kwlog runs. A run with an integrity problem is not scored (reported; it is replaced)."""
    import kwlog_analyse as ka
    runs, tot, lat_pairs = [], {}, []
    for p in paths:
        r, z, info = _kwrun(p)
        runs.append(info)
        if info["problems"]:
            continue
        trials = ka.trials_of(r)
        for design in K_OF:
            for gated in (False, True):
                evs = _events(design, z, gated)
                sc = [ka.score(e, trials) for e, _ in evs]
                key = design + (" + gate" if gated else "")
                t = tot.setdefault(key, dict(n_kw=0, correct=0.0, other_alarm=0.0, quiet_alarm=0.0))
                t["n_kw"] += sc[0]["n_kw"]
                for x in ("correct", "other_alarm", "quiet_alarm"):
                    t[x] += float(np.mean([v[x] for v in sc]))
        f = _latency(_events("F20", z, True)[0][0], trials)                  # F20 with the gate
        u = [_latency(e, trials) for e, _ in _events("U", z, False)]         # today's U, every phase
        for i, v in f.items():
            us = [d[i] for d in u if i in d]
            if us:
                lat_pairs.append((v, float(np.mean(us))))
    a, b = tot.get("F20"), tot.get("F20 + gate")
    ap6 = None if a is None else (b["correct"] >= a["correct"] - a["n_kw"] / 100.0 and
                                  b["other_alarm"] + b["quiet_alarm"] <= a["other_alarm"] + a["quiet_alarm"] + 1)
    lp = np.array(lat_pairs) if lat_pairs else np.zeros((0, 2))
    ap10 = dict(pairs=len(lp), median_F20_gate_s=float(np.median(lp[:, 0])) if len(lp) else None,
                median_U_s=float(np.median(lp[:, 1])) if len(lp) else None,
                median_paired_difference_s=float(np.median(lp[:, 0] - lp[:, 1])) if len(lp) else None)
    return dict(runs=runs, scores=tot, AP6_pass=ap6, AP10=ap10)


def score_room(quiet, background=()):
    """AP7 on all room runs; the launch duty of the quiet runs prices AP5 (model rules; AP5 itself uses the RTL counters);
    a background-sound run is reported with its own duty and power, no threshold."""
    runs, per = [], {}
    for kind, paths in (("quiet", quiet), ("background", background)):
        for p in paths:
            r, z, info = _kwrun(p)
            info["kind"] = kind
            runs.append(info)
            if info["problems"]:
                continue
            for design in K_OF:
                for gated in (False, True):
                    evs = _events(design, z, gated)
                    key = design + (" + gate" if gated else "")
                    d = per.setdefault((kind, key), dict(reports=0.0, hours=0.0, launched=0, scheduled=0))
                    d["reports"] += float(np.mean([len(e) for e, _ in evs]))
                    d["hours"] += info["hours"]
                    n_l = sum(n for _, n in evs)
                    n_s = sum(len(ka_idx) for ka_idx in
                              (np.arange(j, len(z["logits"]), K_OF[design]) for j in range(K_OF[design])))
                    d["launched"] += n_l
                    d["scheduled"] += n_s
                    info[f"{key} launch duty"] = n_l / n_s
            g = gate(z["bm"])
            n = int(3600 / ee.HOP_S)
            info["F20 launch duty per hour"] = [float(g[i:i + n].mean()) for i in range(0, len(g), n)]
            info["F20 launch duty"] = float(g.mean())

    def pooled(kinds, key):
        ds = [per[(k, key)] for k in kinds if (k, key) in per]
        if not ds:
            return None
        rep, hrs = sum(d["reports"] for d in ds), sum(d["hours"] for d in ds)
        return dict(reports_per_hour=rep / hrs, hours=hrs, launch_duty=sum(d["launched"] for d in ds) / sum(d["scheduled"] for d in ds))
    allk = ("quiet", "background")
    tab = {key: {"all": pooled(allk, key), "quiet": pooled(("quiet",), key), "background": pooled(("background",), key)}
           for key in [d + g for d in K_OF for g in ("", " + gate")]}
    # a skip restarts the hold rule, so the gate can add a report: at most 1 report more over R1-R3 (phase mean)
    ap7 = {d: (tab[d]["all"] is not None and tab[d + " + gate"]["all"]["reports_per_hour"] * tab[d]["all"]["hours"]
               <= tab[d]["all"]["reports_per_hour"] * tab[d]["all"]["hours"] + 1.0)
           for d in K_OF}
    price, duty = {}, {}
    for kind in ("quiet", "background"):
        ok = [i for i in runs if i["kind"] == kind and not i["problems"]]
        if not ok:
            continue
        # every room weighs the same, whatever its length (R1 is the pilot's room)
        duty[kind] = {key: float(np.mean([i[f"{key} launch duty"] for i in ok])) for key in ("F20 + gate", "U + gate")}
        q20, qu = duty[kind]["F20 + gate"], duty[kind]["U + gate"]
        pr = dw.Pricing(dict(total=dw.S_RTL_TOTAL, counts={}, words={}))
        price[kind] = {}
        for vname, _ in pr.variants:
            c = _setup(vname)
            u = power(c, 16, 1.0, "d4")
            price[kind][vname] = dict(u_today=u, AP20=power(c, 1, q20, "ap"), APU=power(c, 16, qu, "ap"))
    ap5_model = {}
    for vname in ("baseline", "4.096 MHz clock"):
        b = price.get("quiet", {}).get(vname)
        if b is not None:
            ap5_model[vname] = dict(a=b["AP20"] < b["u_today"], b=b["APU"] <= 0.5 * b["u_today"])
    return dict(runs=runs, table=tab, AP7=ap7, AP7_pass=all(ap7.values()), duty_room_mean=duty, price=price,
                AP5_model_pricing=ap5_model or None)


# ---------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(7)
    ok = True
    # the tracker settles near the lower quartile of a stationary input
    x = rng.integers(40, 120, 200000)
    f = tracker(x)[50000:]
    frac = float((x[50000:] > f).mean())
    ok &= 0.70 <= frac <= 0.80
    print(f"stationary input: P(bm > F) = {frac:.3f} (expected about 0.75)")
    # the gate opens on a loud event and stays closed on the floor
    bm = np.concatenate([rng.integers(60, 100, 3000), np.full(20, 900), rng.integers(60, 100, 3000)])
    g = gate(bm)
    opened = np.nonzero(g)[0] + 49
    ok &= bool(len(opened)) and opened.min() >= 3000 and opened.max() <= 3019 + 49
    ok &= float(g[:2900].mean()) < 0.01
    print(f"loud event at hops 3000-3019: gate open at hops {opened.min() if len(opened) else None}..{opened.max() if len(opened) else None}, "
          f"open on the floor {100 * g[:2900].mean():.2f} %")
    # golden vectors catch broken variants
    bm = rng.integers(20, 3000, 5000) * (rng.random(5000) < 0.1) + rng.integers(40, 120, 5000)
    ref = gate(bm)
    bad = {"down 1": gate(bm, down=1), "S 5": gate(bm, s=5), "A8 19": gate(bm, a8=19),
           "previous hop's W": np.concatenate([[True], swv(bm, 50).max(axis=1)[:-1] >= threshold(bm)[50:]])}
    for n, v in bad.items():
        caught = not np.array_equal(v, ref)
        ok &= caught
        print(f"broken variant '{n}': {'caught' if caught else 'NOT CAUGHT'} ({int((v != ref).sum())} decisions differ)")
    # U never uses a skipped decision's logits (the hardware has none): phase 0, decisions 2 (skipped, strong "yes")
    # and 3 (launched, weak "yes"); summed they would report "yes", alone decision 3 is silence
    L = np.zeros((16 * 6, 12), np.int64)
    L[:, 10] = 5
    L[32, 2], L[48, 2] = 100, 4
    g = np.ones(len(L), bool)
    g[32] = False
    z = dict(logits=L, bm=np.zeros(len(L) + 49, np.int64))
    e_skip = events_for("U", z, g)[0][0]
    e_both = events_for("U", z, None)[0][0]
    u_ok = not any(c == 2 for _, c in e_skip) and any(c == 2 for _, c in e_both)
    ok &= u_ok
    print(f"U after a skipped decision: {'no report (correct)' if u_ok else 'WRONG'}; with both launched: reports {e_both}")
    print("selftest", "PASS" if ok else "FAIL")
    return ok


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if cmd == "pilot":
        pilot()
        return
    if cmd in ("heldout-streams", "score-streams", "score-spoken", "score-room"):
        args = sys.argv[2:]
        jpath = None
        if "--json" in args:
            i = args.index("--json")
            jpath, args = args[i + 1], args[:i] + args[i + 2:]
        if cmd == "heldout-streams":
            heldout_streams(int(args[0]) if args else 8)
            return
        if cmd == "score-streams":
            res = score_streams(args[0] if args else HELDOUT)
        elif cmd == "score-spoken":
            res = score_spoken(args)
        else:
            qi = args.index("--quiet") if "--quiet" in args else None
            bi = args.index("--background") if "--background" in args else len(args)
            res = score_room(args[qi + 1:bi] if qi is not None else [], args[bi + 1:])
        print(json.dumps(res, indent=1, default=float))
        if jpath:
            json.dump(res, open(jpath, "w"), indent=1, default=float)
        return
    out, sens, brk = prices()
    b = out["baseline"]
    print(f"U today (D4, k=16, all powered): {U_TODAY:.2f} uW; D4 k=1 {b['D4 k=1']:.2f}, k=2 {b['D4 k=2']:.2f}")
    print("D4 + clock enables only: " + ", ".join(f"k={k} {b[f'D4+CE k={k}']:.2f}" for k in (1, 2, 16)))
    print(f"{'design':10s}" + "".join(f"  duty {d:<6}" for d in DUTIES))
    for name, _ in DESIGNS:
        print(f"{name:10s}" + "".join(f"  {b[f'{name} duty {d}']:10.2f}" for d in DUTIES))
    print("break-even launch duties (baseline):", {k: round(v, 3) for k, v in brk["baseline"].items()})
    print("D4-AP20 at duty 0.069 over residual leakage and wake time:", {k: round(v, 2) for k, v in sens.items()})
    print("by energy-model variant: D4-AP20 duty 0.069 / D4-APU duty 0.069 / U today; AP20 ratio, APU ratio; break-evens")
    for v, row in out.items():
        u = row["D4 k=16"]
        print(f"  {v:28s} {row['D4-AP20 duty 0.069']:7.2f} / {row['D4-APU duty 0.069']:7.2f} / {u:7.2f}; "
              f"{row['D4-AP20 duty 0.069'] / u:.3f}, {row['D4-APU duty 0.069'] / u:.3f}; "
              + ", ".join(f"{k} {x:.3f}" for k, x in brk[v].items()))
    json.dump(dict(u_today=U_TODAY, by_variant=out, sensitivity=sens, break_even_duty=brk,
                   rule=dict(A8=A8, S=S, DOWN=DOWN, GMIN=GMIN), W_WAKE=W_WAKE, AP_GE=AP_GE,
                   aon_bits=dict(cim_seq=AON_SEQ_BITS, cim_fe=AON_FE_BITS)),
              open(os.path.join(HERE, "ap_prediction.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
