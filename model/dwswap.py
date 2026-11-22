#!/usr/bin/env python3
"""Depthwise swap-in study (docs/DWSWAP_GATES.md, frozen at ce3710d): V, the depthwise layers on two working tiles
written from a dense store, against S, N7 (narrow static depthwise macros), FS (feature-stationary depthwise, digital)
and DIG (a 9-multiplier digital depthwise unit).

Three parts:
  cycles   S's schedule model (streamfuse_sched.py) with V's order (each depthwise layer group by group) and tiles;
           the rewrites scheduled on the cycle model's pass times (one row write per cycle, a group's first write
           after the last pass of the previous group on that tile is done); every pass on a working tile checked
           against the rows it needs (they must hold its group at issue, and no write may land on its tile from its
           issue to its done). V1, V's cycles, G6, and the three broken variants' hazards.
  exact    the cycle model's gathers, passes, rewrites and result writes replayed in cycle order on real clips. The
           engine's two input banks persist from vector to vector and the gather writes only a vector's own rows, as
           cim_seq.sv's masked wide port does, so rows a vector does not write hold an earlier vector's data (random
           at the start). The working tiles persist from group to group and start each inference in the state the
           previous one left (stale cells of group 9 included). A pass on a working tile with a write landing during
           its bit cycles is computed bit plane by bit plane, LSB first, each plane with the tile as it is in that
           cycle. V2 (S through this replay), G1 (V), and the broken variants' wrong values. "adversarial": also every
           row a vector did not write is filled with fresh random values before each of its passes.
  price    every design in the 19 variants of energy_est.py, at U (k = 16) and 40 ms (k = 2): V3, V4, G2-G5, G7.

Usage: dwswap.py cycles | exact [clips] | price | all [clips]     (writes model/dwswap_results.json)
"""
import bisect
import copy
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import cim_net_map as nm
import dscnn_int as d
import energy_est as ee
import streamfuse_sched as ss

HERE = os.path.dirname(os.path.abspath(__file__))
NW = 2                                   # working tiles
WT = 128                                 # their tile ids in the program: 128, 129
DW_LAYERS = (1, 3, 5, 7)
TILE_BITS = 64 * 16 * 8
STORE_BYTES = 2304
# BASELINE_S.md section 4 (RTL, sequencer bench) and section 5 (hop monitor)
S_RTL_LAYER = [4533, 11273, 4531, 11273, 4531, 11273, 4531, 11273, 4531, 92]
S_RTL_TOTAL = 67841
S_WORST_HOP = 80072
HOP_CYCLES = 81920
SKIP = (3, 4)                            # the broken variant "write skipped": layer 3, group 4
MUTANTS = ("tile in use", "write skipped", "early write")
K_U, K_40 = 16, 2


# ---------------------------------------------------------------- the store and the programs
def store_image():
    """The dense store, from the int8 model itself (not from the tile map): layer l, group g, tap k, lane c' at
    byte ((l * 10 + g) * 9 + k) * 7 + c' for groups 0..8, group 9 one byte per tap after them."""
    st, addr = [], {}
    for l in range(4):
        w = d.P[f"L{1 + 2 * l}_w"]                       # [1, 3, 3, 64]
        for g in range(10):
            for k in range(9):
                fy, fx = divmod(k, 3)
                for cp in range(ss.lanes_of(g)):
                    addr[(l, g, k, cp)] = len(st)
                    st.append(int(w[0, fy, fx, 7 * g + cp]))
    assert len(st) == STORE_BYTES
    return np.array(st, np.int64), addr


STORE, ADDR = store_image()


def make_program(order="group", working=True, mutate=None):
    """S's program; order 'group' runs each depthwise layer group by group (for g: for p), 'position' as S
    (for p: for g); working: the depthwise passes on tile WT + g mod 2 (or WT for 'tile in use')."""
    P = ss.build(f"V order={order} working={working} {mutate or ''}", "lbl", 2, sg=True)
    for L in DW_LAYERS:
        seg = P.segments[L]
        if order == "group":
            seg = sorted(seg, key=lambda v: (v.g, v.p))
        for v in seg:
            v.last = False
            if working:
                T = 0 if mutate == "tile in use" else v.g % NW
                v.passes = [(WT + T, res) for (_, res) in v.passes]
        seg[-1].last = True
        P.segments[L] = seg
    vid = 0
    for seg in P.segments:
        for v in seg:
            v.id = vid
            vid += 1
    return P


def pass_times(sim):
    """Issue cycle of every pass, from its first result write (done = issue + 14, first write = done + 6)."""
    t = {}
    for ev in sim.events:
        if ev[2] == "W" and ev[6] == 0:
            key = (ev[4], ev[5])
            t[key] = min(t.get(key, 1 << 40), ev[0] - ss.DONE - 6)
    return t


def group_cells(L, g):
    """The row writes of one group: (row, column, weight), rows in ascending order (tap-major 7 k + c')."""
    l = (L - 1) // 2
    return [(7 * k + cp, cp, int(STORE[ADDR[(l, g, k, cp)]])) for k in range(9) for cp in range(ss.lanes_of(g))]


def write_schedule(P, times, mutate=None):
    """One row write per cycle through one store port. Group g goes to tile g mod 2; its first write lands after the
    last pass of the previous group on that tile is done (broken 'early write': after it is issued). A write
    'lands' at the end of its cycle and is seen from the next cycle on."""
    vecs = {v.id: v for seg in P.segments for v in seg}
    grp = {}
    for (vid, j), t in times.items():
        v = vecs[vid]
        if v.kind == "dw":
            a = grp.setdefault((v.L, v.g), [t, t])
            a[0], a[1] = min(a[0], t), max(a[1], t)
    writes, info = [], []
    port, rel = 0, [0] * NW
    for L in DW_LAYERS:
        for g in range(10):
            T = 0 if mutate == "tile in use" else g % NW
            par = g % NW
            start = max(port, rel[par] + 1)
            cells = [] if (mutate == "write skipped" and (L, g) == SKIP) else group_cells(L, g)
            for i, (r, c, w) in enumerate(cells):
                writes.append((start + i, T, r, c, w, (L, g)))
            last = start + len(cells) - 1
            port = start + len(cells)
            first, lastp = grp[(L, g)]
            info.append(dict(L=L, g=g, tile=T, writes=len(cells), release=rel[par], first_write=start, last_write=last,
                             first_pass=first, last_pass=lastp, slack=first - 1 - last))
            rel[par] = lastp if mutate == "early write" else lastp + ss.DONE
    return writes, info


def steady_tiles(writes):
    """The working tiles (weights and the group each row holds) as one inference leaves them."""
    w = np.zeros((NW, 64, 16), np.int64)
    ver = [[None] * 64 for _ in range(NW)]
    for (land, T, r, c, val, vg) in sorted(writes, key=lambda x: x[0]):
        w[T, r, :] = 0
        w[T, r, c] = val
        ver[T][r] = vg
    return w, ver


def need_rows(g):
    return [7 * k + cp for k in range(9) for cp in range(ss.lanes_of(g))]


def tile_hazards(P, times, writes):
    """Every pass on a working tile: the rows its group needs hold that group at issue, and no write lands on its
    tile from its issue to its done. Data-independent (the weights do not depend on the clip)."""
    vecs = {v.id: v for seg in P.segments for v in seg}
    _, ver = steady_tiles(writes)
    land_by_tile = {T: sorted(x[0] for x in writes if x[1] == T) for T in range(NW)}
    ev = [(x[0], 1, x) for x in writes]
    for (vid, j), t in times.items():
        tile = vecs[vid].passes[j][0]
        if tile >= WT:
            ev.append((t, 0, (vid, j, tile - WT)))
    ev.sort(key=lambda e: (e[0], e[1]))
    hz = []
    for t, kind, x in ev:
        if kind == 1:
            land, T, r, c, val, vg = x
            ver[T][r] = vg
            continue
        vid, j, T = x
        v = vecs[vid]
        bad = [r for r in need_rows(v.g) if ver[T][r] != (v.L, v.g)]
        if bad:
            hz.append(f"t{t}: pass L{v.L} g{v.g} p{v.p} on working tile {T}: {len(bad)} rows hold another group")
        ls = land_by_tile[T]
        n = bisect.bisect_right(ls, t + ss.DONE) - bisect.bisect_left(ls, t)
        if n:
            hz.append(f"t{t}: pass L{v.L} g{v.g} p{v.p} on working tile {T}: {n} writes land before it is done")
    return hz


def schedule(mutate=None, order="group", working=True):
    P = make_program(order, working, mutate)
    sim, r = ss.simulate(P)
    times = pass_times(sim)
    writes, info = write_schedule(P, times, mutate) if working else ([], [])
    hz = tile_hazards(P, times, writes) if working else []
    return P, sim, r, times, writes, info, hz


def cycles(out=print):
    res = {}
    # V1: the schedule model reproduces S (RTL, BASELINE_S.md section 4)
    _, rS = ss.simulate(ss.make("S"))
    v1 = rS["total"] == S_RTL_TOTAL and rS["segments"] == S_RTL_LAYER and rS["hazards"] == 0
    res["V1"] = dict(passed=v1, total=rS["total"], segments=rS["segments"], rtl=S_RTL_LAYER)
    out(f"V1: S's schedule model {rS['total']} cycles, layers {rS['segments']} against the RTL's {S_RTL_TOTAL}, "
        f"{S_RTL_LAYER}: {'pass' if v1 else 'FAIL'}")
    # V4 for the schedule: V's builder with its change off is S, event for event
    P0, sim0, r0, *_ = schedule(order="position", working=False)
    simS, _ = ss.simulate(ss.make("S"))
    same = r0["total"] == rS["total"] and r0["segments"] == rS["segments"] and sim0.events == simS.events
    res["V4_schedule"] = same
    out(f"V4 (schedule): V's program with its order and tiles switched off gives S's events exactly: {'pass' if same else 'FAIL'}")
    # V
    P, sim, r, times, writes, info, hz = schedule()
    sl = [x["slack"] for x in info]
    cnt_same = {k: r["counts"][k] for k in ("reads", "byte_writes", "feat_reads", "chunks", "passes", "requant", "x_rows")} == \
               {k: rS["counts"][k] for k in ("reads", "byte_writes", "feat_reads", "chunks", "passes", "requant", "x_rows")}
    worst = S_WORST_HOP - S_RTL_TOTAL + r["total"]
    g6 = r["total"] <= 1.02 * S_RTL_TOTAL and worst <= HOP_CYCLES and min(sl) >= 0 and not hz and r["hazards"] == 0
    # min(sl) >= 0: every group's writes land before its first pass, so the rewrites never stall a pass and the
    # post-hoc write schedule is exact on the cycle model's pass times
    res["V"] = dict(total=r["total"], segments=r["segments"], buffer_hazards=r["hazards"], tile_hazards=len(hz),
                    first_tile_hazards=hz[:5], row_writes=len(writes), store_reads=len(writes), min_slack=min(sl),
                    slack_by_layer={L: min(x["slack"] for x in info if x["L"] == L) for L in DW_LAYERS},
                    first_group_slack={L: [x["slack"] for x in info if x["L"] == L][:2] for L in DW_LAYERS},
                    counts_equal_S=cnt_same, worst_hop=worst, counts=r["counts"], words=r["words"])
    res["G6"] = dict(passed=g6, network=r["total"], limit=1.02 * S_RTL_TOTAL, worst_hop=worst, hop_limit=HOP_CYCLES)
    out(f"V: {r['total']} cycles, layers {r['segments']}; ACT hazards {r['hazards']}, working-tile hazards {len(hz)}; "
        f"{len(writes)} row writes and store reads per inference; smallest slack of a group's last write before its "
        f"first pass {min(sl)} cycles (by layer {res['V']['slack_by_layer']}); access counts as S: {cnt_same}")
    out(f"G6: network {r['total']} <= {1.02 * S_RTL_TOTAL:.0f}, worst hop {S_WORST_HOP} - {S_RTL_TOTAL} + {r['total']} = "
        f"{worst} <= {HOP_CYCLES}: {'pass' if g6 else 'FAIL'}")
    # the broken variants in the cycle model
    res["mutants_cycle"] = {}
    for mu in MUTANTS:
        _, _, rm, _, wm, im, hm = schedule(mu)
        res["mutants_cycle"][mu] = dict(tile_hazards=len(hm), first=hm[:3], row_writes=len(wm), total=rm["total"])
        out(f"  broken '{mu}': {len(hm)} working-tile hazards ({hm[0] if hm else 'none'}), {len(wm)} row writes")
    return res, (P, sim, times, writes)


# ---------------------------------------------------------------- data replay
def written_rows(v):
    rows = set()
    for k, ch in enumerate(v.chunks):
        if v.kind == "conv1":
            rows.update(range(4 * k, 4 * k + 4))
        elif v.kind == "dw":
            rows.update(range(7 * k, 7 * k + ss.lanes_of(v.g)))
        elif v.kind == "pw":
            rows.update(range(7 * k, 7 * k + ss.lanes_of(k)))
        else:
            rows.add(k)
    return sorted(rows)


def replay_plan(P, sim, times, writes):
    """The data-independent part of the replay: events in cycle order, banks, and the writes inside each pass."""
    vecs = {v.id: v for seg in P.segments for v in seg}
    bank = {}
    for seg in P.segments:
        for i, v in enumerate(seg):
            bank[v.id] = i % 2
    ev = [(e[0], 0, e) for e in sim.events if e[2] == "R"]
    ev += [(t, 1, ("P", vid, j)) for (vid, j), t in times.items()]
    ev += [(x[0], 2, ("TW",) + x) for x in writes]
    ev += [(e[0], 3, e) for e in sim.events if e[2] == "W"]
    ev.sort(key=lambda e: (e[0], e[1]))
    by_tile = {T: sorted((x for x in writes if x[1] == T), key=lambda x: x[0]) for T in range(NW)}
    lands = {T: [x[0] for x in by_tile[T]] for T in range(NW)}
    inside = {}
    for (vid, j), t in times.items():
        tile = vecs[vid].passes[j][0]
        if tile >= WT:
            T = tile - WT
            a, b = bisect.bisect_left(lands[T], t), bisect.bisect_right(lands[T], t + 7)
            if b > a:
                inside[(vid, j)] = by_tile[T][a:b]
    unwritten = {vid: sorted(set(range(64)) - set(written_rows(v))) for vid, v in vecs.items()}
    return dict(vecs=vecs, bank=bank, ev=ev, inside=inside, unwritten=unwritten, tiles0=steady_tiles(writes)[0] if writes else None)


def replay(P, plan, feats, seed, adversarial=False):
    """Returns every layer output [L] -> [N, 125, 64], the logits [N, 12] and the number of bit-plane passes."""
    PR = d.P
    N = len(feats)
    rng = np.random.default_rng(seed)
    feats = np.asarray(feats, np.int64).reshape(N, 49, 10)
    vecs, bank = plan["vecs"], plan["bank"]
    mem = {b: np.zeros((N, n, 8), np.int64) for b, n in P.buffers.items()}
    X = rng.integers(-256, 256, size=(2, N, 64))
    tiles = plan["tiles0"].copy() if plan["tiles0"] is not None else None
    pool = np.zeros((N, 64), np.int64)
    logits = np.zeros((N, 12), np.int64)
    out = {L: np.full((N, ss.NPOS, 64), 999, np.int64) for L in range(9)}
    zin = {L: int(PR[f"L{L}_zin"]) for L in range(10)}
    A, R = {}, {}
    planes = 0

    def result(vid, j):
        key = (vid, j)
        if key not in R:
            v = vecs[vid]
            res = v.passes[j][1]
            L = v.L
            cs = np.array([c for (_, _, _, c) in res])
            acc = A.pop(key)[:, :len(cs)]
            if L == 9:
                y = d.mbqm(acc + PR["L9_b"][cs], PR["L9_q"][0], PR["L9_sh"][0]) + int(PR["L9_zout"])
            else:
                x = acc + PR[f"L{L}_b"][cs]
                y = d.mbqm(x, PR[f"L{L}_q"][cs], PR[f"L{L}_sh"][cs]) + int(PR[f"L{L}_zout"])
                y = np.where(x <= 0, -128, y)                        # S's sign gating (zero point -128)
            R[key] = np.clip(y, -128, 127)
        return R[key]

    for ev in plan["ev"]:
        e = ev[2]
        kind = e[2] if ev[1] in (0, 3) else e[0]
        if kind == "R":
            vid, k = e[3], e[4]
            v = vecs[vid]
            b = bank[vid]
            ch = v.chunks[k]
            if v.kind == "conv1":
                val = np.zeros((N, 4), np.int64)
                if ch is not None:
                    oy, ox, fy = ch[1]
                    iy = 2 * oy - 4 + fy
                    for fx in range(4):
                        ix = 2 * ox - 1 + fx
                        if 0 <= ix < 10:
                            val[:, fx] = feats[:, iy, ix] - zin[0]
                X[b][:, 4 * k: 4 * k + 4] = val
            elif v.kind == "fc":
                a = pool[:, k]
                q = np.where(a > 0, (a + 62) // 125, -((-(a - 62)) // 125))
                X[b][:, k] = np.clip(q, -128, 127) - zin[9]
            else:
                n = ss.lanes_of(v.g) if v.kind == "dw" else ss.lanes_of(k)
                X[b][:, 7 * k: 7 * k + n] = 0 if ch is None else mem[ch[0]][:, ch[1], :n] - zin[v.L]
        elif kind == "P":
            vid, j = e[1], e[2]
            v = vecs[vid]
            x = X[bank[vid]].copy()
            if adversarial and plan["unwritten"][vid]:
                u = plan["unwritten"][vid]
                x[:, u] = rng.integers(-256, 256, size=(N, len(u)))
            tile = v.passes[j][0]
            if tile < WT:
                A[(vid, j)] = x @ nm.TILES[tile].astype(np.int64)
            elif (vid, j) not in plan["inside"]:
                A[(vid, j)] = x @ tiles[tile - WT]
            else:                                                     # bit planes, LSB first, tile as in each cycle
                t = ev[0]
                T = tile - WT
                w = tiles[T].copy()
                pend = list(plan["inside"][(vid, j)])
                acc = np.zeros((N, 16), np.int64)
                xu = x & 0x1FF
                for bit in range(9):
                    while pend and pend[0][0] <= t + bit - 1:
                        _, _, r, c, val, _ = pend.pop(0)
                        w[r, :] = 0
                        w[r, c] = val
                    xb = (xu >> bit) & 1
                    acc += (-(1 << 8) if bit == 8 else (1 << bit)) * (xb @ w)
                A[(vid, j)] = acc
                planes += 1
        elif kind == "TW":
            _, land, T, r, c, val, vg = e
            tiles[T][r, :] = 0
            tiles[T][r, c] = val
        else:                                                          # W
            de, vid, j, i = e[3], e[4], e[5], e[6]
            y = result(vid, j)[:, i]
            dests, L, p, c = vecs[vid].passes[j][1][i]
            if de is dests[0] and L <= 8:
                out[L][:, p, c] = y
            if de[0] == "pool":
                pool[:, c] = (0 if p == 0 else pool[:, c]) + y
            elif de[0] == "logit":
                logits[:, c] = y
            else:
                mem[de[0]][:, de[1], de[2]] = y
    return out, logits, planes


def exact(n=None, batch=500, workers=8, out=print):
    fs = ss.test_clips(n)
    cases = {}
    for name, mu, adv in (("S", None, False), ("V", None, False), ("V, adversarial unused rows", None, True),
                          ("S, adversarial unused rows", None, True)) + tuple((f"broken: {m}", m, False) for m in MUTANTS):
        if name.startswith("S"):
            P = ss.make("S")
            sim, _ = ss.simulate(P)
            plan = replay_plan(P, sim, pass_times(sim), [])
        else:
            P, sim, _, times, writes, _, _ = schedule(mu)
            plan = replay_plan(P, sim, times, writes)
        cases[name] = (P, plan, adv)
    tot = {k: dict(clips=0, bad_clips=0, bad_values=0, bad_logit_clips=0, bit_plane_passes=0) for k in cases}
    t0 = time.time()
    with Pool(workers) as pool:
        for b0 in range(0, len(fs), batch):
            rows = pool.map(ss._clip, fs[b0: b0 + batch])
            feats = np.stack([r[0] for r in rows])
            ref = np.stack([r[1] for r in rows])
            ref_y = np.stack([r[2] for r in rows])
            for k, (P, plan, adv) in cases.items():
                o, y, planes = replay(P, plan, feats, seed=1000 + b0, adversarial=adv)
                got = np.stack([o[L] for L in range(9)], axis=1)
                badv = got != ref
                tot[k]["clips"] += len(rows)
                tot[k]["bad_values"] += int(badv.sum()) + int((y != ref_y).sum())
                tot[k]["bad_clips"] += int((badv.reshape(len(rows), -1).any(axis=1) | (y != ref_y).any(axis=1)).sum())
                tot[k]["bad_logit_clips"] += int((y != ref_y).any(axis=1).sum())
                tot[k]["bit_plane_passes"] = planes
            out(f"  {min(b0 + batch, len(fs))} of {len(fs)} clips, {time.time() - t0:.0f} s: "
                + ", ".join(f"{k} {v['bad_clips']}" for k, v in tot.items()))
    return tot


# ---------------------------------------------------------------- pricing
FS_PASSES = 4 * 64 * 8
FS_NEED_BYTES = 4 * 64 * sum(1 for p in range(ss.NPOS) for t in ss.dw_taps(p) if t is not None)
S_DW_ACT_READS = 37960
S_DW_X_ROWS = 4500 * 9 * 7 + 500 * 9
S_DW_LAYER = 11273
FS_DW_LAYER = 512 * 9 + (S_DW_LAYER - 1250 * 9)         # 9 cycles a pass (9 writes, 9 gathers, 9 cycles issue)
V_CTRL_GE = 300
FA_GE = 3 + 2 * 2.4


def dig_ge(and_ge=1.0, wreg=False, ge_ff=6.0):
    ge = (9 * (64 * and_ge + 56 * FA_GE) + 8 * 18 * FA_GE) * 1.2 + 1000
    return ge + (63 * 8 * ge_ff if wreg else 0)


def dw_split(m):
    """S's depthwise array energy (pJ per inference): the per-row part (multipliers, adder tree) and the per-column
    part (accumulators). Colonnade has no split: all of it counts as per-row (the reading kinder to FS)."""
    tot = m.macro_pass_pj("dw")
    if m.a["macro"] == "colonnade":
        return tot, 0.0
    s, g = ee.SHAPES["dw"], m.g
    al = m.alpha["dw"]["alpha"] if m.a["alpha"] == "measured" else m.a["alpha"]
    arr = s["col_passes"] * s["b"] * al * (g["mult_bit"] * 8 * 64 + ee.tree_adders(64, 8) * g["fa"]) / 1000
    acc = s["col_passes"] * s["b"] * (s["b"] + 14) * (g["fa"] + g["dff"]) / 1000
    assert abs(arr + acc - tot) <= 1e-9 * tot
    return arr, acc


def dig_mac_fj(m):
    """One MAC of the digital unit: an 8 x 8 array multiplier (64 AND at ZigZag's 1-bit multiplier energy, 56 full
    adders) and its share of the 9-input tree (8 adders of 18 bits / 9), at the depthwise inputs' toggle rate
    (0.5 under 'alpha 0.25', ZigZag's validation point); Horowitz's 8 x 8 figure under 'multipliers: Horowitz'."""
    g = m.g
    t = m.alpha["dw"]["toggle_rate"] if m.a["alpha"] == "measured" else 2 * m.a["alpha"]
    mult = (ee.HOROWITZ_MUL32_45NM_PJ * 64 / 1024 * int(ee.NODE) / 45 * 1000 if m.a["mult"] == "horowitz"
            else t * (64 * g["mult_bit"] + 56 * g["fa"]))
    return mult + t * 8 * 18 / 9 * g["fa"]


class Pricing:
    def __init__(self, dV):
        self.tools = json.load(open(os.path.join(HERE, "energy_components.json")))
        self.act = json.load(open(os.path.join(HERE, "energy_activity.json")))
        self.reuse = json.load(open(os.path.join(HERE, "reuse_results.json")))
        sf = json.load(open(os.path.join(HERE, "streamfuse_results.json")))
        self.cc = ee.load_counts("cim")
        ee.check_counts(self.cc, "cim")
        self.dS = sf["cycles"]["S"]
        self.sk = sf["exact"]["S"]["sg_skipped_per_inference"]
        self.dV = dV
        self.parts = ee.ww_parts()
        self.variants = ee.VARIANTS + [("4.096 MHz clock", dict(f_sys_hz=4.096e6))]
        assert [v for v, _ in self.variants] == list(self.parts), "the variants of energy_s_rtl.json"

    def model(self, o):
        return ee.Model(self.tools, self.act, self.cc, self.reuse, **o)

    # leakage, uW by part
    def leak(self, m, design, on=True, cell=2.0, fs_ctrl=False, and_ge=1.0, wreg=False):
        lk = ee.sf_leak(m, self.dS, 0.0, True)
        if not on or design == "S":
            return lk
        G = m.a["cim_group"]
        per_macro = lk["CIM macro logic"] / math.ceil(ee.TILES["cim"] / G)
        tile = m.sram_uw("weights", TILE_BITS)
        ge = 0.0
        if design in ("V", "FS"):
            lk["weight cells"] = (29 + NW * cell) * tile
            lk["CIM macro logic"] = per_macro * math.ceil(31 / G)
            lk["other SRAM"] += m.sram_uw("params", STORE_BYTES * 8)
            ge = V_CTRL_GE if (design == "V" or fs_ctrl) else 0
        elif design == "N7":
            lk["weight cells"] = (29 + 40 * 7 / 16) * tile
            lk["CIM macro logic"] = per_macro * (math.ceil(29 / G) + math.ceil(40 / G) * 7 / 16)
        elif design == "DIG":
            lk["weight cells"] = 29 * tile
            lk["CIM macro logic"] = per_macro * math.ceil(29 / G)
            lk["other SRAM"] += m.sram_uw("params", STORE_BYTES * 8)
            ge = dig_ge(and_ge, wreg)
        lk["digital logic"] += ge * m.a["leak_pw_per_ge"] * 1e-6
        return lk

    # energy of one inference, pJ by part
    def energy(self, m, design, on=True, fs_select="active rows", fs_bits=9, fs_gather="per byte", fs_cycles="own",
               v_writes="charged", dig_store_reads=STORE_BYTES):
        mem = m.mem
        if not on or design in ("S", "N7"):
            return ee.sf_network(m, self.dS, self.sk, 0.0)
        if design == "V":
            e = ee.sf_network(m, self.dV, self.sk, 0.0)
            f = 1.0 if v_writes == "charged" else 0.0
            e["rewrites"] = f * STORE_BYTES * mem["weights"]["write_pJ"]
            e["store reads"] = f * STORE_BYTES * mem["params"]["read_pJ"]
            return e
        if design == "FS":
            dF = copy.deepcopy(self.dS)
            dF["counts"]["reads"]["act"] -= S_DW_ACT_READS
            dF["counts"]["x_rows"] += FS_PASSES * 9 - S_DW_X_ROWS
            if fs_cycles == "own":
                dF["total"] -= 4 * (S_DW_LAYER - FS_DW_LAYER)
            e = ee.sf_network(m, dF, self.sk, 0.0)
            arr, acc = dw_split(m)
            if fs_bits == 8:
                arr8 = arr * 8 / 9
                acc8 = 0.0 if acc == 0 else acc * 8 * (8 + 14) / (9 * (9 + 14))
                fs_arr = arr8 * 9 / 63 + acc8
            else:
                fs_arr = arr * 9 / 63 + acc
            e["CIM array"] += fs_arr - (arr + acc)
            sel = m.select_pj("dw")
            e["tile select"] += (sel * 9 / 63 if fs_select == "active rows" else sel) - sel
            e["gathers"] = FS_NEED_BYTES * mem["act_word"]["read_pJ"] / 7 if fs_gather == "per byte" else 0.0
            e["rewrites"] = FS_PASSES * 9 * mem["weights"]["write_pJ"]
            e["store reads"] = FS_PASSES * 9 * mem["params"]["read_pJ"]
            return e
        if design == "DIG":
            e = ee.sf_network(m, self.dS, self.sk, 0.0)
            e["CIM array"] -= m.macro_pass_pj("dw")
            e["tile select"] -= m.select_pj("dw")
            e["digital unit"] = 288000 * dig_mac_fj(m) / 1000
            e["store reads"] = dig_store_reads * mem["params"]["read_pJ"]
            return e
        raise ValueError(design)

    def power(self, vname, design, k, lk_opt=None, e_opt=None, on=True):
        """uW [E]: S's power at this rate, plus the design's leakage and energy per inference against S's."""
        m = self.model(dict(self.variants)[vname])
        q = self.parts[vname]
        base = ee.ww_power(q, 1 / (ee.HOP_S * k), 1 / (ee.HOP_S * k))
        lS, lD = self.leak(m, "S"), self.leak(m, design, on, **(lk_opt or {}))
        eS, eD = self.energy(m, "S"), self.energy(m, design, on, **(e_opt or {}))
        return base + sum(lD.values()) - sum(lS.values()) + (sum(eD.values()) - sum(eS.values())) / (ee.HOP_S * k) * 1e-6


DESIGNS = ("S", "V", "N7", "FS", "DIG")


def price(cyc, out=print):
    pr = Pricing(dict(total=cyc["V"]["total"], counts=cyc["V"]["counts"], words=cyc["V"]["words"]))
    res = {}
    # V3
    pU, p40 = pr.power("baseline", "S", K_U), pr.power("baseline", "S", K_40)
    ww = json.load(open(os.path.join(HERE, "wholeword_energy.json")))
    v3 = abs(pU - 44.11) < 0.005 and abs(p40 - 68.78) < 0.005
    res["V3"] = dict(passed=v3, U_uW=pU, ms40_uW=p40)
    out(f"V3: S at U {pU:.3f} uW, at 40 ms {p40:.3f} uW (gate: 44.11, 68.78): {'pass' if v3 else 'FAIL'}")
    # V4: every design with its change off is S, in every variant and at both rates
    err = max(abs(pr.power(vn, dz, k, on=False) - pr.power(vn, "S", k)) for vn in pr.parts for dz in DESIGNS[1:] for k in (K_U, K_40))
    res["V4"] = dict(passed=err < 1e-9, max_err_uW=err)
    out(f"V4: V, N7, FS, DIG with their change switched off against S, 19 variants, both rates: largest difference "
        f"{err:.1e} uW -> {'pass' if err < 1e-9 else 'FAIL'}")
    # every design, every variant
    table = {vn: {dz: {k: pr.power(vn, dz, k) for k in (K_U, K_40)} for dz in DESIGNS} for vn in pr.parts}
    res["power_uW"] = table
    b = table["baseline"]
    # by part, baseline
    m0 = pr.model({})
    res["parts"] = {dz: dict(leak_uW=pr.leak(m0, dz), energy_pJ=pr.energy(m0, dz)) for dz in DESIGNS}
    out("\nBy part, baseline variant: leakage, uW, and energy of one inference, nJ:")
    lparts = list(res["parts"]["S"]["leak_uW"])
    ee.tab([[p] + [f"{res['parts'][dz]['leak_uW'][p]:.2f}" for dz in DESIGNS] for p in lparts]
           + [["leakage total"] + [f"{sum(res['parts'][dz]['leak_uW'].values()):.2f}" for dz in DESIGNS]]
           + [["energy per inference, nJ"] + [f"{sum(res['parts'][dz]['energy_pJ'].values()) / 1000:.1f}" for dz in DESIGNS]],
           ["part"] + list(DESIGNS))
    allp = sorted({p for dz in DESIGNS for p in res["parts"][dz]["energy_pJ"]}, key=lambda p: list(res["parts"]["FS"]["energy_pJ"]).index(p)
                  if p in res["parts"]["FS"]["energy_pJ"] else 99)
    out("\nEnergy of one inference by part, nJ, baseline variant:")
    ee.tab([[p] + [f"{res['parts'][dz]['energy_pJ'].get(p, 0) / 1000:.2f}" for dz in DESIGNS] for p in allp], ["part"] + list(DESIGNS))
    out("\nTotal power, uW [E], baseline variant:")
    ee.tab([[f"U (k = {K_U})"] + [f"{b[dz][K_U]:.2f}" for dz in DESIGNS], ["40 ms"] + [f"{b[dz][K_40]:.2f}" for dz in DESIGNS]],
           ["rate"] + list(DESIGNS))
    # gates
    g2 = b["V"][K_U] <= 0.80 * b["S"][K_U] and all(t["V"][K_U] < t["S"][K_U] for t in table.values())
    g3 = b["V"][K_U] <= 0.90 * b["N7"][K_U]
    g4 = b["V"][K_U] <= b["FS"][K_U]
    g5 = b["V"][K_U] <= b["DIG"][K_U]
    S_bits = ee.TILES["cim"] * TILE_BITS
    V_bits = 31 * TILE_BITS + STORE_BYTES * 8
    g7 = V_bits <= 0.50 * S_bits
    rat = [t["V"][K_U] / t["S"][K_U] for t in table.values()]
    res["gates"] = dict(
        G2=dict(passed=g2, V=b["V"][K_U], S=b["S"][K_U], ratio=b["V"][K_U] / b["S"][K_U], ratio_range=[min(rat), max(rat)],
                below_S_everywhere=all(t["V"][K_U] < t["S"][K_U] for t in table.values())),
        G3=dict(passed=g3, V=b["V"][K_U], N7=b["N7"][K_U], ratio=b["V"][K_U] / b["N7"][K_U]),
        G4=dict(passed=g4, V=b["V"][K_U], FS=b["FS"][K_U], diff_uW=b["V"][K_U] - b["FS"][K_U]),
        G5=dict(passed=g5, V=b["V"][K_U], DIG=b["DIG"][K_U], diff_uW=b["V"][K_U] - b["DIG"][K_U]),
        G7=dict(passed=g7, V_bits=V_bits, S_bits=S_bits, ratio=V_bits / S_bits))
    gg = res["gates"]
    out(f"\nG2: V {b['V'][K_U]:.2f} against S {b['S'][K_U]:.2f} uW at U = {gg['G2']['ratio']:.3f} x (<= 0.80), "
        f"{min(rat):.3f}..{max(rat):.3f} x over the variants, below S in all: {gg['G2']['below_S_everywhere']} -> {'pass' if g2 else 'FAIL'}")
    out(f"G3: V against N7 {b['N7'][K_U]:.2f} uW = {gg['G3']['ratio']:.3f} x (<= 0.90) -> {'pass' if g3 else 'FAIL'}")
    out(f"G4: V against FS {b['FS'][K_U]:.2f} uW: {gg['G4']['diff_uW']:+.3f} uW (<= 0) -> {'pass' if g4 else 'FAIL'}")
    out(f"G5: V against DIG {b['DIG'][K_U]:.2f} uW: {gg['G5']['diff_uW']:+.3f} uW (<= 0) -> {'pass' if g5 else 'FAIL'}")
    out(f"G7: V's cells and store {V_bits} bits against S's {S_bits} = {V_bits / S_bits:.3f} (<= 0.50) -> {'pass' if g7 else 'FAIL'}")
    # every variant
    out("\nEvery variant at U, uW [E] (V / S in the last column):")
    ee.tab([[vn] + [f"{t[dz][K_U]:.2f}" for dz in DESIGNS] + [f"{t['V'][K_U] / t['S'][K_U]:.3f}"] for vn, t in table.items()],
           ["variant"] + list(DESIGNS) + ["V / S"])
    out("\nEvery variant at 40 ms, uW [E]:")
    ee.tab([[vn] + [f"{t[dz][K_40]:.2f}" for dz in DESIGNS] for vn, t in table.items()], ["variant"] + list(DESIGNS))
    # where G4 and G5 go in the variants (reported, not gated)
    res["G4_G5_by_variant"] = {vn: dict(V_minus_FS=t["V"][K_U] - t["FS"][K_U], V_minus_DIG=t["V"][K_U] - t["DIG"][K_U],
                                        V_over_N7=t["V"][K_U] / t["N7"][K_U]) for vn, t in table.items()}
    # readings of the costing rules (reported, not gated)
    sens = {}
    for label, dz, lo, eo in (("V, working-tile cells at 1 x", "V", dict(cell=1.0), None),
                              ("V, working-tile cells at 3 x", "V", dict(cell=3.0), None),
                              ("V, rewrites and store reads free", "V", None, dict(v_writes="free")),
                              ("V, both kind readings at once", "V", dict(cell=1.0), dict(v_writes="free")),
                              ("FS, tile select not scaled (narrow reading)", "FS", None, dict(fs_select="all rows")),
                              ("FS, 8-bit passes", "FS", None, dict(fs_bits=8)),
                              ("FS, free gathers (lower bound)", "FS", None, dict(fs_gather="free")),
                              ("FS, S's cycles (sequencer not credited)", "FS", None, dict(fs_cycles="S")),
                              ("FS, with V's 300 GE of control", "FS", dict(fs_ctrl=True), None),
                              ("FS, all kind readings at once", "FS", None, dict(fs_bits=8, fs_gather="free")),
                              ("DIG, AND gates at 1.33 GE", "DIG", dict(and_ge=1.33), None),
                              ("DIG, a 63-byte weight register", "DIG", dict(wreg=True), None),
                              ("DIG, weights from the store every cycle (32,000 reads)", "DIG", None, dict(dig_store_reads=32000)),
                              ("DIG, all unkind readings at once", "DIG", dict(and_ge=1.33, wreg=True), dict(dig_store_reads=32000))):
        p = pr.power("baseline", dz, K_U, lo, eo)
        sens[label] = dict(power_U_uW=p, V_minus=b["V"][K_U] - p if dz != "V" else None,
                           minus_DIG=p - b["DIG"][K_U] if dz == "V" else None, minus_FS=p - b["FS"][K_U] if dz == "V" else None)
    res["readings"] = sens
    out("\nReadings of the costing rules, baseline variant at U (not gated; V's readings against the frozen FS and DIG):")
    ee.tab([[k, f"{v['power_U_uW']:.2f}", "" if v["V_minus"] is None else f"{v['V_minus']:+.3f}",
             "" if v["minus_FS"] is None else f"{v['minus_FS']:+.3f} / {v['minus_DIG']:+.3f}"] for k, v in sens.items()],
           ["reading", "uW", "V minus it", "it minus FS / DIG"])
    vd = [t["V"][K_U] - t["DIG"][K_U] for t in table.values()]
    vf = [t["V"][K_U] - t["FS"][K_U] for t in table.values()]
    res["G4_G5_range"] = dict(V_minus_FS=[min(vf), max(vf)], FS_below_V_in=[vn for vn, t in table.items() if t["FS"][K_U] < t["V"][K_U]],
                              V_minus_DIG=[min(vd), max(vd)], DIG_below_V_in=sum(t["DIG"][K_U] < t["V"][K_U] for t in table.values()))
    out(f"\nOver the 19 variants at U: V minus FS {min(vf):+.2f}..{max(vf):+.2f} uW (FS lower in {res['G4_G5_range']['FS_below_V_in']}); "
        f"V minus DIG {min(vd):+.2f}..{max(vd):+.2f} uW (DIG lower in {res['G4_G5_range']['DIG_below_V_in']} of 19)")
    # cycles of the alternatives and storage area (reported)
    tot_fs = S_RTL_TOTAL - 4 * (S_DW_LAYER - FS_DW_LAYER)
    res["cycles"] = dict(S=S_RTL_TOTAL, V=cyc["V"]["total"], N7=S_RTL_TOTAL, FS=tot_fs, DIG=S_RTL_TOTAL)
    mm = m0.mem
    a_w = mm["weights"]["area_um2"] / (mm["weights"]["width"] * mm["weights"]["depth"])
    a_p = mm["params"]["area_um2"] / (mm["params"]["width"] * mm["params"]["depth"])
    area = dict(S=69 * TILE_BITS * a_w, V=(29 + 2 * NW) * TILE_BITS * a_w + STORE_BYTES * 8 * a_p,
                N7=(29 + 40 * 7 / 16) * TILE_BITS * a_w, FS=(29 + 2 * NW) * TILE_BITS * a_w + STORE_BYTES * 8 * a_p,
                DIG=29 * TILE_BITS * a_w + STORE_BYTES * 8 * a_p)
    res["storage_area_mm2"] = {k: v * 1e-6 for k, v in area.items()}
    out(f"\nNetwork cycles per inference: " + ", ".join(f"{k} {v}" for k, v in res["cycles"].items())
        + f" (FS: {FS_DW_LAYER} a depthwise layer, 9 cycles a pass; DIG: gather-bound, 9 ACT words per 7 channels as S)")
    out("CIM storage area, mm^2 at 28 nm (cells and store; 1R1W cells at 2 x area): "
        + ", ".join(f"{k} {v * 1e-6:.4f}" for k, v in area.items()))
    out(f"DIG size {dig_ge():.0f} GE; FS: {FS_PASSES} passes, {FS_PASSES * 9} row writes, {FS_NEED_BYTES} gather bytes per inference")
    return res


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = os.path.join(HERE, "dwswap_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    cmd = sys.argv[1]
    if cmd in ("cycles", "price", "all"):
        cy, _ = cycles()
        res["cycles"] = cy
    if cmd in ("exact", "all"):
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        res["exact"] = ex = exact(n)
        print("exactness:", json.dumps(ex))
        g1 = ex["V"]["bad_values"] == 0 and ex["V, adversarial unused rows"]["bad_values"] == 0 and ex["V"]["clips"] == 4890
        caught = {m: ex[f"broken: {m}"]["bad_values"] > 0 and res.get("cycles", {}).get("mutants_cycle", {}).get(m, {}).get("tile_hazards", 1) > 0
                  for m in MUTANTS}
        res["G1"] = dict(passed=g1 and all(caught.values()), V_exact=g1, caught=caught)
        res["V2"] = dict(passed=ex["S"]["bad_values"] == 0 and ex["S"]["clips"] == 4890)
        print(f"V2: S through this replay, {ex['S']['clips']} clips, {ex['S']['bad_values']} wrong values -> {'pass' if res['V2']['passed'] else 'FAIL'}")
        print(f"G1: V {ex['V']['bad_values']} wrong values on {ex['V']['clips']} clips ({ex['V, adversarial unused rows']['bad_values']} "
              f"with adversarial unused rows); broken variants caught {caught} -> {'pass' if res['G1']['passed'] else 'FAIL'}")
    if cmd in ("price", "all"):
        res["price"] = price(res["cycles"])
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))


if __name__ == "__main__":
    main()
