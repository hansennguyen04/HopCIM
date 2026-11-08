#!/usr/bin/env python3
"""StreamFuse schedule model (docs/STREAMFUSE_GATES.md): pass order, buffer occupancy, cycles and exactness of
today's sequencer (B0) and of the cross-layer options A, B, C, with and without a second requantisation lane.

Two parts:
  cycle model   a register-for-register model of rtl/cim_seq.sv's network phase (gather stages R, G1, G2 with two
                X banks released 7 cycles after their last pass, issue paced by max(9, results), a 4-entry tag queue,
                done at t0 + 14, a 5-stage requantisation pipeline, layer barriers), extended by the options' buffers.
                Every buffer word carries the version (layer, position, group) it holds; reads check it, writes
                check that the old version has no reads left. The fused options add two interlocks that their
                hardware would need: a gather waits until its source word is written, and a vector waits until
                the words it will write are free. B0 never needs them (its layers are separated by barriers).
  data replay   the cycle model's reads, fills and writes, replayed in cycle order on real data for many clips at
                once, through the option's own buffers, the CIM tiles of cim_net_map.py and TFLite's integer
                requantisation; every layer output and the logits are compared with dscnn_int.

Usage: streamfuse_sched.py cycles            cycle model of every design, V1 and the cycle columns
       streamfuse_sched.py selftest          three broken designs must be caught (hazards or wrong values)
       streamfuse_sched.py exact [clips]     data replay on the GSC test clips (default all 4,890): V2 and G1
       streamfuse_sched.py all               both, writes model/streamfuse_results.json
"""
import glob
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import cim_net_map as nm
import dscnn_int as d

HERE = os.path.dirname(os.path.abspath(__file__))
D_ENG = 9                       # issue interval of a 9-bit pass (ENGINE_SPEC.md)
DONE = D_ENG + 5                # done at t0 + D + 5
BANK_HOLD = 7                   # cim_seq: a bank is released 7 cycles after its last pass starts
TQ = 4                          # tag queue
NPOS, NG = 125, 10              # positions 25 x 5; channel groups of 7 (the last holds 1)
TB = 8                          # tbase
START = 1                       # the go reaches the sequencer one cycle after its grant (calibrated on conv1: V1)

# RTL figures for V1 (SEQUENCER.md section 5, model/energy_counts_cim.log over 101 decisions)
RTL_LAYER = [8034] + [11273, 8032] * 4 + [98]
RTL_TOTAL = 85352
RTL_COUNTS = dict(dw_reads=37960, pw_reads=5000, act_byte_writes=72000, dw_chunks=45000, feat_reads=1175,
                  conv1_chunks=1250, passes=7501, requant=72012)


def lanes_of(g):
    return 1 if g == 9 else 7


def ver(L, p, g):
    """Version id of the word a layer L writes at position p, group g."""
    return (L * 128 + p) * 16 + g


def need_mask(buf, v, wide):
    """Lanes a word must hold before it is read: 8 in a fusion vector of 8-byte words, else the group's (7, or 1)."""
    return 0xFF if buf in wide else (1 << lanes_of(v % 16)) - 1


def word_bits(buf, wide):
    return 64 if buf in wide else 56


def ver_str(v):
    return f"L{v // 2048} p{(v // 16) % 128} g{v % 16}"


# ---------------------------------------------------------------- programs
class Vec:
    __slots__ = ("id", "L", "kind", "p", "g", "chunks", "passes", "last")

    def __init__(self, L, kind, p, g):
        self.L, self.kind, self.p, self.g = L, kind, p, g
        self.chunks = []        # (buf, word, version) | ("feat", (oy, ox, fy)) | ("pool", k) | None (padding)
        self.passes = []        # (tile, [(dest list, L, p, c) per result])
        self.last = False


class Program:
    """Segments of vectors separated by barriers, the buffers, the line-buffer fill lists, the read counts."""

    def __init__(self, name, lanes, sg):
        self.name, self.lanes, self.sg = name, lanes, sg
        self.segments = []      # list of lists of Vec
        self.fills = []         # per segment: list of (src buf, src word, version, dst buf, dst word) or []
        self.buffers = {}       # name -> words
        self.reads = {}         # (buf, version) -> number of reads
        self.wide = set()       # buffers of 8-byte words (fusion vectors read in 8 chunks)
        self.sg_thresh = 0


def dw_taps(p):
    oy, ox = divmod(p, 5)
    out = []
    for k in range(9):
        fy, fx = divmod(k, 3)
        iy, ix = oy - 1 + fy, ox - 1 + fx
        out.append((iy, ix) if 0 <= iy < 25 and 0 <= ix < 5 else None)
    return out


def conv1_vec(p, dest):
    v = Vec(0, "conv1", p, 0)
    oy, ox = divmod(p, 5)
    for fy in range(10):
        iy = 2 * oy - 4 + fy
        v.chunks.append(("feat", (oy, ox, fy)) if 0 <= iy < 49 else None)
    v.passes = [(TB + j, [(dest(0, p, 16 * j + i), 0, p, 16 * j + i) for i in range(16)]) for j in range(4)]
    return v


def dw_vec(L, p, g, src, dest):
    v = Vec(L, "dw", p, g)
    for t in dw_taps(p):
        v.chunks.append(None if t is None else src(L - 1, 5 * t[0] + t[1], g))
    n = lanes_of(g)
    v.passes = [(TB + 4 + 14 * ((L - 1) // 2) + g, [(dest(L, p, 7 * g + i), L, p, 7 * g + i) for i in range(n)])]
    return v


def pw_vec(L, p, src, dest, nchunk=10):
    """nchunk 10: ten 7-byte ACT words; 8: a fusion vector read as eight 8-byte chunks (rows 8k .. 8k + 7)."""
    v = Vec(L, "pw", p, 0)
    for k in range(nchunk):
        v.chunks.append(src(L - 1, p, k))
    v.passes = [(TB + 14 + 14 * ((L - 2) // 2) + j, [(dest(L, p, 16 * j + i), L, p, 16 * j + i) for i in range(16)])
                for j in range(4)]
    return v


def fc_vec():
    v = Vec(9, "fc", 0, 0)
    v.chunks = [("pool", k) for k in range(64)]
    v.passes = [(TB + 60, [([("logit", c)], 9, 0, c) for c in range(12)])]
    return v


def build(name, sched, lanes, sg=True, lb=False, l8_act=False, lb_words=None, fv7=False, mutate=None):
    """sched: 'lbl' layer by layer (B0, S, A), 'fuse' DW->PW forwarding per block (B), 'depth' depth-first (C).
    fv7: the fusion vector laid out like ACT, ten 7-byte words (the first model); else eight 8-byte words."""
    P = Program(name, lanes, sg)
    P.mutate = mutate
    P.sg_thresh = 63 if mutate == "sign gate too wide" else 0      # broken: gates results that are not -128
    FVW, FVN = (7, NG) if fv7 else (8, 8)
    if not fv7:
        P.wide = {"fv", "fv1", "fv2", "fv3", "fv4"}

    def act_word(buf, p, g):
        return buf * 1250 + p * 10 + g

    if sched == "lbl":
        P.buffers["act"] = 2500
        if lb:
            P.buffers["lb"] = lb_words

        for L in range(10):
            seg, fills = [], []
            rbuf, wbuf = (0 if L % 2 else 1), (1 if L % 2 else 0)

            def dest(L_, p, c, wbuf=wbuf):
                g, lane = divmod(c, 7)
                out = []
                if L_ <= 8 and (L_ < 8 or l8_act):
                    out.append(("act", act_word(wbuf, p, g), lane, ver(L_, p, g)))
                if L_ == 8:
                    out.append(("pool", c))
                return out

            if L == 0:
                seg = [conv1_vec(p, dest) for p in range(NPOS)]
            elif L == 9:
                seg = [fc_vec()]
            elif L % 2:                                   # depthwise
                if lb:
                    def src(Lp, pin, g):
                        return ("lb", (50 * (pin // 5) + 10 * (pin % 5) + g) % lb_words, ver(Lp, pin, g))
                    fills = [("act", act_word(rbuf, pin, g), ver(L - 1, pin, g), "lb",
                              (50 * (pin // 5) + 10 * (pin % 5) + g) % lb_words) for pin in range(NPOS) for g in range(NG)]
                else:
                    def src(Lp, pin, g, rbuf=rbuf):
                        return ("act", act_word(rbuf, pin, g), ver(Lp, pin, g))
                seg = [dw_vec(L, p, g, src, dest) for p in range(NPOS) for g in range(NG)]
            else:                                         # pointwise
                def src(Lp, p, k, rbuf=rbuf):
                    return ("act", act_word(rbuf, p, k), ver(Lp, p, k))
                seg = [pw_vec(L, p, src, dest) for p in range(NPOS)]
            seg[-1].last = True
            P.segments.append(seg)
            P.fills.append(fills)

    elif sched == "fuse":
        # one map of 26 rows: level l (conv1 = 0, block b's pointwise = b) row y at physical row (y + 1 - l) mod 26,
        # so a pointwise output row y lands on the input row y - 1 that its depthwise no longer needs
        P.buffers["act"] = 26 * 50
        P.buffers["lb"] = lb_words
        P.buffers["fv"] = 2 * FVN

        def map_word(level, p, g):
            y, x = divmod(p, 5)
            lag = 0 if mutate == "no row lag" else level            # broken: every level on the same rows
            return ((y + 1 - lag) % 26) * 50 + 10 * x + g

        def map_dest(L_, p, c):
            g, lane = divmod(c, 7)
            level = 0 if L_ == 0 else L_ // 2
            if L_ == 8:
                return [("pool", c)]
            return [("act", map_word(level, p, g), lane, ver(L_, p, g))]

        P.segments.append([conv1_vec(p, map_dest) for p in range(NPOS)])
        P.segments[-1][-1].last = True
        P.fills.append([])
        for b in range(1, 5):
            Ld, Lp_ = 2 * b - 1, 2 * b
            lev_in = b - 1

            def src_lb(Lprev, pin, g):
                sh = 1 if mutate == "line buffer slot shifted" else 0       # broken: the neighbouring slot
                return ("lb", (50 * (pin // 5) + 10 * (pin % 5) + g + sh) % lb_words, ver(Lprev, pin, g))

            def fv_dest(L_, p, c):
                g, lane = divmod(c, FVW)
                return [("fv", (p % 2) * FVN + g, lane, ver(L_, p, g))]

            def fv_src(Lprev, p, k):
                return ("fv", (p % 2) * FVN + k, ver(Lprev, p, k))

            seg = []
            for p in range(NPOS + 1):
                if p < NPOS:
                    seg += [dw_vec(Ld, p, g, src_lb, fv_dest) for g in range(NG)]
                if p >= 1:
                    seg.append(pw_vec(Lp_, p - 1, fv_src, map_dest, FVN))
            seg[-1].last = True
            P.segments.append(seg)
            P.fills.append([("act", map_word(lev_in, pin, g), ver(Ld - 1, pin, g), "lb",
                             (50 * (pin // 5) + 10 * (pin % 5) + g) % lb_words) for pin in range(NPOS) for g in range(NG)])
        P.segments.append([fc_vec()])
        P.segments[-1][-1].last = True
        P.fills.append([])

    elif sched == "depth":
        # level l = 0..3 (conv1, blocks 1..3's pointwise) in its own line buffer lb<l>, slots by raster order;
        # block 4's pointwise goes to the pool sums. Waves: in wave t, conv1 row t, then block b row t - b.
        for lv in range(4):
            P.buffers[f"lb{lv}"] = lb_words
        for b in range(1, 5):
            P.buffers[f"fv{b}"] = 2 * FVN

        def lb_slot(p, g):
            return (50 * (p // 5) + 10 * (p % 5) + g) % lb_words

        def dest(L_, p, c):
            g, lane = divmod(c, 7)
            if L_ == 8:
                return [("pool", c)]
            if L_ % 2:                                    # depthwise -> its block's fusion vector
                b = (L_ + 1) // 2
                gf, lf = divmod(c, FVW)
                return [(f"fv{b}", (p % 2) * FVN + gf, lf, ver(L_, p, gf))]
            lv = 0 if L_ == 0 else L_ // 2
            return [(f"lb{lv}", lb_slot(p, g), lane, ver(L_, p, g))]

        def dw_src(Lprev, pin, g):
            lv = 0 if Lprev == 0 else Lprev // 2
            return (f"lb{lv}", lb_slot(pin, g), ver(Lprev, pin, g))

        def pw_src(Lprev, p, k):
            b = (Lprev + 1) // 2
            par = (p + 1) % 2 if mutate == "fusion vector swapped" else p % 2   # broken: the other half
            return (f"fv{b}", par * FVN + k, ver(Lprev, p, k))

        seg = []
        for t in range(25 + 4):
            if t <= 24:
                seg += [conv1_vec(5 * t + x, dest) for x in range(5)]
            for b in range(1, 5):
                r = t - b
                if 0 <= r <= 24:
                    Ld, Lp_ = 2 * b - 1, 2 * b
                    for x in range(6):
                        if x < 5:
                            seg += [dw_vec(Ld, 5 * r + x, g, dw_src, dest) for g in range(NG)]
                        if x >= 1:
                            seg.append(pw_vec(Lp_, 5 * r + x - 1, pw_src, dest, FVN))
        seg[-1].last = True
        P.segments.append(seg)
        P.fills.append([])
        P.segments.append([fc_vec()])
        P.segments[-1][-1].last = True
        P.fills.append([])
    else:
        raise ValueError(sched)

    for si, seg in enumerate(P.segments):                 # line-buffer fills in the gather's order of first use
        if P.fills[si]:
            by_ver = {f[2]: f for f in P.fills[si]}
            order = []
            for v in seg:
                for c in v.chunks:
                    if c is not None and c[0] == "lb" and c[2] in by_ver:
                        order.append(by_ver.pop(c[2]))
            assert not by_ver, "a filled word that is never read"
            P.fills[si] = order
    vid = 0
    for seg in P.segments:
        for v in seg:
            v.id = vid
            vid += 1
            for c in v.chunks:
                if c is not None and c[0] not in ("feat", "pool"):
                    P.reads[(c[0], c[2])] = P.reads.get((c[0], c[2]), 0) + 1
    for fl in P.fills:
        for (sb, sw, vv, db, dw_) in fl:
            P.reads[(sb, vv)] = P.reads.get((sb, vv), 0) + 1
    return P


# ---------------------------------------------------------------- cycle model
class Word:
    __slots__ = ("v", "mask", "need", "left", "ready")

    def __init__(self):
        self.v, self.mask, self.need, self.left, self.ready = -1, 0, 0, 0, 0


class Sim:
    def __init__(self, P, limit=400_000, interlock=True):
        self.P, self.limit, self.interlock = P, limit, interlock
        self.mem = {b: [Word() for _ in range(n)] for b, n in P.buffers.items()}
        self.events = []        # data replay: (cycle, phase, kind, ...); phase 0 reads, 1 writes (read-first memories)
        self.hazards = []
        self.wq = {}            # cycle -> version-table writes landing at the end of that cycle
        self.count = dict(reads={}, byte_writes={}, word_writes={}, feat_reads=0, pool_reads=0, pool_updates=0,
                          chunks={}, passes=0, requant=0, x_rows=0, src_stalls=0, dest_stalls=0, fill_stalls=0)
        self.seg_end = []

    def ready(self, buf, word, v, t):
        if not self.interlock:
            return True
        w = self.mem[buf][word]
        return w.v == v and w.mask == w.need and w.ready <= t

    def free(self, buf, word, v):
        if not self.interlock:
            return True
        w = self.mem[buf][word]
        return w.v == v or w.left == 0

    def read(self, buf, word, v, t):
        w = self.mem[buf][word]
        if not (w.v == v and w.mask == w.need and w.ready <= t):
            self.hazards.append(f"t{t}: read {buf}[{word}] wants {ver_str(v)}, holds {ver_str(w.v)} mask {w.mask:b}")
        w.left -= 1
        self.count["reads"][buf] = self.count["reads"].get(buf, 0) + 1

    def write(self, buf, word, lane, v, t):
        w = self.mem[buf][word]
        if w.v != v:
            if w.left > 0:
                self.hazards.append(f"t{t}: write {buf}[{word}] {ver_str(v)} over {ver_str(w.v)} with {w.left} reads left")
            w.v, w.mask, w.need, w.left = v, 0, need_mask(buf, v, self.P.wide), self.P.reads.get((buf, v), 0)
        w.mask |= w.need if lane is None else (1 << lane)
        w.ready = t + 1

    def run(self):
        t = START
        for si, seg in enumerate(self.P.segments):
            t = self.segment(si, seg, self.P.fills[si], t)
            self.seg_end.append(t)
        return t + 1                                     # done_o the cycle after the last layer end

    def segment(self, si, seg, fills, t0):
        """From lay_start in cycle t0 to lay_end; returns the cycle of lay_end."""
        P, lanes = self.P, self.P.lanes
        nv = len(seg)
        g_on, g_b, g_k, gv = True, 0, 0, 0               # gather: vector gv, chunk g_k, bank g_b
        bbusy, bfull, bcnt, vm = [0, 0], [0, 0], [0, 0], [None, None]
        g1 = g2 = None                                    # (vec, chunk, last chunk, bank)
        i_on, i_b, i_j, pace, tq = True, 0, 0, 0, 0
        eng_ready_at = 0
        inflight = []                                     # (done cycle, vec, pass)
        last_write = -1
        drain_end = -1
        fi = 0
        fill_wq = []                                      # (cycle of the line-buffer write, dst buf, dst word, version, k)
        t = t0 + 1
        while True:
            if t - t0 > self.limit:
                raise RuntimeError(f"{P.name}: segment {si} does not finish (deadlock) at vector {gv} of {nv}")
            # ---- lay_end: front done and back idle, from this cycle's registers
            if not g_on and not i_on and tq == 0 and t > last_write and fi == len(fills) and not fill_wq:
                return t
            # ---- done of the oldest pass: its results drain one (or two) per cycle into the 5-stage pipeline
            popped = False
            if inflight and inflight[0][0] == t:
                _, v, j = inflight.pop(0)
                popped = True
                if t <= drain_end:
                    self.hazards.append(f"t{t}: done while the previous results still drain")
                res = v.passes[j][1]
                drain_end = t + math.ceil(len(res) / lanes) - 1
                for i, (dests, L, p, c) in enumerate(res):
                    wc = t + 6 + i // lanes
                    self.count["requant"] += 1
                    for de in dests:
                        if de[0] == "pool":
                            self.count["pool_updates"] += 1
                        elif de[0] != "logit":
                            self.count["byte_writes"][de[0]] = self.count["byte_writes"].get(de[0], 0) + 1
                            self.wq.setdefault(wc, []).append(de)
                        self.events.append((wc, 1, "W", de, v.id, j, i))
                    last_write = max(last_write, wc)
            # ---- issue (combinational)
            issue = False
            v_i = vm[i_b]
            if i_on and v_i is not None:
                full = bfull[i_b] or (g2 is not None and g2[2] and g2[3] == i_b)
                issue = full and t >= eng_ready_at and pace == 0 and tq != TQ
            # ---- gather request (combinational), with the fused options' interlocks
            g_req = False
            if g_on:
                v = seg[gv]
                ok = (g_k != 0) or (not bbusy[g_b] and bcnt[g_b] == 0)
                if ok and g_k == 0:
                    for (tile, res) in v.passes:
                        for (dests, L, p, c) in res:
                            for de in dests:
                                if de[0] not in ("pool", "logit") and not self.free(de[0], de[1], de[3]):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        self.count["dest_stalls"] += 1
                if ok:
                    ch = v.chunks[g_k]
                    if ch is not None and ch[0] not in ("feat", "pool") and not self.ready(ch[0], ch[1], ch[2], t):
                        ok = False
                        self.count["src_stalls"] += 1
                g_req = ok
            # ---- line-buffer fill: one map word per cycle, raster order, into a free slot
            do_fill = None
            if fi < len(fills):
                sb, sw, vv, db, dw_ = fills[fi]
                if self.ready(sb, sw, vv, t) and self.free(db, dw_, vv):
                    do_fill = fills[fi]
                else:
                    self.count["fill_stalls"] += 1
            # ==== end of cycle t. Reads first (read-first memories), then the writes landing now.
            if g_req:
                v = seg[gv]
                ch = v.chunks[g_k]
                self.count["chunks"][v.kind] = self.count["chunks"].get(v.kind, 0) + 1
                self.count["x_rows"] += 4 if v.kind == "conv1" else (1 if v.kind == "fc" else (
                    lanes_of(v.g) if v.kind == "dw" else (8 if len(v.chunks) == 8 else (1 if g_k == 9 else 7))))
                if ch is not None:
                    if ch[0] == "feat":
                        self.count["feat_reads"] += 1
                    elif ch[0] == "pool":
                        self.count["pool_reads"] += 1
                    else:
                        self.read(ch[0], ch[1], ch[2], t)
                self.events.append((t, 0, "R", v.id, g_k))
            if do_fill is not None:
                sb, sw, vv, db, dw_ = do_fill
                self.read(sb, sw, vv, t)
                self.events.append((t, 0, "FR", (sb, sw), (si, fi)))
                fill_wq.append((t + 1, db, dw_, vv, fi))
                self.count["word_writes"][db] = self.count["word_writes"].get(db, 0) + 1
                fi += 1
            for de in self.wq.pop(t, ()):
                self.write(de[0], de[1], de[2], de[3], t)
            while fill_wq and fill_wq[0][0] == t:
                _, db, dw_, vv, k = fill_wq.pop(0)
                self.write(db, dw_, None, vv, t)
                self.events.append((t, 1, "FW", (db, dw_), (si, k)))
            # registers: gather pipeline G2 <- G1 <- R
            if g2 is not None and g2[2]:
                bfull[g2[3]] = 1
            g2, g1 = g1, None
            if g_req:
                v = seg[gv]
                last_c = g_k == len(v.chunks) - 1
                if g_k == 0:
                    bbusy[g_b] = 1
                    vm[g_b] = v
                g1 = (v, g_k, last_c, g_b)
                if last_c:
                    g_k = 0
                    g_b ^= 1
                    gv += 1
                    g_on = gv < nv
                else:
                    g_k += 1
            if pace:
                pace -= 1
            for b in (0, 1):
                if bcnt[b]:
                    bcnt[b] -= 1
            if popped:
                tq -= 1
            if issue:
                tile, res = v_i.passes[i_j]
                self.count["passes"] += 1
                inflight.append((t + DONE, v_i, i_j))
                tq += 1
                eng_ready_at = t + D_ENG
                pace = math.ceil(len(res) / lanes) - 1
                if i_j == len(v_i.passes) - 1:
                    i_j = 0
                    bbusy[i_b] = bfull[i_b] = 0
                    bcnt[i_b] = BANK_HOLD
                    if v_i.last:
                        i_on = False
                    vm[i_b] = None
                    i_b ^= 1
                else:
                    i_j += 1
            t += 1


def simulate(P, interlock=True):
    s = Sim(P, interlock=interlock)
    total = s.run()
    done = [e + 1 for e in s.seg_end]                     # done after each segment, as a stop-after-layer run
    seg_cycles = [done[0]] + [done[i] - done[i - 1] for i in range(1, len(done))]
    bits = {b: n * word_bits(b, P.wide) for b, n in P.buffers.items()}
    return s, dict(total=total, segments=seg_cycles, hazards=len(s.hazards), first_hazards=s.hazards[:5],
                   counts=s.count, words=dict(P.buffers), storage_bits=sum(bits.values()), bits=bits)


# ---------------------------------------------------------------- data replay
def replay(P, sim, feats):
    """Replay the cycle model's reads, fills and writes on real data (N clips at once); returns every computed
    layer output [L] -> [N, 125, 64], the logits [N, 12] and the number of sign-gated requantisations."""
    PR = d.P
    N = len(feats)
    feats = np.asarray(feats, np.int64).reshape(N, 49, 10)
    vecs = {v.id: v for seg in P.segments for v in seg}
    mem = {b: np.zeros((N, n, 8), np.int64) for b, n in P.buffers.items()}
    pool = np.zeros((N, 64), np.int64)
    logits = np.zeros((N, 12), np.int64)
    out = {L: np.full((N, NPOS, 64), 999, np.int64) for L in range(9)}
    X, R, tmp = {}, {}, {}
    zin = {L: int(PR[f"L{L}_zin"]) for L in range(10)}
    skipped = 0

    def result(vid, j):
        nonlocal skipped
        key = (vid, j)
        if key not in R:
            v = vecs[vid]
            tile, res = v.passes[j]
            acc = X[vid] @ nm.TILES[tile].astype(np.int64)            # [N, 16]
            L = v.L
            cs = np.array([c for (_, _, _, c) in res])
            acc = acc[:, :len(cs)]
            if L == 9:
                y = d.mbqm(acc + PR["L9_b"][cs], PR["L9_q"][0], PR["L9_sh"][0]) + int(PR["L9_zout"])
            else:
                x = acc + PR[f"L{L}_b"][cs]
                y = d.mbqm(x, PR[f"L{L}_q"][cs], PR[f"L{L}_sh"][cs]) + int(PR[f"L{L}_zout"])
                if P.sg:                                                # exactly -128 when x <= 0 (zero point -128)
                    assert int(PR[f"L{L}_zout"]) == -128
                    neg = x <= P.sg_thresh
                    skipped += int(neg.sum())
                    y = np.where(neg, -128, y)
            R[key] = np.clip(y, -128, 127)
        return R[key]

    for ev in sorted(sim.events, key=lambda e: (e[0], e[1])):
        kind = ev[2]
        if kind == "R":
            vid, k = ev[3], ev[4]
            v = vecs[vid]
            if vid not in X:
                X[vid] = np.zeros((N, 64), np.int64)
            ch = v.chunks[k]
            if ch is None:
                continue
            if ch[0] == "feat":
                oy, ox, fy = ch[1]
                iy = 2 * oy - 4 + fy
                for fx in range(4):
                    ix = 2 * ox - 1 + fx
                    if 0 <= ix < 10:
                        X[vid][:, 4 * fy + fx] = feats[:, iy, ix] - zin[0]
            elif ch[0] == "pool":
                a = pool[:, k]
                q = np.where(a > 0, (a + 62) // 125, -((-(a - 62)) // 125))
                X[vid][:, k] = np.clip(q, -128, 127) - zin[9]
            elif ch[0] in P.wide:
                X[vid][:, 8 * k: 8 * k + 8] = mem[ch[0]][:, ch[1], :8] - zin[v.L]
            else:
                n = lanes_of(v.g) if v.kind == "dw" else lanes_of(k)
                r0 = 7 * ((k + 1) % 9) if (getattr(P, "mutate", None) == "tap rows shifted" and v.kind == "dw") else 7 * k
                X[vid][:, r0: r0 + n] = mem[ch[0]][:, ch[1], :n] - zin[v.L]
        elif kind == "FR":
            sb, sw = ev[3]
            tmp[ev[4]] = mem[sb][:, sw, :].copy()
        elif kind == "FW":
            db, dw_ = ev[3]
            mem[db][:, dw_, :] = tmp.pop(ev[4])
        elif kind == "W":
            de, vid, j, i = ev[3], ev[4], ev[5], ev[6]
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
    return out, logits, skipped


# ---------------------------------------------------------------- designs
DESIGNS = {
    # name: (schedule, lanes, sign gating, line buffer, L8 written to ACT)
    "B0": ("lbl", 1, False, False, True),
    "S":  ("lbl", 2, True, False, False),
    "A":  ("lbl", 2, True, True, False),
    "B":  ("fuse", 2, True, True, False),
    "C":  ("depth", 2, True, True, False),
    # for information: without the second lane
    "B0+SG": ("lbl", 1, True, False, False),
    "A, 1 lane": ("lbl", 1, True, True, False),
    "B, 1 lane": ("fuse", 1, True, True, False),
    "C, 1 lane": ("depth", 1, True, True, False),
    # the first model: the fusion vector laid out like ACT, ten 7-byte words
    "B, 7-byte fusion vector": ("fuse", 2, True, True, False, True),
    "C, 7-byte fusion vector": ("depth", 2, True, True, False, True),
}
LB_TRY = {"lbl": range(100, 251, 10), "fuse": range(100, 251, 10), "depth": range(100, 251, 10)}


def make(name, lb_words=None):
    sched, lanes, sg, lb, l8, *fv7 = DESIGNS[name]
    return build(name, sched, lanes, sg=sg, lb=lb, l8_act=l8, lb_words=lb_words, fv7=bool(fv7 and fv7[0]))


def size_line_buffer(name):
    """The smallest line buffer (in 56-bit words, per buffer) with no hazard and no cycle cost against a large one."""
    sched = DESIGNS[name][0]
    _, ref = simulate(make(name, 400))
    curve = []
    best = None
    for w in LB_TRY[sched]:
        try:
            _, r = simulate(make(name, w))
        except RuntimeError:
            curve.append((w, None))
            continue
        curve.append((w, r["total"]))
        if best is None and r["hazards"] == 0 and r["total"] <= ref["total"]:
            best = w
    return best, ref["total"], curve


def cycles_all(out=print):
    res = {}
    for name in DESIGNS:
        lbw, big, curve = (None, None, None)
        if DESIGNS[name][3]:
            lbw, big, curve = size_line_buffer(name)
        P = make(name, lbw)
        sim, r = simulate(P)
        r.update(lb_words=lbw, lb_curve=curve, lb_large_total=big)
        res[name] = r
        out(f"{name:10s} cycles {r['total']:6d}  segments {r['segments']}  hazards {r['hazards']}  "
            f"stalls src {r['counts']['src_stalls']} dest {r['counts']['dest_stalls']} fill {r['counts']['fill_stalls']}  "
            f"storage {r['storage_bits']} bits {r['words']}" + (f"  line buffer {lbw} words (curve {curve})" if lbw else ""))
    return res


def v1(r):
    b = r["B0"]
    c = b["counts"]
    seg_ok = all(abs(a - e) <= 0.005 * e for a, e in zip(b["segments"], RTL_LAYER))
    got = dict(dw_reads=c["reads"]["act"] - 5000 if c["chunks"].get("pw") == 5000 else None, pw_reads=c["chunks"]["pw"],
               act_byte_writes=c["byte_writes"]["act"], dw_chunks=c["chunks"]["dw"], feat_reads=c["feat_reads"],
               conv1_chunks=c["chunks"]["conv1"], passes=c["passes"], requant=c["requant"])
    counts_ok = got == RTL_COUNTS and c["reads"]["act"] == 42960
    return dict(segments=b["segments"], rtl=RTL_LAYER, total=b["total"], rtl_total=RTL_TOTAL,
                exact_layers=sum(a == e for a, e in zip(b["segments"], RTL_LAYER)), counts=got,
                passed=seg_ok and counts_ok and abs(b["total"] - RTL_TOTAL) <= 0.005 * RTL_TOTAL)


# ---------------------------------------------------------------- reference data
def _clip(p):
    import exact_reuse as er
    import mfcc_int as mi
    q = mi.cim_mfcc_int(er.read_wav(p))
    y, acts = d.forward(q, keep=True)
    return q.reshape(49, 10), np.stack([a.reshape(NPOS, 64) for a in acts[:9]]).astype(np.int16), np.asarray(y, np.int16)


def test_clips(n=None):
    import exact_reuse as er
    fs = sorted(glob.glob(os.path.join(er.TEST, "*", "*.wav")))
    if n is not None and n < len(fs):
        fs = fs[:: max(1, len(fs) // n)][:n]
    return fs


def exact_all(names, n=None, batch=500, workers=8, out=print):
    fs = test_clips(n)
    progs = {nm_: make(nm_, lbw) for nm_, lbw in names.items()}
    sims = {k: simulate(P)[0] for k, P in progs.items()}
    tot = {k: dict(clips=0, bad_clips=0, bad_values=0, bad_logit_clips=0, skipped=0) for k in names}
    t0 = time.time()
    with Pool(workers) as pool:
        for b0 in range(0, len(fs), batch):
            rows = pool.map(_clip, fs[b0: b0 + batch])
            feats = np.stack([r[0] for r in rows])
            ref = np.stack([r[1] for r in rows])                       # [N, 9, 125, 64]
            ref_y = np.stack([r[2] for r in rows])
            for k, P in progs.items():
                o, y, sk = replay(P, sims[k], feats)
                got = np.stack([o[L] for L in range(9)], axis=1)       # [N, 9, 125, 64]
                badv = (got != ref)
                badc = badv.reshape(len(rows), -1).any(axis=1) | (y != ref_y).any(axis=1)
                tot[k]["clips"] += len(rows)
                tot[k]["bad_values"] += int(badv.sum()) + int((y != ref_y).sum())
                tot[k]["bad_clips"] += int(badc.sum())
                tot[k]["bad_logit_clips"] += int((y != ref_y).any(axis=1).sum())
                tot[k]["skipped"] += sk
            out(f"  {min(b0 + batch, len(fs))} of {len(fs)} clips, {time.time() - t0:.0f} s: "
                + ", ".join(f"{k} {v['bad_clips']} bad" for k, v in tot.items()))
    for k, v in tot.items():
        v["sg_skipped_per_inference"] = v["skipped"] / max(v["clips"], 1)
    return tot


def self_test(n=50):
    """The checks must be able to fail. Each broken design must be caught: by a deadlock or hazards in the cycle
    model (its version checks), or by wrong values in the data replay. Also: the correct B with its source
    interlock removed (its 24 early reads must show), and B without the one-row lag (a finding, not a fault)."""
    fs = test_clips(n)
    with Pool(8) as pool:
        rows = pool.map(_clip, fs)
    feats = np.stack([r[0] for r in rows])
    ref = np.stack([r[1] for r in rows])
    ref_y = np.stack([r[2] for r in rows])

    def check(P, interlock=True):
        try:
            sim, r = simulate(P, interlock=interlock)
        except RuntimeError:
            return dict(result="deadlock", hazards=None, bad_values=None, cycles=None)
        o, y, _ = replay(P, sim, feats)
        got = np.stack([o[L] for L in range(9)], axis=1)
        bad = int((got != ref).sum()) + int((y != ref_y).sum())
        return dict(result="ran", hazards=r["hazards"], bad_values=bad, cycles=r["total"])

    out = {}
    for name, sched, lbw, mut in (("S", "lbl", None, "sign gate too wide"), ("S", "lbl", None, "tap rows shifted"),
                                  ("C", "depth", 150, "fusion vector swapped"), ("B", "fuse", 120, "line buffer slot shifted")):
        P = build(f"{name}, {mut}", sched, 2, sg=True, lb=(sched != "lbl"), lb_words=lbw, mutate=mut)
        c = check(P)
        c["caught"] = c["result"] == "deadlock" or c["hazards"] > 0 or c["bad_values"] > 0
        out[mut] = c
        print(f"  {name}, {mut}: {c['result']}, hazards {c['hazards']}, wrong values {c['bad_values']} -> "
              f"{'caught' if c['caught'] else 'MISSED'}")
    c = check(make("B", 120), interlock=False)
    out["B without its source interlock"] = c
    print(f"  B without its source interlock: hazards {c['hazards']}, wrong values {c['bad_values']} (the interlock is needed)")
    c = check(build("B, 25-row map", "fuse", 2, sg=True, lb=True, lb_words=120, mutate="no row lag"))
    out["B with a 25-row map"] = c
    print(f"  B with a 25-row map (no row lag): hazards {c['hazards']}, wrong values {c['bad_values']}, cycles {c['cycles']}")
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = os.path.join(HERE, "streamfuse_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if sys.argv[1] in ("cycles", "all"):
        cy = cycles_all()
        res["cycles"] = cy
        res["V1"] = v1(cy)
        print("V1:", res["V1"])
    if sys.argv[1] in ("selftest", "all"):
        res["self_test"] = self_test()
    if sys.argv[1] in ("exact", "all"):
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        names = {k: (res["cycles"][k]["lb_words"] if "cycles" in res else None) for k in ("B0", "S", "A", "B", "C")}
        res["exact"] = exact_all(names, n)
        print("exactness:", res["exact"])
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o if not isinstance(o, (np.integer,)) else int(o))


if __name__ == "__main__":
    main()
