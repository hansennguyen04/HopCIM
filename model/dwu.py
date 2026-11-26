#!/usr/bin/env python3
"""Digital depthwise unit (DWU), the design of docs/DWU_GATES.md: predictions made before its RTL.

The unit sits beside the sequencer's gather. In a depthwise layer it runs on (D1: L1 only; D4: L1, L3, L5, L7) the
layer goes group by group (for g: for p), and each gathered chunk (tap k, 7 channels, x = act - zp, 0 for padding)
is multiplied by the 7 weights of tap k of the group and added into the 7 accumulators of the chunk's X bank. The
gather no longer writes the engine's X bank; the pass of a depthwise vector goes to the unit, which needs no
engine spacing and gives its 7 sums one cycle after the pass is issued (done = issue + 1). Weights: a 63-byte
register (9 taps x 7 lanes), filled from a 360 x 56-bit store: all 9 taps of the first group when the network
starts, and tap k of the next group when chunk k of the group's last vector (p = 124) has been multiplied.

Usage: dwu.py            cycles and counts of S, D1 and D4 (writes model/dwu_prediction.json; frozen with the gates)
       dwu.py rtl        after the RTL: R2 (bench cycles, energy counters), R4 (size), R5 (power from the RTL's own
                         counters); needs sim/logs/seq_tb_cim_d{1,4}.log, model/energy_counts_cim_{s,d4}.log and the two
                         out-of-context runs of syn/ooc.tcl; writes model/dwu_rtl.json
"""
import json
import math
import os

import dwswap as dw
import energy_est as ee
import streamfuse_sched as ss

HERE = os.path.dirname(os.path.abspath(__file__))
DONE_DWU = 1
STORE_WORDS, STORE_W = 360, 56
LANES = 7
ACC_BITS = 20                                 # |sum| <= 9 * 256 * 128 = 294,912 < 2^19
S_DW_X_ROWS = dw.S_DW_X_ROWS


class DwuSim(ss.Sim):
    """S's cycle model; in the unit's segments a pass is done one cycle after its issue and needs no engine gap."""

    def __init__(self, P, dwu_segs, **kw):
        super().__init__(P, **kw)
        self.dwu_segs = dwu_segs

    def segment(self, si, seg, fills, t0):
        if si not in self.dwu_segs:
            return super().segment(si, seg, fills, t0)
        save = ss.DONE, ss.D_ENG
        ss.DONE, ss.D_ENG = DONE_DWU, 1
        try:
            return super().segment(si, seg, fills, t0)
        finally:
            ss.DONE, ss.D_ENG = save


def program(layers):
    P = ss.build(f"DWU {layers}", "lbl", 2, sg=True)
    for L in layers:
        seg = sorted(P.segments[L], key=lambda v: (v.g, v.p))
        for v in seg:
            v.last = False
        seg[-1].last = True
        P.segments[L] = seg
    vid = 0
    for seg in P.segments:
        for v in seg:
            v.id = vid
            vid += 1
    return P


def simulate(layers):
    P = program(layers)
    s = DwuSim(P, set(layers))
    total = s.run()
    done = [e + 1 for e in s.seg_end]
    segs = [done[0]] + [done[i] - done[i - 1] for i in range(1, len(done))]
    return P, s, dict(total=total, segments=segs, hazards=len(s.hazards), counts=s.count, words=dict(P.buffers))


def counts(layers, r):
    n = len(layers)
    c = r["counts"]
    groups = 10 * n
    return dict(
        cycles=r["total"], segments=r["segments"],
        dwu_passes=1250 * n, dwu_chunks=11250 * n, dwu_macs=288000 // 4 * n,            # 72,000 a layer (7 or 1 lanes)
        store_reads=9 + 9 * groups, wreg_writes=9 + 9 * groups,                          # first group + each next group
        engine_passes=7501 - 1250 * n, act_reads=c["reads"]["act"], act_byte_writes=c["byte_writes"]["act"],
        x_rows=c["x_rows"] - S_DW_X_ROWS // 4 * n, requant=c["requant"])


# ---------------------------------------------------------------- size, gate equivalents (analytical)
FA, AND, MUX2, FF = dw.FA_GE, 1.0, 2.33, 6.0


def size_ge():
    """By construction: 7 array multipliers of 9 x 8 bits (72 AND, 64 FA), 7 accumulator adders of 20 bits, two
    banks of 7 x 20-bit accumulators (clear and enable: 1.5 GE a bit; a 2:1 bank read mux), the 504-bit weight
    register and its 9:1 read mux of 56 bits, the loader and the sequencer's changes."""
    parts = dict(multipliers=7 * (72 * AND + 64 * FA), adders=7 * ACC_BITS * FA,
                 accumulators=2 * 7 * ACC_BITS * (FF + 1.5) + 7 * ACC_BITS * MUX2,
                 weight_register=9 * 56 * FF + 8 * 56 * MUX2, loader_and_control=40 * FF + 400)
    return parts, sum(parts.values())


def mac_fj(m, t):
    """A MAC of the unit: the 9 x 8 multiplier and the 20-bit add at the inputs' toggle rate, the accumulator
    register at half that (the study's register convention)."""
    g = m.g
    mult = (ee.HOROWITZ_MUL32_45NM_PJ * 72 / 1024 * int(ee.NODE) / 45 * 1000 if m.a["mult"] == "horowitz"
            else t * (72 * g["mult_bit"] + 64 * g["fa"]))
    return mult + t * ACC_BITS * g["fa"] + ACC_BITS * g["dff"] * 0.5 * t


def price(pr, cnt, ge, vname="baseline", k=dw.K_U, n_layers=4):
    """uW [E] with the study's rules (dwswap.Pricing): S's power at this rate plus the unit's leakage and energy
    against S's. The depthwise tiles of the unit's layers leave the CIM."""
    m = pr.model(dict(pr.variants)[vname])
    q = pr.parts[vname]
    base = ee.ww_power(q, 1 / (ee.HOP_S * k), 1 / (ee.HOP_S * k))
    lS = pr.leak(m, "S")
    lD = dict(lS)
    G = m.a["cim_group"]
    tiles = 69 - 10 * n_layers
    tile = m.sram_uw("weights", dw.TILE_BITS)
    lD["weight cells"] = tiles * tile
    lD["CIM macro logic"] = lS["CIM macro logic"] / math.ceil(69 / G) * math.ceil(tiles / G)
    lD["other SRAM"] += m.sram_uw("params", STORE_WORDS * STORE_W)
    lD["digital logic"] += ge * m.a["leak_pw_per_ge"] * 1e-6
    eS = pr.energy(m, "S")
    d = dict(pr.dS)
    d = json.loads(json.dumps(pr.dS))
    d["total"] = cnt["cycles"]
    d["counts"]["x_rows"] = cnt["x_rows"]
    eD = ee.sf_network(m, d, pr.sk, 0.0)
    frac = n_layers / 4
    eD["CIM array"] -= frac * m.macro_pass_pj("dw")
    eD["tile select"] -= frac * m.select_pj("dw")
    t = m.alpha["dw"]["toggle_rate"] if m.a["alpha"] == "measured" else 2 * m.a["alpha"]
    eD["digital unit"] = cnt["dwu_macs"] * mac_fj(m, t) / 1000
    eD["store reads"] = cnt["store_reads"] * m.mem["params"]["read_pJ"]
    eD["weight register"] = cnt["wreg_writes"] * 56 * m.g["dff"] * 0.5 / 1000
    p = base + sum(lD.values()) - sum(lS.values()) + (sum(eD.values()) - sum(eS.values())) / (ee.HOP_S * k) * 1e-6
    return p, lD, eD


# ---------------------------------------------------------------- after the RTL (docs/DWU_GATES.md R2, R4, R5)
ND = 101                                          # decisions of the energy-count run (tb_cim_mic, divider 5, k = 1)
SYN = os.path.join(HERE, "..", "syn")
OOC_D4 = "ooc_cim_endpoint_FE_DED0_PDM_CLKDIV15_SEQ_S1_SEQ_DWU4"
OOC_S = "ooc_cim_endpoint_FE_DED0_PDM_CLKDIV15_SEQ_S1_SEQ_DWU0"      # S on the same source


def utilisation(d):
    import re
    txt = open(os.path.join(SYN, d, "utilization.rpt")).read()
    def g(label):
        m = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)\s*\|", txt, re.M)
        return float(m[1])
    return dict(lut_logic=g("LUT as Logic"), lutram=g("LUT as Distributed RAM"), srl=g("LUT as Shift Register"),
                ff=g("Register as Flip Flop"), ramb36=g("RAMB36/FIFO*"), ramb18=g("RAMB18"), dsp=g("DSPs"))


def bench_cycles(log):
    import re
    txt = open(os.path.join(HERE, "..", "sim", "logs", log)).read()
    stops = [int(x) for x in re.findall(r"^layer L\d \(stop after it\): \d+ / 8000 mismatches, (\d+) cycles from go", txt, re.M)]
    full = int(re.search(r"full inference (\d+)\.\.(\d+) cycles", txt)[1])
    assert re.search(r"^PASS", txt, re.M)
    pts = stops + [full]
    return full, [pts[0]] + [pts[i] - pts[i - 1] for i in range(1, len(pts))]


def rtl_d(c, cyc):
    """A design's counts per inference from its RTL counters, as energy_est.s_rtl builds S's."""
    return dict(total=cyc, words={"act": 2500},
                counts=dict(reads={"act": (c["seq.gather_tv_lt1"] + c["seq.gather_tv_lt2"]) / ND},
                            byte_writes={"act": c["seq.act_writes"] / ND}, word_writes={},
                            feat_reads=c["seq.gather_tv_lt0"] / ND, requant=c["seq.requant"] / ND,
                            x_rows=(c["engine.xw_rows"] - c["seq.lraw_ring_reads"]) / ND))


def rtl(out=print):
    import re
    pred = json.load(open(os.path.join(HERE, "dwu_prediction.json")))
    p4 = pred["D4"]
    res = {}
    # R2: cycles by layer (sequencer bench) and counters (live path) against the frozen prediction
    for name, log in (("D1", "seq_tb_cim_d1.log"), ("D4", "seq_tb_cim_d4.log")):
        full, segs = bench_cycles(log)
        res[f"{name}_cycles"] = dict(rtl=full, layers=segs, model=pred[name]["cycles"], model_layers=pred[name]["segments"],
                                     equal=(full == pred[name]["cycles"] and segs == pred[name]["segments"]))
        out(f"{name}: RTL {full} cycles, layers {segs}; model {pred[name]['cycles']}, {pred[name]['segments']} -> "
            f"{'equal' if res[name + '_cycles']['equal'] else 'DIFFERENT'}")
    cd, cs = ee.load_counts("cim_d4"), ee.load_counts("cim_s")
    ee.check_counts(ee.load_counts("cim"), "cim")
    net = lambda c: sum(v for k, v in c.items() if re.match(r"engine\.pass_t\d+_b9", k)) / ND
    dw_tiles = [k for k in cd if re.match(r"engine\.pass_t(\d+)_b9", k) and (int(re.match(r"engine\.pass_t(\d+)", k)[1]) - 12) % 14 < 10
                and 12 <= int(re.match(r"engine\.pass_t(\d+)", k)[1]) <= 63]
    got = dict(dwu_passes=cd["seq.dwu_passes"] / ND, dwu_chunks=cd["seq.dwu_chunks"] / ND, dwu_macs=cd["seq.dwu_macs"] / ND,
               store_reads=cd["seq.dwu_store_reads"] / ND, engine_passes=net(cd),
               act_reads=(cd["seq.gather_tv_lt1"] + cd["seq.gather_tv_lt2"]) / ND, act_byte_writes=cd["seq.act_writes"] / ND,
               x_rows=(cd["engine.xw_rows"] - cd["seq.lraw_ring_reads"]) / ND, requant=cd["seq.requant"] / ND)
    want = {k: p4[k] for k in got}
    saved = (cs["seq.run_cycles"] - cd["seq.run_cycles"]) / ND
    full4 = res["D4_cycles"]["rtl"]
    ok_counts = got == want and not dw_tiles and saved == dw.S_RTL_TOTAL - full4
    res["counters"] = dict(rtl=got, model=want, equal=ok_counts, dw_tile_passes=dw_tiles, run_cycles_saved=saved)
    out("counters per inference (RTL / model): " + ", ".join(f"{k} {got[k]:g}/{want[k]:g}" for k in got)
        + f"; engine passes on depthwise tiles: {len(dw_tiles)}; run cycles saved against S {saved:g} "
        f"(benches {dw.S_RTL_TOTAL - full4}) -> {'equal' if ok_counts else 'DIFFERENT'}")
    r2 = all(res[f"{n}_cycles"]["equal"] for n in ("D1", "D4")) and ok_counts and full4 <= 1.02 * dw.S_RTL_TOTAL
    res["R2"] = dict(passed=r2, limit=1.02 * dw.S_RTL_TOTAL)
    out(f"R2: {'pass' if r2 else 'FAIL'} (network {full4} <= {1.02 * dw.S_RTL_TOTAL:.0f}, equal to the model)")
    # R4: size by both counts
    u4, u0 = utilisation(OOC_D4), utilisation(OOC_S)
    du = {k: u4[k] - u0[k] for k in u4}
    ge_ooc = 6 * (du["lut_logic"] + du["srl"]) + 6 * du["ff"] + 1500 * du["dsp"]
    parts, ge_c = size_ge()
    comb = ge_c - (2 * 7 * ACC_BITS + 9 * 56 + 40) * FF               # by construction without its flip-flops
    ge_cons = comb + 6 * du["ff"]
    ge = max(ge_ooc, ge_cons)
    r4 = ge <= 15900
    res["R4"] = dict(passed=r4, ooc_delta=du, ge_ooc=ge_ooc, ge_construction=ge_cons, ge=ge, d4=u4, s=u0)
    out(f"R4: Vivado out of context, D4 against S on the same source: {du['lut_logic']:+.0f} logic LUT, {du['lutram']:+.0f} LUTRAM, "
        f"{du['srl']:+.0f} SRL, {du['ff']:+.0f} FF, {du['dsp']:+.0f} DSP, {du['ramb36']:+g} RAMB36, {du['ramb18']:+g} RAMB18 -> "
        f"{ge_ooc:.0f} GE; by construction with the synthesised flip-flops {ge_cons:.0f} GE -> {ge:.0f} <= 15,900: "
        f"{'pass' if r4 else 'FAIL'}")
    # R5: power from the RTL's counters, with the larger size
    pr = dw.Pricing(dict(total=dw.S_RTL_TOTAL, counts={}, words={}))
    tab = {}
    for vname, o in pr.variants:
        m = pr.model(o)
        q = pr.parts[vname]
        dS_, dD_ = rtl_d(cs, dw.S_RTL_TOTAL), rtl_d(cd, full4)
        eS = ee.sf_network(m, dS_, cs["seq.requant_gated"] / ND, 0.0)
        eD = ee.sf_network(m, dD_, cd["seq.requant_gated"] / ND, 0.0)
        eD["CIM array"] -= m.macro_pass_pj("dw")
        eD["tile select"] -= m.select_pj("dw")
        t = m.alpha["dw"]["toggle_rate"] if m.a["alpha"] == "measured" else 2 * m.a["alpha"]
        eD["digital unit"] = got["dwu_macs"] * mac_fj(m, t) / 1000
        eD["store reads"] = got["store_reads"] * m.mem["params"]["read_pJ"]
        eD["weight register"] = got["store_reads"] * 56 * m.g["dff"] * 0.5 / 1000
        lS = pr.leak(m, "S")
        lD = dict(lS)
        G = m.a["cim_group"]
        lD["weight cells"] = 29 * m.sram_uw("weights", dw.TILE_BITS)
        lD["CIM macro logic"] = lS["CIM macro logic"] / math.ceil(69 / G) * math.ceil(29 / G)
        lD["other SRAM"] += m.sram_uw("params", STORE_WORDS * STORE_W)
        lD["digital logic"] += ge * m.a["leak_pw_per_ge"] * 1e-6
        row = {}
        for k in (dw.K_U, dw.K_40):
            base = ee.ww_power(q, 1 / (ee.HOP_S * k), 1 / (ee.HOP_S * k))
            row[k] = dict(S=base, D4=base + sum(lD.values()) - sum(lS.values()) + (sum(eD.values()) - sum(eS.values())) / (ee.HOP_S * k) * 1e-6)
        tab[vname] = row
        if vname == "baseline":
            res["parts"] = dict(energy_S_pJ=eS, energy_D4_pJ=eD, leak_S_uW=lS, leak_D4_uW=lD)
    b = tab["baseline"]
    V = 28.1323
    r5 = b[dw.K_U]["D4"] <= V
    rng = [t[dw.K_U]["D4"] for t in tab.values()]
    res["R5"] = dict(passed=r5, D4_U=b[dw.K_U]["D4"], S_U=b[dw.K_U]["S"], V_U=V, D4_40=b[dw.K_40]["D4"], S_40=b[dw.K_40]["S"],
                     U_range=[min(rng), max(rng)], below_S_everywhere=all(t[dw.K_U]["D4"] < t[dw.K_U]["S"] for t in tab.values()),
                     by_variant=tab)
    eS_t, eD_t = sum(res["parts"]["energy_S_pJ"].values()), sum(res["parts"]["energy_D4_pJ"].values())
    out(f"R5: D4 from its RTL counters, {ge:.0f} GE: {b[dw.K_U]['D4']:.2f} uW at U (S {b[dw.K_U]['S']:.2f}, swap-in {V:.2f}; "
        f"{b[dw.K_U]['D4'] / b[dw.K_U]['S']:.3f} x S), {b[dw.K_40]['D4']:.2f} at 40 ms (S {b[dw.K_40]['S']:.2f}); "
        f"{min(rng):.2f}..{max(rng):.2f} over the variants; energy {eD_t / 1000:.1f} nJ (S {eS_t / 1000:.1f}), leakage "
        f"{sum(res['parts']['leak_D4_uW'].values()):.2f} uW (S {sum(res['parts']['leak_S_uW'].values()):.2f}) -> {'pass' if r5 else 'FAIL'}")
    ee.tab([[vn, f"{t[dw.K_U]['S']:.2f}", f"{t[dw.K_U]['D4']:.2f}", f"{t[dw.K_U]['D4'] / t[dw.K_U]['S']:.3f}",
             f"{t[dw.K_40]['S']:.2f}", f"{t[dw.K_40]['D4']:.2f}"] for vn, t in tab.items()],
           ["variant", "S at U", "D4 at U", "D4 / S", "S at 40 ms", "D4 at 40 ms"])
    with open(os.path.join(HERE, "dwu_rtl.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rtl":
        rtl()
        return
    out = {}
    _, sS, rS = simulate(())
    assert rS["total"] == dw.S_RTL_TOTAL and rS["segments"] == dw.S_RTL_LAYER, "V: the model with no unit is S"
    out["S"] = dict(cycles=rS["total"], segments=rS["segments"])
    print(f"S (no unit): {rS['total']} cycles, layers {rS['segments']} (RTL {dw.S_RTL_LAYER})")
    sz, ge = size_ge()
    out["size_GE"] = dict(parts=sz, total=ge)
    print(f"size by construction: {ge:.0f} GE: " + ", ".join(f"{k} {v:.0f}" for k, v in sz.items()))
    pr = dw.Pricing(dict(total=rS["total"], counts=rS["counts"], words=rS["words"]))
    for name, layers in (("D1", (1,)), ("D4", (1, 3, 5, 7))):
        _, s, r = simulate(layers)
        assert r["hazards"] == 0
        c = counts(layers, r)
        worst = dw.S_WORST_HOP - dw.S_RTL_TOTAL + r["total"]
        c["worst_hop"] = worst
        pU, lD, eD = price(pr, c, ge, n_layers=len(layers))
        p40, _, _ = price(pr, c, ge, k=dw.K_40, n_layers=len(layers))
        rng = [price(pr, c, ge, vn, n_layers=len(layers))[0] for vn in pr.parts]
        c.update(power_U_uW=pU, power_40_uW=p40, power_U_range=[min(rng), max(rng)],
                 leak_uW=sum(lD.values()), energy_nJ=sum(eD.values()) / 1000,
                 break_even_GE_vs_V=ge + (28.1323 - pU) / 50e-6 if name == "D4" else None)
        out[name] = c
        print(f"{name}: {r['total']} cycles, layers {r['segments']}; worst hop {worst}; {c['dwu_passes']} unit passes, "
              f"{c['dwu_macs']} MACs, {c['store_reads']} store reads; {c['engine_passes']} engine passes; ACT reads "
              f"{c['act_reads']}, X rows {c['x_rows']}; power [E] at U {pU:.2f} uW (variants {min(rng):.2f}..{max(rng):.2f}), "
              f"40 ms {p40:.2f}; energy {c['energy_nJ']:.1f} nJ, leakage {c['leak_uW']:.2f} uW")
    with open(os.path.join(HERE, "dwu_prediction.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
