#!/usr/bin/env python3
"""Energy estimate of HopCIM from exact activity and published models (docs/ENERGY_ESTIMATE.md).

A model-based estimate at 28 nm, not a measurement: nothing here comes from FPGA power. Three kinds of
input, kept apart:

  counted    energy_counts_{cim,ded}.log   RTL simulation of the live path, sim/run_energy_counts.sh
             energy_activity.json          bit activity on real audio, energy_activity.py
             reuse_results.json            activation reuse, exact_reuse.py
  published  energy_components.json        CACTI, Aladdin, AccelForge's Colonnade, energy_tool_run.py
             ZIGZAG below                  the digital CIM equations and 28 nm fit of ZigZag-IMC
                                           (Houshmand, Sun, Verhelst, arXiv 2305.18335), transcribed from
                                           its repository, checked against its chip figures in chips()
  assumed    ASSUME below                  what no tool here gives; sensitivity() varies each one

Usage: energy_est.py               prints the tables, writes energy_results.json
       energy_est.py streamfuse    the StreamFuse designs of docs/STREAMFUSE_GATES.md, streamfuse_energy.json
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOP_S = 0.02
NODE = "28"

# ZigZag-IMC, imc_model_validation/dimc_validation/28nm: fitted 28 nm gate figures (fF, V, um^2)
ZIGZAG = {"vdd": 0.9, "nd2": 0.7, "xor2": 0.7 * 1.5, "dff": 0.7 * 3, "wl": 0.35, "bl": 0.35, "nd2_um2": 0.614}
# Published chip parameters and figures from the same script, for the transcription check
CHIPS = {
    "ISSCC 2022 15.5": dict(reported=36.63, cycles=4, banks=64, out=6, inp=32, mult_bits=8, mult_n=32, tree_bits=9,
                            acc=32, reg=32, ops=2 * 64 * 6 * 32, per_bank_acc=False, channel_tree=None),
    "ISSCC 2023 7.2": dict(reported=19.5, cycles=4, banks=8, out=8, inp=128, mult_bits=1, mult_n=256, tree_bits=2,
                           acc=23, reg=23, ops=2 * 8 * 8 * 128 / 8, per_bank_acc=True, channel_tree=(16, 8)),
}
# Horowitz, ISSCC 2014: 32-bit integer multiply at 45 nm, 0.9 V, pJ (the alternative to Aladdin)
HOROWITZ_MUL32_45NM_PJ = 3.1
# HEEPocrates (X-HEEP in TSMC 65 nm LP), EPFL: 28 uW/MHz at 1.2 V, whole chip running a matrix product.
# Used unscaled, so it is an upper bound for a CV32E20 at 28 nm.
CPU_PJ_PER_CYCLE = 28.0
CPU_CYCLES_PER_DECISION = 263                         # measured on the board (EVALUATION.md section 2)
# Measured 28 nm silicon, for comparison with CACTI: 8T macro in FD-SOI, retention (IEEE S3S 2018)
SILICON_RETENTION_PW_PER_BIT = 7.4

ASSUME = {
    "cim_group": 16,          # tiles sharing one macro's logic; the largest depth among ZigZag's validated chips
    "alpha": "measured",      # or a number: input toggle rate x weight density (ZigZag validates at 0.25)
    "macro": "zigzag",        # or "colonnade"
    "mult": "aladdin",        # or "horowitz"
    "ge_per_lut": 6.0,        # gate equivalents per FPGA LUT (rule of thumb)
    "ge_per_ff": 6.0,         # as ZigZag's area fit
    "ge_per_dsp": 1500.0,
    "logic_activity": 0.1,    # fraction of gates switching per active cycle, no toggle data
    "leak_pw_per_ge": 50.0,   # logic leakage, low-power 28 nm; no tool here gives it
    "sram_pw_per_bit": None,  # None: CACTI's per memory; or a number for every SRAM bit
    "reuse_access": "banked", # "banked": a cache access costs what an ACT access does; "flat": one large array
    "f_sys_hz": 15.36e6,
}

# Block sizes: Vivado out of context, Nexys A7 (syn/ooc_*/utilization_hier.rpt, 2026-11-05)
BLOCKS = {
    "cim_seq": dict(lut=2443, ff=1982, dsp=9),
    "cim_fe": dict(lut=1530, ff=345, dsp=7),
    "mfcc_ded": dict(lut=1447, ff=440, dsp=8),
    "cim_audio": dict(lut=1026, ff=850, dsp=1),
    "pdm_dec": dict(lut=891, ff=710, dsp=1),      # inside cim_audio
    "glue": dict(lut=2, ff=1012, dsp=0),          # endpoint's own registers (mostly the latched results)
}
TILES = {"cim": 69, "ded": 63}                    # front-end 8 tiles (B3: the 2 DCT tiles) + network 61

# group: passes, columns over all passes (per event), rows, bits; per inference or per frame or per DCT row
SHAPES = {
    "conv1": dict(passes=500, col_passes=8000, rows=40, b=9),
    "dw":    dict(passes=5000, col_passes=32000, rows=63, b=9),
    "pw":    dict(passes=2000, col_passes=32000, rows=64, b=9),
    "fc":    dict(passes=1, col_passes=12, rows=64, b=9),
    "fft1":  dict(passes=32, col_passes=512, rows=16, b=17),
    "fft3":  dict(passes=16, col_passes=256, rows=64, b=17),
    "mel":   dict(passes=4, col_passes=56, rows=64, b=22),
    "dct":   dict(passes=2, col_passes=20, rows=40, b=14),
}
NET_GROUPS = ("conv1", "dw", "pw", "fc")
FE_GROUPS = ("fft1", "fft3", "mel")


def load_counts(mode):
    c = {}
    with open(os.path.join(HERE, f"energy_counts_{mode}.log")) as f:
        for ln in f:
            m = re.match(r"EC (\w+) (\w+) (\d+)", ln)
            if m:
                c[f"{m[1]}.{m[2]}"] = int(m[3])
    return c


def check_counts(c, mode):
    """The counters must agree with what the design is known to do."""
    nd, nf, nr = 101, c.get("ep.fe_starts"), 1446
    assert c["seq.requant"] == 72012 * nd, "requantisations per inference"
    assert c["seq.lmq_dct_requant"] == 10 * nr, "window path requantisations per DCT row"
    assert c["seq.act_writes"] == 72000 * nd, "ACT writes per inference"
    assert c["engine.pass_t8_b9"] == 125 * nd and c["engine.pass_t68_b9"] == nd, "passes per tile"
    net_passes = sum(v for k, v in c.items() if re.match(r"engine\.pass_t\d+_b9", k))
    assert net_passes == 7501 * nd, "7,501 passes per inference"
    assert c["engine.pass_t6_b14"] == nr and c["engine.pass_t7_b14"] == nr, "DCT rows"
    assert sum(s["col_passes"] * 0 + s["passes"] for g, s in SHAPES.items() if g in NET_GROUPS) == 7501
    macs = 500 * 640 + 32000 * 9 + 2000 * 1024 + 768
    assert macs == 2656768, "MACs per inference"
    assert sum(SHAPES[g]["passes"] * SHAPES[g]["b"] for g in NET_GROUPS) == 67509, "bit-cycles per inference"
    if mode == "cim":
        assert c["engine.pass_t0_b17"] == 32 * nf and c["engine.pass_t1_b17"] == 16 * nf
        assert c["fe.samples"] == 960 * nf and c["fe.bbuf_writes"] == 512 * nf and c["fe.mag_results"] == 128 * nf
    else:
        assert c["ded.fmem_writes"] == 5120 * nf and c["ded.fmem_reads_fft"] == 4608 * nf
        assert c["ded.mag_results"] == 127 * nf
    return True


def z(a=None):
    """Gate energies of the fit, fJ."""
    v2 = ZIGZAG["vdd"] ** 2
    fa = (2 * ZIGZAG["xor2"] + 3 * ZIGZAG["nd2"]) * v2
    return dict(fa=fa, dff=ZIGZAG["dff"] * v2, nand=ZIGZAG["nd2"] * v2, mult_bit=ZIGZAG["nd2"] / 2 * v2)


def tree_adders(n_inputs, bits):
    depth = int(math.log2(n_inputs))
    return n_inputs * (bits + 1) - (bits + depth + 1)


def chips():
    """Transcription check: the same equations on published chips (TOP/s/W, model against reported)."""
    g = z()
    out = {}
    for name, c in CHIPS.items():
        alpha = 0.5 * 0.5
        mults = alpha * c["banks"] * c["out"] * g["mult_bit"] * c["mult_bits"] * c["mult_n"]
        tree = alpha * c["banks"] * c["out"] * tree_adders(c["inp"], c["tree_bits"]) * g["fa"]
        n_acc = c["banks"] if c["per_bank_acc"] else c["banks"] * c["out"]
        acc = n_acc * c["acc"] * g["fa"]
        reg = n_acc * c["reg"] * g["dff"]
        if c["channel_tree"]:
            bits, n = c["channel_tree"]
            tree += c["banks"] * tree_adders(n, bits) * g["fa"]
        e = (mults + tree + acc + reg) * c["cycles"]
        topsw = c["ops"] / e * 1000
        out[name] = dict(model=topsw, reported=c["reported"], ratio=topsw / c["reported"])
    return out


class Model:
    def __init__(self, tools, activity, counts, reuse, **over):
        self.a = {**ASSUME, **over}
        self.t = tools
        self.mem = tools["cacti"][NODE]
        self.alpha = activity["groups"]
        self.c = counts
        self.reuse = reuse
        g = z()
        self.g = g
        self.e_ff = ZIGZAG["dff"] / 3 * ZIGZAG["vdd"] ** 2 + self.a["logic_activity"] * g["dff"]
        self.e_lut = self.a["ge_per_lut"] * self.a["logic_activity"] * g["nand"]

    # ---------------------------------------------------------------- unit costs, pJ
    def mul(self, name, a, b):
        if self.a["mult"] == "horowitz":
            return HOROWITZ_MUL32_45NM_PJ * a * b / 1024 * int(NODE) / 45
        return self.t["aladdin"][NODE][name]

    def block_cycle_pj(self, b):
        return (BLOCKS[b]["ff"] * self.e_ff + BLOCKS[b]["lut"] * self.e_lut) / 1000

    def macro_pass_pj(self, group):
        """CIM array energy of all passes of a group, as one event's total (pJ)."""
        s = SHAPES[group]
        if self.a["macro"] == "colonnade":
            cp = self.t["colonnade_passes"][NODE]
            if group == "mel":
                return 3 * cp["mel_16"]["pJ_per_pass"] + cp["mel_8"]["pJ_per_pass"]
            if group == "dct":
                return cp["dct_16"]["pJ_per_pass"] + cp["dct_4"]["pJ_per_pass"]
            return s["passes"] * cp[group]["pJ_per_pass"]
        g = self.g
        al = self.alpha[group]["alpha"] if self.a["alpha"] == "measured" else self.a["alpha"]
        col = g["mult_bit"] * 8 * 64 + tree_adders(64, 8) * g["fa"]
        acc = (s["b"] + 14) * (g["fa"] + g["dff"])
        return s["col_passes"] * s["b"] * (al * col + acc) / 1000

    def select_pj(self, group):
        """Selecting a tile in a macro shared by G tiles: ZigZag's local bitline term; 0 when G = 1."""
        G = self.a["cim_group"]
        if G <= 1 or self.a["macro"] == "colonnade":
            return 0.0
        s = SHAPES[group]
        per_weight = (ZIGZAG["wl"] + ZIGZAG["bl"] * G) * ZIGZAG["vdd"] ** 2 * 8
        return s["rows"] * s["col_passes"] * per_weight / 1000

    # ---------------------------------------------------------------- events, pJ, by part
    def network(self):
        c, m, g, nd = self.c, self.mem, self.g, 101
        acc_r = (c["seq.gather_tv_lt1"] + c["seq.gather_tv_lt2"]) / nd
        feat_r = c["seq.gather_tv_lt0"] / nd
        rq = 72012
        byte_w = 0.3 + 0.7 * 8 / 56                     # the wrapper's own width scaling, for a byte write
        rows_net = (c["engine.xw_rows"] - c["seq.lraw_ring_reads"]) / nd
        add_bits = 25 + 56 + 32                          # bias add, rounding, rounding shift and offset
        return {
            "CIM array": sum(self.macro_pass_pj(x) for x in NET_GROUPS),
            "tile select": sum(self.select_pj(x) for x in NET_GROUPS),
            "ACT memory": acc_r * m["act_word"]["read_pJ"] + 72000 * byte_w * m["act_word"]["write_pJ"],
            "FEAT memory": feat_r * m["feat"]["read_pJ"],
            "requant parameters": rq * m["params"]["read_pJ"],
            "requant arithmetic": rq * (self.mul("requant_25x31", 25, 31) + add_bits * g["fa"] / 1000),
            "input and result registers": (rows_net * 9 * g["dff"] * 0.5 + 72012 * 23 * g["dff"] * 0.5) / 1000,
            "sequencer logic": 85352 * self.block_cycle_pj("cim_seq"),
        }

    def frame(self, mode):
        m, g, c = self.mem, self.g, self.c
        nf = c["ep.fe_starts"]
        if mode == "cim":
            busy = sum(v for k, v in c.items() if re.match(r"fe\.st\d+", k) and k != "fe.st0") / nf
            return {
                "CIM array": sum(self.macro_pass_pj(x) for x in FE_GROUPS),
                "tile select": sum(self.select_pj(x) for x in FE_GROUPS),
                "ring memory": 960 * m["ring"]["read_pJ"],
                "buffers": (1024 * m["bbuf"]["read_pJ"] + 512 * m["bbuf"]["write_pJ"]
                            + 128 * (m["ribuf"]["read_pJ"] + m["ribuf"]["write_pJ"])
                            + 128 * (m["magbuf"]["read_pJ"] + m["magbuf"]["write_pJ"])),
                "multipliers": (960 * self.mul("window_16x15", 16, 15) + 4 * 512 * self.mul("twiddle_16x16", 16, 16)
                                + 2 * 128 * self.mul("square_21x21", 21, 21)),
                "registers": (c["engine.x_rows"] / nf * 17 * g["dff"] * 0.5 + 40 * 72 * g["dff"] * 0.5) / 1000,
                "front-end logic": busy * self.block_cycle_pj("cim_fe"),
            }
        busy = sum(v for k, v in c.items() if re.match(r"ded\.st\d+", k) and k != "ded.st0") / nf
        return {
            "CIM array": 0.0,
            "tile select": 0.0,
            "ring memory": 992 * m["ring"]["read_pJ"],
            "buffers": 5120 * m["fmem"]["write_pJ"] + (4608 + 127) * m["fmem"]["read_pJ"],
            "multipliers": (992 * self.mul("window_16x15", 16, 15) + 4 * 2304 * self.mul("butterfly_18x18", 18, 18)
                            + 2 * 127 * (self.mul("square_21x21", 21, 21) + self.mul("mel_18x8", 18, 8))),
            "registers": 40 * 72 * g["dff"] * 0.5 / 1000,
            "front-end logic": busy * self.block_cycle_pj("mfcc_ded"),
        }

    def window_path(self, rows):
        """One hop: copy, maximum, and `rows` DCT rows of the window path."""
        m, g, c = self.mem, self.g, self.c
        cy = self.block_cycle_pj("cim_seq")
        fixed_cyc = sum(c.get(f"seq.ws{i}", 0) for i in (1, 2, 3, 4, 10)) / 150
        row_cyc = sum(c.get(f"seq.ws{i}", 0) for i in (5, 6, 7, 8, 9)) / 1446
        fixed = 40 * m["lraw"]["write_pJ"] + 320 * m["ring"]["write_pJ"] + fixed_cyc * cy
        row = (40 * m["lraw"]["read_pJ"] + 3 * m["feat"]["write_pJ"] + self.macro_pass_pj("dct") + self.select_pj("dct")
               + 50 * self.mul("requant_25x31", 25, 31) + 40 * 14 * g["dff"] * 0.5 / 1000 + row_cyc * cy)
        return fixed + rows * row

    # ---------------------------------------------------------------- power, uW
    def audio_uw(self):
        """Decimator at the PDM bit rate, its FIR at 16 kHz, and the part that sees every clock."""
        pdm = 1.024e6 * self.block_cycle_pj("pdm_dec") * 1e-12 * 1e6
        fir = 16000 * 15 * self.mul("fir_17x16", 17, 16) * 1e-12 * 1e6
        always = (BLOCKS["cim_audio"]["ff"] - BLOCKS["pdm_dec"]["ff"]) * self.e_ff * self.a["f_sys_hz"] * 1e-15 * 1e6
        return pdm + fir + always

    def sram_uw(self, name, bits=None):
        v = self.mem[name]
        if self.a["sram_pw_per_bit"] is not None:
            return (v["width"] * v["depth"] if bits is None else bits) * self.a["sram_pw_per_bit"] * 1e-6
        return v["leak_uW"] if bits is None else bits * v["pW_per_bit"] * 1e-6

    def leak(self, mode):
        """Everything stays powered: no power gating, no retention voltage."""
        a = self.a
        G = a["cim_group"]
        macros = math.ceil(TILES[mode] / G)
        fa_ge = 3 + 2 * 2.4                              # ZigZag's area fit: 3 NAND2 and 2 XOR2 of 2.4 NAND2
        col_ge = tree_adders(64, 8) * fa_ge + 48 * (fa_ge + a["ge_per_ff"]) + 8 * 64
        macro_ge = 16 * col_ge * macros
        names = ["cim_seq", "cim_audio", "glue", "cim_fe" if mode == "cim" else "mfcc_ded"]
        logic_ge = sum(BLOCKS[b]["lut"] * a["ge_per_lut"] + BLOCKS[b]["ff"] * a["ge_per_ff"] + BLOCKS[b]["dsp"] * a["ge_per_dsp"]
                       for b in names)
        mems = ["act_word", "feat", "lraw", "params", "ring"] + (["bbuf", "ribuf", "magbuf"] if mode == "cim" else ["fmem"])
        weights = TILES[mode] * 64 * 16 * 8
        return {
            "weight cells": self.sram_uw("weights", weights),
            "other SRAM": sum(self.sram_uw(n) for n in mems),
            "CIM macro logic": macro_ge * a["leak_pw_per_ge"] * 1e-6,
            "digital logic": logic_ge * a["leak_pw_per_ge"] * 1e-6,
        }, macros

    def rows_per_hop(self, group="always-on, noise RMS 100"):
        return 1 + 48 * self.reuse["groups"][group]["max_change_pct"] / 100

    def budget(self, mode, k):
        net = sum(self.network().values())
        hop = sum(self.frame(mode).values()) + self.window_path(self.rows_per_hop())
        cpu = CPU_CYCLES_PER_DECISION * CPU_PJ_PER_CYCLE
        lk, macros = self.leak(mode)
        rate = 1 / HOP_S
        return {
            "network": net * rate / k * 1e-6,
            "front-end and window path": hop * rate * 1e-6,
            "CPU": cpu * rate / k * 1e-6,
            "audio input": self.audio_uw(),
            **{f"leakage: {n}": v for n, v in lk.items()},
        }, macros

    # ---------------------------------------------------------------- activation reuse
    def reuse_net(self, k, group):
        """Saving of simple reuse against plain k, uW (positive: reuse wins), and its break-even pW per bit."""
        st = self.reuse["groups"][group][f"simple_k{k}"]
        n = self.network()
        rd = st["act_rd_per_hop"] * k / 320000
        wr = st["act_wr_per_hop"] * k / 72000
        pa = st["passes_mean"] / 7501
        cy = st["cycles_1lane"] / 85352
        scale = {"CIM array": pa, "tile select": pa, "ACT memory": None, "FEAT memory": 1.0, "requant parameters": wr,
                 "requant arithmetic": wr, "input and result registers": rd, "sequencer logic": cy}
        m = self.mem
        byte_w = 0.3 + 0.7 * 8 / 56
        acc_r = (self.c["seq.gather_tv_lt1"] + self.c["seq.gather_tv_lt2"]) / 101 * rd
        acc_w = 72000 * wr
        name = "act_plus_32k" if k > 1 else "act_plus_72k"
        am = m[name] if self.a["reuse_access"] == "flat" else m["act_word"]
        after = sum(v * scale[p] for p, v in n.items() if p != "ACT memory")
        after += acc_r * am["read_pJ"] + acc_w * byte_w * am["write_pJ"]
        saved_uw = (sum(n.values()) - after) / (HOP_S * k) * 1e-6
        bits = (72000 if k == 1 else 32000) * 8
        pw = self.a["sram_pw_per_bit"] if self.a["sram_pw_per_bit"] is not None else m[name]["pW_per_bit"]
        leak_uw = bits * pw * 1e-6
        return dict(saved_dynamic_uW=saved_uw, added_leak_uW=leak_uw, net_uW=saved_uw - leak_uw,
                    break_even_pW_per_bit=saved_uw / bits * 1e6, passes_per_decision=st["passes_mean"])

    def area_mm2(self, mode):
        a = self.a
        G = a["cim_group"]
        macros = math.ceil(TILES[mode] / G)
        u = ZIGZAG["nd2_um2"]
        fa = 3 * u + 2 * u * 2.4
        ff = u * 6
        col = tree_adders(64, 8) * fa + 8 * 64 * u + 48 * (fa + ff)
        macro_logic = 16 * col * macros * 1e-6
        cells = TILES[mode] * 64 * 16 * 8 * self.mem["weights"]["area_um2"] / (128 * 4416) * 1e-6
        mems = ["act_word", "feat", "lraw", "params", "ring"] + (["bbuf", "ribuf", "magbuf"] if mode == "cim" else ["fmem"])
        other = sum(self.mem[n]["area_um2"] for n in mems) * 1e-6
        names = ["cim_seq", "cim_audio", "glue", "cim_fe" if mode == "cim" else "mfcc_ded"]
        logic_ge = sum(BLOCKS[b]["lut"] * a["ge_per_lut"] + BLOCKS[b]["ff"] * a["ge_per_ff"] + BLOCKS[b]["dsp"] * a["ge_per_dsp"]
                       for b in names)
        return dict(weight_cells=cells, macro_logic=macro_logic, other_sram=other, digital_logic=logic_ge * u * 1e-6,
                    macros=macros)


# ---------------------------------------------------------------- StreamFuse (docs/STREAMFUSE_GATES.md)
# Costing rules frozen in the gates: added control as a share of the network sequencer (SEQUENCER.md section 6),
# the second requantisation lane as 2 multipliers of 1,500 gate equivalents, line buffers and maps by CACTI at their
# size, fusion vectors as flip-flops. Counts per inference come from model/streamfuse_sched.py.
SF_SHARE = {"B0": 0.0, "S": 0.0, "A": 0.10, "B": 0.25, "C": 0.60}
SF_SEQ = dict(lut=1176, ff=894)
SF_R2_GE = 2 * 1500
SF_MEM = {("act", 2500): "act_word", ("act", 1300): "act_1300", ("act", 1250): "act_1250", ("lb", 120): "lb_120",
          ("lb", 150): "lb_150"}


def sf_mem_name(buf, words):
    return SF_MEM[("lb" if buf.startswith("lb") else buf, words)]


def sf_network(m, d, skipped, share, scale=1.0):
    """Network energy of one inference (pJ, by part) of a design described by the schedule model's counts d."""
    c, mem, g = d["counts"], m.mem, m.g
    byte_w = 0.3 + 0.7 * 8 / 56
    act = fv = 0.0
    for buf, n in d["words"].items():
        if buf.startswith("fv"):
            fv += (c["byte_writes"].get(buf, 0) * 8 * g["dff"] * 0.5 + c["reads"].get(buf, 0) * 64 * 4 * g["nand"] * 0.5) / 1000
            continue
        e = mem[sf_mem_name(buf, n)]
        act += (c["reads"].get(buf, 0) * e["read_pJ"] + c["byte_writes"].get(buf, 0) * byte_w * e["write_pJ"]
                + c["word_writes"].get(buf, 0) * e["write_pJ"])
    rq = c["requant"]
    add_bits = 25 + 56 + 32
    added = share * scale * (SF_SEQ["lut"] * m.e_lut + SF_SEQ["ff"] * m.e_ff) / 1000
    return {
        "CIM array": sum(m.macro_pass_pj(x) for x in NET_GROUPS),
        "tile select": sum(m.select_pj(x) for x in NET_GROUPS),
        "ACT memory": act + fv,
        "FEAT memory": c["feat_reads"] * mem["feat"]["read_pJ"],
        "requant parameters": rq * mem["params"]["read_pJ"],
        "requant arithmetic": ((rq - skipped) * (m.mul("requant_25x31", 25, 31) + add_bits * g["fa"] / 1000)
                               + skipped * 25 * g["fa"] / 1000),
        "input and result registers": (c["x_rows"] * 9 * g["dff"] * 0.5 + rq * 23 * g["dff"] * 0.5) / 1000,
        "sequencer logic": d["total"] * (m.block_cycle_pj("cim_seq") + added),
    }


def sf_leak(m, d, share, r2, scale=1.0):
    """Leakage (uW, by part) with the design's activation buffers, added control and second lane."""
    lk, _ = m.leak("cim")
    a = m.a
    act = 0.0
    fv_ff = 0
    for buf, n in d["words"].items():
        if buf.startswith("fv"):
            fv_ff += n * 64
        else:
            v = m.mem[sf_mem_name(buf, n)]
            act += m.sram_uw(sf_mem_name(buf, n)) if a["sram_pw_per_bit"] is None else v["width"] * v["depth"] * a["sram_pw_per_bit"] * 1e-6
    lk["other SRAM"] += act - m.sram_uw("act_word")
    ge = share * scale * (SF_SEQ["lut"] * a["ge_per_lut"] + SF_SEQ["ff"] * a["ge_per_ff"]) + (SF_R2_GE if r2 else 0) + fv_ff * a["ge_per_ff"]
    lk["digital logic"] += ge * a["leak_pw_per_ge"] * 1e-6
    return lk


def sf_power(m, net_pj, leak):
    """Average total power at 20 / 40 / 80 ms with this network energy and leakage, the rest as today."""
    out = {}
    for k in (1, 2, 4):
        b, _ = m.budget("cim", k)
        base_leak = sum(v for p, v in b.items() if p.startswith("leakage"))
        out[k] = sum(b.values()) - b["network"] + net_pj / (HOP_S * k) * 1e-6 - base_leak + sum(leak.values())
    return out


def streamfuse(out=print):
    tools = json.load(open(os.path.join(HERE, "energy_components.json")))
    act = json.load(open(os.path.join(HERE, "energy_activity.json")))
    reuse = json.load(open(os.path.join(HERE, "reuse_results.json")))
    sf = json.load(open(os.path.join(HERE, "streamfuse_results.json")))
    cc = load_counts("cim")
    check_counts(cc, "cim")
    cy = sf["cycles"]
    skipped = sf["exact"]["S"]["sg_skipped_per_inference"]
    names = ["B0", "S", "A", "B", "C"]
    lanes2 = {"B0": False, "S": True, "A": True, "B": True, "C": True}
    res = dict(skipped_per_inference=skipped, variants={})

    def evaluate(m, name, scale=1.0, design=None):
        dd = cy[design or name]
        sk = 0.0 if name == "B0" else skipped
        net = sf_network(m, dd, sk, SF_SHARE[name], scale)
        lk = sf_leak(m, dd, SF_SHARE[name], lanes2[name], scale)
        return net, lk, sf_power(m, sum(net.values()), lk)

    # V3: B0 from the schedule model's counts reproduces today's network energy and leakage, in every variant
    v3 = []
    for vname, o in VARIANTS:
        m = Model(tools, act, cc, reuse, **o)
        net, lk, _ = evaluate(m, "B0")
        today, tlk = sum(m.network().values()), sum(m.leak("cim")[0].values())
        v3.append(dict(variant=vname, model=sum(net.values()), today=today, leak_model=sum(lk.values()), leak_today=tlk,
                       ok=abs(sum(net.values()) - today) < 1e-6 * today and abs(sum(lk.values()) - tlk) < 1e-9 * max(tlk, 1)))
    res["V3"] = dict(passed=all(x["ok"] for x in v3), rows=v3)
    out(f"V3: B0 from the schedule model's counts against today's energy model, {len(v3)} variants: "
        f"{'all equal' if res['V3']['passed'] else 'DIFFERENT'} (baseline {v3[0]['model'] / 1000:.1f} nJ against {v3[0]['today'] / 1000:.1f})")

    base = Model(tools, act, cc, reuse)
    out(f"\nSign gating: {skipped:.0f} of 72,000 requantisations per inference skip the multiply ({100 * skipped / 72000:.1f} %), "
        f"mean over the {sf['exact']['S']['clips']} test clips")
    out("\nOne inference, nJ, baseline variant (B0 today; S = B0 + sign gating + second lane; options with both):")
    ev = {n: evaluate(base, n) for n in names}
    parts = list(ev["B0"][0])
    tab([[p] + [f"{ev[n][0][p] / 1000:.1f}" for n in names] for p in parts]
        + [["total"] + [f"{sum(ev[n][0].values()) / 1000:.1f}" for n in names]]
        + [["leakage, uW"] + [f"{sum(ev[n][1].values()):.2f}" for n in names]]
        + [["total power at 20 / 40 / 80 ms, uW"] + [" / ".join(f"{ev[n][2][k]:.1f}" for k in (1, 2, 4)) for n in names]],
        ["part"] + names)
    eS = sum(ev["S"][0].values())
    gates = {}
    for n in ("A", "B", "C"):
        e = sum(ev[n][0].values())
        gates[n] = dict(network_nJ=e / 1000, S_nJ=eS / 1000, ratio=e / eS, G2=e <= 0.90 * eS,
                        guard_40ms=ev[n][2][2] <= ev["S"][2][2], power_uW=ev[n][2], S_power_uW=ev["S"][2])
    res["baseline"] = {n: dict(network_pJ=ev[n][0], leak_uW=ev[n][1], power_uW=ev[n][2]) for n in names}
    out("\nG2, network energy against S (pass if <= 0.90), and the guard on total power at 40 ms:")
    tab([[n, f"{g['network_nJ']:.1f}", f"{g['ratio']:.3f}", "pass" if g["G2"] else "FAIL",
          f"{g['power_uW'][2]:.2f} vs {g['S_power_uW'][2]:.2f}", "pass" if g["guard_40ms"] else "FAIL"] for n, g in gates.items()],
        ["option", "nJ", "against S", "G2", "40 ms, uW (S)", "guard"])
    # variants and doubled overheads
    rows = []
    for vname, o in VARIANTS:
        m = Model(tools, act, cc, reuse, **o)
        es = sum(evaluate(m, "S")[0].values())
        r = {n: sum(evaluate(m, n)[0].values()) / es for n in ("A", "B", "C")}
        r2 = {n: sum(evaluate(m, n, 2.0)[0].values()) / es for n in ("A", "B", "C")}
        res["variants"][vname] = dict(ratio=r, ratio_double_overhead=r2)
        rows.append([vname] + [f"{r[n]:.3f} / {r2[n]:.3f}" for n in ("A", "B", "C")])
    out("\nNetwork energy against S in every variant (overheads as frozen / doubled); G2 needs <= 0.900:")
    tab(rows, ["variant", "A", "B", "C"])
    # for information: the designs without the second lane, and B with its first fusion-vector layout
    info = {}
    for label, n, design in (("B0 + sign gating", "S", "B0+SG"), ("B, 1 lane", "B", "B, 1 lane"),
                             ("C, 1 lane", "C", "C, 1 lane"), ("B, 7-byte fusion vector", "B", "B, 7-byte fusion vector")):
        lanes2_save = lanes2[n]
        lanes2[n] = "1 lane" not in label and label != "B0 + sign gating"
        net, lk, pw = evaluate(base, n, design=design)
        lanes2[n] = lanes2_save
        info[label] = dict(network_nJ=sum(net.values()) / 1000, cycles=cy[design]["total"], power_uW=pw)
    res["info"] = info
    out("\nFor information:")
    tab([[k, f"{v['network_nJ']:.1f}", v["cycles"], " / ".join(f"{v['power_uW'][j]:.1f}" for j in (1, 2, 4))] for k, v in info.items()],
        ["design", "nJ", "cycles", "uW at 20 / 40 / 80 ms"])
    res["gates_G2"] = gates
    with open(os.path.join(HERE, "streamfuse_energy.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


S_OOC_ADD = dict(lut=2938 - 2535, ff=2140 - 1982)   # cim_seq out of context, S against R6b (syn/ooc_cim_seq_RQ_SG{1,0}_*)


def s_rtl(out=print):
    """The baseline S priced from the RTL's own counters (sim/run_energy_counts.sh cim 5 1 s), with the costing rules
    of the StreamFuse study, against S as modelled and against R6b from its counters. Writes energy_s_rtl.json."""
    tools = json.load(open(os.path.join(HERE, "energy_components.json")))
    act = json.load(open(os.path.join(HERE, "energy_activity.json")))
    reuse = json.load(open(os.path.join(HERE, "reuse_results.json")))
    sf = json.load(open(os.path.join(HERE, "streamfuse_results.json")))
    cc, cs = load_counts("cim"), load_counts("cim_s")
    check_counts(cc, "cim")
    nd = 101
    wp_cyc = lambda c: sum(c.get(f"seq.ws{i}", 0) for i in range(1, 11))
    # the RTL's counters: every requantisation either multiplies or is gated; L8 no longer reaches ACT
    assert cs["seq.requant"] == 72012 * nd and cs["seq.requant_mults"] + cs["seq.requant_gated"] == cs["seq.requant"]
    assert cs["seq.act_writes"] == 64000 * nd and cs["seq.pool_updates"] == 8000 * nd
    assert all(cs[k] == cc[k] for k in cc if k.startswith("engine.pass_") or k.startswith("seq.gather")), "same passes and gathers"
    assert wp_cyc(cs) == wp_cyc(cc), "the window path is unchanged"
    saved = (cc["seq.run_cycles"] - cs["seq.run_cycles"]) / nd
    seq_log = open(os.path.join(HERE, "..", "sim", "logs", "seq_tb_cim_s.log")).read()
    cyc = int(re.search(r"full inference (\d+)\.\.(\d+) cycles", seq_log)[1])
    assert saved == 85352 - cyc, f"counters save {saved} cycles per inference, the sequencer bench {85352 - cyc}"
    gated = cs["seq.requant_gated"] / nd
    acc_r = (cs["seq.gather_tv_lt1"] + cs["seq.gather_tv_lt2"]) / nd
    rows_net = (cs["engine.xw_rows"] - cs["seq.lraw_ring_reads"]) / nd
    d_rtl = dict(total=cyc, words={"act": 2500},
                 counts=dict(reads={"act": acc_r}, byte_writes={"act": cs["seq.act_writes"] / nd}, word_writes={},
                             feat_reads=cs["seq.gather_tv_lt0"] / nd, requant=cs["seq.requant"] / nd, x_rows=rows_net))
    d_mod = sf["cycles"]["S"]
    sk_mod = sf["exact"]["S"]["sg_skipped_per_inference"]
    res = dict(cycles=cyc, gated_per_inference=gated, mults_per_inference=cs["seq.requant_mults"] / nd,
               act_byte_writes=cs["seq.act_writes"] / nd, model_gated=sk_mod, variants={})
    out(f"S from its RTL counters (tb_cim_mic, 101 decisions, PDM divider 5): {cyc} cycles per inference "
        f"({saved:.0f} fewer than R6b, from the counters and the sequencer bench alike); {gated:.1f} of 72,012 requantisations "
        f"gated ({100 * gated / 72012:.2f} %; the model's test-set mean {sk_mod:.1f}, {100 * sk_mod / 72012:.2f} %); "
        f"{cs['seq.act_writes'] / nd:.0f} ACT byte writes")
    rows = []
    for vname, o in VARIANTS + [("4.096 MHz clock", dict(f_sys_hz=4.096e6))]:
        m = Model(tools, act, cc, reuse, **o)
        b0 = sum(m.network().values())
        net_r = sf_network(m, d_rtl, gated, 0.0)
        net_m = sf_network(m, d_mod, sk_mod, 0.0)
        lk = sf_leak(m, d_mod, 0.0, True)
        pw_r, pw_b0 = sf_power(m, sum(net_r.values()), lk), sf_power(m, b0, m.leak("cim")[0])
        e_r, e_m = sum(net_r.values()), sum(net_m.values())
        res["variants"][vname] = dict(B0_nJ=b0 / 1000, S_rtl_nJ=e_r / 1000, S_model_nJ=e_m / 1000, power_S_rtl_uW=pw_r, power_B0_uW=pw_b0,
                                      parts_S_rtl_pJ=net_r if vname == "baseline" else None)
        rows.append([vname, f"{b0 / 1000:.1f}", f"{e_r / 1000:.1f}", f"{e_m / 1000:.1f}", f"{e_r / e_m:.4f}", f"{100 * (1 - e_r / b0):.1f} %",
                     " / ".join(f"{pw_r[k]:.1f}" for k in (1, 2, 4)), " / ".join(f"{pw_b0[k]:.1f}" for k in (1, 2, 4))])
    out("\nOne inference, nJ [E], and total power, uW [E]:")
    tab(rows, ["variant", "R6b", "S, RTL counts", "S, model", "RTL / model", "S saves", "S at 20 / 40 / 80 ms", "R6b at 20 / 40 / 80 ms"])
    b = res["variants"]["baseline"]
    out("\nS by part, baseline variant, nJ: " + ", ".join(f"{k} {v / 1000:.1f}" for k, v in b["parts_S_rtl_pJ"].items()))
    # The study priced the second lane as 3,000 gate equivalents of leakage and its operations per requantisation. The
    # measured S adds 403 LUT and 158 FF to cim_seq (syn/ooc_cim_seq_RQ_SG*); charging all of it every cycle as well,
    # like the sequencer's control, double-counts the lane's datapath and is an upper bound.
    m = Model(tools, act, cc, reuse)
    extra = S_OOC_ADD["lut"] * m.e_lut / 1000 + S_OOC_ADD["ff"] * m.e_ff / 1000
    lk = sf_leak(m, d_mod, 0.0, True)
    lk["digital logic"] += (S_OOC_ADD["lut"] * m.a["ge_per_lut"] + S_OOC_ADD["ff"] * m.a["ge_per_ff"] - SF_R2_GE) * m.a["leak_pw_per_ge"] * 1e-6
    e_ub = b["S_rtl_nJ"] * 1000 + cyc * extra
    pw_ub = sf_power(m, e_ub, lk)
    res["upper_bound_measured_size"] = dict(S_nJ=e_ub / 1000, saves=1 - e_ub / (b["B0_nJ"] * 1000), power_uW=pw_ub, add=S_OOC_ADD)
    out(f"Upper bound with the measured size (+{S_OOC_ADD['lut']} LUT, +{S_OOC_ADD['ff']} FF, every cycle, and their leakage): "
        f"{e_ub / 1000:.1f} nJ ({100 * (1 - e_ub / (b['B0_nJ'] * 1000)):.1f} % below R6b), "
        + " / ".join(f"{pw_ub[k]:.1f}" for k in (1, 2, 4)) + " uW at 20 / 40 / 80 ms")
    with open(os.path.join(HERE, "energy_s_rtl.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


# B1 (docs/STREAMFUSE_B1.md): cim_seq inside the endpoint, Vivado out of context, B1 against S on the same source
# (syn/ooc_cim_endpoint_*_SEQ_B11 and syn/ooc_cim_endpoint_S_same_source_as_B1): logic and SRL LUTs and FFs are the
# added control; the LUT RAM (line buffer, fusion vector) is priced as memory by the frozen rules instead.
B1_OOC_ADD = dict(lut=(3385 - 2967) + (30 - 28), ff=2334 - 2137, lutram=492 - 274)
SF_MEM[("lb", 128)] = "lb_128"


def b1_rtl(out=print):
    """B1 priced from the RTL's own counters (sim/run_energy_counts.sh cim 5 1 b1) against S from its counters, with the
    StreamFuse costing rules, and against B1 as the schedule model predicts it. Writes energy_b1_rtl.json."""
    tools = json.load(open(os.path.join(HERE, "energy_components.json")))
    act = json.load(open(os.path.join(HERE, "energy_activity.json")))
    reuse = json.load(open(os.path.join(HERE, "reuse_results.json")))
    sf = json.load(open(os.path.join(HERE, "streamfuse_results.json")))
    cc, cs, cb = load_counts("cim"), load_counts("cim_s"), load_counts("cim_b1")
    check_counts(cc, "cim")
    nd = 101
    for k in ("seq.requant", "seq.requant_mults", "seq.requant_gated", "seq.passes", "seq.gather_tv_lt0", "engine.xw_rows"):
        assert cb[k] == cs[k], f"{k}: B1 {cb[k]}, S {cs[k]}"
    assert all(cb[k] == cs[k] for k in cs if k.startswith("engine.pass_")), "same passes on the same tiles"
    per = {k: cb[f"seq.b1_{k}"] / nd for k in ("gather_lb", "gather_fv", "fill_act_reads", "lb_writes", "fv_byte_writes")}
    assert per == dict(gather_lb=9490, gather_fv=1000, fill_act_reads=1250, lb_writes=1250, fv_byte_writes=8000), per
    assert cb["seq.act_writes"] == 56000 * nd
    log = open(os.path.join(HERE, "..", "sim", "logs", "seq_tb_cim_b1.log")).read()
    cyc = int(re.search(r"full inference (\d+)\.\.(\d+) cycles", log)[1])
    cyc_s = int(re.search(r"full inference (\d+)\.\.(\d+) cycles", open(os.path.join(HERE, "..", "sim", "logs", "seq_tb_cim_s.log")).read())[1])
    assert (cs["seq.run_cycles"] - cb["seq.run_cycles"]) / nd == cyc_s - cyc, "the counters and the sequencer bench agree on the saving"
    rows_net = (cb["engine.xw_rows"] - cb["seq.lraw_ring_reads"]) / nd
    act_r = (cb["seq.gather_tv_lt1"] + cb["seq.gather_tv_lt2"]) / nd - per["gather_lb"] - per["gather_fv"] + per["fill_act_reads"]
    common = dict(feat_reads=cb["seq.gather_tv_lt0"] / nd, requant=cb["seq.requant"] / nd, x_rows=rows_net)
    d_b1 = dict(total=cyc, words={"act": 2500, "lb": 128, "fv": 16},
                counts=dict(reads={"act": act_r, "lb": per["gather_lb"], "fv": per["gather_fv"]},
                            byte_writes={"act": cb["seq.act_writes"] / nd, "fv": per["fv_byte_writes"]},
                            word_writes={"lb": per["lb_writes"]}, **common))
    d_s = dict(total=cyc_s, words={"act": 2500},
               counts=dict(reads={"act": (cs["seq.gather_tv_lt1"] + cs["seq.gather_tv_lt2"]) / nd},
                           byte_writes={"act": cs["seq.act_writes"] / nd}, word_writes={}, **common))
    mb = sf["B1"]["B1"]
    d_mod = dict(total=mb["total"], words=mb["words"],
                 counts=dict(reads=mb["reads"], byte_writes=mb["byte_writes"], word_writes=mb["word_writes"],
                             feat_reads=mb["feat_reads"], requant=mb["requant"], x_rows=mb["x_rows"]))
    gated = cb["seq.requant_gated"] / nd
    ctrl_ge = lambda m: B1_OOC_ADD["lut"] * m.a["ge_per_lut"] + B1_OOC_ADD["ff"] * m.a["ge_per_ff"]
    res = dict(cycles=cyc, cycles_S=cyc_s, act_reads=act_r, counts=per, control=B1_OOC_ADD, variants={})

    def b1_energy(m, d, rule):
        """rule 'measured': the OOC-measured control every cycle, with its leakage; 'frozen': B's 25 % share."""
        if rule == "frozen":
            net = sf_network(m, d, gated, SF_SHARE["B"])
            lk = sf_leak(m, d, SF_SHARE["B"], True)
        else:
            net = sf_network(m, d, gated, 0.0)
            net["sequencer logic"] += d["total"] * (B1_OOC_ADD["lut"] * m.e_lut + B1_OOC_ADD["ff"] * m.e_ff) / 1000
            lk = sf_leak(m, d, 0.0, True)
            lk["digital logic"] += ctrl_ge(m) * m.a["leak_pw_per_ge"] * 1e-6
        return net, lk, sf_power(m, sum(net.values()), lk)

    rows = []
    for vname, o in VARIANTS + [("4.096 MHz clock", dict(f_sys_hz=4.096e6))]:
        m = Model(tools, act, cc, reuse, **o)
        net_s = sf_network(m, d_s, gated, 0.0)
        lk_s = sf_leak(m, d_s, 0.0, True)
        pw_s = sf_power(m, sum(net_s.values()), lk_s)
        nb, lb_, pb = b1_energy(m, d_b1, "measured")
        nm_, _, _ = b1_energy(m, d_mod, "measured")
        nf, _, pf = b1_energy(m, d_b1, "frozen")
        es, eb, em, ef = (sum(x.values()) for x in (net_s, nb, nm_, nf))
        res["variants"][vname] = dict(S_nJ=es / 1000, B1_nJ=eb / 1000, B1_model_nJ=em / 1000, B1_frozen_rule_nJ=ef / 1000,
                                      ratio=eb / es, ratio_frozen_rule=ef / es, power_S_uW=pw_s, power_B1_uW=pb, power_B1_frozen_uW=pf,
                                      parts_S_pJ=net_s if vname == "baseline" else None, parts_B1_pJ=nb if vname == "baseline" else None)
        rows.append([vname, f"{es / 1000:.1f}", f"{eb / 1000:.1f}", f"{eb / em:.4f}", f"{eb / es:.3f}", f"{ef / es:.3f}",
                     " / ".join(f"{pw_s[k]:.1f}" for k in (1, 2, 4)), " / ".join(f"{pb[k]:.1f}" for k in (1, 2, 4))])
    out(f"B1 from its RTL counters (tb_cim_mic, 101 decisions): {cyc} cycles per inference (S {cyc_s}); per inference "
        f"{act_r:.0f} ACT word reads (incl. {per['fill_act_reads']:.0f} fill reads), {per['gather_lb']:.0f} line-buffer reads, "
        f"{per['gather_fv']:.0f} fusion-vector chunk reads, {cb['seq.act_writes'] / nd:.0f} ACT and {per['fv_byte_writes']:.0f} "
        f"fusion-vector byte writes; added control as measured: +{B1_OOC_ADD['lut']} LUT, +{B1_OOC_ADD['ff']} FF")
    out("\nOne inference, nJ [E], and total power, uW [E]:")
    tab(rows, ["variant", "S", "B1", "B1 RTL / model", "B1 / S", "B1 / S, frozen 25 % rule", "S at 20 / 40 / 80 ms", "B1 at 20 / 40 / 80 ms"])
    b = res["variants"]["baseline"]
    out("\nBy part, baseline variant, nJ (S -> B1): " + ", ".join(
        f"{k} {b['parts_S_pJ'][k] / 1000:.1f} -> {b['parts_B1_pJ'][k] / 1000:.1f}" for k in b["parts_S_pJ"]))
    with open(os.path.join(HERE, "energy_b1_rtl.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def tab(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    print("  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print("  " + "  ".join(str(v).rjust(w[i]) if i else str(v).ljust(w[i]) for i, v in enumerate(r)))


VARIANTS = [
    ("baseline", {}),
    ("alpha 0.25 (validated)", dict(alpha=0.25)),
    ("Colonnade array, 0.8 V", dict(macro="colonnade")),
    ("multipliers: Horowitz", dict(mult="horowitz")),
    ("logic activity 0.05", dict(logic_activity=0.05)),
    ("logic activity 0.2", dict(logic_activity=0.2)),
    ("6 -> 4 GE per LUT", dict(ge_per_lut=4.0)),
    ("6 -> 10 GE per LUT", dict(ge_per_lut=10.0)),
    ("logic leak 20 pW/GE", dict(leak_pw_per_ge=20.0)),
    ("logic leak 100 pW/GE", dict(leak_pw_per_ge=100.0)),
    ("SRAM 7.4 pW/bit (silicon)", dict(sram_pw_per_bit=7.4)),
    ("SRAM 25 pW/bit", dict(sram_pw_per_bit=25.0)),
    ("tiles per macro 1", dict(cim_group=1)),
    ("tiles per macro 4", dict(cim_group=4)),
    ("tiles per macro 8", dict(cim_group=8)),
    ("tiles per macro 32", dict(cim_group=32)),
    ("reuse: flat array", dict(reuse_access="flat")),
    ("5.12 MHz clock", dict(f_sys_hz=5.12e6)),
]


def main():
    tools = json.load(open(os.path.join(HERE, "energy_components.json")))
    act = json.load(open(os.path.join(HERE, "energy_activity.json")))
    reuse = json.load(open(os.path.join(HERE, "reuse_results.json")))
    cc, cd = load_counts("cim"), load_counts("ded")
    check_counts(cc, "cim")
    check_counts(cd, "ded")
    print("Counts: all exact checks pass (7,501 passes, 72,000 ACT writes, 2,656,768 MACs per inference; frame counts).")
    res = dict(tools=tools["tools"], assume=ASSUME)

    print("\nTranscription of ZigZag-IMC's digital equations against its chip figures (TOP/s/W):")
    ch = chips()
    tab([[n, f"{v['model']:.2f}", f"{v['reported']:.2f}", f"{v['ratio']:.3f}"] for n, v in ch.items()], ["chip", "model", "reported", "ratio"])
    res["chips"] = ch
    print("\nColonnade through AccelForge against the paper (fJ per MAC, 0.8 V, 65 nm):")
    tab([[n, f"{v['model_fJ_per_MAC']:.1f}", f"{v['paper_fJ_per_MAC']:.1f}", f"{v['ratio']:.2f}"] for n, v in tools["colonnade_check"].items()],
        ["case", "model", "paper", "ratio"])

    base = Model(tools, act, cc, reuse)
    print("\nMeasured bit activity (alpha; ZigZag validates at 0.25):")
    tab([[g, f"{v['alpha']:.4f}", f"{v['toggle_rate']:.3f}", f"{v['weight_density']:.3f}"] for g, v in act["groups"].items()],
        ["group", "alpha", "toggle rate", "weight density"])

    print("\nOne network inference, nJ (2,656,768 MACs):")
    models = [("measured alpha", {}), ("alpha 0.25", dict(alpha=0.25)), ("Colonnade", dict(macro="colonnade"))]
    nets = {n: Model(tools, act, cc, reuse, **o).network() for n, o in models}
    parts = list(nets["measured alpha"])
    tab([[p] + [f"{nets[n][p] / 1000:.1f}" for n, _ in models] for p in parts] +
        [["total"] + [f"{sum(nets[n].values()) / 1000:.1f}" for n, _ in models]] +
        [["fJ per MAC, all included"] + [f"{sum(nets[n].values()) * 1000 / 2656768:.0f}" for n, _ in models]],
        ["part"] + [n for n, _ in models])
    res["network_pJ"] = nets

    print("\nFront-end, one frame, nJ: on the CIM (cim_fe) against dedicated (mfcc_ded, B3):")
    fr = {}
    for n, o in models:
        mc = Model(tools, act, cc, reuse, **o)
        md = Model(tools, act, cd, reuse, **o)
        fr[n] = dict(cim=mc.frame("cim"), ded=md.frame("ded"))
    parts = list(fr["measured alpha"]["cim"])
    tab([[p] + [f"{fr[n]['cim'][p] / 1000:.2f} / {fr[n]['ded'][p] / 1000:.2f}" for n, _ in models] for p in parts] +
        [["total"] + [f"{sum(fr[n]['cim'].values()) / 1000:.2f} / {sum(fr[n]['ded'].values()) / 1000:.2f}" for n, _ in models]],
        ["part"] + [n + " (CIM / B3)" for n, _ in models])
    res["frame_pJ"] = fr

    print(f"\nWindow path per hop, nJ: 1 row {base.window_path(1) / 1000:.2f}; 49 rows {base.window_path(49) / 1000:.2f}; "
          f"mean at {base.rows_per_hop():.1f} rows {base.window_path(base.rows_per_hop()) / 1000:.2f}")
    print(f"CPU per decision (upper bound) {CPU_CYCLES_PER_DECISION * CPU_PJ_PER_CYCLE / 1000:.1f} nJ; audio input {base.audio_uw():.2f} uW at 15.36 MHz")

    print("\nAverage power, uW, baseline, one decision every 20 / 40 / 80 ms, cim_fe (B3 total in the last row):")
    bud = {k: base.budget("cim", k)[0] for k in (1, 2, 4)}
    budd = {k: Model(tools, act, cd, reuse).budget("ded", k)[0] for k in (1, 2, 4)}
    rows = [[p] + [f"{bud[k][p]:.2f}" for k in (1, 2, 4)] for p in bud[1]]
    rows.append(["total"] + [f"{sum(bud[k].values()):.1f}" for k in (1, 2, 4)])
    lk = {k: sum(v for p, v in bud[k].items() if p.startswith("leakage")) for k in (1, 2, 4)}
    rows.append(["  of which leakage"] + [f"{lk[k]:.1f} ({100 * lk[k] / sum(bud[k].values()):.0f} %)" for k in (1, 2, 4)])
    rows.append(["total with B3 front-end"] + [f"{sum(budd[k].values()):.1f}" for k in (1, 2, 4)])
    tab(rows, ["part", "20 ms", "40 ms", "80 ms"])
    res["budget_uW"] = dict(cim=bud, ded=budd)

    print("\nArea, mm^2 at 28 nm (not a layout):")
    ar = base.area_mm2("cim")
    print("  " + ", ".join(f"{k} {v:.3f}" if k != "macros" else f"{k} {v}" for k, v in ar.items())
          + f"; total {sum(v for k, v in ar.items() if k != 'macros'):.3f}")
    res["area_mm2"] = ar

    print("\nActivation reuse against the same decision rate without it (simple mode, baseline, stream group ranges):")
    groups = list(reuse["groups"])
    rr = {}
    rows = []
    for k, label in ((1, "20 ms, 2 caches, +72 KB"), (2, "40 ms, 1 cache, +32 KB"), (4, "80 ms, 1 cache, +32 KB")):
        v = [base.reuse_net(k, g) for g in groups]
        rr[k] = v
        lo, hi = min(x["net_uW"] for x in v), max(x["net_uW"] for x in v)
        rows.append([label, f"{min(x['saved_dynamic_uW'] for x in v):.1f}-{max(x['saved_dynamic_uW'] for x in v):.1f}",
                     f"{v[0]['added_leak_uW']:.1f}", f"{lo:+.1f}..{hi:+.1f}",
                     f"{min(x['break_even_pW_per_bit'] for x in v):.1f}-{max(x['break_even_pW_per_bit'] for x in v):.1f}"])
    tab(rows, ["rate, cache", "dynamic saved, uW", "leakage added, uW", "net, uW", "break-even, pW/bit"])
    res["reuse"] = rr

    print("\nThe same streams with reuse off at the next slower rate, as an alternative to reuse:")
    t = {k: sum(bud[k].values()) for k in bud}
    r1 = t[1] - base.reuse_net(1, groups[0])["net_uW"]
    r2 = t[2] - base.reuse_net(2, groups[0])["net_uW"]
    r4 = t[4] - base.reuse_net(4, groups[0])["net_uW"]
    tab([["plain", f"{t[1]:.1f}", f"{t[2]:.1f}", f"{t[4]:.1f}"], ["with simple reuse", f"{r1:.1f}", f"{r2:.1f}", f"{r4:.1f}"]],
        ["", "20 ms", "40 ms", "80 ms"])

    print("\nSensitivity: each row changes one thing against the baseline")
    rows = []
    sens = {}
    g0 = groups[0]
    for name, o in VARIANTS:
        m = Model(tools, act, cc, reuse, **o)
        p = {k: sum(m.budget("cim", k)[0].values()) for k in (1, 2, 4)}
        r2_ = m.reuse_net(2, g0)
        r1_ = m.reuse_net(1, g0)
        d = Model(tools, act, cd, reuse, **o)
        lc, mc_ = m.leak("cim")
        ld, md_ = d.leak("ded")
        fe_dyn = (sum(m.frame("cim").values()) - sum(d.frame("ded").values())) / HOP_S * 1e-6
        fe_leak = sum(lc.values()) - sum(ld.values())
        rows.append([name, f"{sum(m.network().values()) / 1000:.0f}", f"{p[1]:.0f}", f"{p[2]:.0f}", f"{p[4]:.0f}",
                     f"{r1_['net_uW']:+.1f}", f"{r2_['net_uW']:+.1f}", f"{r2_['break_even_pW_per_bit']:.0f}",
                     f"{fe_dyn:+.2f}", f"{fe_leak:+.2f}"])
        sens[name] = dict(net_nJ=sum(m.network().values()) / 1000, total_uW=p, reuse_k1=r1_, reuse_k2=r2_, fe_dyn_uW=fe_dyn, fe_leak_uW=fe_leak)
    tab(rows, ["variant", "net nJ", "20 ms", "40 ms", "80 ms", "reuse@20", "reuse@40", "b/e pW", "FE dyn", "FE leak"])
    print("  20/40/80 ms: total uW. reuse@: net saving of simple reuse, uW. b/e: its break-even at 40 ms.")
    print("  FE dyn, FE leak: cim_fe minus B3, uW at 50 frames per second (positive: the CIM front-end costs more).")
    res["sensitivity"] = sens

    with open(os.path.join(HERE, "energy_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return 0


# ---------------------------------------------------------------- whole-word decisions (docs/WHOLEWORD_GATES.md)
WW_LOGIC_GE = 600                                    # the whole-word test and run state, assumed by the gates


def ww_parts():
    """Per variant, from S priced by its RTL counters (energy_s_rtl.json): S's power with no inference P0 (uW) and its
    energy per decision E (pJ, network + CPU) from P(k) = P0 + E / (20 ms k); the CPU's part is 263 cycles x 28 pJ."""
    s = json.load(open(os.path.join(HERE, "energy_s_rtl.json")))
    e_cpu = CPU_CYCLES_PER_DECISION * CPU_PJ_PER_CYCLE
    over = dict(VARIANTS)
    out = {}
    for name, v in s["variants"].items():
        p = {int(k): x for k, x in v["power_S_rtl_uW"].items()}
        e = (p[1] - p[2]) * 2 * HOP_S * 1e6
        p0 = p[1] - e / HOP_S * 1e-6
        out[name] = dict(p0_uW=p0, e_net_pJ=e - e_cpu, e_cpu_pJ=e_cpu, s_uW=p,
                         leak_pw_per_ge=over.get(name, {}).get("leak_pw_per_ge", ASSUME["leak_pw_per_ge"]))
    return out


def ww_power(parts, net_per_s, cpu_per_s, added_ge=0):
    """Average power (uW) with this many network runs and CPU wakes per second, everything else as S."""
    return (parts["p0_uW"] + (parts["e_net_pJ"] * net_per_s + parts["e_cpu_pJ"] * cpu_per_s) * 1e-6
            + added_ge * parts["leak_pw_per_ge"] * 1e-6)


def ww_check(parts):
    """V4: the uniform rates k = 1, 2, 4 give S's own powers back, in every variant (k = 4 is the independent check)."""
    return max(abs(ww_power(q, 1 / (HOP_S * k), 1 / (HOP_S * k)) - q["s_uW"][k]) for q in parts.values() for k in (1, 2, 4))


def wholeword(out=print):
    """Price the network runs and CPU wakes that model/wholeword.py measured (model/wholeword_results.json), in every
    variant; writes model/wholeword_energy.json."""
    parts = ww_parts()
    err = ww_check(parts)
    b = parts["baseline"]
    out(f"V4: S's power at 20 / 40 / 80 ms from P0 = {b['p0_uW']:.2f} uW and E = {(b['e_net_pJ'] + b['e_cpu_pJ']) / 1000:.2f} nJ "
        f"per decision: {' / '.join(f'{ww_power(b, 50 / k, 50 / k):.1f}' for k in (1, 2, 4))} uW; largest error over all "
        f"variants {err:.2e} uW -> {'pass' if err < 1e-6 else 'FAIL'}")
    res = json.load(open(os.path.join(HERE, "wholeword_results.json")))
    table = {}
    for vname, q in parts.items():
        table[vname] = {}
        for cand, sets in res["rates"].items():
            ge = WW_LOGIC_GE if cand in ("A", "B") else 0
            table[vname][cand] = {s: ww_power(q, r["net_per_hop"] / HOP_S, r["cpu_per_hop"] / HOP_S, ge) for s, r in sets.items()}
    kinds = list(next(iter(res["rates"].values())))
    out(f"\nAverage power, uW [E], baseline variant ({', '.join(kinds)}):")
    tab([[c] + [f"{v:.1f}" for v in table["baseline"][c].values()] for c in res["rates"]], ["design"] + kinds)
    out("\nEvery variant, on the word streams:")
    cands = list(res["rates"])
    tab([[vn] + [f"{table[vn][c]['words']:.1f}" for c in cands] for vn in table], ["variant"] + cands)
    with open(os.path.join(HERE, "wholeword_energy.json"), "w") as f:
        json.dump(dict(v4_err_uW=err, parts=parts, power_uW=table), f, indent=1, default=float)
    return table


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "wholeword":
        wholeword()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "streamfuse":
        streamfuse()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "s_rtl":
        s_rtl()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "b1_rtl":
        b1_rtl()
        sys.exit(0)
    sys.exit(main())
