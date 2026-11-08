#!/usr/bin/env python3
"""Mutation test of the assertions (docs/VERIFICATION.md): each mutant breaks the RTL in one place, in a scratch
copy, and the bench named must then trip the assertion named (or, for "bench", fail its own comparison with the golden
model). A check that cannot fail proves nothing.
Usage: verif_mutants.py [jobs]       writes model/verif_mutants.json. Standard library only.
"""
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL = ["cim_engine", "cim_fe", "mfcc_ded", "cim_log2fix16", "cim_seq", "pdm_dec", "cim_audio", "cim_endpoint"]
# name, file, old text, new text, bench arguments, the assertion that must fail
M = [
    ("engine issues one cycle early", "cim_engine", "gap     <= d_in - 5'd1;", "gap     <= d_in - 5'd2;", ["engine"], "engine.ready_low_d_minus_1"),
    ("engine result one cycle late", "cim_engine", "wait_cnt <= m4.dm8;", "wait_cnt <= m4.dm8 + 5'd1;", ["engine"], "engine.done_at_d_plus_5"),
    ("input bank released too early", "cim_seq", "bcnt[i_b]  <= 3'd7;", "bcnt[i_b]  <= 3'd3;", ["seq", "cim"], "engine.input_bank_stable_port2"),
    ("CPU go granted while the engine is busy", "cim_endpoint", "if (go)           gnt = obi_req_i.req && eng_ready && !fe_owns && !seq_busy;",
     "if (go)           gnt = obi_req_i.req && !fe_owns && !seq_busy;", ["endpoint"], "engine.start_only_when_ready"),
    ("passes too close for requantisation", "cim_seq", ": i_tag.n - 5'd1;", ": i_tag.n - 5'd8;", ["seq", "cim"], "seq.results_drained_before_pop"),
    ("a requantised result lost", "cim_seq", "q1_v <= q0_v; q2_v <= q1_v; q3_v <= q2_v; q4_v <= q3_v;",
     "q1_v <= q0_v; q2_v <= q1_v; q3_v <= q2_v; q4_v <= q3_v && !(q3_d.p == 7'd60 && q3_d.c == 6'd5);", ["seq", "cim"], "seq.requant_keeps_count"),
    ("hop one sample short", "cim_audio", "assign hop_end     = sv && (hs == 9'd319);", "assign hop_end     = sv && (hs == 9'd318);", ["mic", "cim", "5", "1"], "audio.hop_is_320_samples"),
    ("decision counted twice", "cim_audio", "ndec_o  <= ndec_o + 16'd1;", "ndec_o  <= ndec_o + 16'd2;", ["mic", "cim", "5", "1"], "audio.decision_count_steps"),
    ("decision on a hop without the network", "cim_audio", "if (hc_full && net_i) begin", "if (net_i && h_ge49) begin", ["mic", "cim", "5", "2"], "audio.decision_only_on_network_hop"),
    ("stride counter runs past k - 1", "cim_audio", "if (h_ge49) kcnt <= (kcnt >= net_km1_i) ? 4'd0 : kcnt + 4'd1;", "if (h_ge49) kcnt <= kcnt + 4'd1;", ["mic", "cim", "5", "4"], "audio.stride_counter_bounded"),
    ("PDM clock outside 40-60 % duty", "pdm_dec", "pdm_clk_o <= (ccnt >= HI - 1'b1) && (ccnt != CW'(CLKDIV - 1));", "pdm_clk_o <= (ccnt >= CW'(1)) && (ccnt != CW'(CLKDIV - 1));", ["pdm"], "pdm.pdm_clock_duty_40_60"),
    ("interrupt ignores its enable", "cim_endpoint", "assign irq_o = (done_sticky && irq_en[0]) ||", "assign irq_o = (done_sticky) ||", ["seq", "cim"], "endpoint.irq_only_when_enabled"),
    ("sequencer may start on cim_fe's engine", "cim_endpoint", "else if (seq_go)  gnt = obi_req_i.req && !seq_busy && !fe_owns && !eng_busy &&",
     "else if (seq_go)  gnt = obi_req_i.req && !seq_busy &&", ["seq", "cim"], "endpoint.seq_fe_not_with_cim_fe"),
    ("front-end reads past its frame", "cim_fe", "assign samp_addr_o = base_q + 10'(sh_idx);", "assign samp_addr_o = base_q + 10'(sh_idx) + 10'd700;", ["fe", "cim"], "frontend.reads_inside_its_frame"),
    ("B3 reads past its frame", "mfcc_ded", "assign samp_addr_o = base_q + 10'(sh_idx);", "assign samp_addr_o = base_q + 10'(sh_idx) + 10'd700;", ["fe", "ded"], "frontend.reads_inside_its_frame"),
    ("response one cycle late on a read path", "cim_endpoint", "rvalid_q  <= gnt;", "rvalid_q  <= gnt_d;", ["endpoint"], "endpoint.rvalid_follows_gnt"),
    # the baseline S (SEQ_S = 1). "bench" means the bench's own comparison with the golden model must fail.
    ("S: sign gate too wide (x < 64)", "cim_seq", "assign q1_g = sg_on && (q1_x[24] || (q1_x == 25'sd0));", "assign q1_g = sg_on && (q1_x < 25'sd64);",
     ["seq", "cim", "s"], "seq.sign_gate_exact"),
    ("S: sign gate ignores the zero point", "cim_seq", "assign sg_on = RQ_SG && (lay != 4'd9) && (zact_q == 8'h80);", "assign sg_on = RQ_SG && (lay != 4'd9);",
     ["seq", "cim", "s"], "seq.sign_gate_exact"),
    ("S: sign gate also in the FC layer", "cim_seq", "assign sg_on = RQ_SG && (lay != 4'd9) && (zact_q == 8'h80);", "assign sg_on = RQ_SG && (zact_q == 8'h80);",
     ["seq", "cim", "s"], "seq.sign_gate_exact"),
    ("S: a gated result is not -128", "cim_seq", "q4_y   <= q3_g ? 8'h80 :", "q4_y   <= q3_g ? 8'h81 :", ["seq", "cim", "s"], "bench"),
    ("S: lane 1 reads the even parameter bank", "cim_seq", "assign prm_ra = {lay, (k == 0) ? d_ix0 : d_ix1};", "assign prm_ra = {lay, d_ix0};",
     ["seq", "cim", "s"], "bench"),
    ("S: lanes take the wrong parity", "cim_seq", "assign d_sw  = RQ_2L && d_c[0];", "assign d_sw  = RQ_2L && !d_c[0];", ["seq", "cim", "s"], "seq.lane_channel_parity"),
    ("S: the second result of a pair goes to the first one's byte lane", "cim_seq", "cl: (d_cl == 3'd6) ? 3'd0 : d_cl + 3'd1};", "cl: d_cl};",
     ["seq", "cim", "s"], "seq.pair_writes_distinct_lanes"),
    ("S: lane 1's ACT write dropped at one position", "cim_seq", "assign act_we1 = r4_v && (lay != 4'd9) && l8_act;",
     "assign act_we1 = r4_v && (lay != 4'd9) && l8_act && (r4_d.p != 7'd60);", ["seq", "cim", "s"], "bench"),
    ("S: L8 written to ACT in full runs", "cim_seq", "assign l8_act  = !NO_L8A || (lay != 4'd8) || (stop_q == 4'd9);", "assign l8_act  = 1'b1;",
     ["seq", "cim", "s"], "seq.l8_act_only_when_stopped_after_l8"),
    ("S: L8 not written even when the run stops after it", "cim_seq", "assign l8_act  = !NO_L8A || (lay != 4'd8) || (stop_q == 4'd9);",
     "assign l8_act  = !NO_L8A || (lay != 4'd8);", ["seq", "cim", "s"], "bench"),
    ("S: two results drained, three counted", "cim_seq", "d_rem <= d_rem - 5'd2;", "d_rem <= (d_rem > 5'd2) ? d_rem - 5'd3 : 5'd0;", ["seq", "cim", "s"], "bench"),
]


# Not a defect at today's timing: gathering and requantisation already space the network's passes at least 9 cycles apart, so the sequencer
# never reaches the engine while it is not ready. This mutant must change nothing; the check becomes necessary at D > 9 (TIMING_SENSITIVITY.md).
EQUIVALENT = [("sequencer issue ignores ready", "cim_seq", "assign issue   = i_on && i_full && eng_ready_i && (pace == 5'd0) && (tq_cnt != 3'd4);",
               "assign issue   = i_on && i_full && (pace == 5'd0) && (tq_cnt != 3'd4);", ["seq", "cim"], "engine.start_only_when_ready"),
              # S: with two lanes a pass drains in at most 8 cycles, and the engine spaces passes 9 apart, so the drain pacing never binds
              ("S: no drain pacing with two lanes", "cim_seq", "if (issue) pace <= RQ_2L ? 5'((6'(i_tag.n) + 6'd1) >> 1) - 5'd1 :",
               "if (issue) pace <= RQ_2L ? 5'd0 :", ["seq", "cim", "s"], "seq.results_drained_before_pop")]


def run(item):
    i, (name, mod, old, new, bench, want) = item
    wd = tempfile.mkdtemp(prefix="mut_")
    rtl = os.path.join(wd, "rtl")
    os.makedirs(rtl)
    files = []
    for m in RTL:
        s = open(os.path.join(ROOT, "rtl", m + ".sv")).read()
        if m == mod:
            if s.count(old) != 1:
                return dict(name=name, want=want, error=f"pattern occurs {s.count(old)} times")
            s = s.replace(old, new)
            if "gnt_d" in new:                       # a helper register for this mutant
                s = s.replace("logic        rvalid_q;", "logic        rvalid_q, gnt_d;\n  always_ff @(posedge clk_i) gnt_d <= gnt;")
        p = os.path.join(rtl, m + ".sv")
        open(p, "w").write(s)
        files.append(p)
    if bench[0] in ("log2", "pdm", "engine"):
        files = [os.path.join(rtl, {"log2": "cim_log2fix16", "pdm": "pdm_dec", "engine": "cim_engine"}[bench[0]] + ".sv")]
    env = dict(os.environ, OUT=wd, NOCOV="1", VJOBS="4", EXTRA_RTL=" ".join(files), RUN_TIMEOUT="420")
    try:
        r = subprocess.run(["bash", os.path.join(ROOT, "sim", "run_cov.sh"), "one"] + bench, env=env, capture_output=True, text=True)
        logs = [p for p in os.listdir(wd) if p.endswith(".log") and not p.startswith("build_")]
        txt = open(os.path.join(wd, logs[0]), errors="replace").read() if logs else r.stdout
        # a run that hangs or stops early never prints its final lines: each failure also prints when it first occurs
        fails = sorted(set(re.findall(r"^ASSERT \S+\.(\S+) active \d+ fail [1-9]", txt, re.M)) |
                       set(re.findall(r"^ASSERTION FAILED (\S+) ", txt, re.M)))
        first = re.findall(r"^ASSERTION FAILED (.*)$", txt, re.M)[:2]
        bench_pass = bool(re.search(r"^PASS", txt, re.M))
        caught = (not bench_pass) if want == "bench" else (want.split(".")[1] in fails)
        res = dict(name=name, want=want, caught=caught, fails=fails, bench_pass=bench_pass,
                   finished=bool(re.search(r"^(PASS|FAIL)", txt, re.M)), first=first)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    bench_state = "passes" if res["bench_pass"] else ("fails" if res["finished"] else "hangs")
    print(f"{'CAUGHT' if res['caught'] else 'MISSED'}  {name}: wants {want}; assertions failing {res['fails']}; bench {bench_state}", flush=True)
    return res


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with cf.ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(run, enumerate(M)))
        equiv = list(ex.map(run, enumerate(EQUIVALENT)))
    with open(os.path.join(ROOT, "model", "verif_mutants.json"), "w") as f:
        json.dump(dict(mutants=res, equivalent=equiv), f, indent=1)
    caught = sum(bool(r.get("caught")) for r in res)
    for r in equiv:
        print(f"equivalent at today's timing: {r['name']}: bench {'passes' if r['bench_pass'] else 'FAILS'}, assertions failing {r['fails']}")
    if any(r["fails"] or not r["bench_pass"] for r in equiv):
        print("an 'equivalent' mutant changed something: it is not equivalent")
        return 1
    silent = [r["name"] for r in res if r.get("caught") and r.get("bench_pass")]
    print(f"{caught} of {len(res)} mutants caught by the named assertion")
    if silent:
        print("caught only by assertions, the benches alone pass: " + "; ".join(silent))
    return 0 if caught == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
