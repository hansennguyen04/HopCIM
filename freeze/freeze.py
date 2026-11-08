#!/usr/bin/env python3
"""Freeze and check HopCIM's results. Standard library only.

  freeze.py snapshot    copy what lives outside hopcim/ into freeze/: the X-HEEP changes as a patch on its
                        base commit, firmware and board scripts, UART logs, the accuracy gate, the novelty
                        documents, tool versions. Proves that the patch reproduces the live X-HEEP files.
  freeze.py manifest    write freeze/MANIFEST.json (hashes of bitstreams, ELFs, models, dataset) and
                        freeze/RESULTS.md (every headline claim, its evidence, how to reproduce it)
  freeze.py check       recompute all hashes and check every claim against its evidence and the documents;
                        exit status 0 only if everything agrees

Bitstreams, ELFs and the dataset stay outside git: the manifest pins them by hash.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = Path(os.environ.get("HOPCIM_PROJECT", ROOT.parent))   # set it to check a clone made elsewhere
HW = PROJECT / "planenest" / "hw"
XHEEP = HW / "x-heep"
GATE = PROJECT / "kws_gate"
DISCOVERY = PROJECT / "research_discovery" / "20261026_application_driven"
PILOT = PROJECT / "recim-rv" / "accelforge_pilot"
TAG = "freeze-2026-11-08"

BOARD_SCRIPTS = ["board_run.sh", "board_suite.sh", "board_khop.sh", "board_kwlog.sh", "xheep_env.sh"]
GATE_FILES = ["GATE_RESULTS.md", "GATE2_MANIFEST.md", "gate2_eval.py", "gate2_frozen.sha256", "gate2_results.json", "gate2_run.log"]
NOVELTY_FILES = ["NOVELTY_ASSESSMENT.md", "RESEARCH_DISCOVERY.md"]
MODELS = {
    "mlperf_tiny_kws_ref_int8": GATE / "mlperf_tiny_kws" / "trained_models" / "kws_ref_model.tflite",
    "mlperf_tiny_kws_ref_float32": GATE / "mlperf_tiny_kws" / "trained_models" / "kws_ref_model_float32.tflite",
    "dscnn_params": ROOT / "model" / "dscnn_params.npz",
    "mfcc_tables": ROOT / "model" / "mfcc_tables.npz",
}
DATASET = GATE / "data" / "test"
# Which frozen result each bitstream carries
BITSTREAM_USE = {
    "xheep_nexys_bscan_cim128k.bit": "every-k-hops option; the speaker test at 20, 40, 80 ms; the current live design (15.36 MHz)",
    "xheep_nexys_bscan_cim128mic5.bit": "lowest real-time clock (5.12 MHz)",
    "xheep_nexys_bscan_cim128mic.bit": "first live microphone path (R7, 15.36 MHz)",
    "xheep_nexys_bscan_pdm_cim128w_cim.bit": "window path (R5b), cim_fe",
    "xheep_nexys_bscan_pdm_cim128w_ded.bit": "window path (R5b), B3",
    "xheep_nexys_bscan_pdm_cim128s_cim.bit": "layer sequencer (R6b), cim_fe",
    "xheep_nexys_bscan_pdm_cim128s_ded.bit": "layer sequencer (R6b), B3",
}
STALE = [  # statements that were true once and must not come back
    ("docs/HOPCIM_PROPOSAL.md", "The full pipeline and f_min are open"),
    ("docs/HOPCIM_PROPOSAL.md", "The final board measurement with the network on the CIM is open"),
    ("docs/HOPCIM_PROPOSAL.md", "a scripted spoken-keyword test at the board"),
    ("docs/HOPCIM_PROPOSAL.md", "planned: CIC + FIR"),
    ("docs/HOPCIM_PROPOSAL.md", "which is planned for Phase 1"),
    ("docs/HOPCIM_PROPOSAL.md", "the scripted accuracy test is ready"),
    ("docs/LIVE_AUDIO.md", "waits for a speaker"),
    ("docs/EVALUATION.md", "has not been run with a speaker"),
    ("docs/ENERGY_ESTIMATE.md", "**ZigZag-IMC itself has not been run.**"),
    ("docs/HOPCIM_PROPOSAL.md", "the tool itself has not been run"),
    ("docs/LIVE_AUDIO.md", "covers a restart, and the firmware enables the path once"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


def tool_output(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "not available"


def project_path(p):
    """A path as seen from the project folder, wherever this clone is."""
    return f"hopcim/{p.relative_to(ROOT)}" if p.is_relative_to(ROOT) else str(p.relative_to(PROJECT))


def tree_hash(root):
    """Files, names and contents: sha256 over 'relative path, file hash' in sorted order."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(x for x in Path(root).rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(root).as_posix()}\0{sha(p)}\n".encode())
        n += 1
    return h.hexdigest(), n


def xheep_lists():
    changed = [x for x in run("git", "diff", "--name-only", "HEAD", cwd=XHEEP).splitlines() if x]
    others = [x for x in run("git", "ls-files", "--others", "--exclude-standard", cwd=XHEEP).splitlines() if x]
    overlay = [x for x in others if not re.match(r"build_\w*/", x) and x != "hw/ip/hopcim"]
    return changed, overlay


# ------------------------------------------------------------------ snapshot
def verify_patch(patch, changed):
    """The patch on a clean copy of the base commit must reproduce the live files byte for byte."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "archive", "HEAD"], cwd=XHEEP, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=tmp, check=True)
        bad = [c for c in changed if sha(Path(tmp) / c) != sha(XHEEP / c)]
    return bad


def snapshot():
    x = HERE / "xheep"
    if x.exists():
        shutil.rmtree(x)
    (x / "overlay").mkdir(parents=True)
    changed, overlay = xheep_lists()
    base = run("git", "rev-parse", "HEAD", cwd=XHEEP).strip()
    patch = x / "tracked_changes.patch"
    patch.write_bytes(subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=XHEEP, capture_output=True, check=True).stdout)
    bad = verify_patch(patch, changed)
    if bad:
        sys.exit(f"the patch does not reproduce: {bad}")
    for rel in overlay:
        if rel.endswith(".core"):
            sys.exit(f"{rel}: a .core file under hopcim/ would be found by FuseSoC through hw/ip/hopcim")
        d = x / "overlay" / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(XHEEP / rel, d)
    gen = XHEEP / "hw" / "system" / "x_heep_system.sv"
    (x / "generated").mkdir()
    shutil.copy2(gen, x / "generated" / gen.name)
    (x / "BASE.txt").write_text(
        f"https://github.com/x-heep/x-heep.git\n{base}\n\n"
        "To rebuild this tree: clone at that commit, `git apply tracked_changes.patch`, copy overlay/ over it,\n"
        "and link the IP: ln -s <path to hopcim> hw/ip/hopcim. generated/ is what the build generator produced\n"
        "(ignored by X-HEEP's git), kept for comparison.\n")

    b = HERE / "board"
    if b.exists():
        shutil.rmtree(b)
    (b / "logs").mkdir(parents=True)
    for s in BOARD_SCRIPTS:
        shutil.copy2(HW / s, b / s)
    logs = sorted(p for p in HW.glob("*_board_*.uart.log") if p.stat().st_size > 0) + sorted(HW.glob("khop_board_*.log"))
    for p in logs:
        shutil.copy2(p, b / "logs" / p.name)

    for name, src, files in (("gate", GATE, GATE_FILES), ("novelty", DISCOVERY, NOVELTY_FILES)):
        d = HERE / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        for f in files:
            shutil.copy2(src / f, d / f)

    e = HERE / "env"
    if e.exists():
        shutil.rmtree(e)
    e.mkdir()
    env = f"source {HW}/xheep_env.sh >/dev/null; "
    lines = []
    for label, cmd in (("Verilator", "verilator --version"), ("RISC-V GCC", "riscv64-unknown-elf-gcc --version | head -1"),
                       ("Vivado", "vivado -version | head -1"), ("Python, system", "python3 -V"), ("Kernel", "uname -srm")):
        lines.append(f"{label}: " + subprocess.run(["bash", "-c", env + cmd], capture_output=True, text=True).stdout.strip())
    (e / "tools.txt").write_text("\n".join(lines) + "\n")
    for name, py in (("planenest", Path.home() / "venvs/planenest/bin/python"), ("kwsgate", Path.home() / "venvs/kwsgate/bin/python"),
                     ("accelforge_pilot", PILOT / ".venv312/bin/python")):
        (e / f"python_{name}.txt").write_text(tool_output(str(py), "-V") + "\n" + tool_output(str(py), "-m", "pip", "freeze"))
    print(f"snapshot: X-HEEP {base[:12]}, {len(changed)} files patched (patch verified byte for byte), {len(overlay)} overlay files; "
          f"{len(logs)} board logs")


# ------------------------------------------------------------------ manifest
def artefacts():
    bits = {}
    for p in sorted((HW / "bitstreams").glob("*.bit")):
        e = {"sha256": sha(p), "bytes": p.stat().st_size}
        t = p.with_suffix(".timing.rpt")
        if t.exists():
            e["timing_rpt_sha256"] = sha(t)
        if p.name in BITSTREAM_USE:
            e["carries"] = BITSTREAM_USE[p.name]
        bits[p.name] = e
    elfs = {p.name: sha(p) for p in sorted(HW.glob("*.elf"))}
    models = {k: {"path": project_path(p), "sha256": sha(p)} for k, p in MODELS.items()}
    tree, n = tree_hash(DATASET)
    return dict(bitstreams=bits, elfs=elfs, models=models,
                dataset={"name": "Google Speech Commands v2, test split as used by the MLPerf Tiny preprocessing",
                         "path": str(DATASET.relative_to(PROJECT)), "clips": len(list(DATASET.rglob("*.wav"))), "files": n, "tree_sha256": tree})


def source_hashes():
    x = HERE / "xheep"
    tree, n = tree_hash(x / "overlay")
    return {"xheep_base_commit": (x / "BASE.txt").read_text().splitlines()[1],
            "xheep_patch_sha256": sha(x / "tracked_changes.patch"),
            "xheep_overlay": {"files": n, "tree_sha256": tree},
            "xheep_generated_system_sha256": sha(x / "generated" / "x_heep_system.sv")}


def manifest():
    m = dict(tag=TAG, note="Hashes of everything the frozen results depend on that is not in this repository.",
             tools=(HERE / "env" / "tools.txt").read_text().splitlines(), sources=source_hashes(), **artefacts())
    (HERE / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    rows = []
    for c in claims():
        rows.append(f"| {c['id']} | {c['label']} | {c['text']} | `{c['evidence']}` | {c['how']} |")
    (HERE / "RESULTS.md").write_text(
        f"# Frozen results ({TAG})\n\n"
        "Generated by `freeze/freeze.py manifest`. `freeze/freeze.py check` recomputes every line against its "
        "evidence file, and checks that the documents quote the same numbers.\n\n"
        "Labels, as in the proposal: **[M]** measured in this project; **[E]** estimate or model, not measured.\n\n"
        "| ID | Label | Claim | Evidence | Reproduce |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (HERE / "RESULTS_FREEZE.md").write_text(results_freeze_md(m))
    print(f"manifest: {len(m['bitstreams'])} bitstreams, {len(m['elfs'])} ELFs, {len(m['models'])} models, "
          f"dataset {m['dataset']['clips']} clips; {len(rows)} claims")


# ------------------------------------------------------------------ claims
def jget(path, *keys):
    v = json.loads((ROOT / path).read_text())
    for k in keys:
        v = v[k]
    return v


def has(path, pattern):
    return re.search(pattern, (ROOT / path).read_text(errors="replace")) is not None


def speaker():
    sys.path.insert(0, str(ROOT / "model"))
    from score_kwtest import parse
    out = {}
    for k, log in ((1, "hopcim_kwtest_nexys_board_20261107_151805.uart.log"), (2, "hopcim_kwtest_k2_nexys_board_20261107_152243.uart.log"),
                   (4, "hopcim_kwtest_k4_nexys_board_20261107_152720.uart.log")):
        head, prompts, reports, end, _ = parse(str(HERE / "board" / "logs" / log))
        ends = [p[0] for p in prompts[1:]] + [end[0]]
        ok = none = 0
        for (p, kind, word), e in zip(prompts, ends):
            if kind != "keyword":
                continue
            names = [n for a, n in reports if p < a <= e]
            ok += word in names
            none += not names
        out[k] = (ok, none, end[1])
    return out


B = "freeze/board/logs/"
SEQ_LOG = B + "hopcim_seq_nexys_board_20261028_090933.uart.log"
CPU_LOG = B + "hopcim_cpu_nexys_board_20261028_092050.uart.log"
NET1_LOG = B + "hopcim_net1_nexys_board_20261028_090921.uart.log"
STREAM_LOG = B + "hopcim_stream_nexys_board_20261028_091143.uart.log"
LIVE_LOG = B + "hopcim_live_nexys_board_20261105_231010.uart.log"
LIVE5_LOG = B + "hopcim_live_nexys_5m12_board_20261030_080317.uart.log"
KHOP_LOG = B + "khop_board_20261105_231009.log"
UTIL = "syn/ooc_cim_endpoint_FE_DED0_PDM_CLKDIV15/utilization_hier.rpt"


def timing_check():
    h = {x["name"]: x for x in jget("model/timing_results.json", "hop")}
    want = dict(baseline=5, own_clock=5, **{"2bpc_nopipe": 5, "1bpc_nopipe": 5, "1bpc_pipe": 5, "colonnade_bp": 5}, two_steps=8, four_steps=15, colonnade_np=59)
    return (all(h[k]["nmin"] == v for k, v in want.items()) and all(x["at"]["passed"] and x["at"]["overrun"] == 0 and x["at"]["hazards"] == 0 for x in h.values())
            and all(not x["below"]["passed"] and x["below"]["overrun"] > 0 for x in h.values()) and h["baseline"]["worst"] == 97583)


# the two complete ambient runs (the snapshot of 2026-11-07 held 1.46 h of them)
FAR_LOG = "model/kwlog_checks/far_ambient.log"
FAR_TEXT = ("Ambient room, 8.38 h without audio: the firmware rule reported 3.6, 4.1, 2.6 times per hour at 20, 40, 80 ms, all of it in the "
            "first 1.38 h (evening) and none in the 7.0 h after; at 40 ms 0.1 per hour with a window-maximum gate of 2,000 (board)")
FAR_DOC = "**4.1**"


def far_check():
    pooled = [("20 ms  5 in a row", "30\\.0", "3\\.6"), ("40 ms  3 in a row", "34\\.5", "4\\.1"), ("80 ms  3 in a row", "21\\.5", "2\\.6"),
              ("40 ms  level >= 2000, 3 in a row", "1\\.0", "0\\.1")]
    quiet = all(has(FAR_LOG, rf"hour {h} \(1\.00 h\):\s+0\.0,\s+0\.0,\s+0\.0;") for h in range(2, 9))
    return (has(FAR_LOG, r"all runs together, 8\.38 h") and quiet and has(FAR_LOG, r"hour 1 \(1\.00 h\):\s+9\.5,")
            and all(has(FAR_LOG, rf"{name}\s+reports\s+{n},\s+{rate} per hour") for name, n, rate in pooled))


def claims():
    """id, label, text, evidence file, check, documents that must quote it, how to reproduce."""
    py = "`~/venvs/planenest/bin/python`"
    return [
        dict(id="A1", label="[M]", text="Integer CIM-MFCC with the unchanged int8 DS-CNN: 91.78 %, float MFCC 91.66 % (4,890 clips, TFLite default kernels, McNemar p = 0.44)",
             evidence="model/gate2_int_results.json",
             check=lambda: jget("model/gate2_int_results.json", "n") == 4890 and round(jget("model/gate2_int_results.json", "accuracy", "int8|int"), 2) == 91.78
             and round(jget("model/gate2_int_results.json", "accuracy", "int8|ref"), 2) == 91.66 and round(jget("model/gate2_int_results.json", "mcnemar_int8")[2], 2) == 0.44,
             docs=[("docs/HOPCIM_PROPOSAL.md", "91.78 %")], how=f"{py} model/gate2_int.py, needs TensorFlow (`~/venvs/kwsgate`)"),
        dict(id="A2", label="[M]", text="Integer network equals TFLite's reference kernels in every layer on all clips: 91.80 %",
             evidence="model/dscnn_check.log",
             check=lambda: has("model/dscnn_check.log", r"layer mismatches \(clips\): \{'conv1': 0, 'dw1': 0, 'pw1': 0, 'dw2': 0, 'pw2': 0, 'dw3': 0, 'pw3': 0, 'dw4': 0, 'pw4': 0, 'pool': 0, 'fc': 0\}")
             and has("model/dscnn_check.log", r"ours 91\.80 %, TFLite reference kernels 91\.80 %"),
             docs=[("docs/EVALUATION.md", "91.80 %")], how="`model/dscnn_int.py`, needs TensorFlow"),
        dict(id="A3", label="[M]", text="Dedicated front-end B3: 91.60 %, passes the same gate",
             evidence="model/gate2_ded_results.json",
             check=lambda: round(jget("model/gate2_ded_results.json", "accuracy", "int8|int"), 2) == 91.60 and jget("model/gate2_ded_results.json", "gate_pass") is True,
             docs=[("docs/EVALUATION.md", "91.60 %")], how="`model/gate2_int.py`, B3 option"),
        dict(id="A4", label="[M]", text="Front-end tile mapping equals the golden model on 239,610 frames of 4,890 clips",
             evidence="model/cim_map_full.log", check=lambda: has("model/cim_map_full.log", r"4890 clips, 239610 frames, lraw mismatches 0, feature mismatches 0"),
             docs=[("docs/HOPCIM_PROPOSAL.md", "239,610")], how=f"{py} model/cim_map.py"),
        dict(id="B1", label="[M]", text="Network on the CPU alone, plain C: 47,974,801 cycles per inference (board)",
             evidence=CPU_LOG, check=lambda: has(CPU_LOG, r"total 47974801 cycles") and has(CPU_LOG, r"0 wrong -> PASS"),
             docs=[("docs/EVALUATION.md", "47,974,801")], how="`board/board_run.sh hopcim_cpu_nexys.elf <bit>`"),
        dict(id="B2", label="[M]", text="CPU driving the CIM passes: 13,020,998 cycles per inference (board)",
             evidence=NET1_LOG, check=lambda: has(NET1_LOG, r"13020998 cycles") and has(NET1_LOG, r"logit mismatches 0"),
             docs=[("docs/EVALUATION.md", "13,020,998")], how="`board_run.sh hopcim_net1_nexys.elf <bit>`"),
        dict(id="B3", label="[M]", text="Layer sequencer: 85,420 cycles per inference, go to interrupt, 60 runs bit-exact (board)",
             evidence=SEQ_LOG, check=lambda: has(SEQ_LOG, r"sequencer 85420 cycles") and has(SEQ_LOG, r"60 runs bit-exact -> PASS"),
             docs=[("docs/EVALUATION.md", "85,420")], how="`board_run.sh hopcim_seq_nexys.elf <bit>`"),
        dict(id="B4", label="[M]", text="Streaming with the window path in hardware: 100 hops, 51 decisions, no mismatch (board)",
             evidence=STREAM_LOG, check=lambda: has(STREAM_LOG, r"100 hops, 51 decisions, 0 hop and 0 decision mismatches -> PASS"),
             docs=[("docs/EVALUATION.md", "4,447")], how="`board_run.sh hopcim_stream_nexys.elf <bit>`"),
        dict(id="C1", label="[M]", text="Live microphone path: 500 decisions per 10 s, no overrun, 263 CPU-active cycles per decision (board, 15.36 MHz)",
             evidence=LIVE_LOG, check=lambda: has(LIVE_LOG, r"\[10 s\] 500 decisions .*overrun 0; CPU-active per decision 263 cycles"),
             docs=[("docs/EVALUATION.md", "**263**")], how="`board_run.sh hopcim_live_nexys.elf bitstreams/xheep_nexys_bscan_cim128k.bit`"),
        dict(id="C2", label="[M]", text="Real microphone audio: logits equal the golden model at 25 of 25, 13 of 13 and 7 of 7 decisions at 20, 40, 80 ms (board)",
             evidence=KHOP_LOG, check=lambda: all(has(KHOP_LOG, p) for p in (r"25 with the window inside the recording: 0 differ", r"13 with the window inside the recording: 0 differ", r"7 with the window inside the recording: 0 differ")),
             docs=[("docs/LIVE_AUDIO.md", "25 of 25")], how="`board/board_khop.sh`"),
        dict(id="C3", label="[M]", text="Lowest real-time clock 5.12 MHz: no overrun, 500 decisions per 10 s (board)",
             evidence=LIVE5_LOG, check=lambda: has(LIVE5_LOG, r"\[10 s\] 500 decisions .*overrun 0") and has(LIVE5_LOG, r"overrun 0"),
             docs=[("docs/EVALUATION.md", "5.12 MHz")], how="`board_run.sh hopcim_live_nexys_5m12.elf bitstreams/xheep_nexys_bscan_cim128mic5.bit`"),
        dict(id="C4", label="[M]", text="Every k hops: 500, 250, 125 decisions per 10 s with no overrun (board)",
             evidence=KHOP_LOG, check=lambda: all(has(KHOP_LOG, p) for p in (r"\[10 s\] 500 decisions .*overrun 0", r"\[10 s\] 250 decisions .*overrun 0", r"\[10 s\] 125 decisions .*overrun 0")),
             docs=[("docs/LIVE_AUDIO.md", "| **Board:** decisions per 10 s | 500 | 250 | 125 |")], how="`board/board_khop.sh`"),
        dict(id="C5", label="[M]", text="One speaker: 48, 47, 42 of 50 keywords at 20, 40, 80 ms; 0, 1, 6 with nothing reported; no overrun (board)",
             evidence=B + "hopcim_kwtest_nexys_board_20261107_151805.uart.log",
             check=lambda: speaker() == {1: (48, 0, 0), 2: (47, 1, 0), 4: (42, 6, 0)},
             docs=[("docs/LIVE_AUDIO.md", "**48**"), ("docs/LIVE_AUDIO.md", "**42**"), ("docs/EVALUATION.md", "| Keywords correct, of 50 | 48 | 47 | 42 |")],
             how="`model/score_kwtest.py <log>`; a new speaker run gives new numbers"),
        dict(id="D1", label="[M]", text="Gate level: cim_audio RTL equals its Vivado netlist, 0 mismatches over 4,493,107,200 output bits",
             evidence="sim/xsim_audio_nl_div15/run.log", check=lambda: has("sim/xsim_audio_nl_div15/run.log", r"4493107200 output bits compared, 0 skipped \(RTL X\); 0 mismatches") and has("sim/xsim_audio_nl_div15/run.log", r"(?m)^PASS"),
             docs=[("docs/LIVE_AUDIO.md", "4.49 billion")], how="`sim/run_audio_nl.sh`, about 70 min"),
        dict(id="D2", label="[M]", text="Endpoint with cim_fe, out of context: 7,635 LUT, 5,819 FF, 145 DSP",
             evidence=UTIL, check=lambda: has(UTIL, r"\| cim_endpoint\s+\|\s+\(top\) \|\s+7635 \|.*\|\s+5819 \|.*\|\s+145 \|"),
             docs=[("docs/LIVE_AUDIO.md", "7,635 (+18)")], how="`vivado -mode batch -source syn/ooc.tcl -tclargs cim_endpoint FE_DED=0 PDM_CLKDIV=15`"),
        dict(id="D3", label="[M]", text="RTL regression from a clean clone of this repository: 28 of 28 benches pass",
             evidence="freeze/regression_clean_clone.log", check=lambda: has("freeze/regression_clean_clone.log", r"28 of 28 benches pass") and not has("freeze/regression_clean_clone.log", r"(?m)^FAIL"),
             docs=[("docs/HOPCIM_PROPOSAL.md", "hopcim/sim/run_all.sh")], how="`sim/run_all.sh`, about 10 min"),
        dict(id="D4", label="[M]", text="RTL regression in the main tree, with the directed benches of extreme input, stalls and overrun: 28 of 28 benches pass",
             evidence="freeze/regression_main.log", check=lambda: has("freeze/regression_main.log", r"28 of 28 benches pass") and not has("freeze/regression_main.log", r"(?m)^FAIL"),
             docs=[("docs/VERIFICATION.md", "28 of 28 benches pass")], how="`sim/run_all.sh`, about 10 min"),
        dict(id="D5", label="[M]", text="Assertions: 47, all exercised, none failing over the 28 benches; every uncovered line explained",
             evidence="model/verif_results.json",
             check=lambda: len(jget("model/verif_results.json", "assertions")) == 47 and all(v["active"] > 0 and v["fail"] == 0 for v in jget("model/verif_results.json", "assertions").values())
             and len(jget("model/verif_results.json", "benches")) == 28 and all(jget("model/verif_results.json", "benches").values()),
             docs=[("docs/VERIFICATION.md", "47, all exercised, none failing")], how="`sim/run_cov.sh 5`, then `python3 model/verif_report.py`"),
        dict(id="D6", label="[M]", text="Mutation test: 16 of 16 broken designs caught by the assertion named for each",
             evidence="model/verif_mutants.json",
             check=lambda: len(jget("model/verif_mutants.json", "mutants")) == 16 and all(m["caught"] for m in jget("model/verif_mutants.json", "mutants")),
             docs=[("docs/VERIFICATION.md", "16 of 16 broken designs caught")], how="`python3 model/verif_mutants.py 4`, about 8 min"),
        dict(id="D7", label="[M]", text="Post-synthesis, whole endpoint with cim_fe: every layer of a clip and one full inference exact on the Vivado netlist",
             evidence="sim/xsim_endpoint_nl/run.log",
             check=lambda: has("sim/xsim_endpoint_nl/run.log", r"RESULT \(medium\): 0 errors") and has("sim/xsim_endpoint_nl/run.log", r"(?m)^PASS"),
             docs=[("docs/VERIFICATION.md", "`sim/run_endpoint_nl.sh`")], how="`sim/run_endpoint_nl.sh`, about 80 min"),
        dict(id="G1", label="[M]", text="Same-audio logger on the board at 921,600 baud: 2,850 of 2,850 decisions on real microphone audio equal the golden model",
             evidence="model/kwlog_checks/engineering_r1.log", check=lambda: has("model/kwlog_checks/engineering_r1.log", r"2850 decisions recomputed from the recorded audio, 0 differ"),
             docs=[("docs/SAME_AUDIO.md", "2,850 of 2,850")], how="`planenest/hw/board_kwlog.sh`, then `model/kwlog_analyse.py`"),
        dict(id="G2", label="[M]", text="GSC streams, firmware's rule: keyword recall 86.0, 87.0, 86.5 % at 20, 40, 80 ms. Synthetic streams, no board",
             evidence="model/rule_sweep_results.json",
             check=lambda: [round(100 * jget("model/rule_sweep_results.json", "table", n, k, "words", "recall"), 1) for n, k in (("5 in a row", "1"), ("3 in a row", "2"), ("2 in a row", "4"))] == [86.0, 87.0, 86.5],
             docs=[("docs/SAME_AUDIO.md", "86.0 %")], how=f"{py} model/rule_sweep.py run 4; report"),
        dict(id="G3", label="[M]", text=FAR_TEXT,
             evidence="model/kwlog_checks/far_ambient.log", check=far_check,
             docs=[("docs/SAME_AUDIO.md", FAR_DOC)], how=f"{py} model/kwlog_analyse.py --far <runs>; recordings pinned by hash in the log"),
        dict(id="H1", label="[M]", text="Timing sweep in simulation: lowest clock 5.12 MHz at every assumption with D <= 9; 8.19, 15.36, 60.42 MHz at D = 18, 36, 153; bit-exact at the lowest clock, overruns one divider below",
             evidence="model/timing_results.json", check=timing_check,
             docs=[("docs/TIMING_SENSITIVITY.md", "**5.12 MHz**"), ("docs/TIMING_SENSITIVITY.md", "60.42 MHz")], how="`python3 model/timing_sweep.py seq; hop`, about 20 min"),
        dict(id="H2", label="[M]", text="A macro on its own fast clock (D = 1) changes the network's cycles from 85,352 to 85,232; the one-line model is within 1.39 % over 40 timings",
             evidence="model/timing_results.json",
             check=lambda: any(x["d"] == 1 and x["lat"] == 1 and x["cycles"] == 85232 for x in jget("model/timing_results.json", "seq")) and any(x["d"] == 9 and x["lat"] == 5 and x["cycles"] == 85352 for x in jget("model/timing_results.json", "seq")),
             docs=[("docs/TIMING_SENSITIVITY.md", "1.39 %")], how="`python3 model/timing_sweep.py report`"),
        dict(id="E1", label="[M]", text="Activation reuse in software: exact on 174,216 decisions over 3,500 s (0 logit, 0 layer mismatches). Not in RTL",
             evidence="model/reuse_results.json",
             check=lambda: jget("model/reuse_results.json", "decisions") == 174216 and jget("model/reuse_results.json", "logit_mismatches") == 0 and jget("model/reuse_results.json", "layer_mismatches") == 0,
             docs=[("docs/REUSE_EVALUATION.md", "174,216")], how=f"{py} model/exact_reuse.py suite, about 15 min"),
        dict(id="F1", label="[E]", text="Estimate at 28 nm: 1,389 nJ per inference; 110, 75, 58 µW at 20, 40, 80 ms. Not a measurement",
             evidence="model/energy_run.log",
             check=lambda: has("model/energy_run.log", r"total\s+1389\.3") and has("model/energy_run.log", r"total\s+110\.3\s+75\.3\s+57\.9"),
             docs=[("docs/ENERGY_ESTIMATE.md", "**110**"), ("docs/EVALUATION.md", "110 µW")], how="`python3 model/energy_est.py` (see `docs/ENERGY_ESTIMATE.md` §10)"),
        dict(id="F2", label="[E]", text="Estimate: simple reuse saves a net 13.0-14.1 µW at 40 ms with leakage included, using ZigZag-IMC's transcribed equations",
             evidence="model/energy_run.log", check=lambda: has("model/energy_run.log", r"40 ms, 1 cache, \+32 KB\s+15\.8-16\.9\s+2\.8\s+\+13\.0\.\.\+14\.1"),
             docs=[("docs/ENERGY_ESTIMATE.md", "+13.0 to +14.1 µW")], how="`python3 model/energy_est.py`"),
        dict(id="F3", label="[E]", text="ZigZag-IMC itself against the transcribed equations: ratio 1.000 on the pointwise layers, 1.029 overall at the measured activity; with its raw array figure 140, 90, 65 µW",
             evidence="model/zigzag_results.json",
             check=lambda: round(jget("model/zigzag_results.json", "group16_macros1", "rows", "pw", "ratio"), 3) == 1.0 and round(jget("model/zigzag_results.json", "like_for_like", "ratio"), 3) == 1.029
             and [round(jget("model/zigzag_results.json", "like_for_like", "power", k, "with_tool_uW")) for k in ("1", "2", "4")] == [140, 90, 65],
             docs=[("docs/ENERGY_ESTIMATE.md", "140 / 90 / 65 µW")], how="`~/venvs/zigzag/bin/python model/zigzag_run/run_zigzag.py`, about 1 min"),
        dict(id="S1", label="[M]", text="Schedule model of the sequencer equals the RTL (85,352 cycles, all 10 layers, every access count); its replay of B0, S, A, B, C is exact on all 4,890 clips",
             evidence="model/streamfuse_results.json", check=sf_model_check,
             docs=[("docs/STREAMFUSE_STUDY.md", "total 85,352 (all 10 layers equal)")], how=f"{py} model/streamfuse_sched.py all, about 7 min"),
        dict(id="S2", label="[E]", text="Model, not built: S (sign gating, second requantisation lane, no L8 write) 67,841 cycles, 1,119.8 nJ, 68.8 µW at 40 ms; B 0.837 x S and 0.575 x the storage; C 0.821 x S; A fails (0.903)",
             evidence="model/streamfuse_energy.json", check=sf_energy_check,
             docs=[("docs/STREAMFUSE_STUDY.md", "**1,119.8**"), ("docs/STREAMFUSE_STUDY.md", "80,544 (0.575)")],
             how="`python3 model/energy_est.py streamfuse` (after streamfuse_sched.py all)"),
    ]


def sf_model_check():
    r = jget("model/streamfuse_results.json")
    v1 = r["V1"]
    ex = r["exact"]
    return (v1["passed"] and v1["total"] == v1["rtl_total"] == 85352 and v1["exact_layers"] == 10 and v1["segments"] == v1["rtl"]
            and all(ex[k]["clips"] == 4890 and ex[k]["bad_clips"] == 0 and ex[k]["bad_values"] == 0 and ex[k]["bad_logit_clips"] == 0 for k in ("B0", "S", "A", "B", "C"))
            and all(m.get("caught") for k, m in r["self_test"].items() if k not in ("B without its source interlock", "B with a 25-row map"))
            and r["self_test"]["B without its source interlock"]["hazards"] > 0 and r["self_test"]["B with a 25-row map"]["bad_values"] == 0)


def sf_energy_check():
    e = jget("model/streamfuse_energy.json")
    g = e["gates_G2"]
    c = jget("model/streamfuse_results.json", "cycles")
    return (e["V3"]["passed"] and all(round(g[k]["S_nJ"], 1) == 1119.8 and round(g[k]["S_power_uW"]["2"], 1) == 68.8 for k in "ABC") and round(g["B"]["S_power_uW"]["2"], 1) == 68.8 and c["S"]["total"] == 67841
            and round(g["B"]["ratio"], 3) == 0.837 and round(g["C"]["ratio"], 3) == 0.821 and round(g["A"]["ratio"], 3) == 0.903 and not g["A"]["G2"]
            and c["B"]["storage_bits"] == 80544 and c["S"]["storage_bits"] == 140000)


# ------------------------------------------------------------------ RESULTS_FREEZE.md: every thesis table -> claims -> command, output
THESIS_TABLES = [
    ("Accuracy: the unchanged model with the integer CIM-MFCC, TFLite equality, B3", ["A1", "A2", "A3", "A4"]),
    ("Where the cycles go: CPU only, CPU-driven CIM, hardware sequencer, window path", ["B1", "B2", "B3", "B4"]),
    ("Live system: decisions, overrun, CPU-active cycles, real audio, lowest clock, k hops", ["C1", "C2", "C3", "C4"]),
    ("Spoken keywords, same-audio logger, GSC streams, false alarms", ["C5", "G1", "G2", "G3"]),
    ("Macro timing sensitivity", ["H1", "H2"]),
    ("Verification: regression, assertions, mutants, netlists", ["D1", "D3", "D4", "D5", "D6", "D7"]),
    ("FPGA resources", ["D2"]),
    ("Energy estimate [E]", ["F1", "F2", "F3"]),
    ("Activation reuse against a lower cadence", ["E1", "F2"]),
    ("Dataflow options (StreamFuse model study)", ["S1", "S2"]),
]
FINAL_TABLE = [  # the baseline table of docs/HopCIM_Specific_Next_Steps_and_Professor_Advice.md section 18
    ("CPU only", "B1", "measured on the board"),
    ("CPU-driven CIM", "B2", "measured on the board"),
    ("R6b sequencer (today)", "B3, F1, D2", "measured; energy [E]"),
    ("S (sign gating, second lane, no L8 write)", "S2", "model only; RTL not built at this freeze"),
    ("40-ms S", "-", "not built at this freeze"),
    ("StreamFuse-B (optional)", "S2", "model only"),
    ("B3 dedicated MFCC", "A3", "measured; resources in docs/B3_BASELINE.md"),
]
SPEAKER_LOGS = ["hopcim_kwtest_nexys_board_20261107_151805.uart.log", "hopcim_kwtest_k2_nexys_board_20261107_152243.uart.log",
                "hopcim_kwtest_k4_nexys_board_20261107_152720.uart.log"]
MODEL_VERSIONS = [
    ("energy model", ["model/energy_est.py", "model/energy_tool_run.py", "model/energy_components.json", "model/energy_activity.json"]),
    ("StreamFuse schedule model", ["model/streamfuse_sched.py", "model/cim_net_map.py", "model/dscnn_int.py"]),
    ("DeFiNES cross-check", ["model/defines_run/run_defines.py", "model/defines_run/hopcim_hw.py", "model/defines_run/hopcim_dscnn_wl.py"]),
    ("integer golden models", ["model/mfcc_int.py", "model/stream_int.py", "model/pdm_dec.py", "model/dscnn_params.npz", "model/mfcc_tables.npz"]),
]


def results_freeze_md(m):
    cl = {c["id"]: c for c in claims()}
    L = [f"# Results freeze ({TAG})", "",
         "Generated by `freeze/freeze.py manifest`; `freeze/freeze.py check` fails if it is out of date. Every thesis table "
         "is traceable to the commit that tag `" + TAG + "` names, a command, and an output file.", "",
         "## 1. Freeze point", "",
         f"- Git tag `{TAG}` in `hopcim/` (local; no remote). Earlier freezes: `freeze-2026-11-07`, `freeze-2026-11-07b`.",
         f"- X-HEEP base commit `{m['sources']['xheep_base_commit']}`, changes as `freeze/xheep/tracked_changes.patch` "
         f"(sha256 `{m['sources']['xheep_patch_sha256'][:16]}…`), overlay {m['sources']['xheep_overlay']['files']} files.", "",
         "## 2. Model and data", "", "| Item | Path | sha256 |", "|---|---|---|"]
    for k, v in m["models"].items():
        L.append(f"| {k} | `{v['path']}` | `{v['sha256']}` |")
    ds = m["dataset"]
    L += [f"| dataset ({ds['clips']} clips, {ds['files']} files) | `{ds['path']}` | tree `{ds['tree_sha256']}` |", "",
          "## 3. Tools", ""] + [f"- {t}" for t in m["tools"]] + [
          "- Python environments: `freeze/env/python_planenest.txt` (models, energy, StreamFuse), `python_kwsgate.txt` "
          "(TensorFlow, accuracy gate), `python_accelforge_pilot.txt` (CACTI, AccelForge); ZigZag-IMC in `~/venvs/zigzag`, "
          "DeFiNES in `~/venvs/defines` (`~/tools/energy/DeFiNES` at 7097d60)", "",
          "## 4. Bitstreams that carry a result", "", "| Bitstream | sha256 | Carries |", "|---|---|---|"]
    for name, e in m["bitstreams"].items():
        if "carries" in e:
            L.append(f"| `{name}` | `{e['sha256'][:16]}…` | {e['carries']} |")
    L += ["", f"All {len(m['bitstreams'])} bitstreams and {len(m['elfs'])} ELFs are pinned in `freeze/MANIFEST.json`.", "",
          "## 5. Speaker logs", "", "| Log | sha256 |", "|---|---|"]
    for s in SPEAKER_LOGS:
        L.append(f"| `freeze/board/logs/{s}` | `{sha(HERE / 'board' / 'logs' / s)[:16]}…` |")
    L += ["", "## 6. Model versions", "", "| Model | File | sha256 |", "|---|---|---|"]
    for name, files in MODEL_VERSIONS:
        for f in files:
            L.append(f"| {name} | `{f}` | `{sha(ROOT / f)[:16]}…` |")
    L += ["", "## 7. Thesis tables", "", "| Table | Claim | Label | Result | Command | Output |", "|---|---|---|---|---|---|"]
    for title, ids in THESIS_TABLES:
        for i in ids:
            c = cl[i]
            L.append(f"| {title} | {i} | {c['label']} | {c['text']} | {c['how']} | `{c['evidence']}` |")
    L += ["", "## 8. The final baseline table (advice §18) at this freeze", "", "| Configuration | Claims | State |", "|---|---|---|"]
    L += [f"| {a} | {b} | {s} |" for a, b, s in FINAL_TABLE]
    return "\n".join(L) + "\n"


def check():
    fails = []

    def say(ok, what):
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            fails.append(what)

    print("Artefacts against MANIFEST.json")
    m = json.loads((HERE / "MANIFEST.json").read_text())
    a = artefacts()
    for key in ("bitstreams", "elfs", "models", "dataset"):
        say(a[key] == m[key], f"{key}: {len(a[key])} entries" if key != "dataset" else f"dataset: {a[key]['clips']} clips, tree {a[key]['tree_sha256'][:12]}")
    say(source_hashes() == m["sources"], "X-HEEP snapshot hashes")
    rf = HERE / "RESULTS_FREEZE.md"
    say(rf.exists() and rf.read_text() == results_freeze_md(m), "RESULTS_FREEZE.md is current")

    print("Snapshot against the live files")
    changed, overlay = xheep_lists()
    base = run("git", "rev-parse", "HEAD", cwd=XHEEP).strip()
    say(base == m["sources"]["xheep_base_commit"], f"X-HEEP base commit {base[:12]}")
    say(not verify_patch(HERE / "xheep" / "tracked_changes.patch", changed), f"patch reproduces the {len(changed)} live changed files")
    say(all((HERE / "xheep" / "overlay" / r).exists() and sha(HERE / "xheep" / "overlay" / r) == sha(XHEEP / r) for r in overlay), f"{len(overlay)} overlay files equal the live ones")
    say(all(sha(HERE / "board" / s) == sha(HW / s) for s in BOARD_SCRIPTS), "board scripts equal the live ones")
    say(all(sha(HERE / "gate" / f) == sha(GATE / f) for f in GATE_FILES), "accuracy gate files equal the live ones")

    print("Claims against their evidence and the documents")
    for c in claims():
        try:
            ok = bool(c["check"]())
        except Exception as e:                                     # a missing file or key is a failed claim
            ok = False
            print(f"        {c['id']}: {e}")
        docs_ok = all(q in (ROOT / d).read_text() for d, q in c["docs"])
        say(ok and docs_ok, f"{c['id']} {c['label']} {c['text'][:88]}" + ("" if docs_ok else "  (a document does not quote it)"))

    print("Stale statements")
    for d, q in STALE:
        say(q not in (ROOT / d).read_text(), f"{d}: not \"{q}\"")

    print(f"\n{'ALL CHECKS PASS' if not fails else str(len(fails)) + ' CHECKS FAILED'}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=["snapshot", "manifest", "check"])
    a = ap.parse_args()
    if a.command == "snapshot":
        snapshot()
    elif a.command == "manifest":
        manifest()
    else:
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
