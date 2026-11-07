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
TAG = "freeze-2026-11-07"

BOARD_SCRIPTS = ["board_run.sh", "board_suite.sh", "board_khop.sh", "xheep_env.sh"]
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
        dict(id="D3", label="[M]", text="RTL regression from a clean clone of this repository: 17 of 17 benches pass",
             evidence="freeze/regression_clean_clone.log", check=lambda: has("freeze/regression_clean_clone.log", r"17 of 17 benches pass") and not has("freeze/regression_clean_clone.log", r"(?m)^FAIL"),
             docs=[("docs/HOPCIM_PROPOSAL.md", "hopcim/sim/run_all.sh")], how="`sim/run_all.sh`, about 5 min"),
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
    ]


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
