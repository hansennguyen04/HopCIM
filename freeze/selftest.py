#!/usr/bin/env python3
"""The freeze check must be able to fail: break one thing at a time in a throw-away clone, and each break
must be caught by the expected line. Writes nothing in this repository. Usage: selftest.py"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
B = "freeze/board/logs/"


def replace(path, old, new, count=0):
    def f(root):
        p = root / path
        s = p.read_text(errors="replace")
        assert old in s, (path, old)
        p.write_text(s.replace(old, new) if not count else s.replace(old, new, count))
    return f


BREAKS = [
    ("a board log reports other CPU-active cycles", replace(B + "hopcim_live_nexys_board_20261105_231010.uart.log", "per decision 263 cycles", "per decision 264 cycles"), r"FAIL\s+C1"),
    ("the sequencer's cycle count changes", replace(B + "hopcim_seq_nexys_board_20261028_090933.uart.log", "sequencer 85420 cycles", "sequencer 85421 cycles"), r"FAIL\s+B3"),
    ("a spoken keyword disappears from the log", replace(B + "hopcim_kwtest_k4_nexys_board_20261107_152720.uart.log", ">> yes @560\n", ""), r"FAIL\s+C5"),
    ("an overrun appears in a speaker log", replace(B + "hopcim_kwtest_k2_nexys_board_20261107_152243.uart.log", "overrun 0;", "overrun 1;"), r"FAIL\s+C5"),
    ("the reuse evaluation has a mismatch", replace("model/reuse_results.json", '"logit_mismatches": 0', '"logit_mismatches": 1'), r"FAIL\s+E1"),
    ("the energy total changes", replace("model/energy_run.log", "110.3", "110.4"), r"FAIL\s+F1"),
    ("the gate-level check has a mismatch", replace("sim/xsim_audio_nl_div15/run.log", "0 skipped (RTL X); 0 mismatches", "0 skipped (RTL X); 1 mismatches"), r"FAIL\s+D1"),
    ("a document quotes another number", replace("docs/EVALUATION.md", "85,420", "85,421"), r"FAIL\s+B3"),
    ("a stale statement returns", replace("docs/HOPCIM_PROPOSAL.md", "### 15.1 Status", "The full pipeline and f_min are open\n### 15.1 Status"), r"FAIL\s+docs/HOPCIM_PROPOSAL.md"),
    ("a bitstream hash differs", replace("freeze/MANIFEST.json", '"sha256": "', '"sha256": "0', 1), r"FAIL\s+(bitstreams|models)"),
    ("the X-HEEP patch is damaged", replace("freeze/xheep/tracked_changes.patch", "+", "-", 40), r"FAIL\s+X-HEEP snapshot hashes"),
    ("a firmware file differs from the live one", replace("freeze/xheep/overlay/sw/applications/hopcim_live/main.c", "#define", "#undef ", 1), r"FAIL\s+31 overlay"),
    ("a board script differs from the live one", replace("freeze/board/board_run.sh", "#!/", "# !/", 1), r"FAIL\s+board scripts"),
]


def check(root):
    env = {**os.environ, "HOPCIM_PROJECT": str(PROJECT), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(root / "freeze" / "freeze.py"), "check"], capture_output=True, text=True, env=env)


def main():
    missed = 0
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "clean" / "hopcim"
        subprocess.run(["git", "clone", "-q", str(ROOT), str(clean)], check=True)
        r = check(clean)
        ok = r.returncode == 0
        print(f"{'ok    ' if ok else 'FAIL  '}unchanged clone passes")
        missed += not ok
        if not ok:
            print(r.stdout[-1500:])
        for i, (name, breakit, want) in enumerate(BREAKS):
            w = Path(tmp) / f"b{i}" / "hopcim"
            shutil.copytree(clean, w, ignore=shutil.ignore_patterns(".git"))
            breakit(w)
            r = check(w)
            caught = r.returncode != 0 and re.search(want, r.stdout) is not None
            print(f"{'caught' if caught else 'MISSED'}  {name}")
            missed += not caught
    print(f"\n{len(BREAKS)} breaks, {missed} problems")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
