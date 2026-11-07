#!/usr/bin/env python3
"""Score a spoken-keyword test on the board (R7, LIVE_AUDIO.md section 8).

Prompted test (firmware hopcim_kwtest): the log holds the prompts and the reported keywords, each
with its decision number (one decision per 20 ms). Trial i runs from its prompt to the next one (the
last trial to the end line). A keyword trial is
  correct  if the prompted keyword was reported in it,
  wrong    if only other keywords were,
  missed   if nothing was;
a non-keyword or quiet trial is correct if nothing was reported. Every report that is not the first
report of the prompted keyword counts as a false alarm.

Free-running demo (firmware hopcim_live) with --said w1,w2,...: the reported keywords are aligned
with the words said (edit distance) into correct, wrong, missed and extra.

Usage: score_kwtest.py <uart.log> [--said yes,stop,...]
"""
import argparse
import re
import statistics
import sys

KEYWORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
DEC_S = 0.02

RE_HEAD = re.compile(r"^KWTEST rounds (\d+) trials (\d+) period (\d+) lead (\d+)")
RE_PROMPT = re.compile(r"^\[(\d+)/(\d+)\] (?:say: (\S+)( \(not a keyword\))?|\(stay quiet\)) @(\d+)\s*$")
RE_KW = re.compile(r"^>> (\S+)(?: @(\d+))?\s*$")
RE_END = re.compile(r"^end @(\d+): .*overrun (\d+); CPU-active per decision (\d+) cycles \(max (\d+)\)")
RE_STAT = re.compile(r"^\[(\d+) s\] .*overrun (\d+); CPU-active per decision (\d+) cycles \(max (\d+)\)")


def parse(path):
    head, prompts, reports, end, stat = None, [], [], None, None
    for ln in open(path, errors="replace"):
        if m := RE_HEAD.match(ln):
            head = tuple(int(v) for v in m.groups())
        elif m := RE_PROMPT.match(ln):
            kind = "quiet" if m.group(3) is None else ("other" if m.group(4) else "keyword")
            prompts.append((int(m.group(5)), kind, m.group(3) or "(quiet)"))
        elif m := RE_KW.match(ln):
            reports.append((None if m.group(2) is None else int(m.group(2)), m.group(1)))
        elif m := RE_END.match(ln):
            end = tuple(int(v) for v in m.groups())
        elif m := RE_STAT.match(ln):
            stat = tuple(int(v) for v in m.groups())
    return head, prompts, reports, end, stat


def align(said, got):
    """Edit-distance alignment: (correct, wrong, missed, extra) and the aligned pairs."""
    n, m = len(said), len(got)
    d = [[i + j if i == 0 or j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (said[i - 1] != got[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, pairs = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (said[i - 1] != got[j - 1]):
            pairs.append((said[i - 1], got[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((said[i - 1], None))
            i -= 1
        else:
            pairs.append((None, got[j - 1]))
            j -= 1
    pairs.reverse()
    c = sum(a == b for a, b in pairs)
    w = sum(a is not None and b is not None and a != b for a, b in pairs)
    return c, w, sum(b is None for _, b in pairs), sum(a is None for a, _ in pairs), pairs


def hw_line(t):
    return f"hardware: overrun {t[1]}, CPU-active {t[2]} cycles per decision (max {t[3]})" if t else "hardware: no statistics line"


def score_free(reports, stat, said):
    got = [w for _, w in reports]
    print(f"free-running log: {len(got)} keywords reported; {hw_line(stat)}")
    print("  reported: " + " ".join(got))
    if said is None:
        print("  no prompts in the log and no --said list, so nothing to score against")
        return 0
    c, w, mi, ex, pairs = align(said, got)
    print(f"  said {len(said)}: correct {c} ({100 * c / max(len(said), 1):.1f} %), wrong {w}, missed {mi}; "
          f"{ex} extra reports")
    for a, b in pairs:
        if a != b:
            print(f"    said {a or '-':8s} reported {b or '-'}")
    return 0


def score_prompted(head, prompts, reports, end):
    ends = [p[0] for p in prompts[1:]]               # trial i: decisions (prompt i, ends[i]]
    if end is None:
        print("no end line: the test was cut short; the last prompt is not scored")
        trials = prompts[:-1]
    else:
        ends.append(end[0])
        trials = prompts
    t_reports = [[] for _ in trials]
    before = after = 0
    for at, w in reports:
        if at is None:
            continue
        if not trials or at <= trials[0][0]:
            before += 1
            continue
        for i, (p, _, _) in enumerate(trials):
            if p < at <= ends[i]:
                t_reports[i].append((at, w))
                break
        else:
            after += 1
    rounds, n_trial, period, _ = head
    print(f"keyword test: {len(trials)} of {n_trial} trials scored ({rounds} round{'s' * (rounds != 1)}, "
          f"{period * DEC_S:.1f} s each); "
          f"{hw_line(end)}")

    res = {"keyword": [0, 0, 0], "other": [0, 0], "quiet": [0, 0]}     # correct, wrong, missed / correct, reported
    per_kw = {k: [0, 0, []] for k in KEYWORDS}                        # correct, trials, other reports
    fa, lat, fa_list = 0, [], []
    for (p, kind, word), rep in zip(trials, t_reports):
        names = [w for _, w in rep]
        if kind == "keyword":
            per_kw[word][1] += 1
            if word in names:
                res["keyword"][0] += 1
                per_kw[word][0] += 1
                lat.append((rep[names.index(word)][0] - p) * DEC_S)
                extra = list(names)
                extra.remove(word)
            else:
                res["keyword"][1 if names else 2] += 1
                extra = names
            per_kw[word][2] += extra
        else:
            res[kind][0 if not names else 1] += 1
            extra = names
        fa += len(extra)
        fa_list += [(word, w) for w in extra]

    nk = sum(res["keyword"])
    c, w, mi = res["keyword"]
    print(f"  keywords      {nk:3d} trials: correct {c} ({100 * c / max(nk, 1):.1f} %), wrong {w}, missed {mi}")
    for kind, name in (("other", "non-keywords"), ("quiet", "quiet")):
        n = sum(res[kind])
        print(f"  {name:13s} {n:3d} trials: nothing reported in {res[kind][0]}, a keyword reported in {res[kind][1]}")
    print(f"  false alarms  {fa} reports that were not the prompted keyword (or repeated it)")
    if lat:
        lat.sort()
        print(f"  prompt to report (correct trials): median {statistics.median(lat):.2f} s, "
              f"range {lat[0]:.2f} - {lat[-1]:.2f} s (includes reading the prompt and saying the word)")
    if before or after:
        print(f"  not scored: {before} reports before the first prompt, {after} after the last trial")
    print("  per keyword (correct / trials; other reports in its trials):")
    for k in KEYWORDS:
        cc, nn, oth = per_kw[k]
        if nn:
            print(f"    {k:6s} {cc}/{nn}" + (f"   {', '.join(oth)}" if oth else ""))
    alarms = [f"{p} -> {r}" for p, r in fa_list if p not in KEYWORDS]
    if alarms:
        print("  reports in non-keyword and quiet trials: " + ", ".join(alarms))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--said", help="free-running log: the words said, in order, comma separated")
    a = ap.parse_args()
    head, prompts, reports, end, stat = parse(a.log)
    said = None if a.said is None else [w.strip().lower() for w in a.said.split(",") if w.strip()]
    if said is not None and any(w not in KEYWORDS for w in said):
        sys.exit(f"--said: only the keywords {', '.join(KEYWORDS)}")
    if head is None or not prompts:
        return score_free(reports, stat, said)
    return score_prompted(head, prompts, reports, end)


if __name__ == "__main__":
    sys.exit(main())
