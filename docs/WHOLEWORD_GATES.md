# Whole-word decision gates (frozen before the confirmatory test)

Frozen: 2026-11-22, before any of the test data below was generated or scored for these candidates. Baseline: commit
`550e57e`, with `model/` unchanged from it. The question came from the student (2026-11-22): replace the firmware's
5-in-a-row confirmation by deciding on the right 1-s window.

**Not blind.** Three pilots ran the same day on the `rule_sweep` streams (`model/rule_sweep/`, generator seeds
seed + RMS for words 11, 12, speech 21 and noise 31, at RMS 30, 100 and 300). They produced the candidates and every
parameter below. On those streams, at one decision every 20 ms:

| Pilot result | Recall | Speech reports/h | Noise reports/h | Network runs per hop (words / speech / noise) |
|---|---|---|---|---|
| Firmware rule, 20 ms (5 in a row) | 86.0 % | 430 | 3 | 100 % |
| Firmware rule, 40 ms (3 in a row) | 87.0 % | 439 | 8 | 50 % |
| Decision on the exact clip window (oracle) | 82.5 % | 68 | 0 | — |
| A (whole-word runs, every hop) | 83.2 % | 94 | 0 | 100 % |
| B (whole-word trigger, s = 8, at most 4) | 83.2 % | 104 | 0 | 3.3 / 4.9 / 1.6 % |
| U (every 16 hops, sum of 2) | 84.7 % | 114 | 2.2 | 6.25 % |

U was found in the third pilot, which looked for the strongest baseline of B (the same number of network runs spread
evenly). On the pilot streams B was not better than U. The thresholds below are frozen so that the test cannot move
them. Any later change is reported as post hoc.

## Question

The window path keeps the 50 hop maxima of the 1-s window for MLPerf Tiny's peak normalisation. Can it also tell
which windows hold the loud part whole, and if so:
- **A:** can deciding once per run of such windows replace the confirmation rule with far fewer false alarms?
- **B:** can running the network only on such windows save energy beyond what the existing k-hop hardware gives?
- **U:** if not, is the existing hardware at its lowest rate, with a short logit sum, already enough?

## Definitions

Hop `h` has maximum sample `bm[h]` (the window path's block maximum). The window of hop `h` (h ≥ 49) has:
- `W = max bm[h−49 .. h]`, the window maximum;
- `eL = max bm[h−49 .. h−47]`, the edge on the left (its first 3 hops);
- `eR = max bm[h−2 .. h]`, the edge on the right (its last 3 hops).

A window is **whole** if `W > 0`, `32·eL ≤ 11·W` and `32·eR ≤ 11·W` (α = 11/32 = 0.344, m = 3; integer, as hardware
would compute it). A **run** is a maximal set of consecutive whole windows.

| Name | Network runs on | Decision | Reported at |
|---|---|---|---|
| **A** | every hop | per run: argmax of the logits summed over every window of the run; reported if a keyword | the first hop after the run |
| **B** | only whole windows: the first window of each run, then every 8th, at most 4 per run | per run: argmax of the logits summed over those (at most 4) windows; reported if a keyword | the first hop after the run |
| **U** | every 16th hop (`NET_K` = 16, the existing 4-bit field's maximum) | argmax of the logits of the last 2 decisions summed; the firmware's reporting logic ("1 in a row": a keyword when it is the top class and was not the last one reported; reset by a non-keyword) | that decision |
| **F20, F40, F80** | every 1st, 2nd, 4th hop | the firmware rule (`hopcim_live`): 5, 3, 2 in a row | that decision |

Every decision is the golden model's int8 logits of that window. The golden model equals the RTL and the board.

**Firmware-only options** are what needs no new RTL: every rule of `rule_sweep.py` (68 rules) at k = 1, 2, 4, plus
the 19 rules of the third pilot (1–3 in a row; sum of 2–4 with 1–2 in a row; sum of 2 or 4 with margin 5 or 10 and 1–2
in a row; level 1,000 or 2,000 with 1–2 in a row) at k = 1, 2, 4, 8, 12, 16. Each is scored as the mean over its k
phases.

## Test data (never used by the pilots)

| Set | Contents |
|---|---|
| **Streams** | `rule_sweep.make` with fresh seeds: words 501–504 (2.0 h), non-keyword speech 505, 506 (1.0 h), noise 507 (1.0 h), each at RMS 30, 100 and 300. Generator seeds 531–807: none is used by `rule_sweep.py` or `exact_reuse.py`. The words are new draws from the same 4,890 test clips, so clips can repeat; recall is also reported on the words whose clip appears in no pilot stream |
| **Room** | the 8.38-h ambient runs `hopcim_recordings/far_ambient_20261107` and `…b` (board, every 20-ms decision's logits and every hop's maximum, no audio). Scored before only with the firmware-only rules (`kwlog_analyse.py --far`), never with A, B or U |

## Metrics

As `rule_sweep.evaluate`:
- **recall**: a keyword counts as found if the right class is reported within [clip start, clip end + 1 s];
- **wrong**: a different keyword is reported there, and the right one is not;
- **reports per hour**: in the speech streams, the noise streams and the room. Any keyword report there is a false alarm.

**Power [E]** comes from `energy_est.py wholeword`, using S's RTL-counter energies (`model/energy_s_rtl.json`):
- the power of S with no inference,
- plus the measured network runs per second × S's energy per inference,
- plus CPU wakes per second × 263 cycles × 28 pJ (`energy_est.py`'s CPU figure). The CPU wakes on every network run.
- A and B add 600 gate equivalents of logic leakage (the whole-word test and run state; an assumption).

The pass decision uses the baseline variant, and every variant of `ENERGY_ESTIMATE.md` §8 is reported.

## Validity of the tools (before any gate)

| ID | Condition |
|---|---|
| V1 | The scorer reproduces `rule_sweep_report.txt` on the pilot streams: F20 86.0 % / 430 / 3, F40 87.0 % / 439 / 8, F80 86.5 % / 362 / 7 |
| V2 | In every test stream, 10 random decisions equal `dscnn_int.forward` of `stream_int.window_ref` of that window |
| V3 | The room runs pass `kwlog_analyse` integrity, and F20 computed offline from their logits equals the board's own reports hop for hop. The block maxima cover every hop |
| V4 | `energy_est.py wholeword` reproduces S's 97.0 / 68.8 / 54.7 µW at k = 1 / 2 / 4 from the same parts |

If V1–V4 fail, the tool is fixed before any gate is scored.

## Gates

Reference: F20 for A (same network rate); F40 for B and U (the operating point, `EVALUATION.md` §8).

| ID | Gate | Pass if | For |
|---|---|---|---|
| G1 | Speech false alarms | reports/h on the speech streams ≤ **0.40 ×** the reference's | A, B, U |
| G2 | Recall | ≥ the reference's **− 4.0** points | A, B, U |
| G3 | Noise false alarms | reports/h on the noise streams ≤ the reference's | A, B, U |
| G4 | Room false alarms | reports/h over the 8.38 h ≤ the reference's | A, B, U |
| G5 | Power [E] | ≤ **0.75 ×** F40's in each of words, speech, noise and room | B, U |
| G6 | New RTL is needed | recall ≥ R* **+ 2.0** points, where R* is the best recall of any firmware-only option with speech reports/h ≤ the candidate's, noise reports/h ≤ the candidate's + 1.0, and power [E] on the word streams ≤ the candidate's + 2.0 µW | A, B |

G6 is the strongest-baseline test. For B it compares against the lowest rates the existing k-hop field allows. The
best firmware-only option is chosen on the test data, which favours the baseline.

## Decision

- **U** passes G1–G5: propose it as the firmware's low-power mode (`NET_K` = 16 and its rule; no RTL, no new
  bitstream), to be confirmed by a spoken run on the board.
- **A** or **B** passes G1–G4 (B also G5) **and** G6: an RTL proposal (the whole-word test inside the window path's
  `W_MAX` scan, and for B a conditional network start in `cim_audio`). It is built only if the student decides to go
  ahead.
- A candidate that fails G6 gets no RTL. It is reported as a measured negative result.

Reported beside the gates, not gated:
- latency (report hop − clip end);
- results by noise level;
- the held-out-clip recall;
- exact McNemar tests on identical keywords against the reference;
- network runs and CPU wakes per hop.
