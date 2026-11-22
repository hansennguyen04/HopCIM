# Whole-word decisions: results against the frozen gates

Model study, 2026-11-22. The gates are in [`WHOLEWORD_GATES.md`](WHOLEWORD_GATES.md):
- frozen at commit `8e12400`, before any test stream existed;
- SHA-256 `4484fd51…7004e5` (`WHOLEWORD_GATES.sha256`).

Nothing here was built in RTL. Every decision is the golden model's int8 logits, which equal the RTL and the board.
Power is a model estimate [E] (`ENERGY_ESTIMATE.md`), not an FPGA measurement.

## Verdict

| Candidate | G1 speech alarms | G2 recall | G3 noise | G4 room | G5 power | G6 RTL needed | Decision (frozen rule) |
|---|---|---|---|---|---|---|---|
| **A**: one decision per run of whole windows, network every hop | pass | pass | pass | pass | — | **fail** | no RTL |
| **B**: network only on whole windows | pass | **fail** | pass | pass | pass | **fail** | no RTL |
| **U**: existing k-hop field at k = 16, sum of 2 decisions | pass | pass | pass | pass | pass | — | propose as the firmware's low-power mode, after a spoken run on the board |

**What this means:**
- **The localiser works.** The window path's 50 hop maxima do find the whole-word windows: deciding there cuts speech false alarms by 75 %.
- **But choosing windows is not worth new hardware.** The existing hardware, run uniformly at a low rate with a short logit sum, does as well or better at the same or lower power. The network's decision is stable over the ~25 hops during which a word lies whole in the window, so even one decision every 320 ms lands inside that span once or twice per word.
- **Neither A nor B needs RTL.** U needs only a firmware change: `NET_K` = 16 and its rule, on the S bitstream.

## Validity (V1–V4): all pass

| ID | Result |
|---|---|
| V1 | The scorer on the pilot streams gives F20 86.0 % / 430 / 3, F40 87.0 % / 439 / 8, F80 86.5 % / 362 / 7, as `rule_sweep_report.txt` |
| V2 | 210 decisions (10 per test stream) equal `dscnn_int.forward(stream_int.window_ref(…))`: 0 differ |
| V3 | Both room runs: integrity clean, a hop maximum for every hop, decisions contiguous. F20 computed offline equals the board's own reports hop for hop (22 + 8) |
| V4 | The pricing gives S's 97.0 / 68.8 / 54.7 µW at k = 1 / 2 / 4. The largest error over all 19 variants is 6 × 10⁻¹⁴ µW, i.e. P = P0 + E / (20 ms · k) holds exactly, with P0 = 40.58 µW and E = 1,128.15 nJ per decision (network 1,120.79 + CPU 7.36) |

## Results

**Test data:**
- streams: 2.00 h of words (1,446 keywords), 1.00 h of non-keyword speech, 1.00 h of noise, at RMS 30 / 100 / 300;
- room: 8.38 h.

Firmware-only rules are the mean over their k phases.

| Design | Recall | Wrong | Speech /h | Noise /h | Room /h (95 %) | Network runs per hop (words / speech / noise / room) | Power [E], µW (words / speech / noise / room) | Decided, hops after the clip's end: median [p10, p90] |
|---|---|---|---|---|---|---|---|---|
| F20 (5 in a row, 20 ms) | 87.1 % | 5.3 % | 482 | 2.0 | 3.58 (2.41–5.11) | 100 % | 97.0 | −20 [−31, −6] |
| F40 (3 in a row, 40 ms) | 88.2 % | 5.0 % | 477 | 6.0 | 4.12 (2.81–5.67) | 50 % | 68.8 | −19 [−31, −5] |
| F80 (2 in a row, 80 ms) | 87.8 % | 5.2 % | 386 | 6.0 | 3.67 (2.51–5.25) | 25 % | 54.7 | −18 [−30, −4] |
| A | 83.9 % | 6.0 % | 119 | 2.0 | 1.07 (0.49–2.04) | 100 % | 97.0 | +14 [1, 27] |
| B | 82.9 % | 5.9 % | 110 | 5.0 | 1.31 (0.66–2.35) | 3.3 / 4.8 / 1.7 / 0.02 % | 42.4 / 43.3 / 41.6 / 40.6 | +14 [1, 27] |
| U | 85.6 % | 5.3 % | 121 | 1.8 | 2.21 (1.27–3.39) | 6.25 % | 44.1 | −11 [−25, 6] |

### Gates

| Gate | A (against F20) | B (against F40) | U (against F40) |
|---|---|---|---|
| G1 speech /h ≤ 0.40 × reference | 119 ≤ 192.8 ✓ | 110 ≤ 190.6 ✓ | 121 ≤ 190.6 ✓ |
| G2 recall ≥ reference − 4.0 | 83.9 ≥ 83.1 ✓ | 82.9 ≥ 84.2 **✗** | 85.6 ≥ 84.2 ✓ |
| G3 noise /h ≤ reference | 2.0 ≤ 2.0 ✓ | 5.0 ≤ 6.0 ✓ | 1.8 ≤ 6.0 ✓ |
| G4 room /h ≤ reference | 1.07 ≤ 3.58 ✓ | 1.31 ≤ 4.12 ✓ | 2.21 ≤ 4.12 ✓ |
| G5 power ≤ 0.75 × F40 (51.6 µW), every set | — | 43.3 (worst set) ✓ | 44.1 ✓ |
| G6 recall ≥ R* + 2.0 | 83.9 < 86.0 + 2 **✗** | 82.9 < 83.2 + 2 **✗** | — |

**What beats A and B in G6:**
- **For A,** the best qualifying firmware-only option (148 qualify) is "sum of 4, 1 in a row" at k = 8. It gets 86.0 % recall with 101 speech and 1.6 noise reports per hour at 47.6 µW. That is 2.1 points more recall than A, at half its power.
- **For B,** the best of the 17 qualifying options is "sum of 3, 1 in a row" at k = 16. It gets 83.2 % recall with 69 speech and 0.9 noise reports per hour at 44.1 µW. B does not beat it even by a point.
- **Energy is not the reason either.** B's power on the word streams is 0.96 × U's [E] (0.92 × in the room), i.e. 1.7 µW less with words, 3.5 µW less in the room.

### Reported, not gated

**By noise RMS** (recall %, speech reports per hour, noise reports per hour):

| RMS | F20 | F40 | F80 | A | B | U |
|---|---|---|---|---|---|---|
| 30 | 91.6 / 525 / 3.0 | 92.5 / 524 / 6.0 | 92.4 / 416 / 6.8 | 89.3 / 96 / 0.0 | 88.1 / 102 / 3.0 | 90.4 / 105 / 1.9 |
| 100 | 87.8 / 471 / 3.0 | 89.2 / 468 / 7.5 | 88.6 / 386 / 6.8 | 83.7 / 111 / 6.0 | 82.9 / 105 / 12.0 | 86.1 / 120 / 2.2 |
| 300 | 81.8 / 450 / 0.0 | 83.1 / 438 / 4.5 | 82.5 / 357 / 4.5 | 78.7 / 150 / 0.0 | 77.8 / 123 / 0.0 | 80.4 / 138 / 1.3 |

**Held-out clips.** On the 1,177 keywords whose clip appears in no pilot stream, recall is F20 87.0 %, F40 88.1 %, F80 87.7 %, A 83.9 %, B 82.9 % and U 85.7 %. That is the same order as on all keywords, so clip reuse between pilot and test did not favour the candidates.

**Paired, exact McNemar** (keywords found only by the first / only by the second):

| Pair | Only first / only second | p |
|---|---|---|
| A vs F20 | 8 / 54 | 1.7 × 10⁻⁹ |
| B vs F20 | 9 / 69 | 1.4 × 10⁻¹² |
| B vs F40, phase 0 | 6 / 80 | 1.3 × 10⁻¹⁷ |
| B vs F40, phase 1 | 5 / 85 | 7.5 × 10⁻²⁰ |
| U vs F40, phase 0, each of U's 16 phases | — | 1.3 × 10⁻¹⁰ to 0.0016 |

Every candidate's recall loss is real, not noise.

**U by phase.** Over its 16 phases, recall is 85.1–86.2 % and speech reports are 111–131 per hour.

**U's power in every variant.** U is 0.64 × F40 in the baseline variant. Across the 19 variants it ranges from 0.46 × (Colonnade array) to 0.94 × (one tile per macro, where leakage dominates). At 4.096 MHz it is 42.9 µW (`wholeword_energy.txt`).

**Latency.** U decides a median 11 hops (220 ms) before the clip's end, 8 hops later than F40. A and B decide 14 hops after the clip's end, because they wait for the run of whole windows to end.

**Post hoc (chosen on the test data, so not a claim).** At k = 12, "sum of 3, 1 in a row" gets 85.4 % / 98 / 1.5 at 45.3 µW. At k = 8, "sum of 4, 1 in a row" gets 86.0 % / 101 / 1.6 at 47.6 µW. A choice between those and U should be frozen and tested again, ideally on spoken words.

## Errata in the frozen gate document

Two counts and one section reference are wrong in `WHOLEWORD_GATES.md`. The definitions there are exact, and those definitions are what was applied:
- "every rule of `rule_sweep.py` (68 rules)": `rule_sweep.rules()` returns **73** rules;
- "the 19 rules of the third pilot": the list given has **21**;
- "the operating point, `EVALUATION.md` §8": the operating point is stated in **§7**.

After removing duplicates, the firmware-only family has 315 options. The gates were not changed.

## Limits

**The streams are synthetic.** They are isolated GSC clips added to noise beds, 2–4 s apart. Words in running speech, or a word right after another, are not tested. The room run is one room with no speech.

**U at 320 ms is untested on spoken words.** That matters because the real-speaker test of 2026-11-07 found 42 of 50 keywords at 80 ms, against 48 at 20 ms (p = 0.109; `EVALUATION.md` §7). The cause of that 80-ms loss is untested. A spoken run with the same prompts (`hopcim_kwtest`, `-DNET_K=16` and U's rule) is the confirmation the frozen decision asks for, before U becomes a claim.

**U's power is an estimate [E].** It is dominated by the 40.6 µW that does not depend on the network (front end, window path, audio, leakage). That share is the next target if power matters, not the network rate.

**B's noise alarms.** On the noise streams B reported 5.0 per hour (12 per hour at RMS 100), where the pilot showed 0. Whole windows in noise are transients, and summing at most 4 of them does not suppress them.

## The spoken confirmation of U (set 2026-11-22, before the run)

**What runs.** A prompted logger, the same 65 prompts as `hopcim_kwtest` (5 rounds of 10 keywords, 2 other words,
1 quiet trial, 4 s each):
- firmware `planenest/hw/hopcim_kwlog_noaudio_r5_nexys.elf`, built with `-DKWLOG_AUDIO=0 -DKW_TEST=5`;
- it sends every 20-ms decision's logits and each hop's level, and no audio;
- the same tree, rebuilding the room runs' `hopcim_kwlog_noaudio_nexys.elf`, gives an identical loadable image.

Because every decision is logged, F40, U, A and B are scored on the same speech: `wholeword.py spoken <run dir>`. On a
synthetic prompted run built from `words_s501_r30`, that scorer gives the stream scorer's recall to the decimal for all
six designs.

**U is confirmed if both hold** over the pooled spoken runs:
1. its keywords correct (mean over its 16 phases) is at least F40's (mean over its 2 phases) minus 4 points (G2's line);
2. no pairing of a U phase with an F40 phase shows a significant loss: exact McNemar p < 0.05 with F40 ahead.

Other-word and quiet alarms are reported beside them. With one speaker and 50 keyword trials per run, a loss smaller
than about 5 words cannot be told from chance, so two runs (100 trials) are better than one.

## Result of the spoken run (2026-11-22)

**The run.** `hopcim_recordings/spk01_wholeword_run1`, 15:50–15:55:
- one speaker (spk01, the student), 65 prompts (50 keywords, 10 other words, 5 quiet);
- firmware `hopcim_kwlog_noaudio_r5_nexys.elf` (sha256 `34ee2e59…`), bitstream `xheep_nexys_bscan_cim128k.bit`;
- 13,250 decisions, 0 lost, no overrun; `frames.bin` sha256 `2feb6dbc…`;
- the firmware rule computed offline equals the board's 68 reports hop for hop.

**Scores** (mean over phases, with the range over phases):

| Design | Keywords correct of 50 | Phases | Other-word alarms of 10 | Quiet alarms of 5 | Paired with F40 (only F40 / only it; p) |
|---|---|---|---|---|---|
| F20 | 48.0 (96.0 %) | 48 | 2.0 | 0 | 0–2 / 0–1; 0.5–1 |
| F40 | 48.5 (97.0 %) | 47–50 | 2.0 | 0 | — |
| F80 | 48.8 (97.5 %) | 47–50 | 1.8 | 0 | 0–3 / 0–3; 0.25–1 |
| A | 47.0 (94.0 %) | 47 | 1.0 | 0 | 0–3 / 0; 0.25–1 |
| B | 47.0 (94.0 %) | 47 | 0.0 | 0 | 0–3 / 0; 0.25–1 |
| **U** | **48.5 (97.0 %)** | 47–50 | 1.0 | 0 | 0–3 / 0–3; 0.25–1 |

**U is confirmed** by the rule set before the run:
- its 48.5 is at least F40's 48.5 minus 4 points;
- no pairing shows a significant loss (smallest p 0.25).

This is one run of one speaker. It shows no loss of 5 words or more; it cannot show equality.

On these identical words, 80 ms also lost nothing (48.8). The 42 of 50 of the earlier 80-ms speaker run came from a
separate run with different speech, so the rate itself is not shown to cause that loss.

### The second run and the pooled result

**The run.** `spk01_wholeword_run2`, 16:04–16:09, same speaker, firmware and bitstream:
- 13,250 decisions, 0 lost, no overrun; `frames.bin` sha256 `c002a845…`;
- the firmware rule offline equals the board's 64 reports.

**Run 2 alone.** Every design found 45 of 50 (U 44.6, B 43). Five words were missed by every design alike, so the
network itself missed them; no rule changes that.

**Pooled over both runs** (100 keywords, 20 other words, 10 quiet trials):

| Design | Keywords correct of 100 | Phases | Other-word alarms of 20 | Quiet alarms of 10 | Paired with F40 (only F40 / only it; p) |
|---|---|---|---|---|---|
| F20 | 93.0 | 93 | 3.0 | 0 | 0–2 / 0–1; 0.5–1 |
| F40 | 93.5 | 92–95 | 3.0 | 0 | — |
| F80 | 93.8 | 92–95 | 2.5 | 0 | 0–3 / 0–3; 0.25–1 |
| A | 92.0 | 92 | 2.0 | 0 | 0–3 / 0; 0.25–1 |
| B | 90.0 | 90 | 1.0 | 0 | 2–5 / 0; 0.0625–0.5 |
| **U** | **93.1** | 91–95 | 2.0 | 0 | 0–5 / 0–3; 0.125–1 |

**U is confirmed over 100 spoken keywords:**
- 93.1 is at least 93.5 − 4;
- no pairing is significant (smallest p 0.125).

This holds for one speaker, the student. A claim about speakers in general needs more of them (`SAME_AUDIO.md` §4:
consent, codes, no audio needed with this build).

### Why the phases split: conv1's stride (post hoc)

Every design with an even k splits into two groups of phases: F40 gets 50 or 47, F80 50/48/50/47, and U alternates
49–50 with 47–48.

**The three words behind the split.** Two are "left" and one is "stop". On these, the decision changes on every hop.
The word is named on one hop and not the next (`*.*.*.*`) for 15 hops in a row.

**The cause is the network's first layer.** conv1 (10 × 4) has stride 2 in time (`dscnn_int.forward`). A window that
moves by one hop samples the other half of the frames, so the network sees two different inputs on alternate hops. A
word near the decision boundary can be named in one half and not the other.

**It is common.** Of all hops where the class changes, 43 % (synthetic word streams) and 41 % (this run) are such
flip-and-back changes.

**Why even k is affected.** A rate with even k only ever sees one of the two halves. Which half helps a word depends on
when the word was said.

**An odd k sees both halves.** Post hoc, not frozen, so not a claim:
- "sum of 2" at k = 15 gets 86.2 % / 128 / 1.5 on the test streams, against 85.6 % / 121 / 1.8 at k = 16;
- on this spoken run it gets 49.1 of 50, with the worst phase 48, against 48.5 and 47 at k = 16;
- its power is 44.3 µW [E], against 44.1.

Choosing between k = 15 and k = 16 needs a new frozen test.

## U in the live firmware (2026-11-22)

**How to build it.** `hopcim_live` (and `hopcim_kwtest`) built with `-DNET_K=16 -DKW_SUM=2` runs mode U on the S
bitstream. Only the firmware changes. Details are in `LIVE_AUDIO.md` §9.

**Checks:**
- the rule in `sw/kw_rule.h` equals the one scored here: `model/test_kw_rule.py`, 479 cases, 0 differ, 3 of 3 broken copies caught;
- the default builds are byte-identical to those before the change.

**Board demo** (2026-11-22, the first run at k = 16):
- 31 decisions per 10 s, 0 missed, overrun 0, keywords reported as spoken;
- CPU-active 875 cycles per decision, which makes U 44.16 µW [E] instead of 44.11;
- details in `LIVE_AUDIO.md` §9.

**The complete scripted test** of the live build (`hopcim_kwtest_u`, 65 prompts; new speech, not paired):
- 48 of 50 keywords correct, against 48 / 47 / 42 for the earlier 20 / 40 / 80-ms runs;
- 1 of 10 non-keywords and 0 of 5 quiet trials had a report;
- 9 extra reports, against 32 / 35 / 28;
- the median from prompt to report was 1.28 s, against 1.24 s at 40 ms;
- overrun 0. Table in `LIVE_AUDIO.md` §9.

The first attempt stopped at prompt 29: the board script's default timeout is 120 s.

## Files

| File | What it is |
|---|---|
| `model/wholeword.py` | the study: `run` makes the 21 test streams into `model/wholeword/` (git-ignored, with the room runs' parsed logits and hop maxima, no audio); `report` runs V1–V4 and the gates |
| `model/wholeword_report.txt`, `model/wholeword_results.json` | the report and every number above, including all 315 firmware-only options |
| `model/energy_est.py wholeword` | the pricing (`ww_parts`, `ww_power`, `ww_check`); the default run of `energy_est.py` is unchanged (`energy_results.json` byte-identical) |
| `model/wholeword_energy.txt`, `model/wholeword_energy.json` | power of every design in every variant |
| `sw/kw_rule.h`, `sw/applications/hopcim_live/main.c` (`KW_SUM`) | U's rule in the firmware |
| `model/test_kw_rule.py` | the firmware rule against the scored rule, on the host |

**Reproduce** (about 50 core-minutes, then 10 minutes):

```bash
cd hopcim/model
~/venvs/planenest/bin/python wholeword.py run 14
~/venvs/planenest/bin/python wholeword.py report 15 > wholeword_report.txt
~/venvs/planenest/bin/python energy_est.py wholeword > wholeword_energy.txt
```
