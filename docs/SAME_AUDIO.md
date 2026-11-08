# The same-audio experiment: rate, phase and confirmation rule on identical speech

Status (2026-11-07): logger, capture and analysis are built and verified on the board. **The speaker runs have not
been made; they need people at the board (§4).** A false-alarm run in the room and a GSC sweep supply the first
results (§5, §6).

## 1. Why

The earlier spoken test ran once per rate, each time on new speech (`LIVE_AUDIO.md` §9), so its comparison of 20, 40
and 80 ms was not truly paired, and it could not test other rules or phases afterwards.

A decision's logits do not depend on the rate: the front-end and window path run at every hop, and this is verified in
RTL and on the board. So a rate of one decision every k hops at phase p is simply the subsequence of the 20-ms decisions at hops
49 + p + n k. One run at 20 ms therefore holds every rate, phase and rule, on identical speech, and every comparison is
exactly paired.

## 2. The logger

`hopcim_kwlog` (firmware) streams what the hardware does over the UART, at 921,600 baud:

| Build | Sends | For |
|---|---|---|
| `hopcim_kwlog_nexys.elf` (`KW_TEST=5`) | every hop's 320 samples, every decision's 12 logits, the prompts | a spoken run |
| `hopcim_kwlog_r1_nexys.elf` (`KW_TEST=1`) | the same, 13 prompts | a short check |
| `hopcim_kwlog_noaudio_nexys.elf` (`KWLOG_AUDIO=0`, `KW_TEST=0`) | every decision's logits and each hop's maximum, minimum and RMS. **No audio leaves the board** | long false-alarm runs |

- The earlier plan was a class log, because 9600 baud cannot carry audio. The bitstream's UART is configured by software, and at
  15.36 MHz it reaches about 960 kbaud; one hop's frames are 665 bytes per 20 ms, 36 % of 921,600 baud. No new bitstream is needed:
  `xheep_nexys_bscan_cim128k.bit`.
- It uses the same prompts as `hopcim_kwtest`: same seed, same order, 4 s apart.
- Frames carry a checksum and the hop number (`model/kwlog_frames.py`). `frames.bin` holds every received byte, unchanged.
- It polls instead of sleeping, so **its CPU-active cycles say nothing about the live path**; those stay as measured in
  `LIVE_AUDIO.md`.
- It also runs hopcim_live's own rule (5 in a row), so the firmware's reports can be compared with the offline scorer.

## 3. Checks

| Check | Result |
|---|---|
| Link, 1 round at 921,600 baud, twice | 5,782 and 5,770 frames; 0 bytes skipped between frames; 0 hops lost; no overrun |
| **Every decision against the golden model, from the recorded audio** | **2,850 of 2,850 equal, in each run** (2 runs, `model/kwlog_checks/engineering_r1.log` holds the second). The earlier check covered 25 decisions |
| Firmware's reports against the offline rule | identical in hop and class (13 and 1 reports) |
| Scorer self-test, `kwlog_analyse.py --self-test` | rules, trial boundaries, phases, statistics against known values, a corrupted frame stream, broken variants |

The raw audio of these two checks was deleted afterwards.

## 4. The spoken runs (needs people)

The false-alarm logger holds the board while it runs, so stop it first: `pkill -f kwlog_capture`.

```
cd planenest/hw
./board_kwlog.sh hopcim_kwlog_nexys.elf ~/APlus_Project/hopcim_recordings/spk01_run1 0
cd ../../hopcim/model
~/venvs/planenest/bin/python kwlog_analyse.py ~/APlus_Project/hopcim_recordings/spk01_run1
```

About 4.5 min; the prompts appear in the terminal. Say each word once, 20–50 cm from the board, and nothing at `(stay quiet)`.

**Protocol.** Quiet room, normal level. Record the speaker's code, the distance and the room in `notes.txt` beside the recording.
For several speakers, make one run each, then pool them:

```
~/venvs/planenest/bin/python kwlog_analyse.py --pool ~/APlus_Project/hopcim_recordings/spk*_run1
```

**These recordings hold people's voices.** Ask each speaker's consent, and use codes rather than names. `hopcim_recordings/` is outside
the repository; only hashes and derived tables belong in it. Do not record bystanders: false-alarm runs use the build without audio.

**What it answers:**
- whether 80 ms is viable, now on identical speech, with the exact McNemar test over matched words;
- whether phase matters, as the range over phases;
- whether the 80-ms misses follow from run lengths: the untested prediction (L − 4)/4 is compared per word;
- which confirmation rule to use (§5), tested on speech that did not choose it.

## 5. Rules on GSC streams (no board, many speakers)

`model/rule_sweep.py` scores the same rates, phases and rules on GSC streams from the golden model, which equals the hardware bit for bit: 1 h of sparse words (894 words,
736 of them keywords), 0.5 h of non-keyword words only, and 1 h of noise only, at noise RMS 30, 100 and 300. Each word is one GSC clip added to noise, so **these streams are synthetic**. They
choose candidate rules without touching the live recordings; they do not replace them. Recall is the mean over phases; alarms are reports per hour.

**Rate, at the firmware's rule (the hold spans at least 80 ms):**

| A decision every | Hold | Keyword recall | On non-keyword speech | On noise | Paired with 20 ms: found only at 20 ms / only here, per phase |
|---|---:|---:|---:|---:|---|
| 20 ms | 5 | 86.0 % | 430/h | 3/h | |
| 40 ms | 3 | 87.0 % | 439/h | 8/h | 2/8 (p = 0.109), 2/10 (p = 0.039) |
| 80 ms | 2 | 86.5 % | 362/h | 7/h | 4/9, 6/10, 5/7, 4/7 (p = 0.27–0.77) |

So on synthetic words, a slower rate loses nothing: the small differences favour the slower rates, whose hold is no longer. The speaker test's 80-ms loss (42 of 50) does not appear here; only the same-audio
run can tell whether real speech differs, or whether the loss came from the separate takes.

**Rules, at 40 ms:**

| Rule | Keyword recall | On non-keyword speech | On noise | Needs |
|---|---:|---:|---:|---|
| 3 in a row (firmware) | 87.0 % | 439/h | 8/h | class |
| Sum of the last 8 logits, 2 in a row | 86.8 % | 212/h | 5/h | 12 logits per decision |
| Sum of the last 16 logits, 2 in a row | 85.2 % | 98/h | 1/h | 12 logits per decision |
| Sum of 16, mean margin ≥ 5, 2 in a row | 81.6 % | 60/h | 0.5/h | 12 logits per decision |
| Margin ≥ 20, 3 in a row | 68.9 % | 23/h | 0/h | 12 logits per keyword decision |
| Window maximum ≥ 2000, 3 in a row | 86.7 % | 422/h | 1/h | `SEQ_WMAX` per keyword decision |
| Sum of 8, window maximum ≥ 2000, 2 in a row | 86.5 % | 199/h | 0/h | both |

**What this shows.**
- **Smoothing the logits dominates counting decisions:** it halves the alarms on non-keyword speech for 0.2 points of recall.
- **A plain margin threshold is expensive:** it removes alarms, but costs 18 points of recall at margin ≥ 20.
- **With smoothing over the same 320 ms, the rate costs little:** recall 87.4, 86.8 and 86.1 % at 20, 40 and 80 ms (sums of 16, 8 and 4), at 216, 212 and 176 alarms per hour.
- **A level gate targets faint noise only:** it leaves recall and speech alarms nearly unchanged. Its threshold is in microphone units, so GSC cannot set it for this board; a spoken run must (§4).

All of these are firmware rules: no RTL changes. **None is chosen yet.** The cost of smoothing is a few hundred more CPU-active cycles per decision (an estimate, not measured).

Full tables: `model/rule_sweep_report.txt`, `model/rule_sweep_results.json`.

## 6. False alarms in the room

**This corrects an earlier impression twice.** Short checks with nobody speaking gave no reports, and `LIVE_AUDIO.md` §6 found `_unknown_` at every decision of a quiet minute. The first 1.46 h of a long run then did report keywords that nobody prompted, 23.6 per hour at 40 ms. The complete run shows that **all of them came in the evening, while the student was in the room, and none in the 7 hours of the night.**

The board ran the build without audio, in the room in ordinary use, for **8.38 h** in two parts: 22:16–22:40 and 22:41–06:41 (`model/kwlog_checks/far_ambient.log`, which pins both logs by hash). Both parts are clean: 0 bytes skipped, 0 hops lost.
Nobody was prompted to speak or to stay silent, so a keyword actually said nearby would count here as an alarm, and the log cannot tell what was happening in the room.

**Hour by hour**, at 40 ms with the firmware's rule (3 in a row):

| Period | Hours | Reports | Per hour (95 %) | Window maximum, 99.9 % and largest |
|---|---:|---:|---:|---|
| Part 1, 22:16–22:40 | 0.38 | 25.0 | 65 (42–96) | 1,372 |
| Part 2, hour 1, 22:41–23:41 | 1.00 | 9.5 | 9.5 | 1,450 and 3,247 |
| **Evening, together** | **1.38** | **34.5** | **25 (17–34)** | |
| **Part 2, hours 2–8, 23:41–06:41** | **7.00** | **0** | **0 (0–0.53)** | 225–241 and at most 276, each hour |

The sum of 8 and the level gate of 2,000, the other two rules the log breaks down by hour, also report nothing after the first hour of part 2.

**Over all 8.38 h**, an average over an evening and a quiet night, so it describes this night and not a room in use:

| Rule | 20 ms | 40 ms | 80 ms |
|---|---:|---:|---:|
| The firmware's (hold of at least 80 ms), reports per hour | 3.6 | **4.1** (95 %: 2.8–5.7) | 2.6 |

| At 40 ms | Reports | Per hour | 95 % interval | Keyword recall on GSC (§5) |
|---|---:|---:|---:|---:|
| 3 in a row (firmware) | 34.5 | 4.1 | 2.8–5.7 | 87.0 % |
| Sum of 8, 2 in a row | 26.0 | 3.1 | 2.0–4.5 | 86.8 % |
| Sum of 16, 2 in a row | 17.5 | 2.1 | 1.3–3.4 | 85.2 % |
| Margin ≥ 10, 3 in a row | 5.0 | 0.6 | 0.2–1.4 | 80.2 % |
| Margin ≥ 20, 3 in a row | 0.5 | 0.1 | 0.0–0.4 | 68.9 % |
| Window maximum ≥ 1000, 3 in a row | 4.0 | 0.5 | 0.1–1.2 | 87.0 % |
| Window maximum ≥ 2000, 3 in a row | 1.0 | 0.1 | 0.0–0.7 | 86.7 % |
| Sum of 8, window maximum ≥ 2000, 2 in a row | 1.0 | 0.1 | 0.0–0.7 | 86.5 % |

**Why it happens.** The reports come at faint levels. At the 33 reports of the firmware's rule (40 ms, phase 0), the window maximum was 144 to 2,514, median 279, of a full scale of 32,767, and the logit margin at the reporting decision 0 to 20, median 4.
The reference preprocessing normalises each 1-s window by its own maximum, so faint sound reaches the network at full scale. This is a property of keeping the reference pipeline exact, not of the hardware: the 2,850 decisions of §3 equal the golden model.

**Level alone does not explain them, though.** The night was as faint and gave none, and 13 of the 33 evening reports came at a window maximum below the night's 99.9th percentile (241). So the evening's sound differed in kind, not only in level.
**The timing matches a person in the room.** The student was in the room until about 23:00 (their account, given afterwards), and the last report came at about 23:04, minute 22.2 of part 2. Which of their sounds caused the reports (typing, movement, speech) the log, without audio, cannot tell.

**What to do.** In a quiet, empty room the firmware's rule is already quiet: none in 7 hours. What remains open is a room in use, here 25 reports per hour over 1.38 h of evening. A level gate looks like the cheapest remedy there: it left 1 of the evening's 34.5. `SEQ_WMAX` already holds the window maximum, so firmware needs one register read per candidate keyword, with no RTL change.
Whether 2,000 is right on this board **depends on how loud speech is at this microphone, which only a spoken run shows** (§4). A gate that is too high would miss quiet speech. A useful next false-alarm run would be one with people in the room, with notes of what they did. Rates differ
little here, since a slower rate also has fewer chances to report and a shorter hold in decisions.

The recall column comes from GSC streams, not this room, so the two columns are not one experiment.

## 7. Reproduce

```
cd hopcim/model
~/venvs/planenest/bin/python kwlog_analyse.py --self-test
~/venvs/planenest/bin/python rule_sweep.py run 4      # about 30 core-minutes
~/venvs/planenest/bin/python rule_sweep.py report
~/venvs/planenest/bin/python kwlog_analyse.py --far ~/APlus_Project/hopcim_recordings/far_ambient_*
```
Firmware: `planenest/hw/x-heep/sw/applications/hopcim_kwlog/main.c`. Build with `make app PROJECT=hopcim_kwlog TARGET=nexys-a7-100t
COMPILER_FLAGS="-DREFERENCE_CLOCK_Hz=15360000 [-DKW_TEST=1] [-DKWLOG_AUDIO=0 -DKW_TEST=0]"` and copy `sw/build/main.elf`.
