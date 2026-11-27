# HopCIM-AP: gates (frozen before any RTL and before any held-out data)

Frozen: 2026-11-27, before any RTL of HopCIM-AP existed and before any held-out data was generated or recorded. The
freeze commit holds this document, its `.sha256`, `model/hopcim_ap.py`, `model/ap_prediction.json`,
`model/ap_pilot.json` and their logs, as for the DWU and NET2 gates. Baseline: commit `1405549` (S+DWU, "D4", the
default design). The student chose the direction on 2026-11-27
(`research_discovery/20261127_approx_kws/RESEARCH_DISCOVERY.md`, §15).

**Not blind.** The gate rule and its constants were chosen in pilots on data already used before:
- the WHOLEWORD test streams (seeds 501–507);
- the two room runs `hopcim_recordings/far_ambient_20261107{,b}`;
- the two spoken runs `spk01_wholeword_run{1,2}`.

The pilot outputs are `research_discovery/20261127_approx_kws/analysis/ap_pilot*.out` and `model/ap_pilot.json`. The
pilots tried 47 rules: fixed thresholds, a minimum tracker and quantile trackers. Every threshold below was set after
the pilots, so the held-out data must be new.

A review of this draft (2026-11-27, before any held-out data or RTL) changed four things, all on pilot data only:
1. **The U rule after a skip.** The first pilot scoring summed a skipped decision's logits into U's two-pass sum,
   which the hardware cannot do (a skipped decision has no logits). Scored the way the hardware can, with "a skipped
   decision adds nothing", U + gate raised the speech reports from 121.0 to 140.3 per hour. The rule fixed below (a U
   report needs two consecutive launched passes) was then chosen on the same pilot data.
2. **The always-on state.** The logits, `cim_seq`'s CPU-readable registers and the front-end's lraw output were in
   switched domains. The firmware and the next run read them, so they are now always on, and they are priced that way.
3. **The pricing clock.** The first prediction used 81,920-cycle hops (4.096 MHz) but the baseline energy model's
   15.36-MHz clock. Both now come from the energy-model variant: baseline 15.36 MHz, plus a 4.096-MHz variant.
4. **The held-out set** gained a spoken run and a room run with background sound. AP5 uses only the quiet room runs.

A second review (same day, still before any held-out data or RTL) changed five more:
5. **One report of slack in AP7 and AP8.** A skipped decision restarts the hold rule, so the gate can add a report as
   well as remove one. On the pilot speech streams U + gate added 4 reports and removed 7 (over its 16 phases), so a
   test of "not one report more" would be decided by chance. AP7 and AP8 now allow 1 report more over each set, as
   AP6 already allowed 1 alarm more.
6. **Every quiet room weighs the same in AP5.** R1 is the room of the pilot runs, and a longer R1 would otherwise
   dominate the duty. On the pilot's own two room runs (0.38 h from 22:16, then 8 h from 22:41) this weighting gives
   a duty of 0.115, against 0.069 weighted by time. The held-out quiet rooms are ≥ 4 h each, so each is a full sample.
7. **The cycle rule** missed hop 0 (no front-end, one wake) and the CPU-started runs. It is now one rule, also in
   `hopcim_ap.WAKES`. And `PD_SIM` now randomises every storage element without a reset in an off domain: the lraw
   output buffer has no reset, so with resets alone mutant (g) could not be caught.
8. **The gate's level limit** is stated below. The test streams are high-SNR, so only S3 can show it.
9. **Decision count.** A skipped decision does not count in `ndec_o`. AP9 checks `ndec_o` + `AP_NSKIP_D` against
   the hop.

Energy is a model estimate [E], never FPGA power.

## Question

D4 pays for its whole front-end and sequencer state at every hop, and for a network run at every scheduled hop,
whatever the room does. Two properties of HopCIM make this avoidable:
- **the static schedule** gives the exact lifetime of every structure, so each can be powered only while it is live;
- **the window path already computes the 1-s window maximum**, which is a free acoustic-activity signal.

Can the two together give:
1. an exact processing path, with every launched window bit-exact;
2. no measurable loss of keywords;
3. average power that tracks acoustic activity, so that the lowest-latency cadence costs less than today's mode U?

## The design (fixed now)

### Level gate

**Tracker.** In `cim_audio`, in the cycle that ends a hop (`hop_end`), with `bm` the same signed hop maximum that is
latched into `bmax_o`:

```
F_in = max(bm, 1) at hop 0 of a stream (AUD_CTRL enable 0 -> 1), else F
st   = max(1, F_in >> 6)
F    = F_in + st          if bm > F_in (signed)
       max(1, F_in - 3 st) otherwise                                        (F: 16 bits, unsigned)
g    = (20 F) >> 3                                                          (2.5 F; 18 bits, no overflow)
```

F settles at the lower quartile of the hop maxima, because it steps up 1 unit and down 3 units: P(bm > F) = 3/4. The
constants (20, 6, 3) and W below are RTL parameters, not registers.

**Check.** In `cim_seq`, on a run scheduled for the network, in the first cycle of `W_LOG`, when `w_max` holds the
maximum of all 50 ring slots including this hop's:
- if `w_max < g` (signed `w_max`, unsigned `g`), the decision is skipped. The run ends at the window path's end as a
  window-path-only run. `cim_audio` does not latch a class, does not set `pend` and counts a skip.
- Otherwise the network runs as in D4.

The gate applies only to runs started by the hop controller; a CPU-started run is never skipped. The window path
itself runs at every hop, unchanged, so every launched window's features equal the per-window reference.

**Limit (by design).** A decision launches only if the loudest hop of the last second reaches 2.5 × the lower quartile
of the recent hop maxima, about 8 dB in peak terms. A keyword quieter than that over a steady background is skipped,
even where the network would have found it. If the background's lower quartile is above 13,107 (40 % of full scale),
`g` exceeds every sample and nothing launches. The test streams cannot show this limit. Their loudest 1 % of hops reach about
18,000, against median hop maxima of about 85, 280 and 830 at RMS 30, 100 and 300. Even at RMS 300, 95 % of F20's
reports cleared the threshold by at least 2.2 × on every decision of their hold. S3, the spoken run with background sound, is
the test of this limit.

**Registers** (endpoint):
- `AP_CTRL`: [0] gate enable, [1] domain-control enable; both 0 at reset, which gives exactly D4;
- `AP_NSKIP`: skipped decisions since the stream start;
- `AP_NSKIP_D`: `AP_NSKIP` latched when a decision is launched;
- `AP_F`: the tracker, read-only.

`AUD_STATUS`'s decision count (`ndec_o`) keeps its meaning: decisions that raised an interrupt, now the launched
ones. At every launched decision, `ndec_o` + `AP_NSKIP_D` = the number of scheduled decisions so far; at k = 1 and
hop h, h − 48.

**Firmware.** At each decision interrupt it reads the class, the logits and `AP_NSKIP_D`. The difference from the
previous decision's `AP_NSKIP_D` is the number of skipped decisions in between. Each skipped decision counts as a
non-keyword decision:
- F20 and F40: the hold count restarts;
- U: a decision whose previous scheduled decision was skipped reports nothing. Its logits are kept for the next
  decision's two-pass sum.

The model's scoring (`hopcim_ap.events_for`) does the same.

### Clock enables

`cim_audio`'s flip-flops outside the PDM decimator (140 of its 850) change only on a sample, a hop-controller handshake
or a CPU write, together far below 1 % of cycles. They get clock enables. This is functionally invisible. The energy
model counts them as clocked on 10 % of cycles, which is conservative. The decimator is left as in D4, in every design
compared.

### Power domains (coarse)

The wake lead is W = 64 cycles. A unit starts only after every domain it uses has been on for W cycles. With
`AP_CTRL[1]` = 1 the domain controller acts on every start, whether the hop controller or the CPU issues it.

| Domain | Contents | Request | Release |
|---|---|---|---|
| PD_FE | `cim_fe` datapath, its frame buffers (bbuf, ribuf, magbuf) | front-end start request | front-end done |
| PD_CIM | CIM macro periphery (adder trees, accumulators), engine X banks | front-end or sequencer start request, when off | neither the front-end nor the sequencer busy or pending |
| PD_SEQ | `cim_seq` datapath: FSMs, gather, issue, requant lanes, pool, the window path's per-run registers | sequencer start request | sequencer done |
| PD_ACT | activation SRAM | end of the window path of a launched run; the start of a network-only run | sequencer done |
| PD_DWU | depthwise unit and its weight register | as PD_ACT | sequencer done |

On the live path, a hop therefore runs (hop 0 has no front-end: its hop end requests PD_SEQ and PD_CIM):
1. the hop end requests PD_FE and PD_CIM, and the front-end starts W cycles later;
2. front-end done requests PD_SEQ, and the sequencer starts W cycles later;
3. on a launched decision, the window path's end requests PD_ACT and PD_DWU, and the network starts W cycles later.

**Always on:**
- `cim_audio`: PDM decimator, ring control, tracker, hop controller, class, `pend` and the counters;
- the endpoint's configuration and status registers, the AP registers and the domain controller;
- in `cim_seq`: the window path's persistent state (block-maximum ring, valid bits, pointer, `w_max`, `lmx_q`,
  `lmx_ok`, recompute flag) and everything the CPU reads after a run (the logits, the class, `SEQ_STATUS`);
- in `cim_fe`: the lraw output buffer (40 × 32 bits), which the next sequencer run copies in `W_COPY`;
- memories: the audio ring, the lraw ring, FEAT, the requant parameters, the DW weight store, the CIM weight cells
  (retained).

**CPU access.** With `AP_CTRL[1]` = 1, the CPU must not access a switched domain: no ACT window reads, X writes,
CPU engine passes or weight writes. Firmware loads weights and parameters with `AP_CTRL[1]` = 0, then sets it.

**Simulation only (`+define+PD_SIM`):**
- while a domain is off, its registers with a reset are held in reset;
- while a domain is off, every storage element in it without a reset is overwritten with pseudo-random data each
  cycle: memories and register arrays alike (the ACT words, bbuf, ribuf, magbuf, the mel buffer, the pool sums, the
  engine's X banks, the depthwise unit's accumulators and weight register, the requant pipeline). Several of them have
  no reset, among them the lraw output buffer, so resets alone would let mutant (g) below pass;
- an independent bench monitor stops the run if a unit uses a domain less than W cycles after its request, or if
  the CPU accesses a switched domain while it is off.

### Build

- Endpoint parameter `SEQ_AP`, on top of `SEQ_DWU` = 4.
- FuseSoC flag `use_cim_ap`, as a new patch of `hopcim/xheep/patches`.
- Energy counters (`tb/energy_counters.sv`): each domain's on-cycles and wakes, launches, skips and clock-enabled
  cycles.
- Firmware `hopcim_kwlog_ap`: the no-audio kwlog with `AP_CTRL` set and `AP_F`, `AP_NSKIP_D` logged at each decision
  (AP9 only).

## Predictions (`model/hopcim_ap.py`, before any RTL)

**Gate decisions:** exactly `hopcim_ap.gate(bm)`, with bm from hop 0 of the stream. The self-test catches four broken
gate variants and a U rule that would use a skipped decision's logits.

**Cycles** (W = 64), with `AP_CTRL` = 3. A unit start waits W after its request when a domain it needs is off.
Each unit's own domain is released at its done, so every start finds one off. Every run therefore takes the D4 run of
the same stimulus plus W per start (`hopcim_ap.WAKES`):

| Run | Starts that wait | Extra cycles |
|---|---|---:|
| Live hop 0 (no front-end) | sequencer | W |
| Live hop ≥ 1 without a network (hops 1–48, not scheduled, or skipped) | front-end, sequencer | 2W |
| Live launched hop | front-end, sequencer, network | 3W |
| CPU: front-end run; window-path-only run; network-only run | that run | W |
| CPU: window path and network | sequencer, network | 2W |

A live hop's D4 reference is the D4 run with the network on (launched hops) or off (every other hop). The worst hop is
**80,020 + 192 = 80,212** ≤ 81,920 at 4.096 MHz. The network's go→done is unchanged: **67,789**.

**Domain on-cycles and wakes:** each domain's on-cycles are the sum of (release − request) over its on intervals,
with the request and release events from the table above. Its wakes are the number of requests that find it off.

**Power [E]** (µW, `model/ap_prediction.json`). The baseline variant runs at 15.36 MHz. U today = D4 at k = 16, all
powered = **27.96**. Duty is the fraction of scheduled decisions launched; duty 1.0 means domains and clock enables
without the gate.

| Design (launch duty →) | 1.0 | 0.5 | 0.2 | **0.069** (pilot rooms, by time) | 0.02 |
|---|---:|---:|---:|---:|---:|
| D4-AP20 (every hop, F20 rule) | 63.99 | 37.10 | 20.97 | **13.93** (0.50 × U today) | 11.30 |
| D4-AP40 (every 2nd hop) | 37.10 | 23.66 | 15.60 | 12.08 | 10.76 |
| D4-APU (every 16th hop, two passes) | 13.58 | 11.90 | 10.89 | **10.45** (0.37 ×) | 10.29 |

Ablation:

| | k = 1 | k = 2 | k = 16 |
|---|---:|---:|---:|
| D4 as built | 75.80 | 50.28 | 27.96 |
| D4 + clock enables only | 74.37 | 48.86 | 26.53 |
| + domains, no gate (duty 1.0) | 63.99 | 37.10 | 13.58 |
| + gate at the pilot duty | 13.93 | 12.08 | 10.45 |

At k = 16 the domains do most of the work. At k = 1 the gate does.

**Break-even launch duties:**

| | baseline (15.36 MHz) | 4.096 MHz |
|---|---:|---:|
| D4-AP20 below U today | 0.33 | 0.26 |
| D4-AP20 at half of U today | 0.070 | 0.042 |
| D4-APU at half of U today | every duty | 0.67 |

At the pilot duty, D4-AP20 is 0.17–0.62 × U today over the energy-model variants, or 0.40–0.62 × without the
tiles-per-macro variants. D4-APU is 0.14–0.45 ×, or 0.32–0.45 ×. Over residual leakage 0.02–0.2 and wake energy
10–500 µs of leakage, D4-AP20 is 13.5–16.3 µW. The size is assumed at 1,500 gate equivalents; at 50 pW per gate
equivalent, 1,000 more adds 0.05 µW.

**Pilot scores of the fixed rules** (`model/ap_pilot.json`):

| | Recall r30 / r100 / r300 | Launches per hop: words / speech / noise / room | Room reports/h |
|---|---|---|---:|
| F20 | 91.6 / 87.8 / 81.8 | 100 % everywhere | 3.58 |
| F20 + gate | **91.6 / 87.8 / 81.8** | 67.6 / 87.0 / 49.5 / **6.9 %** | **1.91** |
| F40 + gate | 92.5 / 89.2 / 83.1 (unchanged) | 33.8 / 43.5 / 24.7 / 3.5 % | 2.51 (4.12 ungated) |
| U + gate | 90.4 / 86.1 / 80.2 (ungated 90.4 / 86.1 / 80.4) | 4.2 / 5.4 / 3.1 / 0.43 % | 1.42 (2.21 ungated) |

Speech and noise reports per hour with the gate: F20 482 and 2.0 (unchanged); U 120.8 and 1.81 (ungated 121.0 and
1.81).

Spoken runs (`hopcim_ap.py score-spoken`):
- F20: 93/100 keywords correct with and without the gate, 3 alarms (other-word plus quiet) in both;
- U + gate: 93.0 (93.06 ungated);
- F20 with the gate reports a median 0.17 s earlier than today's U (93 paired prompts).

The held-out scoring commands reproduce every pilot number above from the pilot data.

## Held-out data (never used by the pilots)

The spoken and room runs use the same bitstream and firmware as the pilot runs: `xheep_nexys_bscan_cim128k.bit`
(R6b, `b7102d5d…`) with the no-audio kwlog firmware, `hopcim_kwlog_noaudio_r5_nexys.elf` (`34ee2e59…`, prompted) for
S1–S3 and `hopcim_kwlog_noaudio_nexys.elf` (`68364470…`, unprompted) for R1–R3. Both log every hop's maximum and
every decision's logits, and no audio. No no-audio kwlog firmware exists for D4, and none is built: the hop maximum
comes from `cim_audio`, which D4 does not change, and the logits equal the golden model in R6b, S and D4 alike, so
the logged data are what D4 would log. The gate is then applied by `hopcim_ap.gate` on the logged hop maxima. AP1
and AP9 show that it equals the RTL. So the student can record them right after the freeze, while the RTL is being
built.

| Set | Contents | Who |
|---|---|---|
| **Streams** | The WHOLEWORD generators (`rule_sweep.make`) with fresh seeds: words 901–904 (2.0 h), speech 905, 906 (1.0 h), noise 907 (1.0 h), each at RMS 30, 100, 300. Generator seeds (seed + RMS) 931–937, 1001–1007, 1201–1207; none collides with an earlier seed (`hopcim_ap.heldout_streams` asserts it). The streams are new mixtures from the same pools as the pilot streams (GSC test keywords and unknown words, GSC background noise), at high SNR (see the limit above), so this is the weakest held-out test | model |
| **Spoken** | 3 new prompted runs (65 prompts each, 50 of them keywords): S1 at about 0.5 m in a quiet room; S2 at about 2 m in a quiet room; S3 at about 1 m with background sound (TV, music or talk at conversational level) | student |
| **Room** | R1: the usual room (the room of the pilot runs), quiet (no media playing), ≥ 4 h. R2: a different room, quiet, ≥ 4 h; the only room the rule has never seen. R3: background sound (TV, music or talk), ≥ 1 h | student |

All spoken and room runs are recorded after the freeze. A run with an integrity problem (`kwlog_analyse.integrity`,
or decisions not at every hop from hop 49) is not scored; it is reported and recorded again. Missing hop maxima count
as −32768, as in the pilot, and their number is reported.

## Gates

| ID | Gate | Pass if |
|---|---|---|
| AP1 | Exact and safe | With `SEQ_AP` and `AP_CTRL` = 0, every existing bench (`sim/run_all.sh`) passes unchanged. With the domains enabled and `PD_SIM`, bit-exact: (1) the 4,890-clip run (CPU-started network runs); (2) `tb_cim_seq`'s full-run parts (20 clips, synthetic sets, held writes); its stop-after-layer and CPU-pass parts read ACT or use the engine from the CPU, so they run with `AP_CTRL` = 0; (3) `tb_cim_stream` (CPU-started front-end and sequencer); (4) `tb_cim_mic` at k = 1, 2, 4. With the gate also enabled, on a new live-path bench `tb_cim_ap` at k = 1 and k = 16: a noise floor at RMS 30, a GSC keyword, the floor again, a loud onset in ring slot 49, ≥ 150 hops. There the launch/skip sequence and `AP_F` equal `hopcim_ap`, every launched decision's logits equal the golden model, no skipped decision raises an interrupt and `AP_NSKIP_D` is right. Seven broken variants must each be caught: (a) PD_ACT released one cycle before the network's last ACT access; (b) the front-end starts at its request, without the W wait; (c) the gate compares the previous hop's window maximum (checked at `W_PUSH`); (d) the tracker's down step is 1 × st; (e) a skipped decision sets `pend`; (f) the logits in PD_SEQ; (g) the lraw output buffer in PD_FE |
| AP2 | Prediction | On every AP1 bench with the domains enabled, the RTL counters equal the predictions exactly: launches, skips, each domain's on-cycles and wakes, and the cycles of every hop and every CPU-started run (D4's plus `hopcim_ap.WAKES`). A difference is a defect in the RTL or the model, found and reported before scoring |
| AP3 | Real time | Hop monitor, `tb_cim_mic` at the 4.096-MHz divider, k = 1, domains on: (i) gate off (every hop launched, the worst case) and (ii) gate on. Worst hop ≤ 81,920 cycles, no overrun |
| AP4 | Size | Added logic ≤ 2,000 gate equivalents: Vivado out of context, endpoint with `SEQ_AP` against D4 on the same source, 6 per LUT, 6 per FF. Isolation cells, power switches and retention cells are ASIC-only and not counted; the residual leakage and the wake energy stand for them. Power is re-priced with the measured size |
| AP5 | Power [E] | `hopcim_ap.power`, with the cycle constants confirmed by AP2 and the measured size, at the launch duties of the quiet held-out room runs R1 and R2, each room weighing the same (the mean of the two runs' duties; `score-room`). Both conditions in the baseline variant and in the 4.096-MHz variant: (a) D4-AP20 < U today; (b) D4-APU ≤ 0.5 × U today. Reported: every variant, R3's duty and power, and the ablation. (a) fails only above duty 0.33 (0.26 at 4.096 MHz). (b) holds at every duty in the baseline (up to 0.67 at 4.096 MHz), so (b) tests the domains and the measured cycles, not the gate |
| AP6 | Keywords, spoken (held out) | Pooled over S1–S3: F20 with the gate gets ≥ F20 without it − 1 keyword per 100 keyword prompts correct, and its other-word plus quiet alarms are ≤ F20's + 1 (`score-spoken`). The same is reported for F40 and U, and per run |
| AP7 | Room (held out) | Pooled over R1–R3: the reports with the gate are at most 1 more than without it (mean over phases), for F20, F40 and U (`score-room`). Reports per hour and the launch duty are reported per run, and the duty per hour |
| AP8 | Streams (held out) | For F20, F40 and U: recall with the gate ≥ recall without it − 1.0 point at each RMS, and over each of the speech set and the noise set (1 h each), the reports with the gate are at most 1 more than without it (mean over phases; `score-streams`). Every stream's 10-decision check against the per-window reference matches. The launch duty is reported |
| AP9 | Board | The `use_cim_ap` bitstream meets timing at the system clock. With `AP_CTRL` = 0: 60 sequencer runs exact, live path k = 1, 2, 4 with no overrun, real-microphone logits equal to the golden model. With `AP_CTRL` = 3, `hopcim_kwlog_ap` at k = 1, a run of ≥ 10 minutes with no lost hop (otherwise it is recorded again): no overrun, every decision's `AP_F` equals the tracker replayed from the logged hop maxima, `ndec_o` + `AP_NSKIP_D` = h − 48 at every logged decision of hop h, the launched decisions and the skips (from `AP_NSKIP_D`) equal `hopcim_ap.gate()`, and the logits equal the golden model. The student runs the board commands |
| AP10 | Latency (reported, no threshold) | On S1–S3: the median paired time from prompt to the first correct report, F20 with the gate against today's U (`score-spoken`; the pilot gave −0.17 s) |

**Order:**
1. Freeze: one commit with this document, its `.sha256`, `model/hopcim_ap.py`, `model/ap_prediction.json`,
   `model/ap_pilot.json`, `model/hopcim_ap_predict.log` and `model/hopcim_ap_pilot.log`.
2. The student records S1–S3 and R1–R3 at any time after the freeze. AP6, AP7 and AP10 are scored once all six
   exist, in one go, never on part of them.
3. AP8: `heldout-streams`, then `score-streams`.
4. RTL step G: the gate, the tracker, the registers and the clock enables. Then the gate parts of AP1.
5. RTL step P: the domains, `PD_SIM`, the monitor and the counters. Then AP1–AP4.
6. AP5, with the duties of R1 and R2.
7. Bitstream and AP9.

## Decision

- **All pass:** D4-AP becomes a built HopCIM option. The student decides the default cadence.
- **AP1, AP2, AP3 or AP9 fails:** it is a defect; it is fixed and rerun, with no threshold moved.
- **AP4 or AP5 fails:** reported; the student chooses.
- **AP6, AP7 or AP8 fails:** the gate rule does not generalise beyond the pilot data. This is reported. Any change to
  the rule after the freeze is post hoc and is reported as such.
