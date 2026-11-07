# Exact activation reuse: what it would buy (software evaluation)

Status (2026-11-07): evaluated in software only. **It is not in the RTL.** Both conditions for building
it (§8) now hold at the level of a model. The idea is described in
`ExactReuse_CIM_Research_Proposal.md`. Everything here is reproducible with `model/exact_reuse.py`
(§10). Numbers are measured on the integer golden model unless marked as estimates. Energy figures
are model-based estimates (`ENERGY_ESTIMATE.md`).

## 1. Result

Consecutive 1-s windows overlap in 48 of 49 frames, so most of the network's work could in
principle be reused. Reuse can be done exactly, even on the unchanged SAME-padded, strided model:
over 174,216 decisions, every logit and every layer equalled independent full inference.

Its saving at 50 decisions per second is about the same as simply running the network every
40 ms, which needs no memory:

| Option | Extra activation memory | CIM passes per hop, mean | Fewer than now |
|---|---:|---:|---:|
| Now: network every 20 ms | 0 | 7,501 | 0 |
| **Network every 40 ms** (`LIVE_AUDIO.md` §9) | 0 | 3,750 | 50 % |
| Reuse every 20 ms, simple | +72 KB | 3,860–4,091 | 45.5–48.5 % |
| Reuse every 20 ms, with feature comparison and value-aware propagation | +72 KB, plus compare logic | 3,652–3,821 | 49.1–51.3 % |
| **Network every 80 ms** | 0 | 1,875 | 75 % |
| Reuse every 80 ms, simple | +32 KB | 1,087–1,170 | 84.4–85.5 % |

Ranges are over the five stream groups of §2. Extra memory is an estimate (§6).

On synthetic streams, keyword detection did not measurably change at 40 or 80 ms (§5). With a real
speaker, 40 ms showed no loss and 80 ms lost six words of 50 (`LIVE_AUDIO.md` §9). Reuse is not needed
to reach 40 ms. On top of it, reuse removes about half of the remaining passes and shows an estimated net
saving of 13 µW with leakage included (§7).

The worst case does not improve: any change of the window maximum can alter every feature, and a
stream start has no cache, so the deadline must still allow a full recompute. The lowest real-time
clock stays at 5.12 MHz.

## 2. Method

**Reuse.** Window N + lag holds the frames of window N shifted by lag. For an even lag, conv1's
stride-2 grid is aligned, so output row r of the new window can take row r + lag / 2 of the cached
window where both see the same inputs. A position must be recomputed if its receptive field reaches
padding in one window and real data in the other, holds new frames, or holds changed features.
A decision every hop needs two caches (one per stride phase); a decision every 2 or 4 hops needs one.

**Modes**, each including the one before:

| Mode | Rule |
|---|---|
| simple | fixed geometry; a full recompute if the window maximum changed since the cached window |
| feature comparison | fixed geometry, plus the actual difference of the aligned int8 features |
| value-aware | also: a recomputed checkpoint position equal to its cached value does not propagate |
| oracle (a bound, not a design) | only outputs that really change are computed, depthwise by channel group: needs 9 cached tensors and perfect knowledge |

Caches: conv1 and pointwise 1–4, 8,000 B each, the proposal's five checkpoints.

**The model is functional.** Every reused value comes from a cache, and every recomputed value from
the layer functions applied to the reconstructed input, so a wrong dirty mask corrupts later layers
and the logits. It never reads the full run. It is compared with an independent `dscnn_int.forward`
at every decision, in every layer. Cycle figures are estimates from the measured per-layer rates
(`SEQUENCER.md` §5), including about 10 cycles per reused position to re-read it into the pool sum.

**Streams** (16, deterministic, 3,500 s, 174,216 decisions):

| Group | Content | Window maximum changed at |
|---|---|---:|
| Documented stream | the 20 clips of `WINDOW_PATH.md`, back to back | 11.6 % of decisions |
| Dense speech (3 × 200 s) | GSC test clips of all classes, random order | 11.2 % |
| Always-on, noise RMS 30, 100, 500 (4 × 240 s each) | a noise bed of GSC `_silence_` clips scaled to one RMS, with a word every 3 s (±20 %), 80 % keywords, 20 % `_unknown_` | 14.6, 13.2, 10.6 % |

The board's quiet room measured RMS 44–70 (`LIVE_AUDIO.md` §6, §7).

## 3. Exactness

| Check | Result |
|---|---|
| Logits, three modes, lags 2 and 4, all streams | 0 mismatches |
| All 9 layers at every decision | 0 mismatches |
| Geometry against the hand derivation (rows 6, 8, 10, 12, 14; 3,201 passes; full 7,501) | equal |

Exactness follows from determinism: the same int8 inputs give the same int8 outputs, so the task is
bookkeeping. The test can fail, as each repair removed on purpose shows (simple mode, the 951
decisions of the documented stream):

| Broken on purpose | Logits wrong | Class wrong |
|---|---:|---:|
| No padding repair at the start of the window | 798 | 35 |
| One cache for both stride phases | 799 | 312 |
| Window-maximum change ignored | 789 | 52 |

With the feature comparison, a phase mix-up stays correct and only loses the saving: nearly every
aligned cell then differs, so nearly everything is recomputed.

## 4. How many passes

Mean per decision at one decision per hop, the range over the five groups:

| Mode | Passes | Fewer | Median | p99 | Full recomputes | Sequencer cycles, estimate (now 85,352) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed geometry, best case | 3,201 | 57.3 % | | | | |
| Simple | 3,860–4,091 | 45.5–48.5 % | 3,201 | 7,501 | 15.3–20.7 % | 44.2–46.8k |
| Feature comparison | 3,658–3,822 | 49.1–51.2 % | 3,201 | 6,683–6,899 | at stream start | 41.7–43.5k |
| Value-aware | 3,652–3,821 | 49.1–51.3 % | 3,201 | 6,683–6,893 | at stream start | 41.7–43.5k |
| Oracle bound | 3,313–3,458 | 53.9–55.8 % | 2,931–2,950 | | | |

**The proposal's advanced mechanisms add little.**
- The feature comparison adds 2.7–3.6 points.
- Value-aware propagation adds at most 0.2 points. A position only stops propagating if all 64
  channels are equal. Comparing at every activation is not possible with five checkpoints: depthwise
  outputs are not cached, and a recomputed pointwise position needs all 10 of its depthwise groups.
- Even the oracle stays below 56 %: the boundary rows dominate, not dynamic sparsity.

A second requantisation lane (`SEQUENCER.md` §5) would take 85,352 to about 67.9k cycles but skips no
pass, so it is not the baseline to compare reuse's saved work with. With both, reuse would take
about 35.6–37.6k cycles (simple).

## 5. Keyword detection against the decision rate

Hardware option `AUD_CTRL[19:16]`, all kept decisions bit-exact (`LIVE_AUDIO.md` §9). The always-on
streams go through the board's keyword rule, with the hold set to span at least the same 80 ms
(1 + ⌈4/k⌉ decisions: at 20 ms, 5 decisions in a row span 4 intervals). A word can be reported from
its start until 1 s after its end. Every alignment of the stride is averaged.

| Network every | Hold | Recall, noise RMS 30 / 100 / 500 | Alarm on `_unknown_` words | Paired against every hop: hits only every hop / only this rate, p |
|---|---:|---:|---:|---:|
| 20 ms | 5 | 91.2 / 88.5 / 73.4 % | 25.0 / 18.8 / 15.0 % | |
| 40 ms | 3 | 91.4 / 89.1 / 74.8 % | 26.8 / 18.0 / 17.5 % | 2 / 8, p = 0.109 |
| 60 ms | 3 | 90.7 / 87.0 / 71.1 % | 17.3 / 9.4 / 11.1 % | 10 / 2, p = 0.039 |
| **80 ms** | 2 | 91.1 / 88.5 / 73.6 % | 24.1 / 15.2 / 15.0 % | 10 / 6, p = 0.454 |
| 100 ms | 2 | 90.8 / 86.9 / 71.6 % | 19.3 / 10.3 / 12.0 % | 14 / 1, p = 0.001 |
| 160 ms | 2 | 90.2 / 86.1 / 69.7 % | 11.8 / 6.4 / 9.6 % | 22 / 3, p < 0.001 |
| 320 ms | 2 | 88.1 / 84.0 / 65.8 % | 2.6 / 3.9 / 6.9 % | |

The paired test uses 770 keywords, an exact McNemar test at the hardware's alignment (decisions at
hops 49 + nk).
- **At 40 and 80 ms the hold spans exactly 80 ms, as now, and no change in recall is measurable.**
- At 60 ms and from 100 ms up, the same rule spans longer than 80 ms, so it trades recall for fewer
  alarms: this is a stricter rule, not only a lower rate.
- Noise alarms were 0–1 per 960 s at every rate up to 80 ms, too few to compare.

These streams are synthetic: GSC words over GSC noise, every 3 s. The board test with a speaker
(`LIVE_AUDIO.md` §8.1, §9) is the check that matters.

## 6. Memory (estimates, not synthesised)

Now: ACT holds two tensors, 16 KB in 7 RAMB36, and nothing in it has to survive between hops.

| Cached tensors | Total with one work tensor | Extra | ACT block RAM (compact addressing) | Full SoC of 135 tiles (now 82) |
|---|---:|---:|---:|---:|
| 10 (two phases, every hop) | 88 KB | +72 KB | about 28 RAMB36 | about 103 |
| 5 (one phase, every 2 or 4 hops) | 48 KB | +32 KB | about 14 RAMB36 | about 89 |

So it fits this FPGA. For scale, the 69 CIM tiles in use hold about 71 KB.

Fewer tensors keep much of the saving. Best case (fixed geometry, lag 2):

| Tensors per phase | KB per phase | Best choice | Passes | Fewer |
|---:|---:|---|---:|---:|
| 1 | 8 | pw3 | 5,051 | 32.7 % |
| 2 | 16 | pw2, pw4 | 3,921 | 47.7 % |
| 3 | 24 | pw1, pw2, pw4 | 3,561 | 52.5 % |
| 5 | 40 | all | 3,201 | 57.3 % |

The analysis behind this table is about 40 lines of Python and runs in under a second. On one model,
it is not a substantial "compiler" contribution; a second model would need the sequencer's fixed layer
order to change (`SEQUENCER.md` §10).

**Hardware not covered by the proposal.** The pool sums accumulate during the last layer's write-back
(`cim_seq.sv`, first position overwrites), so reused positions would need re-reading or an
incremental update. The gather loop, the ACT addressing and the barrier counts also change. The
arithmetic does not.

## 7. Leakage: what an energy decision would need

**Superseded by `ENERGY_ESTIMATE.md` §7**, which uses exact activity and published tools. There, simple
reuse saves 13.0–14.1 µW net at 40 ms (lowest variant +9.7 µW) and stops paying only at 49–137 pW per
bit, against about 11 from CACTI and 7.4 measured in 28 nm silicon. The thresholds below stay for the
record. They are far too low because they count only MACs and an assumed 10 fJ per ACT byte: with tool
numbers an ACT byte costs about 620 fJ on a read, and a skipped position also skips its requantisation,
its parameter reads and its control cycles.

The cache must keep its contents between hops; today's ACT need not. The proposal's energy terms do
not include this retention cost. The threshold below is the leakage per cached bit at which simple
reuse stops paying, against the same decision rate without a cache:

`leakage per bit < (saved MAC/s × energy per MAC + saved ACT bytes/s × energy per byte) / cached bits`

The saved work is measured (always-on and dense streams). **The energies are assumptions, not data**;
the table is a threshold to compare with a real process, not an energy result.

| Decision every | Saved MAC/s | Saved ACT bytes/s | Cached kbit | Threshold in pW per bit, at 20 / 50 / 100 fJ per MAC and 10 fJ per ACT byte |
|---|---:|---:|---:|---:|
| 20 ms | 63.6 M | 8.92 M | 640 | 2.13 / 5.11 / 10.07 |
| 40 ms | 31.8 M | 4.46 M | 320 | 2.13 / 5.11 / 10.07 |
| 80 ms | 13.4 M | 1.87 M | 320 | 0.89 / 2.15 / 4.23 |

The thresholds are optimistic: they ignore the added control logic and its leakage. Lower decision
rates shrink the saving, but not the leakage.

## 8. Decision

Not built. The rule was to build it only if both hold:
1. the advisor wants energy estimates (`HOPCIM_PROPOSAL.md` §15.3): **yes**, preferred if a published tool
   can do it, as an additional result (2026-11-07);
2. a model that includes leakage, with real process numbers, shows a net saving: **yes in the model**
   (`ENERGY_ESTIMATE.md` §7), with the logic estimate still coarse and ZigZag-IMC's equations not yet run
   in the tool itself.

If built: only the simple mode with one cache, at a decision every 2 hops, probably with fewer tensors
(§6). Rough effort: 4–6 weeks including verification. Feature comparison and value-aware propagation do
not earn their cost.

**Priority: low (2026-11-07).** Meeting the conditions does not make it the next step. It is one candidate
for an optional extension, chosen only after the novelty audit, the same-audio test, the timing sensitivity
and the energy validation (`HOPCIM_PROPOSAL.md` §15.2). Plain 80 ms is estimated at 58 µW, with no memory
and no RTL, against 62 µW for 40 ms with reuse, so reuse earns its 4–6 weeks only if 80 ms loses on real
speech and reuse still wins after the stronger modelling.

## 9. Related work

To be read in full before any claim; checked so far from abstracts or the parts noted.
- StreamiNNC (Kechris et al., arXiv 2408.03223): shift-ability needs retraining with signal padding.
  Its streaming deviates by 3.3–5.8 % NRMSE on pretrained zero-padded networks and 2.03 % on the
  retrained one. Its venue is not yet confirmed.
- Rybakov et al., Interspeech 2020: its library lists DSCNN with stride as not streamable.
- OverFeat (Sermanet et al., 2014): evaluating several offsets to recover what a stride loses is old,
  so "phase-aware" is not itself new.
- LiCo-Net (arXiv 2211.04635): streaming equal to non-streaming, with a changed architecture.
- FeFET streaming (TETC 2024, in the project's novelty assessment): ring-buffered streaming on an analog CIM.

I found no work doing exact reuse of an unchanged, padded and strided int8 CNN on a digital CIM, but
the search was not systematic. The gap, if real, is narrow.

## 10. Reproduce

```
cd hopcim/model
~/venvs/planenest/bin/python exact_reuse.py static              # fixed geometry (about 1 s)
~/venvs/planenest/bin/python exact_reuse.py self-test           # checks with broken variants (a few minutes), reuse_selftest.log
~/venvs/planenest/bin/python exact_reuse.py suite <outdir> 7    # the 16 streams (about 15 min on 7 cores)
~/venvs/planenest/bin/python exact_reuse.py report <outdir>     # the tables, results.json
```

Frozen outputs: `model/reuse_results.json`, `model/reuse_run.log`, `model/reuse_selftest.log`.
