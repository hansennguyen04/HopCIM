# Verification: assertions, coverage, mutation, extreme input, gate level

Status (2026-11-07): done. **No defect was found in the RTL.** Two defects were found in the new checks themselves, and both are described below. This adds confidence in the RTL; it
is simulation, not a proof.

## 1. Summary

| Check | Result |
|---|---|
| Regression, `sim/run_all.sh` | **28 of 28 benches pass** (the 24 of the freeze and 4 new ones, §4). The RTL is unchanged since the freeze |
| Assertions, `tb/cim_assertions.sv` | **47, all exercised, none failing**, over the 28 benches |
| Mutation test, `model/verif_mutants.py` | **16 of 16 broken designs caught** by the assertion named for each; 2 of them only by the assertion, the benches alone pass |
| Coverage, all benches merged | **99 % of lines, 98 % of branches, 97 % of expressions, 93 % of toggle points.** Every uncovered line is explained (§3) |
| Extreme input | **Exact against the golden model**, both front-ends: digital silence, clicks, full-scale square waves, windows with a maximum ≤ 0, saturating PDM |
| Post-synthesis, whole endpoint | **Exact:** 9 layers × 8,000 activation values, and one full inference with its logits, in 85,352 cycles, the RTL's count |

## 2. Assertions

They are bound from outside the RTL, so the synthesised design is untouched. Each counts its activations as well as its failures: a check that never fires proves nothing, and the report rejects it.

| Unit | Checks | What they state |
|---|---:|---|
| `cim_engine` | 9 | start only when ready; done exactly at t0 + D + 5; ready low exactly D − 1 cycles; input bank and tile stable while read; done is a pulse |
| `cim_seq` | 13 | no error flag; a tag for every result; results drained before the next; tag queue bounded; gather only into a free bank; row, slot, position and layer in range; one phase at a time; no requantisation lost or duplicated |
| `cim_fe`, `mfcc_ded` | 4 | done ends busy; reads inside the frame; frame base stable; frame time bounded |
| `cim_audio` | 11 | sample period; 320 samples per hop; hop and decision counts step by one; decisions only on network hops; stride bounded; starts only when allowed |
| `pdm_dec` | 3 | PDM clock period; duty within the microphone's 40–60 %; sample valid is a pulse |
| `cim_endpoint` | 7 | rvalid the cycle after gnt; no gnt without req; one engine master; no CPU go on an owned engine; interrupt exactly when enabled and pending |

Two corrections to the checks, both found by running them:
- A channel range check wrote `d_c < 6'd64`, in which 64 truncates to 0, so it always failed. The check was wrong, not the design; a 6-bit channel cannot leave its range anyway.
- The sample-period check failed once, after a restart, with an interval of 328 cycles instead of 320. The decimator is not reset by a restart (`LIVE_AUDIO.md` §3), so a sample already in the FIR arrives first and its interval includes the stop.
  The check now starts at the second sample of a stream.

### Mutation test

| Broken design | Caught by | Bench alone |
|---|---|---|
| Engine issues one cycle early | `ready_low_d_minus_1` | hangs |
| Engine result one cycle late | `done_at_d_plus_5` | fails |
| CPU go granted while the engine is busy | `start_only_when_ready` | hangs |
| Input bank released too early | `input_bank_stable_port2` | fails |
| Passes too close for requantisation | `results_drained_before_pop`, `no_error_flag` | fails |
| A requantised result lost | `requant_keeps_count` | fails |
| Hop one sample short | `hop_is_320_samples` | fails |
| Decision counted twice | `decision_count_steps` | fails |
| Decision on a hop without the network | `decision_only_on_network_hop` | fails |
| Stride counter runs past k − 1 | `stride_counter_bounded` | fails |
| **PDM clock outside 40–60 % duty** | `pdm_clock_duty_40_60` | **passes** |
| **Response one cycle late** | `rvalid_follows_gnt` | **passes** |
| Interrupt ignores its enable | `irq_only_when_enabled` | fails |
| Sequencer may start on `cim_fe`'s engine | `seq_fe_not_with_cim_fe` | fails |
| Front-end reads past its frame | `reads_inside_its_frame` | fails |
| B3 reads past its frame | `reads_inside_its_frame` | fails |

A 17th mutant is **equivalent at today's timing**: removing `eng_ready_i` from the sequencer's issue changes nothing, because gathering and requantisation already space the passes at least 9 cycles apart. It changes nothing in the
bench, as predicted. It would matter at D > 9.

## 3. Coverage

| Module | Branches | Expressions | Lines | Toggle points |
|---|---:|---:|---:|---:|
| `cim_engine` | 36/38 | 77/78 | 22/22 | 2817/2854 |
| `cim_fe` | 95/98 | 76/76 | 64/65 | 1697/1904 |
| `mfcc_ded` | 60/60 | 49/49 | 41/42 | 2074/2288 |
| `cim_log2fix16` | 10/10 | 4/4 | 4/4 | 318/322 |
| `cim_seq` | 195/200 | 156/158 | 85/85 | 5023/5340 |
| `pdm_dec` | 26/26 | 22/22 | 14/14 | 2556/2568 |
| `cim_audio` | 38/38 | 22/22 | 7/7 | 390/524 |
| `cim_endpoint` | 51/52 | 86/96 | 36/36 | 1709/2112 |
| **Total** | **511/522 (98 %)** | **492/505 (97 %)** | **273/275 (99 %)** | **16584/17912 (93 %)** |

Most of the untoggled points are the high bits of 32-bit counters and addresses that no bench runs long enough to reach.

**The 13 source lines with an uncovered point:**

| Line | Why |
|---|---|
| `cim_seq.sv:143, 144` (requantisation clamps), `cim_engine.sv:71, 72` (input width limits) | **A measuring artefact:** Verilator reports no hits for an `if (...) return` arm inside a function, yet they do run: explicit counters give 164,672 high and 41,923,834 low clamps, and the engine bench now checks widths 0, 25 and 31 |
| `cim_seq.sv:482`, low arm of the feature clamp | **Unreachable by arithmetic:** the lowest possible value is exactly −128, with all mel bins at the floor; the other coefficients stay far above it |
| `cim_seq.sv:828`, the error flag | Only a fault reaches it; the mutation test shows it working |
| `cim_fe.sv:349, 366, 417`, waiting for ready | Unreachable by protocol: ready rises before done, and these states follow a done |
| `cim_fe.sv:444`, `mfcc_ded.sv:362`, `default:` | Defensive; no illegal state occurs |
| `cim_seq.sv:408`, a window path with no new frame and the same maximum | Reachable only by firmware that restarts the window path by hand; the live path never does |
| `cim_endpoint.sv:442`, a CPU sample write in the cycle the window path takes the maximum | Reachable only in the CPU-fed mode, by firmware that writes during a run; the firmware does not |

The last two are real, unexercised paths, outside the live design's use.

**Measured logic activity.** The energy estimate assumes that 10 % of the logic switches per active cycle. Toggle counts of the RTL's own signals give about 0.04 per bit per busy cycle in `cim_seq` and about 0.10 in `cim_fe`. Gates can switch more than named signals do, so this
supports 0.1 as a reasonable to cautious value; the estimate is unchanged.

## 4. New directed tests

They exist because coverage showed what the end-to-end benches cannot reach.

| Bench | Stimulus | Result |
|---|---|---|
| `tb_cim_audio_dir` | The hop controller alone, with a modelled front-end and sequencer that stall and take random times: k = 1, 2, 4, no network, a sequencer that overruns a hop, restarts. Sequencer word, frame base, hop maximum and ring address checked per hop | 0 errors, 117,538 checks, 30 decisions |
| `tb_pdm_dec` on `tb/pdm_vec_sat` | 1 s of full-scale PDM: all ones, all zeros, all ones, a square wave, so the FIR and DC blocker saturate both ways | all 16,000 samples equal the golden model |
| `tb_cim_stream` on `tb/stream_vec_edge_{cim,ded}` | 3 s of digital zeros, single full-scale clicks, a smallest positive peak, a full-scale 1-kHz square wave, then a second that never rises above zero | all 150 hops and 101 decisions exact, features and logits, both front-ends |
| `tb_cim_engine`, `tb_cim_endpoint` | Input widths 0, 25, 31; the `CYCLES` register against the clock | exact |

The overrun test confirms what `LIVE_AUDIO.md` §3 states: the flag is sticky and clears on request, the skipped hop is not counted, and a restart starts everything over.

## 5. Post-synthesis smoke test of the whole endpoint

Until now, gate-level runs covered `cim_audio` and earlier versions of the sequencer. `sim/run_endpoint_nl.sh` runs `tb_cim_seq` under xsim, on the Vivado functional netlist of the whole endpoint, from the routed out-of-context checkpoint with `cim_fe` (the same RTL as the current bitstream).

| Check on the netlist | Result |
|---|---|
| 61 network tiles and 640 parameter entries loaded over the bus | as firmware does |
| Stop-after-layer runs L0 to L8, every output read back through the ACT window | **0 of 72,000 values differ** from the golden model |
| Cycles at each stop, 8,034 to 85,254 | as in RTL |
| One full inference, its 12 logits and class | exact, **85,352 cycles** from go to interrupt, the same as RTL |

The run takes 81 minutes for 691,000 simulated cycles, so it is a smoke test: one clip, not the regression.

It does not run the audio path through the netlist, because reaching the first decision takes 15 million cycles at this speed; `cim_audio`'s own netlist check covers that part (`LIVE_AUDIO.md` §9).

## 6. Limits

- Simulation only: the assertions hold on these stimuli. Nothing here is a formal proof.
- Verilator is 2-state, so X propagation is not checked there; the netlist runs under xsim are 4-state.
- The behavioural engine of `TIMING_SENSITIVITY.md` is a different model and is validated only at the RTL's timing.
- The SoC simulation and the board suite were not re-run in this pass; the RTL they used has not changed. The board ran the logger of `SAME_AUDIO.md`.

## 7. Reproduce

```
cd hopcim
sim/run_all.sh                       # 40 benches (28 R6b, 11 S, 1 that must fail), about 15 min
sim/run_cov.sh 6                     # assertions and coverage, about 30 min: sim/cov/
python3 model/verif_report.py        # model/verif_results.json
python3 model/verif_mutants.py 4     # about 8 min: model/verif_mutants.json
sim/run_endpoint_nl.sh               # the endpoint's netlist under xsim, about 80 min
```

## 8. The baseline S (2026-11-08)

S changed `cim_seq` (`BASELINE_S.md`), behind a parameter that is off by default, so the frozen benches still test R6b
unchanged. All of the checks above were re-run with S added:

| Check | Result |
|---|---|
| Regression, `sim/run_all.sh` | **40 of 40 pass**: the 28 benches above (R6b, unchanged), 11 with S (sequencer, stream, edge vectors, live path at 15.36, 5.12 and 4.096 MHz, k = 1, 2, 4, both front-ends), and one that must fail: R6b at 4.096 MHz overruns |
| Assertions, `sim/run_cov.sh 6` | **51, all exercised, none failing**, over 39 benches (34 regression benches built with the assertions, and the 5 directed runs). The 4 new ones for S (`sign_gate_exact`, `pair_writes_distinct_lanes`, `lane_channel_parity`, `l8_act_only_when_stopped_after_l8`) fired 24,000 to 33.7 million times |
| Mutation test, `model/verif_mutants.py 4` | **27 of 27 caught**: the 16 above and 11 for S; 2 changes equivalent at today's timing, as predicted |
| Coverage, all benches merged | unchanged: 99 % of lines, 98 % of branches, 97 % of expressions, 93 % of toggle points; the same 13 explained lines |
| All 4,890 test clips through the RTL (`sim/run_seq_allclips.sh`) | **0 errors with R6b and with S** (logits and class; 91.80 %) |
| Synthetic set 2 (zero point −128, the sign gate's boundary) | 0 mismatches with R6b and with S |

`sign_gate_exact` recomputes TFLite's whole requantisation for every gated result and requires −128; over the
benches it checked 23.9 million gated results.
