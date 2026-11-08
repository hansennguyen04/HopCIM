# End-to-end evaluation: what each step moved into hardware buys

Status (2026-11-07): every configuration below is measured on the Nexys A7-100T, except where
marked. Every measured cycle count equals Verilator's for the same code. Measured energy is not
claimed (FPGA emulation, see the proposal §9.5); the measured proxies are CPU-active cycles and the
hardware busy time per hop. §9 adds a model-based estimate, labelled as such.

## 1. Configurations

B–E compute the same int8 logits, bit-exact with the TFLite reference kernels. A's network is
bit-exact too; its front-end cost is that of the CPU census kernels (an MFCC of the same size, not
the same integer arithmetic). The model is
the unchanged MLPerf Tiny DS-CNN on the exact integer CIM-MFCC features, 91.80 % on the 4,890 GSC
test clips (B3 front-end: 91.60 %).

| # | Front-end (per frame) | Window path (per hop) | Network (per decision) | Audio input |
|---|---|---|---|---|
| A | CPU (FFT, mel, log, DCT) | CPU | CPU, plain C (`hopcim_cpu`) | CPU copies samples |
| B | CPU drives CIM passes (R4b) | CPU | CPU drives CIM passes (R6a) | CPU |
| C | hardware (R5a) | CPU | layer sequencer (R6b) | CPU |
| D | hardware (R5a) | sequencer (R5b) | sequencer (R6b) | CPU copies samples |
| E | hardware (R5a) | sequencer (R5b) | sequencer (R6b) | hardware (R7, on-board microphone) |

## 2. Per hop, one decision every 20 ms hop (cycles)

| # | CPU-active per hop | Hardware or CPU busy time per hop | Lowest clock for a decision every hop |
|---|---:|---:|---:|
| A | ≈ 48.1 M: front-end 161k + network 47,974,801 | ≈ 48.1 M | ≈ 2.4 GHz |
| B | ≈ 13.17 M: front-end 146,364 + network 13,020,998 | ≈ 13.17 M | ≈ 660 MHz |
| C | ≈ 7.3k typical (front-end I/O 4,743 + window path 2,120 + network control 459); ≈ 109k when the window maximum changes | ≈ 99.5k typical, ≈ 201k worst | 5.0 MHz typical, 10.1 MHz worst |
| D | **4,447** (copying 320 samples 3,847, everything else 600) | 97,512 typical, 102,024 worst | **5.1 MHz** |
| E | **263** (interrupt, PLIC, read and clear the decision, keyword, LED and class-count logic, sleep loop; max 289–292 over the board runs) | ≈ 93.1k typical, 97.7k worst (front-end 7,582 + window path and network) | **5.12 MHz** (the microphone needs 1.024 MHz = f_sys / integer); no overrun at 5.12 MHz in RTL and on the board |

Sources:
- A: network measured on the board; front-end per frame from the earlier CPU census (FFT512
  145k + mel/log/DCT 15.6k).
- B: `hopcim_mfcc` and `hopcim_net1`, board = Verilator.
- C: composed from measured parts (`hopcim_fe_irq`, `hopcim_seq`, `WINDOW_PATH.md` §1).
- D: `hopcim_stream`, board = Verilator.
- E: CPU-active measured on the board (`hopcim_live`, 15.36 MHz: exactly 500 decisions per 10 s,
  no overrun). An earlier build without the LEDs and the class count measured 216 (max 217).
  Hardware time from `tb_cim_stream` (window path + network, go → interrupt) plus the front-end.
  The lowest clock is the smallest integer multiple of the 1.024 MHz PDM clock whose hop
  (20,480 × N cycles) holds the worst case. At 5.12 MHz, `tb_cim_mic` (PDM divider 5) processes
  every hop with no overrun and all decisions bit-exact. The 5.12 MHz bitstream gives 500 decisions
  per 10 s on the board with no overrun (50 s, `LIVE_AUDIO.md` §7).

**What the steps buy:**
- **B over A:** the CIM does the multiplications, but the CPU still moves every operand and
  rescales every result. Neither front-end nor network meets real time; at 15 MHz a decision
  takes 0.88 s.
- **C over B:** the front-end's whole frame path and the network's gather, sequencing and
  requantisation move into hardware. The hardware time per hop drops 132× (13.17M → 99.5k), and
  real time is met at 15 MHz with a 3× margin.
- **D over C:** the per-window normalisation moves into hardware. The CPU-active time per hop is
  now dominated by copying samples, and the worst case falls from about 109k to 4,447.
- **E over D:** the microphone path moves into hardware. The CPU wakes only to read decisions:
  263 cycles per 20 ms hop, 17× less than D. That is 0.09 % of the hop at 15.36 MHz, and about
  183,000× less than A's 48.1 M.

## 3. Network alone

| | Cycles per inference | vs CPU-only |
|---|---:|---:|
| CPU only, plain C, `-O2` (`hopcim_cpu`, board) | 47,974,801 | 1× |
| CPU drives CIM passes (`hopcim_net1`, board) | 13,020,998 | 3.7× |
| **Layer sequencer (`hopcim_seq`, board, go → interrupt)** | **85,420** | **562×** |

CPU-only per layer type: conv1 6.36M, depthwise 7.88M, pointwise 33.0M, pool 0.72M, FC 15k.

The engine is busy 67,509 cycles of the sequencer's 85,352, so 79 %. The depthwise layers run at
the engine's own rate.

## 4. Front-end alone (per 20 ms frame)

| | CPU-active | Latency | Notes |
|---|---:|---:|---|
| CPU only | ≈ 161k | ≈ 161k | census |
| CPU drives CIM passes (R4b) | 146,364 | 146,364 | 1.1× (the MACs alone) |
| `cim_fe` (R5a) | 4,743 (samples, readback, interrupt) | 7,582 | CPU asleep meanwhile |
| B3, dedicated radix-2 (`mfcc_ded`) | 4,743 | 8,674 | same logic cost; H4 not supported (`B3_BASELINE.md`) |

## 5. Hardware cost (Vivado out of context, xc7a100t)

| Endpoint stage | LUT | FF | BRAM tiles | DSP |
|---|---:|---:|---:|---:|
| R5a: engine 128 tiles + `cim_fe` | 3,882 | 2,908 | 34.5 | 135 |
| R6b: + layer sequencer | 5,155 | 3,860 | 44 | 138 |
| R5b: + window path | 6,456 | 4,947 | 46 | 144 |
| R7: + live audio path (`cim_audio`, with the DC blocker) | 7,617 | 5,811 | 46 | 145 |

## 6. Accuracy of the live path

The live path adds the microphone and the decimator, which are not part of the model-preserving
claim: the input is a microphone, not a GSC recording.
- On a simulated 3-s stream (2nd-order ΣΔ model of the microphone), the decisions on the
  decimated audio equal those on the original audio at 94 of 101 hops with `cim_fe` and 96 of
  101 with B3. Every difference is at a class transition of the original (silence/unknown,
  off/on), which the decimated audio moves by 1–4 hops; the original's top-two logit margin
  there is 1–10.
- The decimator is flat within 0.03 dB over the mel band.
- On the board, with the real microphone (`hopcim_micrec`, `LIVE_AUDIO.md` §6), the logits of all
  25 decisions whose window lies inside a 1.5-s recording equal the golden model's on the recorded
  samples.
- A quiet room reads as `_unknown_` at every hop, never `_silence_`. Its noise is mostly 60-Hz
  mains harmonics, but the model's answer stays the same after an offline 300-Hz high-pass: this
  is how the model treats faint room noise scaled to full scale by the window normalisation.
  Neither class is a keyword.
- Spoken keywords (two informal runs, `LIVE_AUDIO.md` §8): 22 and 27 keyword events, with no
  overrun. The scripted test with a speaker (`hopcim_kwtest` + `score_kwtest.py`): 48 of 50 keywords
  at a decision every 20 ms (§7).

## 7. How often the network has to run

Configuration E with the network every k hops (`AUD_CTRL[19:16]`, `LIVE_AUDIO.md` §9). The front-end and
the window path still run every hop, and every decision stays bit-exact.

| A decision every | CIM passes per hop | Engine busy per hop | CPU-active per decision (board) | Decisions per 10 s, overrun (board) | Logits = golden on real microphone audio (board) |
|---|---:|---:|---:|---:|---:|
| 20 ms | 7,501 | 67,509 | 263 | 500, 0 | 25 of 25 |
| 40 ms | 3,750 | 33,755 | 263 | 250, 0 | 13 of 13 |
| 80 ms | 1,875 | 16,877 | 266 | 125, 0 | 7 of 7 |

| With a real speaker (`LIVE_AUDIO.md` §9) | 20 ms | 40 ms | 80 ms |
|---|---:|---:|---:|
| Keywords correct, of 50 | 48 | 47 | 42 |
| Nothing reported, of 50 | 0 | 1 | 6 |

Cost: 18 LUT and 8 FF on the endpoint. The lowest real-time clock stays 5.12 MHz, because a decision
hop still runs the whole network. On synthetic streams, keyword recall did not measurably change at
40 or 80 ms. With a real speaker, 40 ms shows no loss (p = 1.000). 80 ms lost six words (p = 0.109, one
speaker, new speech each run), so 40 ms is the operating point.

## 8. Activation reuse: evaluated in software, not built

Reusing activations across overlapping windows is exact (0 mismatches over 174,216 decisions), and
saves 45.5–51.3 % of the passes at 50 decisions per second: about what 40 ms gives with no memory. It
needs about +72 KB of activation memory, or +32 KB with one decision per 2 or 4 hops, and its worst
case stays a full recompute. Details, and the conditions for building it: `REUSE_EVALUATION.md`.
With published tools and leakage included, it shows a net saving (§9).

## 9. Estimated energy (a model, not a measurement)

At 28 nm, from exact activity counts and published tools (`ENERGY_ESTIMATE.md`). Nothing here comes from
FPGA power.

| Average power, a decision every | 20 ms | 40 ms | 80 ms |
|---|---:|---:|---:|
| Baseline total | 110 µW | 75 µW | 58 µW |
| Range over the sensitivity variants | 94–192 µW | 60–117 µW | 42–84 µW |
| Leakage share of the baseline | 33 % | 48 % | 62 % |
| Baseline with simple activation reuse | 84 µW | 62 µW | 54 µW |

One inference is about 1.39 µJ (1.09–3.01 µJ). The microphone and the X-HEEP host are not included.

## 10. Checks of these results (2026-11-07)

- `TIMING_SENSITIVITY.md`: how the cycle counts and the lowest clock depend on the assumed macro. They hold for an issue interval of at most 9 cycles.
- `SAME_AUDIO.md`: rate, phase and confirmation rule on identical audio, and false alarms per hour. The current rule reported falsely in a room while someone was in it, and not once in the 7-hour night after.
- `VERIFICATION.md`: assertions, coverage, mutation tests, extreme input and a post-synthesis smoke test of the endpoint.
- `ENERGY_ESTIMATE.md` §11: ZigZag-IMC itself against the transcribed equations.

## 11. The baseline S (2026-11-08)

The StreamFuse study (`STREAMFUSE_STUDY.md`) found that once the CPU is gone, requantisation, not the activation
memory, is the largest cost, and that half of all requantisations need no multiply. S builds that finding:
sign-gated requantisation, a second requantisation lane and no L8 write to ACT (`BASELINE_S.md`). It is now the
baseline that any later optimisation must beat.

| | R6b | S |
|---|---:|---:|
| Network, go → interrupt (board) | 85,420 | **67,909** |
| Worst hop (RTL, hop monitor) | 97,583 | **80,072** |
| Lowest real-time clock (RTL and board) | 5.12 MHz | **4.096 MHz** |
| Network energy per inference [E] | 1,389 nJ | **1,121 nJ** |
| Total power at 40 ms [E] | 75.3 µW | **68.8 µW** |
| CPU-active per decision (board) | 263 | 263 |
| Endpoint, LUT / FF / DSP (out of context) | 7,635 / 5,819 / 145 | 8,116 / 5,977 / 147 |

The bottleneck moves again: with S, the depthwise layers (engine-bound, 45,092 of 67,841 cycles) and leakage
(36 of 69 µW at 40 ms [E]) dominate.

## 12. B1: one StreamFuse-B block in RTL (2026-11-08)

The one-block prototype the gates allow (`STREAMFUSE_B1.md`): L1 → line buffer → fusion vector → L2, on top of S.

| | S | B1 |
|---|---:|---:|
| Network, go → interrupt (board) | 67,909 | **67,885** |
| Network cycles, RTL against the schedule model | 67,841 = model | **67,817 = model**, every segment and access count |
| ACT word reads / byte writes per inference | 42,960 / 64,000 | 33,470 / 56,000 |
| Network energy per inference [E] | 1,120.8 nJ | **1,094.1 nJ** (−2.4 %) |
| Total power at 40 ms [E] | 68.8 µW | 68.7 µW |
| Endpoint, LUT / FF (out of context, same source) | 8,126 / 5,980 | 8,761 / 6,182 |
| Live path on the board: overrun, CPU-active, real-audio logits | 0, 263, exact | 0, 263, exact |

B1 is not worth keeping for its own gain; it shows that the schedule model behind the four-block numbers (B: 0.837 × S,
0.575 × the storage) predicts B's mechanisms in RTL exactly, and that their control costs +420 LUT, +197 FF.
