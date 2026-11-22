# R7 — the live audio path: microphone to keyword without the CPU

Status (2026-10-30): built and verified in RTL, end to end with both front-ends, and on the board:
- 500 decisions per 10 s with the CPU asleep, at 263 CPU-active cycles per decision;
- on real microphone audio, the board's logits equal the golden model's (§6).

The lowest real-time clock, 5.12 MHz, is verified in RTL and on the board (§7). Spoken keywords are detected live on the board (§8). The scripted accuracy test has been run with a speaker at a decision every 20, 40 and 80 ms (§9): 48, 47 and 42 of 50 keywords.

Added 2026-11-05: a decision every k hops, verified in RTL and on the board (§9).

## 1. What it does

The Nexys A7-100T has an on-board PDM MEMS microphone (ADMP421, clock J5, data H5, L/R select
pulled low), so no external microphone is needed.

R7 connects it to the CIM endpoint. Every 20 ms hop then runs entirely in hardware:

```
microphone --PDM 1.024 MHz--> pdm_dec (CIC + FIR) --16 kHz--> audio ring --> front-end (frame)
           --> window path (R5b) --> DS-CNN (R6b) --> decision: class, interrupt
```

The CPU loads the tiles and parameters once, enables the audio path, and sleeps. It wakes only on
a decision interrupt, to read the class (and print a detected keyword).

## 2. Decimator (`rtl/pdm_dec.sv`, golden `model/pdm_dec.py`)

**Clock.** The system clock becomes 15.36 MHz, so the PDM clock is 15.36 MHz / 15 = 1.024 MHz
and the audio is 1.024 MHz / 64 = 16 kHz exactly: the model's sample rate.
- MMCM: VCO = 100 MHz × 48 / 5 = 960 MHz, divided by 62.5.
- Build flag `use_cim_mic`.
- Firmware: `REFERENCE_CLOCK_Hz = 15360000`.

The data bit is taken at the rising edge of the PDM clock, as X-HEEP's own `pdm2pcm` does with
this microphone, through a 2-flip-flop synchroniser. The duty cycle is 8/15, within the ADMP421's
40–60 %.

**Filter.**
- CIC: 5th order, decimation 64, 32-bit wrap-around arithmetic, gain 2^30.
- y1 = (c + 2^14) >> 15.
- FIR: 15 taps at 16 kHz in Q14, designed by least squares (numpy only) to undo the CIC droop over
  the model's mel band (20 Hz – 4 kHz, MLPerf Tiny `get_dataset.py`) and roll off above it.

A first-order DC blocker follows the FIR: `acc += ((y − y_prev) << 8) − (acc >> 8)`, output
`sat16((acc + 2^7) >> 8)`, −3 dB at about 10 Hz and −0.95 dB at 20 Hz. The board's microphone
showed a DC offset of about +1177 against room noise of RMS 70. The model normalises each 1-s
window by its maximum sample, so the offset would shift every feature against the training data.

| Property | Value |
|---|---|
| Pass band 100 Hz – 4 kHz | flat within 0.03 dB |
| 5 / 6 / 7 kHz | −2.0 / −11.7 / −45.9 dB |
| Rejection of everything that folds into 0–4 kHz | ≥ 52 dB (CIC alone) |
| 1 kHz tone through the simulated microphone (2nd-order ΣΔ) | SNR 76.9 dB |
| A tone on a DC offset of 1177 | mean after the blocker 0.6, the tone unchanged |
| Cost | one 32-bit adder chain, 5 comb registers, one 17 × 16 multiplier used 15 of every 960 cycles, a 28-bit DC-blocker accumulator |

## 3. Audio path and hop controller (`rtl/cim_audio.sv`)

- Every decimated sample is written straight into the endpoint's 1024-sample audio ring.
- The hop maximum is snapshotted exactly at each 320th sample, so the window path's block maxima
  are those of the reference, even though the next hop's samples keep arriving.
- At the end of hop h, the hop controller:
  1. for h ≥ 1, starts the front-end on frame h−1 (samples 320(h−1) … 320(h−1)+479);
  2. then starts the sequencer: window path; a new frame if h ≥ 1; a stream reset if h = 0; FEAT
     row h mod 49; from hop 49 on, the network;
  3. on its completion, latches the class, counts the decision and raises the decision interrupt.
- A hop takes 307,200 cycles at 15.36 MHz, and its processing about 98k, so an overrun (flagged,
  sticky) never happens.
- Optionally, the network and its decision run only every k hops (§9). The front-end and the window
  path still run at every hop.

**Registers:**

| Register | Bits |
|---|---|
| `AUD_CTRL` (0x34) | [0] enable, [1] network, [14:8] first tile, [19:16] k − 1: a decision every k hops (0, the reset value: every hop) |
| `AUD_STATUS` (0x38) | enabled; decision pending (W1C); overrun (W1C); last class; decisions since the start |
| `AUD_HOPS` (0x3C) | hops since the start |

`IRQ_EN[3]` raises the decision interrupt. `SAMPLES` can be read back, which lets firmware check
the microphone. While the path is enabled, CPU front-end and sequencer starts and `SAMPLES`
writes are ignored.

**Usage rules and limits** (from the verification review, 2026-10-30; none affects the firmware,
which never does these):
- CPU engine passes (`CTRL` go) and `FEAT`/`PARAMS` writes are *not* blocked while the path runs.
  The audio path waits for a CPU pass, but a CPU go granted in the same cycle as an audio-path
  start loses its completion (`DONE_CNT`), and `FEAT`/`PARAMS` writes between hops change the
  stream. Firmware must leave the engine and those buffers alone while `AUD_CTRL[0]` is set.
- An overrun skips that hop's processing (one frame and one block maximum are lost) and leaves the
  FEAT row counter one hop behind. The following decisions, about 49, are computed partly on
  misplaced rows, and then the stream is consistent again. Firmware that sees the overrun flag
  can restart the stream (`AUD_CTRL[0]` 0 → 1). No overrun has occurred at 5.12 MHz or above.
- A restart resets the ring pointer and the hop controller, but not the decimator's filter states
  (CIC, FIR, DC blocker). The first samples after a restart continue from the old states; a sample already
  in the FIR when the path stopped can arrive first, so the first interval can be longer by the stop. `tb_cim_audio_dir` covers restarts and overruns
  (`VERIFICATION.md`), and the firmware enables the path once.

**Pins.** X-HEEP's pad ring owns the I/O buffers of the two microphone pads. An FPGA pad has one
I/O buffer, so the wrapper can neither drive J5 nor read H5 a second time.
- The first attempt gave X-HEEP's PDM ports dummy nets instead of the pins, which left the pad
  ring's buffers without a pin; placement failed ("IBUF … is unplaced after IO placer").
- Now, under `USE_CIM_MIC`, `x_heep_system` gets two ports. `cim_pdm_clk_i` drives the clock pad's
  output (enable 1) instead of `pdm2pcm`'s clock. `cim_pdm_data_o` is the data pad's core-side
  input.
- The wrapper connects them to the endpoint. X-HEEP's `pdm2pcm` still sees the data but no longer
  drives the clock.
- Both the generated `hw/system/x_heep_system.sv` and its template `x_heep_system.sv.tpl` carry the
  change, so it survives a regeneration.

The testharness ties the microphone input low.

## 4. Verification

| Check | Result |
|---|---|
| `tb_pdm_dec`: 1 s of PDM (a GSC clip through the ΣΔ model, 1,024,000 bits) | all 16,000 samples equal `decimate()` |
| `tb_cim_mic`, `cim_fe` and B3: a 3-s speech stream as PDM on the microphone pin. The testbench only enables the path and answers the decision interrupts | all 101 decisions bit-exact (logits, class, count); 150 hops; no overrun; the audio ring equals the golden decimated stream (the slots not overwritten after the stream ends) |
| `tb_cim_mic` at PDM divider 5 (5.12 MHz, §7), `cim_fe` and B3 | all 101 decisions bit-exact; no overrun; the audio ring exact |
| Earlier testbenches, with the audio path present but off | all pass |
| On the board, real microphone audio (`hopcim_micrec`, §6) | the logits of all 25 decisions whose window lies in the recording equal the golden model's |

**The microphone path barely changes the decisions.** On the 3-s stream, the decisions made on the
decimated audio equal those on the original 16 kHz audio at 94 of 101 hops with `cim_fe` and 96 of
101 with B3. Every difference is at a class transition of the original (silence versus unknown,
off versus on), which the decimated audio moves by 1–4 hops (20–80 ms); the original's top-two
logit margin there is 1–10.

## 5. Cost (Vivado out of context, xc7a100t)

| Unit | LUT (memory) | FF | BRAM tiles | DSP | WNS at 15 MHz |
|---|---:|---:|---:|---:|---:|
| `pdm_dec` | 891 | 710 | 0 | 1 | +47.1 ns |
| `cim_audio` (with `pdm_dec`) | 1,021 | 846 | 0 | 1 | +47.4 ns |
| Endpoint with `cim_fe`, R7 | 7,617 (629) | 5,811 | 46 | 145 | +44.4 ns |
| Endpoint with B3, R7 | 7,471 (909) | 5,906 | 45 | 146 | +46.6 ns |

R7 adds about 1,160 LUT, 860 FF and 1 DSP to the R5b endpoint, and no block RAM. The DC blocker
accounts for about 100 LUT and 44 FF of it.

A first build lost the audio ring to LUT-RAM (288 RAM64M, about 1,150 LUTs). The microphone and
the CPU wrote it from two separate statements, which Vivado infers as two write ports. With the
read port, that no longer fits one block RAM. The ring now has one write statement with
multiplexed enable, address and data.

## 6. Bitstream and board

**Bitstream** `xheep_nexys_bscan_cim128mic.bit` (`--flag use_cim --flag use_cim_mic`, `cim_fe`):

| Resource | Value |
|---|---|
| LUT | 42,281 (67 %) |
| FF | 40,213 |
| Block RAM tiles | 82 |
| DSP | 146 |
| System clock | 15.360 MHz; setup +15.94 ns, hold +0.229 ns |

The SPI-slave crossings that every Nexys build reports are explained in `INTEGRATION.md`.

**Live run** (`hopcim_live`, quiet room, 40 s):

| Check | Result |
|---|---|
| Microphone after 1 s (1,024 samples read back) | min −140, max 159, mean 0, RMS 70. Before the DC blocker the mean was +1177 |
| Decisions | 500 per 10 s, one per 20-ms hop; overrun 0 |
| CPU-active per decision | 263 cycles (max 289) |
| Classes | `_unknown_` for every decision |

The 263 cycles cover the interrupt, the PLIC, reading and clearing the decision, the keyword, LED and class-count logic, and the sleep loop. An earlier build without the LEDs and the class count measured 216 cycles (max 217).

**LEDs:**
- LD10–LD13 show the last keyword for 1 s, as 1 (down) … 10 (yes) in binary, with LD10 the least significant bit.
- LD14 is lit while the decision is not `_silence_`.
- LD0 blinks at 1 Hz while decisions arrive.

UART prints `>> <keyword>` when a keyword has been the top class for 5 decisions in a row (100 ms).

**Board against the golden model on real microphone audio.** `hopcim_micrec` records 1.5 s of the microphone (hops 100–174) from the audio ring. It also records every decision the hardware made meanwhile, with its 12 logits. `model/check_micrec.py` recomputes the logits from the recording.

| Check | Result |
|---|---|
| Decisions whose 1-s window lies inside the recording | 25 of the 74 recorded |
| Logits that differ from the golden model | 0 |
| Hop / decision-count alignment, overrun | exact, 0 |

So the live path is bit-exact on the board as well as in RTL, from the decimated microphone samples to the logits.

**The quiet room.** The recording has an RMS of 65:
- 96 % of the power lies below 300 Hz, with its strongest bins at the 60-Hz mains harmonics (62, 125, 188 Hz at a bin width of 31 Hz);
- above 300 Hz, the noise is about 12 LSB RMS.

The model calls every window of it `_unknown_`, also offline after a 100–300 Hz high-pass. So this is how the model treats faint room noise once the window normalisation scales it to full scale. It is not caused by the hum alone, and not by the hardware. GSC's `_silence_` clips are broadband background noise. Both classes mean "no keyword", so this alone is harmless. **A longer run did find false reports in this room while someone was in it, though none in the 7-hour night after:** see `SAME_AUDIO.md` §6.

**Speech.** Pending: it needs someone speaking keywords near the board (§8).

## 7. The lowest real-time clock: 5.12 MHz

A hop is 320 × 64 × N cycles at a PDM divider N (system clock 1.024 MHz × N), and its worst-case processing takes about 97.7k cycles. N = 5 (5.12 MHz, 102,400 cycles per hop) is therefore the smallest divider that still keeps up. At N = 4 a hop has only 81,920 cycles.
- **Build:** `--flag use_cim_mic5`, together with `use_cim_mic`. MMCM: VCO = 100 MHz × 32 / 5 = 640 MHz, divided by 125. Endpoint: `PDM_CLKDIV = 5`.
- **PDM clock:** high for 3 of 5 cycles, a duty cycle of 60 %, at the edge of the ADMP421's 40–60 % range.
- **RTL** (`tb_cim_mic` with `CLKDIV = 5`, both front-ends):
  - all 101 decisions bit-exact;
  - no overrun, including the 24 hops that recompute all 49 frames;
  - the last decision at cycle 15.49 M, against 46.21 M at N = 15.
- **Board** (`xheep_nexys_bscan_cim128mic5.bit`, system clock 5.120 MHz, setup +15.06 ns, hold +0.259 ns; firmware with `REFERENCE_CLOCK_Hz = 5120000`):
  - 2,500 decisions in 50 s, i.e. 500 per 10 s;
  - no overrun;
  - 263 CPU-active cycles per decision, as at 15.36 MHz;
  - the microphone works with the 60 % clock: mean 0, RMS 44;
  - on real microphone audio (`hopcim_micrec` built for 5.12 MHz, 1.5 s, 2026-10-30), the logits of
    all 25 decisions whose window lies in the recording equal the golden model's. The recording's
    spectrum is the room's (85 % of the power below 300 Hz, mains harmonics), not the broadband
    noise of a mis-sampled data line.

  The hardware's cycle counts are the same on the board as in simulation, so the RTL margin at N = 5 (≥ 4.7k cycles per hop) holds on the board too.

## 8. Live demo

```
cd planenest/hw
FOLLOW=1 ./board_run.sh hopcim_live_nexys.elf bitstreams/xheep_nexys_bscan_cim128mic.bit 300
```

`FOLLOW=1` prints the UART output live; the log is also kept in `hopcim_live_nexys_board_<time>.uart.log`. Ctrl-C ends the run and stops OpenOCD and the UART capture too (before this was added, they stayed running and held the cable, and the next run failed with `LIBUSB_ERROR_BUSY`; the script now names the processes to stop). The CPU reset button (CPU_RESETN) resets X-HEEP, which then waits in the boot ROM for a program over JTAG, so after a reset run the command again.

Speak a keyword (down, go, left, no, off, on, right, stop, up, yes) clearly, about 20–50 cm from the board's microphone. A keyword is reported when it is the top class for 5 decisions in a row (100 ms). The same keyword is reported again only after a non-keyword decision. A reported keyword prints `>> <keyword>` and appears on LD10–LD13 for 1 s. The class counts every 10 s show what the model heard at each decision.

**First spoken tests** (2026-10-30, 15.36 MHz, two informal runs: keywords were said near the board, without a list of what was said):

| Measure | Result |
|---|---|
| Keyword events, run 1 (first 80 s) | 22: left 5, yes 6, stop 5, up 6 |
| Keyword events, run 2 (290 s) | 27: yes 7, stop 8, left 6, up 4, right 2, go 1 |
| Classes between the words | `_unknown_` |
| Classes on top for a few decisions only, not reported | `off`, 8 decisions in all |
| Overrun | 0 |
| CPU-active per decision | 263 cycles (max 292) |
| Decisions missed by the firmware while printing at 9600 baud | 77 of 6,000 in run 1 (the hardware made all of them) |

This shows the live chain from microphone to keyword working with the CPU asleep. It does not give an accuracy; the scripted test below does.

### 8.1 Scripted keyword test

The firmware `hopcim_kwtest` is `hopcim_live` built with `KW_TEST` (the number of rounds, default 5), so it uses the same detection rule. It prompts one item every 4 s, after a 5-s lead. Each round holds 13 prompts in a shuffled order:
- the 10 keywords;
- two words of GSC's `_unknown_` class (bird, house, happy, seven, wow, dog, cat, tree, marvin, zero, each once over the 5 rounds);
- one quiet trial.

Five rounds are 65 prompts and take about 4.5 min. The seed is fixed, so every run prompts the same order. Prompts and reported keywords carry the decision number (`@n`, 20 ms per decision). The statistics come once, at the end, followed by `done`, so `board_run.sh` stops by itself.

```
cd planenest/hw
FOLLOW=1 ./board_run.sh hopcim_kwtest_nexys.elf bitstreams/xheep_nexys_bscan_cim128mic.bit 330
python3 ../../hopcim/model/score_kwtest.py $(ls -t hopcim_kwtest_nexys_board_*.uart.log | head -1)
```

**Protocol.**
- Quiet room; speak at a normal level, 20–50 cm in front of the board; the same speaker and distance for the whole run.
- Say each prompted word once, soon after it appears; say nothing at `(stay quiet)`.
- Note the speaker, the distance and the room with the log. Several speakers or distances are separate runs.

**Scoring** (`hopcim/model/score_kwtest.py`). Trial i runs from its prompt to the next prompt.
- A keyword trial is *correct* if the prompted keyword is reported in it, *wrong* if only other keywords are, and *missed* if nothing is.
- A non-keyword or quiet trial is correct if nothing is reported.
- Every other report is a *false alarm*: a wrong keyword, or the prompted keyword reported a second time.
- The script also prints the time from the prompt to the report. This includes reading the prompt and saying the word, so it is not the system's latency.
- For a free-running `hopcim_live` log, `--said yes,stop,...` aligns the reports with the words said (edit distance).

**Checks** (2026-10-30):
- The scorer gives the expected counts on a synthetic log covering every outcome, including a report on a trial boundary and a cut-short log.
- A 1-round build (`-DKW_TEST=1`) ran on the board with nobody speaking: 13 prompts exactly 200 decisions apart, 2 decisions missed while printing, no overrun. The scorer gave 10 keywords missed, the 2 non-keyword trials and the quiet trial without a report, and 0 false alarms.
- The test build's CPU-active cycles (230 per decision) are not the demo's (263): the code differs (no 10-s statistics) and is laid out differently.
- The `hopcim_live` build with the test code added disassembles identically to the ELF of the board runs above.

**What it measures.** Accuracy for one speaker in one room, through the real microphone. It is not comparable with the 91.80 % on the GSC test set: different speakers, a different microphone and room, and the detection rule. The non-keyword words come from GSC's `_unknown_` set, which the model was trained on, so they are the easy case for rejection.

## 9. A decision every k hops

**Why.** Nothing requires a decision every 20 ms: a keyword lasts hundreds of milliseconds, and the
1-s windows overlap almost completely. Running the network less often saves CIM passes in
proportion, with no cache. It is the baseline that activation reuse must beat
(`REUSE_EVALUATION.md`).

**What it does.** `AUD_CTRL[19:16]` = k − 1, for k = 1 … 16; the reset value 0 means every hop, as before.
- The network and its decision run at hops h = 49 + n k.
- The front-end and the window path still run at every hop. So every decision's features, and
  therefore its logits, equal those of the same window with k = 1.
- Hardware: a 4-bit counter in `cim_audio`, reset at the start of a stream.
- Write the field together with the enable. A stride change during a stream takes effect at the
  counter's next wrap, and no firmware does it.
- Earlier bitstreams ignore the field, so firmware built with k > 1 needs `xheep_nexys_bscan_cim128k.bit`.

**Firmware.** `hopcim_live`, `hopcim_kwtest` and `hopcim_micrec` take `-DNET_K=<k>`.
- The keyword hold keeps spanning at least 80 ms: 1 + ⌈4/k⌉ decisions (5, 3, 3, 2, 2 for k = 1 … 5, then 2).
  At k = 1, 5 decisions in a row span 4 intervals.
- Statistics, LED times and test prompts stay in seconds, and the test's `@n` stays in 20-ms units, so
  `score_kwtest.py` needs no change.
- The default builds disassemble identically to the ELFs of the earlier board runs.

**Results** (15.36 MHz; board: `xheep_nexys_bscan_cim128k.bit`, quiet room, nobody speaking, 60 s per
live run; `planenest/hw/board_khop.sh`, log `khop_board_20261105_231009.log`):

| | k = 1 | k = 2 | k = 4 |
|---|---:|---:|---:|
| A decision every | 20 ms | 40 ms | 80 ms |
| CIM passes per hop | 7,501 | 3,750 | 1,875 |
| Engine busy cycles per hop | 67,509 | 33,755 | 16,877 |
| Hardware busy cycles per hop, typical (from the measured parts) | ≈ 93.1k | ≈ 50.5k | ≈ 29.1k |
| **Board:** decisions per 10 s | 500 | 250 | 125 |
| **Board:** overrun | 0 | 0 | 0 |
| **Board:** CPU-active per decision | 263 (max 289) | 263 (max 288) | 266 (max 289) |
| CPU-active per hop | 263 | ≈ 132 | ≈ 67 |
| **Board:** real-microphone decisions with the window in the recording, logits equal to the golden model | 25 of 25 | 13 of 13 | 7 of 7 |

The k = 1 numbers on the new bitstream equal those of §6. A hop without the network takes about 7.8k
cycles (front-end 7,582, window path 207).

**What does not change.** A decision hop still runs the whole network, so the worst-case hop (97.7k
cycles) and the lowest real-time clock (5.12 MHz) stay the same. RTL covers this (below). No 5.12-MHz
bitstream with this option was built.

**Detection.** On synthetic always-on streams (950 words over noise), recall did not measurably change
at 40 ms (p = 0.109) or 80 ms (p = 0.454, 10 words found only at 20 ms, 6 only at 80 ms); see
`REUSE_EVALUATION.md` §5.

**With a real speaker** (2026-11-07: one speaker, one room, the same 65 prompts in the same order, one
run per rate, in the order shown; logs `hopcim_kwtest*_board_20261107_*.uart.log`):

| | 20 ms | 40 ms | 80 ms |
|---|---:|---:|---:|
| Keywords correct, of 50 (95 % interval) | **48** (86.5–98.9 %) | **47** (83.8–97.9 %) | **42** (71.5–91.7 %) |
| Wrong keyword only, or nothing | 2, 0 | 2, 1 | 2, 6 |
| Non-keywords with a report, of 10 | 3 | 2 | 1 |
| Quiet trials with a report, of 5 | 0 | 0 | 0 |
| Reports other than the first of the prompted word | 32 | 35 | 28 |
| Overrun | 0 | 0 | 0 |
| Decisions, and missed while printing | 13,240, 10 | 6,625, 0 | 3,313, 0 |
| CPU-active per decision (max) | 230 (258) | 233 (260) | 233 (259) |
| Keyword class at a decision | 20.8 % | 20.8 % | 18.5 % |

- **40 ms shows no loss.** Against 20 ms, 3 prompts were correct only at 20 ms and 2 only at 40 ms (exact
  McNemar p = 1.000).
- **80 ms is less clear, and its direction is against it:** 8 prompts correct only at 20 ms and 2 only at
  80 ms (p = 0.109). Six keywords got no report at all, while none did at 20 ms. On synthetic streams
  80 ms showed no loss, so the two results disagree.
- **Limits of this comparison.** Prompts are matched by position, but each run is new speech, in a fixed
  order, one speaker, and 50 keywords per rate is few. These p values are a guide, not a test on
  identical audio. The room's level also differed at each start (RMS 48, 31, 57).
- **Most extra reports are word endings** and do not depend on the rate: `stop` → `up`, `off` → `up`,
  `go` → `no`.
- **A possible cause of the 80-ms misses, not tested:** the hold needs the same class at two decisions
  80 ms apart. A run of L hops of one class at 20 ms is then caught only if it is well aligned, with
  probability (L − 4)/4 for L = 5 … 8, while 20 ms always catches it. Real speech may give shorter runs
  than the synthetic words. Printing the class of every decision at 20 ms for one run would settle this:
  each word could then be scored at every rate, alignment and hold on identical audio, since the decisions
  are bit-exact across rates.

**Identical speech.** `SAME_AUDIO.md` replaces this comparison with one that holds all rates, phases and rules on the same audio; its spoken runs are still to be made.

**Operating point: 40 ms**, an estimated 32 % below 20 ms (`ENERGY_ESTIMATE.md`). The commands:

```
cd planenest/hw
for e in hopcim_kwtest_nexys hopcim_kwtest_k2_nexys hopcim_kwtest_k4_nexys; do
  FOLLOW=1 ./board_run.sh $e.elf bitstreams/xheep_nexys_bscan_cim128k.bit 330
  python3 ../../hopcim/model/score_kwtest.py $(ls -t ${e}_board_*.uart.log | head -1)
done
```

One-round builds of `hopcim_kwtest` ran on the board with nobody speaking, at k = 2 and 4: 13 prompts
exactly 200 units (4 s) apart, 1,425 and 713 decisions, none missed, no overrun; the scorer gave 10
keywords missed, nothing reported in the 3 other trials, and 0 false alarms.

**Verification.**

| Check | Result |
|---|---|
| `tb_cim_mic`, k = 1, 2, 3, 4, 5, 16 (`cim_fe`), 1, 2, 4 (B3), PDM divider 15; k = 1, 2, 4 (`cim_fe`), 4 (B3), divider 5 | every decision's logits, class, count and hop exact; no other decision interrupt; no overrun; ring exact |
| k = 1 against the RTL before this option, with the same bench | ends at exactly the same cycle at divider 5: 15,492,293 (`cim_fe`), 15,493,385 (B3). The current bench reads `AUD_CTRL` back after enabling it, which adds 2 cycles to its count |
| The same bench, k > 1: a restart | the first decision again at hop 49 |
| Stride counter broken on purpose (off by one, no reset at a restart, stride ignored, counting before hop 49) | all 4 caught; the unchanged RTL passes |
| Benches with the audio path off: endpoint, front-end, sequencer, stream, both front-ends | all pass |
| Gate level, `tb_cim_audio_nl` (`sim/run_audio_nl.sh`): `cim_audio` RTL against its Vivado netlist, stride changing during the stream, a restart | 0 mismatches over 4.49 billion output bits (103 hops, 31.6 M cycles at divider 15): k = 1, 2, 4, 16, 3, 1, the network off, a restart |

**Cost** (Vivado out of context, xc7a100t):

| Unit | LUT | FF | BRAM tiles | DSP | WNS at 15 MHz |
|---|---:|---:|---:|---:|---:|
| `cim_audio` | 1,026 (+5) | 850 (+4) | 0 | 1 | +48.3 ns |
| Endpoint with `cim_fe` | 7,635 (+18) | 5,819 (+8) | 46 | 145 | +46.9 ns |
| Endpoint with B3 | 7,476 (+5) | 5,914 (+8) | 45 | 146 | +46.4 ns |

Bitstream `xheep_nexys_bscan_cim128k.bit` (FuseSoC build root `build_mick/core`, flags as in §6): 42,306
LUT, 40,221 FF, 82 block RAM tiles, 146 DSP; system clock setup +15.10 ns, hold +0.229 ns.

### The low-power mode U: the summed-logit rule (2026-11-22)

`hopcim_live` and `hopcim_kwtest` also take `-DKW_SUM=<n>`.

**The rule.** The firmware reads each decision's 12 logits and sums those of the last n decisions. It takes the top class
of that sum, ties going to the lower class, and reports a keyword as soon as it becomes that class (a hold of 1). The
code is `sw/kw_rule.h`. LD14 still shows each decision's own class.

**Mode U** is `-DNET_K=16 -DKW_SUM=2`: the network every 320 ms, the sum of the last 2 decisions. It was chosen and
tested in `WHOLEWORD_STUDY.md`, including 100 spoken keywords.

**Checks:**
- **The rule equals the scored one.** `model/test_kw_rule.py` compiles `sw/kw_rule.h` on the host and compares it with
  `kwlog_analyse.apply_rule(("smooth", n, hold))`. It covers 479 cases (the spoken runs, the study's 21 test streams,
  200,000 synthetic decisions with ties) with 0 differing, and catches all 3 of its broken copies.
- **The default is unchanged.** The default builds (`KW_SUM` = 0; k = 1 and 2; live and test) give loadable images
  identical to those before the change.

**ELFs** in `planenest/hw`, for `xheep_nexys_bscan_cim128ks.bit` at 15.36 MHz:
- `hopcim_live_u_nexys.elf` (sha256 `7c142486…`);
- `hopcim_kwtest_u_nexys.elf` (sha256 `ac9cef7e…`).

**At k = 16 the LEDs' integer timing rounds:**
- LD0 toggles every decision, so it blinks at about 1.6 Hz;
- a keyword stays on LD10–LD13 for 3 decisions (0.96 s).

**Board** (2026-11-22; `xheep_nexys_bscan_cim128ks.bit`, 15.36 MHz; `hopcim_live_u_nexys_board_20261122_162128.uart.log`).
This was the first board run at k = 16:
- **Rate:** 31 decisions per 10 s (500 / 16 = 31.25) over 120 s, 0 missed, overrun 0.
- **Keywords:** reported as spoken.
- **CPU-active per decision:** 875 cycles (max 883), against 263 with the hold rule. The extra 610 cycles are the 12 logit
  reads and the two 12-step loops of `kw_rule.h`, about 360 instructions. Per second that is 2,734 CPU-active cycles,
  against 6,575 at 40 ms. Priced with the measured cycles, U is 44.16 µW [E] instead of 44.11.

**The scripted test** `hopcim_kwtest_u_nexys.elf` stopped at prompt 29, because `board_run.sh` was started without a
timeout (default 120 s; the test needs about 265 s). On those 28 trials:
- 18 of 21 keywords were correct;
- 1 of 5 non-keywords and 0 of 2 quiet trials had a report;
- the median from prompt to report was 1.60 s, against 1.24 s at 40 ms in the run of 2026-11-07. That is other speech,
  so it is not paired.

A complete run needs `board_run.sh … 330`.

**The complete scripted test** (2026-11-22, `hopcim_kwtest_u_nexys_board_20261122_163007.uart.log`, timeout 330 s):
829 decisions, 0 missed, overrun 0, CPU-active 825 cycles per decision (max 834). All four runs below are scored by
`score_kwtest.py`, with one speaker in the same room:

| | 20 ms, 5 in a row | 40 ms, 3 in a row | 80 ms, 2 in a row | **U**: 320 ms, sum of 2 |
|---|---:|---:|---:|---:|
| Keywords correct, of 50 | 48 | 47 | 42 | **48** |
| Wrong keyword only, or nothing | 2, 0 | 2, 1 | 2, 6 | 1, 1 |
| Non-keywords with a report, of 10 | 3 | 2 | 1 | 1 |
| Quiet trials with a report, of 5 | 0 | 0 | 0 | 0 |
| Reports other than the first of the prompted word | 32 | 35 | 28 | **9** |
| Prompt to report, median (correct trials) | 1.23 s | 1.24 s | 1.20 s | 1.28 s |
| Network runs per second | 50 | 25 | 12.5 | 3.1 |
| Power [E] (`WHOLEWORD_STUDY.md`) | 97.0 µW | 68.8 µW | 54.7 µW | 44.2 µW |

Each run is new speech, so these four are not paired. The paired comparison, the same speech scored for every design,
is in `WHOLEWORD_STUDY.md`: U 93.1 against F40's 93.5 of 100 keywords.

The 9 extra reports of U are of two kinds:
- "no" just before "go" (3 times);
- "up" just after "stop" or "off" (4 times);
- plus "down" and "stop" on "dog".

Both are confusions of the network that a single summed decision reports at once. The hold rules repeat a keyword
more often instead.

## 10. The baseline S at 4.096 MHz (2026-11-08)

With the baseline S in the sequencer (`BASELINE_S.md`), the worst hop falls from 97,583 to 80,072 cycles, under the
81,920 cycles of a hop at 4.096 MHz (PDM divider 4). Measured on the board (`xheep_nexys_bscan_cim128mic4s.bit`,
`--flag use_cim_s --flag use_cim_mic4`): 6,000 decisions in 120 s with no overrun, 263 CPU-active cycles per
decision, and 25 of 25 decisions on real microphone audio equal to the golden model. The PDM clock is high for 2 of
4 cycles (50 %, inside the ADMP421's 40–60 %). R6b at divider 4 overruns from its first decision (RTL).
