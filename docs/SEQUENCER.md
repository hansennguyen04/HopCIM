# R6b — a hardware layer sequencer: the DS-CNN without the CPU

Status (2026-10-27): built and verified at unit level (bit-exact on every layer and every logit),
as a Vivado gate netlist, on the Verilator SoC and **on the board** (Nexys A7-100T, both
bitstreams). Every per-decision board number equals Verilator's (§8). The first bitstreams exposed
a Vivado synthesis mismatch, which was found and fixed (§8a).

**2026-11-08: the baseline S** (sign-gated requantisation, a second requantisation lane, no L8 write) is built behind
the parameter `SEQ_S` and replaces R6b as the baseline: 67,841 cycles per inference instead of 85,352, exact on all
4,890 test clips, on the board at 15.36 MHz (`BASELINE_S.md`). With `SEQ_S = 0` everything below is unchanged.

**2026-11-25: the default design is S+DWU**, S with the digital depthwise unit (`SEQ_DWU = 4`, `DWU_RESULTS.md`): 67,789
cycles, exact on all 4,890 clips. S stays as the comparison baseline (`DESIGNS.md`).

## 1. Result

`rtl/cim_seq.sv` runs the whole MLPerf Tiny int8 DS-CNN on the CIM by itself. Per decision, the
CPU writes the 49 × 10 int8 features, starts the sequencer and sleeps until its interrupt.

The sequencer does three jobs:
- **gather:** it builds each pass's input vector from its own activation buffers, zero points and
  SAME padding included;
- **sequencing:** it runs all 7,501 passes;
- **requantisation:** it scales every result with TFLite's integer arithmetic and writes it back.

| | CPU-orchestrated (R6a, `hopcim_net1`) | **Layer sequencer (R6b)** |
|---|---:|---:|
| Cycles per inference, board = Verilator | 13,020,998 | **85,420** go → interrupt (152× fewer) |
| Time at 15 MHz | 0.87 s | **5.69 ms** |
| CPU during the inference | busy throughout | **asleep (WFI)** |
| Logits vs TFLite reference kernels | bit-exact | bit-exact |
| Data dependence of the time | ±0.07 % | none (fixed schedule) |

The CPU-orchestrated figure is for the tap-major depthwise layout that the sequencer needs. The
first, channel-major layout measured 12,874,346 cycles. The sequencer itself takes 85,352 cycles
from the grant of its go to done (testbench); the SoC figure adds the go store and the interrupt
entry.

The target was under 10 ms per decision with the CPU asleep. The network now takes 5.69 ms. What
remains on the CPU per decision is listed in §9.

## 2. Architecture

The sequencer is the endpoint's third engine master, next to the CPU and the front-end. Four
stages run concurrently, decoupled by small handshakes:

1. **Gather.** One *chunk* per cycle: up to 8 consecutive X rows of one input vector, written into
   the idle X bank through the engine's new wide port (`ENGINE_SPEC.md`). A chunk is:
   - conv1: the 4 taps fx of filter row fy, from FEAT, values `F − 83`;
   - depthwise: the 7 channels of one tap, one ACT word, values `a + 128`;
   - pointwise: 7 input channels, one ACT word;
   - FC: one pooled value.

   Out-of-bounds taps are written as 0. Stages: read the source, form `x = v − zin` (9 bits),
   rotate into the engine lanes.
2. **Issue.** When a bank's vector is complete, its passes start: 4 for conv1 and pointwise, 1 for
   depthwise and FC. Each pass starts as soon as the engine is ready and the previous pass's
   results can drain, i.e. every `max(9, outputs)` cycles. A pass's tag (position, first channel,
   number of results) goes into a 4-entry FIFO.
3. **Requantisation.** A done pops the oldest tag and latches the 16 results; they drain one per
   cycle through a 5-stage pipeline (§3).
4. **Write-back.** Results go to the other ACT buffer (L0–L8), to the pool sums (L8), or to the
   logits and the running argmax (L9).

Layers are separated by a barrier: a layer starts when every result of the previous one is
written. This costs about 30 cycles per layer.

**Double-buffered input.** The engine now has two X banks. The gather writes vector v + 1 while the
passes of vector v read the other bank. A bank is released 7 cycles after its last pass starts,
so the first new write lands 10 edges after that start, past the pass's last read at `t0 + 9`.

**Buffers** (inside `cim_seq`):

| Buffer | Content | Size | FPGA mapping |
|---|---|---|---|
| FEAT | int8 features, a ring of 49 rows; frame 0 at row `frow` | 49 × 12 B | 4 byte lanes, LUT-RAM |
| ACT | two 25 × 5 × 64 int8 maps, ping-pong; word {buffer, position, c / 7}, lane c mod 7 | 2 × 8,000 B | 7 × RAMB36 |
| PARAMS | per output channel: bias (24 b), multiplier (31 b), e = −shift (4 b); 640 entries | 7.4 KB | 2.5 BRAM |
| POOL | 64 channel sums (16 b) | 128 B | LUT-RAM |
| LOGITS | 12 × int8, class | | FF |

**Layer order and tiles** (relative to `tbase` = 8, as in `model/cim_net_map.py`):

| L | Layer | Vectors | Chunks / vector | Passes / vector | Tiles |
|---|---|---:|---:|---:|---|
| 0 | conv1 10 × 4 / 2 | 125 | 10 | 4 | +0 … +3 |
| 1 + 2l | depthwise 3 × 3 | 1,250 | 9 | 1 | +4 + 14l + g |
| 2 + 2l | pointwise | 125 | 10 | 4 | +14 + 14l + j |
| 9 | FC (after the pool) | 1 | 64 | 1 | +60 |

The depthwise tiles were changed to tap-major rows, `7 (3 fy + fx) + c'`. One tap is then 7
consecutive rows, so a depthwise vector is 9 chunks, one ACT word each.

## 3. Arithmetic (bit-exact to TFLite's reference kernels)

For channel c of layer L, with parameter entry 64 L + c:

```
x = acc + bias                                     25 bits (|acc| < 2^21 by the weight bound)
r = (x · qm + 2^30) >> 31                          = SaturatingRoundingDoublingHighMul(x, qm)
y = clamp(((r + 2^(e−1) − [r < 0]) >> e) + zout)   = RoundingDivideByPOT(r, e) + zout
```

Both closed forms were checked against `dscnn_int.srdhm` / `rdbp`:
- srdhm: 2.14 M values of x (random |x| < 2^24, plus every |x| < 70,000), with random
  multipliers in [2^30, 2^31) and the model's 577 real ones (9 × 64 per channel, 1 for the FC);
- rdbp: the same x values for every e = 0 … 15.

TFLite's nudge `1 − 2^30` for negative products, followed by division toward zero, is the same
as one floor division after adding 2^30. The saturation case (both operands −2^31) cannot occur
with qm in [2^30, 2^31).

The pool replaces the division by 125 with `(a · 8389) >> 20`, exact for all a ≤ 16,062; the sums
need a ≤ 16,062. Only right shifts are implemented (all shifts of this model are −9 … −5); the
firmware refuses a model that needs a left shift.

Hardware: a 25 × 32 multiply in 2 DSPs, with the bias add folded into the DSP pre-adder.

## 4. Verification

**Unit level** (`tb/tb_cim_seq.sv`, `sim/run_seq_tb.sh [ded]`, vectors `model/gen_seq_vectors.py`).
The testbench drives the endpoint over OBI the way firmware does:

| Check | Result |
|---|---|
| Every output of L0 … L8 of clip 0 (stop-after-layer runs, read through the ACT window): 9 × 8,000 values against `dscnn_int` | **0 mismatches** |
| Logits and class, 20 GSC test clips covering all 12 classes | **0 mismatches**, 20/20 classes = golden |
| FEAT as a ring (frame 0 at row 17) | bit-exact |
| Synthetic requantisation set: shifts 0 … 15, multipliers 2^30 and 2^30 + 2^k, activation zero point 5 (no ReLU clamp), conv1 zero point −37; every layer and the logits | **0 mismatches** |
| Interplay: CPU X write, CPU go and FEAT write during a run are held to its end; a `cim_fe` go is held, B3 runs concurrently; a SEQ go waits for a CPU pass in flight and for `cim_fe`; CPU passes are correct after runs | all as specified, both builds |
| Engine contract monitor: no wide write to a bank in the 8 edges after a pass on it started | 0 violations |

The synthetic set exists because the real model exercises only a narrow slice of the arithmetic:
- shifts 5 … 9;
- zero point −128, so every negative value clamps to −128 and its rounding direction never
  shows.

**Mutation tests**:

| Mutation | Caught by |
|---|---|
| srdhm nudge 2^30 − 1 | synthetic set only (the real multipliers have too few trailing zero bits for an exact half) |
| rdbp rounds half up | 1 real clip (the FC layer, the only one without a ReLU clamp); synthetic set: every layer |
| pool constant 63 instead of 62 | 9 of 20 clips |
| FEAT ring wraps at 50 | the ring test |
| X bank released 2 cycles early | contract monitor (23,370 violations). The results stay correct: the gather writes rows in increasing order, so in this design it never overtakes the reads; the margin protects the contract, not this schedule |
| e = 0 guard removed | synthetic set |

**Engine** (`tb_cim_engine`, extended): 600 passes alternating between the banks while the idle
bank is rewritten in random chunks and masks; 3,004 passes bit-exact, timing exact. The endpoint
and both front-end testbenches still pass unchanged.

## 5. Cycles

Per layer, from the stop-after-layer runs (identical for every clip):

| Layer | Cycles | Per pass | Bound |
|---|---:|---:|---|
| conv1 | 8,034 | 16 | requantisation, 16 results per pass at 1 per cycle |
| depthwise (each of 4) | 11,273 | 9 | engine (9-bit inputs), no bubbles |
| pointwise (each of 4) | 8,032 | 16 | requantisation |
| pool + FC | 98 | | |
| **Total** | **85,352** | | |

The engine is busy 7,501 × 9 = 67,509 cycles, 79 % of the run. The depthwise gather sustains 7
values per cycle, exactly the engine's pass rate.

A second requantisation lane (2 more DSPs) would let conv1 and pointwise run at 9 cycles per pass:
about 67.9k cycles, 4.5 ms. It is not needed for the target, so it was not built.

## 6. Cost (Vivado out of context, xc7a100t, 15 MHz)

| Unit | LUT (of which memory) | FF | BRAM tiles | DSP | WNS |
|---|---:|---:|---:|---:|---:|
| `cim_seq` | 1,176 (267) | 894 | 9.5 | 3 | +51.9 ns |
| `cim_engine`, 128 tiles, now with 2 X banks | 2,018 | 1,648 | 32 | 128 | +51.3 ns |
| Endpoint with `cim_fe` and the sequencer | 5,155 (604) | 3,860 | 44 | 138 | +48.3 ns |
| Endpoint with B3 and the sequencer | 4,959 (886) | 3,948 | 43 | 139 | +45.5 ns |

The endpoint before R6b (`B3_BASELINE.md`, `cim_fe`) was 3,882 LUT, 2,908 FF, 34.5 BRAM tiles and
135 DSP. R6b (the sequencer plus the engine's second X bank and wide port) adds about 1,270 LUT,
950 FF, 9.5 BRAM tiles and 3 DSP. All figures are for the final RTL, after the address rewrite
of §8a.

Full SoC on the Nexys A7-100T (128-tile CIM with the sequencer, BSCANE2 debug, PDM):

| Bitstream | LUT | FF | BRAM tiles | DSP | 15 MHz setup / hold |
|---|---:|---:|---:|---:|---|
| `xheep_nexys_bscan_pdm_cim128s_cim.bit` (`cim_fe`) | 39,903 (63 %) | 38,245 | 80 | 139 | +16.24 / +0.222 ns |
| `xheep_nexys_bscan_pdm_cim128s_ded.bit` (B3) | 39,713 | 38,333 | 79 | 140 | +16.20 / +0.222 ns |

As in every earlier build, the only failing paths are the asynchronous crossings of the unused
SPI-slave clock.

A first build had the 1024 × 31 multiplier memory in LUT-RAM (176 RAM64M, about 700 LUTs). Vivado had absorbed its
read register and output stage into the DSP's B pipeline. It is now read one cycle later,
straight into the multiplier, and maps to block RAM.

## 7. Endpoint interface

Registers `SEQ_CTRL` (go, stop-after-layer, FEAT ring row, first tile), `SEQ_STATUS` (busy, done,
error, class, layer), `SEQ_ZP`; windows FEAT, LOGITS, PARAMS, ACT (debug); IRQ_EN bit 2. Details
are in `INTEGRATION.md`; the driver is in `sw/cim.h` (`cim_seq_go`).

## 8. SoC and board

**Firmware** `sw/applications/hopcim_seq`:
- once: builds the 61 tiles from the compact weights, loads the 640 parameter entries and the
  zero points (about 443.5k cycles);
- per decision: writes FEAT, starts the sequencer, sleeps in WFI until its interrupt (PLIC external
  line 3; `mcountinhibit` stops `mcycle` while asleep), then reads class and logits.

Each of the 20 netvec clips runs three times:
1. FEAT in order;
2. FEAT as a ring with frame 0 at row 7k + 3;
3. a streaming step: only the newest frame is rewritten (3 words, same values) into that ring.
   This is the CPU's work for one decision when one frame arrives per hop.

**Board** (Nexys A7-100T, `planenest/hw/board_suite.sh`) and **Verilator SoC**, final RTL, both
builds (`xheep_nexys_bscan_pdm_cim128s_cim.bit` with `cim_fe`, `..._ded.bit` with B3):

| Per decision | Cycles, board = Verilator | At 15 MHz |
|---|---:|---:|
| Sequencer, go → interrupt | 85,420 | 5.69 ms |
| CPU-active, whole window (147 FEAT words, go, wake, class and 12 logits) | 9,677 – 9,679 | 0.65 ms |
| **CPU-active, streaming step (3 FEAT words, go, wake, class and logits)** | **459** | 31 µs |
| Wall, whole window | 95,040 | 6.34 ms |

On both bitstreams and on both models:
- all 60 runs give logits bit-exact to the golden model;
- the classes equal the golden ones and the labels on 20/20 clips;
- the board's log equals Verilator's line for line, except the one-time setup (443,536 vs 443,532
  cycles). `hopcim_net1` shows the same 4-cycle offset in its setup, so it is a fixed start-up
  effect, not part of any measured path.

The schedule is fixed, so every run takes the same number of cycles. CPU-active cycles depend on
how GCC compiles the FEAT loop: other builds of the same program measured 9,382 and 10,118 for
the whole window.

All other HopCIM apps pass on the new bitstreams with their earlier cycle counts unchanged:
`cim_smoke`, `hopcim_mfcc`, `hopcim_fe`, `hopcim_fe_irq`, `hopcim_fe_ded`, `hopcim_fe_irq_ded`, and
`hopcim_net` (20 clips bit-exact).

## 8a. FPGA bring-up: a synthesis mismatch, found and fixed

**Symptom.** The first two bitstreams failed every board run, yet every simulation passed:
- the 60 runs gave wrong logits;
- the sequencer itself completed on schedule (85,420 cycles);
- the front-end apps on the same bitstreams passed.

**Localisation.** `sw/applications/hopcim_seq_dbg` runs clip 0 with stop-after-layer, reads every
conv1 and depthwise-0 output back, and recomputes two conv1 positions through the CPU path:
- conv1 was wrong at exactly the 50 positions with odd ox (25 rows × 2 columns), by a few units;
- the two CPU-path positions and their sequencer outputs were correct.

Testing hypotheses against the board values, exactly one reproduced all of them: at odd ox the
4th tap (column ix0 + 3) received the value 4 columns further on. FEAT lane 0 read word wq + 2
instead of wq + 1.

**Root cause.** The address logic computed that word with `int` arithmetic,
`ixb = ix0 + ((b − ix0) & 3); word = 2'(ixb >>> 2)`.
- Vivado 2024.1's netlist of that block, simulated against its RTL over all 101,250 inputs,
  differs on 40,500, every odd-ox case. Verilator and Vivado's own RTL simulator agree with the
  intended arithmetic.
- A five-line reproduction is in `syn/vivado_miscompile/`.
- The same value written without the negate-and-mask is synthesised correctly.

**Fix.** All gather addresses are now unsigned bit-level logic (with wq = ox / 2: lane 0 reads
wq + 1 for odd ox, lane 3 reads wq − 1 for even ox, every other lane wq). Checks:

| Check | Result |
|---|---|
| New against old RTL, on every case that matters (valid row and tap) | 0 of 101,250 differ |
| New block's Vivado netlist against its RTL | 0 of 101,250 differ |
| Requantiser, pool divider and input offset as Vivado netlists (3 DSPs) against RTL: 300k random vectors over shifts 0..15, multipliers 2^30 … 2^31, all zero points, plus all 32,001 pool sums | 0 differ |
| Vivado netlist of the whole `cim_seq` inside the RTL endpoint and engine, endpoint testbench: every layer and one full inference | 0 of 9 × 8,000 differ, logits exact, cycle counts identical to the RTL |

The engine's X LUT-RAM gets a `CIM_ZERO_INIT` initialiser for 4-state simulators: on the FPGA
it starts at 0, and in xsim the X rows conv1 never writes would otherwise make X × 0 = X. It is
not defined for synthesis or Verilator.

**On the board after the fix:**
- the debug readback is bit-exact (conv1 and depthwise 0: 0 of 8,000 wrong each);
- the full inference gives the golden logits;
- the whole suite passes (§8).

**Lesson for the flow.** Bit-exact RTL simulation is not enough when the RTL uses signed `int`
arithmetic. A layer-wise readback on the board localised the fault in one run; an equivalence
simulation of the synthesised block against its RTL proved the cause before any rebuild.

## 9. What is left on the CPU per decision

1. **FEAT.** With a whole window, the CPU writes 147 FEAT words (9,678 CPU-active cycles per
   decision). When streaming with one new frame per hop, it writes 3 words and advances the ring
   row (459 cycles).
2. **The per-window feature step.** The MLPerf Tiny preprocessing divides each 1-s clip by its peak
   (`get_dataset.py`), and the exact integer MFCC reproduces it: `lraw − log2(max)` before the floor,
   DCT and requantisation. In a sliding window, every frame's features change whenever the window's
   peak changes. Streaming the 20 test clips back to back (951 hops), the peak changes at
   **11.6 % of hops**. The CPU's window path then averages 6.6 frames per decision, about 14k
   cycles; the worst case is 49 frames, about 104k cycles (6.9 ms). Real time is never at risk
   (a hop is 300k cycles), but this is now the largest CPU-active term.

   Next step (R6c): run the window path in the sequencer as a layer before conv1 (49 frames ×
   2 DCT passes plus requantisation, about 3k cycles), from lraw kept by the front-end. The CPU
   would then only start the front-end per frame and the network per hop.

## 10. Limits

- The layer order and shapes are those of this DS-CNN (fixed in the sequencer). Weights,
  requantisation parameters, zero points and the tile base are programmable.
- Right shifts only; pool divisor fixed at 125 positions.
- `frow` must be below 49. The class is that of the last full run (a stop-after run leaves it).
