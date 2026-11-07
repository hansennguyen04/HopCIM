# R5 plan — hardware front-end: the MFCC hop runs without the CPU

Status: R5a (frame path in hardware) built and verified, 2026-10-27; see §5. Builds on R1–R4
(`ENGINE_SPEC.md`, `INTEGRATION.md`, `MFCC_INT_SPEC.md`).

## 1. Why

R4b runs the bit-exact integer front-end with every matrix step on the CIM, orchestrated by the
CPU. It costs 146,278 cycles per frame, only 1.1× better than the CPU-only front-end (160,774).
The CPU epilogue (window, twiddles, magnitude) is 74 % of that time and MMIO is most of the rest,
while the engine computes for 932 cycles. The front-end must therefore move as a whole into
hardware. Putting only the MACs on the CIM does not help.

## 2. Architecture (inside `cim_endpoint`)

A fixed-function front-end unit (`fe`) shares the engine with the CPU port. While `fe` runs,
CPU go writes are held as usual. Per hop (320 new samples) it executes:

| Phase | Work | Unit | Cycles (1 value / cycle) |
|---|---|---|---:|
| PEAK | 480 window products, max \|v\| → `sh` | 1 DSP (16 × 15) | ~480 |
| STEP1 | 32 passes: X-gather `rshift(x·hann, sh)` at stride 32; results → mirror (negate before rounding) → twiddle → B buffer | window DSP + 4 DSP complex multiplier | ~32 × 26 ≈ 830 |
| STEP3 | 16 passes: X-gather 64 values from B; results → `>> 9` → \|X\| → MAG buffer | pipelined isqrt (2 squares, 4 DSP) | ~16 × 74 ≈ 1,200 |
| MEL | 4 passes (one accumulates) over MAG | — | ~280 |
| LOG | 40 × (leading one + 64-entry table) → `lraw` ring (49 frames) | small | ~40 |
| FINISH | `lmq` (1 DSP), DCT 2 passes, requantisation (2 DSP) → int8 feature ring; all 49 frames again only when the window max changes | — | ~180 (new frame) / ~9k (all) |

Total about 3k cycles per hop (0.2 ms at 15 MHz) instead of 146k, with zero CPU cycles. The CPU
is interrupted once per hop (or once per network inference) with the 49 × 10 int8 input ready.
Gathering X overlaps with the previous pass after its 10-cycle X lock, so no X double buffer
is needed.

Storage: sample ring 1024 × 16, B buffer 512 × 34 (one RAMB18), MAG 128 × 21, `lraw` ring
64 × 40 × 24, feature ring 49 × 10 × 8. About 8–10 RAMB18 and 11 DSP in addition to R2, against
135 RAMB36 and 240 DSP on the device (X-HEEP + CIM use 52 and 129).

The window max (step 9 of the spec) is tracked per hop: a 50-entry ring of hop maxima gives the
1 s window max, then `log2fix16` (16 squaring steps). If the max is unchanged, only the new
frame is finished; otherwise all 49 are. This keeps streaming bit-exact with the per-window model.

## 3. Verification

1. Unit TBs per datapath block against `mfcc_int.py` vectors (window/peak, mirror + twiddle,
   isqrt, log, requant).
2. `fe` TB: whole clips through the RTL front-end, `lraw` and features bit-exact against
   `mfcc_int.py` on a sample of test clips, including the frame-level corner cases: silent frames, the
   largest peaks, a window max that changes between hops.
3. SoC: the R4b firmware test with the hardware front-end (bit-exact, CPU-active cycles from a
   sleep counter). Then board.
4. Accuracy is already gated (R4a); bit-exactness carries it over.

## 4. Open choices (to decide before RTL)

- **Sample input.** (a) The CPU or DMA writes 320 samples per hop into the ring (simple; the
  CPU wakes per hop for about 1k cycles). (b) A direct stream PDM → CIC → FIR → ring (the CPU sleeps
  for the whole hop; needs the decimation FIR in hardware). Proposed: (a) first, (b) for the
  live demo.
- **16 kHz on the board.** The board clock is 15 MHz, so the PDM path delivers 15.625 kHz. An
  MMCM output of 15.36 MHz (VCO 960 MHz, M = 48, D = 5, O = 62.5) gives PDM 1.28 MHz → CIC ÷16
  → FIR ÷5 = exactly 16 kHz. The firmware clock constant must follow.
- **Network.** The DS-CNN on the same engine (R5b) needs layer sequencing (conv/pw as passes,
  depthwise as block-diagonal tiles) and requantisation per layer. Proposed after the
  front-end, with CPU-orchestrated passes as its baseline.

## 5. R5a status: the frame path in hardware

`rtl/cim_fe.sv` sits inside `cim_endpoint`. It implements PEAK → FFT1 → FFT3 → MAG → MEL → LOG
for one frame, with the schedule simplified to one pass at a time and a sequential isqrt.
The CPU writes samples into a 1024-entry ring (`SAMPLES`), starts a frame (`FE_CTRL`) and reads
`LRAW[40]`; the window path (normalisation, DCT, requantisation) stays on the CPU for now. Tables are
generated from the frozen model (`model/gen_c.py` → `rtl/fe_tables.svh`).

**Unit level** (`tb/tb_cim_fe.sv`, `sim/run_fe_tb.sh`, through the OBI port). 319 frames: 6 GSC
test clips plus 25 synthetic frames covering silence, full-scale DC/Nyquist/noise, impulses,
block exponents −14..15 and 23 frames with exact-zero entries. The frames sit at rotating ring
bases, so wrap-around is covered. **0 lraw mismatches.**
The interplay checks pass:
- a front-end go waits for the CPU pass in flight;
- DONE_CNT and the CPU results are untouched;
- a CPU X write is held until the front-end finishes.

A mutant that negates the mirrored step-1 outputs after rounding is caught (8 errors in 6 frames).
Front-end latency: 7,476 cycles per frame (528 for a silent frame).

**SoC level** (Verilator, `sw/applications/hopcim_fe`): all 49 frames of the test clip are
bit-exact, lraw and int8 features.

| Per frame | R4b (CPU epilogue) | R5a (hardware frame path) |
|---|---:|---:|
| CPU-active cycles | 146,364 | 6,710 (samples 4,205, lraw read 400, finish 2,105) |
| front-end latency (CPU may sleep) | — | 7,485 |
| whole clip, 49 frames | 7.56 M cycles | 0.98 M cycles |

**Board** (Nexys A7-100T, `bitstreams/xheep_nexys_bscan_pdm_cim_fe.bit`, sha256 `39c90373…e8b3`).
The SoC uses 38,850 LUT (61 %), 38,859 FF, 54.5 BRAM tiles and 136 DSP. The 15 MHz domain meets
+16.6 ns setup and +0.222 ns hold, and there are no hold failures anywhere. `hopcim_fe`,
`hopcim_mfcc` and `cim_smoke` all pass with every reported cycle count identical to Verilator: the
hardware front-end runs bit-exactly on the FPGA.

So R5a needs 21.8× fewer CPU-active cycles than R4b and 24× fewer than the CPU-only front-end
(160,774). Writing the samples is now the largest CPU cost (about 13 cycles per sample over
MMIO). R5b targets it (stream from the PDM path, or packed writes) together with the
hardware window path.

**Cost** (Vivado out of context, xc7a100t, 15 MHz):

| Top | LUT | FF | RAMB18 | DSP | WNS |
|---|---:|---:|---:|---:|---:|
| `cim_fe` | 1,907 | 1,953 | 4 | 7 | +45.0 ns |
| `cim_endpoint` (R2 + ring + front-end) | 4,176 | 4,474 | 37 | 135 | +40.8 ns |
| `cim_endpoint` R2 only, for reference | 2,314 | 2,519 | 32 | 128 | +51.3 ns |

The first version had 7,208 LUT and 13,723 FF. Its buffers were written inside the async-reset FSM
block, so Vivado built them from flip-flops with wide read multiplexers instead of RAM. The writes
now sit in a reset-free block.

