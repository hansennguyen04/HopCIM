# R5b — the window path in hardware: streaming decisions with the CPU asleep

Status (2026-10-28): built and verified:
- in RTL, end to end through the endpoint with both front-ends;
- as a Vivado gate netlist;
- on the Verilator SoC and on the board (Nexys A7-100T, both bitstreams), where every
  per-decision number equals Verilator's.

Per decision, the CPU now does 600 cycles of work beyond copying the new samples.

## 1. What problem it solves

After R6b, the network ran in hardware, but each decision still needed the CPU's per-window
feature step (`mfcc_int.finish`):
1. normalise every frame's `lraw` by the window's peak;
2. apply the floor;
3. compute the DCT (two CIM passes);
4. requantise to int8.

The MLPerf Tiny preprocessing divides every 1-s clip by its maximum sample (`get_dataset.py`), so
in a sliding window all 49 frames change whenever the window's peak changes. On 20 s of streamed
speech that happens at 11.7 % of hops. The CPU then spent on average 6.6 frames × 2,120 cycles per
decision, and 104k cycles (6.9 ms) in the worst case.

R5b moves this step into the layer sequencer, as a phase that runs before the network.

## 2. The algorithm (`model/stream_int.py`)

A decision at hop h (h ≥ 49) uses the last 16,000 samples, i.e. hop blocks h−49 … h of 320
samples each. Its 49 frames are stream frames h−49 … h−1; frame m covers samples 320m … 320m+479.
Every frame's `lraw` depends only on its own samples (per-frame block floating point), so it is
computed once and shared by 49 windows. Only the normalisation differs between windows.

Per hop:
1. push the maximum of the hop's 320 samples into a ring of 50 block maxima;
2. store the new frame's `lraw` (m = h−1) in ring row (h−1) mod 49;
3. lmx16 = log2fix16(window maximum). If it changed since the previous hop, recompute the
   features of all 49 rows; otherwise only the new row.

**Invariant.** After every hop, all rows of arrived frames are normalised with the current lmx16.
So whenever the window is complete, the features equal the per-window reference exactly.

**Checked:** 20 s of GSC speech, 1,000 hops, 951 decisions. For both front-ends (`cim_fe` and B3),
0 windows differ from `cim_mfcc_int` / `mfcc_ded.mfcc` applied to each window independently. All
49 frames were recomputed at 117 hops (11.7 %), one frame at 883.

A window maximum ≤ 0 is undefined in the reference (division by the maximum). The hardware then
uses 1, i.e. lmx16 = 0.

## 3. Hardware

All of it sits in `rtl/cim_seq.sv`, with `rtl/cim_log2fix16.sv`, plus small additions to
`rtl/cim_endpoint.sv`. Constants come from `rtl/wp_tables.svh`, generated from `mfcc_int.py`.

| Part | What it does | Storage |
|---|---|---|
| Hop maximum | the endpoint tracks the maximum of the samples written since the window path last took it | 16-bit register |
| `lraw` ring | 49 frames × 40 values; the new frame is copied from the front-end's `lraw` buffer (40 cycles) | 2 RAMB36 |
| Block ring | 50 hop maxima with valid bits; the window maximum is a 50-cycle scan | LUT-RAM |
| `cim_log2fix16` | floor(log2 v · 2^16) by 16 squarings of a Q30 mantissa, one per cycle | 4 DSP |
| Row datapath | lmq = max(round(((lraw − lmx16) · ln2_Q16) / 2^24), floor) into X; DCT passes on tiles 6 and 7 (xbits 14); acc = 256·y1 + lo; q = clamp(trunc((acc · RQ_M + 83 · 2^44) / 2^44)); 3 FEAT words | pipeline registers |

**Control** (`SEQ_CTRL`):
- `[30]` run the window path first;
- `[29]` window path only (no full window yet);
- `[28]` new frame (copy its `lraw`; the go waits for the front-end to be idle);
- `[23]` stream reset (forget the block maxima).

**Status:** `SEQ_STATUS[12]` (all rows recomputed), `SEQ_LMX` (0x2C), `SEQ_WMAX` (0x30). FEAT
is now also readable.

**Per hop**, as firmware does it:
1. write 320 samples into the ring;
2. start the front-end on frame h−1 and sleep;
3. start the sequencer with the window path (plus the network once h ≥ 49) and sleep;
4. read the class.

## 4. Verification

| Check | Result |
|---|---|
| Streaming model against the per-window reference, 951 decisions, both front-ends | 0 differ |
| `tb_cim_stream`, end to end through the endpoint with the real front-end, 3-s stream (150 hops, 101 decisions, 24 right after a maximum change), `cim_fe` and B3 | every hop's lmx16, window maximum and recompute flag exact; all 49,490 features and all logits and classes exact |
| `tb_cim_log2fix16`, every v = 1 … 32,767 | 0 wrong, latency exact |
| Window maximum → `SEQ_LMX` in the endpoint: 47 values including −32768, −1, 0, 1, 32767 | 0 wrong |
| Earlier testbenches (sequencer, endpoint, front-end ×2, engine) | unchanged, all pass |

**Bug found by the streaming testbench.** In the requantisation,
`ra_acc * $signed(RQ_M) + (57'(IN_ZP) << 44)` mixed one unsigned term into a signed sum. In
SystemVerilog that makes the whole expression unsigned, so the negative accumulator was
zero-extended, and 43 % of the features saturated at +127. The fix makes every operand signed.

## 5. Cycles

From `tb_cim_stream`, go → interrupt:

| Run | Cycles |
|---|---:|
| Window path, new frame only | 207 |
| Window path, all 49 frames | 4,678 – 4,719 |
| Window path + network, new frame only | 85,557 |
| Window path + network, all 49 frames | 90,069 |

The window path adds 205 cycles to a typical decision, and 4.7k when all 49 frames are
recomputed. Its rows are not overlapped; double-buffering would halve the 4.7k, but it is not
needed.

## 6. Gate level, SoC, board, cost

**Gate level** (Vivado netlists simulated against the RTL testbenches; see `SEQUENCER.md` §8a for why):

| Check | Result |
|---|---|
| `cim_log2fix16` netlist (4 DSP), every v = 1 … 32,767 | 0 wrong |
| Netlist of the whole `cim_seq` (with the window path) inside the RTL endpoint, `tb_cim_stream`: 60 hops, 11 decisions, the window-maximum sweep | every hop exact; 0 of 5,390 features wrong; logits exact |

The first gate-level `log2fix16` run hung because the testbench started the unit during the
netlist's 100-ns global set/reset. Starting after it fixed the testbench, not the design.

**SoC and board.** `sw/applications/hopcim_stream` (and `hopcim_stream_ded` for B3) streams 2 s of
GSC speech, 100 hops and 51 decisions, 16 of them right after a change of the window maximum.

Per hop, the CPU:
1. writes the 320 new samples;
2. starts the front-end and sleeps until its interrupt;
3. starts the sequencer (window path, plus the network once the window is complete) and sleeps;
4. reads the class.

| Per decision hop | `cim_fe` | B3 |
|---|---:|---:|
| CPU-active: copying 320 samples | 3,847 | 3,847 |
| **CPU-active: everything else** | **600** | **600** |
| Wall, new frame only | 97,512 (6.5 ms at 15 MHz) | 98,604 |
| Wall, all 49 frames recomputed | 102,024 (6.8 ms) | 103,116 |

Both pass with every hop and every decision bit-exact, on the Verilator SoC and on the board. Every
number in the table is identical on both. The sample copying disappears with the live audio path
(R7, `LIVE_AUDIO.md`).

For comparison, before R5b the CPU spent 2,120 cycles per frame on the window path, and up to
104k per decision when the window maximum changed.

**Cost** (Vivado out of context, 15 MHz):

| Unit | LUT (memory) | FF | BRAM tiles | DSP | WNS |
|---|---:|---:|---:|---:|---:|
| `cim_seq` with the window path | 2,443 (291) | 1,982 | 11.5 | 9 | +50.4 ns |
| Endpoint, `cim_fe` | 6,456 (629) | 4,947 | 46 | 144 | +44.0 ns |
| Endpoint, B3 | 6,277 (909) | 5,044 | 45 | 145 | +46.3 ns |

R5b adds about 1,300 LUT, 1,090 FF, 2 BRAM tiles (the `lraw` ring) and 6 DSP (`log2fix16` 4,
normalisation 1, requantisation 1) to the R6b endpoint.

Full SoC bitstreams (`xheep_nexys_bscan_pdm_cim128w_{cim,ded}.bit`):

| Bitstream | LUT | FF | BRAM tiles | DSP | 15 MHz setup / hold |
|---|---:|---:|---:|---:|---|
| `cim128w_cim` | 41,177 (65 %) | 39,330 | 82 | 145 | +16.59 / +0.220 ns |
| `cim128w_ded` | 41,066 | 39,439 | 81 | 146 | +15.25 / +0.222 ns |

The R6b bitstreams (`cim128s_*`) are kept unchanged.
