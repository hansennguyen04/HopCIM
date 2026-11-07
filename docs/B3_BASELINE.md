# B3 — dedicated MFCC front-end baseline (does reusing the CIM save hardware?)

Status: built and verified, 2026-10-27. Answers hypothesis H4 of `HOPCIM_PROPOSAL.md`.

## 1. What B3 is

`rtl/mfcc_ded.sv` is a dedicated front-end that does not use the CIM. It has the same contract
as `cim_fe` (ring read port, `lraw[40]` per frame), so the same endpoint registers and the same
firmware run on both. FuseSoC flag `use_cim_ded` (endpoint parameter `FE_DED`) selects it.

| Step | `cim_fe` (HopCIM) | `mfcc_ded` (B3) |
|---|---|---|
| Window, block exponent | Same code | Same code |
| FFT | Four-step: 48 CIM passes on int8 DFT tiles + Q14 twiddle multiplier | 512-point radix-2 DIT, 9 stages × 256 butterflies, 2 cycles each; 18-bit data, Q16 twiddles, each stage halved (no overflow); one simple dual-port RAMB18 buffer |
| Magnitude | Exact isqrt, bins 0–127 | Exact isqrt, bins 1–127 |
| Mel | 4 CIM passes on int8 mel tiles | 247 multiply-adds (≤ 2 bands per bin), one shared multiplier |
| Log2, lraw | Same code | Same code (its own scale constant) |
| Mel sums, lraw | LUT-RAM | LUT-RAM |
| Uses the engine? | Yes (tiles 0–5, about 900 engine cycles/frame) | No; the CPU can use the engine meanwhile |

Both are written in the same style (one operation per unit per cycle, sequential square root).
The shared structures are implemented identically.

Golden model: `model/mfcc_ded.py`. Test vectors: `model/gen_fe_vectors.py ded`.

## 2. Accuracy (same gate as gate 2)

`model/gate2_int.py ded`, 4,890 GSC test clips, unchanged int8 DS-CNN:

| Features | int8 accuracy | Δ vs float MFCC | McNemar | Int8 features equal to the reference / within ±1 |
|---|---:|---:|---|---|
| Float reference MFCC | 91.66 % | — | — | — |
| HopCIM CIM-MFCC | 91.78 % | +0.12 | 18 / 24, p = 0.44 | 96.7 % / 99.94 % |
| **B3 dedicated** | **91.60 %** | **−0.06** | 27 / 24, p = 0.78 | 93.9 % / 99.72 % |

Both pass. Halving at every stage costs B3 a little feature fidelity; the CIM pipeline is at
least as accurate as an 18-bit dedicated FFT.

## 3. Verification

| Test | Result |
|---|---|
| `tb_cim_fe` with `DED = 1`: 319 frames, 6 clips + 25 corner cases, ring wrap-around | 0 lraw mismatches; latency 528–8,608 cycles/frame |
| Mutation (butterfly truncates instead of rounding) | Caught: 297 of 319 frames wrong |
| Interplay | B3's go is not held behind a CPU pass, and CPU X writes proceed while B3 runs (the engine stays with the CPU) |
| SoC (Verilator), `hopcim_fe_ded`, `hopcim_fe_irq_ded` | 49 frames bit-exact against the B3 golden model (lraw and int8 features) |

## 4. Cost (Vivado out of context, xc7a100t, 15 MHz)

| Unit | LUT (of which memory) | FF | RAMB18 | DSP | WNS |
|---|---:|---:|---:|---:|---:|
| `cim_fe` (HopCIM front-end) | 1,530 (202) | 345 | 4 | 7 | +48.4 ns |
| **`mfcc_ded` (B3)** | **1,447 (484)** | 440 | 2 | 8 | +49.0 ns |
| Endpoint, 128 tiles, with `cim_fe` | 3,882 | 2,908 | 32 RAMB36 + 5 | 135 | +48.1 ns |
| Endpoint, 128 tiles, with B3 | 3,715 | 2,995 | 32 RAMB36 + 3 | 136 | +45.6 ns |

HopCIM also uses 6 CIM tiles (6 KiB of CIM capacity) for the front-end matrices; B3 uses none.
Both use the 2 DCT tiles of the window path.

**The first cost run was not fair, and was redone.** In that first version, B3 kept its mel sums
in 40 × 40-bit flip-flops with two variable-index read-modify-writes. That gave 4,575 LUT,
against 1,907 for `cim_fe`, whose mel sums were also flip-flops. A competent dedicated design
keeps those sums in a LUT-RAM, and so can `cim_fe`. The table shows both designs with that change.

## 5. Cycles and full-SoC cost

Verilator SoC, final RTL, `hopcim_fe_irq` / `hopcim_fe_irq_ded`: the CPU sleeps in WFI while the
front-end runs, and `mcountinhibit` stops `mcycle` during the sleep.

| Per frame | `cim_fe` | B3 |
|---|---:|---:|
| Front-end latency (go → interrupt) | 7,582 | 8,674 |
| CPU-active (samples + lraw readback + interrupt), CPU asleep during the front-end | 4,743 | 4,743 |
| Wall time | 12,257 | 13,349 |
| Window path (DCT + requantisation, CPU) | 2,120 | 2,120 |

With polling instead of sleeping (`hopcim_fe` / `hopcim_fe_ded`), the latency is 7,521 vs
8,613 cycles. All four runs are bit-exact against their own golden model, lraw and int8 features.
The CPU-active work is identical: in both cases the CPU only writes samples and reads `lraw`.

Full SoC on the Nexys A7-100T (128-tile CIM, BSCANE2 debug, PDM):

| Bitstream | LUT | FF | BRAM tiles | DSP | 15 MHz setup / hold |
|---|---:|---:|---:|---:|---|
| `xheep_nexys_bscan_pdm_cim128_cim.bit` (`cim_fe`) | 38,576 | 37,294 | 70.5 | 136 | +15.96 / +0.213 ns |
| `xheep_nexys_bscan_pdm_cim128_ded.bit` (B3) | 38,458 | 37,380 | 69.5 | 137 | +14.94 / +0.222 ns |

The only failing paths in both are the asynchronous crossings of the unused SPI-slave clock.

**Board** (Nexys A7-100T, `planenest/hw/board_suite.sh`, run twice, once by the student and once
by me, with identical output). All four front-end runs pass on their bitstreams: 49 frames
bit-exact, lraw and int8 features. **Every cycle count in the tables above is identical on the
board.** The R6b bitstreams (`xheep_nexys_bscan_pdm_cim128s_{cim,ded}.bit`, which add the layer
sequencer) reproduce every front-end count again.

This also validates the sleep accounting. On the FPGA the clock gate is a pass-through, so
without `mcountinhibit` around WFI, `mcycle` would also count the ~7,600–8,700 sleeping cycles
per frame. The board reports the same 4,743 CPU-active cycles as simulation, where the core
clock really stops. So the core does sleep in WFI, wakes on the front-end interrupt through the
PLIC, and the sleep is excluded from the count.

## 6. Conclusion for H4

**H4 is not supported.** At matched design style, the dedicated front-end costs about the same logic
as the CIM front-end's epilogue unit:
- 5 % fewer LUTs;
- 1 more DSP;
- 2 fewer RAMB18;
- no CIM capacity.

The cost of an exact MFCC front-end is dominated by what any implementation needs: window, block
exponent, square root, log, one complex multiplier, buffers. A radix-2 butterfly with its
address logic is cheap. Reusing the CIM for the FFT, mel and DCT therefore saves no hardware here;
counting the 6 KiB of CIM capacity, it costs slightly more. B3 also leaves the engine free for
the network.

What survives:
- **Moving the whole front-end into hardware is what matters** (R4b: 1.1× with only the MACs on
  the CIM; R5a or B3: about 22× fewer CPU-active cycles). Which hardware is a wash.
- **Capacity can replace multipliers (an open variant).** Folding the twiddles into 16 step-3
  tiles removes the 4-DSP complex multiplier for +15 CIM tiles, and folding the window removes
  one more DSP for +32 tiles. On an FPGA, and almost certainly in silicon, that is a poor trade.
- **The exact, model-preserving MFCC on a CIM remains a feasible technique.** It is now
  characterised honestly: same accuracy class and same logic cost as a dedicated unit, plus CIM
  capacity.
