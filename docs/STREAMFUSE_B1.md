# B1: StreamFuse B in one block, in RTL

Status (2026-11-08): built behind a parameter (`cim_seq` `FUSE1`, endpoint `SEQ_B1`, FuseSoC `--flag use_cim_b1`
with `use_cim_s`); all off, the sequencer is S cycle for cycle. This is the "one-block prototype" that
`STREAMFUSE_GATES.md` allows once B passed its gates, and `HopCIM_Specific_Next_Steps_and_Professor_Advice.md` §15
names: DW1 → exact requantisation → fusion vector → PW1. **It matched the model exactly** (§2).

## 1. What was built

L1 (depthwise, block 1) and L2 (pointwise, block 1) run as one segment, with the rest of the network as S.

| Part | Design |
|---|---|
| Order | DW(p, g = 0..9), then PW(p − 1), for p = 0..125 (B's order) |
| Line buffer | 128 words of 7 bytes, slot = raster word (10 pin + g) mod 128. The model's smallest hazard-free size is 120; 128 makes the slot the word's low 7 bits (+448 bits) |
| Fill | one map word per cycle over the ACT read port, which the fused gather does not use, in the order of first use: for each output (oy, ox) and group g, the taps that no earlier output read (new rows fy = 1, 2 at oy = 0, else 2; new columns fx = 1, 2 at ox = 0, else 2). 1,250 words |
| Fusion vector | 2 × 8 words of 8 bytes (half p mod 2, word c / 8, lane c mod 8); the pointwise gather reads 8 chunks of 8 rows |
| L2's output | map 1, so no layer writes in place; L3..L8 then use the two maps the other way round. B's one-map, in-place layout is what saves storage; one block cannot, since the other blocks still need two maps |
| Source interlock, fusion vector | a pointwise chunk k of position p waits until the depthwise results of p up to channel 8k + 7 are written (a position counter and a channel counter; DW results are written in order) |
| Source interlock, line buffer | a depthwise tap that is its word's first use waits until the fill has landed that many words (two counters) |
| Overwrite interlock, line buffer | a fill waits until the word it replaces (raster word − 128) has had its last read: that word's last reader (output (min(y+1, 24), min(x+1, 4)), its group, its tap) is compared with the gather's position |
| Overwrite, fusion vector | no check needed: PW(p − 2) is gathered before DW(p) is, and DW(p) writes after its own pass |

The model's per-word version tables became counters and one comparator; the assertions below keep the per-word
tables as a shadow, so the cheaper hardware is checked against the model's rule.

## 2. "B matched the model" (`STREAMFUSE_GATES.md`: cycles within 2 %, counters equal, 0 mismatches)

The prediction came first: `streamfuse_sched.py b1` (a `fuse1` schedule: L0 as S, the fused block with a 128-word
line buffer, L3..L8 with the maps swapped) gave 67,817 cycles and the access counts below before any RTL existed.

| Per inference | Schedule model | RTL | |
|---|---:|---:|---|
| Cycles, whole network | 67,817 | **67,817** | equal (S 67,841) |
| Cycles by segment: L0, fused L1 + L2, L3 .. L8, FC | 4,533, 15,780, 11,273, 4,531, 11,273, 4,531, 11,273, 4,531, 92 | the same, every one | `sim/logs/seq_tb_cim_b1.log`, stop after each layer |
| ACT word reads (gather + fill) | 33,470 | 33,470 | S 42,960 |
| Line-buffer reads / word writes | 9,490 / 1,250 | 9,490 / 1,250 | |
| Fusion-vector chunk reads / byte writes | 1,000 / 8,000 | 1,000 / 8,000 | |
| ACT byte writes | 56,000 | 56,000 | S 64,000 |
| Passes, requantisations, multiplies, gated results | as S | as S | |
| Exactness | 4,890 clips, 0 mismatches in L0..L8 and logits | **4,890 clips, 0 logit or class errors** | `sim/logs/seq_allclips_b1.log` |
| Hazards / source stalls | 0 / 6 (all on the fusion vector) | assertions: 0 failures | |

RTL counts from `sim/run_energy_counts.sh cim 5 1 b1` (101 decisions, `model/energy_counts_cim_b1.log`).

## 3. Verification

| Check | Result |
|---|---|
| `tb_cim_seq`, every layer of clip 0 (L1 from a monitor on the fusion-vector writes), 20 clips, FEAT ring, interplay, both synthetic parameter sets | PASS, 0 mismatches (`sim/logs/seq_tb_cim_b1.log`); also with B3 (`seq ded b1`) |
| `tb_cim_stream`: 150 hops, 101 decisions | PASS |
| `tb_cim_mic`, live path at 15.36 MHz and at 4.096 MHz (k = 1, 4) | PASS, no overrun; worst hop 80,048 (S 80,072) |
| Verilator SoC (X-HEEP with `use_cim_s use_cim_b1`, from `xheep/setup.sh`), `hopcim_seq` | 60 runs bit-exact, **67,885 cycles go to interrupt** (RTL + 68, as for S), CPU-active unchanged |
| Assertions (`tb/cim_assertions.sv`, `sva_seq_b1`, a shadow of both buffers) | 6 new checks, all exercised (e.g. 502,970 line-buffer reads), 0 failures |
| Mutants (`model/verif_mutants.py`) | 6 of 6 caught; one more is equivalent (below) |
| Board, `xheep_nexys_bscan_cim128kb1.bit` (15.36 MHz, built by `xheep/build.sh <tree> b1`), `hopcim_seq` | 60 runs bit-exact, **67,885 cycles go to interrupt**, the Verilator SoC's figure |
| Board, `board_khop.sh 42` | 500, 250, 125 decisions per 10 s at k = 1, 2, 4, **no overrun**; CPU-active 263, 263, 266 cycles per decision (as S); real-microphone logits **exact at 25, 13, 7 of 25, 13, 7** decisions |

| Broken B1 | Caught by |
|---|---|
| fusion-vector source interlock removed | `b1_fv_read_complete`; the bench fails |
| a fill ignores the readers of the word it replaces | `b1_fill_overwrites_dead_words`; the bench fails |
| pointwise reads the other fusion-vector half | `b1_fv_read_complete`; the bench fails |
| the fill writes the neighbouring slot | the bench |
| L2 written to map 0 | the bench |
| a depthwise result also written to ACT | `b1_dw_not_to_act` only: the bench passes, since L2 overwrites those words later; an energy defect only an assertion can see |
| *equivalent:* line-buffer source interlock removed | nothing changes: the fill, in the order of first use, is always ahead of the gather. All 6 source stalls of the model are on the fusion vector; the model without the line-buffer check has the same 67,817 cycles and no hazard |

So of the two interlocks the gates called required, **only the fusion-vector one ever acts** in this order, and the
line buffer's overwrite check (the fill waited 13,593 cycles per inference in the model) is the other one that matters.

## 4. Cost

Vivado 2024.1, out of context, xc7a100t, 15 MHz (`syn/ooc.tcl`), S rerun on the same source for a fair difference
(`syn/ooc_cim_endpoint_S_same_source_as_B1`; S's frozen figure, 8,116 LUT, came from the source before B1 existed and
differs by 10 LUT of synthesis variation in unchanged logic):

| Endpoint | LUT | of it LUT RAM | FF | BRAM tiles | DSP | WNS / WHS |
|---|---:|---:|---:|---:|---:|---:|
| S, same source | 8,126 | 604 | 5,980 | 46.5 | 147 | +46.4 / +0.10 ns |
| **B1** | **8,761** | 822 | **6,182** | 46.5 | 147 | +47.2 / +0.02 ns |
| difference | **+635** | +218 (line buffer, fusion vector) | **+202** | 0 | 0 | |

Whole SoC on the board (`xheep_nexys_bscan_cim128kb1.bit`): 43,418 LUT (S 42,758: +660, +1.5 %), 40,550 FF (+187),
82.5 BRAM tiles and 148 DSP (both as S); the system clock meets timing with +16.3 ns setup and +0.229 ns hold slack.

In `cim_seq`: +418 logic LUT, +2 SRL, +197 FF of control, the rest is the two buffers' LUT RAM.
**The gates priced B's added control at 25 % of R6b's `cim_seq` (294 LUT, 224 FF) for all four blocks; one block
measured +420 LUT and +197 FF**, so the logic is 1.4 × the assumed share, inside the study's doubled-overhead
sensitivity. Much of it (the fill generator, the interlocks, the fused gather sequencing) would be shared by four
blocks, but that is not measured.

## 5. Energy [E]

`energy_est.py b1_rtl` prices B1 from its RTL counters with the frozen StreamFuse rules (line buffer by CACTI at 128
words, 0.598 pJ per read at 28 nm; fusion vector as flip-flops), and charges the measured control every cycle with its
leakage. `energy_b1_rtl.log`:

| One inference, baseline variant | S (RTL counts) | B1 (RTL counts) |
|---|---:|---:|
| ACT memory (with line buffer and fusion vector) | 285.8 nJ | 239.6 nJ |
| Sequencer logic | 155.5 nJ | 175.0 nJ |
| **Network** | **1,120.8 nJ** | **1,094.1 nJ (−2.4 %)** |
| Total power at 20 / 40 / 80 ms | 97.0 / 68.8 / 54.7 µW | 96.3 / 68.7 / 55.0 µW |

- B1 against its own model: ratio 1.0000 in all 19 variants (counts and cycles are equal).
- B1 / S: 0.970 to 0.990 over the variants; with the frozen 25 % rule instead of the measured control, 0.970 to 0.990 too.
- **One block saves 2.4 % of the network energy and nothing at 40 ms** (68.8 → 68.7 µW); at 80 ms the added control
  and buffers leak more than they save (54.7 → 55.0 µW). The model's −16 % for B needs all four blocks and B's one map.

## 6. What this shows, and the decision it leaves

1. **The schedule model predicted the RTL exactly**: every segment's cycles and every access count. The modelled
   numbers for all of B (0.837 × S, 0.575 × the storage) therefore rest on a model that has now been checked against
   RTL in the one block where B's new mechanisms appear, not only against the layer-by-layer designs.
2. **Measured control: +420 LUT, +197 FF for one block**, 1.4 × what the gates assumed for all of B.
3. **Only one of the two interlocks acts in practice** (the fusion vector's); the line buffer's source check never
   waits with this fill order. The overwrite check of the line buffer is essential (its mutant fails).
4. **One block is not worth building for its own sake**: −2.4 % network energy, 0 at 40 ms, more storage than S.

The frozen rule now allows **all four blocks** ("only after one block matched the model"). That is the student's
decision: it adds B's one map of 26 rows with in-place writes (the row lag) and its overwrite interlock, i.e. the part
that saves storage, and should take the network energy toward the modelled 0.837 × S (about −7 % of total power at
40 ms). The novelty is in the evaluation (`STREAMFUSE_NOVELTY.md`), not the mechanism.

## 7. Limits

- One block only. The storage saving of B (one map of 26 rows written in place) is not built, so B1 needs more
  activation storage than S (148,192 against 140,000 bits).
- Energy is the model-based estimate [E] with the frozen rules; on the FPGA the line buffer and the fusion vector are
  LUT RAM, in the estimate a 128-word SRAM and flip-flops.
- With B1, L1 is never in ACT and L2..L8 are in the other map: the debug app `hopcim_seq_dbg`, which reads layers back
  by S's map parity, is not adapted. The production firmware reads only the logits and the class.
- The line-buffer source check is kept although it never binds in this order; a different fill order or a smaller
  buffer could need it.

## 8. Reproduce

```
cd hopcim
~/venvs/planenest/bin/python model/streamfuse_sched.py b1        # the prediction, 4,890-clip replay (~1.5 min)
bash sim/run_seq_tb.sh cim b1 && bash sim/run_seq_allclips.sh b1  # RTL, every layer; all 4,890 clips (~13 min)
bash sim/run_energy_counts.sh cim 5 1 b1 && python3 model/energy_est.py b1_rtl
vivado -mode batch -source syn/ooc.tcl -tclargs cim_endpoint FE_DED=0 PDM_CLKDIV=15 SEQ_S=1 SEQ_B1=1
xheep/build.sh <x-heep tree> b1                                   # bitstream xheep_nexys_bscan_cim128kb1.bit
```
