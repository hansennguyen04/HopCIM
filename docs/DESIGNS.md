# The designs: the default (S+DWU) and the comparison baseline (S)

Decided 2026-11-25 by the student: **the default HopCIM design is S+DWU**, the baseline S with the digital depthwise
unit in all four depthwise layers (`DWU_RESULTS.md`). **S is kept unchanged as the comparison baseline** for the thesis.
"HopCIM" without a qualifier means S+DWU from this date; numbers written before it are S's (or R6b's) and say so.

## 1. How to select each design

| | R6b | **S (baseline)** | **S+DWU (default)** |
|---|---|---|---|
| Endpoint parameters | `SEQ_S = 0` | `SEQ_S = 1`, `SEQ_DWU = 0` | `SEQ_S = 1`, `SEQ_DWU = 4` |
| FuseSoC flags (with `use_cim use_cim_mic`) | – | `use_cim_s` | `use_cim_s use_cim_dwu` |
| Build variant (`xheep/build.sh`), 15.36 MHz | `r6b` | `s` | `dwu` |
| … at 4.096 MHz | – (`r6b5`: 5.12 MHz) | `s4` | `dwu-4` (§5) |
| Verilator SoC | `sim` | `sim-s` | `sim-dwu` |
| Benches (`sim/run_*_tb.sh`) | no option | `s` | `d4` (`d1`: first depthwise layer only) |
| Bitstream, `planenest/hw/bitstreams/` | `…_cim128k.bit` | `…_cim128ks.bit` (`971ddb49…`) | `…_cim128kdwu.bit` (`87954077…`) |
| … 4.096 MHz bitstream | | `…_cim128mic4s.bit` | `…_cim128mic4dwu.bit` (`3b4ec1c7…`) |
| Firmware flag | none | none | **`-DCIM_DWU=4`** |
| Board firmware, `planenest/hw/` | `hopcim_seq_nexys_15m36.elf`, `hopcim_live{,_k2,_k4}_nexys.elf`, `hopcim_micrec…` | the same files | `hopcim_seq_dwu_nexys.elf`, `hopcim_live{,_k2,_k4}_dwu_nexys.elf`, `hopcim_micrec…_dwu_nexys.elf` |

**The firmware must match the bitstream.** S's firmware on the unit's bitstream, or the reverse, runs and gives wrong
logits (`DWU_RESULTS.md`, R6).

```bash
cd planenest/hw
# default
./board_run.sh hopcim_seq_dwu_nexys.elf bitstreams/xheep_nexys_bscan_cim128kdwu.bit 120
TAG=_dwu ./board_khop.sh 42 bitstreams/xheep_nexys_bscan_cim128kdwu.bit
# baseline S
./board_run.sh hopcim_seq_nexys_15m36.elf bitstreams/xheep_nexys_bscan_cim128ks.bit 120
./board_khop.sh 42 bitstreams/xheep_nexys_bscan_cim128ks.bit
```

**The code's own defaults are unchanged, on purpose.** RTL parameters and `CIM_DWU` still default to off, so every
command, hash and frozen result written for R6b, S and B1 stays reproducible; the default is chosen by name (the table
above). Nothing of S was removed or edited: on 2026-11-25 all 18 bitstreams and 33 ELFs of the last freeze, S's among
them, still had their frozen hashes.

## 2. S against S+DWU

Labels: [M] measured (RTL simulation, board or Vivado), [E] energy-model estimate. Sources: `DWU_RESULTS.md` and
`model/dwu_rtl.json`; for S, `BASELINE_S.md` and `freeze/RESULTS.md`.

| | | S | S+DWU | Change |
|---|---|---:|---:|---:|
| Exact on 4,890 test clips (logits and class) | [M] RTL | 0 errors | 0 errors | none |
| Accuracy (the reference model's own) | [M] | 91.80 % | 91.80 % | none |
| Network cycles, sequencer bench | [M] RTL | 67,841 | 67,789 | −0.08 % |
| go → interrupt, board | [M] board | 67,909 | 67,856 | 52 of the 53 from the hardware, 1 from the firmware |
| Worst hop of the live path | [M] RTL | 80,072 | 80,020 | −52 |
| Live path at 15.36 MHz, k = 1, 2, 4 | [M] board | no overrun | no overrun | none |
| CPU-active cycles per decision (k = 1, 2, 4) | [M] board | 263 / 263 / 266 | 264 / 266 / 267 | +1 to +3 |
| CIM passes on the engine, per inference | [M] RTL | 7,501 | 2,501 | −5,000 |
| CIM tiles used | | 69 (5 macros) | 29 (2 macros) | −40 |
| Energy per inference | [E] | 1,120.8 nJ | 1,013.2 nJ | −9.6 % |
| Leakage | [E] | 36.33 µW | 20.52 µW | −43.5 % |
| **Power at mode U** (k = 16) | [E] | 44.11 µW | **27.96 µW** | **−36.6 %** |
| **Power at 40 ms** (k = 2) | [E] | 68.78 µW | **50.28 µW** | **−26.9 %** |
| … range over the 19 model variants, mode U | [E] | | 0.450 to 0.699 × S | lower in all 19 |
| Endpoint, out of context: logic LUT / FF / RAMB18 / DSP | [M] Vivado | 7,489 / 5,980 / 11 / 147 | 8,743 / 6,795 / 13 / 147 | +1,254 / +815 / +2 / 0 |
| Added digital logic, as priced | [E] | – | 12,414 gate equivalents | cost |
| Whole SoC on the FPGA: LUT | [M] Vivado | 42,758 | 44,022 | +3.0 % (other RTL changes since S included) |

Read with care:
- **The gain is power, not speed.** About 98 % of the saving at U is leakage, so it depends on how the model organises
  tiles into macros. The FPGA keeps 128 tile slots in both designs, so it shows no size or power saving.
- **Against the swap-in (V, 28.13 µW at U) the unit's 0.6 % advantage is within the model's uncertainty**; at 40 ms it
  is 6 % (50.28 against 53.63).

## 3. Results that exist for one design only

Do not mix these with the other design's numbers without saying so.

| Result | S | S+DWU |
|---|---|---|
| Real time at 4.096 MHz | RTL and board | RTL (worst hop 80,020 ≤ 81,920); board run in §5 |
| Mode U (k = 16, sum of 2) live on the board | yes (`LIVE_AUDIO.md`) | RTL exact (7 of 7 decisions, 15.36 and 4.096 MHz); board run in §5 |
| Spoken keyword tests | yes | not run (the logits are the same on the test set and on the 45 recorded decisions) |
| MLPerf® Tiny runner, closed and open (unverified) | yes | not run |
| TensorFlow Lite Micro CPU baseline (256 KiB) | yes | not applicable |
| B1 (one fused block) | built on S | not built on the unit |
| Power at 20 and 80 ms [E] | yes (97.0 and 54.7 µW) | not computed |
| Macro-timing sweep (issue intervals 2b, 2b + 8) | yes | not run |
| Assertions, coverage | yes | the unit's benches and 3 mutants; assertions not extended to the unit |

## 4. Keeping S usable

- `sim/run_all.sh` runs S's benches with the unit's (53 in all); run it after any change to `cim_seq.sv`.
- Do not change `SEQ_S = 1, SEQ_DWU = 0` behaviour. A change that must affect both designs is a new parameter.
- Do not overwrite S's bitstream or ELFs (`xheep/build.sh` refuses without `FORCE=1`).

## 5. Board runs of the default design: expectations written before the runs (2026-11-25)

Firmware built 2026-11-25 with `xheep/build.sh <tree> app`, all with `-DCIM_DWU=4`:

| ELF (`planenest/hw/`) | Flags | For |
|---|---|---|
| `hopcim_live_u_dwu_nexys.elf` (`0b3ac9c9…`), `hopcim_kwtest_u_dwu_nexys.elf` (`6dc6a95b…`) | `REFERENCE_CLOCK_Hz=15360000 NET_K=16 KW_SUM=2` | mode U, `…_cim128kdwu.bit` |
| `hopcim_seq_dwu_nexys_4m096.elf` (`fb4e091f…`), `hopcim_live_dwu_nexys_4m096.elf` (`b35710c7…`), `hopcim_micrec_dwu_nexys_4m096.elf` (`1135a8aa…`) | `REFERENCE_CLOCK_Hz=4096000` | 4.096 MHz, `…_cim128mic4dwu.bit` |
| `hopcim_live_u_dwu_nexys_4m096.elf` (`4967e9a3…`) | `REFERENCE_CLOCK_Hz=4096000 NET_K=16 KW_SUM=2` | mode U at 4.096 MHz |

| Run | Expected |
|---|---|
| Mode U live, 15.36 MHz (`hopcim_live_u_dwu_nexys.elf`) | 31 decisions per 10 s (500 / 16 = 31.25), 0 missed, overrun 0; CPU-active near S's 875 cycles per decision |
| Sequencer, 4.096 MHz (`hopcim_seq_dwu_nexys_4m096.elf`) | 60 runs bit-exact; go → interrupt **67,856** (same wake-up code as at 15.36 MHz) |
| Live, 4.096 MHz (`hopcim_live_dwu_nexys_4m096.elf`), about 120 s | 500 decisions per 10 s, overrun 0 (S: 6,000 in 120 s) |
| Live mode U, 4.096 MHz (`hopcim_live_u_dwu_nexys_4m096.elf`) | 31 decisions per 10 s, overrun 0 |
| Microphone recording, 4.096 MHz (`hopcim_micrec_dwu_nexys_4m096.elf`) | 25 of 25 decisions equal to the golden model |

RTL evidence for these rows: `sim/mic_tb_cim_k16_d4.log`, `sim/mic_tb_cim_div4_k16_d4_hm.log` (mode U's k, 7 of 7
decisions, worst hop 80,020), `sim/mic_tb_cim_div4_d4_hm.log` (k = 1).

**The 4.096-MHz bitstream** (`xheep/build.sh <tree> dwu-4`, 42 min; `xheep_nexys_bscan_cim128mic4dwu.bit`, sha256
`3b4ec1c73dd34312…`): 44,022 LUTs, 41,176 registers, 83.5 block-RAM tiles, 148 DSPs. Timing, analysed at 8.192 MHz, stricter
than the real 4.096 MHz: system clock setup slack +15.58 ns, hold +0.259 ns, no failing endpoint. Vivado's design-wide
summary is **not met**, for two reasons that S's 4.096-MHz bitstream shares, with the same hold figures:
- setup −7.720 ns on 64 endpoints: the crossing to X-HEEP's stock, unused SPI-slave clock (S: −7.911);
- hold −0.033 ns (TNS −0.114) on 4 endpoints (`i_seq/d_p_reg[2..5]`): only between `clk_8m192_raw` and `clk_8m192_raw_1`,
  which Vivado derives from the two names (`clk_i`, `sys_clk_pin`) of the board's one 100-MHz input. The same registers meet hold within each
  clock (+0.259 ns). S's bitstream, with the same report, ran 6,000 decisions on the board without error.
