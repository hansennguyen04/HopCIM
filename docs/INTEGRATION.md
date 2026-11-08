# R2/R3 — `cim_endpoint` and its X-HEEP integration

## 1. Endpoint (R2)

`rtl/cim_endpoint.sv` wraps `cim_engine` (R1, `ENGINE_SPEC.md`) as a 32-bit OBI slave with a
256 KiB window and 128 resident tiles. The CPU writes weights and inputs, starts passes and reads
results (R2), feeds audio to the hardware front-end (R5a), or writes the int8 features and lets
the layer sequencer run the DS-CNN to a class (R6b, `SEQUENCER.md`). The parameter `FE_DED` picks
the front-end: `cim_fe`, or the dedicated baseline B3 (`B3_BASELINE.md`).

| Offset | Name | Access | Content |
|---|---|---|---|
| `0x0_0000` | CTRL | W | `[31]` go, `[16]` accumulate, `[12:8]` input width b, `[6:0]` tile. A go is held on the bus (no grant) until the engine can accept it. |
| `0x0_0004` | STATUS | R / W1C | `[0]` ready, `[1]` done (sticky; cleared by a go or by writing 1), `[2]` irq pending |
| `0x0_0008` | IRQ_EN | RW | `[0]` irq on CPU-pass done, `[1]` irq on front-end done, `[2]` irq on sequencer done |
| `0x0_000C` | DONE_CNT | R | completed CPU passes (front-end passes are not counted) |
| `0x0_0010` | FE_CTRL | W | `[31]` go, `[9:0]` ring index of the frame's first sample |
| `0x0_0014` | FE_STATUS | R / W1C | `[0]` busy, `[1]` done (sticky) |
| `0x0_0018` | CYCLES | R | free-running clock counter (wall time) |
| `0x0_0020` | SEQ_CTRL | W | `[31]` go, `[27:24]` stop (0: whole network; s: after layer s − 1), `[21:16]` FEAT row of frame 0, `[6:0]` first network tile. Held until no CPU pass is in flight and `cim_fe` is idle |
| `0x0_0024` | SEQ_STATUS | R / W1C | `[0]` busy, `[1]` done (sticky), `[2]` error (a result lost; never expected), `[7:4]` class, `[11:8]` layer |
| `0x0_0028` | SEQ_ZP | RW | zero points: `[7:0]` conv1 input, `[15:8]` activations, `[23:16]` logits |
| `0x0_0040` | RES_LO[16] | R | result bits `[31:0]` of the last completed CPU pass |
| `0x0_0080` | RES_HI[16] | R | result bits `[47:32]`, sign-extended |
| `0x0_1000` | X[64] | W | input `x[r] = wdata[23:0]` |
| `0x0_2000` | SAMPLES[1024] | W | audio ring, `wdata[15:0]` |
| `0x0_3000` | LRAW[40] | R | lraw of the last front-end frame |
| `0x0_4000` | FEAT | W | int8 features: row r (0..48) `off[9:4]`, word w `off[3:2]` = bytes 4w..4w+3 of the row; a ring of 49 rows |
| `0x0_5000` | LOGITS[12] | R | int8 logits of the last full run, sign-extended |
| `0x0_8000` | PARAMS | W | entry 64 L + c `off[13:4]`; field `off[3:2]`: 0 bias, 1 multiplier, 2 e = −shift |
| `0x1_0000` | ACT | R | debug: activation of buffer `off[15]`, position `off[14:8]`, channel `off[7:2]` |
| `0x2_0000` | WEIGHTS | W | 128 KiB: tile `off[16:10]`, row `off[9:4]`, word `off[3:2]` (4 int8, byte k = column 4q+k) |

X and WEIGHTS writes are held for 10 cycles after a pass starts, while the engine reads them.
While `cim_fe` runs it owns the engine, and CPU go, X and WEIGHTS writes are held. B3 never
touches the engine. While the sequencer runs it owns the engine: CPU go, X, WEIGHTS, FEAT and
PARAMS writes, ACT reads and a `cim_fe` go are held; B3 may run meanwhile. The sequencer
overwrites both X banks. Driver: `sw/cim.h`.

## 2. X-HEEP integration (R3)

X-HEEP tree: `planenest/hw/x-heep` (commit `b3eb3c8` plus local changes). The HopCIM core is
found by FuseSoC through the symlink `hw/ip/hopcim -> ../../../../../hopcim` (`hopcim.core`,
`hopcim:ip:cim_endpoint`). Everything is behind the FuseSoC flag `use_cim` (define `USE_CIM`).
Adding the flag `use_cim_ded` (define `CIM_FE_DED`) selects the B3 front-end.

| File (x-heep) | Change |
|---|---|
| `core-v-mini-mcu.core` | `USE_CIM` parameter; `use_cim ? (USE_CIM=true)` in the `sim` and `nexys-a7-100t` targets; `use_cim ? (hopcim:ip:cim_endpoint)` in `rtl-fpga` |
| `tb/x-heep-tb-utils.core` | `use_cim ? (hopcim:ip:cim_endpoint)` |
| `tb/testharness_pkg.sv` | external slave 3 reserved for the CIM: `CIM_START_ADDRESS = 0xC004_0000`, 256 KiB (`EXT_XBAR_NSLAVE` 3 → 4); unused since the direct tap below |
| `tb/testharness.sv` | `cim_endpoint` on the CPU data master's external-slave port, a direct tap as on the FPGA (`ext_bus` keeps the other masters); interrupt on external line 3 |
| `hw/fpga/xilinx_core_v_mini_mcu_wrapper.sv` | `cim_endpoint` (128 tiles, `FE_DED` from `CIM_FE_DED`) on the CPU data master's external-slave port (`ext_core_data_req_o`, a 1-to-1 tap: it sees 0xC000_0000–0xFFFF_FFFF and decodes `addr[17:0]`, so 0xC004_0000 works unchanged); interrupt on external line 3. Exclusive with `USE_HLS_EXAMPLE`. |
| `sw/applications/cim_smoke/main.c` | SoC smoke test (below) |
| `hw/vendor/pulp_platform/riscv_dbg/src/dmi_bscane_tap.sv` | local patch: BSCANE2 TCK on a BUFG (see below) |
| `hw/fpga/xheep_fpga_support/constraints/common/bscan.xdc`, `core-v-mini-mcu-fpga.core` | new: TCK clock (100 ns) as its own asynchronous group, included with `use_bscane_xilinx` |

Path in both builds: CPU → system bus demux → endpoint, no crossbar. An earlier simulation
build routed the endpoint through the testharness `ext_bus` crossbar. That crossbar can accept a
store before the slave grants it, which hid the CPU loop overhead between back-to-back go
writes: the 8-pass chain took 98 cycles in simulation against 144 on the board. With the
direct tap the two agree cycle for cycle. The DMA cannot reach the endpoint in either build;
the front-end (R5a) moves its own data.

**BSCANE2 debug fix.** The first CIM bitstream programmed and passed timing, but OpenOCD
reported `Debug Module did not become active. dmcontrol=0x0`. The earlier PDM bitstream still
worked on the same board. The BSCANE2 TCK drove the 241 DTM registers on local routing with no
clock defined, so Vivado never timed them. Re-timing the routed design with TCK defined showed 33 hold
violations (worst −0.82 ns, from 1.27 ns of clock skew). Adding the CIM had changed the placement.
Fix: a BUFG on TCK plus `bscan.xdc`. The routed TCK domain then meets +91.6 ns setup and
+0.101 ns hold, and the debug module works.

### Build and run

```bash
cd ~/APlus_Project/planenest/hw/x-heep && source .venv/bin/activate
# Verilator models; keep the binary outside FuseSoC's work directory (a rebuild wipes it)
make verilator-build FUSESOC_FLAGS=--flag=use_cim
cp build/openhwgroup.org_systems_core-v-mini-mcu_1.0.6/sim-verilator/Vtestharness ~/APlus_Project/hopcim/sim/bin/Vtestharness_cim
make verilator-build "FUSESOC_FLAGS=--flag=use_cim --flag=use_cim_ded"   # baseline B3 (-> Vtestharness_ded)
SIMVAR=cim ~/APlus_Project/hopcim/sim/run_soc.sh cim_smoke         # app + run, checks PASS
# FPGA; a separate --build-root must keep the default depth (board files use ../../../).
# Never run two FuseSoC builds at once: the boot-ROM generator runs in place (hw/ip/boot_rom,
# `rm -f *.img *.bin *.sv`) and a concurrent run deletes the other's image.
make vivado-fpga FPGA_BOARD=nexys-a7-100t "FUSESOC_FLAGS=--flag=use_bscane_xilinx --flag=use_cim"
make app PROJECT=cim_smoke TARGET=nexys-a7-100t
../board_run.sh sw/build/main.elf ../bitstreams/xheep_nexys_bscan_pdm_cim.bit
# Live audio (R7): the on-board microphone and a 15.36 MHz system clock (requires use_cim)
fusesoc --cores-root . run --no-export --target=nexys-a7-100t --build-root build_mic/core \
  --flag=use_bscane_xilinx --flag=use_cim --flag=use_cim_mic --build openhwgroup.org:systems:core-v-mini-mcu
make app PROJECT=hopcim_live TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
FOLLOW=1 ../board_run.sh sw/build/main.elf ../bitstreams/xheep_nexys_bscan_cim128mic.bit 60   # FOLLOW=1 prints the UART live
# The same at the lowest real-time clock, 5.12 MHz (PDM divider 5): add --flag=use_cim_mic5 (build root build_mic5/core)
# and build the firmware with -DREFERENCE_CLOCK_Hz=5120000 (bitstream xheep_nexys_bscan_cim128mic5.bit)
# The baseline S (BASELINE_S.md): add --flag=use_cim_s (build root build_micks/core, bitstream xheep_nexys_bscan_cim128ks.bit);
# its lowest real-time clock, 4.096 MHz: --flag=use_cim_s --flag=use_cim_mic4 (build root build_mic4s/core, bitstream
# xheep_nexys_bscan_cim128mic4s.bit) and firmware with -DREFERENCE_CLOCK_Hz=4096000
# Record 1.5 s of the microphone with the decisions made on it, and check them against the golden model:
make app PROJECT=hopcim_micrec TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
../board_run.sh sw/build/main.elf ../bitstreams/xheep_nexys_bscan_cim128mic.bit 150
python hopcim/model/check_micrec.py <the run's .uart.log> [--wav rec.wav]
# A decision every k hops (LIVE_AUDIO.md §9): build root build_mick/core, bitstream xheep_nexys_bscan_cim128k.bit, and any
# of these apps with COMPILER_FLAGS="-DREFERENCE_CLOCK_Hz=15360000 -DNET_K=<k>"; planenest/hw/board_khop.sh measures k = 1, 2, 4
# Scripted keyword test (LIVE_AUDIO.md §8.1): hopcim_live with KW_TEST rounds (default 5; -DKW_TEST=1 for one round)
make app PROJECT=hopcim_kwtest TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
FOLLOW=1 ../board_run.sh sw/build/main.elf ../bitstreams/xheep_nexys_bscan_cim128mic.bit 330
python3 hopcim/model/score_kwtest.py <the run's .uart.log>
```

**X-HEEP files changed for the live audio path (`use_cim_mic`):**

| File | Change |
|---|---|
| `core-v-mini-mcu.core` | parameter `USE_CIM_MIC` and the flag |
| `hw/fpga/xheep_fpga_support/core-v-mini-mcu-fpga.core` | the clock wizard is a fileset per flag |
| `hw/fpga/xheep_fpga_support/scripts/nexys/xilinx_generate_clk_wizard_15m36.tcl` | 15.36 MHz: VCO = 100 MHz × 48 / 5, divided by 62.5 |
| `hw/system/x_heep_system.sv` and `x_heep_system.sv.tpl` | ports `cim_pdm_clk_i` (drives the PDM clock pad) and `cim_pdm_data_o` (the PDM data pad's input) |
| `hw/fpga/xilinx_core_v_mini_mcu_wrapper.sv` | connects them to the endpoint's `pdm_clk_o` / `pdm_data_i` |
| `sw/device/target/nexys-a7-100t/x-heep.h` | `REFERENCE_CLOCK_Hz` can be overridden from the command line |
| `tb/testharness.sv` | ties the endpoint's microphone input low |
| `core-v-mini-mcu.core`, `core-v-mini-mcu-fpga.core`, the wrapper, `xilinx_generate_clk_wizard_5m12.tcl` | `use_cim_mic5`: 5.12 MHz (VCO = 100 MHz × 32 / 5, divided by 125) and the endpoint's `PDM_CLKDIV = 5` |
| `core-v-mini-mcu.core`, the wrapper, `tb/testharness.sv` | `use_cim_s` (`CIM_SEQ_S`): the baseline S in the sequencer, the endpoint's `SEQ_S = 1` (`BASELINE_S.md`); on the FPGA and in the Verilator SoC |
| `core-v-mini-mcu.core`, `core-v-mini-mcu-fpga.core`, the wrapper, `xilinx_generate_clk_wizard_8m192.tcl` | `use_cim_mic4` (`CIM_MIC_4M096`, needs `use_cim_mic` and `use_cim_s`): 4.096 MHz and `PDM_CLKDIV = 4`. No MMCM output divider reaches 4.096 MHz from 100 MHz (the fractional one, CLKOUT0, would need 156.25 > 128), so the MMCM makes 8.192 MHz (VCO 640 MHz, divided by 78.125) without an output buffer, and a BUFGCE in the wrapper passes every second rising edge; timing is analysed at 8.192 MHz |

**Since 2026-11-08 every X-HEEP change above lives in this repository** as a patch series on X-HEEP
`b3eb3c8` (`xheep/`, see `xheep/README.md`): `xheep/setup.sh <dir>` builds an X-HEEP tree from upstream
(clone, `git am` of 3 patches, links to the RTL and to the firmware in `sw/applications/`, `make mcu-gen`),
`xheep/check.sh` checks a tree against it (the working tree `planenest/hw/x-heep` passes), and
`xheep/build.sh <dir> s` builds the S bitstream. A tree made this way from a clean clone produced a
bitstream identical to `xheep_nexys_bscan_cim128ks.bit` apart from the 4 bytes of the build time in its
header. The changes inside vendored directories (`xheep_fpga_support`, `riscv_dbg`, `xheep_spi`) are also in
`hw/vendor/patches/` with a `patch_dir`, so `util/vendor.py` keeps them on a re-vendor. Two hand edits of
generated files (`tb/testharness.sv`, `tb/testharness_pkg.sv`) were moved into their templates.

**Timing of the Nexys builds.** Every bitstream, the stock ones included, reports negative slack
from X-HEEP's SPI-slave pin clock (`spi_slave_clk_pin`, declared at 50 MHz on a 7-segment pin) to
the system clock and back: −3.2 to −5.5 ns at 15 MHz, −13.0 ns at 15.36 MHz. These are unconstrained
crossings of an interface that is unused on this board. The figures quoted for each bitstream are
for the system clock, where all endpoints are met.

## 3. Results

**Unit level** (Verilator 5.040): `tb_cim_engine` 2,401 passes bit-exact, done timing and issue
interval exact; `tb_cim_endpoint` 244 checks, 0 errors (register map, back-to-back go spacing D,
X-write hold, W1C, IRQ).

**Vivado out of context** (xc7a100tcsg324-1, 15 MHz, `syn/ooc.tcl`). Budget: ≤ 10k LUT, ≤ 140 DSP, ≤ 16 RAMB36.

| Top | LUT | FF | RAMB18 | DSP | WNS at 66.7 ns |
|---|---:|---:|---:|---:|---:|
| `cim_engine` | 2,015 | 1,648 | 32 | 128 | +51.8 ns |
| `cim_endpoint` | 2,314 | 2,519 | 32 | 128 | +51.3 ns |

Two inference pitfalls were fixed on the way: 32-bit partial writes into 128-bit BRAM words, and
a registered read-data stage that Vivado absorbed into the DSP B pipeline (BREG = 2). Both made
the weights LUTRAM (16.6k and 11.5k LUTs). The weights are now 32 banks of 512 × 32 with a
registered read address.

**SoC level, Verilator** (`cim_smoke`, CPU-programmed, mcycle): 16 passes (8 single over widths
1–24 and both accumulate modes, 8 back-to-back), all bit-exact against the C reference.

| Step | CPU cycles |
|---|---:|
| load one tile (256 words) | 2,309 (9.0 / word) |
| load X (64 words) | 580 (9.1 / word) |
| read results (32 words) | 211 |
| go → done, b = 1 / 8 / 16 / 24 | 35 / 36 / 49 / 57 |
| 8 back-to-back b = 8 passes | 144 (engine bound 69; CPU loop bound) |

With the CPU moving data, a pass costs about 25× more cycles than it computes; this is what
the R5 front-end removes.

**Board** (Nexys A7-100T, 15 MHz, `bitstreams/xheep_nexys_bscan_pdm_cim.bit`, sha256
`21e501a2…0c77`). The full SoC uses 36,997 LUT (58 %), 36,903 FF, 52 BRAM tiles and 129 DSP. The
15 MHz domain meets +15.6 ns setup and +0.222 ns hold. The only failing paths are the asynchronous
CDC paths of the unused SPI-slave clock, as in the earlier bitstreams.
`cim_smoke` passes: 16 passes, 0 errors, every cycle count in the table identical to Verilator.
`hopcim_mfcc` (R4b) passes: 49 frames bit-exact. Every per-stage cycle count is identical to
Verilator; tile loading differs by 1 cycle. (A later rebuild of the firmware, with the test vector
as global arrays, gives 146,364 cycles/frame, identical on the board and in Verilator.) The R5a
bitstream with the hardware front-end: see `R5_FRONTEND_PLAN.md` §5.
