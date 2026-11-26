# The X-HEEP integration as a patch series

HopCIM runs inside X-HEEP. Everything HopCIM changes or adds in X-HEEP is kept here, in this
repository, and an X-HEEP tree is *built* from it rather than edited by hand:

| | |
|---|---|
| `base.env` | upstream X-HEEP URL and commit (`b3eb3c8`), the expected result tree, the generator config |
| `patches/0001-*.patch` | build environment: SiFive GCC 10.1 multilib flags, Verilator 5.040 version parsing and a lint waiver, `XHEEP_NO_TRACE=1` |
| `patches/0002-*.patch` | Nexys A7 board: debug over the on-board USB (BSCANE2 TCK on a BUFG, `bscan.xdc`), the on-board PDM microphone pins, `configs/kws_nexys.py`, OpenOCD configs |
| `patches/0003-*.patch` | the HopCIM endpoint behind the FuseSoC flags `use_cim`, `use_cim_ded`, `use_cim_mic`, `use_cim_mic5`, `use_cim_s`, `use_cim_mic4`: the FPGA wrapper, the Verilator testharness (templates), the `x_heep_system` microphone ports, the clock wizards (15.36, 5.12, 8.192 MHz) |
| `patches/0004-*.patch` | the flag `use_cim_b1`: StreamFuse B in block 1 (`docs/STREAMFUSE_B1.md`), in the wrapper and the testharness |
| `patches/0005-*.patch` | `configs/kws_nexys_256k.py` (eight 32 KiB RAM banks instead of four, for the CPU-only TensorFlow Lite Micro baseline; the default configuration is unchanged) and C++17 for the firmware (TensorFlow Lite Micro needs it) |
| `patches/0006-*.patch` | the flag `use_cim_dwu`: the digital depthwise unit in every depthwise layer (`docs/DWU_GATES.md`), in the wrapper and the testharness; the firmware fills its store (`-DCIM_DWU=4`) |
| `patches/0007-*.patch` | the flag `use_cim_net2`: the second network's sequencer `cim_gseq` (MLPerf Tiny's ResNet8, `docs/NET2_GATES.md`), the endpoint's `SEQ_NET2 = 1`, in the wrapper and the testharness (build variants `net2-256`, `sim-net2-256`) |
| `patches/0008-*.patch` | `configs/kws_nexys_256k_c180.py`: the 256 KiB configuration with a 180 KiB code section (data 76 KiB), for the second network's CPU-only TensorFlow Lite Micro baseline (`mlperf/build_tflm_resnet.sh`); only the linker scripts differ from `kws_nexys_256k.py`'s output, so its bitstreams run the firmware |
| `generated.sha256` | what `make mcu-gen X_HEEP_CFG=configs/kws_nexys.py` writes in a fresh tree (97 files) |
| `generated.kws_nexys_256k.sha256` | the same for `configs/kws_nexys_256k.py` |
| `setup.sh` | builds a tree: clone at the commit, `git am` the series (8 patches), check the result tree, link this repository in, run the generator (default configuration, or the one given as a second argument, recorded in `<tree>/.hopcim-cfg`), check |
| `check.sh` | checks a tree against the series, the generator output and the links, and reports any other change |
| `build.sh` | bitstreams (`s`, `s4`, `b1`, `b1-4`, `dwu`, `dwu-4`, `r6b`, `r6b5`; `s256` needs a 256 KiB tree), Verilator SoC models (`sim`, `sim-ded`, `sim-s`, `sim-b1`, `sim-dwu`), firmware (`app`) |

The firmware is in `../sw/applications/` (22 apps). `setup.sh` links it into the tree file by file,
so the apps' relative includes (`../../../hw/ip/hopcim/sw/cim.h`) resolve inside the X-HEEP tree.
`hw/ip/hopcim` is a link to this repository; FuseSoC finds `hopcim.core` through it.

## Vendored directories

Three of the changed files lie in directories X-HEEP vendors from other repositories. A re-vendor
(`util/vendor.py --update`) replaces such a directory with upstream and then applies the patches
in its `patch_dir`. The series therefore also adds those patches and the `patch_dir` entries, so a
re-vendor keeps the HopCIM changes:

| Vendored directory | Patches (in X-HEEP) |
|---|---|
| `hw/fpga/xheep_fpga_support` | `hw/vendor/patches/xheep_fpga_support/0001-*.patch` (bscan.xdc, microphone pins), `0002-*.patch` (clock filesets and scripts); `patch_dir` added to `xheep_fpga_support.vendor.hjson` |
| `hw/vendor/pulp_platform/riscv_dbg` | `hw/vendor/patches/pulp_platform_riscv_dbg/0002-bscane-tck-on-a-bufg.patch` (after X-HEEP's own 0001) |
| `hw/vendor/xheep/spi` | `hw/vendor/patches/xheep_spi/0001-*.patch` (lint waiver); `patch_dir` added to `spi.vendor.hjson` |

Checked: each patch directory, applied to the base commit's vendored copy the way `vendor.py`
does (`git apply --directory <dir> -p1`), reproduces the patched directory exactly. A real
re-vendor needs the network and was not run.

## Use

```bash
# a new tree (outside this repository); XHEEP_VENV avoids `make venv`, which needs the network
XHEEP_VENV=~/APlus_Project/planenest/hw/x-heep/.venv xheep/setup.sh ~/xheep-hopcim
xheep/build.sh ~/xheep-hopcim s                     # bitstream of the baseline S, ~50 min
xheep/build.sh ~/xheep-hopcim app hopcim_seq_nexys_15m36 hopcim_seq -DREFERENCE_CLOCK_Hz=15360000
xheep/check.sh                                      # the working tree, planenest/hw/x-heep
# the 256 KiB variant of S, for the CPU-only TensorFlow Lite Micro baseline
XHEEP_VENV=~/APlus_Project/planenest/hw/x-heep/.venv xheep/setup.sh ~/xheep-256k configs/kws_nexys_256k.py
xheep/build.sh ~/xheep-256k s256
```

`setup.sh` needs the tools of `planenest/hw/xheep_env.sh` (RISC-V GCC 10.1, Verilator 5.040,
Vivado 2024.1, verible-verilog-format; set `XHEEP_ENV` to use another script). The generator's
output depends on verible's formatting, so `generated.sha256` holds for this tool set.

## Changing X-HEEP

Edit in a tree made by `setup.sh` (branch `hopcim`, one commit per patch), commit, then regenerate
the series and the recorded tree:

```bash
cd <tree> && git commit --amend      # or a new commit
git format-patch -q --no-signature --zero-commit -o <hopcim>/xheep/patches b3eb3c8..hopcim
git rev-parse 'HEAD^{tree}'          # -> XHEEP_SERIES_TREE in base.env
```

If a change touches a vendored directory, regenerate its `hw/vendor/patches/` entry too. A change
to a generator template or config needs a new `generated.sha256` (`HOPCIM_RECORD=1 setup.sh` on a
fresh tree). `check.sh` fails on anything the series does not explain.

## Why the series replaced the hand-edited tree (2026-11-08)

Until this date the working tree held the changes directly. Rebuilding it from the frozen patch
showed two files that existed only as hand edits of *generated* output: `tb/testharness.sv` and
`tb/testharness_pkg.sv` (the endpoint in the Verilator SoC). `make mcu-gen` overwrites both, and
the freeze patch (`git diff` of tracked files) did not capture them, because X-HEEP's git ignores
them. Patch 0003 moves those lines into the templates. A third generated file,
`hw/system/x_heep_system.sv`, regenerates from the already patched template with one `assign`
statement in a different place (same logic). The firmware lived only in X-HEEP's untracked
`sw/applications/`, and now lives in this repository.
