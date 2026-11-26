# HopCIM

An always-on keyword-spotting microcontroller: X-HEEP's CV32E20 RISC-V core with a bit-serial digital
compute-in-memory (CIM) engine that runs MLPerf Tiny's published int8 DS-CNN, unchanged and bit-exact, with the
MFCC front-end, the layer sequencer and the live microphone path in hardware, so the CPU sleeps between decisions.
FPGA prototype on a Digilent Nexys A7-100T. The default design is **S+DWU** (the baseline S with the digital depthwise
unit); **S** is kept as the comparison baseline (`docs/DESIGNS.md`). Status, questions and plan:
`docs/HOPCIM_PROPOSAL.md`; results: `docs/EVALUATION.md`; all documents: `docs/README.md`.

## Layout

| Folder | What it holds |
|---|---|
| `rtl/` | the SystemVerilog design: `cim_engine`, `cim_endpoint` (OBI slave), front-ends (`cim_fe`, `mfcc_ded`), sequencers (`cim_seq`, `cim_gseq`), audio (`pdm_dec`, `cim_audio`) |
| `hopcim.core` | the FuseSoC core (RTL file list) X-HEEP builds through its `hw/ip/hopcim` link |
| `tb/` | benches (`tb_*.sv`), assertions, activity counters, and their vectors (`*_vec*/`) |
| `sim/` | bench scripts `run_*.sh`; `sim/logs/` their logs; `dwu_jobs/`, `net2_jobs/` long job outputs |
| `model/` | Python golden models, schedule and energy models, vector and table generators, and their results (`model/README.md`) |
| `sw/` | firmware for X-HEEP: `sw/applications/<app>/` and shared headers (`cim.h`, generated tables) |
| `syn/` | Vivado out-of-context scripts (`ooc.tcl`) and reports |
| `xheep/` | X-HEEP as an upstream commit plus a patch series: `setup.sh`, `check.sh`, `build.sh` (`xheep/README.md`) |
| `board/` | board scripts (`board_run.sh`, `board_suite.sh`, `board_khop.sh`, `board_kwlog.sh`) and the tool environment `xheep_env.sh` |
| `mlperf/` | TensorFlow Lite Micro CPU baselines: vendoring and build scripts |
| `third_party/` | TensorFlow Lite Micro headers (vendored, unchanged) |
| `figures/` | thesis figures: data, scripts, `out/` |
| `freeze/` | the results freeze of 2026-11-13: `freeze.py` (snapshot, manifest, check), `RESULTS.md`, copies of the board scripts and gate files |
| `docs/` | specifications, studies, frozen gates and results (`docs/README.md`) |

## Entry points

```bash
source ~/APlus_Project/planenest/hw/xheep_env.sh     # tools: RISC-V GCC, Verilator 5.040, Vivado 2024.1, X-HEEP venv
sim/run_all.sh                                       # the regression: 54 benches, logs in sim/logs/
xheep/setup.sh <dest> [<config>]; xheep/check.sh <tree>; xheep/build.sh <tree> <variant>   # X-HEEP trees, bitstreams, firmware
cd ~/APlus_Project/planenest/hw && ./board_run.sh <elf> <bitstream> [timeout_s]           # a board run
~/venvs/planenest/bin/python freeze/freeze.py check  # every frozen claim against its evidence
python3 figures/figdata.py --check                   # the figure data against the evidence
```

**Board scripts** are versioned here in `board/`, and run through the links of the same names in
`~/APlus_Project/planenest/hw/`: they find the X-HEEP tree, the bitstreams and their log folder next to the path they
are started from, so start them from there (as above), not from `board/`.

## Outside this repository (by path)

| Path under `~/APlus_Project/` | Used for |
|---|---|
| `planenest/hw/` | X-HEEP trees (`x-heep*`), bitstreams, board ELFs and logs, links to `board/` (the folder keeps the name of an earlier project) |
| `planenest/data/cifar-10-batches-py/` | CIFAR-10 test set (ResNet8) |
| `kws_gate/` | the accuracy gate and MLPerf Tiny's KWS models (read by `freeze.py`) |
| `mlperf_tiny/` | MLPerf Tiny's models, runner and the pinned tflite-micro checkout |
| `recim-rv/accelforge_pilot/` | AccelForge and CACTI for `model/energy_tool_run.py` |
| `research_discovery/20261026_application_driven/` | the round-5 report that chose HopCIM (copied by `freeze.py`) |
| `HopCIM_Illustrations/` | verified illustrations; its `run_all.sh` reads files of this repository by path |

## Pinned paths

The frozen gate documents cite files by path and their `.sha256` files record their own paths, and `freeze/freeze.py`
and `HopCIM_Illustrations` read many files by path. Before moving or renaming a file, search for its path in the
repository and in the folders above; never move a file that a `docs/*_GATES.md` cites.
