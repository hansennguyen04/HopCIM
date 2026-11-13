# Paper figures

Every number in these figures is read from the frozen evidence (board logs in `freeze/board/logs/`, model and
energy results in `model/`, synthesis reports in `syn/`), never typed in. `figures_data.json` holds each value
with the file it came from and the pattern or key that finds it; `freeze/freeze.py check` fails if it no longer
equals what the evidence gives.

```bash
~/venvs/defines/bin/python figures/make_figures.py   # rebuild figures_data.json and figures/out/*
python3 figures/figdata.py --check                   # the data against the evidence (standard library only)
```

`out/<name>.pdf` is for the paper (one IEEE column, 3.5 in, no title, TrueType fonts embedded, no creation date,
so a rebuild is byte-identical); `out/<name>.png` has a title, for reading.

| Figure | Shows | Values from | Claims |
|---|---|---|---|
| `fig1_cpu_active_per_hop` | CPU-active cycles per 20-ms hop, configurations A–E of `docs/EVALUATION.md` §2: 48.1 M → 13.2 M → 7,322 → 4,447 → 263 | board logs: CPU census, `hopcim_cpu`, `hopcim_mfcc`, `hopcim_net1`, `hopcim_fe_irq`, `hopcim_seq`, `hopcim_stream`, `hopcim_live` | B1, B2, B4, C1 |
| `fig2_network_cycles` | Network cycles per inference on the board: CPU only 47,974,801; CPU drives the CIM 13,020,998; sequencer R6b 85,420; S 67,909; B1 67,885 | board logs | B1, B2, B3, S7, U4 |
| `fig3_energy_breakdown` | Network energy per inference [E] by component: R6b 1,389, S 1,121, B1 1,094 nJ; four-block B 938 nJ from the schedule model, not built | `model/streamfuse_energy.json`, `model/energy_b1_rtl.json` | F1, S2, S5, U2 |
| `fig4_b1_vs_s` | B1 / S for cycles, activation-memory reads and writes, storage, energy, power at 40 ms, LUT, FF | `model/streamfuse_results.json`, `model/energy_b1_rtl.json`, `syn/` reports | U1, U2, U3 |

Draft captions:

1. *CPU-active cycles per 20-ms hop with one decision every hop, measured on the board. A: CPU only (front-end from
   the CPU census). B: the CPU drives the CIM's passes. C: hardware front-end and layer sequencer, composed from
   measured parts. D: the per-window normalisation in hardware. E: the microphone path in hardware; the CPU wakes
   only to read the decision.*
2. *Network cycles per inference, measured on the board (go to interrupt for the sequencer builds; the CPU-only
   and CPU-driven runs at 15 MHz, the sequencer builds at 15 and 15.36 MHz, which does not change a cycle count).*
3. *Estimated network energy per inference at 28 nm [E], from the RTL's exact activity counters and published
   tools, not from FPGA power. R6b's counts equal the schedule model's; four-block StreamFuse-B is the model's
   prediction, not built.*
4. *B1 (StreamFuse-B in block 1) against S built from the same source. Access counts are the schedule model's,
   which equal the RTL counters (checked in `figdata.py`); energy and power are estimates [E].*

Checks inside `figdata.py`: the energy components of each bar sum to the frozen totals (1,389.3, 1,120.8,
1,094.1 nJ) and four-block B is 0.837 × S; B1's activation reads and both designs' byte writes from the model
equal the RTL counters (`model/energy_counts_cim_{s,b1}.log`, 101 decisions); D and E equal `EVALUATION.md`.

Colours: the first four slots of the data-viz reference palette, in its validated order (blue, orange, aqua,
yellow). Node was not available here, so the palette validator was not re-run; the order is the palette's own
documented passing order for adjacent pairs. Figure 3 labels each wide segment and has a legend, so no value
depends on colour alone.
