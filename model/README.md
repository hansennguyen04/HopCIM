# Models: golden references, schedule and energy models, vector generators

Scripts and the results they write live side by side here, on purpose: the frozen gate documents (`docs/*_GATES.md`)
cite several of these files by path (`dwu.py`, `dwu_prediction.json`, `energy_est.py`, `energy_counts_cim.log`,
`energy_s_rtl.json`, `resnet_int.py`, `resnet_prediction.json`, `resnet_sched.py`), and the freeze check and the
figures read many more. So nothing here is moved; this index groups them instead. Each script's docstring gives its
command line. Scripts import one another by name (run them from this folder or as `python model/<script>.py`).

Python environments: most scripts need only NumPy (`~/venvs/planenest`); the checks against TFLite and the parameter
exports need TensorFlow (`~/venvs/kwsgate`); `energy_tool_run.py` needs the AccelForge environment
(`~/APlus_Project/recim-rv/accelforge_pilot/.venv312`).

## Golden models (integer references the RTL must equal)

| Script | Model | Outputs |
|---|---|---|
| `mfcc_int.py` | R4a: the CIM-MFCC front-end as a pure integer pipeline | `mfcc_tables.npz` (via `make_tables.py`) |
| `mfcc_ded.py` | B3: the dedicated MFCC front-end | |
| `cim_map.py` | the front-end computed only through CIM passes | `cim_map_full.log` |
| `stream_int.py` | R5b: the streaming window path | |
| `pdm_dec.py` | R7: the hardware PDM decimator | |
| `dscnn_int.py` | MLPerf Tiny's int8 DS-CNN, exact against TFLite's reference kernels | `dscnn_params.npz`, `dscnn_check.log` |
| `cim_net_map.py` | the DS-CNN computed only through CIM passes | `cim_net_map.log` |
| `resnet_int.py` | MLPerf Tiny's int8 ResNet8 (second network) | `resnet_params.npz`, `resnet_int_check.log` |
| `resnet_map.py` | ResNet8 on the CIM: tiles, exact pass-level execution, cycles | |

## Schedule, cycle and design-study models

| Script | Study (gates) | Outputs |
|---|---|---|
| `streamfuse_sched.py` | StreamFuse (`STREAMFUSE_GATES.md`) | `streamfuse_*.json`, `streamfuse_run.log` |
| `wholeword.py`, `rule_sweep.py` | whole-word decisions (`WHOLEWORD_GATES.md`), rate and confirmation rule | `wholeword_*`, `rule_sweep_*` |
| `dwswap.py` | depthwise swap-in (`DWSWAP_GATES.md`) | `dwswap_results.json`, `dwswap_run.log` |
| `dwu.py` | the digital depthwise unit (`DWU_GATES.md`); `dwu.py rtl` reprices from RTL counters | `dwu_prediction.json`, `dwu_rtl.json`, `dwu_rtl.log` |
| `resnet_sched.py` | ResNet8 on `cim_gseq` (`NET2_GATES.md`) | `resnet_prediction.json` |
| `exact_reuse.py` | exact activation reuse (software only) | `reuse_*` |
| `timing_sweep.py` | CIM timing sensitivity | `timing_results.json` |

## Energy [E] (`docs/ENERGY_ESTIMATE.md`)

| Script | Role | Outputs |
|---|---|---|
| `energy_tool_run.py` | published-tool numbers (AccelForge, CACTI) | `energy_components.json` |
| `energy_activity.py` | bit activity of the CIM passes on real audio | `energy_activity.json` |
| `energy_est.py` | the energy model | `energy_results.json`, `energy_s_rtl.json`, `energy_b1_rtl.json`, `energy_run.log` |
| `zigzag_run/`, `defines_run/` | ZigZag-IMC and DeFiNES run on HopCIM's macro, as cross-checks | `zigzag_results.json`, `streamfuse_defines.json` |
| (`sim/run_energy_counts.sh`) | RTL activity counters | `energy_counts_*.log` |

## Generators (tables for the firmware, vectors for the benches)

| Script | Writes |
|---|---|
| `gen_c.py`, `gen_net.py` | `sw/mfcc_tables.h`, `sw/dscnn_tables.h` |
| `gen_fe_vectors.py`, `gen_seq_vectors.py`, `gen_seq_allclips.py`, `gen_stream_vectors.py`, `gen_mic_vectors.py` | `tb/*_vec*/` |
| `gen_net2_vectors.py`, `gen_resnet_tables.py` | `tb/net2_vec/`, `sw/resnet_tables.h`, `sw/applications/hopcim_resnet/resvec.h` |

## Accuracy gate, board-run analysis, verification

| Script | Role | Outputs |
|---|---|---|
| `gate2_int.py` | accuracy of the unchanged DS-CNN fed by an integer front-end | `gate2_*` |
| `check_micrec.py` | checks a board recording of the live path | `micrec_4m096_check.log` |
| `score_kwtest.py`, `test_kw_rule.py` | spoken-keyword test scoring; the firmware's summed-logit rule | |
| `kwlog_capture.py`, `kwlog_frames.py`, `kwlog_analyse.py` | same-audio logger runs (`docs/SAME_AUDIO.md`) | `kwlog_checks/` |
| `verif_mutants.py`, `verif_report.py` | mutation test and assertion/coverage report (`docs/VERIFICATION.md`) | `verif_*.json` |
