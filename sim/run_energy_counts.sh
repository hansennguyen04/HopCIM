#!/usr/bin/env bash
# Exact activity counts for the energy estimate: the live path bench (tb_cim_mic) with the counters of
# tb/energy_counters.sv bound in. It must still PASS, so the counts come from a correct run.
#   run_energy_counts.sh [cim|ded] [PDM divider, default 5] [k, default 1] [s: the baseline S, SEQ_S = 1]
# Writes model/energy_counts_<mode>[_k<k>].log. Vectors: as for run_mic_tb.sh.
set -euo pipefail
S=0; SUF=""; ARGS=(); for a in "$@"; do if [ "$a" = s ]; then S=1; SUF=_s; else ARGS+=("$a"); fi; done   # "s": SEQ_S = 1
set -- ${ARGS[@]+"${ARGS[@]}"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
MODE="${1:-cim}"
DED=0; [ "$MODE" = ded ] && DED=1
DIV="${2:-5}"
K="${3:-1}"
TAG="$MODE"; [ "$K" != 1 ] && TAG="${TAG}_k$K"; TAG="$TAG$SUF"
OBJ="$ROOT/sim/obj_ec_$TAG"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_mic -GDED=$DED -GCLKDIV=$DIV -GNET_K=$K -GS=$S -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_mic.sv" "$ROOT/tb/energy_counters.sv" > "$ROOT/sim/build_ec_$TAG.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_ec_$TAG.log"; exit 1; }
cd "$ROOT"
"$OBJ/Vtb_cim_mic" > "$ROOT/model/energy_counts_$TAG.log"
grep -E "^(PASS|FAIL|RESULT|decisions)" "$ROOT/model/energy_counts_$TAG.log"
grep -q "^PASS" "$ROOT/model/energy_counts_$TAG.log"
