#!/usr/bin/env bash
# Lint and simulate the hardware front-end in cim_endpoint against golden vectors, Verilator 5.040.
#   run_fe_tb.sh        cim_fe (vectors tb/fe_vec)       run_fe_tb.sh ded   B3 mfcc_ded (tb/ded_vec)
# Vectors: python model/gen_fe_vectors.py [ded].
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
MODE="${1:-cim}"
DED=0; [ "$MODE" = ded ] && DED=1
OBJ="$ROOT/sim/obj_fe_$MODE"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
echo "[lint] verilator -Wall cim_endpoint, FE_DED=$DED"
verilator --lint-only -Wall -I"$ROOT/rtl" -GFE_DED=$DED "$PKG" "${RTL[@]}" --top-module cim_endpoint
echo "[build] testbench"
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_fe -GDED=$DED -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_fe.sv" > "$ROOT/sim/build_fe_$MODE.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_fe_$MODE.log"; exit 1; }
echo "[run]"
cd "$ROOT"
"$OBJ/Vtb_cim_fe" | tee "$ROOT/sim/fe_tb_$MODE.log"
grep -q "^PASS" "$ROOT/sim/fe_tb_$MODE.log"
