#!/usr/bin/env bash
# Lint and simulate one small unit bench with Verilator 5.040.
#   run_unit_tb.sh log2    cim_log2fix16 (tb/tb_cim_log2fix16.sv)
#   run_unit_tb.sh pdm     pdm_dec against its vectors (tb/tb_pdm_dec.sv, tb/pdm_vec; model/gen_mic_vectors.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
case "${1:?log2 or pdm}" in
  log2) TOP=tb_cim_log2fix16; RTL=("$ROOT/rtl/cim_log2fix16.sv") ;;
  pdm)  TOP=tb_pdm_dec;       RTL=("$ROOT/rtl/pdm_dec.sv") ;;
  *) echo "unknown bench $1" >&2; exit 2 ;;
esac
OBJ="$ROOT/sim/obj_$1"
verilator --lint-only -Wall -I"$ROOT/rtl" "${RTL[@]}" --top-module "$(basename "${RTL[0]}" .sv)" > "$ROOT/sim/lint_$1.log" 2>&1 \
  || { tail -30 "$ROOT/sim/lint_$1.log"; exit 1; }
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module "$TOP" -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "${RTL[@]}" "$ROOT/tb/$TOP.sv" > "$ROOT/sim/build_$1.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_$1.log"; exit 1; }
cd "$ROOT"
"$OBJ/V$TOP" | tee "$ROOT/sim/$1_tb.log"
grep -q "^PASS" "$ROOT/sim/$1_tb.log"
