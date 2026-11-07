#!/usr/bin/env bash
# Lint and simulate cim_engine with Verilator 5.040 (from xheep_env.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
OBJ="$ROOT/sim/obj_engine"
echo "[lint] verilator -Wall rtl/cim_engine.sv"
verilator --lint-only -Wall "$ROOT/rtl/cim_engine.sv" --top-module cim_engine
echo "[build] testbench"
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_engine -Mdir "$OBJ" \
  -MAKEFLAGS "OBJCACHE=" "$ROOT/rtl/cim_engine.sv" "$ROOT/tb/tb_cim_engine.sv" > "$ROOT/sim/build_engine.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_engine.log"; exit 1; }
echo "[run]"
"$OBJ/Vtb_cim_engine" | tee "$ROOT/sim/engine_tb.log"
grep -q "^PASS" "$ROOT/sim/engine_tb.log"
