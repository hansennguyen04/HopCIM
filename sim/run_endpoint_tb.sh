#!/usr/bin/env bash
# Lint and simulate cim_endpoint (with cim_engine) through its OBI port, Verilator 5.040.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
OBJ="$ROOT/sim/obj_endpoint"
echo "[lint] verilator -Wall rtl/cim_endpoint.sv"
verilator --lint-only -Wall -I"$ROOT/rtl" "$PKG" "$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv" --top-module cim_endpoint
echo "[build] testbench"
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_endpoint -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv" "$ROOT/tb/tb_cim_endpoint.sv" \
  > "$ROOT/sim/build_endpoint.log" 2>&1 || { tail -30 "$ROOT/sim/build_endpoint.log"; exit 1; }
echo "[run]"
"$OBJ/Vtb_cim_endpoint" | tee "$ROOT/sim/endpoint_tb.log"
grep -q "^PASS" "$ROOT/sim/endpoint_tb.log"
