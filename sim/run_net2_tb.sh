#!/usr/bin/env bash
# Build and run the ResNet8 bench (cim_gseq, docs/NET2_GATES.md). Vectors in tb/net2_vec are for 4 images (the gate uses 20:
# python model/gen_net2_vectors.py 20).
#   run_net2_tb.sh [NIMG] [IMG0]        env DWU=4: with the digital depthwise unit also in the endpoint
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
NIMG="${1:-4}"; IMG0="${2:-0}"; DWU="${DWU:-0}"; TAG="net2_d$DWU"; OBJ="$ROOT/sim/obj_net2_d$DWU"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_gseq.sv" "$ROOT/rtl/cim_endpoint.sv")
if [ ! -x "$OBJ/Vtb_cim_net2" ] || [ "${REBUILD:-0}" = 1 ]; then
  echo "[lint] verilator -Wall cim_endpoint, SEQ_NET2"
  verilator --lint-only -Wall -I"$ROOT/rtl" -GSEQ_S=1 -GSEQ_DWU=$DWU -GSEQ_NET2=1 "$PKG" "${RTL[@]}" --top-module cim_endpoint
  echo "[build] testbench"
  verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_net2 -GDWU=$DWU -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
    -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_net2.sv" > "$ROOT/sim/logs/build_net2_d$DWU.log" 2>&1 \
    || { tail -30 "$ROOT/sim/logs/build_net2_d$DWU.log"; exit 1; }
fi
cd "$ROOT"
"$OBJ/Vtb_cim_net2" +NIMG=$NIMG +IMG0=$IMG0 | tee "$ROOT/sim/logs/net2_tb_d$DWU.log"
grep -q "^PASS" "$ROOT/sim/logs/net2_tb_d$DWU.log"
