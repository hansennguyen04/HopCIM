#!/usr/bin/env bash
# Lint and simulate the streaming window path (R5b) end to end in cim_endpoint, Verilator 5.040.
#   run_stream_tb.sh     with cim_fe        run_stream_tb.sh ded   with B3 mfcc_ded (FE_DED = 1)
#   a second argument "edge" runs the extreme stream (tb/stream_vec_edge_*; model/gen_stream_vectors.py edge)
# Vectors: python model/gen_fe_vectors.py, gen_seq_vectors.py, gen_stream_vectors.py.
set -euo pipefail
S=0; B1=0; SUF=""; ARGS=(); for a in "$@"; do if [ "$a" = s ]; then S=1; SUF=_s; elif [ "$a" = b1 ]; then S=1; B1=1; SUF=_b1; else ARGS+=("$a"); fi; done   # "s": SEQ_S = 1; "b1": also SEQ_B1 = 1
set -- ${ARGS[@]+"${ARGS[@]}"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
MODE="${1:-cim}"
EDGE="${2:-}"
DED=0; [ "$MODE" = ded ] && DED=1
OBJ="$ROOT/sim/obj_stream_$MODE$EDGE$SUF"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
echo "[lint] verilator -Wall cim_endpoint, FE_DED=$DED"
verilator --lint-only -Wall -I"$ROOT/rtl" -GFE_DED=$DED -GSEQ_S=$S -GSEQ_B1=$B1 "$PKG" "${RTL[@]}" --top-module cim_endpoint
echo "[build] testbench"
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_stream -GDED=$DED -GS=$S -GB1=$B1 -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_stream.sv" > "$ROOT/sim/build_stream_$MODE$EDGE$SUF.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_stream_$MODE$EDGE$SUF.log"; exit 1; }
echo "[run]"
cd "$ROOT"
if [ "$EDGE" = edge ]; then LOG="$ROOT/sim/stream_tb_${MODE}_edge$SUF.log"; "$OBJ/Vtb_cim_stream" "+VEC=tb/stream_vec_edge_$MODE" | tee "$LOG"
else LOG="$ROOT/sim/stream_tb_$MODE$SUF.log"; "$OBJ/Vtb_cim_stream" | tee "$LOG"; fi
grep -q "^PASS" "$LOG"
