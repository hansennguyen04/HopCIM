#!/usr/bin/env bash
# Lint and simulate the layer sequencer in cim_endpoint against golden vectors, Verilator 5.040.
#   run_seq_tb.sh        with cim_fe        run_seq_tb.sh ded   with B3 mfcc_ded (FE_DED = 1)
#   a further argument s: the baseline S in the sequencer (SEQ_S = 1), logs and objects with suffix _s
# Vectors: python model/gen_seq_vectors.py.
set -euo pipefail
S=0; SUF=""; ARGS=(); for a in "$@"; do if [ "$a" = s ]; then S=1; SUF=_s; else ARGS+=("$a"); fi; done   # "s": SEQ_S = 1
set -- ${ARGS[@]+"${ARGS[@]}"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
MODE="${1:-cim}"
DED=0; [ "$MODE" = ded ] && DED=1
OBJ="$ROOT/sim/obj_seq_$MODE$SUF"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
echo "[lint] verilator -Wall cim_endpoint, FE_DED=$DED SEQ_S=$S"
verilator --lint-only -Wall -I"$ROOT/rtl" -GFE_DED=$DED -GSEQ_S=$S "$PKG" "${RTL[@]}" --top-module cim_endpoint
echo "[build] testbench"
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_seq -GDED=$DED -GS=$S -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_seq.sv" > "$ROOT/sim/build_seq_$MODE$SUF.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_seq_$MODE$SUF.log"; exit 1; }
echo "[run]"
cd "$ROOT"
"$OBJ/Vtb_cim_seq" | tee "$ROOT/sim/seq_tb_$MODE$SUF.log"
grep -q "^PASS" "$ROOT/sim/seq_tb_$MODE$SUF.log"
