#!/usr/bin/env bash
# Lint and simulate the live audio path (R7) end to end in cim_endpoint, Verilator 5.040.
#   run_mic_tb.sh        with cim_fe        run_mic_tb.sh ded      with B3 mfcc_ded (FE_DED = 1)
#   a second argument sets the PDM clock divider (default 15, i.e. 15.36 MHz; 5 for 5.12 MHz)
#   a third argument runs the network every k hops (default 1, range 1..16)
#   a further argument s: the baseline S in the sequencer (SEQ_S = 1); b1: S with StreamFuse B in block 1 (SEQ_B1); 4 (4.096 MHz) needs it
#   HOPMON=1 binds tb/hop_monitor.sv (the busy time of every hop; the worst one sets the lowest clock)
# Vectors: python model/gen_fe_vectors.py, gen_seq_vectors.py, gen_mic_vectors.py.
set -euo pipefail
S=0; B1=0; SUF=""; ARGS=(); for a in "$@"; do if [ "$a" = s ]; then S=1; SUF=_s; elif [ "$a" = b1 ]; then S=1; B1=1; SUF=_b1; else ARGS+=("$a"); fi; done   # "s": SEQ_S = 1; "b1": also SEQ_B1 = 1
set -- ${ARGS[@]+"${ARGS[@]}"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
MODE="${1:-cim}"
DED=0; [ "$MODE" = ded ] && DED=1
DIV="${2:-15}"
K="${3:-1}"
TAG="$MODE"; [ "$DIV" != 15 ] && TAG="${TAG}_div$DIV"; [ "$K" != 1 ] && TAG="${TAG}_k$K"; TAG="$TAG$SUF"
EXTRA=(); [ "${HOPMON:-0}" = 1 ] && { EXTRA=("$ROOT/tb/hop_monitor.sv"); TAG="${TAG}_hm"; }
OBJ="$ROOT/sim/obj_mic_$TAG"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
echo "[lint] verilator -Wall cim_endpoint, FE_DED=$DED, PDM_CLKDIV=$DIV"
verilator --lint-only -Wall -I"$ROOT/rtl" -GFE_DED=$DED -GPDM_CLKDIV=$DIV -GSEQ_S=$S -GSEQ_B1=$B1 "$PKG" "${RTL[@]}" --top-module cim_endpoint
echo "[build] testbench"
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_mic -GDED=$DED -GCLKDIV=$DIV -GNET_K=$K -GS=$S -GB1=$B1 -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_mic.sv" ${EXTRA[@]+"${EXTRA[@]}"} > "$ROOT/sim/build_mic_$TAG.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_mic_$TAG.log"; exit 1; }
echo "[run]"
cd "$ROOT"
"$OBJ/Vtb_cim_mic" | tee "$ROOT/sim/mic_tb_$TAG.log"
grep -q "^PASS" "$ROOT/sim/mic_tb_$TAG.log"
