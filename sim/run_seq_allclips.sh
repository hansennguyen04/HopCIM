#!/usr/bin/env bash
# Every GSC test clip (4,890) through the layer sequencer in cim_endpoint: logits and class against the
# golden model, Verilator 5.040. About 15 min.
#   run_seq_allclips.sh [s]      s: the baseline S in the sequencer (SEQ_S = 1)
# Vectors: python model/gen_seq_vectors.py (tiles, parameters) and model/gen_seq_allclips.py (tb/seq_vec_all).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
S=0; SUF=""; [ "${1:-}" = s ] && { S=1; SUF=_s; }
OBJ="$ROOT/sim/obj_seq_all$SUF"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_seq -GS=$S +define+ALLCLIPS -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_seq.sv" > "$ROOT/sim/build_seq_all$SUF.log" 2>&1 \
  || { tail -30 "$ROOT/sim/build_seq_all$SUF.log"; exit 1; }
cd "$ROOT"
LOG="$ROOT/sim/seq_allclips$SUF.log"
{ echo "vectors:"; sed 's/^/  /' tb/seq_vec_all/SHA256SUMS; } > "$LOG"
"$OBJ/Vtb_cim_seq" | tee -a "$LOG" | grep -E "clips|RESULT|PASS|FAIL|MISMATCH" | head -40
grep -q "^PASS" "$LOG"
