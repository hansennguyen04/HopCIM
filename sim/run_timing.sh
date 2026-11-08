#!/usr/bin/env bash
# Timing sensitivity study (docs/TIMING_SENSITIVITY.md): builds a bench with the behavioural engine
# tb/cim_engine_tm.sv in place of rtl/cim_engine.sv. The synthesised RTL is not changed.
#   run_timing.sh build <engine|fe|seq|stream> [cim|ded]    build sim/obj_tm_<bench>_<mode>
#   run_timing.sh build mic [cim|ded] [pdm divider] [k]     build sim/obj_tm_mic_<mode>_div<N>[_k<k>]
# Then run the binary from hopcim/ with plusargs, e.g.
#   sim/obj_tm_seq_cim/Vtb_cim_seq +CIM_DMIN=9 +CIM_LAT=5
# HOLD=1 builds with the sequencer's input-bank hold following the plusarg +CIM_HOLD (see below).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
[ "${1:-}" = build ] || { sed -n 2,9p "$0"; exit 1; }
BENCH="$2"; MODE="${3:-cim}"; DIV="${4:-15}"; K="${5:-1}"; HOLD="${HOLD:-0}"
DED=0; [ "$MODE" = ded ] && DED=1
SEQ="$ROOT/rtl/cim_seq.sv"
SUF=""
if [ "$HOLD" = 1 ]; then
  # the sequencer's input-bank hold is a 3-bit constant tuned to the RTL's timing; a slower macro needs
  # a longer one. This derives a simulation-only copy with a run-time value; the RTL stays as it is.
  mkdir -p "$ROOT/sim/work_tm"
  SEQ="$ROOT/sim/work_tm/cim_seq_hold.sv"
  python3 - "$ROOT/rtl/cim_seq.sv" "$SEQ" <<'PY'
import sys
s = open(sys.argv[1]).read()
n = 0
for a, b in (("bcnt[i_b]  <= 3'd7;", "bcnt[i_b]  <= 16'(hold_cfg);"),
             ("bcnt[b] != 3'd0", "bcnt[b] != 16'd0"),
             ("bcnt[b] <= bcnt[b] - 3'd1;", "bcnt[b] <= bcnt[b] - 16'd1;"),
             ("bcnt[0] <= 3'd0;", "bcnt[0] <= 16'd0;"),
             ("bcnt[1] <= 3'd0;", "bcnt[1] <= 16'd0;"),
             ("(bcnt[g_b] == 3'd0)", "(bcnt[g_b] == 16'd0)")):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
    n += 1
import re
m = re.search(r"logic\s*\[2:0\]\s*bcnt\s*\[2\]\s*;[^\n]*\n", s)
assert m, "bcnt declaration"
s = s[:m.start()] + "logic [15:0] bcnt [2];\n  int unsigned hold_cfg = 7;\n  initial if ($value$plusargs(\"CIM_HOLD=%d\", hold_cfg)) begin end\n" + s[m.end():]
import os
tmp = sys.argv[2] + f".{os.getpid()}"
open(tmp, "w").write(s)
os.replace(tmp, sys.argv[2])                         # parallel builds must never read a half-written file
PY
  SUF="_hold"
fi
RTL=("$ROOT/tb/cim_engine_tm.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$SEQ" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_endpoint.sv")
case "$BENCH" in
  engine) TOP=tb_cim_engine; OBJ="obj_tm_engine"; EXTRA=(); RTL=("$ROOT/tb/cim_engine_tm.sv"); TB=("$ROOT/tb/tb_cim_engine.sv") ;;
  fe)     TOP=tb_cim_fe; OBJ="obj_tm_fe_$MODE"; EXTRA=(-GDED=$DED); TB=("$ROOT/tb/tb_cim_fe.sv") ;;
  seq)    TOP=tb_cim_seq; OBJ="obj_tm_seq_$MODE"; EXTRA=(-GDED=$DED); TB=("$ROOT/tb/tb_cim_seq.sv") ;;
  stream) TOP=tb_cim_stream; OBJ="obj_tm_stream_$MODE"; EXTRA=(-GDED=$DED); TB=("$ROOT/tb/tb_cim_stream.sv") ;;
  mic)    TOP=tb_cim_mic; OBJ="obj_tm_mic_${MODE}_div$DIV"; [ "$K" != 1 ] && OBJ="${OBJ}_k$K"
          EXTRA=(-GDED=$DED -GCLKDIV=$DIV -GNET_K=$K); TB=("$ROOT/tb/hop_monitor.sv" "$ROOT/tb/tb_cim_mic.sv") ;;
  *) echo "unknown bench $BENCH"; exit 1 ;;
esac
OBJ="$ROOT/sim/$OBJ$SUF"
LOG="$ROOT/sim/build_${OBJ##*/}.log"
verilator --binary --timing -Wno-fatal -j "${JOBS:-6}" -O2 --top-module "$TOP" ${EXTRA[@]+"${EXTRA[@]}"} -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" \
  -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "${TB[@]}" > "$LOG" 2>&1 || { tail -30 "$LOG"; exit 1; }
echo "$OBJ/V$TOP"
