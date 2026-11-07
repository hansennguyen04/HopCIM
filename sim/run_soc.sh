#!/usr/bin/env bash
# Build an X-HEEP app (default cim_smoke) and run it on the Verilator model with the CIM endpoint;
# passes if the UART log has a "... -> PASS" line.   usage: [SIMVAR=cim|ded] run_soc.sh [app]
# Models are self-contained binaries kept outside FuseSoC's work directory (a rebuild wipes it):
#   sim/bin/Vtestharness_cim   make verilator-build FUSESOC_FLAGS=--flag=use_cim
#   sim/bin/Vtestharness_ded   make verilator-build "FUSESOC_FLAGS=--flag=use_cim --flag=use_cim_ded"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XH="$HOME/APlus_Project/planenest/hw/x-heep"
SIMVAR="${SIMVAR:-cim}"
BIN="$ROOT/sim/bin/Vtestharness_$SIMVAR"
WORK="$ROOT/sim/work_$SIMVAR"
APP="${1:-cim_smoke}"
LOG="$ROOT/sim/soc_${APP}_$SIMVAR"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
cd "$XH"
source .venv/bin/activate
echo "[app] $APP TARGET=sim"
make app PROJECT="$APP" TARGET=sim > "$LOG.app.log" 2>&1 || { tail -20 "$LOG.app.log"; exit 1; }
mkdir -p "$WORK"
cp sw/build/main.hex "$WORK/main.hex"
cd "$WORK"
echo "[run] $(basename "$BIN") (no waveform)"
XHEEP_NO_TRACE=1 timeout 7200 "$BIN" +firmware=main.hex > "$LOG.run.log" 2>&1 || true
cp uart0.log "$LOG.uart.log"
cat "$LOG.uart.log"
tail -3 "$LOG.run.log"
grep -q -- "-> PASS" "$LOG.uart.log"
