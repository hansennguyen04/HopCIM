#!/usr/bin/env bash
# One same-audio or false-alarm run on the Nexys A7 (hopcim/docs/SAME_AUDIO.md). The UART stream goes
# to <out dir>/frames.bin, as received.
#   usage: board_kwlog.sh <program.elf> <out dir> [seconds, 0 = until the program's end frame]
# Needs the board connected, SW0 up, SW2 down, and no other OpenOCD or Hardware Manager on the cable.
# Ctrl-C stops the run. BIT overrides the bitstream.
set -uo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"; XH="$HW/x-heep"
ELF="$(readlink -f "$1")"; OUT="$(readlink -m "$2")"; SECS="${3:-0}"
BIT="$(readlink -f "${BIT:-$HW/bitstreams/xheep_nexys_bscan_cim128k.bit}")"
UART="${UART:-/dev/ttyUSB1}"
GDB_PORT="${GDB_PORT:-3333}"; TELNET_PORT="${TELNET_PORT:-4444}"; TCL_PORT="${TCL_PORT:-6666}"
source "$HW/xheep_env.sh" >/dev/null
CAP=""; OCD=""; GDB=""
trap 'kill $GDB $OCD $CAP 2>/dev/null; echo; echo "interrupted: $OUT"; exit 130' INT TERM
lsusb | grep -q "0403:6010" || { echo "Nexys A7 (FTDI 0403:6010) not found on USB"; exit 1; }
[ -e "$UART" ] || { echo "$UART missing (try UART=/dev/ttyUSB0)"; exit 1; }
mkdir -p "$OUT"
{ echo "elf $(basename "$ELF") $(sha256sum "$ELF" | cut -d' ' -f1)"
  echo "bit $(basename "$BIT") $(sha256sum "$BIT" | cut -d' ' -f1)"
  echo "start $(date -Is)"; } > "$OUT/run.txt"

echo "[1/3] programming $(basename "$BIT")"
( cd "$XH" && openocd -c "gdb_port disabled" -c "telnet_port disabled" -c "tcl_port disabled" \
    -f tb/nexys-a7-program-fpga.cfg -c "pld load 0 $BIT" -c shutdown ) > "$OUT/program.log" 2>&1 \
  || { echo "programming failed, see $OUT/program.log"
       grep -q LIBUSB_ERROR_BUSY "$OUT/program.log" && echo "the cable is in use; running: $(pgrep -a -x openocd)"
       exit 1; }
sleep 1
echo "[2/3] capture -> $OUT/frames.bin"
READY="$OUT/.ready"; rm -f "$READY"
python3 "$HW/../../hopcim/model/kwlog_capture.py" "$OUT" --port "$UART" --seconds "$SECS" --ready-file "$READY" ${QUIET:+--quiet} & CAP=$!
for _ in $(seq 50); do [ -e "$READY" ] && break; sleep 0.1; done
rm -f "$READY"
echo "[3/3] OpenOCD, GDB load + continue"
( cd "$XH" && exec openocd -c "gdb_port $GDB_PORT" -c "telnet_port $TELNET_PORT" -c "tcl_port $TCL_PORT" \
    -f tb/core-v-mini-mcu-nexys-bscan.cfg ) > "$OUT/openocd.log" 2>&1 & OCD=$!
sleep 4
riscv64-unknown-elf-gdb -batch -ex "set remotetimeout 2000" -ex "target remote localhost:$GDB_PORT" \
  -ex "load" -ex "continue" "$ELF" > "$OUT/gdb.log" 2>&1 & GDB=$!
wait $CAP; rc=$?
kill $GDB $OCD 2>/dev/null; wait 2>/dev/null
echo "end $(date -Is)" >> "$OUT/run.txt"
echo "finished: $OUT"
exit $rc
