#!/usr/bin/env bash
# One-command run of an X-HEEP program on the Nexys A7 through the board's single USB cable.
#   usage: board_run.sh <program.elf> [bitstream.bit] [timeout_s]
# Programs the FPGA (OpenOCD, BSCANE2 flow), starts OpenOCD, logs UART (/dev/ttyUSB1, 9600 baud),
# loads and runs the ELF with GDB in batch mode, and stops when the program prints "done".
# FOLLOW=1 also prints the UART output live. Ctrl-C stops the run (OpenOCD and the UART capture too).
# Requirements: board connected and powered, SW2 down (JTAG boot), Vivado Hardware Manager closed.
set -uo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"; XH="$HW/x-heep"
ELF="$(readlink -f "$1")"; BIT="$(readlink -f "${2:-$HW/bitstreams/xheep_nexys_bscan.bit}")"; TMO="${3:-120}"
UART="${UART:-/dev/ttyUSB1}"
# OpenOCD server ports (override if another OpenOCD is already listening, e.g. GDB_PORT=3334)
GDB_PORT="${GDB_PORT:-3333}"; TELNET_PORT="${TELNET_PORT:-4444}"; TCL_PORT="${TCL_PORT:-6666}"
source "$HW/xheep_env.sh" >/dev/null
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT="${ELF%.elf}_board_$STAMP"
FOLLOW="${FOLLOW:-0}"; CAT=""; OCD=""; GDB=""; TAIL=""
# background jobs ignore Ctrl-C; without this they keep holding the cable and the serial port
trap 'kill $GDB $OCD $CAT $TAIL 2>/dev/null; echo; echo "interrupted: $OUT.uart.log"; exit 130' INT TERM
lsusb | grep -q "0403:6010" || { echo "Nexys A7 (FTDI 0403:6010) not found on USB"; exit 1; }
[ -e "$UART" ] || { echo "$UART missing (try UART=/dev/ttyUSB0)"; exit 1; }

echo "[1/4] programming $(basename "$BIT")"
( cd "$XH" && openocd -c "gdb_port disabled" -c "telnet_port disabled" -c "tcl_port disabled" \
    -f tb/nexys-a7-program-fpga.cfg -c "pld load 0 $BIT" -c shutdown ) > "$OUT.program.log" 2>&1 \
  || { echo "programming failed, see $OUT.program.log"
       if grep -q LIBUSB_ERROR_BUSY "$OUT.program.log"; then
         P="$( { pgrep -x openocd; pgrep -fx "cat $UART"; } | tr '\n' ' ')"
         if [ -n "$P" ]; then echo "the cable is held by an earlier run; stop it with: kill $P"
         else echo "the cable is in use; close the Vivado Hardware Manager"; fi
       fi
       exit 1; }
sleep 1
echo "[2/4] UART capture -> $OUT.uart.log"
stty -F "$UART" 9600 raw -echo -ixon -ixoff cs8 -cstopb -parenb
cat "$UART" > "$OUT.uart.log" & CAT=$!
[ "$FOLLOW" = 1 ] && { tail -s 0.2 -n +1 -f "$OUT.uart.log" 2>/dev/null & TAIL=$!; }   # -s: if it must poll
echo "[3/4] OpenOCD"
( cd "$XH" && exec openocd -c "gdb_port $GDB_PORT" -c "telnet_port $TELNET_PORT" -c "tcl_port $TCL_PORT" \
    -f tb/core-v-mini-mcu-nexys-bscan.cfg ) > "$OUT.openocd.log" 2>&1 & OCD=$!
sleep 4
echo "[4/4] GDB load + continue (timeout ${TMO}s)"
riscv64-unknown-elf-gdb -batch -ex "set remotetimeout 2000" -ex "target remote localhost:$GDB_PORT" \
  -ex "load" -ex "continue" "$ELF" > "$OUT.gdb.log" 2>&1 & GDB=$!
for ((t = 0; t < TMO; t++)); do
  grep -q "^done" "$OUT.uart.log" 2>/dev/null && break
  sleep 1
done
sleep 1
kill $GDB $OCD $CAT $TAIL 2>/dev/null; wait 2>/dev/null
if grep -q "^done" "$OUT.uart.log"; then echo "finished: $OUT.uart.log"; else echo "timeout: see $OUT.*.log"; fi
[ "$FOLLOW" = 1 ] || tail -n 30 "$OUT.uart.log" | cut -c1-160
