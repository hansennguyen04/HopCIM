#!/usr/bin/env bash
# HopCIM board suite (Nexys A7-100T, one USB cable; SW0 up, SW2 down). Runs every HopCIM app on
# the two 128-tile bitstreams with the layer sequencer and the window path (R6b + R5b,
# cim128w_*) and prints the result lines of each run.
#   cim_fe build: cim_smoke, hopcim_mfcc (R4b), hopcim_fe (R5a), hopcim_fe_irq (sleep accounting),
#                 hopcim_net (DS-CNN, CPU-orchestrated, 20 clips), hopcim_net1 (1 clip: the same C
#                 code as the simulation build, so its cycles must equal Verilator's),
#                 hopcim_seq (DS-CNN by the layer sequencer, CPU asleep; 20 clips, three times each),
#                 hopcim_stream (R5b: 2 s of streamed speech, 100 hops, 51 decisions, window path in hardware)
#   B3 build:     hopcim_fe_ded, hopcim_fe_irq_ded, hopcim_seq, hopcim_stream_ded
# Logs: <app>_nexys_board_<time>.uart.log next to this script.   usage: ./board_suite.sh
set -uo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"
cd "$HW"
lsusb | grep -q "0403:6010" || { echo "Nexys A7 (FTDI 0403:6010) not found on USB: connect and power the board"; exit 1; }
run() {                                             # run <bitstream> <app> <timeout_s>
    ./board_run.sh "$2_nexys.elf" "bitstreams/$1" "$3" > /dev/null 2>&1
    local log; log=$(ls -t "$2"_nexys_board_*.uart.log 2>/dev/null | head -1)
    echo "== $2 ($1)"
    grep -E "cycles|per frame|front-end|window path|total|clip [0-9]+:|predictions|PASS|FAIL" "$log" 2>/dev/null \
        || { echo "   no output; see the OpenOCD log"; grep -E "Error" x-heep/openocd_fpga.log | head -3; }
}
B_CIM=xheep_nexys_bscan_pdm_cim128w_cim.bit
B_DED=xheep_nexys_bscan_pdm_cim128w_ded.bit
for a in cim_smoke hopcim_mfcc hopcim_fe hopcim_fe_irq; do run $B_CIM $a 120; done
run $B_CIM hopcim_net 300
run $B_CIM hopcim_net1 120
run $B_CIM hopcim_seq 120
run $B_CIM hopcim_stream 180
for a in hopcim_fe_ded hopcim_fe_irq_ded hopcim_seq; do run $B_DED $a 120; done
run $B_DED hopcim_stream_ded 180
