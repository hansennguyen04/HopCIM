#!/usr/bin/env bash
# Board measurement of the "network every k hops" option (hopcim/docs/LIVE_AUDIO.md section 9), no speaker:
#   hopcim_live for k = 1, 2, 4: decisions, overrun, CPU-active cycles per decision;
#   hopcim_micrec for k = 1, 2, 4: logits on real microphone audio against the golden model.
# Needs the board connected, SW0 up, SW2 down, and no other OpenOCD or Hardware Manager on the cable.
#   usage: board_khop.sh [seconds of each live run, default 42] [bitstream]
#   env TAG: a firmware variant, e.g. TAG=_dwu runs hopcim_live_dwu_nexys.elf, hopcim_live_k2_dwu_nexys.elf, ...
set -uo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"; cd "$HW"
SECS="${1:-42}"
BIT="${2:-bitstreams/xheep_nexys_bscan_cim128k.bit}"
PY="$HOME/venvs/planenest/bin/python"
OUT="khop_board_$(date +%Y%m%d_%H%M%S).log"
fail=0
for k in 1 2 4; do
  T="${TAG:-}"
  live=hopcim_live${T}_nexys.elf; rec=hopcim_micrec${T}_nexys.elf
  [ "$k" != 1 ] && { live=hopcim_live_k${k}${T}_nexys.elf; rec=hopcim_micrec_k${k}${T}_nexys.elf; }
  echo "=== k = $k: live, $SECS s" | tee -a "$OUT"
  ./board_run.sh "$live" "$BIT" "$SECS" > /dev/null 2>&1
  log=$(ls -t "${live%.elf}"_board_*.uart.log | head -1)
  grep -E 'microphone:|network every|^\[[0-9]+ s\]' "$log" | cut -c1-150 | tee -a "$OUT"
  echo "    $log" | tee -a "$OUT"
  echo "=== k = $k: recording against the golden model" | tee -a "$OUT"
  ./board_run.sh "$rec" "$BIT" 150 > /dev/null 2>&1
  log=$(ls -t "${rec%.elf}"_board_*.uart.log | head -1)
  ( cd ../../hopcim/model && "$PY" check_micrec.py --k "$k" "$HW/$log" ) | cut -c1-150 | tee -a "$OUT"
  [ "${PIPESTATUS[0]}" = 0 ] || fail=1
  echo "    $log" | tee -a "$OUT"
done
echo "summary: $OUT; recordings $([ "$fail" = 0 ] && echo 'all exact' || echo 'NOT all exact')" | tee -a "$OUT"
exit $fail
