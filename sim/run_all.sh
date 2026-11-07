#!/usr/bin/env bash
# The whole RTL regression, one summary line per bench. Exit status 0 only if all pass.
#   run_all.sh [jobs, default 4]    about 8 min on 16 cores; writes sim/regression.log
# Not included: run_soc.sh (needs an X-HEEP Verilator build), run_audio_nl.sh (about 70 min, needs Vivado).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${1:-4}"
BENCHES=(
  "run_unit_tb.sh log2" "run_unit_tb.sh pdm" "run_engine_tb.sh" "run_endpoint_tb.sh"
  "run_fe_tb.sh cim" "run_fe_tb.sh ded" "run_seq_tb.sh cim" "run_seq_tb.sh ded"
  "run_stream_tb.sh cim" "run_stream_tb.sh ded"
  "run_mic_tb.sh cim 15 1" "run_mic_tb.sh ded 15 1" "run_mic_tb.sh cim 5 1" "run_mic_tb.sh ded 5 1"
  "run_mic_tb.sh cim 15 2" "run_mic_tb.sh cim 15 3" "run_mic_tb.sh cim 15 4" "run_mic_tb.sh cim 15 5" "run_mic_tb.sh cim 15 16"
  "run_mic_tb.sh ded 15 2" "run_mic_tb.sh ded 15 4" "run_mic_tb.sh cim 5 2" "run_mic_tb.sh cim 5 4" "run_mic_tb.sh ded 5 4"
)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
one() {
  local b="$1" id="$2" t0=$SECONDS
  if bash "$ROOT/sim/"$b > "$TMP/$id.out" 2>&1; then echo "PASS  $b  ($((SECONDS - t0)) s)"
  else echo "FAIL  $b  ($((SECONDS - t0)) s)"; tail -5 "$TMP/$id.out"; fi
}
i=0; for b in "${BENCHES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  one "$b" "$i" >> "$TMP/summary" & i=$((i + 1))
done; wait
sort -k2 "$TMP/summary" | tee "$ROOT/sim/regression.log"
n=$(grep -c "^PASS" "$ROOT/sim/regression.log"); echo "$n of ${#BENCHES[@]} benches pass" | tee -a "$ROOT/sim/regression.log"
[ "$n" -eq "${#BENCHES[@]}" ]
