#!/usr/bin/env bash
# The whole RTL regression, one summary line per bench. Exit status 0 only if all pass.
#   run_all.sh [jobs, default 4]    about 12 min on 16 cores; writes sim/regression.log
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
  "run_unit_tb.sh audio" "run_unit_tb.sh pdm_sat" "run_stream_tb.sh cim edge" "run_stream_tb.sh ded edge"
  # the baseline S (SEQ_S = 1); 4 = the 4.096 MHz system clock
  "run_seq_tb.sh cim s" "run_seq_tb.sh ded s" "run_stream_tb.sh cim s" "run_stream_tb.sh ded s" "run_stream_tb.sh cim edge s"
  "run_mic_tb.sh cim 15 1 s" "run_mic_tb.sh cim 5 1 s" "run_mic_tb.sh cim 4 1 s" "run_mic_tb.sh ded 4 1 s" "run_mic_tb.sh cim 4 2 s" "run_mic_tb.sh cim 4 4 s"
  # B1: S with StreamFuse B in block 1 (SEQ_B1 = 1)
  "run_seq_tb.sh cim b1" "run_seq_tb.sh ded b1" "run_stream_tb.sh cim b1" "run_mic_tb.sh cim 15 1 b1" "run_mic_tb.sh cim 4 1 b1" "run_mic_tb.sh cim 4 4 b1"
  # DWU: S with the digital depthwise unit (SEQ_DWU = 1: L1; 4: every depthwise layer; docs/DWU_GATES.md)
  "run_seq_tb.sh cim d1" "run_seq_tb.sh cim d4" "run_seq_tb.sh ded d4" "run_stream_tb.sh cim d4" "run_mic_tb.sh cim 15 1 d4" "run_mic_tb.sh cim 4 1 d4" "run_mic_tb.sh cim 4 4 d4"
  # expected to fail: R6b overruns at 4.096 MHz
  "!run_mic_tb.sh cim 4 1"
)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
one() {                                               # an entry "!<bench>" passes when the bench fails
  local b="$1" id="$2" t0=$SECONDS want=0
  [ "${b:0:1}" = "!" ] && { want=1; b="${b:1}"; }
  local tag="$b"; [ "$want" = 1 ] && tag="!$b"
  local rc=0; bash "$ROOT/sim/"$b > "$TMP/$id.out" 2>&1 || rc=1
  if [ "$rc" = "$want" ]; then echo "PASS  $tag  ($((SECONDS - t0)) s)"
  else echo "FAIL  $tag  ($((SECONDS - t0)) s)"; tail -5 "$TMP/$id.out"; fi
}
i=0; for b in "${BENCHES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  one "$b" "$i" >> "$TMP/summary" & i=$((i + 1))
done; wait
sort -k2 "$TMP/summary" | tee "$ROOT/sim/regression.log"
n=$(grep -c "^PASS" "$ROOT/sim/regression.log"); echo "$n of ${#BENCHES[@]} benches pass" | tee -a "$ROOT/sim/regression.log"
[ "$n" -eq "${#BENCHES[@]}" ]
