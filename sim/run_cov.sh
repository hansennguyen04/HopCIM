#!/usr/bin/env bash
# Assertions and coverage: all 40 regression benches built with tb/cim_assertions.sv bound in and Verilator
# coverage on (line, branch, toggle). Writes sim/cov/<tag>.log and <tag>.dat; model/verif_report.py reads them.
#   run_cov.sh [jobs, default 6]        about 25 min on 16 cores
#   run_cov.sh one <bench> [mode] [div] [k]    one bench, for the mutation tests; EXTRA_RTL overrides sources
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
ALLRTL=(rtl/cim_engine.sv rtl/cim_fe.sv rtl/mfcc_ded.sv rtl/cim_log2fix16.sv rtl/cim_seq.sv rtl/pdm_dec.sv rtl/cim_audio.sv rtl/cim_endpoint.sv)
OUT="${OUT:-$ROOT/sim/cov}"; mkdir -p "$OUT"
one() {
  local sq=0 b1=0 dwu=0 a; local args=(); for a in "$@"; do if [ "$a" = s ]; then sq=1; elif [ "$a" = b1 ]; then sq=1; b1=1; elif [ "$a" = d1 ]; then sq=1; dwu=1; elif [ "$a" = d4 ]; then sq=1; dwu=4; else args+=("$a"); fi; done; set -- "${args[@]}"
  local bench="$1" mode="${2:-cim}" div="${3:-15}" k="${4:-1}" tag top
  local ded=0; [ "$mode" = ded ] && ded=1
  local files=() gens=() defs=() tb
  case "$bench" in
    audio)  tag=audio_dir; top=tb_cim_audio_dir; files=(rtl/pdm_dec.sv rtl/cim_audio.sv); tb=tb/tb_cim_audio_dir.sv; defs=(-DSVA_AUDIO -DSVA_PDM) ;;
    log2)   tag=log2; top=tb_cim_log2fix16; files=(rtl/cim_log2fix16.sv); tb=tb/tb_cim_log2fix16.sv ;;
    pdm)    tag=pdm; top=tb_pdm_dec; files=(rtl/pdm_dec.sv); tb=tb/tb_pdm_dec.sv; defs=(-DSVA_PDM) ;;
    engine) tag=engine; top=tb_cim_engine; files=(rtl/cim_engine.sv); tb=tb/tb_cim_engine.sv; defs=(-DSVA_ENGINE) ;;
    endpoint) tag=endpoint; top=tb_cim_endpoint; files=("${ALLRTL[@]}"); tb=tb/tb_cim_endpoint.sv; defs=(-DSVA_ENGINE -DSVA_SOC -DSVA_AUDIO -DSVA_PDM) ;;
    fe|seq|stream) tag=${bench}_$mode; top=tb_cim_$bench; files=("${ALLRTL[@]}"); tb=tb/tb_cim_$bench.sv; gens=(-GDED=$ded); defs=(-DSVA_ENGINE -DSVA_SOC -DSVA_AUDIO -DSVA_PDM) ;;
    mic)    tag=mic_${mode}_div${div}_k$k; top=tb_cim_mic; files=("${ALLRTL[@]}"); tb=tb/tb_cim_mic.sv; gens=(-GDED=$ded -GCLKDIV=$div -GNET_K=$k); defs=(-DSVA_ENGINE -DSVA_SOC -DSVA_AUDIO -DSVA_PDM) ;;
    *) echo "unknown bench $bench"; return 2 ;;
  esac
  if [ "$b1" = 1 ]; then gens+=(-GS=1 -GB1=1); tag="${tag}_b1"                # S with StreamFuse B in block 1
  elif [ "$dwu" != 0 ]; then gens+=(-GS=1 -GDWU=$dwu); tag="${tag}_d$dwu"   # S with the digital depthwise unit
  elif [ "$sq" = 1 ]; then gens+=(-GS=1); tag="${tag}_s"; fi               # the baseline S (SEQ_S = 1)
  tag="$tag${TAGSUF:-}"
  [ -n "${EXTRA_RTL:-}" ] && files=($EXTRA_RTL)
  local obj="$OUT/obj_$tag"
  local sv=(); for f in "${files[@]}"; do case "$f" in /*) sv+=("$f");; *) sv+=("$ROOT/$f");; esac; done
  local cov=(--coverage); [ "${NOCOV:-0}" = 1 ] && cov=()
  if ! verilator --binary --timing -Wno-fatal -j "${VJOBS:-4}" -O2 --assert ${cov[@]+"${cov[@]}"} --top-module "$top" ${gens[@]+"${gens[@]}"} ${defs[@]+"${defs[@]}"} \
       -Mdir "$obj" -MAKEFLAGS "OBJCACHE=" -I"$ROOT/rtl" "$PKG" "${sv[@]}" "$ROOT/$tb" "$ROOT/tb/cim_assertions.sv" > "$OUT/build_$tag.log" 2>&1; then
    echo "BUILD-FAIL $tag"; tail -20 "$OUT/build_$tag.log"; return 1
  fi
  ( cd "$ROOT" && ${RUN_TIMEOUT:+timeout "$RUN_TIMEOUT"} stdbuf -oL "$obj/V$top" "+verilator+coverage+file+$OUT/$tag.dat" ${RUNARGS:-} > "$OUT/$tag.log" 2>&1 )
  local pass=0; grep -q "^PASS" "$OUT/$tag.log" && pass=1
  local failed; failed=$(grep -c "ASSERTION FAILED" "$OUT/$tag.log")
  [ "${KEEP_OBJ:-0}" = 1 ] || rm -rf "$obj"
  echo "$tag pass=$pass assertion_failures=$failed"
  [ "$pass" = 1 ] && [ "$failed" = 0 ]
}
if [ "${1:-}" = one ]; then shift; one "$@"; exit $?; fi
JOBS="${1:-6}"
LIST=("log2" "pdm" "engine" "endpoint" "fe cim" "fe ded" "seq cim" "seq ded" "stream cim" "stream ded"
      "mic cim 15 1" "mic ded 15 1" "mic cim 5 1" "mic ded 5 1" "mic cim 15 2" "mic cim 15 3" "mic cim 15 4" "mic cim 15 5" "mic cim 15 16"
      "mic ded 15 2" "mic ded 15 4" "mic cim 5 2" "mic cim 5 4" "mic ded 5 4"
      "seq cim s" "seq ded s" "stream cim s" "stream ded s" "mic cim 15 1 s" "mic cim 5 1 s" "mic cim 4 1 s" "mic ded 4 1 s" "mic cim 4 2 s" "mic cim 4 4 s"
      "seq cim b1" "seq ded b1" "stream cim b1" "mic cim 15 1 b1" "mic cim 4 1 b1" "mic cim 4 4 b1")
for b in "${LIST[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  one $b &
done
# the directed benches of the extreme cases
wait
one audio &
TAGSUF=_sat RUNARGS=+VEC=tb/pdm_vec_sat one pdm &
TAGSUF=_edge RUNARGS=+VEC=tb/stream_vec_edge_cim one stream cim &
TAGSUF=_edge RUNARGS=+VEC=tb/stream_vec_edge_cim one stream cim s &
TAGSUF=_edge RUNARGS=+VEC=tb/stream_vec_edge_ded one stream ded &
wait
