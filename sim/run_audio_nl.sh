#!/usr/bin/env bash
# Gate-level check of cim_audio: its RTL against its Vivado funcsim netlist (tb/tb_cim_audio_nl.sv, xsim).
#   run_audio_nl.sh [routed.dcp [PDM divider]]    about 70 min at divider 15
#     default syn/ooc_cim_audio_TW7/cim_audio_TW7_routed.dcp, divider 15, from
#       vivado -mode batch -source syn/ooc.tcl -tclargs cim_audio TW=7
#     a netlist synthesised with CLKDIV=5 and divider 5 takes a third of the time.
# Use the out-of-context checkpoint: an in-context netlist has optimised boundary ports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
DCP="$(readlink -f "${1:-$ROOT/syn/ooc_cim_audio_TW7/cim_audio_TW7_routed.dcp}")"
DIV="${2:-15}"
WD="$ROOT/sim/xsim_audio_nl_div$DIV"; mkdir -p "$WD"; cd "$WD"
cat > write_nl.tcl <<EOF
open_checkpoint $DCP
write_verilog -force -mode funcsim netlist_raw.v
EOF
vivado -mode batch -nojournal -log write_nl.log -source write_nl.tcl > /dev/null
# the netlist's modules get their own names, next to the RTL's
sed -E 's/\bcim_audio\b/cim_audio_nl/g; s/\bpdm_dec\b/pdm_dec_nl/g' netlist_raw.v > netlist.v
xvlog -nolog netlist.v > xvlog_nl.log
xvlog -nolog -sv "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/tb/tb_cim_audio_nl.sv" > xvlog_tb.log
xelab -nolog -L unisims_ver -relax -generic_top "CLKDIV=$DIV" -s tbnl tb_cim_audio_nl glbl > xelab.log
xsim -nolog tbnl -R | tee run.log
grep -q "^PASS" run.log
