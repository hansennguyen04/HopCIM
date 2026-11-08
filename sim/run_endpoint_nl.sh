#!/usr/bin/env bash
# Post-synthesis smoke test of the whole endpoint: tb_cim_seq on the Vivado functional netlist of
# cim_endpoint (xsim, with the Xilinx cell models), MEDIUM set: every layer of one clip through the
# ACT window, and one full inference with its logits.
#   run_endpoint_nl.sh [routed.dcp]   default: syn/ooc_cim_endpoint_FE_DED0_PDM_CLKDIV15/, from
#     vivado -mode batch -source syn/ooc.tcl -tclargs cim_endpoint FE_DED=0 PDM_CLKDIV=15
#   env NLTAG=<suffix> (work directory sim/xsim_endpoint_nl<suffix>), TBGEN="S=1 B1=1" (the bench's parameters,
#   matching the netlist's SEQ_S / SEQ_B1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
DCP="$(readlink -f "${1:-$ROOT/syn/ooc_cim_endpoint_FE_DED0_PDM_CLKDIV15/cim_endpoint_FE_DED0_PDM_CLKDIV15_routed.dcp}")"
WD="$ROOT/sim/xsim_endpoint_nl${NLTAG:-}"; mkdir -p "$WD"; cd "$WD"
GEN=(); for g in ${TBGEN:-}; do GEN+=(-generic_top "$g"); done
cat > write_nl.tcl <<TCL
open_checkpoint $DCP
write_verilog -force -mode funcsim netlist.v
TCL
vivado -mode batch -nojournal -log write_nl.log -source write_nl.tcl > /dev/null
xvlog -nolog netlist.v > xvlog_nl.log
xvlog -nolog -sv -d NETLIST -d MEDIUM "$PKG" "$ROOT/tb/tb_cim_seq.sv" > xvlog_tb.log
xelab -nolog -L unisims_ver -relax ${GEN[@]+"${GEN[@]}"} -s tbnl tb_cim_seq glbl > xelab.log
ln -sfn "$ROOT/tb" tb                                 # the bench reads tb/seq_vec/...
xsim -nolog tbnl -R | tee run.log
grep -q "^PASS" run.log
