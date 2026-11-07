# Out-of-context implementation on the Nexys A7-100T part (cost check of the HopCIM RTL).
# Usage: vivado -mode batch -source syn/ooc.tcl -tclargs <top> [<name>=<value> ...]
#   tops: cim_engine | cim_fe | mfcc_ded | cim_seq | pdm_dec | cim_audio | cim_endpoint; generics e.g. FE_DED=1
#   reports go to syn/ooc_<top>[_<name><value>...]/
set root [file normalize [file dirname [info script]]/..]
set top  [expr {$argc > 0 ? [lindex $argv 0] : "cim_engine"}]
set gens [lrange $argv 1 end]
set tag  $top
foreach g $gens { append tag "_" [string map {= ""} $g] }
set out  $root/syn/ooc_$tag
file mkdir $out
read_verilog -sv $::env(HOME)/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv
read_verilog -sv [list $root/rtl/cim_engine.sv $root/rtl/cim_fe.sv $root/rtl/mfcc_ded.sv $root/rtl/cim_log2fix16.sv $root/rtl/cim_seq.sv $root/rtl/pdm_dec.sv $root/rtl/cim_audio.sv $root/rtl/cim_endpoint.sv]
set gargs {}
foreach g $gens { lappend gargs -generic $g }
synth_design -top $top -part xc7a100tcsg324-1 -mode out_of_context -include_dirs $root/rtl {*}$gargs
create_clock -name clk -period 66.667 [get_ports clk_i]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/utilization.rpt
report_utilization -hierarchical -file $out/utilization_hier.rpt
report_ram_utilization -file $out/ram.rpt
report_timing_summary -max_paths 5 -file $out/timing.rpt
write_checkpoint -force $out/${tag}_routed.dcp
