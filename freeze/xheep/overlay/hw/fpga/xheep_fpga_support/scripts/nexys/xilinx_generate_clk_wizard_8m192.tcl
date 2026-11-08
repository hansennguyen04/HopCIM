# Copyright 2026 X-HEEP contibutors
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# HopCIM live audio at its lowest real-time clock (use_cim_mic4): 8.192 MHz MMCM output without a buffer (exact: override mode), halved by a BUFGCE in the wrapper to 4.096 MHz: a 1.024 MHz
# PDM clock (/4) and 16 kHz audio (/64) exactly. VCO = 100 MHz * 32 / 5 = 640 MHz, /78.125 (fractional, CLKOUT0).
# Define design macros

set design_name      xilinx_clk_wizard
set in_clk_freq_MHz  100
set out_clk_freq_MHz 8.192
set out_clk_freq_Hz  8192000


# Create block design
create_bd_design $design_name

# Create ports
set clk_100MHz [ create_bd_port -dir I -type clk -freq_hz [ expr $in_clk_freq_MHz * 1000000 ] clk_100MHz ]
set clk_out1_0 [ create_bd_port -dir O -type clk clk_out1_0 ]
set_property -dict [ list CONFIG.FREQ_HZ $out_clk_freq_Hz ] $clk_out1_0

# Create instance and set properties
set clk_wiz_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0 ]
set_property -dict [ list \
 CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {8.192} \
 CONFIG.OVERRIDE_MMCM {true} \
 CONFIG.CLKOUT1_DRIVES {No_buffer} \
 CONFIG.MMCM_DIVCLK_DIVIDE {5} \
 CONFIG.MMCM_CLKFBOUT_MULT_F {32.000} \
 CONFIG.MMCM_CLKOUT0_DIVIDE_F {78.125} \
 CONFIG.PRIM_IN_FREQ $in_clk_freq_MHz \
 CONFIG.USE_LOCKED {false} \
 CONFIG.USE_RESET {false} \
] $clk_wiz_0

# Create port connections
# no output buffer: the feedback is wired here (the wizard then leaves clkfb_in open)
if {[llength [get_bd_pins -quiet clk_wiz_0/clkfb_out]]} { connect_bd_net [get_bd_pins clk_wiz_0/clkfb_out] [get_bd_pins clk_wiz_0/clkfb_in] }
connect_bd_net -net clk_in1_0_1 [ get_bd_ports clk_100MHz ] [ get_bd_pins clk_wiz_0/clk_in1 ]
connect_bd_net -net clk_wiz_0_clk_out1 [ get_bd_ports clk_out1_0 ] [ get_bd_pins clk_wiz_0/clk_out1 ]

# Save and close block design
save_bd_design
close_bd_design $design_name

# create wrapper
set wrapper_path [ make_wrapper -fileset sources_1 -files [ get_files -norecurse xilinx_clk_wizard.bd ] -top ]
add_files -norecurse -fileset sources_1 $wrapper_path
