# Local addition (HopCIM): constraints for the BSCANE2 debug TAP (FuseSoC flag use_bscane_xilinx).
# TCK comes from the FPGA's own JTAG port through BSCANE2 (OpenOCD drives it at 1 MHz). Without a
# clock definition its registers are unclocked and never timed. The DMI crossing to the system
# clock is a cdc_2phase handshake, so TCK is asynchronous to every other clock.
create_clock -name bscan_tck -period 100.000 [get_pins -hierarchical -filter {NAME =~ "*i_dmi_jtag_tap/i_tap_dtmcs/TCK"}]
set_clock_groups -asynchronous -group [get_clocks bscan_tck]
