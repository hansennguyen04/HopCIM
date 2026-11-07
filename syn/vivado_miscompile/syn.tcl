read_verilog -sv mr.sv
synth_design -top mr -part xc7a100tcsg324-1 -mode out_of_context
write_verilog -mode funcsim -force mr_nl.v
