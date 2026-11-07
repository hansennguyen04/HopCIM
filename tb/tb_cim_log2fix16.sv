// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Exhaustive testbench for cim_log2fix16: every v = 1..32767 against model/mfcc_int.py log2fix16
// (tb/stream_vec_cim/log2.hex, from model/gen_stream_vectors.py), and the 16-cycle latency.
`timescale 1ns / 1ps
module tb_cim_log2fix16;
  logic clk = 1'b0, rst_n = 1'b0, start = 1'b0, done;
  logic [14:0] v = '0;
  logic [19:0] y;
  logic [31:0] exp_t [32768];
  int bad = 0, bad_t = 0;
  always #5 clk = ~clk;
  cim_log2fix16 dut (.clk_i(clk), .rst_ni(rst_n), .start_i(start), .v_i(v), .done_o(done), .y_o(y));
  initial begin
    $readmemh("tb/stream_vec_cim/log2.hex", exp_t);
    repeat (3) @(posedge clk);
    #1 rst_n = 1'b1;
    #200;                                                      // past the 100-ns global set/reset of gate-level models
    for (int i = 1; i < 32768; i++) begin
      int n = 0;
      @(posedge clk); #1 v = 15'(i); start = 1'b1;
      @(posedge clk); #1 start = 1'b0;
      while (!done) begin @(posedge clk); #1 n++; end
      if (y !== exp_t[i][19:0]) begin if (bad < 5) $display("v %0d: got %0d expected %0d", i, y, exp_t[i]); bad++; end
      if (n != 16) bad_t++;                             // done 16 edges after the start edge
    end
    $display("RESULT: %0d of 32767 values wrong, %0d latency errors", bad, bad_t);
    if (bad == 0 && bad_t == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
