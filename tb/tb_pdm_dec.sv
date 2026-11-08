// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Testbench for pdm_dec: 1 s of PDM (a GSC clip through model/pdm_dec.py's sigma-delta model) must
// give model/pdm_dec.py decimate()'s 16,000 samples bit-exactly. The microphone model presents the
// next bit after every falling edge of pdm_clk_o (data valid at the rising edge, L/R select low).
// Vectors: python model/pdm_dec.py (tb/pdm_vec).
`timescale 1ns / 1ps
module tb_pdm_dec;
  localparam int NB = 1024000, NS = 16000;
  logic clk = 1'b0, rst_n = 1'b0, en = 1'b0, pdm_clk, pdm_data = 1'b0, v;
  logic signed [15:0] smp;
  logic [0:0]  bits [NB];
  logic [15:0] exp_s [NS];
  int bi = 0, si = 0, bad = 0;
  string vec;
  always #5 clk = ~clk;
  pdm_dec dut (.clk_i(clk), .rst_ni(rst_n), .en_i(en), .pdm_clk_o(pdm_clk), .pdm_data_i(pdm_data),
               .smp_valid_o(v), .smp_o(smp));
  always @(negedge pdm_clk) if (en) begin                   // the microphone's next bit
    bi = bi + 1;
    pdm_data = (bi < NB) ? bits[bi][0] : 1'b0;
  end
  always @(posedge clk) if (v) begin
    if (si < NS && smp !== exp_s[si]) begin
      if (bad < 5) $display("sample %0d: got %0d expected %0d", si, smp, $signed(exp_s[si]));
      bad++;
    end
    si++;
  end
  initial begin
    if (!$value$plusargs("VEC=%s", vec)) vec = "tb/pdm_vec";   // tb/pdm_vec_sat: full-scale, saturating input
    $readmemb({vec, "/bits.hex"}, bits);
    $readmemh({vec, "/samples.hex"}, exp_s);
    pdm_data = bits[0][0];
    repeat (3) @(posedge clk);
    #1 rst_n = 1'b1; en = 1'b1;
    while (si < NS) @(posedge clk);
    $display("RESULT: %0d of %0d samples wrong; %0d bits used, PDM clock 1/15 of the system clock", bad, si, bi);
    if (bad == 0 && si == NS) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
