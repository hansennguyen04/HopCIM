// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Replay of many CIFAR-10 images through cim_gseq in cim_endpoint: logits, class and the cycles per inference.
// +VDIR=<dir with img.hex, logits.hex, cls.hex>  +NIMG=<images>   (tiles, parameters, descriptors from tb/net2_vec)
`timescale 1ns / 1ps

module tb_cim_net2_all;
  import xheep_obi_pkg::*;
  localparam logic [31:0] BASE = 32'hC004_0000;
  localparam int NMAX = 700, NOP = 14, NT = 86;
  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq;
  cim_endpoint #(.TILES(128), .SEQ_S(1'b1), .SEQ_DWU(0), .SEQ_NET2(1'b1)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));
  logic [31:0] tiles [NT * 256];
  logic [31:0] params [640 * 3];
  logic [31:0] desc [NOP * 8];
  logic [23:0] img [NMAX * 1024];
  logic [7:0]  exl [NMAX * 10];
  logic [7:0]  excls [NMAX];
  int errors = 0, nimg = 0;
  longint unsigned cyc = 0, c0, cmin = 64'hFFFF_FFFF_FFFF_FFFF, cmax = 0;
  string vdir;
  always @(posedge clk) cyc <= cyc + 1;
  task automatic tick(); @(posedge clk); #1; endtask
  task automatic wr(input logic [31:0] a, input logic [31:0] d);
    req.req = 1'b1; req.we = 1'b1; req.be = 4'hF; req.addr = a; req.wdata = d;
    forever begin @(negedge clk); if (rsp.gnt) break; end
    tick(); req.req = 1'b0; req.we = 1'b0;
  endtask
  task automatic rd(input logic [31:0] a, output logic [31:0] d);
    req.req = 1'b1; req.we = 1'b0; req.be = 4'hF; req.addr = a; req.wdata = '0;
    forever begin @(negedge clk); if (rsp.gnt) break; end
    tick(); req.req = 1'b0;
    while (!rsp.rvalid) tick();
    d = rsp.rdata;
    tick();
  endtask

  initial begin
    logic [31:0] d;
    bit bad;
    int total_bad;
    total_bad = 0;
    void'($value$plusargs("NIMG=%d", nimg));
    void'($value$plusargs("VDIR=%s", vdir));
    $readmemh("tb/net2_vec/tiles.hex", tiles);
    $readmemh("tb/net2_vec/params.hex", params);
    $readmemh("tb/net2_vec/desc.hex", desc);
    $readmemh({vdir, "/img.hex"}, img);
    $readmemh({vdir, "/logits.hex"}, exl);
    $readmemh({vdir, "/cls.hex"}, excls);
    req = '0;
    repeat (4) tick(); rst_n = 1'b1; repeat (4) tick();
    for (int t = 0; t < NT * 256; t++) wr(BASE + 32'h2_0000 + 32'(t) * 4, tiles[t]);
    for (int e = 0; e < 640; e++) for (int f = 0; f < 3; f++) wr(BASE + 32'hC000 + 32'(e) * 16 + 32'(f) * 4, params[e * 3 + f]);
    for (int o = 0; o < NOP; o++) for (int w = 0; w < 8; w++) wr(BASE + 32'h7000 + 32'(o) * 32 + 32'(w) * 4, desc[o * 8 + w]);
    for (int i = 0; i < nimg; i++) begin
      for (int p = 0; p < 1024; p++) wr(BASE + 32'hF000 + 32'(p) * 4, {8'd0, img[i * 1024 + p]});
      wr(BASE + 32'h7204, 32'd2);
      c0 = cyc;
      wr(BASE + 32'h7200, 32'h8000_0000 | 32'(NOP));
      while (dut.gs_busy) tick();
      if (cyc - c0 < cmin) cmin = cyc - c0;
      if (cyc - c0 > cmax) cmax = cyc - c0;
      bad = 0;
      for (int n = 0; n < 10; n++) begin
        rd(BASE + 32'h7300 + 32'(n) * 4, d);
        if (d[7:0] !== exl[i * 10 + n]) bad = 1;
      end
      rd(BASE + 32'h7204, d);
      if (d[7:4] !== excls[i][3:0] || d[2]) bad = 1;
      if (bad) begin total_bad++; if (total_bad < 4) $display("IMG %0d wrong (status %08x)", i, d); end
    end
    $display("ALL %0d images: %0d with wrong logits, class or lost result; cycles per inference %0d..%0d", nimg, total_bad, cmin, cmax);
    if (total_bad == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
