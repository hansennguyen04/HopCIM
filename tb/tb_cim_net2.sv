// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Bench of cim_gseq in cim_endpoint (SEQ_NET2): ResNet8 against model/resnet_int.py (vectors: python model/gen_net2_vectors.py).
// For every image and every operation the network is run up to and including that operation and the memory the operation wrote is
// compared (ACT through the debug port, the pool registers by hierarchy, the logits and the class through registers).
// +IMG0=<first image index in the vectors> +NIMG=<images>; +QUICK: only images' final result (logits, class), all layers once.
`timescale 1ns / 1ps

module tb_cim_net2 #(
    parameter int DWU = 0
);
  import xheep_obi_pkg::*;
  localparam logic [31:0] BASE = 32'hC004_0000;
  localparam int NIMGMAX = 100, NOP = 14, NT = 86;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq;
  cim_endpoint #(.TILES(128), .SEQ_S(1'b1), .SEQ_DWU(DWU), .SEQ_NET2(1'b1)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));

  logic [31:0] tiles [NT * 256];
  logic [31:0] params [640 * 3];
  logic [31:0] desc [NOP * 8];
  logic [23:0] img [NIMGMAX * 1024];
  logic [7:0]  exp [NIMGMAX * 114688];
  logic [7:0]  exl [NIMGMAX * 10];
  logic [7:0]  exp_pool [NIMGMAX * 64];
  logic [7:0]  excls [NIMGMAX];
  int          errors = 0, nimg, img0, nimgs;
  longint unsigned cyc = 0, cyc_min = 64'hFFFF_FFFF_FFFF_FFFF, cyc_max = 0;
  longint unsigned op_cyc [NOP];
  always @(posedge clk) cyc <= cyc + 1;
  int op_size [NOP] = '{16384, 16384, 16384, 16384, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 4096, 0, 0};
  int op_dst  [NOP] = '{0, 1, 2, 0, 1, 2, 1, 0, 1, 2, 1, 0, 0, 0};
  int op_off  [NOP];

  // Rewrite margin (NET2_GATES.md, Gather: "the gather may start rewriting a bank 1 cycle after the pass that uses it started"; the engine
  // reads row group g at edge t0 + 2 + g). At every edge, no lane of the engine's input-bank write port may write the entry {bank, row group}
  // that the engine's stage 1 reads at the same edge. The frozen design writes a group one edge after its read; a write at the same edge reads
  // the old value in this RTL (the outputs do not change), so only this check sees the margin gone. Added 2026-11-26 after N7's first run.
  int margin_err = 0;
  always @(posedge clk) begin
    if (rst_n && dut.i_engine.v1 && dut.i_engine.xw_we_i) begin
      for (int l = 0; l < 8; l++) begin
        logic [5:0] r;
        r = dut.i_engine.xw_row_i + {3'd0, 3'(l) - dut.i_engine.xw_row_i[2:0]};
        if (dut.i_engine.xw_mask_i[l] && (dut.i_engine.xw_bank_i == dut.i_engine.m1.bank) && (r[5:3] == dut.i_engine.m1.grp)) begin
          if (margin_err < 5) $display("margin: cycle %0d bank %0d row group %0d written at the edge the engine reads it (lane %0d)", cyc,
                                       dut.i_engine.m1.bank, dut.i_engine.m1.grp, l);
          margin_err++;
        end
      end
    end
  end

  task automatic tick(); @(posedge clk); #1; endtask

  task automatic wr(input logic [31:0] a, input logic [31:0] d);
    req.req = 1'b1; req.we = 1'b1; req.be = 4'hF; req.addr = a; req.wdata = d;
    forever begin @(negedge clk); if (rsp.gnt) break; end
    tick(); req.req = 1'b0; req.we = 1'b0;
  endtask
  task automatic rd(input logic [31:0] a, output logic [31:0] d);
    req.req = 1'b1; req.we = 1'b0; req.be = 4'hF; req.addr = a; req.wdata = '0;
    forever begin @(negedge clk); if (rsp.gnt) break; end
    @(posedge clk); #1; req.req = 1'b0;
    d = rsp.rvalid ? rsp.rdata : 32'hDEAD_DEAD;
    while (!rsp.rvalid) begin @(negedge clk); d = rsp.rdata; end
    d = rsp.rdata;
    tick();
  endtask
  task automatic load_img(int i);
    for (int p = 0; p < 1024; p++) wr(BASE + 32'hF000 + 32'(p) * 4, {8'd0, img[i * 1024 + p]});
  endtask

  // run operations 0 .. n - 1 from op0: the bench only supports complete runs (each starts from the image)
  task automatic run_ops(input int n, output longint unsigned cc);
    logic [31:0] st;
    longint unsigned t0;
    wr(BASE + 32'h7204, 32'd2);                              // clear done
    t0 = cyc;
    wr(BASE + 32'h7200, 32'h8000_0000 | 32'(n));
    while (dut.gs_busy) tick();
    cc = cyc - t0;
    repeat (2) tick();
  endtask
  task automatic check_bank(input int i, input int o);
    logic [31:0] d;
    int b, bad;
    bad = 0;
    for (b = 0; b < op_size[o]; b += 4) begin
      rd(BASE + 32'h1_0000 + (32'(op_dst[o]) << 14) + 32'(b), d);
      for (int k = 0; k < 4; k++)
        if (d[8*k +: 8] !== exp[i * 114688 + op_off[o] + b + k]) begin
          if (bad < 3) $display("IMG %0d op %0d byte %0d: %02x expected %02x", i, o, b + k, d[8*k +: 8], exp[i * 114688 + op_off[o] + b + k]);
          bad++; errors++;
        end
    end
    if (bad != 0) $display("IMG %0d op %0d: %0d bytes wrong", i, o, bad);
  endtask

  initial begin
    logic [31:0] d;
    longint unsigned cc;
    $readmemh("tb/net2_vec/tiles.hex", tiles);
    $readmemh("tb/net2_vec/params.hex", params);
    $readmemh("tb/net2_vec/desc.hex", desc);
    $readmemh("tb/net2_vec/img.hex", img);
    $readmemh("tb/net2_vec/exp.hex", exp);
    $readmemh("tb/net2_vec/logits.hex", exl);
    $readmemh("tb/net2_vec/pool.hex", exp_pool);
    $readmemh("tb/net2_vec/cls.hex", excls);
    nimg = 4; img0 = 0;
    void'($value$plusargs("NIMG=%d", nimg));
    void'($value$plusargs("IMG0=%d", img0));
    op_off[0] = 0;
    for (int o = 1; o < NOP; o++) op_off[o] = op_off[o - 1] + op_size[o - 1];
    req = '0;
    repeat (4) tick();
    rst_n = 1'b1;
    repeat (4) tick();
    // debug access to ACT from cim_gseq
    wr(BASE + 32'h7208, 32'd1);
    for (int t = 0; t < NT * 256; t++) wr(BASE + 32'h2_0000 + 32'(t) * 4, tiles[t]);
    for (int e = 0; e < 640; e++) for (int f = 0; f < 3; f++) wr(BASE + 32'hC000 + 32'(e) * 16 + 32'(f) * 4, params[e * 3 + f]);
    for (int o = 0; o < NOP; o++) for (int w = 0; w < 8; w++) wr(BASE + 32'h7000 + 32'(o) * 32 + 32'(w) * 4, desc[o * 8 + w]);
    $display("loaded %0d tiles, 640 parameter entries, %0d descriptors", NT, NOP);
    for (int i = img0; i < img0 + nimg; i++) begin
      int bad0;
      bad0 = errors;
      load_img(i);
      for (int o = 0; o < NOP; o++) begin
        run_ops(o + 1, cc);
        if (i == img0) op_cyc[o] = cc;
        if (o < 12) check_bank(i, o);
        else if (o == 12) begin
          for (int p = 0; p < 64; p++)
            if (dut.g_net2.i_gseq.pout[p] !== $signed(exp_pool[i * 64 + p])) begin
              errors++; if (errors < 20) $display("IMG %0d pool %0d: %0d expected %0d", i, p, dut.g_net2.i_gseq.pout[p], $signed(exp_pool[i * 64 + p]));
            end
        end else begin
          for (int n = 0; n < 10; n++) begin
            rd(BASE + 32'h7300 + 32'(n) * 4, d);
            if (d[7:0] !== exl[i * 10 + n]) begin errors++; $display("IMG %0d logit %0d: %0d expected %0d", i, n, $signed(d[7:0]), $signed(exl[i * 10 + n])); end
          end
          rd(BASE + 32'h7204, d);
          if (d[7:4] !== excls[i][3:0]) begin errors++; $display("IMG %0d class %0d expected %0d", i, d[7:4], excls[i]); end
          if (d[2]) begin errors++; $display("IMG %0d: result lost (err)", i); end
        end
      end
      $display("IMG %0d: %0d errors", i, errors - bad0);
      if (i == img0) begin
        longint unsigned prev;
        prev = 0;
        for (int o = 0; o < NOP; o++) begin
          $display("  op %0d: run of %0d ops %0d cycles, increment %0d", o, o + 1, op_cyc[o], op_cyc[o] - prev);
          prev = op_cyc[o];
        end
      end
      if (cyc_max == 0) begin end
    end
    $display("rewrite margin: %0d violations", margin_err);
    errors += margin_err;
    $display("RESULT: %0d errors", errors);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
