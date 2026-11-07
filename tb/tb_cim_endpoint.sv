// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Testbench for cim_endpoint through its OBI slave port: register map, result readback,
// back-to-back passes (go held until ready) and the X/W write hold after a pass starts.
`timescale 1ns / 1ps

module tb_cim_endpoint;
  import xheep_obi_pkg::*;
  localparam int TILES = 128;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq;

  cim_endpoint #(.TILES(TILES)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));

  byte    W [TILES][64][16];
  int     X [64];
  longint ACC [16];
  int     errors = 0, checks = 0;
  longint unsigned edge_cnt = 0;
  always @(posedge clk) edge_cnt <= edge_cnt + 1;

  task automatic tick(); @(posedge clk); #1; endtask

  // OBI master: hold req until a clock edge samples req && gnt (the OBI handshake), then wait
  // for rvalid. gnt is sampled at the edge, never right after driving req (stale-read hazard).
  task automatic obi(input bit we, input logic [31:0] addr, input logic [31:0] wdata, output logic [31:0] rdata,
                     output longint unsigned t_gnt);
    req.req = 1'b1; req.we = we; req.be = 4'hF; req.addr = addr; req.wdata = wdata;
    do @(posedge clk); while (!rsp.gnt);
    t_gnt = edge_cnt;
    #1 req.req = 1'b0;
    while (!rsp.rvalid) tick();
    rdata = rsp.rdata;
    tick();
  endtask

  task automatic wr(input logic [31:0] addr, input logic [31:0] d);
    logic [31:0] dummy; longint unsigned t;
    obi(1'b1, addr, d, dummy, t);
  endtask

  task automatic rd(input logic [31:0] addr, output logic [31:0] d);
    longint unsigned t;
    obi(1'b0, addr, 32'd0, d, t);
  endtask

  function automatic longint sext(int v, int b);
    longint m = longint'(v) & ((longint'(1) << b) - 1);
    if (m & (longint'(1) << (b - 1))) m -= (longint'(1) << b);
    return m;
  endfunction

  function automatic void model_pass(int t, int b, bit acc);
    for (int n = 0; n < 16; n++) begin
      longint s = 0;
      for (int r = 0; r < 64; r++) s += longint'(W[t][r][n]) * sext(X[r], b);
      ACC[n] = (acc ? ACC[n] : 0) + s;
    end
  endfunction

  task automatic check_results(string tag);
    logic [31:0] lo, hi;
    for (int n = 0; n < 16; n++) begin
      rd(32'hC000_0040 + 4 * n, lo);
      rd(32'hC000_0080 + 4 * n, hi);
      checks++;
      if ({hi[15:0], lo} !== ACC[n][47:0] || hi[31:16] !== {16{ACC[n][47]}}) begin
        if (errors < 10) $display("%s: col %0d got %0h_%08h expected %0d", tag, n, hi, lo, ACC[n]);
        errors++;
      end
    end
  endtask

  task automatic wait_done_cnt(int target);
    logic [31:0] c;
    do rd(32'hC000_000C, c); while (int'(c) < target);
  endtask

  initial begin
    logic [31:0] s, c0;
    longint unsigned t_go1, t_go2, t_xw;
    int passes = 0;
    req = '0;
    for (int n = 0; n < 16; n++) ACC[n] = 0;
    repeat (3) @(posedge clk);
    #1 rst_n = 1'b1;
    tick();
    // weights for 6 tiles through the 64 KiB window
    for (int t = 0; t < 6; t++)
      for (int r = 0; r < 64; r++)
        for (int q = 0; q < 4; q++) begin
          for (int k = 0; k < 4; k++) W[t][r][4*q+k] = byte'($urandom());
          wr(32'hC002_0000 + (t << 10) + (r << 4) + (q << 2),
             {W[t][r][4*q+3], W[t][r][4*q+2], W[t][r][4*q+1], W[t][r][4*q]});
        end
    // 1) single passes with polling of STATUS.done and readback
    for (int it = 0; it < 12; it++) begin
      int b = (it % 3 == 0) ? 8 : ((it % 3 == 1) ? 16 : 24);
      int t = it % 6;
      bit acc = (it % 4 == 3);
      for (int r = 0; r < 64; r++) begin
        X[r] = int'($urandom() & ((32'd1 << b) - 1));
        wr(32'hC000_1000 + 4 * r, X[r]);
      end
      wr(32'hC000_0000, 32'h8000_0000 | (32'(acc) << 16) | (32'(b) << 8) | 32'(t));
      model_pass(t, b, acc);
      passes++;
      do rd(32'hC000_0004, s); while (!s[1]);
      check_results($sformatf("single %0d", it));
    end
    rd(32'hC000_000C, c0);
    if (int'(c0) != passes) begin $display("DONE_CNT %0d != %0d", c0, passes); errors++; end
    // 2) back-to-back: the second go must be granted exactly D (=16) edges after the first
    begin
      logic [31:0] dummy;
      obi(1'b1, 32'hC000_0000, 32'h8000_0000 | (16 << 8) | 0, dummy, t_go1);
      obi(1'b1, 32'hC000_0000, 32'h8001_0000 | (16 << 8) | 1, dummy, t_go2);
      model_pass(0, 16, 0); model_pass(1, 16, 1);
      passes += 2;
      checks++;
      if (t_go2 - t_go1 != 16) begin $display("back-to-back go spacing %0d != 16", t_go2 - t_go1); errors++; end
      wait_done_cnt(passes);
      check_results("back-to-back accumulate");
    end
    // 3) an X write right after go is held until the pass has read X (pass uses the old X)
    begin
      logic [31:0] dummy;
      int old_x0 = X[0];
      obi(1'b1, 32'hC000_0000, 32'h8000_0000 | (8 << 8) | 2, dummy, t_go1);
      obi(1'b1, 32'hC000_1000, 32'h0000_007F, dummy, t_xw);
      checks++;
      if (t_xw - t_go1 < 10) begin $display("X write granted %0d edges after go (< 10)", t_xw - t_go1); errors++; end
      X[0] = old_x0;                                  // the pass must have used the old value
      model_pass(2, 8, 0);
      passes++;
      wait_done_cnt(passes);
      check_results("x-hold");
      X[0] = 32'h7F;
    end
    // 4) interrupt
    begin
      wr(32'hC000_0008, 32'h1);
      wr(32'hC000_0004, 32'h2);                       // clear done (W1C)
      checks++;
      if (irq) begin $display("irq set after clear"); errors++; end
      wr(32'hC000_0000, 32'h8000_0000 | (8 << 8) | 3);
      model_pass(3, 8, 0);
      passes++;
      wait_done_cnt(passes);
      checks++;
      if (!irq) begin $display("irq not raised"); errors++; end
      check_results("irq");
    end
    $display("RESULT: %0d checks, %0d errors, %0d passes", checks, errors, passes);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
