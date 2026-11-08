// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// End-to-end streaming testbench for the window path (R5b) in cim_endpoint. A 3-s speech stream is
// fed hop by hop, as firmware does: 320 samples into the audio ring (which also updates the hop
// maximum), the front-end on the new frame, then the sequencer with the window path, and from hop
// 49 on the network. Checks against model/stream_int.py:
//   per hop       lmx16 (SEQ_LMX), the window maximum (SEQ_WMAX), the recompute-all flag;
//   per decision  all 147 FEAT words (the window's 490 features), the 12 logits and the class.
// Then the window maximum's path to SEQ_LMX: for edge values and every 997th v, a one-sample stream
// (after a stream reset) must give SEQ_LMX = log2fix16(max(v, 1)). (cim_log2fix16 itself is checked
// exhaustively by tb_cim_log2fix16.) +define+SHORT: only the first 60 hops, for gate-level simulation.
// DED = 0: cim_fe front-end (vectors tb/stream_vec_cim), DED = 1: B3 (tb/stream_vec_ded).
// Vectors: python model/gen_fe_vectors.py, gen_seq_vectors.py, gen_stream_vectors.py.
`timescale 1ns / 1ps

module tb_cim_stream #(
    parameter bit DED = 1'b0,
    parameter bit S   = 1'b0,                           // the baseline S in the sequencer (SEQ_S)
    parameter bit B1  = 1'b0                            // with S: StreamFuse B in block 1 (SEQ_B1)
);
  import xheep_obi_pkg::*;
  localparam logic [31:0] BASE = 32'hC004_0000;
  localparam int NT = 61, T0 = 8, HOP = 320, NS = 48000, NH = NS / HOP, ND = NH - 49;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq;

  cim_endpoint #(.TILES(128), .FE_DED(DED), .SEQ_S(S), .SEQ_B1(B1)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));

  logic [31:0] fe_tiles [8 * 256];
  logic [31:0] tiles [NT * 256];
  logic [31:0] params [640 * 3];
  logic [31:0] seq_meta [22];
  logic [15:0] samples [NS];
  logic [31:0] hops [3 * NH];
  logic [7:0]  feat_exp [ND * 490];
  logic [7:0]  logits_exp [ND * 12];
  logic [31:0] meta [2 + ND];
  logic [31:0] log2_exp [32768];
`ifdef SHORT
  localparam int NH_RUN = 60;
`else
  localparam int NH_RUN = NH;
`endif
  int          errors = 0;
  longint unsigned edge_cnt = 0;
  always @(posedge clk) edge_cnt <= edge_cnt + 1;

  task automatic tick(); @(posedge clk); #1; endtask

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

  initial begin
    string vec;
    logic [31:0] d, st, ctrl;
    longint unsigned t_go, cyc;
    longint unsigned c_one_min = '1, c_one_max = 0, c_all_min = '1, c_all_max = 0;    // window path only
    longint unsigned n_one_min = '1, n_one_max = 0, n_all_min = '1, n_all_max = 0;    // + network
    int n_chk_feat = 0, n_dec = 0, bad_hop = 0, bad_feat = 0, bad_logit = 0;
    vec = DED ? "tb/stream_vec_ded" : "tb/stream_vec_cim";
    if ($value$plusargs("VEC=%s", vec)) $display("vectors from %s", vec);  // tb/stream_vec_edge_*: extreme input
    req = '0;
    $readmemh("tb/fe_vec/tiles.hex", fe_tiles);
    $readmemh("tb/seq_vec/tiles.hex", tiles);
    $readmemh("tb/seq_vec/params.hex", params);
    $readmemh("tb/seq_vec/meta.hex", seq_meta);
    $readmemh({vec, "/samples.hex"}, samples);
    $readmemh({vec, "/hops.hex"}, hops);
    $readmemh({vec, "/feat.hex"}, feat_exp);
    $readmemh({vec, "/logits.hex"}, logits_exp);
    $readmemh({vec, "/meta.hex"}, meta);
    $readmemh("tb/stream_vec_cim/log2.hex", log2_exp);
    if (meta[0] != NH || meta[1] != ND) begin $display("ERROR: vectors have %0d hops, %0d decisions", meta[0], meta[1]); $finish; end
    repeat (4) @(posedge clk);
    #1 rst_n = 1'b1;
    tick();
    for (int t = 0; t < 8; t++)                                // front-end and DCT tiles 0..7
      for (int i = 0; i < 256; i++) wr(BASE + 32'h2_0000 + (32'(t) << 10) + (32'(i) << 2), fe_tiles[t * 256 + i]);
    for (int t = 0; t < NT; t++)
      for (int i = 0; i < 256; i++) wr(BASE + 32'h2_0000 + (32'(T0 + t) << 10) + (32'(i) << 2), tiles[t * 256 + i]);
    for (int e = 0; e < 640; e++)
      for (int f = 0; f < 3; f++) wr(BASE + 32'h8000 + (32'(e) << 4) + (32'(f) << 2), params[e * 3 + f]);
    wr(BASE + 32'h28, seq_meta[1]);
    wr(BASE + 32'h08, 32'h4);                                  // IRQ_EN: sequencer done
    $display("loaded 69 tiles and the parameters (FE_DED = %0d); streaming %0d hops", DED, NH);

    for (int h = 0; h < NH_RUN; h++) begin
      int frow = h % 49;
      for (int i = 0; i < HOP; i++)
        wr(BASE + 32'h2000 + 32'(4 * ((HOP * h + i) % 1024)), {16'd0, samples[HOP * h + i]});
      if (h >= 1) begin                                        // the new frame: samples 320 (h - 1) ..
        wr(BASE + 32'h10, 32'h8000_0000 | 32'((HOP * (h - 1)) % 1024));
        do rd(BASE + 32'h14, d); while (!d[1]);
        wr(BASE + 32'h14, 32'h2);
      end
      ctrl = 32'h8000_0000 | 32'h4000_0000 | (h < 49 ? 32'h2000_0000 : 32'h0) | (h >= 1 ? 32'h1000_0000 : 32'h0) |
             (h == 0 ? 32'h0080_0000 : 32'h0) | (32'(frow) << 16) | 32'(T0);
      obi(1'b1, BASE + 32'h20, ctrl, d, t_go);
      while (!irq) tick();
      cyc = edge_cnt - t_go;
      rd(BASE + 32'h24, st);
      wr(BASE + 32'h24, 32'h2);
      if (st[2]) begin $display("ERROR: SEQ_STATUS error at hop %0d", h); errors++; end
      rd(BASE + 32'h2C, d);
      if (d !== hops[3 * h] || st[12] !== hops[3 * h + 1][0]) begin
        if (bad_hop < 5) $display("HOP %0d: lmx16 %0d all %0d, expected %0d all %0d", h, d, st[12], hops[3 * h], hops[3 * h + 1]);
        bad_hop++;
      end
      rd(BASE + 32'h30, d);
      if (d !== {{16{hops[3 * h + 2][15]}}, hops[3 * h + 2][15:0]}) begin
        if (bad_hop < 5) $display("HOP %0d: window maximum %0d, expected %0d", h, $signed(d), $signed(hops[3 * h + 2][15:0]));
        bad_hop++;
      end
      if (h < 49) begin
        if (st[12]) begin if (cyc < c_all_min) c_all_min = cyc; if (cyc > c_all_max) c_all_max = cyc; end
        else if (h >= 1) begin if (cyc < c_one_min) c_one_min = cyc; if (cyc > c_one_max) c_one_max = cyc; end
      end else begin
        int k = h - 49, bl = 0, bf = 0;
        if (st[12]) begin if (cyc < n_all_min) n_all_min = cyc; if (cyc > n_all_max) n_all_max = cyc; end
        else begin if (cyc < n_one_min) n_one_min = cyc; if (cyc > n_one_max) n_one_max = cyc; end
        for (int n = 0; n < 12; n++) begin
          rd(BASE + 32'h5000 + 32'(4 * n), d);
          if (d !== {{24{logits_exp[12 * k + n][7]}}, logits_exp[12 * k + n]}) bl++;
        end
        if (st[7:4] != meta[2 + k][3:0]) bl++;
        for (int r = 0; r < 49; r++) begin                     // ring row r holds window frame (r - frow) mod 49
          int j = (r - frow + 49) % 49;
          for (int w = 0; w < 3; w++) begin
            rd(BASE + 32'h4000 + (32'(r) << 4) + (32'(w) << 2), d);
            for (int b = 0; b < 4; b++)
              if (4 * w + b < 10 && d[8 * b +: 8] !== feat_exp[490 * k + 10 * j + 4 * w + b]) begin
                if (k == 0 && bf < 24) $display("  feat frame %0d (row %0d) k %0d: got %0d expected %0d", j, r, 4 * w + b,
                                                $signed(d[8 * b +: 8]), $signed(feat_exp[490 * k + 10 * j + 4 * w + b]));
                bf++;
              end
          end
        end
        if (bl || bf) begin
          if (bad_logit + bad_feat < 6) $display("DECISION %0d (hop %0d): %0d logit/class errors, %0d feature errors", k, h, bl, bf);
        end
        bad_logit += bl; bad_feat += bf; n_dec++; n_chk_feat += 490;
      end
    end
    errors += bad_hop + bad_feat + bad_logit;
    $display("hops %0d: %0d lmx16/maximum/flag mismatches", NH_RUN, bad_hop);
    begin                                                      // window maximum -> SEQ_LMX
      int bad_lg = 0, n_lg = 0;
      int edge_v [14] = '{-32768, -1, 0, 1, 2, 3, 255, 256, 1000, 16383, 16384, 32766, 32767, 12345};
      for (int i = 0; i < 14 + 33; i++) begin
        int v = (i < 14) ? edge_v[i] : 997 * (i - 14) + 1;
        wr(BASE + 32'h2000, 32'(v) & 32'hFFFF);                // one sample: the hop maximum is v
        obi(1'b1, BASE + 32'h20, 32'hE080_0000 | 32'(T0), d, t_go);   // window path only, stream reset
        while (!irq) tick();
        wr(BASE + 32'h24, 32'h2);
        rd(BASE + 32'h2C, d);
        if (d !== log2_exp[v > 0 ? v : 0]) begin
          if (bad_lg < 5) $display("LOG2: v %0d gives %0h, expected %0h", v, d, log2_exp[v > 0 ? v : 0]);
          bad_lg++;
        end
        n_lg++;
      end
      $display("log2fix16: %0d of %0d window maxima wrong", bad_lg, n_lg);
      errors += bad_lg;
    end
    $display("decisions %0d: %0d of %0d features wrong, %0d logit/class errors", n_dec, bad_feat, n_chk_feat, bad_logit);
    $display("cycles go -> interrupt: window path only %0d..%0d (new frame), %0d..%0d (all 49 frames); with the network %0d..%0d (new frame), %0d..%0d (all)",
             c_one_min, c_one_max, c_all_min, c_all_max, n_one_min, n_one_max, n_all_min, n_all_max);
    $display("RESULT: %0d errors", errors);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end

  initial begin
    #2s $display("TIMEOUT"); $display("FAIL"); $finish;
  end
endmodule
