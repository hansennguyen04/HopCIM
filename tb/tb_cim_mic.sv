// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// End-to-end testbench of the live audio path (R7): a 3-s speech stream as PDM on the microphone
// pin (model/pdm_dec.py sigma-delta model; the next bit after every falling edge of pdm_clk_o).
// After loading tiles and parameters, the "CPU" only enables the audio path (AUD_CTRL) and, on
// every decision interrupt, reads the class and logits and clears the decision; the endpoint
// does all the rest (decimation, ring, front-end, window path, network) by itself. Checks:
//   every decision's 12 logits and class against model/gen_mic_vectors.py, no overrun, the hop
//   count, and the audio ring (read back through SAMPLES) against the golden decimated stream: the
//   ring slots not yet overwritten by the samples decimated after the stream (the microphone pin
//   is then 0) until the last decision.
// DED = 0: cim_fe front-end, 1: B3. CLKDIV: the PDM clock divider, 15 (15.36 MHz) or 5 (5.12 MHz, the
// lowest system clock with a 1.024 MHz PDM clock that still processes every hop in time).
// NET_K: the network runs every NET_K hops (AUD_CTRL[19:16] = NET_K - 1). The decisions must be the
// vectors' decisions 0, NET_K, 2 NET_K, ... (every hop's window has its own golden logits), each at
// hop 49 + n NET_K, with no other decision interrupt.
// Vectors: python model/gen_mic_vectors.py (and gen_fe_vectors.py, gen_seq_vectors.py for the tiles
// and parameters).
`timescale 1ns / 1ps

module tb_cim_mic #(
    parameter bit DED = 1'b0,
    parameter int unsigned CLKDIV = 15,
    parameter int unsigned NET_K = 1,
    parameter bit S = 1'b0                              // the baseline S in the sequencer (SEQ_S)
);
  import xheep_obi_pkg::*;
  localparam logic [31:0] BASE = 32'hC004_0000;
  localparam int NT = 61, T0 = 8, NB = 3072000, NS = 48000, ND = 101;
  localparam int ND_K = (ND - 1) / int'(NET_K) + 1;            // decisions expected with this stride

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq, pdm_clk, pdm_data = 1'b0;

  cim_endpoint #(.TILES(128), .FE_DED(DED), .PDM_CLKDIV(CLKDIV), .SEQ_S(S)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
                                                 .pdm_clk_o(pdm_clk), .pdm_data_i(pdm_data));

  logic [31:0] fe_tiles [8 * 256];
  logic [31:0] tiles [NT * 256];
  logic [31:0] params [640 * 3];
  logic [31:0] seq_meta [22];
  logic [0:0]  bits [NB];
  logic [15:0] samples [NS];
  logic [7:0]  logits_exp [ND * 12];
  logic [31:0] meta [1 + ND];
  int          bi = 0, errors = 0;
  bit          mic_on = 1'b0;
  longint unsigned edge_cnt = 0, t_bits_end = 0;
  always @(posedge clk) edge_cnt <= edge_cnt + 1;

  always @(negedge pdm_clk) if (mic_on) begin              // the microphone's next bit
    bi = bi + 1;
    pdm_data = (bi < NB) ? bits[bi][0] : 1'b0;
    if (bi == NB) t_bits_end = edge_cnt;
  end

  task automatic tick(); @(posedge clk); #1; endtask

  task automatic obi(input bit we, input logic [31:0] addr, input logic [31:0] wdata, output logic [31:0] rdata);
    req.req = 1'b1; req.we = we; req.be = 4'hF; req.addr = addr; req.wdata = wdata;
    do @(posedge clk); while (!rsp.gnt);
    #1 req.req = 1'b0;
    while (!rsp.rvalid) tick();
    rdata = rsp.rdata;
    tick();
  endtask

  task automatic wr(input logic [31:0] addr, input logic [31:0] d);
    logic [31:0] dummy;
    obi(1'b1, addr, d, dummy);
  endtask

  task automatic rd(input logic [31:0] addr, output logic [31:0] d);
    obi(1'b0, addr, 32'd0, d);
  endtask

  initial begin
    string tag;
    logic [31:0] d, st, hops;
    int k = 0, bad_dec = 0, bad_hop = 0, bad_ring = 0, n_post, n0;
    longint unsigned t_last = 0;
    tag = DED ? "ded" : "cim";
    req = '0;
    $readmemh("tb/fe_vec/tiles.hex", fe_tiles);
    $readmemh("tb/seq_vec/tiles.hex", tiles);
    $readmemh("tb/seq_vec/params.hex", params);
    $readmemh("tb/seq_vec/meta.hex", seq_meta);
    $readmemb("tb/mic_vec/bits.hex", bits);
    $readmemh("tb/mic_vec/samples.hex", samples);
    $readmemh({"tb/mic_vec/logits_", tag, ".hex"}, logits_exp);
    $readmemh({"tb/mic_vec/meta_", tag, ".hex"}, meta);
    if (meta[0] != ND) begin $display("ERROR: %0d decisions in the vectors", meta[0]); $finish; end
    repeat (4) @(posedge clk);
    #1 rst_n = 1'b1;
    tick();
    for (int t = 0; t < 8; t++)
      for (int i = 0; i < 256; i++) wr(BASE + 32'h2_0000 + (32'(t) << 10) + (32'(i) << 2), fe_tiles[t * 256 + i]);
    for (int t = 0; t < NT; t++)
      for (int i = 0; i < 256; i++) wr(BASE + 32'h2_0000 + (32'(T0 + t) << 10) + (32'(i) << 2), tiles[t * 256 + i]);
    for (int e = 0; e < 640; e++)
      for (int f = 0; f < 3; f++) wr(BASE + 32'h8000 + (32'(e) << 4) + (32'(f) << 2), params[e * 3 + f]);
    wr(BASE + 32'h28, seq_meta[1]);
    wr(BASE + 32'h08, 32'h8);                                  // IRQ_EN: decisions only
    pdm_data = bits[0][0];
    mic_on = 1'b1;
    wr(BASE + 32'h34, 32'h3 | (32'(T0) << 8) | (32'(NET_K - 1) << 16)); // audio path on, with the network
    rd(BASE + 32'h34, d);
    if (d !== (32'h3 | (32'(T0) << 8) | (32'(NET_K - 1) << 16))) begin $display("ERROR: AUD_CTRL reads %h", d); errors++; end
    $display("audio path enabled (FE_DED = %0d, PDM clock = clk / %0d, network every %0d hops): %0d PDM bits, %0d decisions expected",
             DED, CLKDIV, NET_K, NB, ND_K);

    while (k < ND_K) begin
      int bl, g;
      bl = 0;
      g = k * int'(NET_K);                                      // this decision in the vectors
      while (!irq) tick();
      rd(BASE + 32'h3C, hops);
      if (hops !== 32'(50 + g)) begin                           // decision of hop 49 + g, the last that ended
        if (bad_hop < 5) $display("DECISION %0d: at %0d hops, expected %0d", k, hops, 50 + g);
        bad_hop++;
      end
      rd(BASE + 32'h38, st);
      for (int n = 0; n < 12; n++) begin
        rd(BASE + 32'h5000 + 32'(4 * n), d);
        if (d !== {{24{logits_exp[12 * g + n][7]}}, logits_exp[12 * g + n]}) bl++;
      end
      if (st[7:4] != meta[1 + g][3:0]) bl++;
      if (st[31:16] != 16'(k + 1)) bl++;
      if (st[2]) begin $display("ERROR: overrun at decision %0d", k); errors++; end
      if (bl) begin
        if (bad_dec < 5) $display("DECISION %0d: %0d logit/class/count errors (class %0d, expected %0d)", k, bl, st[7:4], meta[1 + g]);
        bad_dec++;
      end
      wr(BASE + 32'h38, 32'h2);                                 // clear the decision
      t_last = edge_cnt;
      k++;
    end
    // the stream may end on hops without a network: let them pass, with no further decision
    rd(BASE + 32'h3C, hops);
    while (hops < 150 || t_bits_end == 0) begin
      repeat (200) tick();
      if (irq) begin $display("ERROR: an extra decision interrupt"); errors++; break; end
      rd(BASE + 32'h3C, hops);
    end
    if (hops == 150 && (ND - 1) % int'(NET_K) != 0) begin       // hop 149 has no network: its front-end
      repeat (12000) begin                                      // and window path finish, no decision
        tick();
        if (irq) begin $display("ERROR: an extra decision interrupt"); errors++; break; end
      end
    end
    rd(BASE + 32'h38, st);
    if (st[31:16] != 16'(ND_K)) begin $display("ERROR: %0d decisions counted, expected %0d", st[31:16], ND_K); errors++; end
    wr(BASE + 32'h34, 32'h0);                                   // stop
    mic_on = 1'b0;
    n_post = int'((edge_cnt - t_bits_end) / longint'(64 * CLKDIV)) + 2;  // samples after the stream, + margin
    n0 = NS - 1024 + n_post;
    rd(BASE + 32'h3C, hops);
    if (hops < 150) begin $display("ERROR: %0d hops processed", hops); errors++; end
    if (n0 > NS - 600) begin $display("ERROR: only %0d ring samples left to check", NS - n0); errors++; end
    for (int n = n0; n < NS; n++) begin
      rd(BASE + 32'h2000 + 32'(4 * (n % 1024)), d);
      if (d !== {{16{samples[n][15]}}, samples[n]}) begin
        if (bad_ring < 5) $display("RING sample %0d: got %0d expected %0d", n, $signed(d), $signed(samples[n]));
        bad_ring++;
      end
    end
    errors += bad_dec + bad_hop + bad_ring;
    $display("decisions: %0d of %0d wrong, %0d at a wrong hop; ring: %0d of %0d samples wrong; %0d hops; last decision at cycle %0d",
             bad_dec, ND_K, bad_hop, bad_ring, NS - n0, hops, t_last);
    if (NET_K > 1) begin                                        // a restart: the stride starts over at hop 49
      wr(BASE + 32'h38, 32'h6);
      wr(BASE + 32'h34, 32'h3 | (32'(T0) << 8) | (32'(NET_K - 1) << 16));
      while (!irq) tick();
      rd(BASE + 32'h3C, hops);
      rd(BASE + 32'h38, st);
      if (hops !== 32'd50 || st[31:16] != 16'd1) begin
        $display("ERROR: after a restart the first decision is at %0d hops, count %0d", hops, st[31:16]);
        errors++;
      end else $display("restart: first decision at hop 49, count 1");
      wr(BASE + 32'h38, 32'h2);
      wr(BASE + 32'h34, 32'h0);
    end
    $display("RESULT: %0d errors", errors);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end

  initial begin
    #5s $display("TIMEOUT"); $display("FAIL"); $finish;
  end
endmodule
