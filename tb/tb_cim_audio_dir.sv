// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Directed bench of cim_audio's hop controller, alone: the paths the end-to-end benches cannot reach because the real
// front-end and sequencer are always ready in time. Pseudo-random PDM bits, and a modelled front-end and sequencer
// whose starts stall and whose completions come after random delays. Checks:
//   per hop: one front-end start (none at hop 0), one sequencer start, in that order, only when allowed;
//     the frame base 320 (h - 1); the sequencer word's fields (frame row h mod 49, new frame, stream reset, network);
//     the hop maximum, against the samples written to the ring; the ring address;
//   decisions: at hops 49 + n k only, each with its class, one count and one pending flag;
//   overrun: a sequencer that finishes after the next hop's end sets the flag; the flag clears; the hop was skipped and
//     counts as documented in LIVE_AUDIO.md section 3;
//   restart: counters, stride and stream state start over, with the first hop's stream reset.
`timescale 1ns / 1ps

module tb_cim_audio_dir #(
    parameter int unsigned CLKDIV = 5
);
  localparam int unsigned HOP_CYCLES = 320 * 64 * CLKDIV;
  logic clk = 1'b0, rst_n = 1'b0, en = 1'b0, net = 1'b1, pdm_data = 1'b0, pdm_clk;
  logic [3:0] km1 = 4'd0;
  logic [6:0] tbase = 7'd8;
  logic ring_we, fe_go, seq_go, fe_ok = 1'b0, fe_done = 1'b0, seq_ok = 1'b0, seq_done = 1'b0, clr_pend = 1'b0, clr_ovr = 1'b0, pend, ovr;
  logic [9:0] ring_addr, fe_base;
  logic [15:0] ring_data, bmax, ndec;
  logic [31:0] seq_word, nhop;
  logic [3:0] seq_class = 4'd0, cls;
  always #5 clk = ~clk;

  cim_audio #(.TW(7), .CLKDIV(CLKDIV)) dut (
      .clk_i(clk), .rst_ni(rst_n), .en_i(en), .net_i(net), .net_km1_i(km1), .tbase_i(tbase),
      .pdm_clk_o(pdm_clk), .pdm_data_i(pdm_data),
      .ring_we_o(ring_we), .ring_addr_o(ring_addr), .ring_data_o(ring_data), .bmax_o(bmax),
      .fe_ok_i(fe_ok), .fe_done_i(fe_done), .fe_go_o(fe_go), .fe_base_o(fe_base),
      .seq_ok_i(seq_ok), .seq_done_i(seq_done), .seq_class_i(seq_class), .seq_go_o(seq_go), .seq_word_o(seq_word),
      .clr_pend_i(clr_pend), .clr_ovr_i(clr_ovr), .pend_o(pend), .ovr_o(ovr), .class_o(cls), .ndec_o(ndec), .nhop_o(nhop));

  int unsigned seed = 32'h4155_4449;
  function automatic int unsigned rnd(int unsigned n);
    seed = seed * 32'd1664525 + 32'd1013904223;
    return (seed >> 8) % n;
  endfunction

  int errors = 0, checks = 0, stalls_fe = 0, stalls_seq = 0;
  int h = 0;                                         // the hop being processed
  int fe_wait = 0, seq_wait = 0, seq_extra = 0;
  bit fe_run = 1'b0, seq_run = 1'b0;
  int exp_samples = 0;
  logic signed [15:0] run_max = -32768;
  logic [15:0] exp_bmax [$];
  logic [3:0] exp_class = 4'd0;
  logic [15:0] ndec_q = 16'd0;
  int fe_starts = 0, seq_starts = 0, decisions = 0;

  task automatic check(input bit ok, input string what);
    checks++;
    if (!ok) begin
      errors++;
      if (errors < 12) $display("ERROR at %0t: %s", $time, what);
    end
  endtask

  always @(negedge pdm_clk) pdm_data = rnd(2) == 1;

  // the ring and the hop maximum
  always @(posedge clk) if (rst_n && en && ring_we) begin
    check(ring_addr == 10'(exp_samples), $sformatf("ring address %0d, expected %0d", ring_addr, exp_samples % 1024));
    if ($signed(ring_data) > run_max) run_max = $signed(ring_data);
    exp_samples++;
    if (exp_samples % 320 == 0) begin
      exp_bmax.push_back(16'(run_max));
      run_max = -32768;
    end
  end

  // the modelled front-end and sequencer
  always @(negedge clk) begin
    fe_ok <= !fe_run && !seq_run && (rnd(4) != 0);
    seq_ok <= !fe_run && !seq_run && (rnd(4) != 0);
    fe_done <= 1'b0;
    seq_done <= 1'b0;
    if (rst_n && en) begin
      if (3'(dut.hc) == 3'd1 && !fe_ok) stalls_fe++;       // HC_FE
      if (3'(dut.hc) == 3'd3 && !seq_ok) stalls_seq++;     // HC_SEQ
    end
    if (fe_run) begin
      if (fe_wait == 0) begin fe_run = 1'b0; fe_done <= 1'b1; end
      else fe_wait--;
    end
    if (seq_run) begin
      if (seq_wait == 0) begin
        seq_run = 1'b0;
        seq_done <= 1'b1;
        exp_class = 4'(rnd(12));
        seq_class <= exp_class;
      end else seq_wait--;
    end
  end

  always @(posedge clk) if (rst_n && en) begin
    if (fe_go) begin
      check(fe_ok, "front-end start not allowed");
      check(h >= 1, "a front-end start at hop 0");
      // the ring runs on in real time, also over a skipped hop: the frame before the hop that just ended
      check(fe_base == 10'(320 * (exp_samples / 320 - 2)), $sformatf("hop %0d: frame base %0d", h, fe_base));
      fe_starts++;
      fe_run = 1'b1;
      fe_wait = int'(rnd(4000));
    end
    if (seq_go) begin
      bit full = net && (h >= 49) && ((h - 49) % (int'(km1) + 1) == 0);
      check(seq_ok, "sequencer start not allowed");
      check(!fe_run, "sequencer start before the front-end ended");
      check(seq_word[31:30] == 2'b11, "sequencer word: go and window path");
      check(seq_word[29] == !full, $sformatf("hop %0d: network bit %0d", h, !seq_word[29]));
      check(seq_word[28] == (h >= 1), $sformatf("hop %0d: new-frame bit", h));
      check(seq_word[23] == (h == 0), $sformatf("hop %0d: stream-reset bit", h));
      check(seq_word[21:16] == 6'(h % 49), $sformatf("hop %0d: frame row %0d", h, seq_word[21:16]));
      check(seq_word[6:0] == tbase, "first tile");
      check(exp_bmax.size() != 0, "a sequencer start with no finished hop");
      if (exp_bmax.size() != 0) begin
        logic [15:0] b = exp_bmax.pop_front();
        check(bmax == b, $sformatf("hop %0d: hop maximum %0d, expected %0d", h, $signed(bmax), $signed(b)));
      end
      seq_starts++;
      seq_run = 1'b1;
      seq_wait = int'(rnd(20000)) + seq_extra;
      seq_extra = 0;
      h++;
    end
  end

  // every decision: one count, its class, its pending flag
  always @(negedge clk) if (rst_n) begin
    if (en && ndec != ndec_q && ndec != 16'd0) begin
      decisions++;
      check(ndec == ndec_q + 16'd1, $sformatf("the decision count stepped from %0d to %0d", ndec_q, ndec));
      check(cls == exp_class, $sformatf("decision %0d: class %0d, expected %0d", ndec, cls, exp_class));
      check(pend, "a decision without its pending flag");
    end
    ndec_q <= ndec;
  end

  task automatic restart(input int k, input bit with_net);
    en = 1'b0;
    repeat (3) @(negedge clk);
    while (fe_run || seq_run) @(negedge clk);
    repeat (3) @(negedge clk);
    km1 = 4'(k - 1);
    net = with_net;
    h = 0;
    exp_samples = 0;
    run_max = -32768;
    exp_bmax.delete();
    fe_starts = 0;
    seq_starts = 0;
    en = 1'b1;
  endtask

  task automatic wait_hops(input int n);
    while (h < n) @(negedge clk);
  endtask

  task automatic check_stream(input int k, input string name);
    int exp_dec = (h > 49) ? (h - 49 + k - 1) / k : 0;
    while (seq_run) @(negedge clk);
    repeat (2) @(negedge clk);
    check(int'(ndec) == exp_dec, $sformatf("%s: %0d decisions after %0d hops, expected %0d", name, ndec, h, exp_dec));
    check(fe_starts == h - 1, $sformatf("%s: %0d front-end starts after %0d hops", name, fe_starts, h));
    check(seq_starts == h, $sformatf("%s: %0d sequencer starts after %0d hops", name, seq_starts, h));
  endtask

  initial begin
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    repeat (3) @(negedge clk);

    // a decision every hop, with stalls and random processing times
    en = 1'b1;
    wait_hops(60);
    check_stream(1, "every hop");
    check(pend, "no pending flag after decisions");
    clr_pend = 1'b1; @(negedge clk); clr_pend = 1'b0;
    @(negedge clk);
    check(!pend, "the pending flag did not clear");
    check(!ovr, "an overrun without cause");

    // a decision every 2 hops, then every 4: only those hops carry the network (word checked above)
    restart(2, 1'b1);
    wait_hops(60);
    check_stream(2, "every 2 hops");
    restart(4, 1'b1);
    wait_hops(70);
    check_stream(4, "every 4 hops");

    // no network: the window path alone, and no decision
    restart(1, 1'b0);
    wait_hops(55);
    while (seq_run) @(negedge clk);
    repeat (2) @(negedge clk);
    check(ndec == 16'd0, "a decision with the network off");

    // overrun: a sequencer that takes longer than a hop. The flag sets at the next hop's end, that hop is skipped.
    restart(1, 1'b1);
    wait_hops(52);
    begin
      int h_before, nhop_before;
      seq_extra = int'(HOP_CYCLES) + 2000;
      wait_hops(53);                                 // this hop's start carried the long delay
      nhop_before = int'(nhop);
      h_before = h;
      check(!ovr, "the flag before the long hop ended");
      while (!ovr) @(negedge clk);
      check(int'(nhop) == nhop_before, "a skipped hop was counted");
      while (seq_run) @(negedge clk);
      check(ovr, "the overrun flag is sticky");
      clr_ovr = 1'b1; @(negedge clk); clr_ovr = 1'b0;
      @(negedge clk);
      check(!ovr, "the overrun flag did not clear");
      // the block maximum of the skipped hop was never taken: keep the queue aligned with the hardware
      if (exp_bmax.size() != 0) exp_bmax.delete();
      wait_hops(h_before + 1);
      check(!ovr, "a new overrun without cause");
    end

    // restart: everything starts over, with the stream reset at hop 0
    en = 1'b0;
    repeat (3) @(negedge clk);
    while (fe_run || seq_run) @(negedge clk);
    check(ovr == 1'b0, "overrun after the restart's stop");
    restart(1, 1'b1);
    repeat (3) @(negedge clk);
    check(nhop == 32'd0 && ndec == 16'd0, "the counters did not restart");
    wait_hops(52);
    check_stream(1, "after a restart");

    check(stalls_fe > 0 && stalls_seq > 0, "the starts never stalled, so this bench proved nothing about stalls");
    $display("RESULT: %0d checks, %0d errors; %0d decisions; starts stalled %0d cycles (front-end), %0d cycles (sequencer)", checks, errors, decisions, stalls_fe, stalls_seq);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end

  initial begin
    #10s $display("TIMEOUT"); $display("FAIL"); $finish;
  end
endmodule
