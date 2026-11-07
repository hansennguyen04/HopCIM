// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Gate-level check of the hop controller and decimator: cim_audio's RTL and its Vivado funcsim netlist
// (module cim_audio_nl) side by side on the same inputs, every output compared at every falling edge.
// Inputs: pseudo-random PDM bits, front-end and sequencer handshakes with pseudo-random delays, and,
// after the first full window, the stride changing while the stream runs (k = 2, 4, 16, 3, 1), a
// network switched off for a while, then a restart. Run by sim/run_audio_nl.sh under xsim, after the
// 100-ns global set/reset of the netlist. Values the RTL still holds as X (unreset filter states) are
// not compared; the summary says how many comparisons were made.
`timescale 1ns / 1ps

module tb_cim_audio_nl #(
    parameter int unsigned CLKDIV = 15                // as the netlist was synthesised
);
  localparam int unsigned HOP_CYCLES = 320 * 64 * CLKDIV;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  logic en = 1'b0, net = 1'b0, pdm_data = 1'b0;
  logic [3:0] km1 = 4'd0;
  logic [6:0] tbase = 7'd8;
  logic fe_ok = 1'b0, fe_done = 1'b0, seq_ok = 1'b0, seq_done = 1'b0, clr_pend = 1'b0, clr_ovr = 1'b0;
  logic [3:0] seq_class = 4'd0;

  typedef struct packed {
    logic        pdm_clk, ring_we;
    logic [9:0]  ring_addr;
    logic [15:0] ring_data, bmax;
    logic        fe_go;
    logic [9:0]  fe_base;
    logic        seq_go;
    logic [31:0] seq_word;
    logic        pend, ovr;
    logic [3:0]  cls;
    logic [15:0] ndec;
    logic [31:0] nhop;
  } out_t;
  out_t a, b;

  cim_audio #(.TW(7), .CLKDIV(CLKDIV)) u_rtl (
      .clk_i(clk), .rst_ni(rst_n), .en_i(en), .net_i(net), .net_km1_i(km1), .tbase_i(tbase),
      .pdm_clk_o(a.pdm_clk), .pdm_data_i(pdm_data),
      .ring_we_o(a.ring_we), .ring_addr_o(a.ring_addr), .ring_data_o(a.ring_data), .bmax_o(a.bmax),
      .fe_ok_i(fe_ok), .fe_done_i(fe_done), .fe_go_o(a.fe_go), .fe_base_o(a.fe_base),
      .seq_ok_i(seq_ok), .seq_done_i(seq_done), .seq_class_i(seq_class), .seq_go_o(a.seq_go), .seq_word_o(a.seq_word),
      .clr_pend_i(clr_pend), .clr_ovr_i(clr_ovr), .pend_o(a.pend), .ovr_o(a.ovr), .class_o(a.cls), .ndec_o(a.ndec), .nhop_o(a.nhop));

  cim_audio_nl u_nl (
      .clk_i(clk), .rst_ni(rst_n), .en_i(en), .net_i(net), .net_km1_i(km1), .tbase_i(tbase),
      .pdm_clk_o(b.pdm_clk), .pdm_data_i(pdm_data),
      .ring_we_o(b.ring_we), .ring_addr_o(b.ring_addr), .ring_data_o(b.ring_data), .bmax_o(b.bmax),
      .fe_ok_i(fe_ok), .fe_done_i(fe_done), .fe_go_o(b.fe_go), .fe_base_o(b.fe_base),
      .seq_ok_i(seq_ok), .seq_done_i(seq_done), .seq_class_i(seq_class), .seq_go_o(b.seq_go), .seq_word_o(b.seq_word),
      .clr_pend_i(clr_pend), .clr_ovr_i(clr_ovr), .pend_o(b.pend), .ovr_o(b.ovr), .class_o(b.cls), .ndec_o(b.ndec), .nhop_o(b.nhop));

  // deterministic stimulus
  int unsigned seed = 32'h4B48_4F50;
  function automatic int unsigned rnd(int unsigned n);
    seed = seed * 32'd1664525 + 32'd1013904223;
    return (seed >> 8) % n;
  endfunction

  always @(negedge a.pdm_clk) pdm_data <= rnd(2) == 1;

  // handshakes: driven from the RTL's starts, the same for both
  int fe_wait = -1, seq_wait = -1, n_fe = 0, n_seq = 0, n_seq_net = 0, n_dec = 0;
  always @(posedge clk) begin
    fe_ok <= rnd(4) != 0;
    seq_ok <= rnd(4) != 0;
    seq_class <= 4'(rnd(12));
    clr_ovr <= rnd(100000) == 0;
    clr_pend <= (a.pend === 1'b1) && (rnd(8) == 0);
    fe_done <= 1'b0;
    seq_done <= 1'b0;
    if (a.fe_go === 1'b1) begin fe_wait <= 8 + int'(rnd(2000)); n_fe <= n_fe + 1; end
    else if (fe_wait == 0) begin fe_done <= 1'b1; fe_wait <= -1; end
    else if (fe_wait > 0) fe_wait <= fe_wait - 1;
    if (a.seq_go === 1'b1) begin
      seq_wait <= 8 + int'(rnd(a.seq_word[29] ? 200 : 20000));
      n_seq <= n_seq + 1;
      if (!a.seq_word[29]) n_seq_net <= n_seq_net + 1;
    end
    else if (seq_wait == 0) begin seq_done <= 1'b1; seq_wait <= -1; end
    else if (seq_wait > 0) seq_wait <= seq_wait - 1;
  end

  longint unsigned cmp = 0, bad = 0, skipped = 0;
  bit check_on = 1'b0;
  always @(negedge clk) if (check_on) begin
    if ($isunknown(a)) begin                         // compare the bits the RTL knows
      for (int i = 0; i < $bits(out_t); i++)
        if (a[i] !== 1'bx && a[i] !== 1'bz) begin
          cmp++;
          if (a[i] !== b[i]) bad++;
        end else skipped++;
    end else begin
      cmp += $bits(out_t);
      if (a !== b) begin
        if (bad < 10) $display("t=%0t rtl %h netlist %h (hop %0d)", $time, a, b, a.nhop);
        bad++;
      end
    end
  end

  task automatic hops(int n);
    repeat (n * HOP_CYCLES) @(posedge clk);
    $display("t = %0t: hop %0d, %0d decisions, %0d bits compared, %0d mismatches", $time, a.nhop, a.ndec, cmp, bad);
  endtask

  initial begin
    #200ns;                                          // past the netlist's global set/reset
    @(posedge clk);
    #1 rst_n = 1'b1;
    repeat (4) @(posedge clk);
    check_on = 1'b1;
    #1 net = 1'b1; en = 1'b1;
    hops(52);                                        // first full window, k = 1
    #1 km1 = 4'd1;  hops(6);                         // k = 2
    #1 km1 = 4'd3;  hops(9);                         // k = 4
    #1 km1 = 4'd15; hops(18);                        // k = 16
    #1 km1 = 4'd2;  hops(7);                         // k = 3
    #1 net = 1'b0;  hops(3);                         // no network
    #1 net = 1'b1; km1 = 4'd0; hops(3);              // k = 1
    #1 en = 1'b0;   hops(1);                         // stop
    #1 km1 = 4'd3; en = 1'b1; hops(4);               // restart
    check_on = 1'b0;
    $display("%0d output bits compared, %0d skipped (RTL X); %0d mismatches", cmp, skipped, bad);
    $display("front-end starts %0d, sequencer starts %0d, with the network %0d; decisions in the last stream %0d, overrun %b",
             n_fe, n_seq, n_seq_net, a.ndec, a.ovr);
    if (bad == 0 && n_seq_net > 10) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
