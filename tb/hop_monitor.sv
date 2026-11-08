// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// The busy time of every hop of the live audio path: from the end of a hop (cim_audio leaves HC_IDLE)
// to the end of its processing (back to HC_IDLE). Bound from outside the RTL, simulation only; it drives
// nothing. The lowest real-time clock follows from the worst hop (docs/TIMING_SENSITIVITY.md).
`timescale 1ns / 1ps
module hop_monitor (input logic clk_i, input logic en_i, input logic [2:0] hc_i, input logic ovr_i, input logic full_i);
  longint unsigned t = 0, max_all = 0, max_net = 0, max_nonet = 0, n = 0, sum_net = 0, n_net = 0, first_net = 0;
  logic [2:0] hc_q = 3'd0;
  always_ff @(posedge clk_i) begin
    hc_q <= hc_i;
    if (en_i && hc_i != 3'd0) t <= (hc_q == 3'd0) ? 1 : t + 1;
    if (hc_i == 3'd0 && hc_q != 3'd0) begin
      n <= n + 1;
      if (t > max_all) max_all <= t;
      if (full_i) begin
        if (t > max_net) max_net <= t;
        sum_net <= sum_net + t;
        n_net <= n_net + 1;
      end else if (t > max_nonet) max_nonet <= t;
    end
  end
  final $display("HOPMON %0d hops: worst %0d cycles; with the network: worst %0d, mean %0d; without: worst %0d",
                 n, max_all, max_net, n_net == 0 ? 0 : sum_net / n_net, max_nonet);
endmodule

bind cim_audio hop_monitor i_hop_monitor (.clk_i(clk_i), .en_i(en_i), .hc_i(3'(hc)), .ovr_i(ovr_o), .full_i(hc_full));
