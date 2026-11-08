// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_engine, behavioural and with run-time timing: SIMULATION ONLY, for the timing sensitivity study
// (docs/TIMING_SENSITIVITY.md, sim/run_timing.sh). Same module name and ports as rtl/cim_engine.sv, which
// it replaces at compile time; the synthesised RTL is not touched. Results are computed exactly as the
// RTL does; only when they appear changes.
//
// Timing, from plusargs (defaults: the RTL's contract):
//   a pass with effective input width b, accepted at edge t0:
//     issue interval  D = max(+CIM_DMIN, ceil(b / +CIM_BPC) * +CIM_CPB) + +CIM_D0
//     the next pass may start at t0 + D; done and results at edge t0 + D + +CIM_LAT
//   +CIM_DMIN=8  +CIM_BPC=1 (input bits per step)  +CIM_CPB=1 (cycles per step)  +CIM_D0=0  +CIM_LAT=5
//
// Input hold. A bit-serial macro reads its input for the whole interval D, so this model samples the
// input bank at the end of it (edge t0 + D - 1) and counts every write to that bank in the meantime:
// such a write would be a hazard in a real macro, and here it changes the result. A short final
// message prints the count ("CIM_TM ... hazards").

module cim_engine #(
    parameter int unsigned TILES = 64,
    parameter int unsigned XB    = 24,
    parameter int unsigned ACCW  = 48,
    localparam int unsigned TW   = $clog2(TILES)
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,
    input  logic                   w_we_i,
    input  logic [TW-1:0]          w_tile_i,
    input  logic [5:0]             w_row_i,
    input  logic [1:0]             w_word_i,
    input  logic [31:0]            w_data_i,
    input  logic                   x_we_i,
    input  logic [5:0]             x_idx_i,
    input  logic [XB-1:0]          x_data_i,
    input  logic                   xw_we_i,
    input  logic                   xw_bank_i,
    input  logic [5:0]             xw_row_i,
    input  logic [7:0]             xw_mask_i,
    input  logic [XB-1:0]          xw_data_i [8],
    input  logic                   start_i,
    input  logic [TW-1:0]          tile_i,
    input  logic [4:0]             xbits_i,
    input  logic                   accum_i,
    input  logic                   xbank_i,
    output logic                   ready_o,
    output logic                   done_o,
    output logic                   busy_o,
    output logic signed [ACCW-1:0] res_o [16]
);
  localparam int unsigned N = 16;
  localparam int unsigned QD = 64;                   // passes in flight the model can hold

  int unsigned dmin = 8, bpc = 1, cpb = 1, d0 = 0, lat = 5;
  initial begin
    if ($value$plusargs("CIM_DMIN=%d", dmin)) begin end
    if ($value$plusargs("CIM_BPC=%d", bpc)) begin end
    if ($value$plusargs("CIM_CPB=%d", cpb)) begin end
    if ($value$plusargs("CIM_D0=%d", d0)) begin end
    if ($value$plusargs("CIM_LAT=%d", lat)) begin end
    if (dmin + lat < 2) begin dmin = 1; lat = 1; end
  end

  function automatic int unsigned eff_bits(input logic [4:0] b);
    if (b == 5'd0) return 1;
    if (int'(b) > int'(XB)) return XB;
    return int'(b);
  endfunction

  function automatic int unsigned pass_d(input int unsigned b);
    int unsigned steps = (b + bpc - 1) / bpc;
    int unsigned d = steps * cpb;
    if (d < dmin) d = dmin;
    return d + d0;
  endfunction

  function automatic longint sext_b(input logic [XB-1:0] v, input int unsigned b);
    logic [63:0] r;
    for (int k = 0; k < 64; k++) r[k] = (k < int'(b)) ? v[k] : v[b-1];
    return longint'(r);
  endfunction

  logic signed [7:0] w [TILES][64][16];
  logic [XB-1:0]     x [2][64];
  `ifdef CIM_ZERO_INIT
  initial for (int b = 0; b < 2; b++) for (int r = 0; r < 64; r++) x[b][r] = '0;
  `endif

  // passes in flight, in issue order
  longint unsigned now = 0;
  longint unsigned q_sample [QD], q_done [QD];
  int unsigned     q_tile [QD], q_b [QD], q_bank [QD];
  bit              q_acc [QD], q_ok [QD];
  longint          q_res [QD][N];
  int unsigned     q_head = 0, q_samp = 0, q_tail = 0;   // counters, modulo QD on use
  longint          acc [N];
  int unsigned     gap = 0;
  longint unsigned hazards = 0, passes = 0, max_inflight = 0, d_sum = 0;

  assign ready_o = (gap == 0);
  assign busy_o  = (q_head != q_tail);

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      done_o <= 1'b0;
      for (int n = 0; n < int'(N); n++) begin
        res_o[n] <= '0;
        acc[n] = 0;
      end
      q_head = 0;
      q_samp = 0;
      q_tail = 0;
      gap    = 0;
    end else begin
      // the sample of a pass is taken before this edge's writes land
      while (q_samp != q_tail && q_sample[q_samp % QD] <= now) begin
        int unsigned s = q_samp % QD;
        for (int n = 0; n < int'(N); n++) begin
          longint y = 0;
          for (int r = 0; r < 64; r++) y += longint'(w[q_tile[s]][r][n]) * sext_b(x[q_bank[s]][r], q_b[s]);
          acc[n] = (q_acc[s] ? acc[n] : 64'd0) + y;
          q_res[s][n] = acc[n];
        end
        q_ok[s] = 1'b1;
        q_samp++;
      end
      // a done is sampled high at edge q_done: it rises after edge q_done - 1
      done_o <= 1'b0;
      if (done_o) q_head++;                          // the pass whose done was seen at this edge
      begin
        int unsigned nh = q_head;
        if (nh != q_tail && q_done[nh % QD] == now + 1) begin
          done_o <= 1'b1;
          for (int n = 0; n < int'(N); n++) res_o[n] <= ACCW'(q_res[nh % QD][n]);
          if (!q_ok[nh % QD]) $display("CIM_TM ERROR: done before the sample of a pass");
        end
      end
      if (gap != 0) gap--;
      if (start_i && ready_o) begin
        int unsigned b = eff_bits(xbits_i), d = pass_d(b), t = q_tail % QD;
        q_tile[t]   = int'(tile_i);
        q_b[t]      = b;
        q_acc[t]    = accum_i;
        q_bank[t]   = int'(xbank_i);
        q_ok[t]     = 1'b0;
        q_sample[t] = now + longint'(d) - 1 > now ? now + longint'(d) - 1 : now + 1;
        q_done[t]   = now + longint'(d) + longint'(lat);
        q_tail++;
        gap = d - 1;
        passes++;
        d_sum += longint'(d);
        if (longint'(q_tail - q_head) > longint'(max_inflight)) max_inflight = longint'(q_tail - q_head);
        if (q_tail - q_head > QD) $display("CIM_TM ERROR: more than %0d passes in flight", QD);
      end
      now++;
    end
  end

  // writes: weights, and the two input ports (port 2 has priority, as in the RTL)
  always @(posedge clk_i) begin
    if (w_we_i) for (int k = 0; k < 4; k++) w[w_tile_i][w_row_i][4 * int'(w_word_i) + k] <= w_data_i[8 * k +: 8];
    if (xw_we_i) begin
      for (int l = 0; l < 8; l++) if (xw_mask_i[l]) begin
        logic [5:0] row = xw_row_i + {3'd0, 3'(l) - xw_row_i[2:0]};
        x[xw_bank_i][row] <= xw_data_i[l];
        for (int unsigned i = q_samp; i != q_tail; i++)
          if (q_bank[i % QD] == int'(xw_bank_i)) hazards++;
      end
    end else if (x_we_i) begin
      x[0][x_idx_i] <= x_data_i;
      for (int unsigned i = q_samp; i != q_tail; i++)
        if (q_bank[i % QD] == 0) hazards++;
    end
  end

  final $display("CIM_TM dmin %0d bpc %0d cpb %0d d0 %0d lat %0d: %0d passes, mean D %0.2f, at most %0d in flight, %0d hazards",
                 dmin, bpc, cpb, d0, lat, passes, passes == 0 ? 0.0 : real'(d_sum) / real'(passes), max_inflight, hazards);
endmodule
