// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// pdm_dec: PDM microphone interface and decimator of the live audio path (R7), bit-exact to
// model/pdm_dec.py decimate(). pdm_clk_o = clk / CLKDIV (1.024 MHz from 15.36 MHz), high for the
// last CLKDIV/2 + 1 cycles of each period. One data bit per rising edge of pdm_clk_o, through a
// 2-flip-flop synchroniser (as X-HEEP's pdm2pcm, which works with the Nexys A7's ADMP421 with its
// L/R select low). With x = 2 b - 1:
//   CIC  5 integrators per bit, chained in one cycle (as cumsum), 32-bit wrap-around; after every
//        64th bit the 5 combs (initial state 0) give c, and y1 = (c + 2^14) >> 15
//   FIR  PD_NTAP taps (pdm_tables.svh, Q14), one multiply-accumulate per cycle:
//        y = sat16((sum h[k] y1[n-k] + 2^13) >> 14)
//   DC blocker (the microphone's PDM has a DC offset): acc += ((y - y_prev) << 8) - (acc >> 8),
//        smp_o = sat16((acc + 2^7) >> 8), with a one-cycle smp_valid_o (16 kHz)
// Explicit-width logic, every signed expression with signed operands only (docs/SEQUENCER.md §8a,
// docs/WINDOW_PATH.md §4).

module pdm_dec #(
    parameter int unsigned CLKDIV = 15
) (
    input  logic               clk_i,
    input  logic               rst_ni,
    input  logic               en_i,                 // clock the microphone and decimate
    output logic               pdm_clk_o,
    input  logic               pdm_data_i,
    output logic               smp_valid_o,
    output logic signed [15:0] smp_o
);
`include "pdm_tables.svh"
  localparam int unsigned CW = $clog2(CLKDIV);
  localparam logic [CW-1:0] HI = CW'(CLKDIV / 2);    // pdm_clk_o rises when the count reaches HI

  // ------------------------------------------------------------------ PDM clock and bit
  logic [CW-1:0] ccnt;
  logic [1:0]    dsync;
  logic          bit_ev;                              // a new bit this cycle
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ccnt      <= '0;
      pdm_clk_o <= 1'b0;
      dsync     <= 2'b00;
    end else begin
      dsync <= {dsync[0], pdm_data_i};
      if (!en_i) begin
        ccnt      <= '0;
        pdm_clk_o <= 1'b0;
      end else begin
        ccnt      <= (ccnt == CW'(CLKDIV - 1)) ? '0 : ccnt + 1'b1;
        pdm_clk_o <= (ccnt >= HI - 1'b1) && (ccnt != CW'(CLKDIV - 1));
      end
    end
  end
  assign bit_ev = en_i && (ccnt == HI);

  // ------------------------------------------------------------------ CIC
  logic [31:0] ig [PD_NCIC];                         // integrators
  logic [31:0] dl [PD_NCIC];                         // comb delays
  logic [31:0] in_x, i_nx [PD_NCIC], c_nx [PD_NCIC + 1];
  logic [5:0]  bcnt;                                 // bit within the 64-bit block
  logic        dec_ev;                               // the 64th bit: a CIC output
  assign in_x = dsync[1] ? 32'd1 : 32'hFFFF_FFFF;
  always_comb begin
    i_nx[0] = ig[0] + in_x;
    for (int k = 1; k < int'(PD_NCIC); k++) i_nx[k] = ig[k] + i_nx[k-1];
    c_nx[0] = i_nx[PD_NCIC-1];
    for (int k = 0; k < int'(PD_NCIC); k++) c_nx[k+1] = c_nx[k] - dl[k];
  end
  assign dec_ev = bit_ev && (bcnt == 6'(PD_R - 1));

  logic signed [16:0] y1;                            // (c + 2^14) >> 15
  logic signed [31:0] c_out;
  assign c_out = $signed(c_nx[PD_NCIC]);
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < int'(PD_NCIC); k++) begin
        ig[k] <= 32'd0;
        dl[k] <= 32'd0;
      end
      bcnt <= 6'd0;
      y1   <= 17'sd0;
    end else if (bit_ev) begin
      for (int k = 0; k < int'(PD_NCIC); k++) ig[k] <= i_nx[k];
      bcnt <= bcnt + 6'd1;
      if (dec_ev) begin
        for (int k = 0; k < int'(PD_NCIC); k++) dl[k] <= c_nx[k];
        y1 <= 17'((33'(c_out) + 33'sd16384) >>> 15);
      end
    end
  end

  // ------------------------------------------------------------------ FIR
  logic signed [16:0] hist [PD_NTAP];                // y1[n], y1[n-1], ...
  logic [4:0]         mk;                            // 0 idle, 1..NTAP tap mk-1, NTAP+1 output
  logic [3:0]         ti;                            // tap index of step mk
  logic signed [37:0] acc, prod;
  logic signed [37:0] rnd;
  logic               dec_q;                         // y1 was just updated
  logic signed [15:0] y_fir, dc_prev;                // FIR output (saturated), its previous value
  logic signed [27:0] dc_acc, dc_nx, dc_out;         // DC blocker accumulator (Q8)
  assign ti   = ((mk == 5'd0) || (mk > 5'(PD_NTAP))) ? 4'd0 : 4'(mk - 5'd1);
  assign prod = 38'(hist[ti]) * 38'(PD_H[ti]);
  assign rnd  = (acc + 38'sd8192) >>> PD_HFRAC;
  assign y_fir  = (rnd > 38'sd32767) ? 16'sh7FFF : (rnd < -38'sd32768) ? 16'sh8000 : rnd[15:0];
  assign dc_nx  = dc_acc + ((28'(y_fir) - 28'(dc_prev)) <<< 8) - (dc_acc >>> 8);
  assign dc_out = (dc_nx + 28'sd128) >>> 8;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < int'(PD_NTAP); k++) hist[k] <= 17'sd0;
      mk          <= 5'd0;
      acc         <= 38'sd0;
      dec_q       <= 1'b0;
      smp_valid_o <= 1'b0;
      smp_o       <= 16'sd0;
      dc_acc      <= 28'sd0;
      dc_prev     <= 16'sd0;
    end else begin
      smp_valid_o <= 1'b0;
      dec_q       <= dec_ev;
      if (dec_q) begin                               // shift in y1, start the MAC
        hist[0] <= y1;
        for (int k = 1; k < int'(PD_NTAP); k++) hist[k] <= hist[k-1];
        acc <= 38'sd0;
        mk  <= 5'd1;
      end else if ((mk != 5'd0) && (mk <= 5'(PD_NTAP))) begin
        acc <= acc + prod;
        mk  <= mk + 5'd1;
      end else if (mk == 5'(PD_NTAP + 1)) begin       // round, saturate, DC block, output
        mk          <= 5'd0;
        dc_acc      <= dc_nx;
        dc_prev     <= y_fir;
        smp_valid_o <= 1'b1;
        smp_o       <= (dc_out > 28'sd32767) ? 16'sh7FFF : (dc_out < -28'sd32768) ? 16'sh8000 : dc_out[15:0];
      end
    end
  end

endmodule
