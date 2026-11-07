// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_audio: the live audio path (R7). The on-board PDM microphone feeds pdm_dec (16 kHz), whose
// samples go straight into the endpoint's audio ring; a hop controller then runs every hop
// without the CPU:
//   hop h = samples 320 h .. 320 h + 319. When its last sample is written:
//   - the hop's maximum sample is snapshotted (bmax_o) for the window path;
//   - h >= 1: the front-end starts on the frame of samples 320 (h - 1) .. 320 (h - 1) + 479;
//   - then the sequencer: window path (new frame if h >= 1, stream reset if h = 0, FEAT row of
//     frame 0 = h mod 49), and, from hop 49 on and with net_i, the network at hops
//     h = 49 + n (net_km1_i + 1), so one decision every net_km1_i + 1 hops. The front-end and the
//     window path still run at every hop: every decision's features equal the per-window reference;
//   - a decision latches the class and sets pend_o (the endpoint's decision interrupt).
// The sequencer's work per hop (about 100k cycles) is far below a hop (307,200 cycles at 15.36 MHz);
// a hop that ends before the previous one is processed sets ovr_o (never expected).
// Unsigned explicit-width logic only (docs/SEQUENCER.md §8a).

module cim_audio #(
    parameter int unsigned TW     = 7,
    parameter int unsigned CLKDIV = 15                // PDM clock = clk / CLKDIV (1.024 MHz at 15.36 MHz)
) (
    input  logic          clk_i,
    input  logic          rst_ni,
    input  logic          en_i,                       // run (a 0 -> 1 restarts the stream)
    input  logic          net_i,                      // run the network on full windows
    input  logic [3:0]    net_km1_i,                  // ... every net_km1_i + 1 hops (0: every hop)
    input  logic [TW-1:0] tbase_i,                    // first network tile
    output logic          pdm_clk_o,
    input  logic          pdm_data_i,
    // audio ring
    output logic          ring_we_o,
    output logic [9:0]    ring_addr_o,
    output logic [15:0]   ring_data_o,
    output logic [15:0]   bmax_o,                     // maximum sample of the last complete hop
    // front-end and sequencer
    input  logic          fe_ok_i,                    // a front-end start is possible now
    input  logic          fe_done_i,
    output logic          fe_go_o,
    output logic [9:0]    fe_base_o,
    input  logic          seq_ok_i,                   // a sequencer start is possible now
    input  logic          seq_done_i,
    input  logic [3:0]    seq_class_i,
    output logic          seq_go_o,
    output logic [31:0]   seq_word_o,                 // SEQ_CTRL word of the start
    // status
    input  logic          clr_pend_i,
    input  logic          clr_ovr_i,
    output logic          pend_o,                     // a new decision (sticky)
    output logic          ovr_o,                      // a hop overran (sticky)
    output logic [3:0]    class_o,                    // class of the last decision
    output logic [15:0]   ndec_o,                     // decisions since the start
    output logic [31:0]   nhop_o                      // hops since the start
);

  typedef enum logic [2:0] {HC_IDLE, HC_FE, HC_FEW, HC_SEQ, HC_SEQW} hstate_e;

  logic               sv;
  logic signed [15:0] smp;
  pdm_dec #(.CLKDIV(CLKDIV)) i_dec (
      .clk_i,
      .rst_ni,
      .en_i,
      .pdm_clk_o,
      .pdm_data_i,
      .smp_valid_o(sv),
      .smp_o      (smp)
  );

  logic [9:0]  wptr;                                  // ring index of the next sample
  logic [8:0]  hs;                                    // sample within the hop
  logic [15:0] blk;                                   // running maximum of the hop (signed)
  logic        hop_end, en_q;
  logic [5:0]  frow;                                  // h mod 49
  logic [3:0]  kcnt;                                  // hops since the last network, from hop 49 on
  logic        h_ge1, h_ge49;
  hstate_e     hc;
  logic        hc_frame, hc_first, hc_full;
  logic [5:0]  hc_frow;
  logic [15:0] smax;

  assign smax        = ($signed(smp) > $signed(blk)) ? smp : blk;
  assign hop_end     = sv && (hs == 9'd319);
  assign ring_we_o   = sv;
  assign ring_addr_o = wptr;
  assign ring_data_o = smp;
  assign fe_go_o     = (hc == HC_FE) && fe_ok_i;
  assign seq_go_o    = (hc == HC_SEQ) && seq_ok_i;
  assign seq_word_o  = 32'h8000_0000 | 32'h4000_0000 | ((hc_full && net_i) ? 32'h0 : 32'h2000_0000) |
                       (hc_frame ? 32'h1000_0000 : 32'h0) | (hc_first ? 32'h0080_0000 : 32'h0) |
                       (32'(hc_frow) << 16) | 32'(tbase_i);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      en_q      <= 1'b0;
      wptr      <= 10'd0;
      hs        <= 9'd0;
      blk       <= 16'h8000;
      bmax_o    <= 16'h8000;
      frow      <= 6'd0;
      kcnt      <= 4'd0;
      h_ge1     <= 1'b0;
      h_ge49    <= 1'b0;
      hc        <= HC_IDLE;
      hc_frame  <= 1'b0;
      hc_first  <= 1'b0;
      hc_full   <= 1'b0;
      hc_frow   <= 6'd0;
      fe_base_o <= 10'd0;
      pend_o    <= 1'b0;
      ovr_o     <= 1'b0;
      class_o   <= 4'd0;
      ndec_o    <= 16'd0;
      nhop_o    <= 32'd0;
    end else begin
      en_q <= en_i;
      if (clr_pend_i) pend_o <= 1'b0;
      if (clr_ovr_i) ovr_o <= 1'b0;
      if (en_i && !en_q) begin                        // start of a stream
        wptr   <= 10'd0;
        hs     <= 9'd0;
        blk    <= 16'h8000;
        frow   <= 6'd0;
        kcnt   <= 4'd0;
        h_ge1  <= 1'b0;
        h_ge49 <= 1'b0;
        hc     <= HC_IDLE;
        ndec_o <= 16'd0;
        nhop_o <= 32'd0;
      end else if (en_i) begin
        if (sv) begin
          wptr <= wptr + 10'd1;
          if (hop_end) begin
            hs     <= 9'd0;
            bmax_o <= smax;
            blk    <= 16'h8000;
          end else begin
            hs  <= hs + 9'd1;
            blk <= smax;
          end
        end
        unique case (hc)
          HC_IDLE: if (hop_end) begin
            hc_frame  <= h_ge1;
            hc_first  <= !h_ge1;
            hc_full   <= h_ge49 && (kcnt == 4'd0);        // this hop runs the network
            if (h_ge49) kcnt <= (kcnt >= net_km1_i) ? 4'd0 : kcnt + 4'd1;
            hc_frow   <= frow;
            fe_base_o <= wptr - 10'd639;              // (index of the hop's last sample + 1) - 640
            hc        <= h_ge1 ? HC_FE : HC_SEQ;
            frow      <= (frow == 6'd48) ? 6'd0 : frow + 6'd1;
            h_ge1     <= 1'b1;
            if (frow == 6'd48) h_ge49 <= 1'b1;        // hop 48 done: from hop 49 on
            nhop_o    <= nhop_o + 32'd1;
          end
          HC_FE:   if (fe_ok_i) hc <= HC_FEW;
          HC_FEW:  if (fe_done_i) hc <= HC_SEQ;
          HC_SEQ:  if (seq_ok_i) hc <= HC_SEQW;
          default: if (seq_done_i) begin              // HC_SEQW
            if (hc_full && net_i) begin
              class_o <= seq_class_i;
              ndec_o  <= ndec_o + 16'd1;
              pend_o  <= 1'b1;
            end
            hc <= HC_IDLE;
          end
        endcase
        if (hop_end && (hc != HC_IDLE)) ovr_o <= 1'b1;
      end
    end
  end

endmodule
