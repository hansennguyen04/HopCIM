// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_seq: layer sequencer that runs the MLPerf Tiny int8 DS-CNN on cim_engine without the CPU
// (R6b). It gathers each pass's input vector from its own buffers (zero point and SAME padding
// included), runs all 7,501 passes, requantises every result with TFLite's integer arithmetic,
// writes it back and reports the argmax of the logits. See docs/SEQUENCER.md.
//
// Layers L, tiles relative to tbase (model/cim_net_map.py):
//   L0     conv1 10x4 stride 2, FEAT 49x10 -> 25x5x64   tiles +0 .. +3            4 passes/position
//   L1+2l  depthwise 3x3, l = 0..3                      tiles +4+14l .. +13+14l  10 passes/position
//   L2+2l  pointwise 64 -> 64                           tiles +14+14l .. +17+14l  4 passes/position
//   pool   average over the 25x5 map (channel sums accumulated during L8)
//   L9     fully connected 64 -> 12                     tile +60                  1 pass
// Every pass has xbits 9 and no accumulation; its input is x = v - zin, and padding taps are 0.
// Requantisation of channel c of layer L uses parameter entry 64 L + c (bias, qm, e = -shift):
//   y = clamp(rdbp(srdhm(acc + bias, qm), e) + zout, -128, 127)
//   srdhm(x, qm) = (x qm + 2^30) >> 31          TFLite's SaturatingRoundingDoublingHighMul for
//                                              qm in [2^30, 2^31) (its saturation case cannot occur)
//   rdbp(r, e)   = (r + 2^(e-1) - (r < 0)) >> e  RoundingDivideByPOT (e = 0: r)
// Only right shifts (shift <= 0) are supported, which is all this model uses.
//
// Buffers: FEAT (features written by the CPU; a ring of 49 rows, frame 0 at row frow_i), ACT (two
// 25x5x64 int8 maps, ping-pong between layers; word {buffer, position, c / 7}, lane c % 7), POOL
// (64 channel sums), the parameter memory, and the logits.
// Pipeline: gather (one chunk of up to 8 X rows per cycle into the idle engine bank) -> passes
// (one every max(9, outputs) cycles) -> requantisation (one result per cycle) -> write-back.
// Layers are separated by a barrier: a layer starts when every result of the previous one is
// written.
//
// Window path (R5b, wp_i): before the network, the sequencer computes the features of the current
// 1-s window from the front-end's per-frame lraw, as model/stream_int.py: it keeps the lraw of the
// last 49 frames and the maxima of the last 50 hops, and recomputes all 49 feature rows only when
// log2fix16(window maximum) changed, otherwise only the new frame's row. Each row is
// mfcc_int.finish: normalisation into X, the two DCT passes (tiles D1, D2) and requantisation.
//
// Options of the baseline S (docs/BASELINE_S.md, found by docs/STREAMFUSE_STUDY.md); all off is R6b:
//   RQ_SG   sign-gated requantisation: in L0..L8 with output zero point -128, acc + bias <= 0 gives
//           exactly -128 (srdhm and rdbp of x <= 0 are <= 0), so the multiplier and the shift hold
//           their registers and the result is -128
//   RQ_2L   a second requantisation lane: two results per cycle, lane 0 takes the even channel of
//           each pair and lane 1 the odd one, so the parameter memory is split into an even and an
//           odd bank (no extra capacity); passes are paced by max(9, ceil(outputs / 2))
//   NO_L8A  L8 is not written to ACT (the pool sums are its only consumer), except in a run that
//           stops after L8, so that it can be read back
//
// Option B1 (FUSE1; docs/STREAMFUSE_B1.md, StreamFuse B in block 1 only; built with S):
//   L1 and L2 run as one segment (lay = 1). Vectors in the order DW(p, g = 0..9), PW(p - 1), for
//   p = 0..125. Depthwise taps are read from a line buffer of 128 words (slot = raster word
//   10 pin + g mod 128), filled from map 0 one word per cycle in the order of first use, over the
//   ACT read port that the segment's gather does not need. Depthwise results go to a fusion
//   vector (2 x 8 words of 8 bytes, half p mod 2, word c / 8, lane c mod 8) instead of ACT; the
//   pointwise gather reads it in 8 chunks of 8 rows. L2 is written to map 1, so L3..L8 read and
//   write the maps the other way round. Interlocks: a depthwise tap that is a word's first use
//   waits until that word has landed in the line buffer (fill count against the gather's count of
//   first uses); a pointwise chunk waits until its 8 channels of the position are written; a fill
//   waits until the word it replaces (raster word - 128) has had its last read, by comparing the
//   gather's position with that word's last reader. The fusion vector needs no overwrite check:
//   PW(p - 2) is gathered before DW(p) is.

module cim_seq #(
    parameter int unsigned TW = 7,                   // engine tile index width
    parameter int unsigned XB = 24,                  // engine input width
    parameter bit          RQ_SG  = 1'b0,
    parameter bit          RQ_2L  = 1'b0,
    parameter bit          NO_L8A = 1'b0,
    parameter bit          FUSE1  = 1'b0             // B1: L1 -> L2 fused (see above)
) (
    input  logic               clk_i,
    input  logic               rst_ni,
    // run control
    input  logic               start_i,              // run the network (ignored while busy_o)
    input  logic [TW-1:0]      tbase_i,              // first conv1 tile
    input  logic [3:0]         stop_i,               // 0: whole network; s: stop after layer s - 1
    input  logic [5:0]         frow_i,               // FEAT row of frame 0 (0..48)
    input  logic [7:0]         zin0_i,               // zero points (int8): conv1 input,
    input  logic [7:0]         zact_i,               //   every other activation,
    input  logic [7:0]         zfc_i,                //   logits
    output logic               busy_o,
    output logic               done_o,               // one-cycle pulse at the end of a run
    output logic               err_o,                // a result was lost (sticky until the next run)
    output logic [3:0]         layer_o,              // current (or last) layer
    output logic [3:0]         class_o,              // argmax of the logits (first maximum)
    // window path (R5b); with start_i
    input  logic               wp_i,                 // run the window path before the network
    input  logic               wp_only_i,            //   and stop after it
    input  logic               wp_frame_i,           //   a new frame: copy its lraw from the front-end
    input  logic               wp_reset_i,           //   start of a stream: forget the block maxima
    output logic               wp_all_o,             // the last window path recomputed all 49 rows
    output logic [19:0]        lmx_o,                // log2fix16(window maximum), Q16
    output logic [15:0]        wmax_o,               // window maximum
    input  logic [15:0]        bmax_i,               // maximum sample of the current hop (endpoint)
    output logic               bmax_take_o,          //   pushed into the block ring this cycle
    output logic               lraw_rd_o,            // front-end lraw read (copy)
    output logic [5:0]         lraw_addr_o,
    input  logic [31:0]        lraw_i,
    // CPU side (the endpoint grants these only while !busy_o)
    input  logic               feat_we_i,            // FEAT word {row, w}: bytes 4w .. 4w+3 of a row
    input  logic [7:0]         feat_addr_i,
    input  logic [31:0]        feat_data_i,
    input  logic               prm_we_i,             // parameter entry 64 L + c, field 0 bias,
    input  logic [9:0]         prm_addr_i,           //   1 qm, 2 e
    input  logic [1:0]         prm_field_i,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0]        prm_data_i,           // bias [23:0], qm [30:0], e [3:0]
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [31:0]        feat_rdata_o,         // FEAT word at feat_addr_i, the cycle after (CPU)
    input  logic               act_re_i,             // ACT read of buffer, position, channel
    input  logic               act_buf_i,
    input  logic [6:0]         act_pos_i,
    input  logic [5:0]         act_ch_i,
    output logic [7:0]         act_rdata_o,          //   valid the cycle after act_re_i
    input  logic [3:0]         logit_addr_i,
    output logic [7:0]         logit_o,
    // engine (driven while busy_o)
    output logic               xw_we_o,
    output logic               xw_bank_o,
    output logic [5:0]         xw_row_o,
    output logic [7:0]         xw_mask_o,
    output logic [XB-1:0]      xw_data_o [8],
    output logic               go_o,                 // start a pass (only asserted with eng_ready_i)
    output logic [TW-1:0]      tile_o,
    output logic               xbank_o,
    output logic [4:0]         xbits_o,
    input  logic               eng_ready_i,
    input  logic               eng_done_i,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic signed [47:0] res_i [16]            // |acc| < 2^21 (network), < 2^26 (DCT)
    /* verilator lint_on UNUSEDSIGNAL */
);

  localparam int unsigned AW = 24;                   // latched accumulator width

  typedef enum logic [1:0] {T_CONV1, T_DW, T_PW, T_FC} ltype_e;

  typedef struct packed {                            // one pass, from issue to its results
    logic [6:0] p;                                   // output position 5 oy + ox
    logic [5:0] c0;                                  // first output channel
    logic [3:0] cg0;                                 // c0 / 7
    logic [2:0] cl0;                                 // c0 % 7
    logic [4:0] n;                                   // results
    logic       pw;                                  // FUSE1: a pointwise pass of the fused segment
  } tag_t;

  typedef struct packed {                            // destination of one result
    logic [6:0] p;
    logic [5:0] c;
    logic [3:0] cg;
    logic [2:0] cl;
    logic       pw;                                  // FUSE1: of the fused segment's pointwise layer
  } dst_t;

  function automatic logic signed [8:0] s9(input logic [7:0] v);
    return {v[7], v};
  endfunction

  // TFLite int8 average pool of 125 values: (s + 62) / 125 for s > 0, else (s - 62) / 125, both
  // truncated toward zero; (a * 8389) >> 20 == a / 125 for every a <= 16062 (|s| <= 16000)
  function automatic logic signed [8:0] pooled(input logic signed [15:0] s);
    logic [14:0] a;
    logic [8:0]  q;
    a = s[15] ? 15'(-s) : 15'(s);
    q = 9'((28'(a + 15'd62) * 28'd8389) >> 20);
    return s[15] ? -q : q;
  endfunction

  // rdbp(r, e) + zout, clamped to int8
  function automatic logic [7:0] rq_out(input logic signed [24:0] r, input logic [3:0] e,
                                        input logic [7:0] zo);
    logic signed [26:0] t;
    t = 27'(r);
    if (e != 4'd0) t = t + (27'sd1 <<< (e - 4'd1)) - (r[24] ? 27'sd1 : 27'sd0);
    t = (t >>> e) + 27'(signed'(zo));
    if (t > 27'sd127) return 8'h7F;
    if (t < -27'sd128) return 8'h80;
    return t[7:0];
  endfunction

  // ================================================================== run and layer control
  logic          run, done_q, err_q;
  logic [3:0]    lay, stop_q;
  logic [5:0]    frow_q;
  logic [TW-1:0] tbase_q;
  logic [7:0]    zin0_q, zact_q, zfc_q;
  ltype_e        lt;
  logic [1:0]    blk;                                // block l of L1..L8
  logic [TW-1:0] tl_base, t14;
  logic          rbuf, wbuf;                         // ACT buffer read / written by the layer
  logic [1:0]    npass_m1;                           // passes per vector - 1
  logic [5:0]    nchunk_m1;                          // chunks per vector - 1
  logic [5:0]    rstep;                              // X rows per chunk step
  logic          lay_start, lay_end, fin, front_done, back_idle;
  logic          wp_on, net_on, wp_fin;              // window-path phase, network phase, end of the first
  logic          wp_only_q, wp_frame_q;
  logic          fz;                                 // FUSE1: the fused segment (L1 and L2, lay = 1)
  logic          swp;                                // FUSE1: L3..L8 use the maps the other way round

  always_comb begin
    if (lay == 4'd0)      lt = T_CONV1;
    else if (lay == 4'd9) lt = T_FC;
    else if (lay[0])      lt = T_DW;
    else                  lt = T_PW;
  end
  assign fz   = FUSE1 && (lay == 4'd1);
  assign swp  = FUSE1 && (lay >= 4'd3);
  assign blk  = 2'((lay - 4'd1) >> 1);
  assign t14  = TW'({blk, 4'd0}) - TW'({blk, 1'b0});
  assign rbuf = swp ? lay[0] : !lay[0];
  assign wbuf = swp ? !lay[0] : lay[0];              // the fused segment's L2 goes to map 1 (write-back)
  always_comb begin
    unique case (lt)
      T_CONV1: begin tl_base = tbase_q;               npass_m1 = 2'd3; nchunk_m1 = 6'd9;  rstep = 6'd4; end
      T_DW:    begin tl_base = tbase_q + TW'(4) + t14;  npass_m1 = 2'd0; nchunk_m1 = 6'd8;  rstep = 6'd7; end
      T_PW:    begin tl_base = tbase_q + TW'(14) + t14; npass_m1 = 2'd3; nchunk_m1 = 6'd9;  rstep = 6'd7; end
      default: begin tl_base = tbase_q + TW'(60);     npass_m1 = 2'd0; nchunk_m1 = 6'd63; rstep = 6'd1; end
    endcase
  end

  // FUSE1: a stop after L1 or after L2 both end with the fused segment
  assign fin       = ((stop_q != 4'd0) && ((lay == stop_q - 4'd1) || (fz && (stop_q == 4'd3)))) || (lay == 4'd9);
  assign lay_end   = net_on && front_done && back_idle;
  assign lay_start = (start_i && !run && !wp_i) || (lay_end && !fin) || (wp_fin && !wp_only_q);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      run     <= 1'b0;
      done_q  <= 1'b0;
      lay     <= 4'd0;
      stop_q  <= 4'd0;
      frow_q  <= 6'd0;
      tbase_q <= '0;
      zin0_q  <= 8'd0;
      zact_q  <= 8'd0;
      zfc_q   <= 8'd0;
      wp_on      <= 1'b0;
      net_on     <= 1'b0;
      wp_only_q  <= 1'b0;
      wp_frame_q <= 1'b0;
    end else begin
      done_q <= 1'b0;
      if (start_i && !run) begin
        run     <= 1'b1;
        lay     <= 4'd0;
        wp_on      <= wp_i;
        net_on     <= !wp_i;
        wp_only_q  <= wp_only_i;
        wp_frame_q <= wp_frame_i;
        stop_q  <= stop_i;
        frow_q  <= frow_i;
        tbase_q <= tbase_i;
        zin0_q  <= zin0_i;
        zact_q  <= zact_i;
        zfc_q   <= zfc_i;
      end else if (wp_fin) begin
        wp_on <= 1'b0;
        if (wp_only_q) begin
          run    <= 1'b0;
          done_q <= 1'b1;
        end else net_on <= 1'b1;
      end else if (lay_end) begin
        if (fin) begin
          run    <= 1'b0;
          net_on <= 1'b0;
          done_q <= 1'b1;
        end else lay <= fz ? 4'd3 : lay + 4'd1;
      end
    end
  end

  // ================================================================== window path (R5b)
  // W_COPY  new frame's lraw (front-end buffer) -> lraw ring row w_new = frow - 1 (mod 49)
  // W_PUSH  the hop's maximum sample (endpoint) -> block ring slot hptr (50 slots)
  // W_MAX   window maximum over the valid slots; W_LOG lmx16 = log2fix16(max(wmax, 1)) (cim_log2fix16)
  // rows    all 49 if lmx16 changed or after a stream reset, else only the new row (if any):
  //   W_LMQ  lmq[n] = lraw == ZERO ? FLOOR : max(((lraw - lmx16) LN2_Q16 + 2^23) >> 24, FLOOR),
  //          X bank 0 row n (n = 0..39), one per cycle (stages A read, B subtract, C multiply)
  //   W_PASS DCT passes on tiles D1, D2 (xbits 14); W_WAIT both results
  //   W_RQ   acc = 256 y1[k] + lo_k; q = clamp(trunc((acc RQ_M + IN_ZP 2^RQ_F) / 2^RQ_F)), k = 0..9
  //   W_FEAT the row's 3 FEAT words
`include "wp_tables.svh"

  typedef enum logic [3:0] {W_IDLE, W_COPY, W_PUSH, W_MAX, W_LOG, W_LMQ, W_PASS, W_WAIT, W_RQ,
                            W_FEAT, W_FIN} wstate_e;
  wstate_e      ws;
  logic [6:0]   w_n;                                 // counter within a state
  logic [5:0]   w_new, w_row, w_last;                // newest frame's row, row computed, last row
  logic [1:0]   w_d;                                 // DCT results received
  logic [5:0]   hptr;                                // next block slot
  logic [49:0]  bm_v;                                // slot holds a block maximum
  logic [15:0]  bm_m [50];
  logic [15:0]  w_max;                               // window maximum (signed)
  logic [19:0]  lmx_q;                               // log2fix16(window maximum)
  logic         lmx_ok, wall_q;
  logic [14:0]  lg_v;
  logic         lg_done;
  logic [19:0]  lmx_new;
  logic         lmx_chg;

  assign w_new = (frow_q == 6'd0) ? 6'd48 : frow_q - 6'd1;

  // log2fix16 of max(wmax, 1), started on entry to W_LOG
  assign lg_v = ($signed(w_max) <= 16'sd0) ? 15'd1 : w_max[14:0];
  cim_log2fix16 i_log2 (
      .clk_i,
      .rst_ni,
      .start_i((ws == W_LOG) && (w_n == 7'd0)),
      .v_i    (lg_v),
      .done_o (lg_done),
      .y_o    (lmx_new)
  );
  assign lmx_chg = !lmx_ok || (lmx_new != lmx_q);

  // lraw ring: row r, mel n at 40 r + n (49 x 40 words)
  function automatic logic [10:0] lr_addr(input logic [5:0] row, input logic [5:0] n);
    return {row, 5'd0} + {2'd0, row, 3'd0} + {5'd0, n};
  endfunction
  (* ram_style = "block" *) logic [31:0] lr_m [2048];
  logic        cp_v;                                 // copy: front-end word read in the previous cycle
  logic [5:0]  cp_n;
  logic [31:0] cp_d;
  logic [10:0] lr_ra_q;
  logic [31:0] lr_rd;
  always_ff @(posedge clk_i) begin
    if (cp_v) lr_m[lr_addr(w_new, cp_n)] <= cp_d;
    lr_ra_q <= lr_addr(w_row, w_n[5:0]);
    cp_n    <= w_n[5:0];
    cp_d    <= lraw_i;
  end
  assign lr_rd       = lr_m[lr_ra_q];
  assign lraw_rd_o   = (ws == W_COPY);
  assign lraw_addr_o = w_n[5:0];
  assign bmax_take_o = (ws == W_PUSH);

  // W_LMQ pipeline: A = ring word (address issued in W_LMQ), B = lraw - lmx16, C = (B) LN2_Q16,
  // D = rounded, floored lmq driving the X write
  logic        la_v, lb_v, lc_v, ld_v, lb_z, lc_z;
  logic [5:0]  la_n, lb_n, lc_n, ld_n;
  logic signed [32:0] lb_d;
  logic signed [49:0] lc_p;
  logic signed [25:0] lc_t;
  logic signed [XB-1:0] ld_x;                        // lmq (fits 14 bits, WP_B_DCT)
  assign lc_t = 26'((lc_p + 50'sd8388608) >>> 24);

  // DCT results and requantisation
  logic signed [31:0] y1 [16];
  logic signed [31:0] y2 [4];
  logic        ra_v, rb_v;
  logic [3:0]  ra_k, rb_k;
  logic signed [35:0] ra_acc;
  logic signed [56:0] rb_v57;
  logic signed [12:0] rq_t;
  logic [7:0]  fb [12];                              // the row's features (fb[10], fb[11] = 0)
  logic signed [31:0] rq_lo;
  logic        wp_go, wp_fw_we;
  logic [TW-1:0] wp_tile;
  logic [7:0]  wp_fw_addr;
  logic [31:0] wp_fw_data;

  always_comb begin
    if (w_n[3:0] < 4'd6) rq_lo = y1[w_n[3:0] + 4'd10];
    else                 rq_lo = y2[w_n[1:0] + 2'd2];   // k = 6..9 -> 0..3
  end
  assign rq_t       = 13'((rb_v57 + (rb_v57[56] ? 57'sd17592186044415 : 57'sd0)) >>> WP_RQ_F);
  assign wp_go      = (ws == W_PASS) && eng_ready_i;
  assign wp_tile    = (w_n == 7'd0) ? TW'(WP_T_D1) : TW'(WP_T_D2);
  assign wp_fw_we   = (ws == W_FEAT);
  assign wp_fw_addr = {w_row, w_n[1:0]};
  always_comb begin
    unique case (w_n[1:0])
      2'd0:    wp_fw_data = {fb[3], fb[2], fb[1], fb[0]};
      2'd1:    wp_fw_data = {fb[7], fb[6], fb[5], fb[4]};
      default: wp_fw_data = {fb[11], fb[10], fb[9], fb[8]};
    endcase
  end
  assign wp_fin = (ws == W_FIN);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ws     <= W_IDLE;
      w_n    <= 7'd0;
      w_row  <= 6'd0;
      w_last <= 6'd0;
      w_d    <= 2'd0;
      hptr   <= 6'd0;
      bm_v   <= '0;
      w_max  <= 16'h8000;
      lmx_q  <= 20'd0;
      lmx_ok <= 1'b0;
      wall_q <= 1'b0;
      cp_v   <= 1'b0;
      la_v   <= 1'b0;
      lb_v   <= 1'b0;
      lc_v   <= 1'b0;
      ld_v   <= 1'b0;
      ra_v   <= 1'b0;
      rb_v   <= 1'b0;
    end else begin
      cp_v <= (ws == W_COPY) && (w_n < 7'd40);
      la_v <= (ws == W_LMQ) && (w_n < 7'd40);
      lb_v <= la_v;
      lc_v <= lb_v;
      ld_v <= lc_v;
      ra_v <= (ws == W_RQ) && (w_n < 7'd10);
      rb_v <= ra_v;
      if (wp_on && eng_done_i) w_d <= w_d + 2'd1;
      unique case (ws)
        W_IDLE: if (start_i && !run && wp_i) begin
          if (wp_reset_i) begin
            bm_v   <= '0;
            hptr   <= 6'd0;
            lmx_ok <= 1'b0;
          end
          w_n <= 7'd0;
          ws  <= wp_frame_i ? W_COPY : W_PUSH;
        end
        W_COPY: begin                                // 40 reads, the last write one cycle later
          w_n <= w_n + 7'd1;
          if (w_n == 7'd40) ws <= W_PUSH;
        end
        W_PUSH: begin
          bm_v[hptr] <= 1'b1;
          hptr  <= (hptr == 6'd49) ? 6'd0 : hptr + 6'd1;
          w_max <= 16'h8000;
          w_n   <= 7'd0;
          ws    <= W_MAX;
        end
        W_MAX: begin
          if (bm_v[w_n[5:0]] && ($signed(bm_m[w_n[5:0]]) > $signed(w_max))) w_max <= bm_m[w_n[5:0]];
          w_n <= w_n + 7'd1;
          if (w_n == 7'd49) begin
            w_n <= 7'd0;
            ws  <= W_LOG;
          end
        end
        W_LOG: begin                                 // start the log2 unit, wait for its result
          if (w_n == 7'd0) w_n <= 7'd1;
          if (lg_done) begin
            w_n <= 7'd0;
            ws  <= W_FIN;                            // overridden below when rows are computed
            if (lmx_chg) begin
              w_row <= 6'd0;
              w_last <= 6'd48;
              ws    <= W_LMQ;
            end else if (wp_frame_q) begin
              w_row <= w_new;
              w_last <= w_new;
              ws    <= W_LMQ;
            end
          end
        end
        W_LMQ: begin                                 // 40 addresses, then drain stages A..D
          w_n <= w_n + 7'd1;
          if (w_n == 7'd43) begin
            w_n <= 7'd0;
            w_d <= 2'd0;
            ws  <= W_PASS;
          end
        end
        W_PASS: if (eng_ready_i) begin
          w_n <= w_n + 7'd1;
          if (w_n == 7'd1) ws <= W_WAIT;
        end
        W_WAIT: if (w_d == 2'd2) begin
          w_n <= 7'd0;
          ws  <= W_RQ;
        end
        W_RQ: begin                                  // 10 issues, then drain the 2 stages
          w_n <= w_n + 7'd1;
          if (w_n == 7'd11) begin
            w_n <= 7'd0;
            ws  <= W_FEAT;
          end
        end
        W_FEAT: begin
          w_n <= w_n + 7'd1;
          if (w_n == 7'd2) begin
            w_n <= 7'd0;
            if (w_row == w_last) ws <= W_FIN;
            else begin
              w_row <= w_row + 6'd1;
              ws    <= W_LMQ;
            end
          end
        end
        default: ws <= W_IDLE;                       // W_FIN: one cycle (wp_fin)
      endcase
      if ((ws == W_LOG) && lg_done) begin
        lmx_q  <= lmx_new;
        lmx_ok <= 1'b1;
        wall_q <= lmx_chg;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (ws == W_PUSH) bm_m[hptr] <= bmax_i;
    // stage A -> B
    lb_n <= la_n;
    lb_z <= (lr_rd == 32'h8000_0000);
    lb_d <= {lr_rd[31], lr_rd} - {13'd0, lmx_q};
    // B -> C
    lc_n <= lb_n;
    lc_z <= lb_z;
    lc_p <= lb_d * $signed({1'b0, 16'(WP_LN2_Q16)});
    // C -> D
    ld_n <= lc_n;
    ld_x <= XB'((lc_z || (lc_t < 26'(WP_FLOOR_Q8))) ? 26'(WP_FLOOR_Q8) : lc_t);
    la_n <= w_n[5:0];
    // DCT results
    if (wp_on && eng_done_i && (w_d == 2'd0)) for (int n = 0; n < 16; n++) y1[n] <= res_i[n][31:0];
    if (wp_on && eng_done_i && (w_d == 2'd1)) for (int n = 0; n < 4; n++) y2[n] <= res_i[n][31:0];
    // requantisation: R0 combine, R1 multiply + zero point, R2 truncate and clamp
    ra_k   <= w_n[3:0];
    ra_acc <= {y1[w_n[3:0]][27:0], 8'd0} + 36'(rq_lo);
    rb_k   <= ra_k;
    // every operand signed: one unsigned term would make the sum unsigned and zero-extend ra_acc
    rb_v57 <= ra_acc * $signed({1'b0, 20'(WP_RQ_M)}) + $signed(57'(WP_IN_ZP) << WP_RQ_F);
    if (rb_v) fb[rb_k] <= ($signed(rq_t) > 13'sd127) ? 8'h7F : ($signed(rq_t) < -13'sd128) ? 8'h80 : rq_t[7:0];
    if ((ws == W_IDLE) && start_i) begin
      fb[10] <= 8'd0;
      fb[11] <= 8'd0;
    end
  end

  // ================================================================== gather
  // A vector is the input of one pass group: conv1 and pointwise one per position (used by 4
  // passes), depthwise one per (position, channel group), FC one. A chunk writes up to 8
  // consecutive X rows: conv1 the 4 taps fx of filter row fy (rows 4 fy ..), depthwise the 7
  // channels of tap k (rows 7 k ..), pointwise 7 input channels (rows 7 g ..), FC one pooled value.
  // Stage R reads FEAT / ACT / POOL, G1 forms x = v - zin (0 for padding), G2 drives the wide port.
  logic          g_on, g_b, g_req, g_lastc, g_lastv;
  logic [6:0]    g_p;                                // position 5 oy + ox
  logic [4:0]    g_oy;
  logic [2:0]    g_ox;
  logic [3:0]    g_g;                                // depthwise channel group
  logic [5:0]    g_k, g_row;                         // chunk, its first X row
  logic [1:0]    g_fy, g_fx;                         // depthwise tap of chunk g_k
  logic [3:0]    g_n;                                // rows in the chunk
  logic [1:0]    bbusy, bfull;                       // per X bank: holds a vector / it is complete
  logic [2:0]    bcnt [2];                           // per X bank: cycles until its last pass has read it
  logic [6:0]    vm_p [2];                           // per X bank: the vector's position,
  logic [3:0]    vm_g [2];                           //   channel group,
  logic [1:0]    vm_last;                            //   last vector of the layer
  logic [1:0]    vm_pw;                              //   FUSE1: a pointwise vector of the fused segment
  logic          g_pw;                               // FUSE1: the vector is PW(g_p - 1) of the fused segment
  ltype_e        gk;                                 // kind of the vector being gathered
  logic [5:0]    g_ncm1, g_rst;                      // its chunks - 1, X rows per chunk step
  logic          g_src;                              // FUSE1: the chunk's source word is there (interlock)

  assign gk      = fz ? (g_pw ? T_PW : T_DW) : lt;
  assign g_ncm1  = fz ? (g_pw ? 6'd7 : 6'd8) : nchunk_m1;
  assign g_rst   = fz ? (g_pw ? 6'd8 : 6'd7) : rstep;
  assign g_lastc = (g_k == g_ncm1);
  assign g_lastv = fz ? (g_pw && (g_p == 7'd125))
                      : ((lt == T_FC) || ((g_p == 7'd124) && ((lt != T_DW) || (g_g == 4'd9))));
  // a new vector needs its bank free: no vector in it, and its last pass past its X reads
  assign g_req   = g_on && ((g_k != 6'd0) || (!bbusy[g_b] && (bcnt[g_b] == 3'd0))) && g_src;
  always_comb begin
    unique case (gk)
      T_CONV1: g_n = 4'd4;
      T_DW:    g_n = (g_g == 4'd9) ? 4'd1 : 4'd7;
      T_PW:    g_n = fz ? 4'd8 : ((g_k == 6'd9) ? 4'd1 : 4'd7);
      default: g_n = 4'd1;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      g_on  <= 1'b0;
      g_b   <= 1'b0;
      g_p   <= 7'd0;
      g_oy  <= 5'd0;
      g_ox  <= 3'd0;
      g_g   <= 4'd0;
      g_k   <= 6'd0;
      g_row <= 6'd0;
      g_fy  <= 2'd0;
      g_fx  <= 2'd0;
      g_pw  <= 1'b0;
    end else if (lay_start) begin
      g_on  <= 1'b1;
      g_b   <= 1'b0;
      g_p   <= 7'd0;
      g_oy  <= 5'd0;
      g_ox  <= 3'd0;
      g_g   <= 4'd0;
      g_k   <= 6'd0;
      g_row <= 6'd0;
      g_fy  <= 2'd0;
      g_fx  <= 2'd0;
      g_pw  <= 1'b0;
    end else if (g_req) begin
      if (g_lastc) begin
        g_k   <= 6'd0;
        g_row <= 6'd0;
        g_fy  <= 2'd0;
        g_fx  <= 2'd0;
        g_b   <= !g_b;
        if (g_lastv) g_on <= 1'b0;
        if ((gk == T_DW) && (g_g != 4'd9)) g_g <= g_g + 4'd1;
        else if (fz && !g_pw && (g_p != 7'd0)) begin  // fused: DW(p, 9) -> PW(p - 1)
          g_g  <= 4'd0;
          g_pw <= 1'b1;
        end else begin                                // fused: PW(p - 1) -> DW(p + 1, 0), or PW(124) last
          g_g  <= 4'd0;
          g_pw <= fz && g_pw && (g_p == 7'd124);
          g_p  <= g_p + 7'd1;
          if (g_ox == 3'd4) begin
            g_ox <= 3'd0;
            g_oy <= g_oy + 5'd1;
          end else g_ox <= g_ox + 3'd1;
        end
      end else begin
        g_k   <= g_k + 6'd1;
        g_row <= g_row + g_rst;
        if (g_fx == 2'd2) begin
          g_fx <= 2'd0;
          g_fy <= g_fy + 2'd1;
        end else g_fx <= g_fx + 2'd1;
      end
    end
  end

  // stage R: source addresses, in unsigned bit-level logic. (A first version used signed int
  // arithmetic, e.g. ix0 + ((b - ix0) & 3); Vivado 2024.1 synthesised lane 0's word of it wrongly
  // for odd ox, while Verilator and xsim simulate it as intended. See docs/SEQUENCER.md.)
  //   conv1: input row iy = 2 oy - 4 + fy, columns ix0 .. ix0 + 3 with ix0 = 2 ox - 1. Lane b of
  //   FEAT holds the columns = b mod 4: with wq = ox / 2, ox even -> lanes 0..2 read word wq,
  //   lane 3 word wq - 1 (column ix0; invalid for ox = 0); ox odd -> lane 0 reads word wq + 1
  //   (column ix0 + 3), lanes 1..3 word wq. Tap fx comes from lane (ix0 + fx) mod 4.
  //   depthwise: input position (oy - 1 + fy, ox - 1 + fx).
  logic [7:0]  fa [4];                               // FEAT lane b address {ring row, word}
  logic [3:0]  c_ixv;                                // conv1: column of tap fx in bounds
  logic [1:0]  c_ix0;                                // conv1: ix0 mod 4
  logic        g_tv;                                 // conv1 row / depthwise tap in bounds
  logic [11:0] ga;                                   // ACT word {buffer, position, group}
  logic [5:0]  c_iy4;                                // conv1: iy + 4 = 2 oy + fy (0..57)
  logic [6:0]  c_s;                                  // conv1: frow + iy (valid rows: 0..96)
  logic [5:0]  c_row;                                // conv1: FEAT ring row
  logic [1:0]  c_wq, c_w0, c_w3;
  logic [4:0]  d_y1;                                 // depthwise: input row + 1 (0..26)
  logic [2:0]  d_x1;                                 // depthwise: input column + 1 (0..6)
  logic [6:0]  d_pin;                                // depthwise: input position 5 (y1 - 1) + (x1 - 1)

  assign c_iy4 = {g_oy, 1'b0} + g_k;
  assign c_s   = {1'b0, frow_q} + {1'b0, c_iy4} - 7'd4;
  assign c_row = (c_s >= 7'd49) ? 6'(c_s - 7'd49) : c_s[5:0];
  assign c_wq  = g_ox[2:1];
  assign c_w0  = g_ox[0] ? c_wq + 2'd1 : c_wq;
  assign c_w3  = g_ox[0] ? c_wq : c_wq - 2'd1;
  assign fa[0] = {c_row, c_w0};
  assign fa[1] = {c_row, c_wq};
  assign fa[2] = {c_row, c_wq};
  assign fa[3] = {c_row, c_w3};
  assign c_ix0 = {!g_ox[0], 1'b1};
  assign c_ixv = {g_ox != 3'd4, 1'b1, 1'b1, g_ox != 3'd0};
  assign d_y1  = g_oy + {3'd0, g_fy};
  assign d_x1  = g_ox + {1'b0, g_fx};
  assign d_pin = 7'({1'b0, d_y1, 2'b00} + {3'd0, d_y1} + {5'd0, d_x1} - 8'd6);
  logic [11:0] f_ra;                                 // FUSE1: the line-buffer fill's ACT word (below)
  /* verilator lint_off UNUSEDSIGNAL */            // FUSE1 only
  logic        f_on, f1_v, f_start;                  // FUSE1: fill running, word landing, segment start
  /* verilator lint_on UNUSEDSIGNAL */
  always_comb begin
    if (gk == T_CONV1)   g_tv = (c_iy4 >= 6'd4) && (c_iy4 <= 6'd52);
    else if (gk == T_DW) g_tv = (d_y1 >= 5'd1) && (d_y1 <= 5'd25) && (d_x1 >= 3'd1) && (d_x1 <= 3'd5);
    else                 g_tv = 1'b1;
    if (fz)              ga = f_ra;                  // the fused gather reads no ACT: the fill has the port
    else if (lt == T_DW) ga = {rbuf, d_pin, g_g};
    else                 ga = {rbuf, g_p, g_k[3:0]};
  end

  // FUSE1 gather sources: line-buffer slot of the tap's word, fusion-vector word of the chunk
  /* verilator lint_off UNUSEDSIGNAL */            // FUSE1 only
  logic [6:0]  g_rw;                                 // raster word 10 pin + g of the tap, mod 128 (its slot)
  logic [6:0]  lb_ra;
  logic [3:0]  fv_ra;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [6:0]  g_pwp;                                // position of the pointwise vector
  logic        g_fu;                                 // the tap is its word's first use
  logic [10:0] gq, fl_cnt;                           // first uses requested by the gather, words landed
  logic [6:0]  fv_wp, fv_wc;                         // fusion vector: position being written, channels of it
  assign g_rw  = 7'({d_pin, 3'b000} + {2'b00, d_pin, 1'b0} + {6'd0, g_g});
  assign lb_ra = g_rw[6:0];
  assign g_pwp = g_p - 7'd1;
  assign fv_ra = {g_pwp[0], g_k[2:0]};
  // first reader of input (iy, ix) is output (max(iy - 1, 0), max(ix - 1, 0))
  assign g_fu  = g_tv && ((g_fy == 2'd2) || ((g_oy == 5'd0) && (g_fy == 2'd1)))
                      && ((g_fx == 2'd2) || ((g_ox == 3'd0) && (g_fx == 2'd1)));
  always_comb begin
    if (!fz)       g_src = 1'b1;
    else if (g_pw) g_src = (fv_wp > g_pwp) || ((fv_wp == g_pwp) && (fv_wc >= {4'(g_k[2:0]) + 4'd1, 3'b000}));
    else           g_src = !g_fu || (fl_cnt > gq);
  end

  // memories read in stage R (data in G1)
  logic [7:0] feat_q [4];
  logic [7:0] act_q [7];
  logic [7:0] lb_q [7];                              // FUSE1: line buffer
  logic [7:0] fv_q [8];                              // FUSE1: fusion vector
  logic signed [15:0] pool_rd;                       // pool sum of channel g_k (FC gather; write-back)

  // FEAT: written by the CPU, or by the window path while it runs; read by the gather, or by the
  // CPU while the sequencer is idle (data the cycle after, feat_rdata_o)
  logic        feat_we;
  logic [7:0]  feat_wa;
  logic [31:0] feat_wd;
  assign feat_we = feat_we_i || wp_fw_we;
  assign feat_wa = wp_fw_we ? wp_fw_addr : feat_addr_i;
  assign feat_wd = wp_fw_we ? wp_fw_data : feat_data_i;
  for (genvar b = 0; b < 4; b++) begin : g_feat
    (* ram_style = "distributed" *) logic [7:0] fm [256];
    always_ff @(posedge clk_i) begin
      if (feat_we) fm[feat_wa] <= feat_wd[8*b +: 8];
      feat_q[b] <= fm[run ? fa[b] : feat_addr_i];
    end
  end
  assign feat_rdata_o = {feat_q[3], feat_q[2], feat_q[1], feat_q[0]};

  // stage G1
  logic        g1_v, g1_b, g1_last, g1_tv, g1_fz;
  ltype_e      g1_kind;
  logic [5:0]  g1_row;
  logic [3:0]  g1_n, g1_ixv;
  logic [1:0]  g1_ix0;
  logic signed [15:0] g1_pool;
  logic signed [8:0]  gx [8];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) g1_v <= 1'b0;
    else         g1_v <= g_req;
  end
  always_ff @(posedge clk_i) begin
    if (g_req) begin
      g1_b    <= g_b;
      g1_last <= g_lastc;
      g1_tv   <= g_tv;
      g1_row  <= g_row;
      g1_n    <= g_n;
      g1_ixv  <= c_ixv;
      g1_ix0  <= c_ix0;
      g1_pool <= pool_rd;
      g1_kind <= gk;
      g1_fz   <= fz;
    end
  end

  always_comb begin
    for (int i = 0; i < 8; i++) gx[i] = 9'sd0;
    unique case (g1_kind)
      T_CONV1:
        for (int f = 0; f < 4; f++)
          if (g1_tv && g1_ixv[f]) gx[f] = s9(feat_q[2'(g1_ix0 + 2'(f))]) - s9(zin0_q);
      T_DW, T_PW:
        if (g1_fz && (g1_kind == T_PW))                // fused pointwise: 8 channels of the fusion vector
          for (int i = 0; i < 8; i++) gx[i] = s9(fv_q[i]) - s9(zact_q);
        else
          for (int i = 0; i < 7; i++)
            if (g1_tv) gx[i] = s9(g1_fz ? lb_q[i] : act_q[i]) - s9(zact_q);
      default: gx[0] = pooled(g1_pool) - s9(zact_q);
    endcase
  end

  // stage G2: rows rotated into engine lanes (row r in lane r % 8)
  logic        g2_v, g2_b, g2_last;
  logic [5:0]  g2_row;
  logic [7:0]  g2_mask;
  logic signed [8:0] g2_x [8];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) g2_v <= 1'b0;
    else         g2_v <= g1_v;
  end
  always_ff @(posedge clk_i) begin
    if (g1_v) begin
      g2_b    <= g1_b;
      g2_last <= g1_last;
      g2_row  <= g1_row;
      for (int l = 0; l < 8; l++) begin
        g2_x[l]    <= gx[3'(l) - g1_row[2:0]];
        g2_mask[l] <= ({1'b0, 3'(l) - g1_row[2:0]} < g1_n);
      end
    end
  end

  // engine wide port: the window path writes one row per cycle, the network gather up to 8
  assign xw_we_o   = wp_on ? ld_v : g2_v;
  assign xw_bank_o = wp_on ? 1'b0 : g2_b;
  assign xw_row_o  = wp_on ? ld_n : g2_row;
  assign xw_mask_o = wp_on ? (8'd1 << ld_n[2:0]) : g2_mask;
  for (genvar l = 0; l < 8; l++) begin : g_xw
    assign xw_data_o[l] = wp_on ? ld_x : XB'(g2_x[l]);
  end

  // ================================================================== issue
  // The passes of the vector in bank i_b start once it is complete (also in the cycle its last
  // chunk is written), each when the engine is ready and the previous pass's results can drain
  // (one result per cycle: passes are max(9, outputs) cycles apart).
  logic          i_on, i_b, i_full, issue, i_lastp;
  logic [1:0]    i_j;                                // pass within the vector
  logic [4:0]    pace;
  logic [2:0]    tq_cnt;
  tag_t          i_tag;

  assign i_full  = bfull[i_b] || (g2_v && g2_last && (g2_b == i_b));
  assign issue   = i_on && i_full && eng_ready_i && (pace == 5'd0) && (tq_cnt != 3'd4);
  assign i_lastp = (i_j == (fz ? (vm_pw[i_b] ? 2'd3 : 2'd0) : npass_m1));
  logic [TW-1:0] n_tile;                             // the network's pass tile
  assign go_o    = wp_on ? wp_go : issue;
  assign tile_o  = wp_on ? wp_tile : n_tile;
  assign xbank_o = wp_on ? 1'b0 : i_b;
  assign xbits_o = wp_on ? 5'(WP_B_DCT) : 5'd9;

  always_comb begin
    if (fz && vm_pw[i_b]) begin                      // fused pointwise L2: tiles of block 0
      n_tile = tbase_q + TW'(14) + TW'(i_j);
      i_tag  = '{p: vm_p[i_b], c0: {i_j, 4'd0}, cg0: {1'b0, i_j, 1'b0}, cl0: {i_j, 1'b0}, n: 5'd16, pw: 1'b1};
    end else begin
      unique case (lt)
        T_DW: begin
          n_tile = tl_base + TW'(vm_g[i_b]);
          i_tag  = '{p: vm_p[i_b], c0: 6'(7 * int'(vm_g[i_b])), cg0: vm_g[i_b], cl0: 3'd0,
                     n: (vm_g[i_b] == 4'd9) ? 5'd1 : 5'd7, pw: 1'b0};
        end
        T_FC: begin
          n_tile = tl_base;
          i_tag  = '{p: 7'd0, c0: 6'd0, cg0: 4'd0, cl0: 3'd0, n: 5'd12, pw: 1'b0};
        end
        default: begin                               // pass j: channels 16 j .. (16 j = 14 j + 2 j)
          n_tile = tl_base + TW'(i_j);
          i_tag  = '{p: vm_p[i_b], c0: {i_j, 4'd0}, cg0: {1'b0, i_j, 1'b0}, cl0: {i_j, 1'b0}, n: 5'd16, pw: 1'b0};
        end
      endcase
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      i_on    <= 1'b0;
      i_b     <= 1'b0;
      i_j     <= 2'd0;
      pace    <= 5'd0;
      bbusy   <= 2'b00;
      bfull   <= 2'b00;
      bcnt[0] <= 3'd0;
      bcnt[1] <= 3'd0;
      vm_last <= 2'b00;
      vm_pw   <= 2'b00;
    end else begin
      if (issue) pace <= RQ_2L ? 5'((6'(i_tag.n) + 6'd1) >> 1) - 5'd1 : i_tag.n - 5'd1;   // drain time
      else if (pace != 5'd0) pace <= pace - 5'd1;
      for (int b = 0; b < 2; b++)
        if (bcnt[b] != 3'd0) bcnt[b] <= bcnt[b] - 3'd1;
      if (lay_start) begin
        i_on  <= 1'b1;
        i_b   <= 1'b0;
        i_j   <= 2'd0;
        bbusy <= 2'b00;
        bfull <= 2'b00;
      end else begin
        if (g_req && (g_k == 6'd0)) begin
          bbusy[g_b]   <= 1'b1;
          vm_last[g_b] <= g_lastv;
          vm_pw[g_b]   <= g_pw;
        end
        if (g2_v && g2_last) bfull[g2_b] <= 1'b1;
        if (issue) begin
          if (i_lastp) begin                         // the bank is read until 9 edges after the start
            i_j        <= 2'd0;
            i_b        <= !i_b;
            bbusy[i_b] <= 1'b0;
            bfull[i_b] <= 1'b0;
            bcnt[i_b]  <= 3'd7;
            if (vm_last[i_b]) i_on <= 1'b0;
          end else i_j <= i_j + 2'd1;
        end
      end
    end
  end
  always_ff @(posedge clk_i) begin
    if (g_req && (g_k == 6'd0)) begin
      vm_p[g_b] <= g_pw ? g_pwp : g_p;
      vm_g[g_b] <= g_g;
    end
  end

  // ================================================================== results
  // Tags of the passes in flight (at most 2), in issue order; a done pops the oldest and latches
  // its results, which drain one per cycle (RQ_2L: two) into the requantisation pipeline.
  tag_t        tq [4];
  logic [1:0]  tq_wp, tq_rp;
  logic        pop;
  tag_t        t_pop;
  logic signed [AW-1:0] lat [16];
  logic [3:0]  d_i;
  logic [4:0]  d_rem;
  logic [6:0]  d_p;
  logic [5:0]  d_c;
  logic [3:0]  d_cg;
  logic [2:0]  d_cl;
  logic        d_two;                                // RQ_2L: two results drain this cycle
  logic        d_sw;                                 // RQ_2L: the first result has an odd channel
  logic        d_pw;                                 // FUSE1: the pass is the fused segment's pointwise
  logic [3:0]  d_lay;                                // layer of its parameters
  dst_t        d_d1, d_d2;                           // destinations of the first and the second result
  /* verilator lint_off UNUSEDSIGNAL */            // used with RQ_2L only
  logic [4:0]  d_ix0, d_ix1;                         // RQ_2L: parameter index (channel / 2), lanes 0, 1
  /* verilator lint_on UNUSEDSIGNAL */

  assign pop   = run && net_on && eng_done_i;
  assign t_pop = tq[tq_rp];
  assign d_two = RQ_2L && (d_rem > 5'd1);
  assign d_sw  = RQ_2L && d_c[0];
  assign d_d1  = '{p: d_p, c: d_c, cg: d_cg, cl: d_cl, pw: d_pw};
  assign d_d2  = '{p: d_p, c: d_c + 6'd1, cg: (d_cl == 3'd6) ? d_cg + 4'd1 : d_cg, cl: (d_cl == 3'd6) ? 3'd0 : d_cl + 3'd1,
                   pw: d_pw};
  assign d_lay = (fz && d_pw) ? 4'd2 : lay;
  assign d_ix0 = 5'((7'(d_c) + 7'd1) >> 1);          // the even channel of the pair (d_c or d_c + 1)
  assign d_ix1 = d_c[5:1];                           // the odd one

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      tq_wp  <= 2'd0;
      tq_rp  <= 2'd0;
      tq_cnt <= 3'd0;
      d_rem  <= 5'd0;
      d_i    <= 4'd0;
      d_p    <= 7'd0;
      d_c    <= 6'd0;
      d_cg   <= 4'd0;
      d_cl   <= 3'd0;
      d_pw   <= 1'b0;
      err_q  <= 1'b0;
    end else begin
      if (issue) tq_wp <= tq_wp + 2'd1;
      if (pop && (tq_cnt != 3'd0)) tq_rp <= tq_rp + 2'd1;
      tq_cnt <= tq_cnt + ((issue) ? 3'd1 : 3'd0) - ((pop && (tq_cnt != 3'd0)) ? 3'd1 : 3'd0);
      if (start_i && !run) err_q <= 1'b0;
      else if (pop && ((tq_cnt == 3'd0) || (d_rem > (RQ_2L ? 5'd2 : 5'd1)))) err_q <= 1'b1;
      if (pop) begin
        d_rem <= t_pop.n;
        d_i   <= 4'd0;
        d_p   <= t_pop.p;
        d_c   <= t_pop.c0;
        d_cg  <= t_pop.cg0;
        d_cl  <= t_pop.cl0;
        d_pw  <= t_pop.pw;
      end else if (d_two) begin
        d_rem <= d_rem - 5'd2;
        d_i   <= d_i + 4'd2;
        d_c   <= d_c + 6'd2;
        if (d_cl >= 3'd5) begin
          d_cl <= d_cl - 3'd5;
          d_cg <= d_cg + 4'd1;
        end else d_cl <= d_cl + 3'd2;
      end else if (d_rem != 5'd0) begin
        d_rem <= d_rem - 5'd1;
        d_i   <= d_i + 4'd1;
        d_c   <= d_c + 6'd1;
        if (d_cl == 3'd6) begin
          d_cl <= 3'd0;
          d_cg <= d_cg + 4'd1;
        end else d_cl <= d_cl + 3'd1;
      end
    end
  end
  always_ff @(posedge clk_i) begin
    if (issue) tq[tq_wp] <= i_tag;
    if (pop) for (int n = 0; n < 16; n++) lat[n] <= AW'(res_i[n]);
  end

  // requantisation: Q0 result and parameter address, Q1 + bias, Q2 * qm, Q3 srdhm, Q4 rdbp + zout.
  // The parameter memories register their read address (a block RAM's own register). qm is read
  // one cycle later, so it feeds the multiplier straight from the RAM: with a data register in
  // between, Vivado absorbs that register and the RAM's output stage into the DSP B pipeline and
  // is left with an asynchronous read that only LUTRAM can implement.
  logic [23:0] bias_q;                               // lane 0
  logic [30:0] qm_q;
  logic [3:0]  e_q;
  /* verilator lint_off UNUSEDSIGNAL */            // lane 1: used with RQ_2L only
  logic [23:0] bias1_q;
  logic [30:0] qm1_q;
  logic [3:0]  e1_q;
  /* verilator lint_on UNUSEDSIGNAL */

  if (!RQ_2L) begin : g_prm1
    (* ram_style = "block" *) logic [23:0] bias_m [1024];
    (* ram_style = "block" *) logic [30:0] qm_m [1024];
    (* ram_style = "block" *) logic [3:0]  e_m [1024];
    logic [9:0]  prm_ra, prm_ra_q, prm_ra_q2;

    assign prm_ra = {d_lay, d_c};
    always_ff @(posedge clk_i) begin
      if (prm_we_i && (prm_field_i == 2'd0)) bias_m[prm_addr_i] <= prm_data_i[23:0];
      if (prm_we_i && (prm_field_i == 2'd1)) qm_m[prm_addr_i] <= prm_data_i[30:0];
      if (prm_we_i && (prm_field_i == 2'd2)) e_m[prm_addr_i] <= prm_data_i[3:0];
      prm_ra_q  <= prm_ra;
      prm_ra_q2 <= prm_ra_q;
    end
    assign bias_q  = bias_m[prm_ra_q];               // in Q1
    assign e_q     = e_m[prm_ra_q];
    assign qm_q    = qm_m[prm_ra_q2];                // in Q2
    assign bias1_q = '0;
    assign e1_q    = '0;
    assign qm1_q   = '0;
  end else begin : g_prm2
    // bank k holds the channels c with c mod 2 = k, entry {L, c / 2}; lane k reads bank k
    for (genvar k = 0; k < 2; k++) begin : g_bank
      (* ram_style = "block" *) logic [23:0] bias_m [512];
      (* ram_style = "block" *) logic [30:0] qm_m [512];
      (* ram_style = "block" *) logic [3:0]  e_m [512];
      logic [8:0] wa, prm_ra, prm_ra_q, prm_ra_q2;
      logic       we;

      assign wa     = {prm_addr_i[9:6], prm_addr_i[5:1]};
      assign we     = prm_we_i && (prm_addr_i[0] == 1'(k));
      assign prm_ra = {d_lay, (k == 0) ? d_ix0 : d_ix1};
      always_ff @(posedge clk_i) begin
        if (we && (prm_field_i == 2'd0)) bias_m[wa] <= prm_data_i[23:0];
        if (we && (prm_field_i == 2'd1)) qm_m[wa] <= prm_data_i[30:0];
        if (we && (prm_field_i == 2'd2)) e_m[wa] <= prm_data_i[3:0];
        prm_ra_q  <= prm_ra;
        prm_ra_q2 <= prm_ra_q;
      end
      if (k == 0) begin : g_l0
        assign bias_q = bias_m[prm_ra_q];
        assign e_q    = e_m[prm_ra_q];
        assign qm_q   = qm_m[prm_ra_q2];
      end else begin : g_l1
        assign bias1_q = bias_m[prm_ra_q];
        assign e1_q    = e_m[prm_ra_q];
        assign qm1_q   = qm_m[prm_ra_q2];
      end
    end
  end

  // sign gate: exact -128 for acc + bias <= 0 when the output zero point is -128 (L0..L8)
  logic sg_on;
  assign sg_on = RQ_SG && (lay != 4'd9) && (zact_q == 8'h80);

  // lane 0 (R6b's only lane; with RQ_2L the even channel of each pair)
  logic                 q0_v, q1_v, q2_v, q3_v, q4_v;
  dst_t                 q0_d, q1_d, q2_d, q3_d, q4_d;
  logic signed [AW-1:0] q0_acc;
  logic signed [24:0]   q1_x, q3_r;
  logic [3:0]           q1_e, q2_e, q3_e;
  logic signed [56:0]   q2_m;
  logic [7:0]           q4_y;
  logic                 q1_g, q2_g, q3_g;            // sign-gated: x <= 0 (q1_g in Q1, from q1_x)

  assign q1_g = sg_on && (q1_x[24] || (q1_x == 25'sd0));
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      q0_v <= 1'b0; q1_v <= 1'b0; q2_v <= 1'b0; q3_v <= 1'b0; q4_v <= 1'b0;
    end else begin
      q0_v <= d_sw ? d_two : (d_rem != 5'd0);
      q1_v <= q0_v; q2_v <= q1_v; q3_v <= q2_v; q4_v <= q3_v;
    end
  end
  always_ff @(posedge clk_i) begin
    q0_acc <= lat[d_sw ? d_i + 4'd1 : d_i];
    q0_d   <= d_sw ? d_d2 : d_d1;
    q1_x   <= 25'(q0_acc) + 25'(signed'(bias_q));
    q1_e   <= e_q;
    q1_d   <= q0_d;
    if (!q1_g) q2_m <= q1_x * $signed({1'b0, qm_q});        // held when gated
    q2_g   <= q1_g;
    q2_e   <= q1_e;
    q2_d   <= q1_d;
    if (!q2_g) q3_r <= 25'((q2_m + 57'sd1073741824) >>> 31);
    q3_g   <= q2_g;
    q3_e   <= q2_e;
    q3_d   <= q2_d;
    q4_y   <= q3_g ? 8'h80 : rq_out(q3_r, q3_e, (lay == 4'd9) ? zfc_q : zact_q);
    q4_d   <= q3_d;
  end

  // lane 1 (RQ_2L): the odd channel of each pair; constant idle without RQ_2L
  logic                 r0_v, r1_v, r2_v, r3_v, r4_v;
  /* verilator lint_off UNUSEDSIGNAL */            // read only inside g_lane1
  dst_t                 r0_d, r1_d, r2_d, r3_d;
  logic signed [AW-1:0] r0_acc;
  logic signed [24:0]   r1_x, r3_r;
  logic [3:0]           r1_e, r2_e, r3_e;
  logic signed [56:0]   r2_m;
  logic                 r1_g, r2_g, r3_g;
  /* verilator lint_on UNUSEDSIGNAL */
  dst_t                 r4_d;
  logic [7:0]           r4_y;

  if (RQ_2L) begin : g_lane1
    assign r1_g = sg_on && (r1_x[24] || (r1_x == 25'sd0));
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        r0_v <= 1'b0; r1_v <= 1'b0; r2_v <= 1'b0; r3_v <= 1'b0; r4_v <= 1'b0;
      end else begin
        r0_v <= d_sw ? (d_rem != 5'd0) : d_two;
        r1_v <= r0_v; r2_v <= r1_v; r3_v <= r2_v; r4_v <= r3_v;
      end
    end
    always_ff @(posedge clk_i) begin
      r0_acc <= lat[d_sw ? d_i : d_i + 4'd1];
      r0_d   <= d_sw ? d_d1 : d_d2;
      r1_x   <= 25'(r0_acc) + 25'(signed'(bias1_q));
      r1_e   <= e1_q;
      r1_d   <= r0_d;
      if (!r1_g) r2_m <= r1_x * $signed({1'b0, qm1_q});
      r2_g   <= r1_g;
      r2_e   <= r1_e;
      r2_d   <= r1_d;
      if (!r2_g) r3_r <= 25'((r2_m + 57'sd1073741824) >>> 31);
      r3_g   <= r2_g;
      r3_e   <= r2_e;
      r3_d   <= r2_d;
      r4_y   <= r3_g ? 8'h80 : rq_out(r3_r, r3_e, (lay == 4'd9) ? zfc_q : zact_q);
      r4_d   <= r3_d;
    end
  end else begin : g_lane1_off
    assign {r0_v, r1_v, r2_v, r3_v, r4_v} = '0;
    assign {r0_d, r1_d, r2_d, r3_d, r4_d} = '0;
    assign {r0_acc, r1_x, r3_r, r1_e, r2_e, r3_e, r2_m, r4_y} = '0;
    assign {r1_g, r2_g, r3_g} = '0;
  end

  assign back_idle  = (tq_cnt == 3'd0) && (d_rem == 5'd0) && !q0_v && !q1_v && !q2_v && !q3_v && !q4_v
                      && !r0_v && !r1_v && !r2_v && !r3_v && !r4_v;
  assign front_done = !g_on && !i_on && !f_on && !f1_v;

  // ================================================================== write-back
  // ACT (L0..L8; NO_L8A: L8 only in a run that stops after it), the pool sums (L8; the first
  // position overwrites), the logits and argmax (L9). With RQ_2L both lanes write in one cycle:
  // consecutive channels, so always different ACT byte lanes and different pool banks.
  logic        act_we, act_we1, l8_act;
  logic [11:0] act_wa, act_wa1, act_ra;
  logic [3:0]  cdiv;                                 // CPU read: channel / 7, channel % 7
  logic [2:0]  cmod, act_lane_q;
  logic [7:0]  logit [16];
  logic [7:0]  best, best0;
  logic [3:0]  cls, cls0;

  assign l8_act  = !NO_L8A || (lay != 4'd8) || (stop_q == 4'd9);
  // FUSE1: the fused segment's depthwise results go to the fusion vector, its pointwise ones to map 1
  assign act_we  = q4_v && (lay != 4'd9) && l8_act && !(fz && !q4_d.pw);
  assign act_we1 = r4_v && (lay != 4'd9) && l8_act && !(fz && !r4_d.pw);
  assign act_wa  = {fz ? 1'b1 : wbuf, q4_d.p, q4_d.cg};
  assign act_wa1 = {fz ? 1'b1 : wbuf, r4_d.p, r4_d.cg};
  assign cdiv    = 4'((int'(act_ch_i) * 37) >> 8);
  assign cmod    = 3'(act_ch_i - 6'(7 * int'(cdiv)));
  assign act_ra  = run ? ga : {act_buf_i, act_pos_i, cdiv};

  for (genvar l = 0; l < 7; l++) begin : g_act
    (* ram_style = "block" *) logic [7:0] am [4096];
    logic w0, w1;
    assign w0 = act_we && (q4_d.cl == 3'(l));
    assign w1 = act_we1 && (r4_d.cl == 3'(l));
    always_ff @(posedge clk_i) begin
      if (w0 || w1) am[w0 ? act_wa : act_wa1] <= w0 ? q4_y : r4_y;
      act_q[l] <= am[act_ra];
    end
  end

  if (!RQ_2L) begin : g_pool1
    logic signed [15:0] pool_m [64];
    always_ff @(posedge clk_i)
      if (q4_v && (lay == 4'd8))
        pool_m[q4_d.c] <= ((q4_d.p == 7'd0) ? 16'sd0 : pool_m[q4_d.c]) + 16'(signed'(q4_y));
    assign pool_rd = pool_m[g_k];
  end else begin : g_pool2                           // lane 0: even channels, lane 1: odd channels
    logic signed [15:0] pool_e [32], pool_o [32];
    always_ff @(posedge clk_i) begin
      if (q4_v && (lay == 4'd8))
        pool_e[q4_d.c[5:1]] <= ((q4_d.p == 7'd0) ? 16'sd0 : pool_e[q4_d.c[5:1]]) + 16'(signed'(q4_y));
      if (r4_v && (lay == 4'd8))
        pool_o[r4_d.c[5:1]] <= ((r4_d.p == 7'd0) ? 16'sd0 : pool_o[r4_d.c[5:1]]) + 16'(signed'(r4_y));
    end
    assign pool_rd = g_k[0] ? pool_o[g_k[5:1]] : pool_e[g_k[5:1]];
  end

  always_ff @(posedge clk_i) begin
    if (act_re_i) act_lane_q <= cmod;
  end

  // ================================================================== FUSE1: line buffer and fusion vector
  // Fill: for each output (oy, ox) in raster order and group g, the taps that are first uses, in tap
  // order: new rows fy = 1, 2 if oy = 0 else 2; new columns fx = 1, 2 if ox = 0 else 2 (outputs with
  // ox = 4 or oy = 24 have none). 1,250 words, one per cycle: F0 reads map 0, F1 writes the slot.
  /* verilator lint_off UNUSEDSIGNAL */            // FUSE1 only
  logic [4:0]  f_oy, f_iy, f_yo, f_ly;
  logic [2:0]  f_ox, f_ix, f_xo, f_lx;
  logic [3:0]  f_g, f_go, f_lk;
  logic        f_a, f_b, f_amax, f_bmax, f_free, f_b1, f_b2, f_old;
  logic [6:0]  f_pin, f_lp, f1_slot;
  logic [6:0]  f_rw;                                 // raster word mod 128: the slot
  logic [14:0] g_key, f_key;
  logic        fv_we0, fv_we1;
  /* verilator lint_on UNUSEDSIGNAL */

  assign f_start = lay_end && !fin && FUSE1 && (lay == 4'd0);
  assign f_iy    = (f_oy == 5'd0) ? {4'd0, f_a} : f_oy + 5'd1;
  assign f_ix    = (f_ox == 3'd0) ? {2'd0, f_b} : f_ox + 3'd1;
  assign f_pin   = 7'({f_iy, 2'b00} + {2'b00, f_iy} + {4'd0, f_ix});
  assign f_rw    = 7'({f_pin, 3'b000} + {2'b00, f_pin, 1'b0} + {6'd0, f_g});
  assign f_ra    = {1'b0, f_pin, f_g};
  assign f_amax  = (f_oy != 5'd0) || f_a;
  assign f_bmax  = (f_ox != 3'd0) || f_b;
  // the word in its slot before: raster word - 128 = 50 (iy - 2) + 10 (ix - 2) + (g - 8), with borrows
  assign f_b1    = (f_g < 4'd8);
  assign f_go    = f_b1 ? f_g + 4'd2 : f_g - 4'd8;
  assign f_b2    = ({1'b0, f_ix} < (f_b1 ? 4'd3 : 4'd2));
  assign f_xo    = f_b2 ? f_ix + (f_b1 ? 3'd2 : 3'd3) : f_ix - (f_b1 ? 3'd3 : 3'd2);
  assign f_old   = (f_iy >= (f_b2 ? 5'd3 : 5'd2));
  assign f_yo    = f_iy - (f_b2 ? 5'd3 : 5'd2);
  // its last reader: output (min(y + 1, 24), min(x + 1, 4)), group g_old, tap 3 fy + fx
  assign f_ly    = (f_yo == 5'd24) ? 5'd24 : f_yo + 5'd1;
  assign f_lx    = (f_xo == 3'd4) ? 3'd4 : f_xo + 3'd1;
  assign f_lk    = ((f_yo == 5'd24) ? 4'd3 : 4'd0) + ((f_xo == 3'd4) ? 4'd1 : 4'd0);
  assign f_lp    = 7'({f_ly, 2'b00} + {2'b00, f_ly} + {4'd0, f_lx});
  assign f_key   = {f_lp, f_go, f_lk};
  // the gather's next chunk (a pointwise vector comes after all depthwise vectors of position g_p)
  assign g_key   = {g_p, g_pw ? 4'd10 : g_g, g_pw ? 4'd0 : g_k[3:0]};
  assign f_free  = !f_old || !g_on || (g_key > f_key);

  assign fv_we0 = q4_v && fz && !q4_d.pw;
  assign fv_we1 = r4_v && fz && !r4_d.pw;

  if (FUSE1) begin : g_fuse
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        f_on   <= 1'b0;
        f1_v   <= 1'b0;
        f_oy   <= 5'd0;
        f_ox   <= 3'd0;
        f_g    <= 4'd0;
        f_a    <= 1'b0;
        f_b    <= 1'b0;
        fl_cnt <= 11'd0;
        gq     <= 11'd0;
        fv_wp  <= 7'd0;
        fv_wc  <= 7'd0;
      end else begin
        f1_v <= f_on && f_free;
        if (f1_v) fl_cnt <= fl_cnt + 11'd1;
        if (g_req && fz && !g_pw && g_fu) gq <= gq + 11'd1;
        // both lanes of one cycle come from one pass, so from one position
        if (fv_we0 || fv_we1) begin
          if ((fv_we0 ? q4_d.p : r4_d.p) != fv_wp) begin
            fv_wp <= fv_we0 ? q4_d.p : r4_d.p;
            fv_wc <= {6'd0, fv_we0} + {6'd0, fv_we1};
          end else fv_wc <= fv_wc + {6'd0, fv_we0} + {6'd0, fv_we1};
        end
        if (f_start) begin
          f_on   <= 1'b1;
          f_oy   <= 5'd0;
          f_ox   <= 3'd0;
          f_g    <= 4'd0;
          f_a    <= 1'b0;
          f_b    <= 1'b0;
          fl_cnt <= 11'd0;
          gq     <= 11'd0;
          fv_wp  <= 7'd0;
          fv_wc  <= 7'd0;
        end else if (f_on && f_free) begin
          if (!f_bmax) f_b <= 1'b1;
          else begin
            f_b <= 1'b0;
            if (!f_amax) f_a <= 1'b1;
            else begin
              f_a <= 1'b0;
              if (f_g != 4'd9) f_g <= f_g + 4'd1;
              else begin
                f_g <= 4'd0;
                if (f_ox != 3'd3) f_ox <= f_ox + 3'd1;
                else begin
                  f_ox <= 3'd0;
                  if (f_oy == 5'd23) f_on <= 1'b0;
                  else f_oy <= f_oy + 5'd1;
                end
              end
            end
          end
        end
      end
    end
    always_ff @(posedge clk_i) if (f_on && f_free) f1_slot <= f_rw;

    // line buffer: 128 x 7 bytes (F1 writes the ACT word read in F0)
    for (genvar l = 0; l < 7; l++) begin : g_lb
      (* ram_style = "distributed" *) logic [7:0] lbm [128];
      always_ff @(posedge clk_i) begin
        if (f1_v) lbm[f1_slot] <= act_q[l];
        lb_q[l] <= lbm[lb_ra];
      end
    end
    // fusion vector: byte lane c mod 8, word {p mod 2, c / 8}; the two lanes have consecutive channels
    for (genvar l = 0; l < 8; l++) begin : g_fv
      (* ram_style = "distributed" *) logic [7:0] fvm [16];
      logic w0, w1;
      assign w0 = fv_we0 && (q4_d.c[2:0] == 3'(l));
      assign w1 = fv_we1 && (r4_d.c[2:0] == 3'(l));
      always_ff @(posedge clk_i) begin
        if (w0 || w1) fvm[w0 ? {q4_d.p[0], q4_d.c[5:3]} : {r4_d.p[0], r4_d.c[5:3]}] <= w0 ? q4_y : r4_y;
        fv_q[l] <= fvm[fv_ra];
      end
    end
  end else begin : g_fuse_off
    assign {f_on, f1_v, f_oy, f_ox, f_g, f_a, f_b, fl_cnt, gq, fv_wp, fv_wc, f1_slot} = '0;
    for (genvar l = 0; l < 7; l++) begin : g_lb0
      assign lb_q[l] = 8'd0;
    end
    for (genvar l = 0; l < 8; l++) begin : g_fv0
      assign fv_q[l] = 8'd0;
    end
  end

  // argmax, first maximum: lane 0 before lane 1 (an L9 pair is channels 2k, 2k + 1)
  always_comb begin
    best0 = best;
    cls0  = cls;
    if (q4_v && (lay == 4'd9) && ((q4_d.c == 6'd0) || ($signed(q4_y) > $signed(best)))) begin
      best0 = q4_y;
      cls0  = q4_d.c[3:0];
    end
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < 16; i++) logit[i] <= 8'd0;
      best <= 8'd0;
      cls  <= 4'd0;
    end else begin
      if (q4_v && (lay == 4'd9)) logit[q4_d.c[3:0]] <= q4_y;
      if (r4_v && (lay == 4'd9)) logit[r4_d.c[3:0]] <= r4_y;
      best <= best0;
      cls  <= cls0;
      if (r4_v && (lay == 4'd9) && ((r4_d.c == 6'd0) || ($signed(r4_y) > $signed(best0)))) begin
        best <= r4_y;
        cls  <= r4_d.c[3:0];
      end
    end
  end

  assign act_rdata_o = act_q[act_lane_q];
  assign logit_o     = logit[logit_addr_i];
  assign busy_o      = run;
  assign done_o      = done_q;
  assign err_o       = err_q;
  assign layer_o     = fz ? 4'd2 : lay;              // FUSE1: the fused segment ends with L2
  assign class_o     = cls;
  assign wp_all_o    = wall_q;
  assign lmx_o       = lmx_q;
  assign wmax_o      = w_max;

endmodule
