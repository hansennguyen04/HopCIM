// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_endpoint: OBI slave wrapper of cim_engine for X-HEEP, with a hardware MFCC front-end and a
// network layer sequencer next to the CPU port. The CPU loads weights, writes inputs and starts
// passes (R2), writes audio samples and lets the front-end compute lraw of a frame (R5a), or
// writes the int8 features and lets the sequencer run the DS-CNN to a class (R6b). FE_DED selects
// the front-end: 0 = cim_fe, which runs its matrix steps on the engine; 1 = mfcc_ded (baseline
// B3), a dedicated radix-2 front-end that leaves the engine to the CPU or the sequencer.
//
// Address map (offset = addr[17:0], 256 KiB window, 32-bit word accesses only):
//   0x0_0000 CTRL      W  [31] go, [16] accum, [12:8] xbits, [6:0] tile. A go write is held
//                         (no grant) until the engine is ready, so back-to-back passes need no polling.
//   0x0_0004 STATUS    R  [0] ready, [1] done (sticky, W1C, also cleared by go), [2] irq pending
//   0x0_0008 IRQ_EN    RW [0] irq_o while STATUS.done, [1] while FE_STATUS.done, [2] while SEQ_STATUS.done,
//                         [3] while AUD_STATUS.pending
//   0x0_000C DONE_CNT  R  number of completed CPU passes (front-end passes are not counted)
//   0x0_0010 FE_CTRL   W  [31] go, [9:0] ring index of the frame's first sample. Held until the
//                         engine has no pass in flight; the frame is samples base .. base+479.
//   0x0_0014 FE_STATUS R  [0] busy, [1] done (sticky, W1C, also cleared by a front-end go)
//   0x0_0018 CYCLES    R  free-running clock-cycle counter (wall time; the CPU's mcycle can stop)
//   0x0_0020 SEQ_CTRL  W  [31] go, [30] window path first (R5b), [29] window path only, [28] new
//                         frame (copy the front-end's lraw), [27:24] stop (0: whole network, s: after
//                         layer s - 1), [23] stream reset, [21:16] FEAT row of frame 0, [6:0] first
//                         network tile. Held until no engine pass is in flight and cim_fe is idle
//                         (with [28], until no front-end runs). Reads back the last go word.
//   0x0_0024 SEQ_STATUS R [0] busy, [1] done (sticky, W1C, also cleared by go), [2] error (a result
//                         was lost), [7:4] class (argmax of the logits), [11:8] layer, [12] the last
//                         window path recomputed all 49 frames
//   0x0_0028 SEQ_ZP    RW [7:0] conv1 input zero point, [15:8] activation zp, [23:16] logits zp
//   0x0_002C SEQ_LMX   R  log2fix16 of the window maximum (Q16), from the last window path
//   0x0_0030 SEQ_WMAX  R  window maximum (signed), from the last window path
//   0x0_0034 AUD_CTRL  RW live audio path (R7, cim_audio): [0] enable (a 0 -> 1 starts a stream:
//                         PDM microphone -> 16 kHz ring -> front-end -> window path -> network per
//                         hop, without the CPU), [1] run the network on full windows, [14:8] first
//                         network tile, [19:16] k - 1: the network and its decision every k hops
//                         (front-end and window path still every hop). While enabled, CPU front-end
//                         and sequencer gos and SAMPLES writes are ignored.
//   0x0_0038 AUD_STATUS R [0] enabled, [1] decision pending (sticky, W1C), [2] overrun (sticky, W1C),
//                         [7:4] class of the last decision, [31:16] decisions since the start
//   0x0_003C AUD_HOPS  R  hops since the start
//   0x0_0040 RES_LO    R  16 words: bits [31:0] of the latched results of the last CPU pass
//   0x0_0080 RES_HI    R  16 words: bits [47:32], sign-extended
//   0x0_1000 X         W  64 words: input x[r] = wdata[23:0] (word r)
//   0x0_2000 SAMPLES   RW 1024 words: audio ring, sample i = wdata[15:0] (word i); reads (sign-extended)
//                         are held while a front-end runs
//   0x0_3000 LRAW      R  40 words: lraw of the last front-end frame (0x8000_0000 = exact zero)
//   0x0_4000 FEAT      RW int8 features: row r (0..48) = off[9:4], word w = off[3:2] holds bytes
//                         4w .. 4w+3 of the row (10 per row)
//   0x0_5000 LOGITS    R  12 words: int8 logits of the last full run, sign-extended
//   0x0_8000 PARAMS    W  entry 64 L + c = off[13:4], field off[3:2]: 0 bias, 1 qm, 2 e = -shift
//   0x1_0000 ACT       R  debug: activation of buffer off[15], position off[14:8], channel off[7:2]
//   0x2_0000 WEIGHTS   W  128 KiB: tile = off[16:10], row = off[9:4], word = off[3:2]; 4 int8 per word
// X and WEIGHTS writes are held for 10 cycles after a pass starts (the engine reads them then).
// While cim_fe runs, CPU go, X and WEIGHTS writes are held; SAMPLES writes and all reads are
// served (the front-end reads only the 480 samples of its frame). cim_fe uses tiles 0..5.
// While the sequencer runs, CPU go, X, WEIGHTS, FEAT and PARAMS writes, ACT, FEAT and LRAW reads
// and (with cim_fe) a front-end go are held; it overwrites both X banks. mfcc_ded (FE_DED) may run
// meanwhile. Every SAMPLES write also updates the hop maximum, which a window path takes (R5b).

module cim_endpoint
  import xheep_obi_pkg::*;
#(
    parameter int unsigned TILES  = 128,
    parameter bit          FE_DED = 1'b0,
    parameter int unsigned PDM_CLKDIV = 15,            // PDM clock = clk / 15 (1.024 MHz at 15.36 MHz)
    parameter bit          SEQ_S  = 1'b0,              // baseline S in the sequencer: sign-gated requantisation,
                                                       //   a second requantisation lane, no L8 write to ACT
    parameter bit          SEQ_B1 = 1'b0               // with SEQ_S: StreamFuse B in block 1 (docs/STREAMFUSE_B1.md)
) (
    input  logic           clk_i,
    input  logic           rst_ni,
    // word accesses only (be ignored); the region is decoded upstream (addr[31:18] ignored)
    /* verilator lint_off UNUSEDSIGNAL */
    input  xheep_obi_req_t obi_req_i,
    /* verilator lint_on UNUSEDSIGNAL */
    output xheep_obi_rsp_t obi_rsp_o,
    output logic           irq_o,
    // on-board PDM microphone (R7; tie pdm_data_i low and leave pdm_clk_o open if unused)
    output logic           pdm_clk_o,
    input  logic           pdm_data_i
);

  localparam int unsigned TW = $clog2(TILES);

  // ------------------------------------------------------------------ decode
  logic [17:0] off;
  logic        is_w, is_x, is_reg, is_samp, is_lraw, is_feat, is_logit, is_prm, is_act;
  assign off      = obi_req_i.addr[17:0];
  assign is_w     = off[17];
  assign is_x     = (off[17:12] == 6'h01);
  assign is_samp  = (off[17:12] == 6'h02);
  assign is_lraw  = (off[17:8] == 10'h030);
  assign is_reg   = (off[17:8] == 10'h000);
  assign is_feat  = (off[17:12] == 6'h04);
  assign is_logit = (off[17:12] == 6'h05);
  assign is_prm   = (off[17:14] == 4'b0010);
  assign is_act   = (off[17:16] == 2'b01);

  logic wr, go, fe_go, seq_go, data_wr, seq_mem, samp_rd, gnt;
  // live audio path (R7): AUD_CTRL and cim_audio's outputs
  logic [14:0]   aud_ctrl;
  logic [3:0]    aud_net_km1;
  logic          aud_en, aud_fe_go, aud_seq_go, aud_ring_we, aud_pend, aud_ovr;
  logic [9:0]    aud_fe_base, aud_ring_addr;
  logic [15:0]   aud_ring_data, aud_bmax, aud_ndec;
  logic [31:0]   aud_seq_word, aud_nhop;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0]   seq_cfg;                              // SEQ_CTRL word of a start (CPU or audio path)
  /* verilator lint_on UNUSEDSIGNAL */
  logic [3:0]    aud_class;
  assign aud_en = aud_ctrl[0];
  logic eng_ready, eng_done, eng_busy, fe_busy, fe_done, fe_owns, seq_busy;
  assign fe_owns = fe_busy && !FE_DED;                 // the front-end drives the engine
  logic [3:0] lock;                                    // cycles left in which X/W writes are held
  assign wr      = obi_req_i.req && obi_req_i.we;
  assign go      = wr && is_reg && (off[7:0] == 8'h00) && obi_req_i.wdata[31];
  assign fe_go   = wr && is_reg && (off[7:0] == 8'h10) && obi_req_i.wdata[31];
  assign seq_go  = wr && is_reg && (off[7:0] == 8'h20) && obi_req_i.wdata[31];
  assign data_wr = wr && (is_x || is_w);
  assign seq_mem = obi_req_i.req && (obi_req_i.we ? (is_feat || is_prm) : (is_act || is_feat || is_lraw));
  assign samp_rd = obi_req_i.req && !obi_req_i.we && is_samp;
  always_comb begin
    if (go)           gnt = obi_req_i.req && eng_ready && !fe_owns && !seq_busy;
    else if (fe_go)   gnt = obi_req_i.req && !fe_busy && (FE_DED || (!eng_busy && !seq_busy));
    else if (seq_go)  gnt = obi_req_i.req && !seq_busy && !fe_owns && !eng_busy &&
                            !(obi_req_i.wdata[28] && fe_busy);
    else if (data_wr) gnt = obi_req_i.req && (lock == 4'd0) && !fe_owns && !seq_busy;
    else if (seq_mem) gnt = obi_req_i.req && !seq_busy;
    else if (samp_rd) gnt = obi_req_i.req && !fe_busy;  // the ring's read port is the front-end's
    else              gnt = obi_req_i.req;
  end

  // ------------------------------------------------------------------ audio ring
  (* ram_style = "block" *) logic [15:0] ring [1024];
  logic [9:0]  samp_addr;                              // the front-end's read address
  logic [15:0] samp_q;
  logic        ring_we;                                // one write port: the microphone or the CPU (two
  logic [9:0]  ring_wa;                                // write statements would be two ports, and the
  logic [15:0] ring_wd;                                // ring would no longer fit a block RAM)
  assign ring_we = aud_en ? aud_ring_we : (gnt && wr && is_samp);
  assign ring_wa = aud_en ? aud_ring_addr : off[11:2];
  assign ring_wd = aud_en ? aud_ring_data : obi_req_i.wdata[15:0];
  always_ff @(posedge clk_i) begin
    if (ring_we) ring[ring_wa] <= ring_wd;
    samp_q <= ring[fe_busy ? samp_addr : off[11:2]];
  end

  // ------------------------------------------------------------------ front-end
  logic          fe_x_we, fe_pass, fe_accum;
  logic [5:0]    fe_x_idx;
  logic [23:0]   fe_x_data;
  logic [TW-1:0] fe_tile;
  logic [4:0]    fe_xbits;
  logic [31:0]   lraw_rdata;
  logic          seq_lraw_rd;                          // the sequencer copies the front-end's lraw (R5b)
  logic [5:0]    seq_lraw_addr;
  logic signed [47:0] res [16];

  if (FE_DED) begin : g_fe_ded
    mfcc_ded i_fe (
        .clk_i,
        .rst_ni,
        .start_i    ((gnt && fe_go && !aud_en) || aud_fe_go),
        .base_i     (aud_fe_go ? aud_fe_base : obi_req_i.wdata[9:0]),
        .busy_o     (fe_busy),
        .done_o     (fe_done),
        .samp_addr_o(samp_addr),
        .samp_i     (signed'(samp_q)),
        .lraw_addr_i(seq_lraw_rd ? seq_lraw_addr : off[7:2]),
        .lraw_o     (lraw_rdata)
    );
    assign fe_x_we   = 1'b0;
    assign fe_x_idx  = '0;
    assign fe_x_data = '0;
    assign fe_pass   = 1'b0;
    assign fe_tile   = '0;
    assign fe_xbits  = '0;
    assign fe_accum  = 1'b0;
  end else begin : g_fe_cim
    cim_fe #(.TW(TW)) i_fe (
        .clk_i,
        .rst_ni,
        .start_i    ((gnt && fe_go && !aud_en) || aud_fe_go),
        .base_i     (aud_fe_go ? aud_fe_base : obi_req_i.wdata[9:0]),
        .busy_o     (fe_busy),
        .done_o     (fe_done),
        .samp_addr_o(samp_addr),
        .samp_i     (signed'(samp_q)),
        .x_we_o     (fe_x_we),
        .x_idx_o    (fe_x_idx),
        .x_data_o   (fe_x_data),
        .go_o       (fe_pass),
        .tile_o     (fe_tile),
        .xbits_o    (fe_xbits),
        .accum_o    (fe_accum),
        .eng_ready_i(eng_ready),
        .eng_done_i (eng_done),
        .res_i      (res),
        .lraw_addr_i(seq_lraw_rd ? seq_lraw_addr : off[7:2]),
        .lraw_o     (lraw_rdata)
    );
  end

  // ------------------------------------------------------------------ layer sequencer
  logic          seq_done, seq_err, seq_xw_we, seq_xw_bank, seq_pass, seq_xbank;
  logic          seq_wp_all, seq_bmax_take;
  logic [3:0]    seq_layer, seq_class;
  logic [4:0]    seq_xbits;
  logic [19:0]   seq_lmx;
  logic [15:0]   seq_wmax, blk_max;
  logic [31:0]   feat_rdata;
  logic [5:0]    seq_xw_row;
  logic [7:0]    seq_xw_mask, act_rdata, logit_rdata;
  logic [23:0]   seq_xw_data [8];
  logic [TW-1:0] seq_tile;
  logic [31:0]   seq_zp;

  // ------------------------------------------------------------------ live audio path (R7)
  assign seq_cfg = aud_seq_go ? aud_seq_word : obi_req_i.wdata;
  cim_audio #(.TW(TW), .CLKDIV(PDM_CLKDIV)) i_audio (
      .clk_i,
      .rst_ni,
      .en_i       (aud_en),
      .net_i      (aud_ctrl[1]),
      .net_km1_i  (aud_net_km1),
      .tbase_i    (aud_ctrl[8+TW-1:8]),
      .pdm_clk_o,
      .pdm_data_i,
      .ring_we_o  (aud_ring_we),
      .ring_addr_o(aud_ring_addr),
      .ring_data_o(aud_ring_data),
      .bmax_o     (aud_bmax),
      .fe_ok_i    (!fe_busy && (FE_DED || (!eng_busy && !seq_busy))),
      .fe_done_i  (fe_done),
      .fe_go_o    (aud_fe_go),
      .fe_base_o  (aud_fe_base),
      .seq_ok_i   (!seq_busy && !fe_busy && !eng_busy),
      .seq_done_i (seq_done),
      .seq_class_i(seq_class),
      .seq_go_o   (aud_seq_go),
      .seq_word_o (aud_seq_word),
      .clr_pend_i (gnt && wr && is_reg && (off[7:0] == 8'h38) && obi_req_i.wdata[1]),
      .clr_ovr_i  (gnt && wr && is_reg && (off[7:0] == 8'h38) && obi_req_i.wdata[2]),
      .pend_o     (aud_pend),
      .ovr_o      (aud_ovr),
      .class_o    (aud_class),
      .ndec_o     (aud_ndec),
      .nhop_o     (aud_nhop)
  );

  cim_seq #(.TW(TW), .XB(24), .RQ_SG(SEQ_S), .RQ_2L(SEQ_S), .NO_L8A(SEQ_S), .FUSE1(SEQ_S && SEQ_B1)) i_seq (
      .clk_i,
      .rst_ni,
      .start_i     ((gnt && seq_go && !aud_en) || aud_seq_go),
      .tbase_i     (seq_cfg[TW-1:0]),
      .stop_i      (seq_cfg[27:24]),
      .frow_i      (seq_cfg[21:16]),
      .zin0_i      (seq_zp[7:0]),
      .zact_i      (seq_zp[15:8]),
      .zfc_i       (seq_zp[23:16]),
      .busy_o      (seq_busy),
      .done_o      (seq_done),
      .err_o       (seq_err),
      .layer_o     (seq_layer),
      .class_o     (seq_class),
      .wp_i        (seq_cfg[30]),
      .wp_only_i   (seq_cfg[29]),
      .wp_frame_i  (seq_cfg[28]),
      .wp_reset_i  (seq_cfg[23]),
      .wp_all_o    (seq_wp_all),
      .lmx_o       (seq_lmx),
      .wmax_o      (seq_wmax),
      .bmax_i      (aud_en ? aud_bmax : blk_max),
      .bmax_take_o (seq_bmax_take),
      .lraw_rd_o   (seq_lraw_rd),
      .lraw_addr_o (seq_lraw_addr),
      .lraw_i      (lraw_rdata),
      .feat_we_i   (gnt && wr && is_feat),
      .feat_addr_i (off[9:2]),
      .feat_data_i (obi_req_i.wdata),
      .prm_we_i    (gnt && wr && is_prm),
      .prm_addr_i  (off[13:4]),
      .prm_field_i (off[3:2]),
      .prm_data_i  (obi_req_i.wdata),
      .feat_rdata_o(feat_rdata),
      .act_re_i    (gnt && !obi_req_i.we && is_act),
      .act_buf_i   (off[15]),
      .act_pos_i   (off[14:8]),
      .act_ch_i    (off[7:2]),
      .act_rdata_o (act_rdata),
      .logit_addr_i(off[5:2]),
      .logit_o     (logit_rdata),
      .xw_we_o     (seq_xw_we),
      .xw_bank_o   (seq_xw_bank),
      .xw_row_o    (seq_xw_row),
      .xw_mask_o   (seq_xw_mask),
      .xw_data_o   (seq_xw_data),
      .go_o        (seq_pass),
      .tile_o      (seq_tile),
      .xbank_o     (seq_xbank),
      .xbits_o     (seq_xbits),
      .eng_ready_i (eng_ready),
      .eng_done_i  (eng_done),
      .res_i       (res)
  );

  // ------------------------------------------------------------------ engine (CPU, front-end or sequencer)
  cim_engine #(.TILES(TILES), .XB(24), .ACCW(48)) i_engine (
      .clk_i,
      .rst_ni,
      .w_we_i   (gnt && wr && is_w),
      .w_tile_i (off[10+TW-1:10]),
      .w_row_i  (off[9:4]),
      .w_word_i (off[3:2]),
      .w_data_i (obi_req_i.wdata),
      .x_we_i   (fe_owns ? fe_x_we : (gnt && wr && is_x)),
      .x_idx_i  (fe_owns ? fe_x_idx : off[7:2]),
      .x_data_i (fe_owns ? fe_x_data : obi_req_i.wdata[23:0]),
      .xw_we_i  (seq_xw_we),
      .xw_bank_i(seq_xw_bank),
      .xw_row_i (seq_xw_row),
      .xw_mask_i(seq_xw_mask),
      .xw_data_i(seq_xw_data),
      .start_i  (seq_busy ? seq_pass : (fe_owns ? fe_pass : (gnt && go))),
      .tile_i   (seq_busy ? seq_tile : (fe_owns ? fe_tile : obi_req_i.wdata[TW-1:0])),
      .xbits_i  (seq_busy ? seq_xbits : (fe_owns ? fe_xbits : obi_req_i.wdata[12:8])),
      .accum_i  (seq_busy ? 1'b0 : (fe_owns ? fe_accum : obi_req_i.wdata[16])),
      .xbank_i  (seq_busy && seq_xbank),
      .ready_o  (eng_ready),
      .done_o   (eng_done),
      .busy_o   (eng_busy),
      .res_o    (res)
  );

  // ------------------------------------------------------------------ registers
  logic signed [47:0] res_lat [16];
  logic        done_sticky, fe_done_sticky, seq_done_sticky, cpu_done;
  logic [3:0]  irq_en;
  logic [31:0] done_cnt, ctrl_q, cycles, seq_ctrl_q;

  assign cpu_done = eng_done && !fe_owns && !seq_busy; // passes while cim_fe or the sequencer runs are theirs

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      lock           <= 4'd0;
      done_sticky    <= 1'b0;
      fe_done_sticky <= 1'b0;
      seq_done_sticky <= 1'b0;
      irq_en         <= 4'd0;
      aud_ctrl       <= 15'd0;
      aud_net_km1    <= 4'd0;
      done_cnt       <= 32'd0;
      ctrl_q         <= 32'd0;
      cycles         <= 32'd0;
      seq_ctrl_q     <= 32'd0;
      seq_zp         <= 32'd0;
      for (int n = 0; n < 16; n++) res_lat[n] <= '0;
    end else begin
      cycles <= cycles + 32'd1;
      if (lock != 4'd0) lock <= lock - 4'd1;
      if (gnt && go) begin
        lock        <= 4'd10;
        ctrl_q      <= obi_req_i.wdata;
        done_sticky <= 1'b0;
      end
      if (gnt && wr && is_reg && (off[7:0] == 8'h04) && obi_req_i.wdata[1]) done_sticky <= 1'b0;
      if (gnt && wr && is_reg && (off[7:0] == 8'h08)) irq_en <= obi_req_i.wdata[3:0];
      if (gnt && wr && is_reg && (off[7:0] == 8'h34)) begin
        aud_ctrl    <= obi_req_i.wdata[14:0];
        aud_net_km1 <= obi_req_i.wdata[19:16];
      end
      if (cpu_done) begin                              // a completion wins over a same-cycle go; with
        done_sticky <= 1'b1;                           // back-to-back passes track passes via DONE_CNT
        done_cnt    <= done_cnt + 32'd1;
        for (int n = 0; n < 16; n++) res_lat[n] <= res[n];
      end
      if (gnt && fe_go) fe_done_sticky <= 1'b0;
      if (gnt && wr && is_reg && (off[7:0] == 8'h14) && obi_req_i.wdata[1]) fe_done_sticky <= 1'b0;
      if (fe_done) fe_done_sticky <= 1'b1;
      if (gnt && seq_go) begin
        seq_ctrl_q      <= obi_req_i.wdata;
        seq_done_sticky <= 1'b0;
      end
      if (gnt && wr && is_reg && (off[7:0] == 8'h24) && obi_req_i.wdata[1]) seq_done_sticky <= 1'b0;
      if (gnt && wr && is_reg && (off[7:0] == 8'h28)) seq_zp <= {8'd0, obi_req_i.wdata[23:0]};
      if (seq_done) seq_done_sticky <= 1'b1;
    end
  end

  assign irq_o = (done_sticky && irq_en[0]) || (fe_done_sticky && irq_en[1]) || (seq_done_sticky && irq_en[2]) ||
                 (aud_pend && irq_en[3]);

  // ------------------------------------------------------------------ read data / response
  logic [31:0] rdata_d, rdata_q;
  logic        rvalid_q;
  always_comb begin
    rdata_d = 32'd0;
    if (is_reg) begin
      unique casez (off[7:0])
        8'h00:      rdata_d = ctrl_q;
        8'h04:      rdata_d = {29'd0, irq_o, done_sticky, eng_ready};
        8'h08:      rdata_d = {28'd0, irq_en};
        8'h0C:      rdata_d = done_cnt;
        8'h14:      rdata_d = {30'd0, fe_done_sticky, fe_busy};
        8'h18:      rdata_d = cycles;
        8'h20:      rdata_d = seq_ctrl_q;
        8'h24:      rdata_d = {19'd0, seq_wp_all, seq_layer, seq_class, 1'b0, seq_err, seq_done_sticky, seq_busy};
        8'h28:      rdata_d = seq_zp;
        8'h2C:      rdata_d = {12'd0, seq_lmx};
        8'h30:      rdata_d = {{16{seq_wmax[15]}}, seq_wmax};
        8'h34:      rdata_d = {12'd0, aud_net_km1, 1'b0, aud_ctrl};
        8'h38:      rdata_d = {aud_ndec, 8'd0, aud_class, 1'b0, aud_ovr, aud_pend, aud_en};
        8'h3C:      rdata_d = aud_nhop;
        8'b01??_??00: rdata_d = res_lat[off[5:2]][31:0];
        8'b10??_??00: rdata_d = {{16{res_lat[off[5:2]][47]}}, res_lat[off[5:2]][47:32]};
        default:    rdata_d = 32'd0;
      endcase
    end else if (is_lraw) rdata_d = lraw_rdata;
    else if (is_logit) rdata_d = {{24{logit_rdata[7]}}, logit_rdata};
  end

  // ACT (BRAM) and FEAT (registered read) data come in the rvalid cycle
  logic act_rd_q, feat_rd_q, samp_rd_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rvalid_q  <= 1'b0;
      rdata_q   <= 32'd0;
      act_rd_q  <= 1'b0;
      feat_rd_q <= 1'b0;
      samp_rd_q <= 1'b0;
    end else begin
      rvalid_q  <= gnt;
      rdata_q   <= rdata_d;
      act_rd_q  <= gnt && !obi_req_i.we && is_act;
      feat_rd_q <= gnt && !obi_req_i.we && is_feat;
      samp_rd_q <= gnt && !obi_req_i.we && is_samp;
    end
  end

  // maximum sample written since the window path last took it (signed; -32768 = none)
  logic samp_wr;
  assign samp_wr = gnt && wr && is_samp;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) blk_max <= 16'h8000;
    else if (seq_bmax_take) blk_max <= samp_wr ? obi_req_i.wdata[15:0] : 16'h8000;
    else if (samp_wr && ($signed(obi_req_i.wdata[15:0]) > $signed(blk_max))) blk_max <= obi_req_i.wdata[15:0];
  end

  assign obi_rsp_o.gnt    = gnt;
  assign obi_rsp_o.rvalid = rvalid_q;
  assign obi_rsp_o.rdata  = act_rd_q ? {{24{act_rdata[7]}}, act_rdata} :
                            feat_rd_q ? feat_rdata : samp_rd_q ? {{16{samp_q[15]}}, samp_q} : rdata_q;

endmodule
