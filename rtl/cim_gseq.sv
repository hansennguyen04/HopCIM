// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_gseq: descriptor-driven layer sequencer that runs MLPerf Tiny's int8 ResNet8 (a published model, unchanged) on
// cim_engine without the CPU (docs/NET2_GATES.md). Built in the endpoint with SEQ_NET2; cim_seq is not used by it.
//
// Operations (16 records of 8 words, written by the CPU while idle; fields below): conv, add, pool, fc.
//   conv K x K (K = 1, 3), stride 1 or 2, SAME padding, Cin 3 (image), 16, 32 or 64, Cout 16, 32, 64, square maps 8, 16, 32.
//   A position's input vector is tap-major (row = tap x Cin + c, tap = fy x K + fx) and is cut into R row tiles of 64 rows. For each
//   of the Cc column tiles (16 outputs) the R row passes run in order and accumulate in the engine; the last one's 16 sums are
//   requantised (srdhm and rdbp of TFLite, two per cycle) and written as two 8-channel words.
//   add: two maps of the same shape, both rescaled (left shift 20), summed, rescaled; two elements per cycle.
//   pool: average of an 8 x 8 map of 64 channels, half away from zero. fc: 64 pool results -> 10 logits.
// Memories: ACT, three banks of 2,048 words of 64 bits (a map in raster order, word = position x (C / 8) + group); IMG, 32 x 32
// pixels of 3 bytes; parameters, entry 64 L + c (bias, qm, e = -shift) in an even and an odd bank; pool results and logits.
// Timing (docs/NET2_GATES.md section 2): the gather addresses one word per cycle; the word is written into the engine's input bank
// two cycles later and the vector is complete at the third; the pass starts then, when the engine is ready and the
// previous one has started 9 cycles ago. A bank may be rewritten from the cycle after its pass started, in ascending
// row groups (the engine reads group g at edge t0 + 2 + g).

module cim_gseq #(
    parameter int unsigned TW = 7,
    parameter int unsigned XB = 24
) (
    input  logic                  clk_i,
    input  logic                  rst_ni,
    input  logic                  start_i,            // run operations 0 .. nops_i - 1 (idle only)
    input  logic [3:0]            nops_i,
    output logic                  busy_o,
    output logic                  done_o,             // one cycle at the end of the run
    output logic                  err_o,              // a result was lost (sticky, cleared by start)
    output logic [3:0]            class_o,            // argmax of the logits (first maximum)
    input  logic                  dsc_we_i,           // record word {operation, w}, w = 0 .. 7
    input  logic [6:0]            dsc_addr_i,
    input  logic [31:0]           dsc_data_i,
    input  logic                  prm_we_i,           // parameter entry 64 L + c
    input  logic [9:0]            prm_addr_i,
    input  logic [1:0]            prm_field_i,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0]           prm_data_i,         // [31] unused
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                  img_we_i,           // pixel y x 32 + x: bytes 0, 1, 2 = channels 0, 1, 2
    input  logic [9:0]            img_addr_i,
    input  logic [23:0]           img_data_i,
    input  logic                  dbg_re_i,           // ACT read while idle: {bank, word, half}, data in the next cycle
    input  logic [13:0]           dbg_addr_i,
    output logic [31:0]           dbg_rdata_o,
    input  logic [3:0]            logit_addr_i,
    output logic [7:0]            logit_o,
    output logic                  xw_we_o,            // engine input banks
    output logic                  xw_bank_o,
    output logic [5:0]            xw_row_o,
    output logic [7:0]            xw_mask_o,
    output logic [XB-1:0]         xw_data_o [8],
    output logic                  go_o,               // engine passes
    output logic [TW-1:0]         tile_o,
    output logic                  accum_o,
    output logic                  xbank_o,
    output logic [4:0]            xbits_o,
    input  logic                  eng_ready_i,
    input  logic                  eng_done_i,
    input  logic signed [47:0]    res_i [16]
);

  localparam logic [1:0] OP_CONV = 2'd0, OP_ADD = 2'd1, OP_POOL = 2'd2, OP_FC = 2'd3;
  assign xbits_o = 5'd9;

  // ------------------------------------------------------------------ descriptors
  // w0: [1:0] type, [2] K = 3, [3] stride 2, [5:4] Cin code (0: 3, 1: 16, 2: 32, 3: 64), [7:6] log2 H - 3 (0: 8, 1: 16, 2: 32),
  //     [9:8] source bank (A), [11:10] destination bank, [12] RELU, [18:13] Cout - 1, [22:19] R, [25:23] Cc, [29:26] parameter
  //     layer, [31:30] source bank B (add)
  // w1: [6:0] tile base, [15:8] zin (A), [23:16] zout, [31:24] zB;  w2, w3, w4: qA, qB, qO;  w5: [4:0] eA, [9:5] eB, [14:10] eO
  (* ram_style = "distributed" *) logic [31:0] dsc_m [128];
  always_ff @(posedge clk_i) if (dsc_we_i) dsc_m[dsc_addr_i] <= dsc_data_i;

  logic [31:0] dr [8], nr [8];                        // current operation's record, the next one's
  logic [1:0]  o_type, o_cin, o_hl, o_src, o_dst, o_srcb;
  logic        o_k3, o_s2, o_relu;
  logic [5:0]  o_coutm1;
  logic [3:0]  o_R, o_lpar;
  logic [2:0]  o_C;
  logic [6:0]  o_tbase;
  logic [7:0]  o_za, o_zo, o_zb;
  logic [30:0] o_qa, o_qb, o_qo;
  logic [4:0]  o_ea, o_eb, o_eo;
  assign {o_srcb, o_lpar, o_C, o_R, o_coutm1, o_relu, o_dst, o_src, o_hl, o_cin, o_s2, o_k3, o_type} = dr[0];
  assign {o_zb, o_zo, o_za} = dr[1][31:8];
  assign o_tbase = dr[1][6:0];
  assign o_qa = dr[2][30:0];
  assign o_qb = dr[3][30:0];
  assign o_qo = dr[4][30:0];
  assign {o_eo, o_eb, o_ea} = dr[5][14:0];

  // ------------------------------------------------------------------ top control
  typedef enum logic [2:0] {S_IDLE, S_FETCH, S_LOAD, S_GO, S_RUN} st_e;
  st_e         st;
  logic [3:0]  op, nops, fet_op;
  logic [2:0]  fet_w;
  logic        fet_on, nr_v, op_go, op_done;
  assign busy_o = (st != S_IDLE);
  assign op_go  = (st == S_GO);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st <= S_IDLE; op <= '0; nops <= '0; fet_op <= '0; fet_w <= '0; fet_on <= 1'b0; nr_v <= 1'b0; done_o <= 1'b0;
      for (int i = 0; i < 8; i++) begin dr[i] <= '0; nr[i] <= '0; end
    end else begin
      done_o <= 1'b0;
      if (fet_on) begin
        nr[fet_w] <= dsc_m[{fet_op, fet_w}];
        fet_w     <= fet_w + 3'd1;
        if (fet_w == 3'd7) begin fet_on <= 1'b0; nr_v <= 1'b1; end
      end
      unique case (st)
        S_IDLE: if (start_i) begin
          nops <= nops_i; op <= '0; fet_op <= '0; fet_w <= '0; fet_on <= 1'b1; nr_v <= 1'b0; st <= S_FETCH;
        end
        S_FETCH: if (nr_v && !fet_on) st <= S_LOAD;
        S_LOAD: begin                                 // nr -> dr, and fetch the next operation's record behind the running one
          for (int i = 0; i < 8; i++) dr[i] <= nr[i];
          nr_v <= 1'b0;
          if (op + 4'd1 < nops) begin fet_op <= op + 4'd1; fet_w <= '0; fet_on <= 1'b1; end
          st <= S_GO;
        end
        S_GO: st <= S_RUN;
        S_RUN: if (op_done) begin
          if (op + 4'd1 >= nops) begin st <= S_IDLE; done_o <= 1'b1; end
          else begin op <= op + 4'd1; st <= S_FETCH; end
        end
        default: st <= S_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------------ activation and image memories
  logic        ar_re [3];
  logic [10:0] ar_a  [3];
  logic [63:0] ar_q  [3];
  logic        aw_we;
  logic [1:0]  aw_b;
  logic [10:0] aw_a;
  logic [63:0] aw_d;
  for (genvar b = 0; b < 3; b++) begin : g_act
    (* ram_style = "block" *) logic [63:0] m [2048];
    always_ff @(posedge clk_i) begin
      if (aw_we && (aw_b == 2'(b))) m[aw_a] <= aw_d;
      if (ar_re[b]) ar_q[b] <= m[ar_a[b]];
    end
  end
  (* ram_style = "block" *) logic [23:0] img_m [1024];
  logic [9:0]  img_ra;
  logic [23:0] img_q;
  always_ff @(posedge clk_i) begin
    if (img_we_i) img_m[img_addr_i] <= img_data_i;
    img_q <= img_m[img_ra];
  end
  logic signed [7:0] pout [64];                        // pool results
  logic [7:0]  logits [10];
  logic signed [7:0] logit_best;
  assign logit_o = logits[logit_addr_i];

  // ------------------------------------------------------------------ shapes of the running operation
  logic        is_fc, is_conv, is_img;
  logic [4:0]  h_in, out_m1;                           // input map size (8, 16, 32 -> 5 bits hold 32 - 0)
  logic [5:0]  h6;                                     // 8, 16 or 32
  logic [3:0]  hl;                                     // log2 of the input map size
  logic [1:0]  c_gl;                                   // log2 (Cin / 8), Cin = 16, 32, 64
  logic [3:0]  c_nt;                                   // taps per row tile
  logic [2:0]  c_wm1;                                  // ACT words per tap - 1
  logic [3:0]  c_nk;                                   // taps in this row tile
  logic [3:0]  ntaps;
  logic [2:0]  o_gl;                                   // log2 (Cout / 8): 1, 2, 3
  assign is_fc   = (o_type == OP_FC);
  assign is_conv = (o_type == OP_CONV);
  assign is_img  = is_conv && (o_cin == 2'd0);
  assign hl      = is_fc ? 4'd0 : 4'd3 + {2'd0, o_hl};
  assign h6      = 6'd1 << hl;
  assign out_m1  = is_fc ? 5'd0 : 5'((h6 >> o_s2) - 6'd1);
  assign ntaps   = o_k3 ? 4'd9 : 4'd1;
  assign o_gl    = (o_coutm1 == 6'd15) ? 3'd1 : (o_coutm1 == 6'd31) ? 3'd2 : 3'd3;
  always_comb begin
    unique case (o_cin)
      2'd0:    begin c_gl = 2'd0; c_nt = 4'd9; c_wm1 = 3'd0; end   // image: 9 taps, 3 bytes each
      2'd1:    begin c_gl = 2'd1; c_nt = 4'd4; c_wm1 = 3'd1; end
      2'd2:    begin c_gl = 2'd2; c_nt = 4'd2; c_wm1 = 3'd3; end
      default: begin c_gl = 2'd3; c_nt = 4'd1; c_wm1 = 3'd7; end
    endcase
    h_in = 5'(h6 - 6'd1);
  end

  // ------------------------------------------------------------------ gather
  logic        g_on, g_vec, g_bank;                    // more vectors to gather; a vector is in progress; its bank
  logic [4:0]  g_oy, g_ox;
  logic [2:0]  g_c;
  logic [3:0]  g_r, g_k;                               // row tile, tap of the row tile
  logic [2:0]  g_w;                                    // word of the tap
  logic [3:0]  g_tap;
  logic [1:0]  g_fy, g_fx;
  logic signed [6:0] g_iy, g_ix;
  logic        g_pad, g_adv, g_lastw, g_lastpos;
  logic        fr;                                     // first word of a vector this cycle
  logic [1:0]  occ, full;
  logic [10:0] g_wa;

  assign g_tap    = is_img ? g_k : 4'(({2'd0, g_r} << (o_cin == 2'd1 ? 2 : (o_cin == 2'd2 ? 1 : 0))) + {2'd0, g_k});
  assign g_fy     = 2'(g_tap / 4'd3);
  assign g_fx     = 2'(g_tap % 4'd3);
  // rows tile r holds min(nt, ntaps - r nt) taps; fc and K = 1: one tap
  assign c_nk     = (4'(g_r * c_nt) + c_nt > ntaps) ? (ntaps - 4'(g_r * c_nt)) : c_nt;
  assign g_lastw  = (g_w == c_wm1) && (g_k == ((is_img) ? 4'd8 : (c_nk - 4'd1)));
  assign g_lastpos = (g_ox == out_m1) && (g_oy == out_m1);
  logic pad1;                                          // SAME padding of 1 on the top and left for 3 x 3 stride 1
  assign pad1     = o_k3 && !o_s2;
  always_comb begin
    g_iy = signed'({2'd0, g_oy}) * (o_s2 ? 7'sd2 : 7'sd1) + signed'({5'd0, g_fy}) - signed'({6'd0, pad1});
    g_ix = signed'({2'd0, g_ox}) * (o_s2 ? 7'sd2 : 7'sd1) + signed'({5'd0, g_fx}) - signed'({6'd0, pad1});
    if (is_fc) g_pad = 1'b0;
    else g_pad = (g_iy < 7'sd0) || (g_ix < 7'sd0) || (g_iy > signed'({2'b00, h_in})) || (g_ix > signed'({2'b00, h_in}));
  end
  assign g_wa   = 11'(((11'(g_iy[4:0]) << hl) + 11'(g_ix[4:0])) << c_gl) + 11'(g_w);
  assign img_ra = {g_iy[4:0], g_ix[4:0]};
  assign g_adv  = g_on && (g_vec || !occ[g_bank]);
  assign fr     = g_adv && !g_vec;

  // read ports: gather (one bank), add (two banks), pool (one bank), debug
  logic        p_re, a_re;
  logic [10:0] p_wa, a_wa;
  logic        p_on, a_on;
  always_comb begin
    for (int b = 0; b < 3; b++) begin ar_re[b] = 1'b0; ar_a[b] = '0; end
    if (st == S_IDLE) begin
      if (dbg_re_i && (dbg_addr_i[13:12] != 2'd3)) begin ar_re[dbg_addr_i[13:12]] = 1'b1; ar_a[dbg_addr_i[13:12]] = dbg_addr_i[11:1]; end
    end else if (is_conv || is_fc) begin
      if (g_adv && !is_img && !is_fc && !g_pad && (o_src != 2'd3)) begin ar_re[o_src] = 1'b1; ar_a[o_src] = g_wa; end
    end else if (p_on) begin
      if (p_re) begin ar_re[o_src] = 1'b1; ar_a[o_src] = p_wa; end
    end else if (a_on) begin
      if (a_re) begin
        ar_re[o_src]  = 1'b1; ar_a[o_src]  = a_wa;
        ar_re[o_srcb] = 1'b1; ar_a[o_srcb] = a_wa;
      end
    end
  end
  logic       dbg_h_q;
  logic [1:0] dbg_b_q;
  always_ff @(posedge clk_i) begin dbg_h_q <= dbg_addr_i[0]; dbg_b_q <= dbg_addr_i[13:12]; end
  assign dbg_rdata_o = dbg_h_q ? ar_q[dbg_b_q][63:32] : ar_q[dbg_b_q][31:0];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      g_on <= 1'b0; g_vec <= 1'b0; g_bank <= 1'b0; g_oy <= '0; g_ox <= '0; g_c <= '0; g_r <= '0; g_k <= '0; g_w <= '0;
    end else if (op_go) begin
      g_on <= is_conv || is_fc;
      g_vec <= 1'b0; g_bank <= 1'b0; g_oy <= '0; g_ox <= '0; g_c <= '0; g_r <= '0; g_k <= '0; g_w <= '0;
    end else if (g_adv) begin
      if (!g_lastw) begin
        g_vec <= 1'b1;
        if (g_w == c_wm1) begin g_w <= '0; g_k <= g_k + 4'd1; end
        else g_w <= g_w + 3'd1;
      end else begin                                   // vector finished: next row tile, column tile, position
        g_vec <= 1'b0; g_w <= '0; g_k <= '0; g_bank <= ~g_bank;
        if (g_r + 4'd1 != o_R) g_r <= g_r + 4'd1;
        else begin
          g_r <= '0;
          if (g_c + 3'd1 != o_C) g_c <= g_c + 3'd1;
          else begin
            g_c <= '0;
            if (g_lastpos) g_on <= 1'b0;
            else if (g_ox == out_m1) begin g_ox <= '0; g_oy <= g_oy + 5'd1; end
            else g_ox <= g_ox + 5'd1;
          end
        end
      end
    end
  end

  // stage 1 (memory output valid)
  logic        s1_v, s1_last, s1_bank, s1_pad;
  logic [5:0]  s1_row;
  logic [2:0]  s1_fw;
  logic        xw_last;                                // the write in the output registers is the vector's last word
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin s1_v <= 1'b0; s1_last <= 1'b0; s1_bank <= 1'b0; s1_pad <= 1'b0; s1_row <= '0; s1_fw <= '0; end
    else begin
      s1_v    <= g_adv;
      s1_last <= g_adv && g_lastw;
      s1_bank <= g_bank;
      s1_pad  <= g_pad;
      s1_fw   <= g_w;
      s1_row  <= is_img ? 6'(6'd3 * g_k) : 6'(((6'(g_k)) << (6'(c_gl) + 6'd3)) + 6'({g_w, 3'b000}));
    end
  end
  // stage 2: x = v - zin (9 bits); the write is registered, so the engine sees it in the next cycle
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      xw_we_o <= 1'b0; xw_bank_o <= 1'b0; xw_row_o <= '0; xw_mask_o <= '0; xw_last <= 1'b0;
      for (int l = 0; l < 8; l++) xw_data_o[l] <= '0;
    end else begin
      xw_we_o   <= s1_v;
      xw_bank_o <= s1_bank;
      xw_row_o  <= s1_row;
      xw_last   <= s1_last;
      for (int l = 0; l < 8; l++) xw_data_o[l] <= '0;
      if (is_img) begin                                // 3 rows: row, row + 1, row + 2 in lanes (row + i) mod 8
        xw_mask_o <= '0;
        for (int i = 0; i < 3; i++) begin
          logic [2:0] rr;
          rr = s1_row[2:0] + 3'(i);
          xw_mask_o[rr] <= 1'b1;
          xw_data_o[rr] <= s1_pad ? XB'(0) : XB'(9'(signed'(img_q[8*i +: 8])) - 9'(signed'(o_za)));
        end
      end else begin
        xw_mask_o <= 8'hFF;
        for (int l = 0; l < 8; l++) begin
          logic signed [7:0] v;
          v = is_fc ? pout[{s1_fw, 3'(l)}] : signed'(ar_q[o_src][8*l +: 8]);
          xw_data_o[l] <= s1_pad ? XB'(0) : XB'(9'(v) - 9'(signed'(o_za)));
        end
      end
    end
  end
  // a bank is complete when its last word has been written (one cycle after the write is presented)
  logic pass_go, i_bank, i_on;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin occ <= 2'b00; full <= 2'b00; end
    else if (op_go) begin occ <= 2'b00; full <= 2'b00; end
    else begin
      for (int b = 0; b < 2; b++) begin
        // occupied from the first word of a vector until the pass that uses it starts
        if (pass_go && (i_bank == 1'(b))) occ[b] <= fr && (g_bank == 1'(b));
        else if (fr && (g_bank == 1'(b))) occ[b] <= 1'b1;
        if (pass_go && (i_bank == 1'(b))) full[b] <= 1'b0;
        else if (xw_we_o && xw_last && (xw_bank_o == 1'(b))) full[b] <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------ issue
  logic [3:0]  i_r;
  logic [2:0]  i_c;
  logic [4:0]  i_oy, i_ox;
  assign pass_go = i_on && full[i_bank] && eng_ready_i;
  assign go_o    = pass_go;
  assign accum_o = (i_r != 4'd0);
  assign xbank_o = i_bank;
  assign tile_o  = TW'({4'd0, o_tbase} + 11'(i_r) * 11'(o_C) + 11'(i_c));
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin i_on <= 1'b0; i_bank <= 1'b0; i_r <= '0; i_c <= '0; i_oy <= '0; i_ox <= '0; end
    else if (op_go) begin i_on <= is_conv || is_fc; i_bank <= 1'b0; i_r <= '0; i_c <= '0; i_oy <= '0; i_ox <= '0; end
    else if (pass_go) begin
      i_bank <= ~i_bank;
      if (i_r + 4'd1 != o_R) i_r <= i_r + 4'd1;
      else begin
        i_r <= '0;
        if (i_c + 3'd1 != o_C) i_c <= i_c + 3'd1;
        else begin
          i_c <= '0;
          if (i_ox == out_m1) begin
            i_ox <= '0;
            if (i_oy == out_m1) i_on <= 1'b0; else i_oy <= i_oy + 5'd1;
          end else i_ox <= i_ox + 5'd1;
        end
      end
    end
  end

  // ------------------------------------------------------------------ results: the last pass of each chain
  logic signed [23:0] lat [16];
  logic        lat_v, rq_take;
  logic [9:0]  lat_pos;                                // output position, raster
  logic [2:0]  lat_c, d_c;
  logic [3:0]  d_r;
  logic [4:0]  d_oy, d_ox;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      d_oy <= '0; d_ox <= '0; d_c <= '0; d_r <= '0; lat_v <= 1'b0; lat_pos <= '0; lat_c <= '0; err_o <= 1'b0;
      for (int n = 0; n < 16; n++) lat[n] <= '0;
    end else begin
      if (start_i && (st == S_IDLE)) err_o <= 1'b0;
      if (op_go) begin d_oy <= '0; d_ox <= '0; d_c <= '0; d_r <= '0; lat_v <= 1'b0; end
      else begin
        if (rq_take) lat_v <= 1'b0;
        if (eng_done_i && (st == S_RUN) && (is_conv || is_fc)) begin
          if (d_r + 4'd1 != o_R) d_r <= d_r + 4'd1;
          else begin
            d_r <= '0;
            if (lat_v && !rq_take) err_o <= 1'b1;        // the previous result was not taken: lost
            for (int n = 0; n < 16; n++) lat[n] <= res_i[n][23:0];
            lat_v   <= 1'b1;
            lat_pos <= 10'(10'(d_oy) * 10'({1'b0, out_m1} + 6'd1)) + 10'(d_ox);
            lat_c   <= d_c;
            if (d_c + 3'd1 != o_C) d_c <= d_c + 3'd1;
            else begin
              d_c <= '0;
              if (d_ox == out_m1) begin d_ox <= '0; d_oy <= d_oy + 5'd1; end
              else d_ox <= d_ox + 5'd1;
            end
          end
        end
      end
    end
  end

  // ------------------------------------------------------------------ requantisation: 2 lanes
  // one pair per cycle: q pair and parameter address, q+1 memory output and accumulator, q+2 acc + bias, q+3 x qm, q+4 srdhm,
  // q+5 rdbp + zout + clamp (result)
  logic        rq_on;
  logic [2:0]  rq_i, rq_npm1;
  logic [9:0]  rq_pos;
  logic [2:0]  rq_c;
  logic signed [23:0] rlat [16];
  logic [5:1]  rq_ch;
  logic [8:0]  pa;
  assign rq_npm1 = is_fc ? 3'd4 : 3'd7;
  assign rq_take = lat_v && (!rq_on || (rq_i == rq_npm1));
  assign rq_ch   = 5'({rq_c, 3'd0}) + 5'(rq_i);          // channel / 2 of the pair

  assign pa      = {o_lpar, rq_ch};
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_on <= 1'b0; rq_i <= '0; rq_pos <= '0; rq_c <= '0;
      for (int n = 0; n < 16; n++) rlat[n] <= '0;
    end else if (op_go) rq_on <= 1'b0;
    else if (rq_take) begin
      for (int n = 0; n < 16; n++) rlat[n] <= lat[n];
      rq_on <= 1'b1; rq_i <= '0; rq_pos <= lat_pos; rq_c <= lat_c;
    end else if (rq_on) begin
      if (rq_i == rq_npm1) rq_on <= 1'b0; else rq_i <= rq_i + 3'd1;
    end
  end

  logic [23:0] pb_q [2];
  logic [30:0] pq_q [2];
  logic [3:0]  pe_q [2];
  for (genvar b = 0; b < 2; b++) begin : g_prm
    (* ram_style = "block" *) logic [23:0] bias_m [512];
    (* ram_style = "block" *) logic [30:0] qm_m   [512];
    (* ram_style = "block" *) logic [3:0]  e_m    [512];
    logic we;
    assign we = prm_we_i && (prm_addr_i[0] == 1'(b));
    always_ff @(posedge clk_i) begin
      if (we && (prm_field_i == 2'd0)) bias_m[prm_addr_i[9:1]] <= prm_data_i[23:0];
      if (we && (prm_field_i == 2'd1)) qm_m[prm_addr_i[9:1]]   <= prm_data_i[30:0];
      if (we && (prm_field_i == 2'd2)) e_m[prm_addr_i[9:1]]    <= prm_data_i[3:0];
      pb_q[b] <= bias_m[pa];
      pq_q[b] <= qm_m[pa];
      pe_q[b] <= e_m[pa];
    end
  end

  typedef struct packed {
    logic       v;
    logic [9:0] pos;
    logic [2:0] c;
    logic [2:0] i;
  } tag_t;
  tag_t t1, t2, t3, t4, t5;
  logic signed [23:0] lx1 [2];
  logic signed [24:0] s2 [2];
  logic [30:0] q2 [2];
  logic [3:0]  e2 [2], e3 [2], e4 [2];
  logic signed [55:0] p3 [2];
  logic signed [31:0] h4 [2];
  logic [7:0]  y5 [2];

  function automatic logic signed [31:0] rdbp(input logic signed [31:0] r, input logic [4:0] e);
    logic signed [32:0] n;
    n = 33'(r) + ((e == 5'd0) ? 33'sd0 : (33'sd1 <<< (e - 5'd1))) - (((r < 0) && (e != 5'd0)) ? 33'sd1 : 33'sd0);
    return (e == 5'd0) ? r : 32'(n >>> e);
  endfunction
  function automatic logic [7:0] clamp8(input logic signed [31:0] v, input logic signed [7:0] lo);
    if (v < 32'(lo)) return lo;
    if (v > 32'sd127) return 8'sd127;
    return v[7:0];
  endfunction
  logic signed [7:0] o_lo;
  assign o_lo = o_relu ? signed'(o_zo) : -8'sd128;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin t1 <= '0; t2 <= '0; t3 <= '0; t4 <= '0; t5 <= '0; end
    else begin t1 <= '{v: rq_on, pos: rq_pos, c: rq_c, i: rq_i}; t2 <= t1; t3 <= t2; t4 <= t3; t5 <= t4; end
  end
  always_ff @(posedge clk_i) begin
    for (int b = 0; b < 2; b++) begin
      lx1[b] <= rlat[{rq_i, 1'(b)}];                    // valid with pb_q (q + 1)
      s2[b]  <= 25'(lx1[b]) + 25'(signed'(pb_q[b]));
      q2[b]  <= pq_q[b];
      e2[b]  <= pe_q[b];
      (* use_dsp = "yes" *) p3[b] <= 56'(s2[b]) * 56'(signed'({1'b0, q2[b]}));
      e3[b]  <= e2[b];
      h4[b]  <= 32'((p3[b] + 56'sd1073741824) >>> 31);
      e4[b]  <= e3[b];
      y5[b]  <= clamp8(rdbp(h4[b], {1'b0, e4[b]}) + 32'(signed'(o_zo)), o_lo);
    end
  end

  // ------------------------------------------------------------------ write-back of conv / fc results
  logic [47:0] wbuf;
  logic [63:0] wb_d;
  logic [10:0] wb_a;
  logic        wb_v;
  logic [12:0] res_left;                               // chain results still to be written (conv) or 1 (fc)
  logic [10:0] ob;
  logic        o_last;
  assign ob     = 11'((11'(t5.pos) << o_gl) + 11'({t5.c, 1'b0}) + 11'(t5.i[2]));
  assign o_last = t5.v && (t5.i == rq_npm1);
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin wbuf <= '0; wb_d <= '0; wb_a <= '0; wb_v <= 1'b0; end
    else begin
      wb_v <= 1'b0;
      if (t5.v && !is_fc) begin
        if (t5.i[1:0] != 2'd3)
          for (int b = 0; b < 2; b++) wbuf[8*({t5.i[1:0], 1'(b)}) +: 8] <= y5[b];
        if (t5.i[1:0] == 2'd3) begin
          wb_v <= 1'b1;
          wb_a <= ob;
          wb_d <= {y5[1], y5[0], wbuf[47:0]};
        end
      end
    end
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int n = 0; n < 10; n++) logits[n] <= '0;
      class_o <= '0;
    end else if (is_fc && t5.v) begin
      logic signed [7:0] best;
      logic [3:0] bi;
      for (int b = 0; b < 2; b++) logits[{t5.i[2:0], 1'(b)} & 4'hF] <= y5[b];
      best = (t5.i == 3'd0) ? -8'sd128 : logit_best;
      bi   = (t5.i == 3'd0) ? 4'd0 : class_o;
      if (t5.i == 3'd0) begin
        best = signed'(y5[0]); bi = 4'd0;
        if (signed'(y5[1]) > best) begin best = signed'(y5[1]); bi = 4'd1; end
      end else for (int b = 0; b < 2; b++)
        if (signed'(y5[b]) > best) begin best = signed'(y5[b]); bi = {t5.i, 1'(b)}; end
      logit_best <= best;
      class_o    <= bi;
    end
  end
  // ------------------------------------------------------------------ add: two elements per cycle
  logic        a_run;
  logic [10:0] a_w, a_nwm1;
  logic [1:0]  a_sub;
  assign a_on  = a_run && (o_type == OP_ADD);
  assign a_nwm1 = 11'((12'd1 << (2 * hl + 4'(o_gl))) - 12'd1);   // words per map - 1: H^2 x C / 8
  assign a_re  = a_on && (a_sub == 2'd0);
  assign a_wa  = a_w;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin a_run <= 1'b0; a_w <= '0; a_sub <= '0; end
    else if (op_go) begin a_run <= (o_type == OP_ADD); a_w <= '0; a_sub <= '0; end
    else if (a_on) begin
      a_sub <= a_sub + 2'd1;
      if (a_sub == 2'd3) begin
        if (a_w == a_nwm1) a_run <= 1'b0;
        a_w <= a_w + 11'd1;
      end
    end
  end
  // the word is read at sub 0 and its 4 pairs are processed in the next 4 cycles
  logic        b1_v;
  logic [1:0]  b1_sub;
  logic [10:0] b1_w;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin b1_v <= 1'b0; b1_sub <= '0; b1_w <= '0; end
    else begin b1_v <= a_on; b1_sub <= a_sub; b1_w <= a_w; end
  end
  logic        av [9];
  logic [1:0]  asub [9];
  logic [10:0] aw_w [9];
  logic signed [8:0]  da [2], db [2];
  logic signed [40:0] ma [2], mb [2];
  logic signed [31:0] ha [2], hb [2], ra [2], rb [2], sm [2];
  logic signed [62:0] m2 [2];
  logic signed [31:0] h2 [2];
  logic [7:0]         ya [2];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < 9; i++) begin av[i] <= 1'b0; asub[i] <= '0; aw_w[i] <= '0; end
    end else begin
      av[0] <= b1_v; asub[0] <= b1_sub; aw_w[0] <= b1_w;
      for (int i = 1; i < 9; i++) begin av[i] <= av[i-1]; asub[i] <= asub[i-1]; aw_w[i] <= aw_w[i-1]; end
    end
  end
  always_ff @(posedge clk_i) begin
    for (int k = 0; k < 2; k++) begin
      logic signed [7:0] xa, xb;
      xa = signed'(ar_q[o_src][8*({b1_sub, 1'(k)}) +: 8]);
      xb = signed'(ar_q[o_srcb][8*({b1_sub, 1'(k)}) +: 8]);
      da[k] <= 9'(xa) - 9'(signed'(o_za));
      db[k] <= 9'(xb) - 9'(signed'(o_zb));
      ma[k] <= 41'(da[k]) * 41'(signed'({1'b0, o_qa}));
      mb[k] <= 41'(db[k]) * 41'(signed'({1'b0, o_qb}));
      ha[k] <= 32'((ma[k] + 41'sd1024) >>> 11);          // srdhm((x - z) << 20, q)
      hb[k] <= 32'((mb[k] + 41'sd1024) >>> 11);
      ra[k] <= rdbp(ha[k], o_ea);
      rb[k] <= rdbp(hb[k], o_eb);
      sm[k] <= ra[k] + rb[k];
      m2[k] <= 63'(sm[k]) * 63'(signed'({1'b0, o_qo}));
      h2[k] <= 32'((m2[k] + 63'sd1073741824) >>> 31);
      ya[k] <= clamp8(rdbp(h2[k], o_eo) + 32'(signed'(o_zo)), o_lo);
    end
  end
  // y of the pair is valid 8 stages after the read; av index: ar_q at b1, da 0, ma 1, ha 2, ra 3, sm 4, m2 5, h2 6, ya 7
  logic [47:0] a_buf;
  logic        aw_v;
  logic [10:0] aw_wa;
  logic [63:0] aw_wd;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin a_buf <= '0; aw_v <= 1'b0; aw_wa <= '0; aw_wd <= '0; end
    else begin
      aw_v <= 1'b0;
      if (av[7]) begin
        if (asub[7] != 2'd3)
          for (int k = 0; k < 2; k++) a_buf[8*({asub[7], 1'(k)}) +: 8] <= ya[k];
        if (asub[7] == 2'd3) begin
          aw_v <= 1'b1; aw_wa <= aw_w[7]; aw_wd <= {ya[1], ya[0], a_buf[47:0]};
        end
      end
    end
  end

  // ------------------------------------------------------------------ pool: sum of 64 positions, 8 channels per word
  logic        p_run;
  logic [9:0]  p_w;                                    // word counter (512 words for 8 x 8 x 64)
  logic signed [14:0] psum [64];
  assign p_on = p_run && (o_type == OP_POOL);
  assign p_re = p_on && (p_w < 10'd512);
  assign p_wa = 11'(p_w);
  logic [3:0]  p_st;                                   // 0 reading / draining, 1 convert
  logic [2:0]  p_c;
  logic        p_v1;
  logic [2:0]  p_w1;
  logic        p_done;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      p_run <= 1'b0; p_w <= '0; p_v1 <= 1'b0; p_w1 <= '0; p_c <= '0; p_st <= '0; p_done <= 1'b0;
      for (int n = 0; n < 64; n++) begin psum[n] <= '0; pout[n] <= '0; end
    end else begin
      p_done <= 1'b0;
      if (op_go) begin
        p_run <= (o_type == OP_POOL); p_w <= '0; p_st <= '0; p_c <= '0; p_v1 <= 1'b0;
        for (int n = 0; n < 64; n++) psum[n] <= '0;
      end else if (p_run) begin
        p_v1 <= p_re;
        p_w1 <= p_w[2:0];
        if (p_re) p_w <= p_w + 10'd1;
        if (p_v1) for (int l = 0; l < 8; l++) psum[{p_w1, 3'(l)}] <= psum[{p_w1, 3'(l)}] + 15'(signed'(ar_q[o_src][8*l +: 8]));
        if (p_st == 4'd0) begin
          if ((p_w == 10'd512) && !p_v1 && !p_re) begin p_st <= 4'd1; p_c <= '0; end
        end else begin
          for (int l = 0; l < 8; l++) pout[{p_c, 3'(l)}] <= (psum[{p_c, 3'(l)}] >= 0) ?
              8'((32'(psum[{p_c, 3'(l)}]) + 32'sd32) >>> 6) : 8'(-((-32'(psum[{p_c, 3'(l)}]) + 32'sd32) >>> 6));
          p_c <= p_c + 3'd1;
          if (p_c == 3'd7) begin p_run <= 1'b0; p_done <= 1'b1; end
        end
      end
    end
  end

  // ------------------------------------------------------------------ ACT write port, operation end
  always_comb begin
    aw_we = 1'b0; aw_b = o_dst; aw_a = wb_a; aw_d = wb_d;
    if (wb_v) begin aw_we = 1'b1; aw_b = o_dst; aw_a = wb_a; aw_d = wb_d; end
    else if (aw_v) begin aw_we = 1'b1; aw_b = o_dst; aw_a = aw_wa; aw_d = aw_wd; end
  end
  logic [12:0] res_tot;
  assign res_tot = (13'(out_m1) + 13'd1) * (13'(out_m1) + 13'd1) * 13'(o_C);
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) res_left <= '0;
    else if (op_go) res_left <= res_tot;
    else if (o_last && !is_fc) res_left <= res_left - 13'd1;
  end
  // the operation ends when the last result's last word has been written
  logic conv_end;
  logic o_last_q, o_last_q2;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin o_last_q <= 1'b0; o_last_q2 <= 1'b0; conv_end <= 1'b0; end
    else begin
      o_last_q  <= o_last && (is_fc || (res_left == 13'd1));
      o_last_q2 <= o_last_q;
      conv_end  <= o_last_q2;
    end
  end
  logic add_end;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin add_end <= 1'b0; end
    else begin
      add_end <= aw_v && (aw_wa == a_nwm1) && (o_type == OP_ADD);
    end
  end
  assign op_done = ((is_conv || is_fc) && conv_end) || ((o_type == OP_ADD) && add_end) || ((o_type == OP_POOL) && p_done);

endmodule
