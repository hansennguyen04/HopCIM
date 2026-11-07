// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_fe: hardware MFCC front-end, frame path (R5a). For one frame of 480 samples read from the
// sample ring it computes lraw[40] of the integer CIM-MFCC bit-exactly (model/mfcc_int.py:
// frame_lraw; docs/MFCC_INT_SPEC.md). The engine runs the three matrix steps on the front-end
// tiles (slots 0..5, layout model/cim_map.py); this unit does the window, block exponent,
// twiddles, magnitude and log, and moves data between its buffers and the engine.
//
// Schedule (each pass is gathered, run and scattered before the next one starts):
//   PEAK  480 window products -> peak -> block exponent sh (peak 0: every lraw is ZERO)
//   FFT1  32 x { x[r] = rnd(s[32r + n2] * hann, sh), r < 15 (x[15] = 0); pass F1;
//                mirror (negated before rounding), twiddle -> B[k1][n2] }
//   FFT3  16 x { x = [Br | Bi][k1]; pass F3; >> 9 -> re/im of bins k1 + 16 k2, k2 < 8 }
//   MAG   128 x isqrt(re^2 + im^2), one result bit per cycle
//   MEL   MEL00 on bins 0..63; MEL01 (same x) then MEL11 on bins 64..127 (accumulate); MEL12;
//         each result set is copied, one per cycle, into a 40-entry LUT-RAM
//   LOG   40 x (leading one + 64-entry log2 table, Q8) * 256 + 65536 (sh + 5) - K16
// lraw of an exact zero (peak 0 or mel 0) reads as ZERO = 0x8000_0000.

module cim_fe #(
    parameter int unsigned TW = 6                    // tile index width of the engine
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,
    // control
    input  logic                   start_i,          // start one frame (ignored while busy)
    input  logic [9:0]             base_i,           // ring index of sample 0 of the frame
    output logic                   busy_o,
    output logic                   done_o,           // one-cycle pulse: lraw complete
    // sample ring read port (data one cycle after the address)
    output logic [9:0]             samp_addr_o,
    input  logic signed [15:0]     samp_i,
    // engine (driven by this unit while busy_o)
    output logic                   x_we_o,
    output logic [5:0]             x_idx_o,
    output logic [23:0]            x_data_o,
    output logic                   go_o,
    output logic [TW-1:0]          tile_o,
    output logic [4:0]             xbits_o,
    output logic                   accum_o,
    input  logic                   eng_ready_i,
    input  logic                   eng_done_i,
    input  logic signed [47:0]     res_i [16],
    // lraw read port (combinational)
    input  logic [5:0]             lraw_addr_i,
    output logic [31:0]            lraw_o
);

  `include "fe_tables.svh"

  localparam logic [31:0] ZERO = 32'h8000_0000;

  typedef enum logic [4:0] {
    S_IDLE, S_PEAK, S_EXP, S_G1, S_P1, S_W1, S_S1, S_G3, S_P3, S_W3, S_S3, S_MAG,
    S_GM, S_PM, S_WM, S_WR, S_LOG, S_DONE
  } state_t;

  state_t      state;
  logic [9:0]  base_q;
  logic [6:0]  cnt;                                  // element counter of the current phase
  logic [4:0]  n2;
  logic [3:0]  k1;
  logic [1:0]  mstep;                                // mel pass 0..3
  logic        silent;
  logic signed [5:0] sh_q;                           // block exponent, -14..15
  logic [29:0] peak;
  logic [1:0]  fine_q;                               // PEAK: sample = 4 cnt + fine_q; cnt 0x7F = end

  // ---------------------------------------------------------------- arithmetic helpers
  // (the temporaries are wider than the results; their top bits are dropped on purpose)
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic signed [23:0] bfp(input logic signed [30:0] v, input logic signed [5:0] sh);
    logic signed [31:0] t;
    if (sh > 0) t = (32'(v) + (32'sd1 <<< (sh - 6'sd1))) >>> sh;
    else t = 32'(v) <<< (-sh);
    return t[23:0];
  endfunction

  function automatic logic signed [15:0] rsh11(input logic signed [33:0] v);
    logic signed [33:0] t;
    t = (v + 34'sd1024) >>> 11;
    return t[15:0];
  endfunction

  function automatic logic signed [16:0] rsh14(input logic signed [32:0] v);
    logic signed [32:0] t;
    t = (v + 33'sd8192) >>> 14;
    return t[16:0];
  endfunction

  function automatic logic signed [20:0] rsh9(input logic signed [31:0] v);
    logic signed [31:0] t;
    t = (v + 32'sd256) >>> 9;
    return t[20:0];
  endfunction

  function automatic logic [13:0] log2_q8(input logic [39:0] v);   // v >= 1
    int unsigned e;
    logic [39:0] m;
    e = 0;
    for (int unsigned i = 0; i < 40; i++) if (v[i]) e = i;
    m = (e >= 6) ? (v >> (e - 6)) : (v << (6 - e));
    return 14'(e * 256) + 14'(FE_LOG_LUT[m[5:0]]);
  endfunction

  /* verilator lint_on UNUSEDSIGNAL */

  // ---------------------------------------------------------------- buffers
  logic [33:0] bbuf [512];                           // {Br, Bi} at {k1, n2}
  logic [41:0] ribuf [128];                          // {re, im} at bin
  logic [20:0] magbuf [128];
  logic [39:0] mel_ram [64];                         // 40 used; LUT-RAM (write port, async read)
  logic [5:0]  wr_base;                              // first mel band of the result set being copied
  logic [31:0] lraw_q [64];
  assign lraw_o = lraw_q[lraw_addr_i];

  // ---------------------------------------------------------------- sample x Hann pipeline
  // issue at c (ring and ROM addresses), data at c+1, product at c+2 (PEAK and FFT1 gather)
  logic        sh_issue, sh_zero;
  logic [8:0]  sh_idx;
  logic [5:0]  sh_tag;
  logic        sp1_v, sp1_z, sp2_v, sp2_z;
  logic [5:0]  sp1_tag, sp2_tag;
  logic [15:0] hann_q;
  logic signed [30:0] prod_q;
  logic [29:0] prod_abs;

  always_comb begin
    sh_issue = 1'b0;
    sh_zero  = 1'b0;
    sh_idx   = 9'd0;
    sh_tag   = 6'd0;
    if (state == S_PEAK && cnt[6:0] != 7'h7F) begin
      sh_issue = 1'b1;
      sh_idx   = 9'({cnt[6:0], 2'b00}) | 9'(fine_q);
    end else if (state == S_G1 && cnt < 7'd16) begin
      sh_issue = 1'b1;
      sh_zero  = (cnt == 7'd15);
      sh_idx   = sh_zero ? 9'd0 : 9'({cnt[3:0], n2});
      sh_tag   = 6'(cnt);
    end
  end

  assign samp_addr_o = base_q + 10'(sh_idx);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sp1_v <= 1'b0; sp1_z <= 1'b0; sp1_tag <= '0;
      sp2_v <= 1'b0; sp2_z <= 1'b0; sp2_tag <= '0;
    end else begin
      sp1_v <= sh_issue; sp1_z <= sh_zero; sp1_tag <= sh_tag;
      sp2_v <= sp1_v;    sp2_z <= sp1_z;   sp2_tag <= sp1_tag;
    end
  end

  always_ff @(posedge clk_i) begin
    hann_q <= FE_HANN[sh_idx];
    prod_q <= 31'(samp_i * $signed({1'b0, hann_q}));
  end
  assign prod_abs = prod_q[30] ? 30'(-prod_q) : 30'(prod_q);

  // ---------------------------------------------------------------- FFT1 scatter: mirror + twiddle
  // issue at c (k1 = cnt): A from res_i, twiddle ROM; c+1 registered; c+2 products; write at c+2
  logic        t1_v, t2_v;
  logic [3:0]  t1_k, t2_k;
  logic signed [15:0] ar_q, ai_q;
  logic [31:0] tw_q;
  logic signed [31:0] m_rr, m_ii, m_ri, m_ir;
  logic [3:0]  j_re, j_im;
  logic signed [15:0] ar_d, ai_d;
  logic        s1_issue;

  assign s1_issue = (state == S_S1) && (cnt < 7'd16);

  always_comb begin
    logic [3:0] k;
    k    = cnt[3:0];
    j_re = (k <= 4'd8) ? k : 4'd0 - k;               // 16 - k for k = 9..15
    j_im = (j_re == 4'd0 || j_re == 4'd8) ? 4'd9 : 4'd8 + j_re;
    ar_d = rsh11(34'($signed(res_i[j_re][31:0])));      // part-selects are unsigned: re-sign
    if (j_re == 4'd0 || j_re == 4'd8) ai_d = 16'sd0;
    else if (k < 4'd8) ai_d = rsh11(34'($signed(res_i[j_im][31:0])));
    else ai_d = rsh11(-34'($signed(res_i[j_im][31:0])));
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      t1_v <= 1'b0; t2_v <= 1'b0; t1_k <= '0; t2_k <= '0;
    end else begin
      t1_v <= s1_issue; t1_k <= cnt[3:0];
      t2_v <= t1_v;     t2_k <= t1_k;
    end
  end

  always_ff @(posedge clk_i) begin
    ar_q <= ar_d;
    ai_q <= ai_d;
    tw_q <= FE_TW[{cnt[3:0], n2}];
    m_rr <= ar_q * $signed(tw_q[31:16]);
    m_ii <= ai_q * $signed(tw_q[15:0]);
    m_ri <= ar_q * $signed(tw_q[15:0]);
    m_ir <= ai_q * $signed(tw_q[31:16]);
    if (t2_v) bbuf[{t2_k, n2}] <= {rsh14(33'(m_rr) - 33'(m_ii)), rsh14(33'(m_ri) + 33'(m_ir))};
  end

  // ---------------------------------------------------------------- buffer reads for gathers
  logic [33:0] bq;                                   // FFT3 gather: B read
  logic [20:0] mq;                                   // MEL gather: |X| read
  logic [41:0] riq;                                  // MAG: re/im read
  logic        g_v;                                  // a gather read is in flight (data at c+1)
  logic [5:0]  g_idx;

  always_ff @(posedge clk_i) begin
    bq  <= bbuf[{k1, cnt[4:0]}];
    mq  <= magbuf[{mstep[1], cnt[5:0]}];
    riq <= ribuf[cnt];
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      g_v <= 1'b0; g_idx <= '0;
    end else begin
      g_v   <= ((state == S_G3) || (state == S_GM)) && (cnt < 7'd64);
      g_idx <= cnt[5:0];
    end
  end

  // ---------------------------------------------------------------- engine X writes and passes
  always_comb begin
    x_we_o   = 1'b0;
    x_idx_o  = '0;
    x_data_o = '0;
    if (state == S_G1 && sp2_v) begin
      x_we_o   = 1'b1;
      x_idx_o  = sp2_tag;
      x_data_o = sp2_z ? 24'd0 : bfp(prod_q, sh_q);
    end else if (state == S_G3 && g_v) begin      // x[n2] = Br, x[32 + n2] = Bi
      x_we_o   = 1'b1;
      x_idx_o  = g_idx;
      x_data_o = g_idx[5] ? 24'($signed(bq[16:0])) : 24'($signed(bq[33:17]));
    end else if (state == S_GM && g_v) begin
      x_we_o   = 1'b1;
      x_idx_o  = g_idx;
      x_data_o = 24'(mq);
    end
  end

  always_comb begin
    go_o    = 1'b0;
    tile_o  = '0;
    xbits_o = 5'd8;
    accum_o = 1'b0;
    unique case (state)
      S_P1: begin go_o = 1'b1; tile_o = TW'(FE_T_F1); xbits_o = 5'(FE_B_F1); end
      S_P3: begin go_o = 1'b1; tile_o = TW'(FE_T_F3); xbits_o = 5'(FE_B_F3); end
      S_PM: begin
        go_o    = 1'b1;
        xbits_o = 5'(FE_B_MEL);
        accum_o = (mstep == 2'd2);
        unique case (mstep)
          2'd0: tile_o = TW'(FE_T_MEL00);
          2'd1: tile_o = TW'(FE_T_MEL01);
          2'd2: tile_o = TW'(FE_T_MEL11);
          default: tile_o = TW'(FE_T_MEL12);
        endcase
      end
      default: ;
    endcase
  end

  // ---------------------------------------------------------------- magnitude (sequential isqrt)
  logic [5:0]  mph;                                  // 0 read, 1 squares, 2 sum, 3..23 iterate, 24 write
  logic signed [41:0] sq_re, sq_im;
  logic [42:0] rem, root, bitm;

  // ---------------------------------------------------------------- buffer writes
  // (kept out of the reset block so that the buffers map to RAM rather than flip-flops)
  logic [31:0] lraw_d;
  logic [39:0] mel_rd;
  assign mel_rd = mel_ram[cnt[5:0]];
  always_comb begin
    if (silent || mel_rd == '0) lraw_d = ZERO;
    else lraw_d = 32'(log2_q8(mel_rd)) * 32'd256 + 32'(65536 * (int'(sh_q) + 5)) - 32'(FE_K16);
  end

  always_ff @(posedge clk_i) begin
    if (state == S_S3) ribuf[{cnt[2:0], k1}] <= {rsh9(res_i[{1'b0, cnt[2:0]}][31:0]), rsh9(res_i[{1'b1, cnt[2:0]}][31:0])};
    if (state == S_MAG && mph == 6'd24) magbuf[cnt] <= root[20:0];
    if (state == S_LOG) lraw_q[cnt[5:0]] <= lraw_d;
    if (state == S_WR) mel_ram[wr_base + cnt[5:0]] <= res_i[cnt[3:0]][39:0];   // res_i holds until the next pass
  end

  // ---------------------------------------------------------------- control
  assign busy_o = (state != S_IDLE);
  assign done_o = (state == S_DONE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state  <= S_IDLE;
      base_q <= '0;
      cnt    <= '0;
      fine_q <= '0;
      n2     <= '0;
      k1     <= '0;
      mstep  <= '0;
      silent <= 1'b0;
      sh_q   <= '0;
      peak   <= '0;
      mph    <= '0;
      wr_base <= '0;
      sq_re  <= '0;
      sq_im  <= '0;
      rem    <= '0;
      root   <= '0;
      bitm   <= '0;
    end else begin
      unique case (state)
        S_IDLE: if (start_i) begin
          state  <= S_PEAK;
          base_q <= base_i;
          cnt    <= '0;
          fine_q <= '0;
          peak   <= '0;
        end
        // 480 products: issue while cnt != 0x7F; finish when the pipeline is empty
        S_PEAK: begin
          if (cnt != 7'h7F) begin
            fine_q <= fine_q + 2'd1;
            if (fine_q == 2'd3) cnt <= (cnt == 7'd119) ? 7'h7F : cnt + 7'd1;
          end
          if (sp2_v && prod_abs > peak) peak <= prod_abs;
          if (cnt == 7'h7F && !sp1_v && !sp2_v) state <= S_EXP;
        end
        S_EXP: begin
          int unsigned e;
          e = 0;
          for (int unsigned i = 0; i < 30; i++) if (peak[i]) e = i;
          sh_q   <= 6'(signed'(e) - 14);
          silent <= (peak == '0);
          n2     <= '0;
          k1     <= '0;
          cnt    <= '0;
          state  <= (peak == '0) ? S_LOG : S_G1;
        end
        S_G1: begin
          if (cnt < 7'd16) cnt <= cnt + 7'd1;
          else if (!sp1_v && !sp2_v) state <= S_P1;
        end
        S_P1: if (eng_ready_i) state <= S_W1;
        S_W1: if (eng_done_i) begin
          state <= S_S1;
          cnt   <= '0;
        end
        S_S1: begin
          if (cnt < 7'd16) cnt <= cnt + 7'd1;
          else if (!t1_v && !t2_v) begin
            cnt <= '0;
            n2  <= n2 + 5'd1;
            state <= (n2 == 5'd31) ? S_G3 : S_G1;
          end
        end
        S_G3: begin
          if (cnt < 7'd64) cnt <= cnt + 7'd1;
          else if (!g_v) state <= S_P3;
        end
        S_P3: if (eng_ready_i) state <= S_W3;
        S_W3: if (eng_done_i) begin
          state <= S_S3;
          cnt   <= '0;
        end
        S_S3: begin
          if (cnt == 7'd7) begin
            cnt <= '0;
            k1  <= k1 + 4'd1;
            if (k1 == 4'd15) begin
              state <= S_MAG;
              mph   <= '0;
            end else state <= S_G3;
          end else cnt <= cnt + 7'd1;
        end
        S_MAG: begin
          unique case (mph)
            6'd0: mph <= 6'd1;                       // riq valid next cycle
            6'd1: begin
              sq_re <= $signed(riq[41:21]) * $signed(riq[41:21]);
              sq_im <= $signed(riq[20:0]) * $signed(riq[20:0]);
              mph   <= 6'd2;
            end
            6'd2: begin
              rem  <= 43'(sq_re) + 43'(sq_im);
              root <= '0;
              bitm <= 43'(1) << 40;
              mph  <= 6'd3;
            end
            6'd24: begin
              mph <= '0;
              if (cnt == 7'd127) begin
                state <= S_GM;
                cnt   <= '0;
                mstep <= '0;
              end else cnt <= cnt + 7'd1;
            end
            default: begin                           // 21 iterations, mph 3..23
              if (rem >= root + bitm) begin
                rem  <= rem - (root + bitm);
                root <= (root >> 1) + bitm;
              end else root <= root >> 1;
              bitm <= bitm >> 2;
              mph  <= mph + 6'd1;
            end
          endcase
        end
        S_GM: begin
          if (cnt < 7'd64) cnt <= cnt + 7'd1;
          else if (!g_v) state <= S_PM;
        end
        S_PM: if (eng_ready_i) state <= S_WM;
        S_WM: if (eng_done_i) begin
          cnt <= '0;
          if (mstep == 2'd1) begin                   // MEL01 result stays in the engine (accumulated)
            mstep <= 2'd2;
            state <= S_GM;                           // gather bins 64..127
          end else begin
            wr_base <= (mstep == 2'd0) ? 6'd0 : ((mstep == 2'd2) ? 6'd16 : 6'd32);
            state   <= S_WR;
          end
        end
        S_WR: begin                                  // copy 16 (8 for MEL12) results, one per cycle
          if (cnt == ((mstep == 2'd3) ? 7'd7 : 7'd15)) begin
            cnt   <= '0;
            mstep <= mstep + 2'd1;
            unique case (mstep)
              2'd0: state <= S_PM;                   // MEL01 on the same x
              2'd2: state <= S_PM;                   // MEL12 on the same x
              default: state <= S_LOG;
            endcase
          end else cnt <= cnt + 7'd1;
        end
        S_LOG: begin
          if (cnt == 7'd39) state <= S_DONE;
          else cnt <= cnt + 7'd1;
        end
        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
