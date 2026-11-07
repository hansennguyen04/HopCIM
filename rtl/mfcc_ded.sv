// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// mfcc_ded: baseline B3, a dedicated MFCC front-end (frame path) that does not use the CIM.
// Same contract as cim_fe (ring read port, lraw[40] out), bit-exact with model/mfcc_ded.py:
//   PEAK  480 window products -> peak -> block exponent sh (peak 0: every lraw is ZERO)
//   LOAD  512 x { x = rnd(s * hann, sh), 0 from sample 480 on } -> FFT buffer at bitrev(i)
//   FFT   9 stages x 256 radix-2 DIT butterflies, 2 cycles each on a simple dual-port buffer
//         (one read, one write per cycle): t = rnd((b * w) >> 16) with Q16 twiddles,
//         a' = (a + t + 1) >> 1, b' = (a - t + 1) >> 1
//   MAG   bins 1..127: isqrt(re^2 + im^2), one result bit per cycle; mel on the fly (<= 2 bands per
//         bin) as two read-modify-writes of a 40-entry LUT-RAM with one shared multiplier
//   LOG   40 x (leading one + 64-entry log2 table, Q8) * 256 + 65536 (sh - 6) - K16D
// Written in the same style as cim_fe (one operation per unit per cycle) so that the two costs
// compare like for like. lraw of an exact zero reads as ZERO = 0x8000_0000.

module mfcc_ded (
    input  logic               clk_i,
    input  logic               rst_ni,
    // control
    input  logic               start_i,              // start one frame (ignored while busy)
    input  logic [9:0]         base_i,               // ring index of sample 0 of the frame
    output logic               busy_o,
    output logic               done_o,               // one-cycle pulse: lraw complete
    // sample ring read port (data one cycle after the address)
    output logic [9:0]         samp_addr_o,
    input  logic signed [15:0] samp_i,
    // lraw read port (combinational)
    input  logic [5:0]         lraw_addr_i,
    output logic [31:0]        lraw_o
);

  `include "ded_tables.svh"

  localparam logic [31:0] ZERO = 32'h8000_0000;

  typedef enum logic [2:0] {S_IDLE, S_PEAK, S_EXP, S_LOAD, S_FFT, S_MAG, S_LOG, S_DONE} state_t;

  state_t      state;
  logic [9:0]  base_q;
  logic [9:0]  cnt;                                  // element / cycle counter of the current phase
  logic [3:0]  stage;
  logic [6:0]  bin;
  logic [4:0]  mph;
  logic        silent;
  logic signed [5:0] sh_q;
  logic [29:0] peak;

  // ---------------------------------------------------------------- arithmetic helpers
  // (the temporaries are wider than the results; their top bits are dropped on purpose)
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic signed [17:0] bfp(input logic signed [30:0] v, input logic signed [5:0] sh);
    logic signed [31:0] t;
    if (sh > 0) t = (32'(v) + (32'sd1 <<< (sh - 6'sd1))) >>> sh;
    else t = 32'(v) <<< (-sh);
    return t[17:0];
  endfunction

  function automatic logic signed [18:0] rsh16(input logic signed [36:0] v);
    logic signed [36:0] t;
    t = (v + 37'sd32768) >>> 16;
    return t[18:0];
  endfunction

  function automatic logic signed [17:0] half(input logic signed [19:0] v);
    logic signed [19:0] t;
    t = (v + 20'sd1) >>> 1;
    return t[17:0];
  endfunction

  function automatic logic [13:0] log2_q8(input logic [39:0] v);   // v >= 1
    int unsigned e;
    logic [39:0] m;
    e = 0;
    for (int unsigned i = 0; i < 40; i++) if (v[i]) e = i;
    m = (e >= 6) ? (v >> (e - 6)) : (v << (6 - e));
    return 14'(e * 256) + 14'(DED_LOG_LUT[m[5:0]]);
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  function automatic logic [8:0] bitrev9(input logic [8:0] i);
    for (int k = 0; k < 9; k++) bitrev9[k] = i[8 - k];
  endfunction

  // ---------------------------------------------------------------- FFT buffer (simple dual port)
  logic [35:0] fmem [512];                           // {re, im}, 18 bits each
  logic [8:0]  raddr, waddr;
  logic        we;
  logic [35:0] wdata, rdata;

  always_ff @(posedge clk_i) begin
    if (we) fmem[waddr] <= wdata;
    rdata <= fmem[raddr];
  end

  // ---------------------------------------------------------------- sample x Hann pipeline
  // issue at c (ring and ROM addresses), data at c+1, product at c+2 (PEAK and LOAD)
  logic        sh_issue, sh_zero;
  logic [8:0]  sh_idx;
  logic        sp1_v, sp1_z, sp2_v, sp2_z;
  logic [8:0]  sp1_tag, sp2_tag;
  logic [15:0] hann_q;
  logic signed [30:0] prod_q;
  logic [29:0] prod_abs;

  always_comb begin
    sh_issue = 1'b0;
    sh_zero  = 1'b0;
    sh_idx   = 9'd0;
    if (state == S_PEAK && cnt < 10'd480) begin
      sh_issue = 1'b1;
      sh_idx   = cnt[8:0];
    end else if (state == S_LOAD && cnt < 10'd512) begin
      sh_issue = 1'b1;
      sh_zero  = (cnt >= 10'd480);
      sh_idx   = sh_zero ? 9'd0 : cnt[8:0];
    end
  end

  assign samp_addr_o = base_q + 10'(sh_idx);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sp1_v <= 1'b0; sp1_z <= 1'b0; sp1_tag <= '0;
      sp2_v <= 1'b0; sp2_z <= 1'b0; sp2_tag <= '0;
    end else begin
      sp1_v <= sh_issue; sp1_z <= sh_zero; sp1_tag <= cnt[8:0];
      sp2_v <= sp1_v;    sp2_z <= sp1_z;   sp2_tag <= sp1_tag;
    end
  end

  always_ff @(posedge clk_i) begin
    hann_q <= DED_HANN[sh_idx];
    prod_q <= 31'(samp_i * $signed({1'b0, hann_q}));
  end
  assign prod_abs = prod_q[30] ? 30'(-prod_q) : 30'(prod_q);

  // ---------------------------------------------------------------- butterfly pipeline
  // Butterfly k: cycle 2k reads a = X[i0], cycle 2k+1 reads b = X[i1]; the products of b and the
  // twiddle are registered at 2k+2, the rounded t at 2k+3; a' is written at 2k+4 and b' at 2k+5.
  logic [7:0]  bk;                                   // butterfly index within the stage
  logic        rd_v;                                 // a butterfly read is issued this cycle
  logic [8:0]  i0, i1;
  logic [7:0]  twi;
  logic [35:0] tw_q;
  logic        r1_v, r1_odd, p_v, t_v, w1_v;
  logic [8:0]  r1_i0, r1_i1, p_i0, p_i1, t_i0, t_i1, w1_i1;
  logic signed [17:0] aq_re, aq_im, a1_re, a1_im, a2_re, a2_im;
  logic signed [35:0] p_rr, p_ii, p_ri, p_ir;
  logic signed [18:0] t_re, t_im;
  logic [35:0] x1_q;

  assign bk   = cnt[8:1];
  assign rd_v = (state == S_FFT) && (cnt < 10'd512);
  always_comb begin
    logic [8:0] h, m, j;
    h   = 9'd1 << stage;
    m   = 9'(bk) & (h - 9'd1);
    j   = 9'(bk) >> stage;
    i0  = (j << (stage + 4'd1)) | m;
    i1  = i0 | h;
    twi = 8'(m << (4'd8 - stage));
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      r1_v <= 1'b0; r1_odd <= 1'b0; p_v <= 1'b0; t_v <= 1'b0; w1_v <= 1'b0;
    end else begin
      r1_v  <= rd_v;
      r1_odd <= cnt[0];
      p_v   <= r1_v && r1_odd;                       // b arrived: products registered
      t_v   <= p_v;
      w1_v  <= t_v;
    end
  end

  always_ff @(posedge clk_i) begin
    tw_q  <= DED_TW[twi];                            // same butterfly in both read cycles
    r1_i0 <= i0;
    r1_i1 <= i1;
    if (r1_v && !r1_odd) begin                       // a arrived
      aq_re <= $signed(rdata[35:18]);
      aq_im <= $signed(rdata[17:0]);
    end
    // b arrived (r1_v && r1_odd): products of b and the twiddle
    p_rr  <= $signed(rdata[35:18]) * $signed(tw_q[35:18]);
    p_ii  <= $signed(rdata[17:0])  * $signed(tw_q[17:0]);
    p_ri  <= $signed(rdata[35:18]) * $signed(tw_q[17:0]);
    p_ir  <= $signed(rdata[17:0])  * $signed(tw_q[35:18]);
    a1_re <= aq_re;
    a1_im <= aq_im;
    p_i0  <= r1_i0;
    p_i1  <= r1_i1;
    // rounded twiddle product
    t_re  <= rsh16(37'(p_rr) - 37'(p_ii));
    t_im  <= rsh16(37'(p_ri) + 37'(p_ir));
    a2_re <= a1_re;
    a2_im <= a1_im;
    t_i0  <= p_i0;
    t_i1  <= p_i1;
    // b' waits one cycle for the write port
    x1_q  <= {half(20'(a2_re) - 20'(t_re)), half(20'(a2_im) - 20'(t_im))};
    w1_i1 <= t_i1;
  end

  // ---------------------------------------------------------------- buffer port control
  always_comb begin
    raddr = '0;
    waddr = '0;
    we    = 1'b0;
    wdata = '0;
    unique case (state)
      S_LOAD: begin                                  // bit-reversed load
        waddr = bitrev9(sp2_tag);
        we    = sp2_v;
        wdata = {sp2_z ? 18'sd0 : bfp(prod_q, sh_q), 18'sd0};
      end
      S_FFT: begin
        raddr = cnt[0] ? i1 : i0;
        if (t_v) begin                               // a' (even cycle)
          waddr = t_i0;
          we    = 1'b1;
          wdata = {half(20'(a2_re) + 20'(t_re)), half(20'(a2_im) + 20'(t_im))};
        end else if (w1_v) begin                     // b' (odd cycle)
          waddr = w1_i1;
          we    = 1'b1;
          wdata = x1_q;
        end
      end
      S_MAG: raddr = 9'(bin);
      default: ;
    endcase
  end

  // ---------------------------------------------------------------- magnitude, mel, log
  logic [21:0] melw;                                 // {m_lo, w_lo, w_hi} of the current bin
  logic signed [35:0] sq_re, sq_im;
  logic [36:0] rem, root, bitm;
  logic [39:0] mel_ram [64];                         // 40 used; LUT-RAM (write port, async read)
  logic [5:0]  mel_ra;
  logic [39:0] mel_rd, mprod;
  logic [31:0] lraw_q [64];
  logic [31:0] lraw_d;
  assign lraw_o = lraw_q[lraw_addr_i];
  assign mel_ra = (state == S_LOG) ? cnt[5:0] : ((mph == 5'd21) ? melw[21:16] : melw[21:16] + 6'd1);
  assign mel_rd = mel_ram[mel_ra];
  assign mprod  = 40'(root[17:0]) * 40'((mph == 5'd21) ? melw[15:8] : melw[7:0]);

  always_comb begin
    if (silent || mel_rd == '0) lraw_d = ZERO;
    else lraw_d = 32'(log2_q8(mel_rd)) * 32'd256 + 32'(65536 * (int'(sh_q) - 6)) - 32'(DED_K16);
  end

  // buffers outside the reset block, so that they map to RAM rather than flip-flops
  always_ff @(posedge clk_i) begin
    if (state == S_LOAD && cnt < 10'd40) mel_ram[cnt[5:0]] <= '0;          // clear the mel sums
    else if (state == S_MAG && (mph == 5'd21 || (mph == 5'd22 && melw[21:16] != 6'd39)))
      mel_ram[mel_ra] <= mel_rd + mprod;
    if (state == S_LOG) lraw_q[cnt[5:0]] <= lraw_d;
  end

  // ---------------------------------------------------------------- control
  assign busy_o = (state != S_IDLE);
  assign done_o = (state == S_DONE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state  <= S_IDLE;
      base_q <= '0;
      cnt    <= '0;
      stage  <= '0;
      bin    <= '0;
      mph    <= '0;
      silent <= 1'b0;
      sh_q   <= '0;
      peak   <= '0;
      melw   <= '0;
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
          peak   <= '0;
        end
        S_PEAK: begin
          if (cnt < 10'd480) cnt <= cnt + 10'd1;
          if (sp2_v && prod_abs > peak) peak <= prod_abs;
          if (cnt == 10'd480 && !sp1_v && !sp2_v) state <= S_EXP;
        end
        S_EXP: begin
          int unsigned e;
          e = 0;
          for (int unsigned i = 0; i < 30; i++) if (peak[i]) e = i;
          sh_q   <= 6'(signed'(e) - 14);
          silent <= (peak == '0);
          cnt    <= '0;
          state  <= (peak == '0) ? S_LOG : S_LOAD;
        end
        S_LOAD: begin
          if (cnt < 10'd512) cnt <= cnt + 10'd1;
          else if (!sp1_v && !sp2_v) begin
            state <= S_FFT;
            stage <= '0;
            cnt   <= '0;
          end
        end
        S_FFT: begin                                 // 516 cycles per stage: 256 x 2 + drain
          if (cnt == 10'd515) begin
            cnt <= '0;
            if (stage == 4'd8) begin
              state <= S_MAG;
              bin   <= 7'd1;
              mph   <= '0;
            end else stage <= stage + 4'd1;
          end else cnt <= cnt + 10'd1;
        end
        S_MAG: begin
          unique case (mph)
            5'd0: begin                              // rdata and the mel entry valid next cycle
              melw <= DED_MEL[bin];
              mph  <= 5'd1;
            end
            5'd1: begin
              sq_re <= $signed(rdata[35:18]) * $signed(rdata[35:18]);
              sq_im <= $signed(rdata[17:0]) * $signed(rdata[17:0]);
              mph   <= 5'd2;
            end
            5'd2: begin
              rem  <= 37'(sq_re) + 37'(sq_im);
              root <= '0;
              bitm <= 37'(1) << 34;
              mph  <= 5'd3;
            end
            5'd21: mph <= 5'd22;                     // mel, lower band (buffer write block)
            5'd22: begin                             // mel, upper band
              mph <= '0;
              if (bin == 7'd127) begin
                state <= S_LOG;
                cnt   <= '0;
              end else bin <= bin + 7'd1;
            end
            default: begin                           // 18 isqrt iterations, mph 3..20
              if (rem >= root + bitm) begin
                rem  <= rem - (root + bitm);
                root <= (root >> 1) + bitm;
              end else root <= root >> 1;
              bitm <= bitm >> 2;
              mph  <= mph + 5'd1;
            end
          endcase
        end
        S_LOG: begin
          if (cnt == 10'd39) state <= S_DONE;
          else cnt <= cnt + 10'd1;
        end
        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
