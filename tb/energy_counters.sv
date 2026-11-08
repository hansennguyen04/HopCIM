// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Activity counters for the energy estimate (docs/ENERGY_ESTIMATE.md, model/energy_est.py). Bound
// from outside the RTL, simulation only (sim/run_energy_counts.sh); they drive nothing in the design.
// Each block prints its counts at the end of the run as lines "EC <block> <name> <count>".
`timescale 1ns / 1ps

module ec_engine #(parameter int unsigned TW = 7) (
    input logic clk_i, input logic start_i, input logic ready_i, input logic [TW-1:0] tile_i,
    input logic [4:0] xbits_i, input logic xw_we_i, input logic [7:0] xw_mask_i, input logic x_we_i,
    input logic w_we_i
);
  longint unsigned pass[128][32];
  longint unsigned xw_rows = 0, x_rows = 0, w_words = 0;
  always_ff @(posedge clk_i) begin
    if (start_i && ready_i) pass[7'(tile_i)][xbits_i] <= pass[7'(tile_i)][xbits_i] + 1;
    if (xw_we_i) xw_rows <= xw_rows + longint'($countones(xw_mask_i));
    if (x_we_i && !xw_we_i) x_rows <= x_rows + 1;
    if (w_we_i) w_words <= w_words + 1;
  end
  final begin
    for (int t = 0; t < 128; t++) for (int b = 0; b < 32; b++)
      if (pass[t][b] != 0) $display("EC engine pass_t%0d_b%0d %0d", t, b, pass[t][b]);
    $display("EC engine xw_rows %0d", xw_rows);
    $display("EC engine x_rows %0d", x_rows);
    $display("EC engine w_words %0d", w_words);
  end
endmodule

module ec_seq (
    input logic clk_i, input logic run_i, input logic g_req_i, input logic [1:0] lt_i,
    input logic g_tv_i, input logic q0_v_i, input logic q4_v_i, input logic act_we_i,
    input logic [3:0] lay_i, input logic cp_v_i, input logic feat_we_i, input logic la_v_i,
    input logic [3:0] ws_i, input logic ra_v_i, input logic prm_we_i, input logic go_i, input logic eng_ready_i,
    // the baseline S: lane 1 of the requantisation, the sign gate (both 0 in R6b)
    input logic r0_v_i, input logic r4_v_i, input logic act_we1_i, input logic q1_v_i, input logic q1_g_i,
    input logic r1_v_i, input logic r1_g_i,
    // B1 (FUSE1): the fused segment's gather sources, the line-buffer fill, the fusion-vector writes (0 otherwise)
    input logic fz_i, input logic g_pw_i, input logic f_rd_i, input logic f1_v_i, input logic fv_we0_i, input logic fv_we1_i
);
  longint unsigned ws_cyc[16];
  longint unsigned g_lt[4], g_lt_tv[4];
  longint unsigned run_cyc = 0, q0 = 0, q4 = 0, act_w = 0, pool_up = 0, cp = 0, feat_w = 0, la = 0, ra = 0, prm = 0, passes = 0;
  longint unsigned r0 = 0, mults = 0, gated = 0;
  longint unsigned g_lb = 0, g_fv = 0, f_rd = 0, lb_w = 0, fv_w = 0;
  always_ff @(posedge clk_i) begin
    ws_cyc[ws_i] <= ws_cyc[ws_i] + 1;
    if (run_i) run_cyc <= run_cyc + 1;
    if (g_req_i) begin
      g_lt[lt_i] <= g_lt[lt_i] + 1;
      if (g_tv_i) g_lt_tv[lt_i] <= g_lt_tv[lt_i] + 1;
    end
    q0 <= q0 + longint'(q0_v_i) + longint'(r0_v_i);                    // requantisations, both lanes
    q4 <= q4 + longint'(q4_v_i) + longint'(r4_v_i);
    act_w <= act_w + longint'(act_we_i) + longint'(act_we1_i);
    if (lay_i == 4'd8) pool_up <= pool_up + longint'(q4_v_i) + longint'(r4_v_i);
    if (r0_v_i) r0 <= r0 + 1;
    mults <= mults + longint'(q1_v_i && !q1_g_i) + longint'(r1_v_i && !r1_g_i);   // multiplies executed
    gated <= gated + longint'(q1_v_i && q1_g_i) + longint'(r1_v_i && r1_g_i);
    if (cp_v_i) cp <= cp + 1;
    if (feat_we_i) feat_w <= feat_w + 1;
    if (la_v_i) la <= la + 1;
    if (ra_v_i) ra <= ra + 1;
    if (prm_we_i) prm <= prm + 1;
    if (go_i && eng_ready_i) passes <= passes + 1;
    if (g_req_i && fz_i && !g_pw_i && g_tv_i) g_lb <= g_lb + 1;     // depthwise taps from the line buffer
    if (g_req_i && fz_i && g_pw_i) g_fv <= g_fv + 1;                 // pointwise chunks from the fusion vector
    if (f_rd_i) f_rd <= f_rd + 1;                                    // fill: ACT word reads
    if (f1_v_i) lb_w <= lb_w + 1;                                    // fill: line-buffer word writes
    fv_w <= fv_w + longint'(fv_we0_i) + longint'(fv_we1_i);           // fusion-vector byte writes
  end
  final begin
    for (int i = 0; i < 16; i++) if (ws_cyc[i] != 0) $display("EC seq ws%0d %0d", i, ws_cyc[i]);
    for (int i = 0; i < 4; i++) begin
      $display("EC seq gather_lt%0d %0d", i, g_lt[i]);
      $display("EC seq gather_tv_lt%0d %0d", i, g_lt_tv[i]);
    end
    $display("EC seq run_cycles %0d", run_cyc);
    $display("EC seq requant %0d", q0);
    $display("EC seq writeback %0d", q4);
    $display("EC seq act_writes %0d", act_w);
    $display("EC seq pool_updates %0d", pool_up);
    $display("EC seq lraw_ring_writes %0d", cp);
    $display("EC seq feat_writes %0d", feat_w);
    $display("EC seq lraw_ring_reads %0d", la);
    $display("EC seq lmq_dct_requant %0d", ra);
    $display("EC seq passes %0d", passes);
    if (r0 != 0 || gated != 0) begin                                   // only with the baseline S options
      $display("EC seq requant_lane1 %0d", r0);
      $display("EC seq requant_mults %0d", mults);
      $display("EC seq requant_gated %0d", gated);
    end
    if (f_rd != 0) begin                                               // only with B1
      $display("EC seq b1_gather_lb %0d", g_lb);
      $display("EC seq b1_gather_fv %0d", g_fv);
      $display("EC seq b1_fill_act_reads %0d", f_rd);
      $display("EC seq b1_lb_writes %0d", lb_w);
      $display("EC seq b1_fv_byte_writes %0d", fv_w);
    end
  end
endmodule

module ec_fe (
    input logic clk_i, input logic [4:0] state_i, input logic sh_issue_i, input logic sh_zero_i,
    input logic s1_issue_i, input logic t2_v_i, input logic g_v_i, input logic [5:0] mph_i,
    input logic start_i
);
  longint unsigned st[32];
  longint unsigned samp = 0, s1 = 0, bbuf_w = 0, gv = 0, mag = 0, start = 0;
  always_ff @(posedge clk_i) begin
    st[state_i] <= st[state_i] + 1;
    if (sh_issue_i && !sh_zero_i) samp <= samp + 1;
    if (s1_issue_i) s1 <= s1 + 1;
    if (t2_v_i) bbuf_w <= bbuf_w + 1;
    if (g_v_i) gv <= gv + 1;
    if (state_i == 5'd11 && mph_i == 6'd24) mag <= mag + 1;     // S_MAG: results
    if (start_i) start <= start + 1;
  end
  final begin
    for (int i = 0; i < 32; i++) if (st[i] != 0) $display("EC fe st%0d %0d", i, st[i]);
    $display("EC fe samples %0d", samp);
    $display("EC fe twiddle_scatter %0d", s1);
    $display("EC fe bbuf_writes %0d", bbuf_w);
    $display("EC fe gather_reads %0d", gv);
    $display("EC fe mag_results %0d", mag);
    $display("EC fe starts %0d", start);
  end
endmodule

module ec_ded (
    input logic clk_i, input logic [2:0] state_i, input logic we_i, input logic rd_v_i, input logic sh_issue_i,
    input logic [4:0] mph_i, input logic start_i
);
  longint unsigned st[8];
  longint unsigned w = 0, r = 0, samp = 0, mph21 = 0, start = 0;
  always_ff @(posedge clk_i) begin
    st[state_i] <= st[state_i] + 1;
    if (we_i) w <= w + 1;
    if (rd_v_i) r <= r + 1;
    if (sh_issue_i) samp <= samp + 1;
    if (state_i == 3'd5 && mph_i == 5'd21) mph21 <= mph21 + 1;   // S_MAG: one bin's mel update
    if (start_i) start <= start + 1;
  end
  final begin
    for (int i = 0; i < 8; i++) if (st[i] != 0) $display("EC ded st%0d %0d", i, st[i]);
    $display("EC ded fmem_writes %0d", w);
    $display("EC ded fmem_reads_fft %0d", r);
    $display("EC ded window_products %0d", samp);
    $display("EC ded mag_results %0d", mph21);
    $display("EC ded starts %0d", start);
  end
endmodule

module ec_endpoint (input logic clk_i, input logic ring_we_i, input logic aud_seq_go_i, input logic aud_fe_go_i, input logic aud_en_i);
  longint unsigned ring_w = 0, seq_go = 0, fe_go = 0, cyc = 0;
  always_ff @(posedge clk_i) begin
    if (ring_we_i) ring_w <= ring_w + 1;
    if (aud_seq_go_i) seq_go <= seq_go + 1;
    if (aud_fe_go_i) fe_go <= fe_go + 1;
    if (aud_en_i) cyc <= cyc + 1;
  end
  final begin
    $display("EC ep ring_writes %0d", ring_w);
    $display("EC ep seq_starts %0d", seq_go);
    $display("EC ep fe_starts %0d", fe_go);
    $display("EC ep enabled_cycles %0d", cyc);
  end
endmodule

bind cim_engine ec_engine #(.TW(TW)) u_ec (.clk_i(clk_i), .start_i(start_i), .ready_i(ready_o), .tile_i(tile_i),
    .xbits_i(xbits_i), .xw_we_i(xw_we_i), .xw_mask_i(xw_mask_i), .x_we_i(x_we_i), .w_we_i(w_we_i));
bind cim_seq ec_seq u_ec (.clk_i(clk_i), .run_i(run), .g_req_i(g_req), .lt_i(2'(lt)), .g_tv_i(g_tv), .q0_v_i(q0_v),
    .q4_v_i(q4_v), .act_we_i(act_we), .lay_i(lay), .cp_v_i(cp_v), .feat_we_i(feat_we), .la_v_i(la_v), .ws_i(4'(ws)),
    .ra_v_i(ra_v), .prm_we_i(prm_we_i), .go_i(go_o), .eng_ready_i(eng_ready_i), .r0_v_i(r0_v), .r4_v_i(r4_v),
    .act_we1_i(act_we1), .q1_v_i(q1_v), .q1_g_i(q1_g), .r1_v_i(r1_v), .r1_g_i(r1_g),
    .fz_i(fz), .g_pw_i(g_pw), .f_rd_i(f_on && f_free), .f1_v_i(f1_v), .fv_we0_i(fv_we0), .fv_we1_i(fv_we1));
bind cim_fe ec_fe u_ec (.clk_i(clk_i), .state_i(5'(state)), .sh_issue_i(sh_issue), .sh_zero_i(sh_zero),
    .s1_issue_i(s1_issue), .t2_v_i(t2_v), .g_v_i(g_v), .mph_i(mph), .start_i(start_i));
bind mfcc_ded ec_ded u_ec (.clk_i(clk_i), .state_i(3'(state)), .we_i(we), .rd_v_i(rd_v), .sh_issue_i(sh_issue),
    .mph_i(mph), .start_i(start_i));
bind cim_endpoint ec_endpoint u_ec (.clk_i(clk_i), .ring_we_i(ring_we), .aud_seq_go_i(aud_seq_go), .aud_fe_go_i(aud_fe_go), .aud_en_i(aud_en));
