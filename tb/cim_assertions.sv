// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Assertions on the HopCIM RTL, bound from outside it, simulation only (sim/run_cov.sh). They drive
// nothing. Every check counts its activations as well as its failures, and the final lines
//   ASSERT <unit>.<name> active <n> fail <n>
// let model/verif_report.py reject a check that was never exercised: a check that never fires proves
// nothing. A failure prints once per check at its first occurrence and fails the bench's run.
`timescale 1ns / 1ps

`define CHK(idx, cond, msg) \
  begin active[idx] <= active[idx] + 1; \
    if (!(cond)) begin fail[idx] <= fail[idx] + 1; \
      if (fail[idx] == 0) $display("ASSERTION FAILED %s %m %s at %0t", NAME[idx], msg, $time); end end

// ---------------------------------------------------------------------- the engine
module sva_engine #(parameter int unsigned TW = 7) (
    input logic clk_i, input logic rst_ni, input logic start_i, input logic ready_o, input logic done_o, input logic busy_o,
    input logic [4:0] xbits_i, input logic xbank_i, input logic [TW-1:0] tile_i, input logic x_we_i, input logic xw_we_i,
    input logic xw_bank_i, input logic [7:0] xw_mask_i, input logic w_we_i, input logic [TW-1:0] w_tile_i
);
  localparam int N = 9;
  longint unsigned active[N] = '{default: 0}, fail[N] = '{default: 0};
  localparam string NAME[N] = '{"start_only_when_ready", "done_at_d_plus_5", "no_done_without_pass", "ready_low_d_minus_1",
                                "input_bank_stable_port2", "input_bank_stable_port1", "tile_not_written_in_pass",
                                "busy_covers_pass", "done_is_a_pulse"};
  longint unsigned now = 0, t_ready = 0;
  longint unsigned done_at[8], t_bank[2] = '{0, 0}, tile_until = 0;
  bit started[2] = '{1'b0, 1'b0};
  logic [TW-1:0] tile_q;
  int h = 0, t = 0;
  logic done_q = 1'b0;

  function automatic int unsigned pass_d(input logic [4:0] b);
    int unsigned e = (b == 0) ? 1 : ((int'(b) > 24) ? 24 : int'(b));
    return e > 8 ? e : 8;
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      h <= 0; t <= 0; now <= 0; done_q <= 1'b0;
    end else begin
      now <= now + 1;
      done_q <= done_o;
      if (start_i) `CHK(0, ready_o, "start while the engine is not ready")
      if (start_i && ready_o) begin
        done_at[t % 8] <= now + longint'(pass_d(xbits_i)) + 5;
        t <= t + 1;
        t_ready <= now + longint'(pass_d(xbits_i));
        t_bank[xbank_i] <= now;
        started[xbank_i] <= 1'b1;
        tile_until <= now + 9;
        tile_q <= tile_i;
      end
      if (done_o) begin
        `CHK(2, h != t, "done with no pass in flight")
        if (h != t) `CHK(1, done_at[h % 8] == now, "done not at t0 + D + 5")
        h <= h + 1;
      end else if (h != t && done_at[h % 8] <= now) `CHK(1, 1'b0, "done missing at t0 + D + 5")
      if (now < t_ready) `CHK(3, !ready_o, "ready before t0 + D")
      else if (now >= 1) `CHK(3, ready_o, "not ready at t0 + D")
      if (xw_we_i && xw_mask_i != 8'd0) `CHK(4, !(started[xw_bank_i] && now > t_bank[xw_bank_i] && now <= t_bank[xw_bank_i] + 8),
                                            "wide-port write to a bank a pass reads")
      if (x_we_i && !xw_we_i) `CHK(5, !(started[0] && now > t_bank[0] && now <= t_bank[0] + 8), "port-1 write to the bank a pass reads")
      if (w_we_i) `CHK(6, !(now <= tile_until && tile_until != 0 && w_tile_i == tile_q), "weight write to the tile a pass reads")
      if (h != t) `CHK(7, busy_o, "a pass in flight without busy")
      if (done_o || done_q) `CHK(8, !(done_o && done_q), "done high for two cycles")
    end
  end
  final for (int i = 0; i < N; i++) $display("ASSERT engine.%s active %0d fail %0d", NAME[i], active[i], fail[i]);
endmodule

// ---------------------------------------------------------------------- the sequencer
module sva_seq (
    input logic clk_i, input logic rst_ni, input logic busy_o, input logic done_o, input logic err_o, input logic [3:0] lay,
    input logic pop, input logic [2:0] tq_cnt, input logic [4:0] d_rem, input logic issue, input logic g2_v, input logic g2_b,
    input logic [1:0] bbusy, input logic [2:0] bcnt0, input logic [2:0] bcnt1, input logic g_req, input logic g_b, input logic [5:0] g_k,
    input logic act_we, input logic [11:0] act_wa, input logic [6:0] d_p, input logic [5:0] d_c, input logic d_rem_nz,
    input logic [5:0] w_row, input logic [5:0] w_new, input logic [5:0] hptr, input logic [3:0] ws, input logic wp_on,
    input logic net_on, input logic go_o, input logic eng_ready_i, input logic [6:0] g_p, input logic q0_v, input logic q4_v
);
  localparam int N = 13;
  longint unsigned active[N] = '{default: 0}, fail[N] = '{default: 0};
  localparam string NAME[N] = '{"no_error_flag", "layer_in_range", "pop_has_a_tag", "results_drained_before_pop", "tag_queue_bounded",
                                "gather_starts_on_free_bank", "go_only_when_ready", "position_in_range", "row_in_range",
                                "block_slot_in_range", "one_phase_at_a_time", "done_ends_busy", "requant_keeps_count"};
  logic busy_q = 1'b0;
  longint signed rq_in = 0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      busy_q <= 1'b0; rq_in <= 0;
    end else begin
      busy_q <= busy_o;
      if (busy_o) `CHK(0, !err_o, "the sequencer's error flag is set")
      if (busy_o) `CHK(1, lay <= 4'd9, "layer out of range")
      if (pop) `CHK(2, tq_cnt != 3'd0, "a result without a tag")
      if (pop) `CHK(3, d_rem <= 5'd1, "new results before the previous ones drained")
      if (busy_o) `CHK(4, tq_cnt <= 3'd4, "tag queue overflow")
      if (g_req && g_k == 6'd0) `CHK(5, !bbusy[g_b] && ((g_b ? bcnt1 : bcnt0) == 3'd0), "gather into a bank still in use")
      if (go_o) `CHK(6, eng_ready_i, "go while the engine is not ready")
      if (d_rem_nz) `CHK(7, d_p < 7'd125, "write-back position out of range")
      if (wp_on) `CHK(8, w_row < 6'd49 && w_new < 6'd49, "feature row out of range")
      if (wp_on) `CHK(9, hptr < 6'd50, "block ring slot out of range")
      if (busy_o) `CHK(10, !(wp_on && net_on), "window path and network at once")
      if (done_o) `CHK(11, busy_q && !busy_o, "done without the end of a run")
      if (busy_o && !busy_q) rq_in <= 0;
      else rq_in <= rq_in + longint'(q0_v) - longint'(q4_v);
      if (done_o) `CHK(12, rq_in == 0, "a requantisation lost or duplicated")
    end
  end
  final for (int i = 0; i < N; i++) $display("ASSERT seq.%s active %0d fail %0d", NAME[i], active[i], fail[i]);
endmodule

// ---------------------------------------------------------------------- the front-ends (cim_fe, mfcc_ded)
module sva_frontend (
    input logic clk_i, input logic rst_ni, input logic start_i, input logic busy_o, input logic done_o,
    input logic [9:0] samp_addr_o, input logic [9:0] base_q
);
  localparam int N = 4;
  longint unsigned active[N] = '{default: 0}, fail[N] = '{default: 0};
  localparam string NAME[N] = '{"done_ends_busy", "reads_inside_its_frame", "base_stable_while_busy", "frame_time_bounded"};
  logic done_q = 1'b0, busy_q = 1'b0;
  logic [9:0] base_qq = '0;
  longint unsigned run = 0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      done_q <= 1'b0; busy_q <= 1'b0; run <= 0;
    end else begin
      done_q <= done_o;
      busy_q <= busy_o;
      base_qq <= base_q;
      run <= busy_o ? run + 1 : 0;
      if (done_q) `CHK(0, !busy_o, "a front-end still busy after done")
      if (busy_o) `CHK(1, 10'(samp_addr_o - base_q) < 10'd640, "a front-end read outside its frame")
      if (busy_o && busy_q) `CHK(2, base_q == base_qq, "the frame base changed during a frame")
      if (busy_o) `CHK(3, run < 200000, "a frame takes too long")
    end
  end
  final for (int i = 0; i < N; i++) $display("ASSERT frontend.%s active %0d fail %0d", NAME[i], active[i], fail[i]);
endmodule

// ---------------------------------------------------------------------- what the benches must reach
// Counted from the design's own signals, because coverage of a function's `if (...) return` is not reliable.
module cov_seq (
    input logic clk_i, input logic q3_v, input logic signed [24:0] q3_r, input logic [3:0] q3_e, input logic [7:0] zo,
    input logic rb_v, input logic signed [12:0] rq_t
);
  longint unsigned hi = 0, lo = 0, fhi = 0, flo = 0;
  logic signed [26:0] t;
  always_comb begin
    t = 27'(q3_r);
    if (q3_e != 4'd0) t = t + (27'sd1 <<< (q3_e - 4'd1)) - (q3_r[24] ? 27'sd1 : 27'sd0);
    t = (t >>> q3_e) + 27'(signed'(zo));
  end
  always_ff @(posedge clk_i) begin
    if (q3_v && t > 27'sd127) hi <= hi + 1;
    if (q3_v && t < -27'sd128) lo <= lo + 1;
    if (rb_v && rq_t > 13'sd127) fhi <= fhi + 1;
    if (rb_v && rq_t < -13'sd128) flo <= flo + 1;
  end
  final begin
    $display("COVER seq.requant_clamp_high %0d", hi);
    $display("COVER seq.requant_clamp_low %0d", lo);
    $display("COVER seq.feature_clamp_high %0d", fhi);
    $display("COVER seq.feature_clamp_low %0d", flo);
  end
endmodule

// ---------------------------------------------------------------------- the audio path
module sva_audio #(parameter int unsigned CLKDIV = 15) (
    input logic clk_i, input logic rst_ni, input logic en_i, input logic sv, input logic hop_end, input logic [8:0] hs,
    input logic [2:0] hc, input logic [3:0] kcnt, input logic [3:0] net_km1_i, input logic [15:0] ndec_o, input logic [31:0] nhop_o,
    input logic pend_o, input logic fe_go_o, input logic fe_ok_i, input logic seq_go_o, input logic seq_ok_i, input logic seq_done_i,
    input logic fe_done_i, input logic hc_full, input logic [5:0] frow, input logic [9:0] wptr
);
  localparam int N = 11;
  longint unsigned active[N] = '{default: 0}, fail[N] = '{default: 0};
  localparam string NAME[N] = '{"sample_period", "hop_is_320_samples", "hop_count_steps", "decision_count_steps", "decision_only_on_network_hop",
                                "stride_counter_bounded", "fe_go_when_allowed", "seq_go_when_allowed", "row_in_range", "wptr_follows_samples",
                                "pending_after_decision"};
  longint unsigned now = 0, t_sv = 0;
  bit en_q = 1'b0, en_qq = 1'b0;
  int n_sv = 0;
  logic [15:0] ndec_q = '0;
  logic [31:0] nhop_q = '0;
  logic [9:0] wptr_q = '0;
  logic sv_q = 1'b0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      now <= 0; n_sv <= 0; en_q <= 1'b0; en_qq <= 1'b0;
      ndec_q <= '0; nhop_q <= '0; wptr_q <= '0; sv_q <= 1'b0;
    end else begin
      now <= now + 1;
      en_q <= en_i;
      en_qq <= en_i && en_q;
      ndec_q <= ndec_o;
      nhop_q <= nhop_o;
      wptr_q <= wptr;
      sv_q <= sv;
      if (!en_i) n_sv <= 0;
      if (en_i && sv) begin
        // from the second sample of a stream: a restart does not reset the decimator (LIVE_AUDIO.md section 3), so a
        // sample already in the FIR when the path stopped can arrive first, and its interval includes the stop
        if (n_sv >= 2) `CHK(0, now - t_sv == longint'(64 * CLKDIV), $sformatf("sample period is %0d, not 64 PDM clocks (%0d)", now - t_sv, 64 * CLKDIV))
        t_sv <= now;
        n_sv <= n_sv + 1;
      end
      if (en_i && hop_end) `CHK(1, hs == 9'd319, "hop end not at the 320th sample")
      if (en_i && en_qq && nhop_o != nhop_q) `CHK(2, nhop_o == nhop_q + 1, "hop count jumped")
      if (en_i && en_qq && ndec_o != ndec_q) begin
        `CHK(3, ndec_o == ndec_q + 16'd1, "decision count jumped")
        `CHK(4, hc_full, "a decision on a hop without the network")
        `CHK(10, pend_o, "a decision without its pending flag")
      end
      if (en_i) `CHK(5, kcnt <= net_km1_i || net_km1_i == 4'd0, "stride counter past k - 1")
      if (fe_go_o) `CHK(6, fe_ok_i, "front-end start not allowed")
      if (seq_go_o) `CHK(7, seq_ok_i, "sequencer start not allowed")
      if (en_i) `CHK(8, frow < 6'd49, "frame row out of range")
      if (en_i && en_q && sv_q) `CHK(9, wptr == wptr_q + 10'd1 || wptr == 10'd0, "ring pointer not one step per sample")
    end
  end
  final for (int i = 0; i < N; i++) $display("ASSERT audio.%s active %0d fail %0d", NAME[i], active[i], fail[i]);
endmodule

// ---------------------------------------------------------------------- the decimator
module sva_pdm #(parameter int unsigned CLKDIV = 15) (
    input logic clk_i, input logic rst_ni, input logic en_i, input logic pdm_clk_o, input logic smp_valid_o
);
  localparam int N = 3;
  longint unsigned active[N] = '{default: 0}, fail[N] = '{default: 0};
  localparam string NAME[N] = '{"pdm_clock_period", "valid_is_a_pulse", "pdm_clock_duty_40_60"};
  longint unsigned now = 0, t_rise = 0, t_fall = 0;
  bit have = 1'b0, have_f = 1'b0;
  logic pc_q = 1'b0, v_q = 1'b0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      now <= 0; have <= 1'b0; have_f <= 1'b0; pc_q <= 1'b0; v_q <= 1'b0;
    end else begin
      now <= now + 1;
      pc_q <= pdm_clk_o;
      v_q <= smp_valid_o;
      if (!en_i) begin have <= 1'b0; have_f <= 1'b0; end
      if (en_i && pdm_clk_o && !pc_q) begin
        if (have) `CHK(0, now - t_rise == longint'(CLKDIV), "PDM clock period")
        t_rise <= now;
        have <= 1'b1;
      end
      if (en_i && !pdm_clk_o && pc_q) begin
        if (have) `CHK(2, (now - t_rise) * 10 >= 4 * CLKDIV && (now - t_rise) * 10 <= 6 * CLKDIV, "PDM clock outside 40-60 % duty")
        t_fall <= now;
        have_f <= 1'b1;
      end
      if (smp_valid_o || v_q) `CHK(1, !(smp_valid_o && v_q), "sample valid for two cycles")
    end
  end
  final for (int i = 0; i < N; i++) $display("ASSERT pdm.%s active %0d fail %0d", NAME[i], active[i], fail[i]);
endmodule

// ---------------------------------------------------------------------- the endpoint: bus and ownership
module sva_endpoint (
    input logic clk_i, input logic rst_ni, input logic req, input logic gnt, input logic rvalid, input logic go, input logic fe_owns,
    input logic seq_busy, input logic fe_pass, input logic seq_pass, input logic fe_busy, input logic aud_en, input logic irq_o,
    input logic [31:0] irq_en, input logic done_sticky, input logic fe_done_sticky, input logic seq_done_sticky, input logic aud_pend
);
  localparam int N = 7;
  longint unsigned active[N] = '{default: 0}, fail[N] = '{default: 0};
  localparam string NAME[N] = '{"rvalid_follows_gnt", "no_gnt_without_req", "one_engine_master_seq_fe", "cpu_go_not_on_owned_engine",
                                "irq_only_when_enabled", "irq_when_enabled_and_pending", "seq_fe_not_with_cim_fe"};
  logic gnt_q = 1'b0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) gnt_q <= 1'b0;
    else begin
      gnt_q <= gnt;
      if (gnt_q || rvalid) `CHK(0, rvalid == gnt_q, "rvalid is not the cycle after gnt")
      if (gnt) `CHK(1, req, "gnt without req")
      if (seq_pass || fe_pass) `CHK(2, !(seq_pass && fe_pass && fe_owns), "two masters start the engine")
      if (gnt && go) `CHK(3, !fe_owns && !seq_busy, "CPU go while the engine has an owner")
      begin
        logic pend = (irq_en[0] && done_sticky) || (irq_en[1] && fe_done_sticky) || (irq_en[2] && seq_done_sticky) || (irq_en[3] && aud_pend);
        if (irq_o) `CHK(4, pend, "irq without an enabled pending source")
        if (pend) `CHK(5, irq_o, "enabled pending source without irq")
      end
      if (seq_busy) `CHK(6, !fe_owns, "the sequencer runs while cim_fe owns the engine")
    end
  end
  final for (int i = 0; i < N; i++) $display("ASSERT endpoint.%s active %0d fail %0d", NAME[i], active[i], fail[i]);
endmodule

`ifdef SVA_ENGINE
bind cim_engine sva_engine #(.TW(TW)) u_sva (.clk_i, .rst_ni, .start_i, .ready_o, .done_o, .busy_o, .xbits_i, .xbank_i, .tile_i,
    .x_we_i, .xw_we_i, .xw_bank_i, .xw_mask_i, .w_we_i, .w_tile_i);
`endif
`ifdef SVA_SOC
bind cim_seq sva_seq u_sva (.clk_i, .rst_ni, .busy_o, .done_o, .err_o, .lay, .pop, .tq_cnt, .d_rem, .issue, .g2_v, .g2_b, .bbusy,
    .bcnt0(bcnt[0]), .bcnt1(bcnt[1]), .g_req, .g_b, .g_k, .act_we, .act_wa, .d_p, .d_c, .d_rem_nz(d_rem != 5'd0), .w_row, .w_new, .hptr,
    .ws(4'(ws)), .wp_on, .net_on, .go_o, .eng_ready_i, .g_p, .q0_v, .q4_v);
`endif
`ifdef SVA_AUDIO
bind cim_audio sva_audio #(.CLKDIV(CLKDIV)) u_sva (.clk_i, .rst_ni, .en_i, .sv, .hop_end, .hs, .hc(3'(hc)), .kcnt, .net_km1_i, .ndec_o, .nhop_o,
    .pend_o, .fe_go_o, .fe_ok_i, .seq_go_o, .seq_ok_i, .seq_done_i, .fe_done_i, .hc_full, .frow, .wptr);
`endif
`ifdef SVA_PDM
bind pdm_dec sva_pdm #(.CLKDIV(CLKDIV)) u_sva (.clk_i, .rst_ni, .en_i, .pdm_clk_o, .smp_valid_o);
`endif
`ifdef SVA_SOC
bind cim_fe sva_frontend u_sva (.clk_i, .rst_ni, .start_i, .busy_o, .done_o, .samp_addr_o, .base_q);
bind mfcc_ded sva_frontend u_sva (.clk_i, .rst_ni, .start_i, .busy_o, .done_o, .samp_addr_o, .base_q);
bind cim_seq cov_seq u_cov (.clk_i, .q3_v, .q3_r, .q3_e, .zo((lay == 4'd9) ? zfc_q : zact_q), .rb_v, .rq_t);
bind cim_endpoint sva_endpoint u_sva (.clk_i, .rst_ni, .req(obi_req_i.req), .gnt, .rvalid(obi_rsp_o.rvalid), .go, .fe_owns, .seq_busy, .fe_pass, .seq_pass,
    .fe_busy, .aud_en, .irq_o, .irq_en(32'(irq_en)), .done_sticky, .fe_done_sticky, .seq_done_sticky, .aud_pend);
`endif
