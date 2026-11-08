// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Self-checking testbench for cim_engine: bit-exact results against an integer reference model
// and the exact timing contract (done at t0 + D + 5, next start accepted at t0 + D). The last phase
// alternates passes between the two input banks and rewrites the idle bank through the wide port
// (random chunks and masks) while each pass runs, as the layer sequencer does.
`timescale 1ns / 1ps

module tb_cim_engine;
  localparam int TILES = 128, XB = 24, ACCW = 48;
  localparam int TW = $clog2(TILES);

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  logic                  w_we = 1'b0;
  logic [TW-1:0]         w_tile = '0;
  logic [5:0]            w_row = '0;
  logic [1:0]            w_word = '0;
  logic [31:0]           w_data = '0;
  logic                  x_we = 1'b0;
  logic [5:0]            x_idx = '0;
  logic [XB-1:0]         x_data = '0;
  logic                  xw_we = 1'b0, xw_bank = 1'b0, xbank = 1'b0;
  logic [5:0]            xw_row = '0;
  logic [7:0]            xw_mask = '0;
  logic [XB-1:0]         xw_data [8];
  logic                  start = 1'b0, accum = 1'b0;
  logic [TW-1:0]         tile = '0;
  logic [4:0]            xbits = 5'd8;
  logic                  ready, done;
  logic signed [ACCW-1:0] res [16];

  cim_engine #(.TILES(TILES), .XB(XB), .ACCW(ACCW)) dut (
      .clk_i(clk), .rst_ni(rst_n),
      .w_we_i(w_we), .w_tile_i(w_tile), .w_row_i(w_row), .w_word_i(w_word), .w_data_i(w_data),
      .x_we_i(x_we), .x_idx_i(x_idx), .x_data_i(x_data),
      .xw_we_i(xw_we), .xw_bank_i(xw_bank), .xw_row_i(xw_row), .xw_mask_i(xw_mask), .xw_data_i(xw_data),
      .start_i(start), .tile_i(tile), .xbits_i(xbits), .accum_i(accum), .xbank_i(xbank),
      .ready_o(ready), .done_o(done), .busy_o(), .res_o(res));

  // ---------------------------------------------------------------- reference model state
  byte          W   [TILES][64][16];
  int unsigned  X   [2][64];                      // raw 24-bit patterns written to the DUT, per bank
  longint       ACC [16];

  typedef struct {
    longint unsigned t0;
    int              d;
    longint          exp_res [16];
  } exp_t;
  exp_t   expq[$];
  longint unsigned edge_cnt = 0;
  int     errors = 0, timing_errors = 0, checked = 0, ready_errors = 0, ready_checked = 0;
  longint unsigned prev_t0 = 0;
  int     prev_d = 0;

  always @(posedge clk) edge_cnt <= edge_cnt + 1;

  // monitor: every done (sampled at an edge) is matched, in order, with the oldest expectation
  always @(posedge clk) begin
    if (rst_n && done) begin
      exp_t e;
      if (expq.size() == 0) begin
        $display("ERROR: done without an outstanding pass at edge %0d", edge_cnt);
        errors++;
      end else begin
        e = expq.pop_front();
        if (edge_cnt != e.t0 + longint'(e.d) + 5) begin
          if (timing_errors < 10)
            $display("TIMING: done at edge %0d, expected %0d (t0 %0d, D %0d)", edge_cnt, e.t0 + e.d + 5, e.t0, e.d);
          timing_errors++;
        end
        for (int n = 0; n < 16; n++) begin
          if (res[n] !== ACCW'(e.exp_res[n])) begin
            if (errors < 10) $display("MISMATCH col %0d: got %0d expected %0d", n, res[n], e.exp_res[n]);
            errors++;
          end
        end
        checked++;
      end
    end
  end

  function automatic longint sext(int unsigned v, int b);
    longint m;
    m = longint'(v) & ((longint'(1) << b) - 1);
    if (m & (longint'(1) << (b - 1))) m -= (longint'(1) << b);
    return m;
  endfunction

  task automatic tick();
    @(posedge clk); #1;
  endtask

  task automatic load_tile(int t);
    for (int r = 0; r < 64; r++)
      for (int q = 0; q < 4; q++) begin
        w_we = 1'b1; w_tile = TW'(t); w_row = 6'(r); w_word = 2'(q);
        w_data = {W[t][r][4*q+3], W[t][r][4*q+2], W[t][r][4*q+1], W[t][r][4*q]};
        tick();
      end
    w_we = 1'b0;
  endtask

  task automatic load_x();
    for (int r = 0; r < 64; r++) begin
      x_we = 1'b1; x_idx = 6'(r); x_data = XB'(X[0][r]);
      tick();
    end
    x_we = 1'b0;
  endtask

  // port 2: rows row0 .. row0 + 7 of a bank, lanes in mask (rows above 63 must be masked off)
  task automatic wide_write(bit bank, int row0, logic [7:0] mask, int b);
    xw_we = 1'b1; xw_bank = bank; xw_row = 6'(row0); xw_mask = mask;
    for (int l = 0; l < 8; l++) begin
      int r = row0 + ((l - row0) & 7);
      int unsigned u = $urandom();
      u = ($urandom_range(0, 9) == 0) ? (u & 24'hFFFFFF) : (u & ((32'd1 << b) - 1));
      xw_data[l] = XB'(u);
      if (mask[l]) X[bank][r] = u;
    end
    tick();
    xw_we = 1'b0;
  endtask

  // rewrite a whole bank in random chunks of 1..8 consecutive rows, some with holes
  task automatic wide_fill(bit bank, int b);
    int r = 0;
    while (r < 64) begin
      int n = $urandom_range(1, 8);
      logic [7:0] m = '0;
      if (n > 64 - r) n = 64 - r;
      for (int i = 0; i < n; i++) m[(r + i) % 8] = 1'b1;
      if ($urandom_range(0, 3) == 0) wide_write(bank, r, m & 8'($urandom()), b);   // partial first
      wide_write(bank, r, m, b);
      r += n;
    end
  endtask

  // issue one pass as soon as ready; push its expectation (reference model update)
  task automatic issue(int t, int b, bit acc, bit bank = 1'b0);
    exp_t e;
    int beff, d;
    bit waited;
    beff = (b == 0) ? 1 : (b > XB ? XB : b);
    d = (beff > 8) ? beff : 8;
    waited = !ready;
    while (!ready) tick();
    start = 1'b1; tile = TW'(t); xbits = 5'(b); accum = acc; xbank = bank;
    for (int n = 0; n < 16; n++) begin
      longint s = 0;
      for (int r = 0; r < 64; r++) s += longint'(W[t][r][n]) * sext(X[bank][r], beff);
      ACC[n] = (acc ? ACC[n] : 0) + s;
      e.exp_res[n] = ACC[n];
    end
    @(posedge clk);
    e.t0 = edge_cnt;                                // pre-edge value, same convention as the monitor
    e.d = d;
    if (waited) begin                               // back-to-back: accepted exactly D after the last start
      ready_checked++;
      if (e.t0 != prev_t0 + longint'(prev_d)) begin
        if (ready_errors < 10) $display("READY: accepted at %0d, expected %0d", e.t0, prev_t0 + prev_d);
        ready_errors++;
      end
    end
    prev_t0 = e.t0;
    prev_d = d;
    expq.push_back(e);
    #1 start = 1'b0;
  endtask

  task automatic wait_idle();
    while (expq.size() != 0) tick();
    repeat (3) tick();
  endtask

  function automatic void random_x(int b, bit allow_wide);
    for (int r = 0; r < 64; r++) begin
      int unsigned u = $urandom();
      if (allow_wide && ($urandom_range(0, 9) == 0)) X[0][r] = u & 24'hFFFFFF;  // out-of-range: tests sext
      else X[0][r] = u & ((b >= 32) ? 32'hFFFFFFFF : ((32'd1 << b) - 1));
    end
  endfunction

  initial begin
    int n_random = 2400;
    int widths [8] = '{1, 4, 8, 9, 12, 16, 20, 24};
    for (int t = 0; t < TILES; t++)
      for (int r = 0; r < 64; r++)
        for (int n = 0; n < 16; n++)
          case (t)
            0: W[t][r][n] = -8'sd128;
            1: W[t][r][n] = 8'sd127;
            2: W[t][r][n] = 8'sd0;
            3: W[t][r][n] = (r == n) ? 8'sd1 : 8'sd0;
            default: W[t][r][n] = byte'($urandom());
          endcase
    for (int n = 0; n < 16; n++) ACC[n] = 0;
    repeat (4) @(posedge clk);
    #1 rst_n = 1'b1;
    tick();
    for (int t = 0; t < TILES; t++) load_tile(t);
    $display("loaded %0d tiles", TILES);

    // directed: extremes and identity
    for (int r = 0; r < 64; r++) X[0][r] = 32'h800000;       // -2^23 at 24 bits
    load_x();
    issue(0, 24, 0); issue(1, 24, 0); issue(2, 24, 0); issue(3, 24, 0);
    wait_idle();
    for (int r = 0; r < 64; r++) X[0][r] = 32'(r * 1000 - 20000);
    load_x();
    issue(3, 24, 0); issue(3, 16, 0); issue(3, 8, 0);          // identity: x truncated to b bits
    wait_idle();
    // directed: accumulate chain over 6 passes
    random_x(16, 0); load_x();
    issue(5, 16, 0); for (int k = 0; k < 5; k++) issue(6 + k, 16, 1);
    wait_idle();
    // directed: input widths out of range: 0 acts as 1, and anything above XB as XB
    random_x(24, 0); load_x();
    issue(7, 0, 0); issue(8, 25, 0); issue(9, 31, 0); issue(10, 1, 0);
    wait_idle();
    $display("directed: %0d passes checked, %0d errors, %0d timing errors", checked, errors, timing_errors);

    // random: new vector, then a burst of back-to-back passes on random tiles
    while (checked < n_random) begin
      int b = widths[$urandom_range(0, 7)];
      random_x(b, 1);
      load_x();
      repeat ($urandom_range(1, 6)) issue($urandom_range(0, TILES - 1), b, $urandom_range(0, 1));
      wait_idle();
    end
    wait_idle();
    $display("random: %0d passes checked, %0d errors", checked, errors);

    // double-buffered: pass k reads bank k % 2 while the other bank is rewritten through port 2;
    // every few passes port 1 rewrites bank 0 during a bank-1 pass instead
    wide_fill(1'b0, 9); wide_fill(1'b1, 9);
    for (int k = 0; k < 600; k++) begin
      bit bank = 1'(k % 2);
      int b = ($urandom_range(0, 1) == 0) ? 9 : widths[$urandom_range(0, 7)];
      issue($urandom_range(0, TILES - 1), b, $urandom_range(0, 3) == 0, bank);
      if (bank && ($urandom_range(0, 4) == 0)) begin
        random_x(9, 1); load_x();
      end else wide_fill(!bank, 9);
    end
    wait_idle();
    $display("RESULT: %0d passes checked, %0d value mismatches, %0d done-timing errors; %0d back-to-back starts checked, %0d issue-interval errors",
             checked, errors, timing_errors, ready_checked, ready_errors);
    if (errors == 0 && timing_errors == 0 && ready_errors == 0 && ready_checked > 0) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
