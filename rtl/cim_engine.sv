// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_engine: FPGA emulation of a bit-serial digital CIM built from 64-row x 16-column int8
// macros (one macro, called a tile, is active per pass). See docs/ENGINE_SPEC.md.
//
// Timing contract (cycles are clock edges at which signals are sampled):
//   a pass accepted at edge t0 (start_i && ready_o) asserts done_o at edge t0 + D + 5,
//   D = max(8, b), b = effective input width (1..XB); the next pass may start at edge t0 + D.
// Results: res_o[n] = (accum ? previous : 0) + sum_r W[tile][r][n] * sext_b(x[r]), x being the
// input bank named at the start of the pass (two banks of 64 rows, so the next input vector can
// be written while a pass runs).
// The emulation computes 8 rows x 16 columns per cycle (128 multipliers) from BRAM-resident
// weights; results are bit-exact, the SoC-visible timing is that of the modelled CIM, and no
// CIM energy or area is emulated.

module cim_engine #(
    parameter int unsigned TILES = 64,               // resident tiles (1 KiB of weights each)
    parameter int unsigned XB    = 24,               // maximum input width
    parameter int unsigned ACCW  = 48,               // accumulator width
    localparam int unsigned TW   = $clog2(TILES)
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,
    // weight write: w_data_i[8k+7:8k] is the weight of column 4*w_word_i+k of row w_row_i
    input  logic                   w_we_i,
    input  logic [TW-1:0]          w_tile_i,
    input  logic [5:0]             w_row_i,
    input  logic [1:0]             w_word_i,
    input  logic [31:0]            w_data_i,
    // input vector register file (the pass's bank must stay stable for 9 edges after the pass is
    // accepted). Port 1 writes row x_idx_i of bank 0 (CPU, front-end). Port 2 writes up to 8
    // consecutive rows xw_row_i .. xw_row_i + 7 of bank xw_bank_i at once (layer sequencer):
    // xw_data_i[l] is the row in lane l (row % 8 == l), xw_mask_i[l] enables it. Port 2 has priority.
    input  logic                   x_we_i,
    input  logic [5:0]             x_idx_i,
    input  logic [XB-1:0]          x_data_i,
    input  logic                   xw_we_i,
    input  logic                   xw_bank_i,
    input  logic [5:0]             xw_row_i,
    input  logic [7:0]             xw_mask_i,
    input  logic [XB-1:0]          xw_data_i [8],
    // pass interface
    input  logic                   start_i,
    input  logic [TW-1:0]          tile_i,
    input  logic [4:0]             xbits_i,           // 0 is treated as 1, values > XB as XB
    input  logic                   accum_i,
    input  logic                   xbank_i,          // input bank the pass reads
    output logic                   ready_o,
    output logic                   done_o,
    output logic                   busy_o,           // a pass is in flight (done not yet given)
    output logic signed [ACCW-1:0] res_o [16]
);

  localparam int unsigned N     = 16;                // columns per tile
  localparam int unsigned LANES = 8;                 // rows per cycle
  localparam int unsigned AW    = TW + 3;            // lane address {tile, row group}
  localparam int unsigned PW    = XB + 8;            // product width
  localparam int unsigned SW    = PW + 3;            // width of a sum of 8 products

  typedef struct packed {
    logic       first;                               // first row group of a pass
    logic       last;                                // last row group of a pass
    logic       accum;
    logic       bank;                                // input bank
    logic [4:0] xb;                                  // effective input width
    logic [4:0] dm8;                                 // D - 8
    logic [2:0] grp;                                 // row group (rows 8*grp .. 8*grp+7)
  } meta_t;

  function automatic logic [4:0] eff_bits(input logic [4:0] b);
    if (b == 5'd0) return 5'd1;
    if (b > 5'(XB)) return 5'(XB);
    return b;
  endfunction

  function automatic logic [4:0] pass_d(input logic [4:0] b);
    return (b > 5'd8) ? b : 5'd8;
  endfunction

  // sign-extend the low b bits of v (what a b-plane bit-serial macro sees)
  function automatic logic signed [XB-1:0] sext_b(input logic [XB-1:0] v, input logic [4:0] b);
    logic [XB-1:0] r;
    for (int k = 0; k < int'(XB); k++) r[k] = (k < int'(b)) ? v[k] : v[b-5'd1];
    return r;
  endfunction

  // ------------------------------------------------------------------ issue control
  logic          issuing;
  logic [2:0]    grp;
  logic [TW-1:0] tile_q;
  logic [4:0]    xb_q, dm8_q, gap;
  logic          acc_q, xbank_q;
  logic [4:0]    b_in, d_in;

  assign b_in    = eff_bits(xbits_i);
  assign d_in    = pass_d(b_in);
  assign ready_o = (gap == 5'd0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      issuing <= 1'b0;
      grp     <= 3'd0;
      tile_q  <= '0;
      xb_q    <= 5'd8;
      dm8_q   <= 5'd0;
      acc_q   <= 1'b0;
      xbank_q <= 1'b0;
      gap     <= 5'd0;
    end else begin
      if (gap != 5'd0) gap <= gap - 5'd1;
      if (issuing) begin
        grp <= grp + 3'd1;
        if (grp == 3'd7) issuing <= 1'b0;
      end
      if (start_i && ready_o) begin                  // overrides the end of the previous pass
        issuing <= 1'b1;
        grp     <= 3'd0;
        tile_q  <= tile_i;
        xb_q    <= b_in;
        dm8_q   <= d_in - 5'd8;
        acc_q   <= accum_i;
        xbank_q <= xbank_i;
        gap     <= d_in - 5'd1;
      end
    end
  end

  // ------------------------------------------------------------------ weight lanes (BRAM)
  // Lane l holds rows 8g+l of every tile, split into four 32-bit word banks (columns 4q..4q+3)
  // so that every write is full width: Vivado maps 32-bit partial writes of a 128-bit word to
  // LUTRAM. One bank is a 512 x 32 RAMB18; 8 lanes x 4 banks = 32 RAMB18. The read address is
  // registered (stage 1) rather than the read data: a data register here is absorbed into the
  // DSP B pipeline, which leaves an asynchronous read that only LUTRAM can implement.
  logic [AW-1:0] raddr, waddr;
  logic [31:0]   wword [LANES][4];                   // bank outputs (stage 1)
  assign raddr = {tile_q, grp};
  assign waddr = {w_tile_i, w_row_i[5:3]};

  for (genvar l = 0; l < int'(LANES); l++) begin : g_lane
    for (genvar q = 0; q < 4; q++) begin : g_word
      (* ram_style = "block" *) logic [31:0] mem [TILES*8];
      logic [AW-1:0] ra_q;
      always_ff @(posedge clk_i) begin
        if (w_we_i && (w_row_i[2:0] == 3'(l)) && (w_word_i == 2'(q))) mem[waddr] <= w_data_i;
        if (issuing) ra_q <= raddr;
      end
      assign wword[l][q] = mem[ra_q];
    end
  end

  // ------------------------------------------------------------------ pipeline
  logic  v1, v2, v3, v4;
  // later stages carry the whole record; xb and grp are only consumed by stage 2
  /* verilator lint_off UNUSEDSIGNAL */
  meta_t m1, m2, m3, m4;
  /* verilator lint_on UNUSEDSIGNAL */
  logic signed [XB-1:0]  xop [LANES];                 // stage 2 operands
  logic [N*8-1:0]        wq  [LANES];
  (* use_dsp = "yes" *) logic signed [PW-1:0] prod [LANES][N];  // stage 3
  logic signed [SW-1:0]  s_lo [N], s_hi [N];         // stage 4 partial sums
  logic signed [ACCW-1:0] acc [N];
  logic signed [ACCW-1:0] acc_next [N];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
      m1 <= '0;   m2 <= '0;   m3 <= '0;   m4 <= '0;
    end else begin
      v1 <= issuing;
      m1 <= '{first: (grp == 3'd0), last: (grp == 3'd7), accum: acc_q, bank: xbank_q, xb: xb_q, dm8: dm8_q,
              grp: grp};
      v2 <= v1; m2 <= m1;
      v3 <= v2; m3 <= m2;
      v4 <= v3; m4 <= m3;
    end
  end

  // ------------------------------------------------------------------ input register file
  // Row r of bank k is entry {k, r[5:3]} of lane r[2:0]: one 16-entry LUT-RAM per lane, one write
  // and one read per cycle. A port-2 write of rows row0 .. row0 + 7 gives lane l the row
  // row0 + ((l - row0) mod 8), i.e. entry row0[5:3], or row0[5:3] + 1 for the lanes that wrap.
  logic [XB-1:0] xrd [LANES];                        // stage-2 read of lane l
  for (genvar l = 0; l < int'(LANES); l++) begin : g_xlane
    (* ram_style = "distributed" *) logic [XB-1:0] xm [16];
    logic          xwe;
    logic [3:0]    xwa;
    logic [XB-1:0] xwd;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [5:0]    xwr;                              // port-2 row of this lane (its lane bits are l)
    /* verilator lint_on UNUSEDSIGNAL */
    assign xwr = xw_row_i + {3'd0, 3'(l) - xw_row_i[2:0]};
    always_comb begin
      if (xw_we_i) begin
        xwe = xw_mask_i[l];
        xwa = {xw_bank_i, xwr[5:3]};
        xwd = xw_data_i[l];
      end else begin
        xwe = x_we_i && (x_idx_i[2:0] == 3'(l));
        xwa = {1'b0, x_idx_i[5:3]};
        xwd = x_data_i;
      end
    end
    always_ff @(posedge clk_i) begin
      if (xwe) xm[xwa] <= xwd;
    end
`ifdef CIM_ZERO_INIT                                 // FPGA LUT-RAM starts at 0; lets a 4-state simulator
    initial for (int i = 0; i < 16; i++) xm[i] = '0;  // model that (rows never written times 0 weights)
`endif
    assign xrd[l] = xm[{m1.bank, m1.grp}];
  end

  // stage 2: operands (row 8*grp + l uses lane l)
  always_ff @(posedge clk_i) begin
    for (int l = 0; l < int'(LANES); l++) begin
      xop[l] <= sext_b(xrd[l], m1.xb);
      wq[l]  <= {wword[l][3], wword[l][2], wword[l][1], wword[l][0]};
    end
  end

  // stage 3: 128 products
  always_ff @(posedge clk_i) begin
    for (int l = 0; l < int'(LANES); l++)
      for (int n = 0; n < int'(N); n++)
        prod[l][n] <= PW'(xop[l]) * PW'($signed(wq[l][n*8 +: 8]));
  end

  // stage 4: partial adder trees (4 + 4 products per column)
  always_ff @(posedge clk_i) begin
    for (int n = 0; n < int'(N); n++) begin
      s_lo[n] <= SW'(prod[0][n]) + SW'(prod[1][n]) + SW'(prod[2][n]) + SW'(prod[3][n]);
      s_hi[n] <= SW'(prod[4][n]) + SW'(prod[5][n]) + SW'(prod[6][n]) + SW'(prod[7][n]);
    end
  end

  // stage 5: accumulate
  always_comb begin
    for (int n = 0; n < int'(N); n++)
      acc_next[n] = ((m4.first && !m4.accum) ? ACCW'(0) : acc[n]) + ACCW'(s_lo[n]) + ACCW'(s_hi[n]);
  end

  logic       pending;
  logic [4:0] wait_cnt;
  assign done_o = pending && (wait_cnt == 5'd0);
  assign busy_o = issuing || v1 || v2 || v3 || v4 || pending;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int n = 0; n < int'(N); n++) begin
        acc[n]   <= '0;
        res_o[n] <= '0;
      end
      pending  <= 1'b0;
      wait_cnt <= 5'd0;
    end else begin
      if (pending) begin
        if (wait_cnt == 5'd0) pending <= 1'b0;
        else wait_cnt <= wait_cnt - 5'd1;
      end
      if (v4) begin
        for (int n = 0; n < int'(N); n++) acc[n] <= acc_next[n];
        if (m4.last) begin
          for (int n = 0; n < int'(N); n++) res_o[n] <= acc_next[n];
          pending  <= 1'b1;
          wait_cnt <= m4.dm8;
        end
      end
    end
  end

endmodule
