// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Testbench for the hardware front-end (cim_fe, or with DED = 1 the B3 dedicated front-end mfcc_ded,
// inside cim_endpoint) through the OBI port: loads
// the front-end tiles, writes each frame's 480 samples into the ring at a rotating base (ring
// wrap-around included), runs the front-end and checks lraw[40] bit-exactly against the golden
// model (vectors from model/gen_fe_vectors.py). Also checks the CPU / front-end interplay: a
// front-end go waits for an in-flight CPU pass (cim_fe), DONE_CNT and the CPU results are left
// alone, and a CPU X write is held while cim_fe runs (B3 leaves the engine to the CPU).
// Vectors: tb/fe_vec (cim_fe) or tb/ded_vec (B3).
`timescale 1ns / 1ps

module tb_cim_fe #(
    parameter bit DED = 1'b0
);
  import xheep_obi_pkg::*;
  localparam int MAXF = 512;
  localparam logic [31:0] BASE = 32'hC004_0000;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq;

  cim_endpoint #(.TILES(128), .FE_DED(DED)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));

  logic [31:0] tiles [8 * 256];
  logic [15:0] samples [MAXF * 480];
  logic [31:0] lraw_exp [MAXF * 40];
  int          nframes, errors = 0;
  longint unsigned edge_cnt = 0;
  always @(posedge clk) edge_cnt <= edge_cnt + 1;

  task automatic tick(); @(posedge clk); #1; endtask

  // OBI master: hold req until a clock edge samples req && gnt, then wait for rvalid
  task automatic obi(input bit we, input logic [31:0] addr, input logic [31:0] wdata, output logic [31:0] rdata,
                     output longint unsigned t_gnt);
    req.req = 1'b1; req.we = we; req.be = 4'hF; req.addr = addr; req.wdata = wdata;
    do @(posedge clk); while (!rsp.gnt);
    t_gnt = edge_cnt;
    #1 req.req = 1'b0;
    while (!rsp.rvalid) tick();
    rdata = rsp.rdata;
    tick();
  endtask

  task automatic wr(input logic [31:0] addr, input logic [31:0] d);
    logic [31:0] dummy; longint unsigned t;
    obi(1'b1, addr, d, dummy, t);
  endtask

  task automatic rd(input logic [31:0] addr, output logic [31:0] d);
    longint unsigned t;
    obi(1'b0, addr, 32'd0, d, t);
  endtask

  function automatic logic signed [7:0] wt(int t, int r, int n);   // weight of tile t, row r, column n
    logic [31:0] w = tiles[t * 256 + r * 4 + n / 4];
    return w[8 * (n % 4) +: 8];
  endfunction

  function automatic longint sext(longint v, int b);
    longint m = v & ((longint'(1) << b) - 1);
    if (m & (longint'(1) << (b - 1))) m -= (longint'(1) << b);
    return m;
  endfunction

  task automatic write_frame(int f, int base);
    for (int i = 0; i < 480; i++) wr(BASE + 32'h2000 + 4 * ((base + i) % 1024), {16'd0, samples[f * 480 + i]});
  endtask

  task automatic check_lraw(int f);
    logic [31:0] s;
    for (int m = 0; m < 40; m++) begin
      rd(BASE + 32'h3000 + 4 * m, s);
      if (s !== lraw_exp[f * 40 + m]) begin
        if (errors < 10) $display("frame %0d mel %0d: lraw %08h expected %08h", f, m, s, lraw_exp[f * 40 + m]);
        errors++;
      end
    end
  endtask

  initial begin
    int fd, bad_frames = 0;
    logic [31:0] s, c, lo, hi, d;
    longint unsigned t0, t_cpu, t_fe, t_x, cyc, cyc_min = '1, cyc_max = 0, cyc_sum = 0;
    longint x [64];
    longint ref_res [16];
    string vec = DED ? "tb/ded_vec" : "tb/fe_vec";
    $readmemh({vec, "/tiles.hex"}, tiles);
    $readmemh({vec, "/samples.hex"}, samples);
    $readmemh({vec, "/lraw.hex"}, lraw_exp);
    fd = $fopen({vec, "/nframes.txt"}, "r");
    if (fd == 0 || $fscanf(fd, "%d", nframes) != 1) $fatal(1, "cannot read %s/nframes.txt", vec);
    $fclose(fd);
    req = '0;
    repeat (3) @(posedge clk);
    #1 rst_n = 1'b1;
    tick();
    for (int i = 0; i < 8 * 256; i++) wr(BASE + 32'h2_0000 + 4 * i, tiles[i]);

    // 1) interplay: CPU pass in flight, then a front-end go, then a CPU X write while it runs
    write_frame(0, 0);
    for (int r = 0; r < 64; r++) begin
      x[r] = longint'($urandom() & 32'h3FFF);
      wr(BASE + 32'h1000 + 4 * r, 32'(x[r]));
    end
    for (int n = 0; n < 16; n++) begin
      ref_res[n] = 0;
      for (int r = 0; r < 64; r++) ref_res[n] += longint'(wt(6, r, n)) * sext(x[r], 14);
    end
    obi(1'b1, BASE + 32'h0, 32'h8000_0000 | (14 << 8) | 6, d, t_cpu);        // CPU pass, tile 6, b = 14
    obi(1'b1, BASE + 32'h10, 32'h8000_0000, d, t_fe);                        // front-end, base 0
    obi(1'b1, BASE + 32'h1000, 32'h0, d, t_x);                               // held while busy
    rd(BASE + 32'h14, s);
    if (!DED) begin                                                          // cim_fe owns the engine
      if (t_fe < t_cpu + 14 + 5) begin $display("front-end go granted %0d edges after the CPU pass", t_fe - t_cpu); errors++; end
      if (s[0] || !s[1]) begin $display("CPU X write completed while the front-end was busy (FE_STATUS %h)", s); errors++; end
    end else begin                                                           // B3 leaves the engine to the CPU
      if (t_fe >= t_cpu + 14) begin $display("B3 go held %0d edges behind a CPU pass", t_fe - t_cpu); errors++; end
      if (!s[0]) begin $display("B3 finished before the CPU X write (FE_STATUS %h)", s); errors++; end
      do rd(BASE + 32'h14, s); while (!s[1]);
    end
    rd(BASE + 32'h0C, c);
    if (c != 1) begin $display("DONE_CNT %0d != 1 after a front-end run", c); errors++; end
    for (int n = 0; n < 16; n++) begin
      rd(BASE + 32'h40 + 4 * n, lo);
      rd(BASE + 32'h80 + 4 * n, hi);
      if ({hi[15:0], lo} !== ref_res[n][47:0]) begin $display("CPU result col %0d changed: %h_%h", n, hi, lo); errors++; end
    end
    check_lraw(0);
    $display("interplay: front-end go held %0d edges behind the CPU pass, X write held %0d edges", t_fe - t_cpu, t_x - t_fe);

    // 2) every frame, rotating ring base (wraps around 1024)
    for (int f = 0; f < nframes; f++) begin
      int base = (f * 320 + 7) % 1024;
      int e0 = errors;
      write_frame(f, base);
      t0 = edge_cnt;
      wr(BASE + 32'h10, 32'h8000_0000 | 32'(base));
      do rd(BASE + 32'h14, s); while (!s[1]);
      cyc = edge_cnt - t0;
      cyc_sum += cyc;
      if (cyc < cyc_min) cyc_min = cyc;
      if (cyc > cyc_max) cyc_max = cyc;
      check_lraw(f);
      if (errors != e0) bad_frames++;
    end
    $display("RESULT: %0d frames, %0d bad frames, %0d errors; front-end go to done %0d..%0d edges (mean %0d)",
             nframes, bad_frames, errors, cyc_min, cyc_max, cyc_sum / longint'(nframes));
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end
endmodule
