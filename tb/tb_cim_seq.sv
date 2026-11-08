// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// Self-checking testbench for the layer sequencer in cim_endpoint (R6b). Loads the 61 network
// tiles and the parameter memory over OBI, as firmware does, then checks:
//   1. stop-after-layer runs on clip 0: every output of L0..L8 (125 positions x 64 channels),
//      read through the ACT window, against dscnn_int (= TFLite reference kernels);
//   2. full runs on 20 clips: int8 logits and class against the golden model, and the cycles per
//      inference (SEQ go granted -> interrupt);
//   3. FEAT as a ring: a clip written rotated by 17 rows and run with frame 0 at row 17;
//   4. interplay: CPU X/go/FEAT accesses during a run are held until it ends; a front-end go is
//      held (cim_fe) or runs concurrently (B3, DED = 1); a SEQ go waits for the front-end
//      (cim_fe) and for a CPU pass in flight; CPU passes work after a run;
//   5. synthetic requantisation parameters and zero points (shifts 0..15, power-of-two
//      multipliers, no ReLU clamp): every layer and the logits of clip 0;
//   6. the same with zero point -128 (the baseline S's sign gate is active) and acc + bias = -1, 0,
//      +1 at one position of every channel: every layer and the logits of clip 0.
// B1 (StreamFuse B in block 1, SEQ_B1): L1 never reaches ACT, so its outputs are taken from the
// sequencer's fusion-vector writes (a monitor; not in NETLIST runs); L2..L8 are in the other map.
// A monitor checks the engine contract throughout: no wide-port write to an X bank in the 8
// edges after a pass on that bank started (its reads run until 9 edges after the start).
// +define+NETLIST simulates a Vivado functional netlist of the endpoint instead of the RTL (no
// monitor); +define+QUICK loads only the conv1 and depthwise-0 tiles and checks only L0 and L1;
// +define+MEDIUM checks every layer and one full inference (for slow gate-level simulation);
// +define+ALLCLIPS runs every GSC test clip (tb/seq_vec_all, model/gen_seq_allclips.py) and checks only
// the logits and the class of each (sim/run_seq_allclips.sh).
// Vectors: python model/gen_seq_vectors.py.
`timescale 1ns / 1ps

module tb_cim_seq #(
    parameter bit DED = 1'b0,
    parameter bit S   = 1'b0,                           // the baseline S in the sequencer (SEQ_S)
    parameter bit B1  = 1'b0                            // with S: StreamFuse B in block 1 (SEQ_B1)
);
  import xheep_obi_pkg::*;
  localparam logic [31:0] BASE = 32'hC004_0000;
  localparam int NT = 61, T0 = 8;                    // network tiles 8..68

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  xheep_obi_req_t req;
  xheep_obi_rsp_t rsp;
  logic irq;

`ifdef NETLIST
  cim_endpoint dut (.clk_i(clk), .rst_ni(rst_n), .\obi_req_i[req] (req.req), .\obi_req_i[we] (req.we),
      .\obi_req_i[be] (req.be), .\obi_req_i[addr] (req.addr), .\obi_req_i[wdata] (req.wdata),
      .\obi_rsp_o[gnt] (rsp.gnt), .\obi_rsp_o[rvalid] (rsp.rvalid), .\obi_rsp_o[rdata] (rsp.rdata), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));
`else
  cim_endpoint #(.TILES(128), .FE_DED(DED), .SEQ_S(S), .SEQ_B1(B1)) dut (.clk_i(clk), .rst_ni(rst_n), .obi_req_i(req), .obi_rsp_o(rsp), .irq_o(irq),
      .pdm_clk_o(), .pdm_data_i(1'b0));
`endif
`ifdef QUICK
  localparam int NT_LOAD = 14, NSTOP = 2;
`elsif MEDIUM
  localparam int NT_LOAD = NT, NSTOP = 9;
`else
  localparam int NT_LOAD = NT, NSTOP = 9;
`endif

`ifdef ALLCLIPS
  localparam int NCMAX = 4890;
`else
  localparam int NCMAX = 20;
`endif
  logic [31:0] tiles [NT * 256];
  logic [31:0] params [640 * 3];
  logic [31:0] feat [NCMAX * 147];
  logic [7:0]  logits_exp [NCMAX * 12];
  logic [7:0]  acts_exp [9 * 8000];
  logic [31:0] meta [2 + NCMAX];
  logic [31:0] s_params [640 * 3];
  logic [7:0]  s_acts [9 * 8000];
  logic [7:0]  s_logits [12];
  logic [31:0] s_meta [2];
  logic [31:0] s2_params [640 * 3];
  logic [7:0]  s2_acts [9 * 8000];
  logic [7:0]  s2_logits [12];
  logic [31:0] s2_meta [2];
  int          nclips, errors = 0, contract_err = 0;
  longint unsigned edge_cnt = 0;
  longint unsigned cyc_min = '1, cyc_max = 0;
  always @(posedge clk) edge_cnt <= edge_cnt + 1;

  task automatic tick(); @(posedge clk); #1; endtask

  // B1: L1 as written into the fusion vector (position p, channel c), and the map of layer L
  logic [7:0] l1_cap [8000];
  bit         l1_got [8000];
`ifndef NETLIST
  if (B1) begin : g_l1mon
    always @(posedge clk) begin
      if (dut.i_seq.fv_we0) begin
        l1_cap[int'(dut.i_seq.q4_d.p) * 64 + int'(dut.i_seq.q4_d.c)] = dut.i_seq.q4_y;
        l1_got[int'(dut.i_seq.q4_d.p) * 64 + int'(dut.i_seq.q4_d.c)] = 1'b1;
      end
      if (dut.i_seq.fv_we1) begin
        l1_cap[int'(dut.i_seq.r4_d.p) * 64 + int'(dut.i_seq.r4_d.c)] = dut.i_seq.r4_y;
        l1_got[int'(dut.i_seq.r4_d.p) * 64 + int'(dut.i_seq.r4_d.c)] = 1'b1;
      end
    end
  end
`endif
  function automatic int map_of(int L);
    return (B1 && (L >= 2)) ? (L + 1) % 2 : L % 2;
  endfunction
  // layer L's output at (p, c): from ACT, or for B1's L1 from the monitor (NETLIST: not checked, ok = 0)
  task automatic get_act(int L, int p, int c, output logic [31:0] v, output bit ok);
    ok = 1'b1;
    if (B1 && (L == 1)) begin
`ifdef NETLIST
      ok = 1'b0; v = 32'd0;
`else
      v = l1_got[p * 64 + c] ? {{24{l1_cap[p * 64 + c][7]}}, l1_cap[p * 64 + c]} : 32'hDEAD_BEEF;
`endif
    end else rd(BASE + 32'h1_0000 + (32'(map_of(L)) << 15) + (32'(p) << 8) + (32'(c) << 2), v);
  endtask

`ifndef NETLIST
  // engine contract monitor (edge_cnt is the pre-edge count, as in obi())
  longint unsigned last_start [2] = '{0, 0};
  bit              started [2] = '{1'b0, 1'b0};
  always @(posedge clk) begin
    if (rst_n) begin
      if (dut.i_engine.xw_we_i && (dut.i_engine.xw_mask_i != 8'd0) && started[dut.i_engine.xw_bank_i] &&
          (edge_cnt > last_start[dut.i_engine.xw_bank_i]) && (edge_cnt <= last_start[dut.i_engine.xw_bank_i] + 8)) begin
        if (contract_err < 5) $display("CONTRACT: X bank %0d written at edge %0d, %0d edges after a pass on it started",
                                       dut.i_engine.xw_bank_i, edge_cnt, edge_cnt - last_start[dut.i_engine.xw_bank_i]);
        contract_err++;
      end
      if (dut.i_engine.start_i && dut.i_engine.ready_o) begin
        last_start[dut.i_engine.xbank_i] = edge_cnt;
        started[dut.i_engine.xbank_i] = 1'b1;
      end
    end
  end
`endif

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

  function automatic logic signed [7:0] wt(int t, int r, int n);   // weight of network tile t
    logic [31:0] w = tiles[(t - T0) * 256 + r * 4 + n / 4];
    return w[8 * (n % 4) +: 8];
  endfunction

  function automatic logic [31:0] seq_word(int stop, int frow);
    return 32'h8000_0000 | (32'(stop) << 24) | (32'(frow) << 16) | 32'(T0);
  endfunction

  task automatic load_feat(int clip, int frow);
    for (int r = 0; r < 49; r++)
      for (int w = 0; w < 3; w++)
        wr(BASE + 32'h4000 + (32'((r + frow) % 49) << 4) + (32'(w) << 2), feat[clip * 147 + r * 3 + w]);
  endtask

  // wait for the sequencer's interrupt, check its status, clear done
  task automatic seq_finish(input longint unsigned t_go, output longint unsigned cyc, output logic [31:0] st);
    while (!irq) tick();
    cyc = edge_cnt - t_go;
    rd(BASE + 32'h24, st);
    if (st[0] || !st[1] || st[2]) begin
      $display("ERROR: SEQ_STATUS 0x%08h after the interrupt (busy %0d, done %0d, error %0d)", st, st[0], st[1], st[2]);
      errors++;
    end
    wr(BASE + 32'h24, 32'h2);
    if (irq) begin $display("ERROR: irq still high after W1C"); errors++; end
  endtask

  task automatic check_logits(int clip, logic [31:0] st, string tag);
    logic [31:0] v;
    int bad = 0;
    for (int n = 0; n < 12; n++) begin
      rd(BASE + 32'h5000 + 32'(4 * n), v);
      if (v !== {{24{logits_exp[clip * 12 + n][7]}}, logits_exp[clip * 12 + n]}) bad++;
    end
    if (st[7:4] != meta[2 + clip][3:0]) bad++;
    if (bad != 0) begin
      $display("MISMATCH %s clip %0d: %0d logit/class errors (class %0d, expected %0d)", tag, clip, bad, st[7:4], meta[2 + clip]);
      errors += bad;
    end
  endtask

  task automatic run_clip(int clip, int frow, string tag);
    logic [31:0] st, d; longint unsigned t_go, cyc;
    load_feat(clip, frow);
    obi(1'b1, BASE + 32'h20, seq_word(0, frow), d, t_go);
    seq_finish(t_go, cyc, st);
    check_logits(clip, st, tag);
    if (cyc < cyc_min) cyc_min = cyc;
    if (cyc > cyc_max) cyc_max = cyc;
    $display("  %s clip %2d: class %2d (golden %2d), %0d cycles go -> interrupt", tag, clip, st[7:4], meta[2 + clip], cyc);
  endtask

  // CPU pass on network tile t with x[r] = 9-bit values, checked against the tile weights
  task automatic cpu_pass_check(int t, int seed);
    logic [31:0] lo, hi;
    int x [64];
    for (int r = 0; r < 64; r++) begin
      x[r] = ((r * 37 + seed * 11) % 511) - 255;
      wr(BASE + 32'h1000 + 32'(4 * r), 32'(x[r]));
    end
    wr(BASE + 32'h0, 32'h8000_0000 | (32'd9 << 8) | 32'(t));
    do rd(BASE + 32'h4, lo); while (!lo[1]);
    for (int n = 0; n < 16; n++) begin
      longint s = 0;
      for (int r = 0; r < 64; r++) s += longint'(wt(t, r, n)) * longint'(x[r]);
      rd(BASE + 32'h40 + 32'(4 * n), lo); rd(BASE + 32'h80 + 32'(4 * n), hi);
      if ({hi[15:0], lo} !== 48'(s)) begin
        $display("MISMATCH CPU pass tile %0d col %0d: got %0d expected %0d", t, n, $signed({hi[15:0], lo}), s);
        errors++;
      end
    end
  endtask

  initial begin
    logic [31:0] st, d;
    longint unsigned t_go, t_a, cyc;
    req = '0;
    $readmemh("tb/seq_vec/tiles.hex", tiles);
    $readmemh("tb/seq_vec/params.hex", params);
`ifdef ALLCLIPS
    $readmemh("tb/seq_vec_all/feat.hex", feat);
    $readmemh("tb/seq_vec_all/logits.hex", logits_exp);
    $readmemh("tb/seq_vec_all/meta.hex", meta);
`else
    $readmemh("tb/seq_vec/feat.hex", feat);
    $readmemh("tb/seq_vec/logits.hex", logits_exp);
    $readmemh("tb/seq_vec/meta.hex", meta);
`endif
    $readmemh("tb/seq_vec/acts.hex", acts_exp);
    $readmemh("tb/seq_vec/synth_params.hex", s_params);
    $readmemh("tb/seq_vec/synth_acts.hex", s_acts);
    $readmemh("tb/seq_vec/synth_logits.hex", s_logits);
    $readmemh("tb/seq_vec/synth_meta.hex", s_meta);
    $readmemh("tb/seq_vec/synth2_params.hex", s2_params);
    $readmemh("tb/seq_vec/synth2_acts.hex", s2_acts);
    $readmemh("tb/seq_vec/synth2_logits.hex", s2_logits);
    $readmemh("tb/seq_vec/synth2_meta.hex", s2_meta);
    nclips = int'(meta[0]);
`ifdef NETLIST
    #200;                                             // after the netlist's 100-ns global set/reset
`endif
    repeat (4) @(posedge clk);
    #1 rst_n = 1'b1;
    tick();

    for (int t = 0; t < NT_LOAD; t++)
      for (int r = 0; r < 64; r++)
        for (int q = 0; q < 4; q++)
          wr(BASE + 32'h2_0000 + (32'(T0 + t) << 10) + (32'(r) << 4) + (32'(q) << 2), tiles[t * 256 + r * 4 + q]);
    for (int e = 0; e < 640; e++)
      for (int f = 0; f < 3; f++) wr(BASE + 32'h8000 + (32'(e) << 4) + (32'(f) << 2), params[e * 3 + f]);
    wr(BASE + 32'h28, meta[1]);
    wr(BASE + 32'h08, 32'h4);                         // IRQ_EN: sequencer done
    rd(BASE + 32'h28, d);
    if (d !== meta[1]) begin $display("ERROR: SEQ_ZP reads 0x%08h", d); errors++; end
    $display("loaded %0d network tiles and 640 parameter entries (FE_DED = %0d, SEQ_S = %0d, SEQ_B1 = %0d)", NT, DED, S, B1);
`ifdef ALLCLIPS
    begin
      int bad_clips = 0, e0;
      for (int k = 0; k < nclips; k++) begin
        logic [31:0] st2; longint unsigned tg, cy;
        e0 = errors;
        load_feat(k, 0);
        obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, tg);
        seq_finish(tg, cy, st2);
        check_logits(k, st2, "all");
        if (errors != e0) bad_clips++;
        if (cy < cyc_min) cyc_min = cy;
        if (cy > cyc_max) cyc_max = cy;
        if ((k + 1) % 500 == 0) $display("  %0d clips, %0d with a logit or class error", k + 1, bad_clips);
      end
      $display("engine contract: %0d violations", contract_err);
      errors += contract_err;
      $display("RESULT (all clips): %0d clips, %0d with a logit or class error, %0d errors; %0d..%0d cycles go -> interrupt",
               nclips, bad_clips, errors, cyc_min, cyc_max);
      if (errors == 0) $display("PASS"); else $display("FAIL");
      $finish;
    end
`endif

    // 1. every layer of clip 0
    load_feat(0, 0);
    for (int s = 1; s <= NSTOP; s++) begin
      int L = s - 1, bad = 0;
      bit ok;
      for (int i = 0; i < 8000; i++) l1_got[i] = 1'b0;
      obi(1'b1, BASE + 32'h20, seq_word(s, 0), d, t_go);
      seq_finish(t_go, cyc, st);
      if (st[11:8] != 4'((B1 && (L == 1)) ? 2 : L)) begin $display("ERROR: stopped in layer %0d, expected %0d", st[11:8], L); errors++; end
      for (int p = 0; p < 125; p++)
        for (int c = 0; c < 64; c++) begin
          get_act(L, p, c, d, ok);
          if (ok && (d !== {{24{acts_exp[L * 8000 + p * 64 + c][7]}}, acts_exp[L * 8000 + p * 64 + c]})) begin
            if (bad < 4) $display("  L%0d p %0d c %0d: got %0d expected %0d", L, p, c, $signed(d), $signed(acts_exp[L * 8000 + p * 64 + c]));
            bad++;
          end
        end
      $display("layer L%0d (stop after it): %0d / 8000 mismatches, %0d cycles from go", L, bad, cyc);
      errors += bad;
    end
`ifdef MEDIUM
    run_clip(0, 0, "full");
`endif
`ifdef QUICK
    $display("RESULT (quick): %0d errors", errors);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
`elsif MEDIUM
    $display("RESULT (medium): %0d errors", errors);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
`endif

    // 2. full runs
    for (int k = 0; k < nclips; k++) run_clip(k, 0, "full");

    // 3. FEAT ring
    run_clip(1, 17, "ring frame0@17");

    // 4. interplay
    load_feat(2, 0);
    obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, t_go);       // CPU X write during a run: held
    obi(1'b1, BASE + 32'h1000, 32'h5, d, t_a);
    if (t_a - t_go < 50000) begin $display("ERROR: X write granted %0d cycles into a run", t_a - t_go); errors++; end
    seq_finish(t_go, cyc, st); check_logits(2, st, "X-write-held");
    obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, t_go);       // FEAT write during a run: held
    obi(1'b1, BASE + 32'h4000 + (32'd48 << 4), feat[2 * 147 + 48 * 3], d, t_a);
    if (t_a - t_go < 50000) begin $display("ERROR: FEAT write granted %0d cycles into a run", t_a - t_go); errors++; end
    seq_finish(t_go, cyc, st); check_logits(2, st, "FEAT-write-held");
    rd(BASE + 32'h0C, d);
    obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, t_go);       // CPU go during a run: held, then runs
    obi(1'b1, BASE + 32'h0, 32'h8000_0000 | (32'd9 << 8) | 32'(T0), d, t_a);
    if (t_a - t_go < 50000) begin $display("ERROR: CPU go granted %0d cycles into a run", t_a - t_go); errors++; end
    seq_finish(t_go, cyc, st); check_logits(2, st, "CPU-go-held");
    do rd(BASE + 32'h4, d); while (!d[1]);
    cpu_pass_check(T0 + 1, 1);                                // CPU passes work after runs
    cpu_pass_check(T0 + 60, 2);
    rd(BASE + 32'h0C, d);
    if (d != 32'd3) begin $display("ERROR: DONE_CNT %0d, expected 3 CPU passes", d); errors++; end
    begin                                                     // SEQ go waits for a CPU pass in flight
      logic [31:0] lo, hi; longint unsigned t_c; int x [64];
      for (int r = 0; r < 64; r++) begin x[r] = (r * 13) % 200 - 100; wr(BASE + 32'h1000 + 32'(4 * r), 32'(x[r])); end
      obi(1'b1, BASE + 32'h0, 32'h8000_0000 | (32'd9 << 8) | 32'(T0 + 2), d, t_c);
      obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, t_go);
      if (t_go - t_c < 14) begin $display("ERROR: SEQ go granted %0d cycles after a CPU go", t_go - t_c); errors++; end
      seq_finish(t_go, cyc, st); check_logits(2, st, "after-CPU-pass");
      for (int n = 0; n < 16; n++) begin
        longint s = 0;
        for (int r = 0; r < 64; r++) s += longint'(wt(T0 + 2, r, n)) * longint'(x[r]);
        rd(BASE + 32'h40 + 32'(4 * n), lo); rd(BASE + 32'h80 + 32'(4 * n), hi);
        if ({hi[15:0], lo} !== 48'(s)) begin $display("MISMATCH CPU pass before a run, col %0d", n); errors++; end
      end
    end
    // front-end go during a run: held with cim_fe, concurrent with B3; SEQ go during a front-end frame
    for (int i = 0; i < 480; i++) wr(BASE + 32'h2000 + 32'(4 * i), 32'((i * 397) % 4001 - 2000));
    obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, t_go);
    obi(1'b1, BASE + 32'h10, 32'h8000_0000, d, t_a);
    if (!DED && (t_a - t_go < 50000)) begin $display("ERROR: cim_fe go granted %0d cycles into a run", t_a - t_go); errors++; end
    if (DED && (t_a - t_go > 20)) begin $display("ERROR: B3 go held %0d cycles by a run", t_a - t_go); errors++; end
    seq_finish(t_go, cyc, st); check_logits(2, st, "with-front-end");
    do rd(BASE + 32'h14, d); while (!d[1]);
    wr(BASE + 32'h14, 32'h2);
    obi(1'b1, BASE + 32'h10, 32'h8000_0000, d, t_a);         // front-end first
    obi(1'b1, BASE + 32'h20, seq_word(0, 0), d, t_go);
    if (!DED && (t_go - t_a < 5000)) begin $display("ERROR: SEQ go granted %0d cycles into a cim_fe frame", t_go - t_a); errors++; end
    if (DED && (t_go - t_a > 20)) begin $display("ERROR: SEQ go held %0d cycles by B3", t_go - t_a); errors++; end
    seq_finish(t_go, cyc, st); check_logits(2, st, "after-front-end");
    do rd(BASE + 32'h14, d); while (!d[1]);
    $display("interplay checks done");

    // 5. synthetic requantisation parameters and zero points; 6. the same with zero point -128
    for (int set = 0; set < 2; set++) begin
      string tag = set ? "synthetic parameters 2 (zero point -128)" : "synthetic parameters";
      for (int e = 0; e < 640; e++)
        for (int f = 0; f < 3; f++) wr(BASE + 32'h8000 + (32'(e) << 4) + (32'(f) << 2), set ? s2_params[e * 3 + f] : s_params[e * 3 + f]);
      wr(BASE + 32'h28, set ? s2_meta[0] : s_meta[0]);
      load_feat(0, 0);
      for (int s = 1; s <= 10; s++) begin
        int L = s - 1, bad = 0;
        bit ok;
        for (int i = 0; i < 8000; i++) l1_got[i] = 1'b0;
        obi(1'b1, BASE + 32'h20, seq_word(s == 10 ? 0 : s, 0), d, t_go);
        seq_finish(t_go, cyc, st);
        if (s == 10) begin
          for (int n = 0; n < 12; n++) begin
            logic [7:0] y = set ? s2_logits[n] : s_logits[n];
            rd(BASE + 32'h5000 + 32'(4 * n), d);
            if (d !== {{24{y[7]}}, y}) bad++;
          end
          if (st[7:4] != (set ? s2_meta[1][3:0] : s_meta[1][3:0])) bad++;
          $display("%s, logits and class: %0d errors", tag, bad);
        end else begin
          for (int p = 0; p < 125; p++)
            for (int c = 0; c < 64; c++) begin
              logic [7:0] a = set ? s2_acts[L * 8000 + p * 64 + c] : s_acts[L * 8000 + p * 64 + c];
              get_act(L, p, c, d, ok);
              if (ok && (d !== {{24{a[7]}}, a})) bad++;
            end
          $display("%s, layer L%0d: %0d / 8000 mismatches", tag, L, bad);
        end
        errors += bad;
      end
    end
    $display("engine contract: %0d violations", contract_err);
    errors += contract_err;

    $display("RESULT: %0d errors; full inference %0d..%0d cycles (go -> interrupt) = %.2f..%.2f ms at 15 MHz",
             errors, cyc_min, cyc_max, real'(cyc_min) / 15.0e3, real'(cyc_max) / 15.0e3);
    if (errors == 0) $display("PASS"); else $display("FAIL");
    $finish;
  end

  initial begin                                       // watchdog
`ifdef ALLCLIPS
    #20s $display("TIMEOUT"); $display("FAIL"); $finish;
`else
    #400ms $display("TIMEOUT"); $display("FAIL"); $finish;
`endif
  end
endmodule
