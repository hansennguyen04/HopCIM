// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// cim_log2fix16: floor(log2(v) * 2^16) for 1 <= v <= 32767, as model/mfcc_int.py log2fix16
// (the window path's normalisation, R5b). e = floor(log2 v), m = v << (30 - e) (Q30 mantissa),
// then 16 times m = m^2 >> 30, the next fraction bit being m >= 2^31 (then m >>= 1); one squaring
// per cycle. start_i loads v_i; done_o pulses 16 cycles later, with y_o = 65536 e + fraction.
// Unsigned bit-level logic only (see docs/SEQUENCER.md §8a).

module cim_log2fix16 (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,
    input  logic [14:0] v_i,                         // 1 .. 32767
    output logic        done_o,
    output logic [19:0] y_o
);

  logic [3:0]  e, e0;
  logic [30:0] m;
  logic [15:0] f;
  logic [4:0]  step;                                 // 0: idle, 1..16: squaring step
  /* verilator lint_off UNUSEDSIGNAL */
  logic [61:0] sq;                                   // m^2: bits [29:0] are dropped by the >> 30
  /* verilator lint_on UNUSEDSIGNAL */

  always_comb begin                                  // leading one of v (v >= 1)
    e0 = 4'd0;
    for (int i = 1; i < 15; i++) if (v_i[i]) e0 = 4'(i);
  end
  assign sq = {31'd0, m} * {31'd0, m};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      e      <= 4'd0;
      m      <= 31'd0;
      f      <= 16'd0;
      step   <= 5'd0;
      done_o <= 1'b0;
    end else begin
      done_o <= 1'b0;
      if (start_i) begin
        e    <= e0;
        m    <= 31'({16'd0, v_i} << (5'd30 - {1'b0, e0}));
        f    <= 16'd0;
        step <= 5'd1;
      end else if (step != 5'd0) begin
        m <= sq[61] ? sq[61:31] : sq[60:30];
        f <= {f[14:0], sq[61]};
        if (step == 5'd16) begin
          step   <= 5'd0;
          done_o <= 1'b1;
        end else step <= step + 5'd1;
      end
    end
  end

  assign y_o = {e, f};

endmodule
