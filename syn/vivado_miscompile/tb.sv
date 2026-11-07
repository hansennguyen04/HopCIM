`timescale 1ns/1ps
module tb;
  logic [2:0] ox; logic [1:0] a, b, c, d;
  mr_nl_ref r (.ox, .w0(a), .w0_nomask(c));
  mr        n (.ox, .w0(b), .w0_nomask(d));
  initial begin
    for (int i = 0; i < 5; i++) begin
      ox = 3'(i); #1;
      $display("ox %0d (ix0 %0d): w0 rtl %0d netlist %0d | (ix0+3)>>>2 rtl %0d netlist %0d", i, 2 * i - 1, a, b, c, d);
    end
    $finish;
  end
endmodule
