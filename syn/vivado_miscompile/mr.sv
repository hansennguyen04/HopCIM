// minimal reproduction: word of the column = 0 mod 4 among ix0 .. ix0 + 3, ix0 = 2 ox - 1
module mr (input logic [2:0] ox, output logic [1:0] w0, output logic [1:0] w0_nomask);
  always_comb begin : blk
    int ix0, ixb;
    ix0 = 2 * int'(ox) - 1;
    ixb = ix0 + ((0 - ix0) & 3);
    w0  = 2'(ixb >>> 2);
    w0_nomask = 2'((ix0 + 3) >>> 2);        // same value for ix0 = 1 mod 4, written without the mask
  end
endmodule
