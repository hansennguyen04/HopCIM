# Vivado 2024.1 synthesis mismatch (found during the R6b board bring-up)

`mr.sv` computes, for `ix0 = 2·ox − 1`, the word index of the column ≡ 0 (mod 4) among
`ix0 … ix0 + 3`:

```systemverilog
ixb = ix0 + ((0 - ix0) & 3);   // int arithmetic
w0  = 2'(ixb >>> 2);
```

| ox | ix0 | expected (Verilator 5.040, xsim RTL) | Vivado 2024.1 netlist (`synth_design`, xc7a100t, OOC) |
|---:|---:|---:|---:|
| 0 | −1 | 0 | 0 |
| 1 | 1 | **1** | **2** |
| 2 | 3 | 1 | 1 |
| 3 | 5 | **2** | **3** |
| 4 | 7 | 2 | 2 |

The same value written as `(ix0 + 3) >>> 2` is synthesised correctly.

Reproduce:
1. `vivado -mode batch -source syn.tcl` writes the netlist `mr_nl.v`.
2. Simulate `tb.sv` with `mr.sv` renamed to `mr_nl_ref` against the netlist (xsim with
   `-L unisims_ver`).

In `cim_seq` the expression selected FEAT lane 0's word for conv1. On the board, conv1 was wrong
at the 50 positions with odd ox (the 4th tap read 4 columns too far), while every simulation passed.

Fix: `cim_seq` computes all gather addresses in unsigned bit-level logic. Its Vivado netlist
equals the RTL on all 101,250 address inputs; the old and new RTL agree on every case that is used.
