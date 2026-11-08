# R1 — `cim_engine`: FPGA emulation of a bit-serial digital CIM (one active macro)

Status: specification, frozen before RTL. Scope: digital design only, evaluated by RTL simulation (Verilator) and FPGA tools (Vivado, Nexys A7-100T).

## 1. What is modelled

The modelled CIM is a bank of resident **macros**. Each macro holds a 64-row × 16-column array of int8 weights (1 KiB). One macro is active per pass.

A **pass** on tile `t`:
- applies an input vector `x[0..63]` of `b` bits each (two's complement, `1 ≤ b ≤ 24`) bit-serially;
- produces 16 exact integer dot products `y[n] = Σ_r W_t[r][n] · x_b[r]`, where `x_b` is `x` truncated to `b` bits and sign-extended (what a `b`-plane bit-serial macro sees);
- either overwrites the 16 accumulators or adds to them (`accum`), so reductions longer than 64 rows chain over several passes.

**Timing contract.** A pass accepted at cycle `t0` has:
- **issue interval** `D = max(8, b)`: the next pass may start at `t0 + D`;
- **result** `done` asserted at `t0 + D + 5`, with `res` valid from that cycle.

This is the timing of a `b`-cycle bit-serial macro with a fixed 5-cycle pipeline. The emulation cannot go below 8 cycles, so `D ≥ 8`. How much depends on this timing: `TIMING_SENSITIVITY.md`.

**Capacity.** 64 tiles = 64 KiB of resident weights (enough for the KWS DS-CNN plus the MFCC matrices, ~35 KiB). Later, the mapped network needed 69 tiles, so the engine has had 128 since R6. This specification keeps its text as frozen before the RTL. Weights are written as 32-bit words (4 weights).

## 2. How the FPGA emulates it (claim boundary)

| Aspect | Emulation | Faithful to the model? |
|---|---|---|
| Weight storage | 8 row-lanes × 4 word banks of 512 × 32 bit (32 RAMB18 = 16 RAMB36 tiles), row `r` in lane `r mod 8`, columns 4q..4q+3 in bank q; registered read address | Capacity and addressing: yes |
| Arithmetic | 128 multipliers (DSP48) + 16 adder trees: 8 rows × 16 columns per cycle, a 64-row pass in 8 cycles | Results: bit-exact |
| Pass timing | Issue interval and `done` delayed to the contract above for `b > 8` | SoC-visible cycles: yes |
| Energy, area of a real DCIM | Not emulated | **No claim from the FPGA**; a model-based estimate is in `ENERGY_ESTIMATE.md` |

The SoC, firmware and data movement therefore see exactly the cycle behaviour of the modelled CIM. FPGA power or LUT counts are not CIM energy or area.

## 3. Interface (single clock)

| Group | Signals | Notes |
|---|---|---|
| Weight write | `w_we, w_tile[6:0], w_row[5:0], w_word[1:0], w_data[31:0]` | `w_data[8k+7:8k]` = weight of column `4·w_word + k` (128 tiles since R6) |
| Input write, port 1 | `x_we, x_idx[5:0], x_data[23:0]` | One row of input bank 0 (CPU, front-end) |
| Input write, port 2 | `xw_we, xw_bank, xw_row[5:0], xw_mask[7:0], xw_data[8][23:0]` | Up to 8 consecutive rows `xw_row .. xw_row + 7` of either bank at once; `xw_data[l]` is the row in lane `l = row mod 8`, `xw_mask[l]` enables it; added for R6b (layer sequencer). Port 2 has priority |
| Pass | `start, tile[6:0], xbits[4:0], accum, xbank` → `ready`, `done`, `res[16][48]` | `start` is accepted when `ready=1`; the pass reads input bank `xbank` |
| Status | `busy` | a pass is in flight (accepted, `done` not yet given); added for R5a, where the endpoint starts the front-end only when the engine is idle |

**Two input banks (R6b).** Each bank holds 64 rows; row `r` of bank `k` is entry `{k, r[5:3]}` of
lane `r[2:0]`, one 16-entry LUT-RAM per lane. A pass reads row group `g` (rows `8g .. 8g+7`) of its
bank at edge `t0 + 2 + g`, so the next input vector can be written into the other bank while a
pass runs. With bank 0 and port 1 the engine behaves exactly as before.

Hazards that the R2 wrapper must avoid:
- writing a tile while a pass reads it;
- writing the bank a pass reads during the 10 cycles after `start` (the endpoint holds CPU X
  writes for 10 cycles; the sequencer releases a bank 7 cycles after its last pass, so its first
  write lands 10 edges after that pass started).

## 4. Verification plan (R1 exit criteria)

1. **Verilator lint:** `-Wall` clean on the RTL.
2. **Self-checking testbench** against an integer reference model:
   - directed tests: zero weights, extreme values (−128 × −2^23), sign extension of out-of-range inputs, accumulate chains;
   - ≥ 2,000 random passes over random tiles and widths `b ∈ {1, 4, 8, 9, 12, 16, 20, 24}`;
   - back-to-back passes issued the cycle `ready` rises;
   - (R6b) 600 passes alternating between the banks while the idle bank is rewritten through
     port 2 in random chunks and masks, or through port 1. Mutations that ignore the bank or
     mis-map the wrapping lanes give 5,712 and 8,734 mismatches; the earlier phases miss both.

   **Pass:** 0 mismatches, and every pass meets the timing contract exactly.
3. **Vivado out-of-context** implementation on `xc7a100tcsg324-1` at 15 MHz: report LUT/FF/DSP/BRAM and slack. **Budget:** ≤ 10k LUT, ≤ 140 DSP, ≤ 16 RAMB36.
