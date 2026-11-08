# HopCIM — Specific Next-Step Execution Plan
## Professor-style research decision, novelty review, implementation priorities, and exit gates

**Date:** 2026-11-08  
**Project:** HopCIM  
**Platform:** X-HEEP + digital bit-serial CIM + Nexys A7 FPGA emulation  
**Application:** Always-on keyword spotting with the unchanged MLPerf Tiny int8 DS-CNN

---

# 0. Student-requested instruction

> **About "Manual IEEE Xplore novelty search + advisor claim freeze", please do the IEEE Xplore novelty search for me and act as my professor and decide which is the best advice.**

This document therefore includes: (1) an IEEE-Xplore-targeted novelty review, (2) a professor-style decision on the thesis claim, (3) a frozen priority order for the next work, and (4) explicit BUILD / DO NOT BUILD / OPTIONAL decisions with quantitative exit gates.

---

# 1. Professor decision

Do **not** make StreamFuse, ExactReuse, or any individual micro-optimization the thesis headline.

The strongest thesis story is:

> **HopCIM is an exact, model-preserving, CPU-light always-on KWS system on a digital CIM-equipped RISC-V MCU. The work experimentally shows how the bottleneck moves as progressively more of the pipeline—MACs, data movement, requantization, feature extraction, window processing, and audio input—is moved from the CPU into hardware.**

Recommended thesis title:

> **HopCIM: Exact Model-Preserving Keyword Spotting on a Digital CIM-Equipped RISC-V Microcontroller**

Alternative, more architecture-focused title:

> **HopCIM: Where the Bottleneck Moves in an Exact CIM-Accelerated Keyword-Spotting Microcontroller**

Recommended central research question:

> **What must move into hardware—not just into the CIM MAC array—to make an exact pretrained always-on KWS pipeline efficient on a CIM-equipped RISC-V microcontroller?**

The thesis contributions should be organized as:

1. **Exact model-preserving pipeline:** standard MFCC-style front-end, unchanged pretrained int8 DS-CNN, no retraining, bit-exact network behavior against the TFLite reference.
2. **Measured bottleneck migration:** MAC-only offload is weak; hardware gather/control/requantization and front-end autonomy create the major performance gain.
3. **End-to-end live system:** microphone → decimator → MFCC/window path → CIM network → keyword, with the CPU mostly asleep.
4. **Design-space evaluation:** CPU, CPU-driven CIM, autonomous CIM, dedicated MFCC, CIM MFCC, decision cadence, ExactReuse, cheap exact requantization optimizations, and optional StreamFuse.

---

# 2. IEEE-Xplore-targeted novelty search

## 2.1 Search families used

The search targeted combinations of:

```text
compute-in-memory / computing-in-memory / CIM
+
MFCC / FFT / Fourier / DFT / mel / DCT
+
keyword spotting / speech / audio
+
RISC-V
+
pretrained / unchanged model
```

Representative queries:

```text
"compute-in-memory" MFCC keyword spotting
"computing-in-memory" FFT keyword spotting
"SRAM CIM" keyword spotting MFCC
"digital CIM" keyword spotting MFCC
"compute-in-memory" RISC-V keyword spotting
"serial FFT" keyword spotting CIM
"MFCC-CNN" computing-on-memory keyword spotting
```

This was an IEEE-Xplore-targeted web search supplemented by exact-title and DOI searches for known close works. It is strong enough for research positioning, but it is not identical to a logged-in human manually paging every Xplore result. The thesis should therefore still use cautious wording such as **"to our knowledge"**.

## 2.2 Closest works found

### Shan et al., ISSCC 2020

**A 510nW 0.41V Low-Memory Low-Computation Keyword-Spotting Chip Using Serial FFT-Based MFCC and Binarized Depthwise Separable Convolutional Neural Network in 28nm CMOS**  
DOI: `10.1109/ISSCC19947.2020.9063000`

Establishes complete low-power KWS hardware, serial FFT-based MFCC, and DS-CNN. It does **not** map the standard MFCC onto the same digital CIM as the network, and the network is binarized/co-designed.

**Implication:** hardware MFCC + KWS and serial-FFT MFCC are already taken.

### Jia et al., ISCAS 2022

**A 11.6 μW Computing-on-Memory-Boundary Keyword Spotting Processor with Joint MFCC-CNN Ternary Quantization**  
DOI: `10.1109/ISCAS48785.2022.9937206`

Establishes KWS with a memory-compute macro, joint MFCC/CNN optimization, and mel+DCT matrix computation near memory. MFCC/model arithmetic is changed and quantized jointly.

**Implication:** mel/DCT-on-memory and KWS front-end processing near memory are not new.

### RV-Flash, ASP-DAC 2025

**RISC-V Driven Orchestration of Vector Processing Units and eFlash Compute-in-Memory Arrays for Fast and Accurate Keyword Spotting**  
DOI: `10.1145/3658617.3697697`

Establishes RISC-V + CIM for KWS, but uses a changed/reparameterized hybrid KWS model split between eFlash CIM and vector hardware.

**Implication:** "RISC-V + CIM + KWS" is taken.

### Hybrid FFT-BFNet, IEEE TVLSI 2026

**A 95.3% 12-Class, 108-nJ/Inference Keyword Spotting Chip With Hybrid FFT-BFNet Architecture and Exponent-Aware Nonuniform Quantization in 65-nm CMOS**  
DOI: `10.1109/TVLSI.2025.3632791`

Establishes FFT/network resource sharing and a very efficient complete KWS chip, but changes the network architecture and quantization.

**Implication:** FFT/network hardware sharing is not new by itself.

### KWS-CIM, IEEE TCAS-I 2026

**KWS-CIM: A Charge-Domain eDRAM-Based Compute-in-Memory Macro for Ternary DFSMN Keyword Spotting**  
DOI: `10.1109/TCSI.2026.3717749`

Establishes fabricated CIM specifically for KWS using ternary/binary arithmetic and a DFSMN model.

**Implication:** CIM for KWS is clearly not new.

### Heterogeneous TinyML KWS SoC, IEEE JSSC 2026

**A Heterogeneous TinyML SoC With Systematic Minimum Energy Searching and Management for Keyword Spotting**  
DOI: `10.1109/JSSC.2025.3619427`

Establishes system-level KWS energy optimization and CIM-assisted wake-up.

**Implication:** general system-level KWS energy optimization is crowded.

## 2.3 Novelty verdict

The search did **not** find an IEEE-Xplore-indexed work clearly demonstrating all of the following together:

```text
standard MFCC including FFT
+
mapped through the same digital multi-bit CIM infrastructure
+
unchanged pretrained int8 KWS model
+
no retraining
+
RISC-V MCU integration
+
end-to-end live real-time hardware demonstration
```

The narrow HopCIM system claim therefore remains **plausible**, but the thesis must avoid broad novelty claims.

Already taken individually:

```text
KWS + CIM
RISC-V + CIM + KWS
hardware MFCC + KWS
serial FFT MFCC
mel/DCT near memory
layer fusion
depth-first execution
activation forwarding
resident-weight cross-layer scheduling
```

## 2.4 Claim I would approve as professor

Cautious first-style claim:

> **To our knowledge, HopCIM is the first demonstrated RISC-V MCU system in which a complete model-preserving MFCC front-end and an unchanged pretrained int8 KWS network share one exact multi-bit digital CIM datapath, with bit-exact reference behavior and live real-time FPGA validation.**

Safer thesis claim, which I prefer:

> **HopCIM demonstrates that an unchanged MFCC-trained int8 KWS model can be deployed end-to-end on a digital CIM-equipped RISC-V MCU without retraining, and experimentally quantifies where the performance and energy bottlenecks migrate as MACs, control, data movement, and feature processing are progressively moved into hardware.**

Do **not** claim:

- first CIM KWS system;
- first RISC-V CIM KWS;
- first MFCC hardware accelerator;
- first FFT-on-CIM;
- first mel/DCT-on-memory;
- first layer fusion / DW→PW forwarding / depth-first CIM;
- first activation reuse;
- measured silicon CIM energy.

---

# 3. P0 — Reproducibility freeze

**Status:** substantially done.

Reported state:

```text
freeze commit: aec9897
tag: freeze-2026-11-07b
gates frozen before results: a80834c
self-test: 23/23 deliberate breaks caught
```

Remaining action: create/update `RESULTS_FREEZE.md` containing:

```text
git commit / tag
TFLite SHA256
Vivado / Verilator versions
Python environment
board / bitstream SHA256
headline cycles
accuracy
speaker logs
energy-model version
StreamFuse-model version
commands used to reproduce each thesis table
```

**Exit:** every thesis table is traceable to `commit + command + output file`.

---

# 4. P1 — Freeze thesis claim with advisor

Bring the advisor the professor recommendation from Sections 1–2.

Ask only these questions:

1. Is the **system + bottleneck-migration** claim acceptable as the main thesis contribution?
2. Is cycle-faithful FPGA emulation plus a published/tool energy model sufficient, or is a stronger ASIC estimate expected?
3. Is a conference paper expected, or only a strong Master's thesis?
4. If publication novelty must be increased, should effort go into stronger evidence or one optional StreamFuse-B RTL prototype?

**My professor decision:** freeze the thesis around the system and bottleneck-migration story now. Do not wait for a stronger StreamFuse mechanism claim.

---

# 5. P2 — Build the new mandatory baseline S

Define:

\[
S = \text{sign-gated requantization} + \text{second requantization lane} + \text{redundant L8 ACT-write removal}.
\]

This becomes the baseline for every later optimization.

## 5.1 Exact sign-gated requantization

For layers L0–L8, output zero point is `-128`. If:

\[
acc+bias \le 0,
\]

the final clamped output is exactly `-128`, so bypass the expensive requantization multiply/shift path.

Measured software result:

```text
36,338 / 72,000 ≈ 50.5 %
```

of requantizations satisfy the condition across all 4,890 test clips.

RTL fast path:

```text
if signed(acc_plus_bias) <= 0:
    out = -128
else:
    normal_exact_requant()
```

Verification:

- negative / zero / positive boundary cases;
- multiplier/shift extremes;
- all existing sequencer tests;
- full 4,890-clip model replay;
- RTL vs golden;
- mutation test with intentionally wrong gate threshold.

**Exit:** zero mismatches.

## 5.2 Second requantization lane

Expected model result:

```text
85,352 → 67,841 cycles/inference
```

Measure after RTL:

```text
actual cycles
LUT / FF / DSP
WNS
gate/logit correctness
```

**Exit:** zero mismatches and no timing regression that invalidates the predicted throughput.

## 5.3 Remove redundant L8 ACT write

Eliminate the unnecessary L8 activation round-trip if the pool/final path can consume it directly.

**Exit:** exact pool sums and logits.

---

# 6. P3 — Verify S end-to-end on FPGA

Run:

1. unit regression;
2. sequencer regression;
3. full SoC simulation;
4. synthesis;
5. board regression;
6. live microphone test.

Record:

```text
network cycles
worst-hop cycles
CPU-active cycles
overrun
LUT / FF / BRAM / DSP
all logit comparisons
```

**Exit:** zero functional mismatches.

---

# 7. P4 — Test the ~4.10-MHz operating point

Model prediction with S:

```text
worst hop ≈ 80,072 cycles
```

At 4.096 MHz, a 20-ms hop provides:

```text
81,920 cycles
```

(~2.3% model margin).

Build the nearest clock configuration compatible with the 1.024-MHz PDM clock and run at least 500 live decisions, preferably several thousand.

Check:

```text
overrun = 0
sampled logits = golden
PDM clock correct
audio remains exactly 16 kHz
```

If successful, update the minimum real-time clock from 5.12 MHz to approximately 4.10 MHz. If it fails, report the measured limit honestly.

---

# 8. P5 — Recompute the energy model using real S RTL counts

Once S exists, remove the current assumed S control counts.

Re-count:

```text
requant multiplies actually executed
PARAM reads
ACT reads/writes
X-bank traffic
engine busy cycles
sequencer active cycles
```

Update `ENERGY_ESTIMATE.md` for:

```text
B0
S
20-ms S
40-ms S
80-ms S
```

Keep all energy results labeled `[E]`.

---

# 9. P6 — Same-audio 20/40/80-ms experiment

The current live trials used different speech recordings per rate, so run one 20-ms-resolution capture and derive all rates offline from the exact same audio.

Evaluate:

```text
20 ms
40 ms phase 0
40 ms phase 1
80 ms phases 0,1,2,3
```

Also separate network cadence from the keyword hold rule, e.g.:

```text
80 ms + current 2-decision rule
80 ms + one strong decision
80 ms + 2-of-3 rule
```

Current operating point remains **40 ms** unless identical-audio evidence clearly supports 80 ms.

---

# 10. P7 — Validate the CIM timing assumption

Current model:

\[
D=\max(8,b), \qquad t_{done}=t_0+D+5.
\]

Make issue interval and result pipeline configurable.

Recommended sweep:

\[
D=\max(D_{min},\alpha b+\beta)
\]

with representative points:

```text
alpha = 1.0, 1.5, 2.0
beta  = 0, 4, 8
Dmin  = 8
pipeline = 3, 5, 8, 16
```

At minimum include:

```text
baseline:          max(8,b)
conservative:      max(8,2b)
very conservative: max(8,2b+8)
```

For each point recompute:

```text
network cycles
worst-hop cycles
minimum real-time clock
engine utilization
```

Final wording should describe HopCIM as a **parameterized cycle-faithful CIM architectural model**, not physical timing measured from a fabricated macro.

---

# 11. P8 — Run ZigZag-IMC itself

The current energy model transcribes ZigZag-IMC equations. Run the actual tool using equivalent macro/mapping parameters.

Compare:

```text
array energy per pass
energy per inference
relative ordering of designs
mapping sensitivity
```

If disagreement exceeds roughly 10–15%, investigate before thesis freeze. Even if it agrees, all results remain `[E]` rather than silicon measurements.

---

# 12. P9 — Improve digital-logic energy estimates

Use an open ASIC flow only for conventional digital blocks:

```text
sequencer
requantizer
window path
audio controller
optional StreamFuse controller
```

Recommended methodology split:

```text
CIM array      → ZigZag / CiMLoop / published macro
SRAM           → CACTI / OpenRAM
digital RTL    → ASIC synthesis/activity estimate
activity       → exact RTL counters
```

Do **not** synthesize the FPGA-emulated CIM and call its result a real CIM macro.

Priority: medium, after S and the macro timing sensitivity.

---

# 13. P10 — Stronger live-system evidence

After the cadence is frozen, test approximately 3–5 speakers. A practical target is:

```text
10 keywords × 5 repetitions per speaker
```

Compare only the relevant final operating points, likely 20 and 40 ms.

Record microphone distance, room RMS, speaker ID, word, prediction, and detection latency.

## False alarms per hour

Collect long streams of:

```text
quiet room
ordinary conversation
TV/podcast speech
music
office/background noise
```

Report:

\[
FAR=\frac{\text{false keyword triggers}}{\text{hours}}.
\]

This is more meaningful for an always-on KWS system than another small batch of prompted words.

---

# 14. P11 — Verification polish

Add assertions for:

```text
start && !ready → pass not accepted
done → outstanding pass exists
X-bank read → no destructive overwrite
layer_done → all outputs committed
decision IRQ → logits valid
supported clock → no audio overrun
```

Run Verilator coverage and inspect untested FSM states, padding cases, bank hazards, interrupts, reset, and overrun paths.

Run a targeted full-endpoint post-synthesis/netlist smoke test for:

```text
one full inference
window-max-change path
bank switching
IRQ
one short microphone stream
```

Do not spend months on exhaustive gate-level simulation.

---

# 15. P12 — StreamFuse decisions

## A — line-buffer-only

**DO NOT BUILD.**

It fails the frozen energy and storage gates and is dominated by stronger alternatives.

## B — one-map + DW→PW forwarding

**OPTIONAL ONE-BLOCK RTL PROTOTYPE, only after S.**

Model relative to S:

```text
network energy: 1119.8 → 937.6 nJ  (~−16.3%)
40-ms total:     68.8 → 63.9 µW    (~−7%)
cycles:          67,841 → 67,745   (~unchanged)
storage:         1.0× → 0.575×
```

First RTL target only:

```text
DW1 → exact requant → fusion vector → PW1
```

Required mechanisms:

```text
source-ready interlock
overwrite interlock
fusion vector
line buffer
8-byte fusion-vector gather
```

Expand to all four blocks only if one-block RTL confirms the modeled traffic, cycles, and manageable control overhead.

## C — full depth-first StreamFuse

**DO NOT BUILD FOR THIS THESIS.**

It has much higher complexity, weak novelty, and almost no system-level gain over B.

---

# 16. P13 — ExactReuse decision

**DO NOT BUILD AS THE NEXT STEP.**

Keep it as a design-space/negative result comparing temporal reuse with reduced inference cadence. Reconsider only if a future application requirement forces high decision cadence and stronger modeling still shows a compelling benefit.

---

# 17. New mandatory baseline rule

From now on, every new optimization must compare against:

\[
\boxed{S}
\]

not only against the old 85,352-cycle R6b design.

Any new idea must beat S on at least one meaningful system metric without losing correctness.

---

# 18. Final thesis baseline table to freeze

| Configuration | Network cycles | CPU-active | Accuracy | ACT traffic | Energy [E] | Resources |
|---|---:|---:|---:|---:|---:|---:|
| CPU-only | measured | measured | reference | — | optional | CPU |
| CPU-driven CIM | measured | measured | exact | measured | [E] | measured |
| R6b | 85,352 | measured | exact | measured | 1,389 nJ [E] | measured |
| **S** | measure RTL | measure | exact | measure | recompute [E] | measure |
| 40-ms S | — | measure | live test | — | [E] | same |
| optional StreamFuse-B | measure | measure | exact | measure | [E] | measure |
| B3 dedicated MFCC | measured | measured | measured | — | [E] | measured |

---

# 19. Thesis narrative to freeze

```text
1. CIM makes MACs cheap.

2. MFCC and CPU orchestration become bottlenecks.

3. Moving only MACs gives weak gains.

4. Moving the full front-end, gather, sequencing, and exact
   requantization into hardware makes the CPU almost idle.

5. The bottleneck moves again inside the accelerator:
   requantization, activation memory, control, and leakage dominate.

6. Cheap exact optimization S therefore comes before complex
   activation-dataflow mechanisms.

7. More elaborate reuse/fusion works, but gives diminishing
   whole-system returns.

8. The main lesson is not "CIM solves everything";
   the bottleneck continually migrates outside the MAC array.
```

---

# 20. Immediate work order

## Immediate

```text
1. Update RESULTS_FREEZE.md
2. Present/freeze thesis positioning with advisor
3. Implement S
4. Verify S
5. Test ~4.10-MHz live operation
```

## Next

```text
6. Recompute energy using S RTL activity
7. Same-audio cadence experiment
8. CIM timing sensitivity sweep
9. Run ZigZag-IMC itself
```

## Evidence strengthening

```text
10. 3–5 speakers
11. FAR/hour
12. assertions + coverage
13. targeted endpoint netlist smoke test
```

## Optional publication extension

```text
14. StreamFuse-B one-block RTL
15. expand B only if the one-block model→RTL prediction is accurate
```

## Explicitly do not prioritize

```text
StreamFuse A
StreamFuse C
ExactReuse RTL
large new architecture rewrites
new model training solely to create novelty
```

---

# 21. Stop conditions

Stop adding architecture and move fully into thesis/paper writing when:

```text
[ ] thesis claim frozen with advisor
[ ] S verified on RTL + board
[ ] final minimum real-time clock measured
[ ] final 20/40/80-ms cadence decision made
[ ] macro timing sensitivity completed
[ ] energy tool cross-check completed
[ ] multi-speaker/FAR evidence completed
[ ] results manifest frozen
```

After this point, additional StreamFuse work should only be done if it is explicitly needed for a conference paper.

---

# 22. Final professor advice

If I were supervising this project, my advice would be:

> **Do not chase another large architecture simply to increase novelty. HopCIM already has enough technical depth for a strong Master's thesis. The next job is to strengthen the claim, replace weak assumptions with sensitivity analysis or independent tools, implement the cheap exact baseline S, and produce a clean end-to-end evaluation.**

For publication, frame the paper around:

> **the migration of bottlenecks in a model-preserving CIM-accelerated KWS MCU**

rather than around:

> **a new fusion/reuse mechanism.**

If one additional hardware extension is desired, permit only:

\[
\boxed{\text{StreamFuse-B one-block prototype}}
\]

after S is complete.

The priority is:

\[
\boxed{\text{evidence quality} > \text{claim quality} > \text{simple exact optimization} > \text{new complex RTL}}
\]

---

# 23. Key literature identified in the novelty review

1. Shan et al., ISSCC 2020 — *A 510nW 0.41V Low-Memory Low-Computation Keyword-Spotting Chip Using Serial FFT-Based MFCC and Binarized Depthwise Separable Convolutional Neural Network in 28nm CMOS*. DOI: `10.1109/ISSCC19947.2020.9063000`.
2. Jia et al., ISCAS 2022 — *A 11.6 μW Computing-on-Memory-Boundary Keyword Spotting Processor with Joint MFCC-CNN Ternary Quantization*. DOI: `10.1109/ISCAS48785.2022.9937206`.
3. Kang et al., ASP-DAC 2025 — *RISC-V Driven Orchestration of Vector Processing Units and eFlash Compute-in-Memory Arrays for Fast and Accurate Keyword Spotting*. DOI: `10.1145/3658617.3697697`.
4. Fu et al., IEEE TVLSI 2026 — *A 95.3% 12-Class, 108-nJ/Inference Keyword Spotting Chip With Hybrid FFT-BFNet Architecture and Exponent-Aware Nonuniform Quantization in 65-nm CMOS*. DOI: `10.1109/TVLSI.2025.3632791`.
5. *KWS-CIM: A Charge-Domain eDRAM-Based Compute-in-Memory Macro for Ternary DFSMN Keyword Spotting*, IEEE TCAS-I 2026. DOI: `10.1109/TCSI.2026.3717749`.
6. *A Heterogeneous TinyML SoC With Systematic Minimum Energy Searching and Management for Keyword Spotting*, IEEE JSSC 2026. DOI: `10.1109/JSSC.2025.3619427`.

These should be cited prominently because they define the closest novelty boundaries around HopCIM.
