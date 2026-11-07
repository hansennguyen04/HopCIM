# ExactReuse-CIM
## Exact Padded/Strided Incremental Streaming Inference + CIM Reuse Compiler for HopCIM

**Project type:** Master's thesis research direction / potential conference-paper extension  
**Base platform:** HopCIM on X-HEEP + Nexys A7-100T  
**Application:** Always-on keyword spotting (KWS)  
**Base model:** Unchanged quantized MLPerf Tiny DS-CNN  
**Primary research goal:** Reduce redundant CIM computation across overlapping streaming inference windows while preserving the exact outputs of independent full-window inference.

---

# 1. Executive Summary

HopCIM already implements an end-to-end always-on keyword-spotting pipeline:

```text
PDM microphone
    ↓
CIC + FIR + DC blocker
    ↓
16 kHz PCM
    ↓
integer MFCC front-end
    ↓
streaming window normalization
    ↓
int8 MLPerf Tiny DS-CNN
    ↓
keyword decision
```

The current system is highly CPU-light: the CPU mostly sleeps while the hardware front-end, CIM engine, window path, and neural-network sequencer perform the work.

However, one major source of computational redundancy remains:

> HopCIM performs a complete DS-CNN inference every 20 ms even though consecutive one-second input windows overlap almost completely.

The current DS-CNN mapping uses:

| Layer | CIM passes per inference |
|---|---:|
| Conv1 | 500 |
| Depthwise ×4 | 5,000 |
| Pointwise ×4 | 2,000 |
| Fully connected | 1 |
| **Total** | **7,501** |

The existing sequencer takes about **85,352 cycles per inference**, with the CIM engine busy for about **67,509 cycles**.

This proposal develops a new execution method:

> **Reuse intermediate CNN activations across overlapping windows while explicitly repairing the effects of stride, SAME padding, quantization, and window-dependent MFCC changes, so that the optimized execution produces exactly the same int8 outputs as independently running the original pretrained TFLite model on every window.**

The main proposed contribution is therefore not generic streaming inference. Streaming reuse is already known.

The intended contribution is:

> **Exact, model-preserving incremental execution of an unchanged padded/strided quantized CNN on digital CIM, using phase-aware activation caches, exact dirty-region propagation, and a compiler that generates the reuse schedule automatically.**

---

# 2. Why This Direction Is a Natural Extension of HopCIM

HopCIM has already solved several earlier bottlenecks.

## 2.1 Existing HopCIM progress

The current research progression is approximately:

```text
CPU MFCC + CPU CNN
        ↓
CIM-assisted arithmetic
        ↓
hardware MFCC front-end
        ↓
hardware layer sequencer
        ↓
hardware window normalization
        ↓
live microphone path
        ↓
CPU almost completely asleep
```

The main remaining hardware workload is now the neural network itself.

The current R6b sequencer executes all **7,501 CIM passes for every decision**, even when nearly all input information overlaps with the previous decision.

Therefore the next meaningful research question is no longer:

> "How do we reduce CPU work?"

It becomes:

> "How do we reduce redundant accelerator work?"

This keeps the next research contribution directly connected to HopCIM's existing architecture.

---

# 3. Core Observation: Consecutive Windows Overlap Almost Completely

A KWS decision uses 49 MFCC frames.

At decision `N`:

```text
[f0 f1 f2 ... f46 f47 f48]
```

One hop later:

```text
   [f1 f2 ... f47 f48 f49]
```

Thus:

- 48 of 49 MFCC frame positions overlap.
- Only one new frame enters every 20 ms.

A naïve implementation still evaluates the entire CNN again.

This means a large amount of convolution work is potentially redundant.

However, simply shifting cached activations is **not correct** because HopCIM's network contains:

- stride-2 convolution;
- SAME padding;
- multiple depthwise 3×3 convolutions;
- exact integer requantization;
- window-dependent MFCC normalization.

These are the reasons the research problem is non-trivial.

---

# 4. Why Generic Incremental CNN Reuse Is Not Novel Enough

Several prior works already exploit overlap between streaming inputs.

## 4.1 Shan et al., IEEE JSSC 2021

Weiwei Shan et al. proposed framewise incremental computation for keyword spotting.

Reference:

> W. Shan et al., "A 510-nW Wake-Up Keyword-Spotting Chip Using Serial-FFT-Based MFCC and Binarized Depthwise Separable CNN in 28-nm CMOS," IEEE Journal of Solid-State Circuits, vol. 56, no. 1, pp. 151-164, 2021.  
> DOI: 10.1109/JSSC.2020.3029097

Their work shows that incremental KWS computation is already established.

However, their architecture modifies the network so incremental execution becomes easier.

Therefore HopCIM should **not** claim:

- first incremental KWS;
- first framewise CNN reuse;
- first streaming CNN reuse.

---

## 4.2 StreamiNNC, AAAI 2025

Kechris et al. proposed StreamiNNC:

> C. Kechris, J. Dan, J. Miranda, and D. Atienza, "Don't Think It Twice: Exploit Shift Invariance for Efficient Online Streaming Inference of CNNs," AAAI 2025.

Paper:
https://arxiv.org/abs/2408.03223

Code:
https://github.com/esl-epfl/streaminnc

This work explicitly identifies:

- zero padding;
- pooling;
- alignment;

as complications for streaming reuse.

It demonstrates that overlapping-window reuse is useful, but pretrained zero-padded networks do not automatically reproduce conventional full-window inference exactly.

This is highly relevant prior art.

---

## 4.3 Continual Inference

Continual Inference provides stateful step-wise neural execution for PyTorch.

Reference:

> L. Hedegaard and A. Iosifidis, "Continual Inference: A Library for Efficient Online Inference with Deep Neural Networks in PyTorch," 2022.

Paper:
https://arxiv.org/abs/2204.03418

Repository:
https://github.com/lukashedegaard/continual-inference

This is another important comparison because it demonstrates general continual neural execution.

---

## 4.4 DeltaCNN

DeltaCNN reuses sparse changes between video frames.

Reference:

> M. Parger et al., "DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos," CVPR 2022.

DOI:
10.1109/CVPR52688.2022.01217

This shows that dynamic activation-difference propagation is also established.

Therefore the research novelty cannot simply be:

> "Skip unchanged CNN computation."

---

# 5. Proposed Novelty Boundary

The intended novelty is the combination of:

1. **unchanged pretrained network;**
2. **SAME padding preserved;**
3. **stride preserved;**
4. **TFLite quantization preserved;**
5. **per-window preprocessing preserved;**
6. **bit-exact independent-window outputs;**
7. **hardware execution on a digital CIM accelerator;**
8. **compiler-generated phase/cache/dependency schedule.**

The proposed research question is:

> **Can an independently evaluated sliding-window quantized CNN reuse intermediate activations while preserving its exact original padded/strided semantics, without retraining or modifying the deployed model?**

The strongest defensible contribution would be:

> **An exact incremental execution method for sliding-window quantized CNNs that explicitly handles stride phase, padding boundaries, and dynamically changing quantized inputs, implemented as a digital-CIM reuse controller and generated automatically from the model graph.**

A final publication should **not use the word "first"** unless a later systematic literature review verifies that precise claim.

---

# 6. HopCIM Network Characteristics

The unchanged MLPerf Tiny int8 DS-CNN used by HopCIM is approximately:

```text
Input: 49 × 10 int8 MFCC
    ↓
Conv 10 × 4, stride 2
    ↓
25 × 5 × 64
    ↓
Depthwise 3 × 3
    ↓
Pointwise 1 × 1
    ↓
Depthwise 3 × 3
    ↓
Pointwise 1 × 1
    ↓
Depthwise 3 × 3
    ↓
Pointwise 1 × 1
    ↓
Depthwise 3 × 3
    ↓
Pointwise 1 × 1
    ↓
Average Pool
    ↓
Fully Connected 12 classes
```

The reference semantics include:

- SAME padding;
- int8 activations;
- per-channel quantized multipliers;
- TFLite `MultiplyByQuantizedMultiplier`;
- TFLite rounding behavior;
- int8 average-pooling rounding;
- exact argmax classification.

The current HopCIM integer implementation already verifies every layer against TFLite reference kernels.

This provides an excellent golden reference for the proposed reuse mechanism.

---

# 7. Problem 1: Stride Breaks Simple One-Hop Reuse

The first convolution has temporal stride 2.

If the input shifts by one MFCC frame:

```text
Window N:
[f0 f1 f2 ... f48]

Window N+1:
   [f1 f2 ... f49]
```

the convolution grid does not align with the previous output.

Conceptually:

```text
N     → stride phase 0
N+1   → stride phase 1
N+2   → stride phase 0
N+3   → stride phase 1
```

Therefore a single cache is insufficient.

---

# 8. Proposed Solution 1: Phase-Aware Caches

Maintain separate activation state for each stride phase.

For this model:

```text
phase_count = 2
```

because the first temporal stride is 2.

Use:

```text
Phase-0 cache
Phase-1 cache
```

A new decision accesses the cache corresponding to:

```text
phase = hop_index mod 2
```

Compare:

```text
Window N
[f0 f1 f2 ... f48]

Window N+2
      [f2 f3 ... f50]
```

These windows are separated by two input frames, so the stride-2 convolution grid is aligned.

This enables exact reuse of properly aligned interior activations.

---

# 9. Problem 2: SAME Padding Breaks Translation Equivalence

Even when stride phase aligns, SAME padding creates a second problem.

Interior convolution outputs can often be shifted safely.

Boundary outputs cannot.

For example:

```text
Old window:
PAD | input input input ... input | PAD

New aligned window:
PAD | shifted input ... new input | PAD
```

The left and right convolution boundaries use different padding/input combinations after the shift.

Therefore reused outputs near the boundaries may be wrong.

---

# 10. Proposed Solution 2: Boundary Repair

For every output position, determine whether its complete receptive field corresponds exactly to a previous cached receptive field.

If yes:

```text
REUSE
```

If no:

```text
DIRTY
```

Only dirty outputs are recomputed.

Conceptually:

```text
Conv1 output rows

D D D R R R R R R R R R R R R R R R R D D D

D = dirty
R = reusable
```

The exact number depends on:

- kernel size;
- padding;
- stride;
- window displacement;
- layer geometry.

This should be generated automatically by the reuse compiler.

---

# 11. Exact Dirty-Region Propagation

After the first layer, dirty positions propagate through subsequent layers.

Define a binary mask:

```text
dirty_l[p] = 1
```

if position `p` in layer `l` must be recomputed.

For a convolution layer:

```text
dirty_(l+1)[p] = 1
```

if the receptive field of output `p` intersects any dirty input position.

Mathematically:

\[
D_{l+1}(p)=
\bigvee_{q \in RF_l(p)} D_l(q)
\]

where:

- \(D_l\) is the dirty bitmap at layer \(l\);
- \(RF_l(p)\) is the input receptive field of output \(p\).

For a 3×3 depthwise convolution:

```text
input dirty region:
       XXXXX

output dirty region:
      XXXXXXX
```

For a 1×1 pointwise convolution:

```text
dirty region unchanged spatially
```

because each output position only depends on the same input spatial position.

This makes dependency propagation deterministic and exact.

---

# 12. Problem 3: HopCIM MFCC Normalization Can Change Old Frames

The MLPerf preprocessing used by HopCIM normalizes each one-second window by its maximum sample.

Therefore the 49 MFCC frames are not always independent.

When the one-second maximum changes:

```text
old frame features
```

may change even though the raw audio corresponding to that frame is unchanged.

HopCIM currently handles this correctly by recomputing all affected MFCC feature rows.

In the existing streaming measurements, the window maximum changed on approximately 11.7% of hops in one evaluated stream.

A simple CNN reuse controller that assumes only the new frame changes would therefore be incorrect.

---

# 13. Proposed Solution 3: Exact Quantized Feature-Difference Detection

Instead of predicting whether features changed, compare the actual quantized feature rows.

For the new window:

\[
F^{new}
\]

and the cached aligned same-phase window:

\[
F^{old}
\]

compare:

\[
\Delta F_{t,f}
=
\left(
F^{new}_{t,f}
\neq
F^{old}_{aligned(t),f}
\right)
\]

Because the comparison occurs **after**:

- normalization;
- flooring;
- DCT;
- requantization;

it automatically captures:

- window-maximum changes;
- quantization;
- saturation;
- rounding;
- new incoming frames.

The CNN reuse mechanism therefore operates on the true int8 model input.

This makes reuse correctness independent of why the feature changed.

---

# 14. Dynamic Dirty Input Mask

The feature comparison produces an initial dirty mask:

```text
49 × 10 feature matrix

0 = exactly unchanged
1 = changed
```

Example:

```text
row 0   0000000000
row 1   0000000000
...
row 43  0011000000
row 44  0000000000
row 45  1111111111
...
```

This mask becomes the input to the compiler-generated dependency propagation rules.

The hardware therefore recomputes only outputs whose exact input dependencies changed.

---

# 15. Proposed Solution 4: Value-Aware Dirty Propagation

A conservative dependency mask says:

> If an input changed, recompute all dependent outputs.

However, after recomputation the new quantized output may be exactly equal to the previously cached output.

Example:

```text
old input changed
        ↓
recompute convolution
        ↓
old output = 37
new output = 37
```

Then downstream layers do not need to be marked dirty.

Therefore after every recomputed activation:

```text
if new_value == old_cached_value:
    downstream_dirty = 0
else:
    downstream_dirty = 1
```

This produces **exact dynamic sparsity**.

There is no approximation threshold.

The reuse rule is:

\[
\text{reuse allowed}
\iff
\text{old int8 value} = \text{new int8 value}
\]

This means the optimization preserves bit-exact outputs.

---

# 16. Proposed Hardware: Exact Reuse Controller

Add a new hardware block around the existing sequencer.

```text
                     ┌─────────────────────────┐
FEAT ───────────────►│ Feature Difference Unit │
                     └─────────────┬───────────┘
                                   │
                              dirty bitmap
                                   ↓
                     ┌─────────────────────────┐
                     │ Exact Reuse Controller  │
                     │                         │
                     │ • phase tracking        │
                     │ • circular addressing   │
                     │ • boundary repair       │
                     │ • dirty propagation     │
                     │ • dirty iterator        │
                     └─────────────┬───────────┘
                                   │
                             dirty positions
                                   ↓
                     ┌─────────────────────────┐
                     │ Existing HopCIM         │
                     │ gather → CIM → requant  │
                     └─────────────┬───────────┘
                                   │
                           recomputed value
                                   ↓
                        old/new comparison
                          ↙             ↘
                     same              changed
                       ↓                  ↓
                 stop propagation   mark downstream
```

The key principle is:

> **Do not change the arithmetic datapath unless necessary. Change which positions are scheduled.**

This minimizes implementation risk.

---

# 17. Reuse the Existing HopCIM Datapath

The existing sequencer already performs:

```text
gather
  ↓
issue CIM pass
  ↓
receive result
  ↓
requantize
  ↓
write activation
```

The new controller should sit above this flow.

Current behavior:

```text
for every output position:
    compute
```

New behavior:

```text
for every output position:
    if dirty:
        compute
    else:
        reuse cached value
```

Therefore most of the existing verified arithmetic remains unchanged.

---

# 18. Circular Activation Cache

Copying entire activation maps every hop would waste memory bandwidth.

Instead store activations in a circular temporal buffer.

Each major activation tensor is approximately:

\[
25 \times 5 \times 64
=
8000 \text{ int8 values}
\]

or about:

\[
8 \text{ KB}
\]

Use a logical row offset:

```text
cache_base[phase]
```

For each same-phase update:

```text
cache_base = (cache_base + shift_rows) mod 25
```

Logical activation rows are mapped onto existing physical SRAM rows without copying the full tensor.

Only dirty rows/positions are rewritten.

---

# 19. Multi-Phase Cache Organization

For stride-2 alignment:

```text
Phase 0 state
Phase 1 state
```

Each cache entry should include:

```text
activation values
valid bits
epoch/version
logical row base
```

An epoch prevents stale cache reuse after:

- reset;
- model change;
- tile reload;
- feature-buffer discontinuity;
- stream restart.

---

# 20. Cache Size Is a Major Research Trade-Off

If five intermediate checkpoints are cached:

```text
Conv1
PW1
PW2
PW3
PW4
```

and each is approximately 8 KB:

\[
5 \times 8 \text{ KB}
=
40 \text{ KB/phase}
\]

For two phases:

\[
40 \times 2
=
80 \text{ KB}
\]

This is substantial.

Therefore the research must not claim that activation reuse is automatically efficient.

The key question becomes:

> **Does the energy and latency saved by skipped CIM work exceed the area and energy cost of the activation cache?**

This should become a central research question.

---

# 21. Proposed Cache-Budget-Aware Optimization

Do not always cache every possible checkpoint.

Consider cache budgets such as:

```text
8 KB
16 KB
24 KB
32 KB
48 KB
80 KB
```

For each candidate checkpoint set:

\[
C
\]

estimate:

\[
E(C)
=
E_{\text{CIM}}(C)
+
E_{\text{cache}}(C)
+
E_{\text{control}}(C)
\]

subject to:

\[
M_{\text{cache}}(C)
\le
M_{\text{budget}}
\]

Possible optimization goals:

### Minimum energy

\[
\min_C E(C)
\]

### Minimum latency

\[
\min_C T(C)
\]

### Minimum energy under deadline

\[
\min_C E(C)
\]

subject to:

\[
T(C) \le 20 \text{ ms}
\]

### Minimum energy-delay product

\[
\min_C E(C)\cdot T(C)
\]

The cache-selection problem turns the work into a more general architecture methodology.

---

# 22. ExactReuse Compiler / Analyzer

The strongest version of this project should not hard-code the reuse rules manually.

Build a small compiler/analyzer.

Suggested file:

```text
tools/reuse_analyzer.py
```

Input:

```text
model graph
layer dimensions
kernel size
stride
padding
quantization
CIM mapping
window displacement
cache budget
```

Output:

```text
phase count
phase alignment
boundary dirty masks
dirty propagation tables
checkpoint candidates
cache placement
expected CIM passes
expected cache reads/writes
estimated cycles
generated sequencer parameters
```

---

# 23. Compiler Analysis Flow

```text
TFLite / model description
          ↓
Parse operators
          ↓
Extract:
• shape
• kernel
• stride
• padding
• tensor lifetime
• quantization
          ↓
Compute temporal stride phase
          ↓
Compute aligned reusable regions
          ↓
Compute boundary repair masks
          ↓
Propagate dependency regions
          ↓
Generate checkpoint candidates
          ↓
Estimate:
• pass savings
• cache bytes
• cache accesses
• latency
          ↓
Select cache configuration
          ↓
Generate RTL tables / firmware metadata
```

---

# 24. Phase Count Analysis

For HopCIM:

```text
first convolution temporal stride = 2
```

Therefore:

```text
phase_count = 2
```

For a more general model with multiple temporal strides:

\[
P
=
\prod_l s_l
\]

may provide a conservative phase period, although the compiler should reduce it where operator structure permits.

This generalization should be verified carefully before claiming support for arbitrary networks.

---

# 25. Static Boundary Analysis

The compiler can statically determine which outputs cannot be reused because of padding and window displacement.

Inputs:

```text
input length
kernel
stride
padding mode
aligned window shift
```

Output:

```text
reusable output range
dirty left boundary
dirty right boundary
```

This logic can be generated once at compile time.

---

# 26. Dynamic Difference Analysis

Static analysis determines:

```text
what may be reusable
```

Dynamic analysis determines:

```text
what is actually unchanged
```

These two mechanisms should be combined.

```text
final_dirty =
static_boundary_dirty
OR
dynamic_input_change_dirty
```

This protects correctness while allowing additional reuse when actual quantized values remain unchanged.

---

# 27. Preliminary Analytical Feasibility Estimate

Before RTL, perform a first geometric estimate.

For two aligned windows separated by two MFCC frames, a possible conservative pattern for the current network is:

| Stage | Approx. dirty spatial positions | Approx. CIM passes |
|---|---:|---:|
| Conv1 | 30 / 125 | ~120 |
| DW1 + PW1 | 40 / 125 | ~560 |
| DW2 + PW2 | 50 / 125 | ~700 |
| DW3 + PW3 | 60 / 125 | ~840 |
| DW4 + PW4 | 70 / 125 | ~980 |
| Pool + FC | full | ~1 |
| **Total** | | **~3,201** |

Current:

\[
7501
\]

Potential reuse case:

\[
\approx3201
\]

Approximate reduction:

\[
\frac{7501-3201}{7501}
\approx
57\%
\]

This is **only a feasibility estimate**.

It must not be reported as a result until verified by the exact software model and RTL.

---

# 28. Research Questions

## RQ1 — Correctness

Can an unchanged sliding-window quantized CNN reuse intermediate activations despite:

- SAME padding;
- temporal stride;
- TFLite quantization;
- window-dependent preprocessing;

while remaining bit-exact with independent full-window inference?

---

## RQ2 — Computation Reduction

How much can exact reuse reduce:

- CIM passes;
- CIM active cycles;
- total inference latency;
- data movement;

relative to complete recomputation?

---

## RQ3 — Cache Trade-Off

When does the energy saved by skipped CIM operations exceed the:

- SRAM area;
- SRAM read/write energy;
- reuse-controller cost;

of the activation cache?

---

## RQ4 — Generality

Can a compiler automatically generate correct reuse schedules for more than one streaming CNN architecture?

---

# 29. Main Hypotheses

Suggested hypotheses:

## H1 — Exactness

The reuse architecture produces:

\[
0
\]

mismatching final int8 logits against independent full inference over the complete evaluation stream.

---

## H2 — Computation reduction

Average CIM-pass count decreases by at least:

\[
30\%
\]

relative to full recomputation.

This is a suggested go/no-go threshold, not a predicted result.

---

## H3 — Energy usefulness

After including activation-cache energy, the reuse architecture decreases modeled accelerator energy per decision.

A useful target might be:

\[
15\%-20\%
\]

or more.

Again, this is a research target, not a claimed result.

---

## H4 — Generality

The compiler reproduces correct reuse schedules for at least two model/workload configurations.

---

# 30. Strong Baselines

The paper must compare against strong alternatives.

## B0 — Full current HopCIM

Current R6b:

```text
7,501 passes
~85,352 cycles
```

---

## B1 — Strengthened HopCIM

The current sequencer is partly limited by one requantization lane.

HopCIM's own analysis indicates that adding a second requantization lane could reduce full-network execution to roughly:

```text
~67.9k cycles
```

This baseline is important.

Otherwise reviewers can argue that a much simpler hardware modification could provide the same benefit.

---

## B2 — Naïve sliding reuse

Reuse aligned activations without proper padding repair.

Expected:

```text
fewer calculations
but incorrect outputs
```

This demonstrates why the research problem exists.

---

## B3 — Static exact boundary reuse

Use:

```text
phase handling
+
padding repair
```

but no dynamic value comparison.

---

## B4 — Dynamic FEAT difference

Add exact feature comparison.

---

## B5 — Value-aware propagation

Add old/new activation comparison after recomputation.

This should provide the strongest reuse.

---

# 31. Key Ablation Studies

Run at least these ablations:

1. **No phase awareness**
2. **No padding repair**
3. **No normalization-change handling**
4. **Static boundary reuse only**
5. **Dynamic FEAT comparison**
6. **Value-aware propagation**
7. **Different cache sizes**
8. **One versus two phase caches**
9. **Current versus strengthened two-requantizer baseline**

The purpose is to show exactly which mechanism creates the improvement.

---

# 32. Implementation Plan

## Phase 1 — Software feasibility model

Do **not** begin with RTL.

Extend:

```text
model/dscnn_int.py
model/stream_int.py
```

Create:

```text
model/exact_reuse.py
```

Implement:

```text
FullReference
ExactReuseReference
```

For each decision:

```python
assert reuse_logits == full_logits
assert reuse_class == full_class
```

Preferably verify checkpoint tensors too:

```text
Conv1
PW1
PW2
PW3
PW4
pool
logits
```

---

# 33. Software Go/No-Go Gate

Continue only if:

```text
0 logit mismatches
```

and average pass reduction is approximately:

```text
>= 30%
```

on realistic continuous streams.

If the gain is much smaller, stop before RTL.

This prevents months of work on a weak research direction.

---

# 34. Phase 2 — Compiler/Analyzer

Implement:

```text
tools/reuse_analyzer.py
```

Responsibilities:

- parse model geometry;
- calculate phase period;
- compute alignment;
- compute boundary masks;
- propagate dirty regions;
- estimate cache occupancy;
- estimate CIM pass count;
- choose checkpoint placement;
- generate RTL lookup tables.

---

# 35. Phase 3 — RTL Design

Suggested modules:

```text
rtl/reuse_ctrl.sv
rtl/reuse_dirty.sv
rtl/reuse_cache.sv
```

Modify:

```text
rtl/cim_seq.sv
```

but preserve the existing arithmetic path wherever possible.

New controller responsibilities:

```text
phase selection
cache validity
logical/physical row mapping
dirty-mask iteration
reuse decision
cache update
downstream dirty propagation
```

---

# 36. Phase 4 — FPGA Integration

Integrate into the existing:

```text
cim_endpoint
```

and keep the same:

```text
X-HEEP
Nexys A7
audio pipeline
MFCC
network weights
TFLite model
```

This gives a clean apples-to-apples comparison.

---

# 37. Verification Strategy

The verification standard should be unusually strict.

---

## 37.1 Golden Reference

Continue using:

```text
TensorFlow Lite BUILTIN_REF
```

as the neural reference.

The full HopCIM integer model should remain the intermediate golden model.

The new optimization must preserve:

```text
every final int8 logit
```

not merely the predicted class.

---

## 37.2 Python Differential Testing

For every streaming decision:

```text
full inference
vs
reuse inference
```

Compare:

```text
input features
dirty masks
layer outputs
logits
class
```

Store the first mismatch with full state for debugging.

---

## 37.3 cocotb + Verilator

Use:

- Verilator;
- cocotb.

cocotb lets the Python reference directly drive RTL simulation.

Recommended flow:

```text
Python exact model
       ↓
     cocotb
       ↓
    Verilator
       ↓
      RTL
```

Useful checks:

```python
assert rtl_phase == py_phase
assert rtl_dirty == py_dirty
assert rtl_logits == py_logits
```

Public tools:

Verilator:
https://github.com/verilator/verilator

cocotb:
https://www.cocotb.org/

---

# 38. Formal Verification

Use SymbiYosys selectively for controller properties.

Tool:
https://github.com/YosysHQ/sby

Do not try to formally verify the entire neural network.

Prove local properties such as:

### Cache safety

```text
reuse -> cache_valid
```

### Phase safety

```text
reuse -> cached_phase == current_phase
```

### Epoch safety

```text
cache_epoch != current_epoch -> !reuse
```

### Dirty scheduling

```text
dirty_position -> eventually_recomputed
```

### Layer completion

```text
layer_done -> all_dirty_outputs_committed
```

### No stale reuse after reset

```text
reset -> all_cache_valid = 0
```

These proofs would strengthen the reliability story significantly.

---

# 39. Mutation Testing

Intentionally inject controller errors.

Examples:

```text
wrong phase
wrong cache offset
ignore left padding
ignore right padding
skip one dirty position
ignore normalization update
reuse invalid cache entry
wrong epoch
incorrect dirty dilation
```

The regression suite must catch every mutation.

Report something like:

```text
12/12 injected faults detected
```

rather than only stating that simulation passed.

---

# 40. Gate-Level Verification

The existing HopCIM project already discovered synthesis-related signedness issues.

Therefore the new design should repeat:

```text
RTL
↓
Vivado synthesis
↓
gate-level simulation
↓
compare with Python/RTL reference
```

At minimum test:

- phase transitions;
- cache wrap-around;
- normalization changes;
- reset;
- boundary dirty regions;
- long continuous streams.

---

# 41. Public Benchmark 1 — MLPerf Tiny KWS

Keep the existing MLPerf Tiny DS-CNN / Speech Commands setup.

This preserves continuity with current HopCIM results.

For streaming tests, construct deterministic continuous streams containing:

```text
speech clips
silence
background noise
keyword transitions
non-keyword transitions
```

Store a manifest:

```text
clip ID
start offset
noise source
noise gain
seed
```

so every experiment is reproducible.

---

# 42. Public Benchmark 2 — MLPerf Tiny Streaming Wakeword

MLPerf Tiny v1.3 introduced a dedicated continuous streaming wake-word benchmark.

Reference:
https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/

The benchmark uses continuous audio and measures wake-word detection behavior in a realistic streaming context.

This is highly relevant because ExactReuse-CIM is specifically designed for continuous overlapping inference.

Possible evaluation levels:

### Minimum

Run the reuse compiler / cycle model on the public streaming network.

### Stronger

Map the second model into the HopCIM software mapping model.

### Best

Run it in actual RTL.

The second workload would substantially improve claims of generality.

---

# 43. Prior-Work Software Baselines

Where practical, compare or analyze against:

## StreamiNNC

https://github.com/esl-epfl/streaminnc

Use primarily to compare:

```text
streaming reuse
padding behavior
alignment
exactness
```

---

## Continual Inference

https://github.com/lukashedegaard/continual-inference

Use as a reference for stateful continual model execution.

The comparison does not need to be cycle-for-cycle hardware benchmarking if model semantics differ.

A conceptual and software-level comparison may be sufficient if clearly explained.

---

# 44. FPGA Metrics

For every design report:

| Metric | Purpose |
|---|---|
| CIM passes / decision | direct compute reduction |
| engine active cycles | accelerator activity |
| network cycles | latency |
| total hop cycles | real-time behavior |
| p50 / p90 / p95 / p99 | input-dependent behavior |
| worst-case cycles | hard deadline |
| cache reads | memory traffic |
| cache writes | memory traffic |
| LUT | logic overhead |
| FF | controller state |
| BRAM | cache overhead |
| DSP | arithmetic overhead |
| Fmax / WNS | timing impact |

Do not report only average latency.

---

# 45. Power and Energy Evaluation

The current FPGA implementation emulates the CIM using FPGA resources.

Therefore:

> FPGA power is **not** equivalent to physical CIM silicon energy.

The study should explicitly separate three evaluation levels.

---

## 45.1 FPGA Dynamic Energy

Use Vivado post-route activity.

Possible flow:

```text
post-route simulation
        ↓
SAIF
        ↓
Vivado read_saif
        ↓
report_power
```

Use the result only as:

> FPGA implementation energy comparison.

Do not claim:

> actual CIM silicon energy.

---

# 46. CiMLoop for CIM Energy Modeling

Use the open-source CiMLoop framework:

Paper/project:
https://emze.csail.mit.edu/cimloop

Repository:
https://github.com/mit-emze/cimloop

CiMLoop can help model:

- CIM computation energy;
- data movement;
- architecture;
- mapping;
- different CIM assumptions.

The key research question is:

> How much energy does skipping CIM passes save compared with the extra SRAM read/write energy caused by the activation cache?

Do sensitivity analysis rather than relying on one assumed macro.

Sweep:

```text
CIM energy/pass
SRAM energy/access
cache size
reuse rate
```

Then identify the break-even region.

---

# 47. ASIC-Style Controller and SRAM Evaluation

For the additional digital hardware:

## OpenROAD

https://github.com/The-OpenROAD-Project/OpenROAD

Use it for:

```text
RTL → synthesis → place/route → timing/area
```

Possible public platforms:

```text
SKY130
GF180
Nangate45
ASAP7
```

---

## OpenRAM

https://openram.org/

Use OpenRAM to model the activation-cache SRAM.

It can provide:

```text
SRAM area
timing
power model
layout/netlist
```

Keep this analysis separate from FPGA and modeled CIM data.

Do not combine unrelated technology numbers into one fake total-silicon result.

---

# 48. Experimental Design Matrix

Recommended central evaluation table:

| Design | Exact | Retraining | Cache | CIM passes | Cycles | BRAM/SRAM | Energy |
|---|---|---|---:|---:|---:|---:|---:|
| Full R6b | Yes | No | 0 | 7,501 | 85,352 | baseline | baseline |
| R6b + 2RQ | Yes | No | 0 | 7,501 | ~67.9k est. | baseline | measure |
| Naïve reuse | No | No | varies | low | low | varies | — |
| Phase-aware | Yes | No | varies | measure | measure | measure | measure |
| Boundary-aware | Yes | No | varies | measure | measure | measure | measure |
| Dynamic FEAT | Yes | No | varies | measure | measure | measure | measure |
| Value-aware | Yes | No | varies | minimum | minimum | maximum | measure |

The `~67.9k` strengthened baseline must be measured after implementation before publication.

---

# 49. Cache-Budget Sweep

Recommended configurations:

```text
0 KB
8 KB
16 KB
24 KB
32 KB
48 KB
80 KB
```

For each configuration plot:

```text
CIM passes
cycles
cache accesses
energy
area
```

Create a Pareto frontier:

```text
energy
  ↑
  |
  |       x
  |   x
  | x
  +----------------→ cache area
```

The goal is to identify whether there is a useful middle point rather than blindly maximizing cache size.

---

# 50. Statistical Reporting

Because dynamic reuse depends on the audio stream, report distributions.

At minimum:

```text
mean
median
p90
p95
p99
maximum
```

Also report:

```text
dirty FEAT cells / decision
dirty positions / layer
CIM passes / layer
cache hits
cache misses
value-aware cancellation rate
normalization-change rate
```

This explains why the architecture works.

---

# 51. Go/No-Go Research Gates

## Gate 1 — Software exactness

Requirement:

```text
0 final logit mismatches
```

across the selected continuous test streams.

If this fails, do not proceed to RTL.

---

## Gate 2 — Computation reduction

Suggested threshold:

\[
\ge 30\%
\]

average CIM-pass reduction.

If the reduction is below approximately 10–15%, the direction is probably too weak.

---

## Gate 3 — Cache cost

Estimate SRAM energy and area.

Continue only if:

```text
saved CIM energy
>
extra cache + controller energy
```

over a reasonable technology range.

---

## Gate 4 — RTL benefit

After implementation, compare against:

```text
current R6b
strengthened R6b + second requantizer
```

If a simple baseline improvement performs nearly as well, the paper contribution becomes weak.

---

# 52. Suggested Development Timeline

## Weeks 1–2

Software exact reuse:

```text
phase alignment
boundary repair
dirty propagation
```

---

## Weeks 3–4

Add:

```text
dynamic FEAT comparison
value-aware propagation
```

Run long-stream analysis.

---

## Weeks 5–6

Build cache-budget model.

Use:

```text
pass counts
cache access counts
rough energy
```

Make go/no-go decision.

---

## Weeks 7–9

Build compiler/analyzer.

Generate reuse tables automatically.

---

## Weeks 10–14

RTL implementation:

```text
reuse_ctrl
dirty maps
phase cache
sequencer changes
```

---

## Weeks 15–17

Verification:

```text
cocotb
Verilator
formal properties
mutation tests
gate-level simulation
```

---

## Weeks 18–20

Evaluation:

```text
Nexys A7
Vivado resources
SAIF power
CiMLoop
cache sweep
public streaming benchmark
```

---

## Weeks 21–24

Paper/thesis writing:

```text
architecture
methodology
ablation
related work
limitations
reproducibility package
```

---

# 53. Expected Contribution Set

If successful, the thesis/paper could claim:

## Contribution 1 — Exact streaming reuse algorithm

A method for reusing intermediate activations in sliding-window quantized CNN inference while preserving:

```text
stride
SAME padding
quantization
full-window semantics
```

without retraining.

---

## Contribution 2 — Exact dirty-region propagation

A static/dynamic dependency mechanism combining:

```text
phase alignment
boundary repair
actual input differences
value-aware output equality
```

to determine exactly which outputs require recomputation.

---

## Contribution 3 — Digital CIM reuse architecture

A hardware controller that converts sparse dirty positions into:

```text
skipped CIM passes
```

while reusing HopCIM's existing arithmetic datapath.

---

## Contribution 4 — Cache-budget-aware reuse compiler

A compiler/analyzer that determines:

```text
phase structure
cache placement
reuse regions
expected passes
cache requirements
```

from network geometry.

---

## Contribution 5 — Rigorous evaluation

An evaluation across:

```text
bit-exact correctness
CIM passes
latency
cache cost
FPGA resources
FPGA dynamic energy
modeled CIM energy
streaming benchmark
```

---

# 54. What Would Count as a Strong Result?

A strong result might look like:

```text
0 mismatching logits
>30% fewer CIM passes
>20% lower total modeled energy
manageable cache increase
deadline always satisfied
```

A particularly strong paper might achieve something like:

```text
passes:   -50%
cycles:   -40%
energy:   -25%
cache:    +16–32 KB
accuracy: identical
```

These numbers are **examples of useful outcomes**, not predictions.

---

# 55. What Would Falsify the Research Hypothesis?

The idea should be considered unsuccessful if:

### Case 1

```text
passes saved < 10%
```

after exactness constraints.

### Case 2

```text
cache energy > saved CIM energy
```

for realistic architectures.

### Case 3

The required cache is so large that:

```text
area/energy becomes worse than full recomputation.
```

### Case 4

A simple second requantization lane nearly matches the benefit with negligible overhead.

### Case 5

The approach only works for one model and cannot be meaningfully generalized.

Negative results can still contribute to the thesis but would weaken regular-conference publication potential.

---

# 56. Publication Positioning

Do **not** claim:

```text
first incremental KWS
first streaming CNN reuse
first activation reuse
first delta CNN
```

These are already established.

Potential positioning:

> We propose a model-preserving incremental execution method for independently evaluated sliding-window quantized CNNs. Unlike approaches that redesign or retrain models for continual execution, the proposed method preserves the original stride, padding, quantization, and independent-window semantics. A phase-aware reuse controller and compiler identify exact reusable activation regions and eliminate redundant digital-CIM passes without changing the pretrained model.

This wording should still undergo a final literature review before publication.

---

# 57. Comparison Against Prior Work

A useful comparison table for the paper:

| Work | Overlap reuse | Same pretrained model | SAME padding preserved | Stride-aware | Exact independent-window output | CIM hardware |
|---|---|---|---|---|---|---|
| Shan et al. JSSC 2021 | Yes | Modified architecture/training | Avoided/constrained | Constrained | For modified model | No |
| Continual Inference | Yes | Converted operators | Framework-dependent | Yes | Continual semantics | No |
| DeltaCNN CVPR 2022 | Yes | Yes | General | General | Can use approximation | GPU |
| StreamiNNC AAAI 2025 | Yes | Often | Padding is a central difficulty | Alignment handling | Nonzero deviation in pretrained cases | No |
| **ExactReuse-CIM** | Yes | **Yes** | **Yes** | **Phase-aware** | **Target: bit-exact** | **Yes** |

This table must be updated after reading the complete original papers before publication.

---

# 58. Reproducibility Requirements

Freeze and publish where permitted:

```text
Git commit
RTL commit
Python model commit
TFLite model SHA256
dataset manifest
stream-generation seed
Verilator version
Vivado version
cocotb version
compiler version
build flags
FPGA bitstream hash
CiMLoop configuration
OpenROAD configuration
OpenRAM configuration
```

Recommended commands:

```bash
make verify-python
make verify-rtl
make verify-formal
make verify-mutations
make benchmark-gsc-stream
make benchmark-mlperf-stream
make synth
make fpga-power
make cim-energy
make reproduce-paper
```

---

# 59. Recommended Repository Structure

```text
hopcim/
├── model/
│   ├── dscnn_int.py
│   ├── stream_int.py
│   ├── exact_reuse.py
│   └── reuse_reference.py
│
├── tools/
│   ├── reuse_analyzer.py
│   ├── gen_reuse_tables.py
│   ├── make_stream_manifest.py
│   └── analyze_reuse_results.py
│
├── rtl/
│   ├── cim_seq.sv
│   ├── reuse_ctrl.sv
│   ├── reuse_dirty.sv
│   └── reuse_cache.sv
│
├── tb/
│   ├── test_reuse.py
│   ├── test_phase.py
│   ├── test_padding.py
│   ├── test_cache_wrap.py
│   └── formal/
│
├── benchmark/
│   ├── gsc_stream/
│   └── mlperf_streaming/
│
├── energy/
│   ├── cimloop/
│   └── fpga_power/
│
└── results/
```

---

# 60. Recommended First Experiment

Before writing any RTL:

1. Run the existing integer DS-CNN on a continuous GSC stream.
2. Save every layer output for every decision.
3. Compare decision `N` with decision `N+2`.
4. Determine exactly which activation positions are identical after phase alignment.
5. Implement conservative padding-aware reuse.
6. Propagate dirty regions.
7. Count required CIM passes.
8. Verify every final logit against full inference.

The first research question should be answered numerically:

> **Across a realistic continuous speech stream, how many of HopCIM's 7,501 CIM passes can be eliminated while preserving every output logit exactly?**

Only after answering this should RTL development begin.

---

# 61. Recommended Decision Rule

If software results show:

```text
>= 30% average pass reduction
0 logit mismatches
reasonable cache budget
```

then proceed with ExactReuse-CIM.

If not, stop and pivot before investing months in RTL.

This makes the project research-driven rather than implementation-driven.

---

# 62. Final Proposed Research Story

The complete thesis/paper story would be:

```text
HopCIM already moves the whole KWS pipeline into hardware.
        ↓
The CPU is no longer the main bottleneck.
        ↓
The accelerator still recomputes almost the same CNN every 20 ms.
        ↓
Existing incremental CNN techniques do not automatically preserve
independent padded/strided quantized model semantics.
        ↓
We introduce exact phase-aware reuse.
        ↓
A compiler derives padding boundaries, stride phases,
dependency propagation and cache placement.
        ↓
Hardware skips only provably redundant CIM passes.
        ↓
All logits remain bit-exact with full TFLite inference.
        ↓
We quantify when skipped compute is worth the added cache.
```

This is a substantially stronger research narrative than simply extending HopCIM with another accelerator optimization.

---

# 63. Final Recommendation

**Primary direction:** pursue ExactReuse-CIM.

**First priority:** software feasibility and exactness.

**Second priority:** quantify cache/computation trade-offs.

**Third priority:** only then implement RTL.

The strongest possible final contribution is not:

> "HopCIM can reuse activations."

It is:

> **A compiler-guided, phase- and boundary-aware digital-CIM execution method that removes redundant sliding-window neural computation while preserving the exact behavior of an unchanged pretrained quantized model.**

That is the research hypothesis that should now be tested.

---

# References and Public Tools

## Research papers

1. W. Shan et al., "A 510-nW Wake-Up Keyword-Spotting Chip Using Serial-FFT-Based MFCC and Binarized Depthwise Separable CNN in 28-nm CMOS," IEEE JSSC, 2021. DOI: 10.1109/JSSC.2020.3029097.

2. C. Kechris, J. Dan, J. Miranda, and D. Atienza, "Don't Think It Twice: Exploit Shift Invariance for Efficient Online Streaming Inference of CNNs," AAAI 2025.  
   https://arxiv.org/abs/2408.03223

3. L. Hedegaard and A. Iosifidis, "Continual Inference: A Library for Efficient Online Inference with Deep Neural Networks in PyTorch," 2022.  
   https://arxiv.org/abs/2204.03418

4. M. Parger et al., "DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos," CVPR 2022. DOI: 10.1109/CVPR52688.2022.01217.

5. T. Andrulis, J. S. Emer, and V. Sze, "CiMLoop: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool," ISPASS 2024.

## Benchmark resources

MLPerf Tiny:
https://github.com/mlcommons/tiny

MLPerf Tiny Streaming benchmark:
https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/

## Streaming inference tools

StreamiNNC:
https://github.com/esl-epfl/streaminnc

Continual Inference:
https://github.com/lukashedegaard/continual-inference

## RTL and verification

Verilator:
https://github.com/verilator/verilator

cocotb:
https://www.cocotb.org/

SymbiYosys:
https://github.com/YosysHQ/sby

## CIM modeling

CiMLoop:
https://github.com/mit-emze/cimloop

## Physical design / SRAM

OpenROAD:
https://github.com/The-OpenROAD-Project/OpenROAD

OpenRAM:
https://openram.org/

---

# Important Claim Boundary

This document proposes a research direction.

The following are **not yet measured results**:

- percentage of CIM passes saved;
- cycle reduction;
- cache size needed for the optimum implementation;
- energy reduction;
- FPGA resource overhead;
- generality across a second model.

All such values must be experimentally established before being used as research claims.

The ~57% pass-reduction example is only a preliminary geometric feasibility estimate.

The key success criterion remains:

\[
\boxed{
\text{lower accelerator work}
\quad+\quad
\text{manageable cache cost}
\quad+\quad
\text{zero output mismatch}
}
\]

