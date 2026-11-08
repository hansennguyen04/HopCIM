# HopCIM-StreamFuse Research Proposal
## Cross-Layer Line Buffering + Depthwise/Pointwise Direct Forwarding

**Project:** HopCIM  
**Direction:** Quantization-preserving, communication-minimal cross-layer dataflow for a resident-weight digital CIM  
**Goal:** Reduce intermediate activation traffic, activation-buffer size, and estimated energy while preserving the unchanged pretrained model and exact TFLite quantization semantics.

---

Yes. After looking at your current HopCIM dataflow and recent accelerator/CIM literature, I think your instinct is good, but I would narrow it substantially.

The promising bottleneck is **not weight fetching from the CPU anymore**—HopCIM already keeps the weights resident in the CIM. It is also not MFCC-to-CNN transfer, because that tensor is tiny. The remaining data-movement opportunity is mainly:

> **Repeated access to intermediate activation maps, especially around the four depthwise + pointwise blocks.**

I would investigate a new direction I’ll call **HopCIM-StreamFuse**:

> **A quantization-preserving, cross-layer streaming dataflow that keeps intermediate activations in small line buffers / local vectors and forwards them directly between depthwise and pointwise layers instead of repeatedly materializing whole feature maps in ACT SRAM.**

I think this is a better match to what you are asking than activation reuse.

---

# 1. What bottlenecks are common in this class of accelerator?

A recurring lesson in recent CIM literature is that accelerating the MAC itself does **not** automatically make the whole accelerator efficient. System-level efficiency becomes limited by mapping, activation movement, intermediate buffers, peripheral processing, and underutilization. A 2024 TCAD study of digital CIM running full MLPerf workloads found that dataflow/macro-level optimization improved macro utilization by 3.04× and reduced EDP to 0.34× the initial design. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10318181?utm_source=chatgpt.com)

Recent CIM-specific dataflow research reaches the same conclusion. MIREDO, ASP-DAC 2026, explicitly models memory transfer, synchronization and CIM-specific stalls and reports up to 3.2× improvement through dataflow optimization alone. [ci-lab.net](https://www.ci-lab.net/uploads/paper/aspdac26_miredo.pdf?utm_source=chatgpt.com) A 2025 dataflow-centric CIM study likewise treats capacity, bandwidth, reuse and memory access costs as first-order system variables rather than focusing only on MAC throughput. [IEEE Xplore](https://ieeexplore.ieee.org/document/11096377/?utm_source=chatgpt.com)

For your type of DS-CNN, the recurring bottlenecks are approximately:

| Bottleneck                          | General literature             | HopCIM status                               |
| ----------------------------------- | ------------------------------ | ------------------------------------------- |
| Weight movement                     | Major in normal accelerators   | **Mostly solved** — weights resident in CIM |
| CPU↔accelerator orchestration       | Often significant              | **Solved** by R6b sequencer                 |
| Requantization / nonlinear epilogue | Common end-to-end bottleneck   | **Still present**                           |
| Depthwise mapping/utilization       | **Major**                      | **Still important**                         |
| Intermediate activation traffic     | **Major**                      | **Still important**                         |
| Full feature-map buffering          | Major in lightweight CNNs      | **Present: 2×8 KB ACT**                     |
| Repeated spatial-window fetch       | Major for DW convolution       | **Present**                                 |
| Temporal redundant decisions        | KWS-specific issue             | Already studied with \\(k\\)-hop            |
| Cache leakage                       | Important if adding reuse SRAM | ExactReuse exposed this problem             |

In other words, your project has already removed many of the obvious bottlenecks. That actually makes the activation dataflow problem more interesting.

---

# 2. Why depthwise-separable networks are especially problematic

Depthwise convolution reduces MAC count dramatically, but that comes with a side effect:

\\[ \text{arithmetic intensity} = \frac{\text{operations}} {\text{bytes moved}} \\]

becomes low.

So although DW convolution looks computationally cheap, moving its input/output data can dominate.

This is widely reported. A 2024 study of fused depthwise-pointwise GPU kernels explicitly describes depthwise/pointwise operations as memory-access bottlenecked; their fusion reduced memory accesses by up to 83% and gave up to 3.7× kernel speedup. [arXiv](https://arxiv.org/abs/2404.19331?utm_source=chatgpt.com)

On an ultra-low-power GAP8 device, fusing DW/PW kernels reduced activation movement between memory levels by up to 52.97%, although end-to-end latency improved by a more modest 11.4%. [arXiv](https://arxiv.org/abs/2406.12478?utm_source=chatgpt.com)

DepFiN attacked the same issue with **depth-first execution**: instead of computing an entire layer, storing the whole feature map, and then starting the next layer, it switches between layers so intermediate data can remain local. Its motivation is exactly the cost of full intermediate feature maps. [IEEE Xplore](https://ieeexplore.ieee.org/document/9919231/?utm_source=chatgpt.com)

And FuseCU at DAC 2025 derives profitable operator fusion from communication cost; it reports roughly 39–64% data-movement reductions, but much smaller speedups of roughly 1.14–1.33×. [IEEE Xplore](https://ieeexplore.ieee.org/document/11132765/?utm_source=chatgpt.com)

That last result is an important warning for us:

> **Reducing memory access can substantially reduce energy without producing an equally large latency reduction.**

That is likely to be true for HopCIM too.

---

# 3. This problem is already visible inside HopCIM

Your current sequencer stores activations as:

\\[ 2\times25\times5\times64 \\]

int8 values.

That's:

\\[ 2\times8000\text{ B} = \boxed{16\text{ KB}} \\]

implemented using seven RAMB36 blocks.

The execution is layer-by-layer and ping-pongs between the two ACT maps. The depthwise gather uses one ACT word for each tap, while pointwise gathers channel groups from ACT.    SEQUENCER

Conceptually:

```
PW0 output
    ↓
write whole ACT map
    ↓
read overlapping 3×3 windows
    ↓
DW1
    ↓
write whole DW map
    ↓
read whole DW map
    ↓
PW1
    ↓
write whole PW map
    ↓
...
```

This is correct and fast enough, but not data-movement minimal.

---

# 4. There are actually two different types of unnecessary traffic

## A. Re-reading overlapping 3×3 neighborhoods

Consider one depthwise layer:

\\[ 25\times5\times64. \\]

For a 3×3 convolution, neighboring output pixels reuse many of the same input activations.

For example:

```
Output 0 needs

a b c
d e f
g h i

Output 1 needs

b c j
e f k
h i l
```

Values `b,c,e,f,h,i` are read again.

Your current depthwise gather reconstructs the window by reading activation words for the required taps.

A classical line-buffer architecture instead does:

```
SRAM
 ↓
one sequential stream
 ↓
┌───────────────────┐
│ Row -2            │
│ Row -1            │
│ Current row       │
└───────────────────┘
        ↓
      3×3 window
```

and reuses values locally.

For your exact \\(25\times5\\) geometry, the number of valid 3×3 spatial references is:

\\[ 949 \\]

per channel.

Across 64 channels:

\\[ 949\times64 = 60,736 \\]

activation-value uses per depthwise layer.

But there are only:

\\[ 25\times5\times64 = 8,000 \\]

unique input activations.

So the theoretical reuse factor is approximately:

\\[ \boxed{7.59\times} \\]

for the input tensor.

This does **not** mean you get 7.59× speedup. It means your large activation memory is serving the same values repeatedly.

---

# 5. The second waste: DW output is immediately consumed by PW

Your block is:

```
Depthwise 3×3
     ↓
requantize
     ↓
ACT SRAM
     ↓
Pointwise 1×1
```

But pointwise needs exactly the 64 depthwise outputs for that same spatial position.

Therefore this:

```
DW result
 ↓
write SRAM
 ↓
read SRAM
 ↓
PW
```

can instead become:

```
DW result
 ↓
TFLite-exact requantization
 ↓
small 64-byte local vector
 ↓
PW immediately
```

The important point is:

> We are **not algebraically merging** DW and PW.

That would generally break TFLite semantics because there is quantization/requantization between them.

Instead we preserve:

```
DW
 → exact requantization
 → PW
```

but eliminate the unnecessary memory round trip.

This is much safer for HopCIM because model preservation remains intact.

---

# 6. My proposed direction: HopCIM-StreamFuse

I would investigate:

## Quantization-Preserving Cross-Layer Streaming Dataflow for a Resident-Weight Digital CIM

The idea has three connected mechanisms.

### 1. Rolling activation line buffers

Instead of repeatedly gathering 3×3 neighborhoods from the 8-KB ACT map, sequentially stream each input activation into a 3-row circular buffer.

For one DS-CNN layer:

\\[ 3\times5\times64 = 960\text{ B}. \\]

Then construct all neighboring 3×3 windows from that local storage.

---

### 2. Direct DW→PW result forwarding

After the ten channel groups of one spatial position have been processed:

```
DW CIM passes
      ↓
requantizer
      ↓
64-byte fusion vector
      ↓
PW CIM passes
```

No DW feature map is written to the main ACT SRAM.

A particularly nice HopCIM implementation may be able to use the idle **X bank** as part of this forwarding path:

```
CIM DW result
    ↓
requantize
    ↓
write centered PW input directly into X bank
    ↓
launch PW tile
```

That is quite specific to your two-X-bank bit-serial engine.

---

### 3. Depth-first execution across blocks

The more ambitious version avoids full PW feature-map materialization too.

Instead:

```
Conv1
 ↓
DW1 → PW1
       ↓
       line buffer for DW2
            ↓
          DW2 → PW2
                   ↓
                 line buffer DW3
                        ↓
                       ...
```

Once enough rows exist to compute the next 3×3 operation, switch the CIM tile ID and proceed downstream.

Because HopCIM already stores all layer weights in CIM tiles, switching layers does **not require reloading weights**.

That is one reason this idea fits HopCIM unusually well.

---

# 7. Why resident weights are important

Many fused accelerators have to worry about:

```
compute layer A
↓
load weights B
↓
compute layer B
↓
reload A
...
```

which can destroy the benefit of fine-grained layer switching.

HopCIM is different.

All network tiles are already resident.

Current mapping:

\\[ 61\text{ network tiles} \\]

and the sequencer simply selects the appropriate tile for each pass.

So a schedule such as:

```
DW1 tile
DW1 tile
...
PW1 tile
PW1 tile
DW2 tile
...
```

doesn't imply a weight-transfer operation.

That makes fine-grained layer interleaving much more practical.

This is where I see a plausible HopCIM-specific contribution.

---

# 8. How much memory could this reduce?

Current main activation storage:

\\[ 2\times8000\text{ B} = 16\text{ KB}. \\]

A fully depth-first version could approximately use a few line buffers instead.

One 3-row buffer:

\\[ 3\times5\times64 = 960\text{ B}. \\]

Four DW blocks:

\\[ 4\times960 = 3840\text{ B}. \\]

Add local vectors, control information and banking, and you might end up in the vicinity of:

\\[ \boxed{4-6\text{ KB}} \\]

instead of 16 KB.

That is only an architectural estimate. FPGA BRAM granularity may make the physical result less dramatic.

But it is worth synthesizing.

---

# 9. More importantly: memory accesses could fall much further

Using your ACT organization, one depthwise layer requires roughly 9,490 valid channel-group/tap word accesses from the full activation store, while feeding the tensor once into a rolling buffer would require roughly 1,250 source words.

So approximately:

\\[ 9490 \rightarrow 1250 \\]

or:

\\[ \boxed{\sim7.6\times} \\]

fewer reads from the large activation store for that part of the layer.

Across all four depthwise blocks, approximately:

\\[ 37,960 \rightarrow 5,000 \\]

large-buffer reads.

Then direct DW→PW forwarding removes another class of transactions:

```
DW writes → ACT
PW reads  ← ACT
```

which is roughly another 10,000 packed-word transactions across the four blocks in your current organization.

My rough analytical estimate is therefore that a carefully fused schedule might reduce **large ACT-buffer transactions by roughly 70–80%**.

That is not yet a measurement. The first thing to do is instrument the existing sequencer and obtain the exact count.

---

# 10. But here is the critical limitation

This idea probably **will not dramatically reduce network latency by itself**.

Your network currently takes:

\\[ 85,352\text{ cycles}. \\]

But the CIM engine itself is active for:

\\[ 67,509\text{ cycles} \\]

or 79% of the run.

And your depthwise layers already run with **no engine bubbles**. Your own document estimates that adding a second requantization lane would reduce the network to about 67.9k cycles, already close to the engine bound.    SEQUENCER

Therefore:

\\[ \boxed{ \text{dataflow fusion mostly attacks memory energy/area, not CIM arithmetic cycles} } \\]

A reviewer will notice this immediately.

So I would **not sell the proposal as primarily a speedup technique**.

The strongest story is:

> **Reduce activation-memory footprint and access energy while preserving exact model behavior and maintaining engine-rate execution.**

Any latency reduction is a secondary benefit.

---

# 11. If you want runtime improvement too, there is a useful prior technique

Your depthwise layers account for:

\\[ 5000 / 7501 \approx66.7\\% \\]

of all network CIM passes.

And they consume around 40 KiB of mapped CIM capacity despite only about 2.25 KiB of useful depthwise weights.

This is precisely the problem tackled by:

**Wang, Tsai, Liu, ASP-DAC 2026 — Input Reuse, Weight-Stationary Dataflow and Mapping Strategy for Depthwise Convolution in Computing-in-Memory Neural Network Accelerators.**

They rearrange the CIM mapping so the input is reused better and outputs are reconstructed from partial sums. They report up to **5.19× speedup** for depthwise mapping and up to **2.87× model-wise energy efficiency**, with 6.3% macro-area overhead. [DOI](https://doi.org/10.1109/ASP-DAC66049.2026.11420629?utm_source=chatgpt.com)

This means:

> You cannot claim “we discovered input reuse for depthwise CIM.”

That is taken.

But I would use their mapping as an **important baseline or optional ingredient**.

A more ambitious HopCIM paper could combine:

```
published intra-layer DW reuse mapping
                  +
new inter-layer DW→PW stream fusion
                  +
resident-weight tile scheduling
```

and ask whether jointly optimizing **within-layer reuse and between-layer communication** produces a better system-level design.

The first part is prior art; the second/third parts would have to carry your novelty.

---

# 12. Why I prefer this over sparse compression

You also mentioned “filter down the data size.”

A natural idea is:

```
lots of zeros
↓
compress activation
↓
fetch only nonzeros
```

There is extensive prior art here.

Thanos, DATE 2025, specifically targets KWS and exploits zero-skipping across time, feature and frequency domains; it reports very large latency and energy reductions. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10993014?utm_source=chatgpt.com)

ROZK, JSSC 2026, explicitly discusses problems caused by encoded sparse data—load imbalance, irregular memory footprint and routing—and instead uses local zero-skipping. [IEEE Xplore](https://ieeexplore.ieee.org/document/11153790/?utm_source=chatgpt.com)

Digital CIM processors also increasingly support dual-sided sparsity and compressed activation processing. [IEEE Xplore](https://ieeexplore.ieee.org/document/10562243/?utm_source=chatgpt.com)

For HopCIM, there is another problem:

Even if several rows in one CIM input are zero, your bit-serial macro still performs the pass unless the architecture can exploit that sparsity.

So activation compression might reduce SRAM traffic but **not necessarily reduce CIM cycles**.

I would profile sparsity, but I would not make it the main research direction.

---

# 13. Why I also would not focus on dynamic weight loading

Another possible idea is:

> Store compact weights outside CIM and load only each active layer into a smaller number of CIM tiles.

That could reduce CIM capacity.

But it creates a new problem:

\\[ \text{weight loading latency + energy}. \\]

PIPECIM explicitly introduces fine-grained ping-pong weight loading to hide these stalls, showing that this is an established CIM research problem. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10701033?utm_source=chatgpt.com)

MIREDO also treats weight reload/mode-switch stalls as an important dataflow constraint. [ci-lab.net](https://www.ci-lab.net/uploads/paper/aspdac26_miredo.pdf?utm_source=chatgpt.com)

And in your system the model is only \~21.5 KiB, so retaining weights is actually one of HopCIM's strengths.

I would not deliberately throw that advantage away.

---

# 14. Literature positioning of the proposed fusion direction

The prior art around this idea is substantial, so the novelty claim has to be precise.

| Work                           | Main idea                                                                                                                                   | Difference from proposed HopCIM direction           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| DepFiN, JSSC 2023              | Depth-first layer fusion                                                                                                                    | Conventional CNN accelerator; not CIM               |
| TileFlow, MICRO 2023           | Models fusion dataflows                                                                                                                     | Framework, not your hardware mechanism              |
| Fused DW/PW, 2024              | GPU fusion                                                                                                                                  | GPU architecture                                    |
| GAP8 DSC fusion, 2024          | Reduce memory transfers                                                                                                                     | MCU kernels                                         |
| FuseCU, DAC 2025               | Communication-optimal operator fusion                                                                                                       | General spatial accelerators                        |
| EDEA, 2025                     | Dedicated dual DW/PW engines                                                                                                                | Two dedicated engines                               |
| Ir-Ws, ASP-DAC 2026            | CIM depthwise input reuse                                                                                                                   | Intra-layer DW mapping                              |
| MIREDO, ASP-DAC 2026           | General CIM dataflow optimizer                                                                                                              | Broad compiler/DSE                                  |
| **Proposed HopCIM-StreamFuse** | **One resident-weight CIM dynamically switches between quantized DW/PW operators while intermediate features stay in line buffers/X banks** | **Single-CIM, exact-TFLite, cross-layer streaming** |

The fusion concept itself is absolutely not new.

The possible novelty is the **specific CIM execution mechanism and schedule**.

---

# 15. A stronger novelty formulation

I would not say:

> “We propose depthwise-pointwise fusion.”

Too weak.

I would formulate the research problem as:

> **Can a weight-resident digital CIM execute a depthwise-separable network in a communication-minimal cross-layer order, preserving exact quantization boundaries, so that intermediate feature maps do not need to be materialized in the accelerator SRAM?**

Then the architecture consists of:

```
resident CIM weight tiles
        +
fine-grained layer switching
        +
rolling line buffers
        +
result-to-X forwarding
        +
exact requantization boundary
        +
communication-aware scheduler
```

That is much more defensible.

---

# 16. The important twist: preserve the quantization boundary

This is where your work can differ from some fusion designs.

You don't do:

\\[ PW(DW(x)) \\]

as one algebraically merged convolution.

You explicitly preserve:

\\[ q\_{DW} = Q(DW(x)) \\]

then compute:

\\[ PW(q\_{DW}). \\]

Therefore:

```
DW CIM
 ↓
TFLite exact requantization
 ↓
forward qDW locally
 ↓
PW CIM
```

must equal:

```
DW
 ↓
requantize
 ↓
SRAM
 ↓
read
 ↓
PW
```

bit-for-bit.

So:

\\[ \boxed{ \text{memory semantics change; model semantics do not} } \\]

That fits HopCIM's existing “model-preserving” identity extremely well.

---

# 17. What I would build

I would **not initially implement the full depth-first four-block architecture**.

Use staged research gates:

1. **Profile the current system.** Add counters for ACT reads/writes, X-bank writes, PARAM reads, CIM passes, gather stalls, requant stalls and layer barriers. Get the exact data-movement breakdown per layer.
2. **Software/cycle simulator.** Model three schedules: current layer-by-layer, 3-row line-buffer DW, and line-buffer + direct DW→PW forwarding. Ensure all arithmetic is unchanged.
3. **Prototype one DSC block only.** Implement `DW1 → requant → PW1` with a 960-B rolling line buffer plus a 64-B fusion vector/result-to-X path. Verify every output against `dscnn_int.py`.
4. **Only if worthwhile, generalize to four blocks.** Then add the depth-first scheduler where PW outputs directly populate the next block's line buffer.
5. **Finally evaluate the published Ir-Ws depthwise mapping as a baseline/optional combination.** Do not claim it as your contribution.

That would be my one list; the rest of the evaluation can be expressed in tables.

---

# 18. Go/no-go thresholds

Before committing months of RTL effort, I would require approximately:

| Metric                  | Gate                                                |
| ----------------------- | --------------------------------------------------- |
| TFLite/logit mismatches | **0**                                               |
| Large ACT SRAM accesses | **≥60% reduction**                                  |
| ACT storage             | preferably **≥50% reduction**                       |
| Network latency         | no regression; ideally ≥10% improvement             |
| Added LUT/DSP           | small relative to current sequencer                 |
| Modeled total energy    | **≥15% reduction** after line-buffer leakage        |
| Second workload/model   | schedule should also help another DS-CNN-like model |

The 60% memory-access target looks plausible from the geometry.

The 15% energy target is intentionally much harder.

---

# 19. Strong baseline you absolutely need

Do not compare only against current 85k-cycle R6b.

You need:

\\[ B_0 = \text{current R6b} \\]

and:

\\[ B_1 = \text{R6b + second requantization lane}. \\]

Because \\(B_1\\) should be around:

\\[ 67.9\text{k cycles} \\]

for only two extra DSPs.    SEQUENCER

If StreamFuse gives:

```
85k → 70k
```

but a second requantizer gives:

```
85k → 68k
```

then StreamFuse cannot claim its main value is performance.

Its value must be demonstrated in:

```
activation accesses
buffer size
dynamic energy
```

against that strengthened baseline.

This is exactly the kind of baseline lesson you learned from ExactReuse.

---

# 20. How I would benchmark memory traffic

For every design, report:

| Metric                   | Current | Line buffer | DW/PW fused | Full StreamFuse                 |
| ------------------------ | ------- | ----------- | ----------- | ------------------------------- |
| CIM passes               | 7,501   | same        | same        | same unless combined with Ir-Ws |
| ACT read transactions    | measure | measure     | measure     | measure                         |
| ACT writes               | measure | measure     | measure     | measure                         |
| Line-buffer reads/writes | 0       | measure     | measure     | measure                         |
| X-bank writes            | measure | measure     | measure     | measure                         |
| Requant results          | same    | same        | same        | same                            |
| Activation SRAM size     | 16 KB   | ?           | ?           | target \~4–6 KB                 |
| Network cycles           | 85,352  | ?           | ?           | ?                               |
| Energy / inference       | model   | model       | model       | model                           |

This makes the paper much more convincing than only reporting “memory reduced by X%.”

---

# 21. Energy modeling

For this idea, energy is central.

Use:

\\[ E\_\text{total} = N\_\text{CIM}E\_\text{CIM} + N\_\text{ACT-r}E\_\text{ACT-r} + N\_\text{ACT-w}E\_\text{ACT-w} + N\_\text{LB}E\_\text{LB} + E\_\text{requant} + E\_\text{control} + P\_\text{leak}T. \\]

Notice that:

\\[ N\_\text{CIM} \\]

does not initially decrease.

So the proposal only wins if:

\\[ \Delta E\_\text{memory} > E\_\text{linebuffer+control}. \\]

This is the exact thing you should prove.

---

# 22. Public tools I would use

**Timeloop/Accelergy** is useful for establishing the memory hierarchy and counting/mapping data accesses. It explicitly models working sets, memory access, wires, compute energy and mapping trade-offs. It can also model keeping tensors on-chip across layers. [timeloop.csail.mit.edu](https://timeloop.csail.mit.edu/?utm_source=chatgpt.com)

**TileFlow** is particularly relevant because it was designed to model **operator-fusion dataflows** and calculates data-movement volume and resource utilization for fused schedules. It reports 1.28× average speedup for convolution chains in its own evaluation. [IEEE Xplore](https://ieeexplore.ieee.org/document/10411361/?utm_source=chatgpt.com)

**CiMLoop** should remain your CIM-specific energy tool. It is open source and explicitly models CIM circuits, architecture, operand values, mapping and system energy. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10590023?utm_source=chatgpt.com)

For the actual design, keep using your existing:

```
Python integer reference
→ Verilator
→ gate-level Vivado
→ Nexys A7
```

plus SAIF-based FPGA dynamic-power comparison.

The analytical tools should complement your hardware measurement, not replace it.

---

# 23. How this would fit the existing HopCIM paper/thesis story

Your current message is:

> Moving only MACs to CIM is insufficient; the surrounding data movement and epilogue have to move into hardware.

The proposed direction is a very natural next observation:

> **Even after the CPU is removed, a layer-by-layer accelerator still spends storage and energy moving intermediate activations. The next optimization level is therefore cross-layer dataflow.**

That produces a nice progression:

```
CPU bottleneck
   ↓
move MACs
   ↓
data movement + epilogue dominate

move sequencer + epilogue
   ↓
CPU disappears

now:
intermediate activation movement remains
   ↓
cross-layer StreamFuse
```

That is a coherent research evolution, not a disconnected extra project.

---

# 24. My ranking after this literature review

| Idea                                                  | Expected benefit           | Novelty potential                | HopCIM fit    | Verdict             |
| ----------------------------------------------------- | -------------------------- | -------------------------------- | ------------- | ------------------- |
| Activation compression / zero skip                    | Medium                     | Low                              | Medium        | Don't lead with it  |
| ASP-DAC-style depthwise remapping                     | **High**                   | **Very low: directly published** | High          | Baseline/ingredient |
| Dynamic weight streaming                              | Medium                     | Low–moderate                     | Poor–medium   | Not preferred       |
| Exact bit-width reduction                             | Medium                     | Low–moderate                     | High          | Secondary           |
| More temporal activation reuse                        | Medium                     | Low                              | Medium        | Already rejected    |
| **Cross-layer line-buffer + DW/PW direct forwarding** | **High traffic reduction** | **Moderate**                     | **Very high** | **Best direction**  |
| Full communication-minimal compiler                   | Potentially high           | Moderate–high if generalized     | Medium        | Ambitious extension |

So my recommendation is:

\\[ \boxed{\textbf{HopCIM-StreamFuse}} \\]

but with an important qualification:

> **Do not start with RTL. First prove that ACT-buffer accesses are significant enough in an energy model to matter.**

The very first experiment should be an **activation-traffic census of the existing `cim_seq`**, broken down by Conv1, each DW, each PW, and pool. If it confirms the analytical \~7.6× repeated DW fetch pattern and shows that ACT-buffer energy is a meaningful fraction of total system energy, then this is a much more defensible next research direction than ExactReuse-CIM