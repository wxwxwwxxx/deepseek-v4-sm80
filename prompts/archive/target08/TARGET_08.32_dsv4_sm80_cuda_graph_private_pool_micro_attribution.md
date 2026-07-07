# TARGET 08.32: DSV4 SM80 CUDA Graph Private-Pool Micro Attribution

## Status

Parallel TARGET 08 follow-up.

Run this when capacity/headroom is the priority.  It is independent from TARGET
08.31:

- TARGET 08.31 investigates persistent SWA/cache lifecycle memory.
- TARGET 08.32 investigates CUDA graph private-pool memory.

Both affect serving capacity, but they have different mechanisms and should not
be mixed in one child thread.

## Goal

Attribute and reduce, if a clear low-risk fix exists, the unusually large DSV4
CUDA graph capture memory cost on A100/sm80.

The known full-model number is about:

```text
18.8-19.0 GiB/rank first-graph CUDA graph private-pool cost
```

Previous TARGET 08.06/08.07 work proved this is real and repeatable, but did
not identify the internal owner.  This target should avoid immediately loading
the full model and should instead build partial-model and microbench probes that
isolate graph-private allocations by subgraph.

## Non-Goals

- Do not start by loading `/models/DeepSeek-V4-Flash` full weights.
- Do not run a broad full-model benchmark matrix until the micro attribution has
  a concrete hypothesis.
- Do not change prefix-cache semantics.
- Do not implement FP8 KV/cache, INT8 MoE, or quantized communication here.
- Do not redesign the whole cache/workspace manager unless the evidence proves
  it is necessary and there is no smaller fix.
- Do not spend time optimizing sub-`1 GiB/rank` graph-memory effects.

## Baseline Evidence

Read these first:

- `performance_milestones/target08_cuda_graph_memory_attribution/README.md`
- `performance_milestones/target08_bf16_cache_graph_memory_attribution/README.md`
- `performance_milestones/target08_prefix_cache_memory_ledger/README.md`
- `prompts/archive/target08/TARGET_08.06_dsv4_sm80_cuda_graph_memory_attribution.md`
- `prompts/archive/target08/TARGET_08.07_dsv4_sm80_bf16_cache_graph_memory_attribution.md`

Known results:

- `[1]` graph alone costs about `18.795 GiB/rank`.
- `[16]` graph alone costs about `18.828 GiB/rank`.
- `[1,2,4,8,16]` costs about `19.037 GiB/rank`.
- Later buckets reuse the first graph pool and add only about `0.05-0.08 GiB`
  per bucket.
- Explicit graph input buffers are tiny, at most about `7.9 MiB`.
- Greedy sample capture changed memory by `0.000 GiB`.
- Captured compressed-location metadata changed memory by `0.000 GiB`.
- `max_seq_len` changed memory by `0.000 GiB` in the tested range.
- `num_pages` from 64 to 128 changed graph delta by only about `0.035 GiB`.
- Disabling all tested BF16 projection/shared-expert caches removed about
  `1.588 GiB/rank` persistent pre-capture memory but changed the graph delta by
  only about `+0.057 GiB/rank`.

Interpretation to carry forward:

```text
The big owner is probably the private pool preserving captured runtime
allocation/workspace shape for full decode forward, not KV pages, graph input
buffers, greedy sampling, direct metadata, or BF16 cache tensors themselves.
```

## Source References

Mini graph/runtime:

- `python/minisgl/engine/graph.py`
- `python/minisgl/engine/engine.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

Mini DSV4 subgraphs:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/moe_impl.py`
- `python/minisgl/kernel/marlin_wna16.py`

SGLang/vLLM reference, if source comparison helps:

- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/`
- `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## Required Approach

### 1. Build A Lightweight Graph-Memory Harness

Create a milestone-local script, for example:

```text
performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/graph_private_pool_micro.py
```

The harness should:

- run each case in a fresh Python process when comparing graph private-pool
  memory;
- avoid full model weight loading by default;
- allocate synthetic tensors matching DSV4 decode shapes;
- warm up once before capture when appropriate;
- capture one graph with `torch.cuda.CUDAGraph`;
- optionally reuse a graph pool for later buckets to reproduce old behavior;
- report free memory, allocated/reserved memory, peak memory, and graph delta;
- emit JSON plus a Markdown summary.

At minimum report:

- `free_before_bytes`;
- `free_after_bytes`;
- `free_delta_bytes`;
- `allocated_before_bytes`;
- `allocated_after_bytes`;
- `allocated_delta_bytes`;
- `reserved_before_bytes`;
- `reserved_after_bytes`;
- `reserved_delta_bytes`;
- `peak_allocated_bytes`;
- `peak_reserved_bytes`;
- capture elapsed time;
- graph replay sanity;
- input/output/workspace tensor bytes that were allocated outside capture.

Use `torch.cuda.memory_snapshot()` or allocator history only if it stays
lightweight and produces useful attribution.  The primary evidence should be
controlled A/B capture deltas, not giant snapshots.

### 2. Establish Controls

Run controls before DSV4-specific probes:

- empty graph;
- tensor copy / static graph input staging only;
- one small elementwise chain;
- one BF16 matmul using DSV4-like shapes;
- repeated BF16 matmul loop with `N=1,2,4,8,16,43`;
- out-of-place versus preallocated-output variants where PyTorch/kernel APIs
  support it;
- optional NCCL/PyNCCL all-reduce graph-capture control with synthetic BF16 and
  FP32 tensors.

These controls answer whether the 19 GiB class cost can be reproduced by:

- captured tensor temporaries;
- cuBLAS/cuBLASLt workspace;
- repeated layer out-of-place allocation;
- graph-private allocator behavior independent of DSV4 attention.

### 3. Probe DSV4 Attention Subgraphs Without Full Weights

Use synthetic tensors and, when possible, existing mini kernels directly.
Prioritize decode shapes for bucket `1` and `16`.

Probe these owners separately:

- SWA attention boundary;
- C4 sparse attention / C4A path;
- C128 compressed attention path;
- C4 indexer select/top-k/index metadata path;
- q/kv norm + RoPE + cache-store boundary;
- cache gather/dequant/store helpers if already available;
- direct graph metadata copy/deforest helpers.

For each case, compare:

- eager warmup peak allocation;
- capture delta;
- replay correctness/sanity;
- explicit preallocated workspace bytes;
- projected full-model cost if repeated across the relevant number of layers.

The most important question:

```text
Does any single attention subgraph, or repeated 43-layer synthetic loop, explain
multi-GiB graph private-pool growth?
```

### 4. Probe One-Layer And Repeated-Layer Decode Skeletons

If direct kernel probes are not enough, build a partial decode-layer skeleton
that still avoids full model weights:

- instantiate one DSV4 decode layer or a small owner-specific module with
  random/synthetic weights only if the constructor path is cheap;
- otherwise create a callable that mimics the same kernel sequence using
  synthetic tensors and existing kernels;
- run `N=1,2,4,8,16,43` repetitions inside capture;
- separate attention-only, MoE-only, projection-only, and attention+MLP
  skeletons.

Do not load model checkpoints for this phase.  The point is to find whether
private-pool memory scales with repeated captured temporary allocation, not to
measure model quality.

### 5. Compare With Full-Model Capture Only After A Hypothesis

Only after the micro/partial harness identifies a likely owner, run a small
full-model confirmation.

Use the promoted baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

If the candidate fix does not touch prefix cache, a faster non-prefix
confirmation is acceptable first.  Still run a prefix smoke before any
promotion-style recommendation.

## Candidate Fix Directions

Only attempt a PoC if attribution finds a material owner.

Possible fixes:

- preallocate graph-consumed output/workspace tensors outside capture;
- replace out-of-place repeated temporaries with explicit workspace or `out=`
  variants where safe;
- avoid hidden allocator calls inside captured helper boundaries;
- split a huge captured subgraph only if microbench proves memory drops enough
  and launch overhead remains acceptable;
- move certain setup/staging work outside capture if it is static and not needed
  for replay;
- align with SGLang/vLLM graph preparation if source review shows they avoid
  the same private-pool cost with a mature mechanism.

A fix is worth a full-model PoC only if it plausibly saves at least:

```text
>= 2 GiB/rank
```

or if it unlocks significantly more `--num-pages`/context capacity with no
latency regression.

## Required Analysis

The final README must include:

- old TARGET 08.06/08.07 evidence recap;
- micro harness design and exact commands;
- control-case memory table;
- DSV4 subgraph memory table;
- one-layer/repeated-layer scaling table;
- explanation of whether the cost is:
  - PyTorch graph allocator behavior;
  - captured out-of-place tensor temporaries;
  - cuBLAS/cuBLASLt workspace;
  - attention C4/C128/indexer workspace;
  - communication workspace;
  - full-model composition;
  - still unattributed;
- if a candidate fix exists, a before/after micro result;
- if no fix exists, the smallest next evidence step.

Use a simple projection formula:

```text
projected_full_model_delta =
  measured_one_owner_delta * number_of_layers_or_owner_count
```

and label it as a projection until full-model confirmation exists.

## Gates

Pass this target if it produces one of:

1. a concrete owner explaining at least `2 GiB/rank` of the graph private-pool
   cost plus a fix/PoC recommendation;
2. a concrete owner explaining most of the `18.8-19.0 GiB/rank` cost, even if
   the fix requires a later target;
3. strong evidence that the private-pool cost is a PyTorch/CUDA graph allocator
   property of the full captured decode graph and cannot be reduced locally
   without a major graph/workspace redesign.

Stop early if:

- the harness starts loading full model weights before micro attribution;
- the baseline controls cannot measure graph memory reliably;
- a probe repeatedly OOMs before producing comparable memory data;
- all tested owners are below `1 GiB/rank` projected effect and no scaling trend
  points to the 19 GiB class owner;
- the only possible next step is a broad workspace manager rewrite.

## Deliverables

Write results under:

```text
performance_milestones/target08_cuda_graph_private_pool_micro_attribution/
```

Include:

- `README.md`;
- `scripts/` for the micro harness and summary builder;
- `summaries/` JSON/Markdown tables;
- raw logs or symlinks;
- any small code changes needed for measurement hooks;
- recommendation for one of:
  - full-model confirmation/fix target;
  - graph/workspace manager design target;
  - defer graph memory because no tractable owner was found.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read this file,
`prompts/target.md`, `prompts/TARGET_08_radix_prefix_dsv4.md`, and the TARGET
08.06/08.07 reports.  Start by writing the synthetic graph-memory harness.  Do
not load full DSV4 weights until a micro or partial-layer result points to a
specific owner.
