# TARGET 08.33: DSV4 SM80 Indexer Capture Static-Width Audit

## Status

Active TARGET 08 capacity follow-up after TARGET 08.32.

Run this before any broader CUDA graph workspace redesign.  TARGET 08.32
proved that many synthetic graph-private-pool suspects are too small, but it
did not exercise every real full-model call-site shape.  This target focuses on
one concrete high-impact hypothesis in the DSV4 C4 indexer logits path.

## Goal

Prove or disprove whether the `~18.8-19.0 GiB/rank` first CUDA graph capture
memory cost is materially caused by an over-large captured indexer logits
workspace, especially this capture-time width calculation:

```text
static_max_seq_len = page_table.shape[1] * page_size
```

If the hypothesis is true, implement the smallest correct fix and confirm that
graph capture memory drops without breaking text smoke, prefix-cache behavior,
or graph replay.  If it is false, leave precise instrumentation and a narrowed
next hypothesis.

## Why This Target Exists

TARGET 08.32 found no synthetic owner above the multi-GiB class:

- empty/PyTorch graph overhead was small;
- simple out-of-place temporaries were small;
- BF16 matmul and cuBLASLt workspace were small;
- synthetic SWA/C4/C128 attention was small;
- repeated C4 indexer/topk skeleton `N=21` reached only about
  `0.111 GiB/rank`;
- synthetic metadata/deforest helpers and NCCL controls were small.

However, the real mini indexer logits path allocates dense FP32 logits shaped
like:

```text
[rows, max_seq_len]
```

during capture.  If `page_table.shape[1]` is already a token-slot width in mini
and is multiplied by `page_size=256` again, the captured logits can be inflated
by `256x`.

Example scale:

```text
32768 token slots * 256 = 8388608 static positions
bs=16 FP32 logits = 16 * 8388608 * 4 ~= 512 MiB
21 C4/indexer layers ~= 10.5 GiB projected captured logits alone
```

This is large enough to plausibly explain a major fraction of the first-graph
private-pool cost when combined with related buffers and graph allocator
lifetime.

## Non-Goals

- Do not run another broad synthetic graph-memory sweep.
- Do not redesign the graph/workspace manager unless this target proves the
  indexer hypothesis is false and leaves a specific new owner.
- Do not load the full DSV4 checkpoint until the instrumentation and a minimal
  width A/B are ready.
- Do not implement FP8 KV/cache, INT8 MoE, quantized communication, or SWA
  lifecycle changes here.
- Do not optimize sub-`1 GiB/rank` graph-memory effects in this target.

## Source References

Primary mini code:

- `python/minisgl/kernel/deepseek_v4.py`
  - `indexer_bf16_logits_fallback`
  - `indexer_fp8_logits_fallback`
  - `indexer_fp8_paged_logits_fallback`
- `python/minisgl/kernel/triton/deepseek_v4.py`
  - `indexer_bf16_logits`
  - `indexer_fp8_logits`
  - `indexer_fp8_paged_logits`
- `python/minisgl/attention/deepseek_v4.py`
  - C4/indexer metadata creation and component page table handling
- `python/minisgl/engine/engine.py`
  - global page-table allocation
- `python/minisgl/engine/graph.py`
  - CUDA graph capture flow and memory measurement boundary

Historical evidence:

- `performance_milestones/target08_cuda_graph_memory_attribution/README.md`
- `performance_milestones/target08_bf16_cache_graph_memory_attribution/README.md`
- `performance_milestones/target08_cuda_graph_private_pool_micro_attribution/README.md`
- `prompts/TARGET_08.32_dsv4_sm80_cuda_graph_private_pool_micro_attribution.md`

Useful reference implementations:

- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## Required Approach

### 1. Establish Page-Table Semantics

Before changing behavior, write down the exact semantics of every page table
that can reach the indexer logits functions:

- global engine page table;
- DSV4 decode metadata `page_table`;
- `c4_page_table`;
- `c128_page_table`;
- `c4_indexer_page_table`;
- component page-table cache rows, if enabled;
- capture dummy request table.

For each table record:

- shape;
- whether width means token slots or logical pages;
- whether entries store token locations or page ids;
- whether the table has already been divided by `page_size`;
- the intended `page_size` to use when gathering cache.

Compare mini behavior with SGLang/vLLM source where possible.  Do not assume
the table is page-based just because the local variable is named
`page_table`.

### 2. Add Focused Capture Instrumentation

Add opt-in diagnostics, guarded by an environment flag such as:

```text
MINISGL_DSV4_INDEXER_CAPTURE_WIDTH_DEBUG=1
```

The diagnostics should be low-volume and rank-aware.  During CUDA graph capture
only, log or collect per call:

- function name;
- backend path (`triton`, `triton_fp8`, `triton_fp8_paged`, fallback);
- layer id if available, otherwise call index;
- `rows`;
- `q.shape`;
- `cache.shape`;
- `seq_lens.min/max`;
- `page_table.shape`;
- `page_size`;
- current `static_max_seq_len`;
- candidate corrected widths:
  - `page_table.shape[1]`;
  - `seq_lens.max()`;
  - `page_table.shape[1] * page_size`;
  - `ceil(seq_lens.max() / page_size) * page_size`;
- implied FP32 logits bytes for each candidate.

The final README must include a table of the real captured widths.  If the log
is large, write JSONL under the milestone directory and summarize it.

### 3. Split Graph Capture Memory Stages

Instrument `GraphRunner` with an opt-in stage memory ledger, for example:

```text
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1
```

Measure at least:

- before `GraphCaptureBuffer.init`;
- after `GraphCaptureBuffer.init`;
- after `attn_backend.init_capture_graph`;
- after `prepare_for_capture`;
- after `stage_capture_metadata`;
- after warmup `model.forward()` outside `torch.cuda.graph`;
- after actual `torch.cuda.graph` capture;
- after optional `gc.collect()` / `torch.cuda.empty_cache()` sanity point.

Report both free-memory delta and PyTorch allocated/reserved deltas.  This tells
whether the cost appears during warmup, actual capture, or persistent setup.

### 4. Minimal Width A/B Before Full Macro

Create a small, reversible opt-in switch for the candidate corrected indexer
capture width.  Example:

```text
MINISGL_DSV4_INDEXER_CAPTURE_WIDTH_MODE=current
MINISGL_DSV4_INDEXER_CAPTURE_WIDTH_MODE=table_width
MINISGL_DSV4_INDEXER_CAPTURE_WIDTH_MODE=seq_len_aligned
```

Start with diagnostic mode only.  Only enable behavior change after page-table
semantics are clear.

The first A/B should use a single graph bucket and the smallest full-model run
that exercises the real call sites, for example:

```text
--allow-dsv4-cuda-graph --cuda-graph-bs 1
```

or:

```text
--allow-dsv4-cuda-graph --cuda-graph-bs 16
```

Do not run the full matrix until a single-bucket result shows a meaningful
effect.

### 5. Correctness And Replay Gates

For any behavior change, run:

- DSV4 text smoke with page size `256`;
- graph replay zero-eager check for the tested bucket;
- prefix-cache smoke if the change can touch component page tables;
- a small logits/top-k sanity check if available, comparing current mode versus
  corrected-width mode under the same prompt/layout.

The corrected width must not truncate valid positions.  If it does, stop and
document the exact table semantics instead of forcing the optimization.

## Candidate Fix Directions

Only choose a fix after the width audit proves the current width is too large.

Possible fixes:

- if the incoming indexer table width is already token slots, set capture
  `static_max_seq_len = page_table.shape[1]`;
- if the incoming indexer table width is logical pages, keep
  `page_table.shape[1] * page_size`;
- pass an explicit `max_seq_len` from metadata instead of inferring from table
  shape;
- normalize the table contract so indexer logits receives either a compact
  page table plus `page_size`, or a token-slot loc table plus `page_size=1`
  semantics;
- if dense logits are still the owner after width correction, open a later
  target to avoid dense `[rows, max_seq_len]` logits and compute top-k/select
  directly.

Prefer the smallest contract fix that aligns with SGLang/vLLM behavior.

## Required Analysis

The final README must include:

- recap of TARGET 08.32 and why this target is narrower;
- page-table semantics table;
- real capture indexer-width table;
- graph capture stage memory table;
- current-width versus candidate-width memory A/B;
- projected memory saved per rank and equivalent extra KV pages/tokens;
- correctness/replay results;
- whether the `~19 GiB/rank` first-graph cost is now explained, partially
  explained, or still unattributed;
- the next target recommendation.

Use this projection formula:

```text
projected_dense_logits_bytes =
  rows * static_max_seq_len * sizeof(float32) * active_indexer_owner_count
```

Label projections as projections until a full-model graph capture confirms
them.

## Gates

Pass this target if it produces one of:

1. proof that the indexer capture width is over-expanded, plus a corrected
   width PoC that saves at least `2 GiB/rank` graph capture memory;
2. proof that the width is semantically correct, with instrumentation showing
   indexer logits is not a multi-GiB owner;
3. proof that indexer dense logits explains a material cost but requires a
   later direct-topk/select redesign.

Stop early if:

- page-table semantics cannot be determined from code and instrumentation;
- the corrected width risks truncating valid attention/indexer positions;
- single-bucket A/B shows less than `1 GiB/rank` effect and instrumentation
  shows no large repeated indexer logits allocation;
- full-model runs are attempted before the instrumentation is in place.

## Deliverables

Write results under:

```text
performance_milestones/target08_indexer_capture_static_width_audit/
```

Include:

- `README.md`;
- scripts or command snippets used for instrumentation and A/B;
- JSON/JSONL logs for indexer width and graph capture stages;
- raw logs or symlinks;
- any small code changes required for debug flags or candidate width mode;
- final recommendation:
  - promote a width fix;
  - open direct-topk/select redesign;
  - or return to broader graph-private owner attribution.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read `prompts/target.md`,
`prompts/TARGET_08_radix_prefix_dsv4.md`, this file, and the TARGET 08.32
report.  Start by mapping the exact page-table semantics that reach
`indexer_*_logits_fallback`, then add opt-in capture-width and graph-stage
instrumentation.  Do not run a broad benchmark matrix until a single-bucket
current-width versus candidate-width A/B explains at least `1 GiB/rank`.
