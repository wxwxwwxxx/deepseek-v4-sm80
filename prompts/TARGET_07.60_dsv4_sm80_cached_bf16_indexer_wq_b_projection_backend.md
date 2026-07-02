# TARGET 07.60: DSV4 SM80 Cached BF16 `indexer.wq_b` Projection Backend

## Goal

Extend the cached BF16 dequantized-weight projection backend to the remaining
large `_quantized_linear_fp8_kernel` owner: `indexer.wq_b`.

TARGET 07.58 and TARGET 07.59 promoted owner-scoped cached BF16 paths for
`attn.q_wqb` and `attn.wo_b`.  The current promoted opt-in stack is:

```text
q_wqb cached BF16 + wo_b cached BF16
```

Macro baseline for this target:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `49.6585` | `122.1863` | `127` | `0` |
| 4096/1024/batch4 | `98.6953` | `121.8705` | `1023` | `0` |

TARGET 07.59's fresh profile shows the next largest same-contract owner is now
`indexer.wq_b`:

```text
indexer.wq_b intrinsic _quantized_linear_fp8_kernel: 0.364997s
indexer.wq_b activation quant:                     0.005293s
indexer.wq_b copy/layout:                           0.012928s
```

This target should finish the three-owner cached BF16 projection sequence:

1. `attn.q_wqb`;
2. `attn.wo_b`;
3. `indexer.wq_b`.

After this target, do not continue projection work by inertia.  Run a fresh
profile and choose the next bottleneck from evidence.

## Win Condition

Primary implementation gate:

- implement a graph-safe opt-in cached BF16 dequantized-weight path for
  `indexer.wq_b` on top of the promoted `q_wqb + wo_b` cached BF16 path; and
- reduce focused `indexer.wq_b` local projection compute time by at least
  `30%`, or improve 4096/128/batch4 output throughput by at least `3%` over the
  q_wqb+wo_b baseline `49.6585 output tok/s`; and
- preserve CUDA graph replay with eager decode count `0`; and
- pass text smoke.

Long-decode gate:

- if 4096/128 passes the focused/profile gate, run 4096/1024/batch4 and require
  at least `2%` output throughput gain over `98.6953 output tok/s`, or explain
  from profile evidence why the gain does not carry.

Post-sequence reprofile gate:

- after implementing `indexer.wq_b`, capture a fresh 4096/128/batch4 rank0
  profile and classify the top remaining owners/buckets.
- the final decision must name the next bottleneck.  Do not end with "continue
  projection generally."

Memory gate:

- record incremental `indexer.wq_b` cache bytes/rank;
- record combined `q_wqb + wo_b + indexer.wq_b` cache bytes/rank;
- convert both to KV tokens/pages using measured benchmark metadata;
- if incremental `indexer.wq_b` cache costs much more than the expected
  `0.33 GiB/rank`, stop and explain before promotion.

## Current Baseline

Current opt-in baseline from TARGET 07.59:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `49.6585` | `122.1863` | `127` | `0` |
| 4096/1024/batch4 | `98.6953` | `121.8705` | `1023` | `0` |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- vLLM 4096/128/batch4: about `82.28 output tok/s`;
- vLLM 4096/1024/batch4: about `202.03 output tok/s`.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.58_dsv4_sm80_cached_bf16_projection_backend.md`
- `prompts/TARGET_07.59_dsv4_sm80_cached_bf16_wo_b_projection_backend.md`
- `performance_milestones/target07_cached_bf16_projection_backend/README.md`
- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/README.md`
- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/summaries/nsys_target0759_qwqb_wob_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`
- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/summaries/qwqb_wob_memory_ledger.md`

Mini source areas:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/engine/engine.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM source areas:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

## vLLM Comparison Requirement

Keep the comparison focused:

- compare mini's `indexer.wq_b` projection boundary against the corresponding
  vLLM indexer/query projection source boundary;
- record whether vLLM applies `Fp8LinearMethod`, BF16 dequant fallback, Marlin
  FP8, or another backend at the source level;
- do not claim runtime parity unless an actual vLLM probe/profile is run.

This target is still a mini-owned speed-first cached-weight experiment.  The
major vLLM parity pass should happen after q_wqb, wo_b, and indexer.wq_b have
all been handled and a fresh mini profile is available.

## Scope

In scope:

- owner-scoped cached BF16 dequantized-weight backend for `indexer.wq_b`;
- preserving the existing q_wqb and wo_b cached BF16 paths;
- using only C4/indexer layers that actually own `DSV4Indexer`;
- focused microbench for real serving-sharded `indexer.wq_b`;
- text smoke and macro benchmark with page size 256;
- memory ledger for incremental indexer cache and combined three-owner cache;
- fresh post-implementation rank0 profile and top-bucket decision.

Out of scope:

- changing FP8 indexer cache/logits kernels;
- changing indexer compressor, indexer weights projection, topk, or cache store;
- changing `attn.wo_a`, `wq_a/wkv`, shared experts, MoE/Marlin, or `lm_head`;
- removing activation fake-quant;
- communication/NCCL optimization;
- full model/layer `torch.compile`;
- full `fp8_ds_mla` KV-cache E2E;
- broad graph/layout cleanup before the post-sequence reprofile.

## Implementation Guidance

Expected new opt-in toggle:

```text
MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE=1
```

Expected new benchmark/text-smoke variant should include all three cached BF16
owners, for example:

```text
..._qwqbbf16cache_wobbf16cache_idxwqbbf16cache_...
```

Implementation notes:

- reuse the cached FP8-to-BF16 helper from TARGET 07.58/07.59 if possible;
- prepare indexer `wq_b` BF16 caches during the explicit model preparation
  phase, after weights are loaded and before CUDA graph capture;
- only prepare layers with an actual `indexer` object, i.e. C4 layers in this
  model;
- graph capture and replay must only read the persistent BF16 cache;
- if the indexer `wq_b` cache is missing or stale in forward, raise a clear
  error rather than rebuilding;
- route all active `indexer.wq_b` call sites through the cached path only when
  the new toggle is enabled;
- do not route `indexer.weights_proj`, `indexer.compressor`, FP8 cache logits,
  or other indexer subpaths through this toggle.

The active `indexer.wq_b` path is used to prepare query vectors for sparse
indexer selection.  Keep the surrounding rotary/indexer math unchanged.

## Memory Ledger

The milestone README must include:

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | `360,710,144` | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | 43 | `[4096, 1024]` | `360,710,144` | `0.3359` | `4744.04` | `18.53` |
| `indexer.wq_b` | 21 | `[8192, 1024]` | `352,321,536` | `0.3281` | `4633.71` | `18.10` |
| total | 107 | mixed | `1,073,741,824` | `1.0000` | `14121.79` | `55.16` |

Reference estimates use the TARGET 07.59 `76034.41 bytes/token/rank` value.
The target must recompute final values from actual benchmark artifacts and
report expected-vs-measured peak memory deltas.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/
```

with:

- `README.md`;
- `scripts/`;
- `raw/`;
- `summaries/`.

Record inherited q_wqb+wo_b baseline, 07.59 profile evidence, and the expected
indexer memory ledger.

### 2. Build Or Extend Real-Weight Microbench

Measure real serving `indexer.wq_b` weights on a C4 layer such as layer 2.

At minimum, report for `M = 1, 4, 8, 16`:

- current FP8 wrapper projection ms;
- current intrinsic `_quantized_linear_fp8_kernel` ms;
- cached BF16 `F.linear` ms;
- cached total projection ms;
- max absolute and relative error against current path.

Confirm shape and layer count:

- local shape should be `[8192, 1024]`;
- cached layers should be `21` for `/models/DeepSeek-V4-Flash`.

### 3. Implement `indexer.wq_b` Cached BF16 Path

Implement the new path only for `indexer.wq_b`, with q_wqb and wo_b caches still
enabled.

Do not enable:

- `indexer.weights_proj`;
- `indexer.compressor`;
- FP8 indexer cache logits;
- `attn.wo_a`;
- `wq_a/wkv`;
- shared experts;
- `lm_head`.

### 4. Validate Correctness And Graph Semantics

Required:

- py_compile for touched Python files;
- focused microbench correctness;
- text smoke with page size 256;
- CUDA graph replay preserved;
- eager decode count `0`;
- no default behavior change when new toggles are off;
- evidence that q_wqb, wo_b, and indexer.wq_b cached BF16 allocations happen
  before graph capture and are not rebuilt inside replay.

Suggested smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_three_owner_cached_variant> \
  --page-size 256 \
  --output performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/text_smoke_qwqb_wob_idxwqb.json
```

### 5. Macro Gate

Run 4096/128/batch4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_three_owner_cached_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/macro_qwqb_wob_idxwqb_4096x128_bs4_np128 \
  --keep-going
```

If the short run passes the focused/profile gate or the macro gate, run
4096/1024/batch4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_three_owner_cached_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/macro_qwqb_wob_idxwqb_4096x1024_bs4_np128 \
  --keep-going
```

### 6. Fresh Post-Sequence Profile

Capture a fresh 4096/128/batch4 rank0 Nsight profile after the three-owner
cached BF16 path is working.

Classify at least:

- `attn.q_wqb`;
- `attn.wo_b` local projection and all-reduce separately;
- `indexer.wq_b`;
- `attn.wo_a`;
- shared expert projection/copy-layout;
- graph/layout cluster;
- MoE/Marlin;
- sparse attention/indexer/cache buckets;
- remaining `_quantized_linear_fp8_kernel` total.

This profile is mandatory because this target closes the original
`_quantized_linear_fp8_kernel` three-owner sequence.

## Decision Rules

End with exactly one decision:

- `Decision: promote three-owner cached BF16 and run vLLM parity reprofile`
  if correctness, graph, memory, focused, and macro/profile gates pass.
- `Decision: pivot to wo_b communication/overlap`
  if `wo_b` all-reduce is now a top blocker after indexer.wq_b is reduced.
- `Decision: pivot to activation-quant/projection contract`
  if q_wqb/wo_b/indexer cached BF16 matmuls are fast but activation fake-quant
  or wrapper overhead dominates the projection owners.
- `Decision: pivot to wo_a/shared-expert graph-layout`
  if post-sequence profile shows those owners dominate.
- `Decision: stop cached BF16 expansion`
  if indexer.wq_b misses correctness, graph replay, memory, or focused profile
  gates.

Do not choose a broad next target without profile evidence.

## Stop Rules

Hard stops:

- indexer.wq_b cached path misses both focused projection and macro/profile
  gates;
- graph replay is lost or eager decode becomes nonzero;
- cached BF16 weights are allocated or rebuilt during graph replay;
- measured memory delta is much larger than expected and cannot be explained;
- text smoke fails and one focused fix does not restore it;
- implementation starts touching unrelated indexer or attention owners to show
  a win.

## Expected Output

Create:

- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/README.md`
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/scripts/`
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/`
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/summaries/`

The README must include:

- inherited q_wqb+wo_b baseline;
- vLLM `indexer.wq_b` source comparison;
- implementation summary and exact toggles/variant name;
- microbench before/after;
- text smoke and graph replay status;
- 4096/128 macro result;
- 4096/1024 macro result if gate is reached;
- memory ledger with incremental and combined cache cost;
- fresh post-sequence profile summary;
- final decision and exact next target recommendation.
