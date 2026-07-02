# TARGET 07.59: DSV4 SM80 Cached BF16 `wo_b` Projection Backend

## Goal

Extend the TARGET 07.58 cached BF16 dequantized-weight projection backend from
`attn.q_wqb` to `attn.wo_b`, while keeping the result owner-scoped, graph-safe,
and memory-accounted.

TARGET 07.58 proved that the cached BF16 projection backend is a real win for
`attn.q_wqb`:

| Workload | 07.54/07.55 baseline output tok/s | q_wqb cached BF16 output tok/s | Gain |
| --- | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `47.9464` | `+11.33%` |
| 4096/1024/batch4 | `87.0831` | `92.5170` | `+6.24%` |

The `q_wqb` cache also corrected the earlier memory estimate: actual TP8 local
`q_wqb` shape is `[4096, 1024]`, so 43 layers require only
`360,710,144 bytes/rank` (`0.3359 GiB/rank`), about `4744` KV tokens or `18.53`
pages at page size 256.

TARGET 07.57 showed `attn.wo_b` has nearly the same intrinsic projection
weight as `attn.q_wqb`:

```text
attn.q_wqb _quantized_linear_fp8_kernel  0.404178s
attn.wo_b  _quantized_linear_fp8_kernel  0.403710s
```

The key difference is that `wo_b` is row-parallel and includes an all-reduce
after the local projection.  This target must therefore separate local
projection compute from `wo_b` communication before deciding whether cached
BF16 should be promoted further.

## Win Condition

Primary implementation gate:

- implement a graph-safe opt-in cached BF16 dequantized-weight path for
  `attn.wo_b` on top of the promoted `q_wqb` cached BF16 path; and
- reduce focused `attn.wo_b` local projection compute time by at least `30%`,
  or improve 4096/128/batch4 output throughput by at least `5%` over the
  q_wqb-cached baseline `47.9464 output tok/s`; and
- preserve CUDA graph replay with eager decode count `0`; and
- pass text smoke.

Secondary long-decode gate:

- if 4096/128 passes, run 4096/1024/batch4 and require at least `3%` output
  throughput gain over the q_wqb-cached baseline `92.5170 output tok/s`, or
  explain from profile evidence why the short-decode gain does not carry.

Communication-aware gate:

- report `wo_b` local projection compute separately from row-parallel
  all-reduce.
- if local projection improves but macro gain is under `5%`, this can still be
  a useful result only if a fresh profile proves all-reduce or another named
  owner is now masking the compute gain.

Memory gate:

- record incremental `wo_b` cache bytes/rank, total `q_wqb + wo_b` cache
  bytes/rank, peak memory delta, and equivalent KV tokens/pages lost.
- if the incremental `wo_b` cache costs much more than the expected
  `0.34 GiB/rank`, stop and explain before promotion.

## Current Baseline

The baseline for this target is the promoted TARGET 07.58 `q_wqb` cached BF16
variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `47.9464` | `112.3221` | `127` | `0` |
| 4096/1024/batch4 | `92.5170` | `112.3307` | `1023` | `0` |

Previous reference lines remain:

- old serving victory line: `114.07 output tok/s`;
- vLLM 4096/128/batch4: about `82.28 output tok/s`;
- vLLM 4096/1024/batch4: about `202.03 output tok/s`.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.58_dsv4_sm80_cached_bf16_projection_backend.md`
- `performance_milestones/target07_cached_bf16_projection_backend/README.md`
- `performance_milestones/target07_cached_bf16_projection_backend/summaries/qwqb_cached_bf16_microbench.md`
- `performance_milestones/target07_cached_bf16_projection_backend/summaries/qwqb_memory_ledger.md`
- `performance_milestones/target07_projection_gemm_backend_parity/README.md`
- `performance_milestones/target07_projection_gemm_backend_parity/summaries/nsys_target0757_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`

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

Keep the same vLLM comparison discipline as TARGET 07.58:

- vLLM's `Fp8LinearMethod.apply` has a BF16 dequant fallback boundary when the
  faster FP8 backend is unavailable;
- vLLM models place `wo_b` behind a row-parallel linear boundary;
- mini's target path should match the high-level idea of a BF16 dequantized
  weight boundary, while caching the dequantized weights once for repeated
  decode use.

Record exact vLLM files/functions consulted in the milestone README.  Do not
claim runtime parity with vLLM unless an actual vLLM profile or probe was run.

## Scope

In scope:

- owner-scoped cached BF16 dequantized-weight backend for `attn.wo_b`;
- preserving the existing TARGET 07.58 `attn.q_wqb` cached BF16 path;
- row-parallel all-reduce correctness and timing separation;
- real-weight microbench for `wo_b`;
- text smoke and macro benchmark with page size 256;
- memory ledger for incremental `wo_b` and combined `q_wqb + wo_b`;
- optional fresh Nsight profile if macro and microbench disagree.

Out of scope:

- enabling cached BF16 for `indexer.wq_b` in this target;
- changing `wo_a`, `wq_a/wkv`, shared experts, MoE/Marlin, or `lm_head`;
- removing activation fake-quant;
- changing row-parallel communication algorithms;
- full model/layer `torch.compile`;
- full `fp8_ds_mla` KV-cache E2E;
- broad graph/layout cleanup.

## Implementation Guidance

Expected new opt-in toggle:

```text
MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE=1
```

Expected new benchmark/text-smoke variant should include both q_wqb and wo_b
cache names, for example:

```text
..._qwqbbf16cache_wobbf16cache_...
```

Implementation notes:

- reuse the TARGET 07.58 cached FP8-to-BF16 weight helper if possible;
- build `wo_b` cached BF16 weights during the explicit model preparation phase,
  after weights are loaded and before CUDA graph capture;
- graph capture and replay must only read the persistent BF16 cache;
- if the `wo_b` cache is missing or stale in forward, raise a clear error
  rather than rebuilding;
- preserve the row-parallel all-reduce behavior of `DSV4Linear.forward`;
- if adding a cached-weight forward helper for row-parallel linears, make the
  reduction explicit and keep the all-reduce label stable enough for profiling;
- do not accidentally route `indexer.wq_b` or other FP8 linears through the new
  cache.

Important: `DSV4Linear.forward_fp8_cached_bf16_weight()` from TARGET 07.58
returns only local `F.linear` output.  For `wo_b`, the implementation must add
or preserve the row-parallel all-reduce after the cached local projection.

## Memory Ledger

The milestone README must include:

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | `360,710,144` | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | | | | | | |
| total | | | | | | |

Reference estimate for TP8 `/models/DeepSeek-V4-Flash`:

- `attn.wo_b` local shape should be `[4096, 1024]`;
- expected incremental `wo_b` cache: `360,710,144 bytes/rank`,
  about `0.3359 GiB/rank`;
- expected combined `q_wqb + wo_b`: `721,420,288 bytes/rank`,
  about `0.6719 GiB/rank`;
- using the prior `76034.41 bytes/token/rank`, combined cache costs about
  `9488` KV tokens/rank, or about `37.1` pages.

These are estimates.  The target must compute final values from benchmark
artifacts and report expected-vs-measured peak memory deltas.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_cached_bf16_wo_b_projection_backend/
```

with:

- `README.md`;
- `scripts/`;
- `raw/`;
- `summaries/`.

Record inherited q_wqb-cached baseline, 07.58 memory ledger, and 07.57 `wo_b`
owner evidence.

### 2. Build Or Extend Real-Weight Microbench

Measure real serving-sharded `attn.wo_b` weights, not the earlier unsharded
shape assumption.

At minimum, report for `M = 1, 4, 8, 16`:

- current FP8 wrapper local projection ms;
- current intrinsic `_quantized_linear_fp8_kernel` ms;
- cached BF16 local `F.linear` ms;
- cached total local projection ms;
- optional all-reduce-inclusive time if the microbench runs under TP;
- max absolute and relative error against current path.

If all-reduce is not included in microbench, say so clearly and rely on macro or
Nsight for communication timing.

### 3. Implement `attn.wo_b` Cached BF16 Path

Implement the new path only for `attn.wo_b`, with q_wqb cache still enabled.

Do not enable:

- `indexer.wq_b`;
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
- evidence that q_wqb and wo_b cached BF16 allocations happen before graph
  capture and are not rebuilt inside replay.

Suggested smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_qwqb_wob_cached_variant> \
  --page-size 256 \
  --output performance_milestones/target07_cached_bf16_wo_b_projection_backend/raw/text_smoke_qwqb_wob.json
```

### 5. Macro Gate

Run 4096/128/batch4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_qwqb_wob_cached_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_cached_bf16_wo_b_projection_backend/raw/macro_qwqb_wob_4096x128_bs4_np128 \
  --keep-going
```

If the short run passes the gate or if focused profiling proves a large compute
win, run 4096/1024/batch4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_qwqb_wob_cached_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_cached_bf16_wo_b_projection_backend/raw/macro_qwqb_wob_4096x1024_bs4_np128 \
  --keep-going
```

### 6. Profile If Needed

Capture a focused rank0 Nsight profile if:

- 4096/128 macro gain is below `5%`;
- 4096/1024 does not improve by at least `3%`;
- microbench improves but macro barely moves;
- memory or graph behavior is unexpected.

Classify at least:

- `attn.q_wqb` local projection;
- `attn.wo_b` local projection;
- `attn.wo_b` row-parallel all-reduce;
- remaining `_quantized_linear_fp8_kernel`;
- `indexer.wq_b`;
- graph/layout cluster.

## Decision Rules

End with exactly one decision:

- `Decision: promote q_wqb + wo_b cached BF16`
  if correctness passes and macro/profile/memory gates pass.
- `Decision: continue cached BF16 to indexer.wq_b`
  if q_wqb + wo_b is promoted and `indexer.wq_b` remains the largest
  same-contract projection owner.
- `Decision: pivot to wo_b communication/overlap`
  if local `wo_b` projection improves substantially but row-parallel all-reduce
  is now the dominant blocker.
- `Decision: pivot to activation-quant/projection contract`
  if cached BF16 matmul is fast but activation fake-quant or wrapper overhead
  becomes the largest part of q_wqb/wo_b projection.
- `Decision: stop cached BF16 expansion`
  if wo_b fails correctness, graph replay, memory, or focused profile gates.

Do not end with "keep optimizing projection generally."  Name the exact next
owner or backend contract.

## Stop Rules

Hard stops:

- `wo_b` cached path misses both focused projection and macro gates;
- row-parallel all-reduce correctness is uncertain;
- graph replay is lost or eager decode becomes nonzero;
- cached BF16 weights are allocated or rebuilt during graph replay;
- measured memory delta is much larger than expected and cannot be explained;
- text smoke fails and one focused fix does not restore it;
- implementation starts touching unrelated projection owners to show a win.

## Expected Output

Create:

- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/README.md`
- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/`
- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/raw/`
- `performance_milestones/target07_cached_bf16_wo_b_projection_backend/summaries/`

The README must include:

- inherited q_wqb-cached baseline;
- vLLM `wo_b` source comparison;
- implementation summary and exact toggles/variant name;
- microbench before/after;
- text smoke and graph replay status;
- 4096/128 macro result;
- 4096/1024 macro result if gate is reached;
- memory ledger with incremental and combined cache cost;
- compute-vs-all-reduce breakdown for `wo_b` if a profile is captured;
- final decision and exact next target recommendation.
