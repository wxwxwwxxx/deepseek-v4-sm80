# TARGET 07.58: DSV4 SM80 Cached BF16 Projection Backend

## Goal

Test an opt-in cached BF16 dequantized-weight backend for the dominant
DeepSeek V4 SM80 FP8 projection contract, with explicit VRAM and KV-cache
capacity accounting.

TARGET 07.57 showed that the largest remaining projection/GEMM contract is
mini's `_quantized_linear_fp8_kernel` across:

| Owner | 4096/128 decode-envelope intrinsic s | Real-weight wrapper ms | Cached-dequant BF16 `F.linear` ms |
| --- | ---: | ---: | ---: |
| `attn.q_wqb` | `0.404178` | about `0.412` | about `0.053` |
| `attn.wo_b` | `0.403710` | about `0.660` | about `0.052` |
| `indexer.wq_b` | `0.364756` | about `0.168` | about `0.019` |

The evidence says the current SM80 small-M FP8 projection path is dominated by
repeated weight decode, scale handling, and Triton kernel inefficiency.  This
target should first spend memory to remove that repeated work, then measure
whether the macro speedup is worth the lost KV-cache capacity.

This is a speed-first opt-in target.  The default exact path must remain
unchanged.

## Win Condition

Primary implementation gate:

- implement a graph-safe opt-in cached BF16 dequantized-weight path for
  `attn.q_wqb`; and
- reduce `attn.q_wqb` focused projection time by at least `30%`, or improve
  4096/128/batch4 output throughput by at least `5%` over the active baseline
  `43.0685 output tok/s`; and
- preserve CUDA graph replay with eager decode count `0`; and
- pass text smoke.

Secondary expansion gate:

- only add `attn.wo_b` after `q_wqb` clears the primary gate, or after focused
  profiling proves `q_wqb` improved substantially but macro is masked by
  another equally large `_quantized_linear_fp8_kernel` owner.
- only add `indexer.wq_b` after `q_wqb` and/or `wo_b` results justify extending
  the same backend contract.

Long-decode gate:

- if 4096/128 passes, run 4096/1024/batch4 and require at least `3%` output
  throughput gain over `87.0831 output tok/s`, or explain from profile evidence
  why the short-decode gain does not carry over.

Memory gate:

- record the extra cached BF16 weight memory per rank and the equivalent KV
  cache token/page capacity loss.
- if the path needs more than `3 GiB/rank` for the first `q_wqb` slice, stop and
  explain the discrepancy before expanding to `wo_b`.

## Current Baseline

Active promoted opt-in baseline from TARGET 07.54/07.55:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/batch4 | `87.0831` | `104.3427` | `1023` | `0` |

TARGET 07.56 static scale cache reached only `43.2194 output tok/s` on
4096/128/batch4 (`+0.35%`) and was not promoted.

TARGET 07.57 was attribution-only.  It did not change the runtime baseline,
but it identified `_quantized_linear_fp8_kernel` as the next backend contract:

```text
attn.q_wqb      0.404178s
attn.wo_b       0.403710s
indexer.wq_b    0.364756s
-------------------------
contract total  1.172645s
```

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- vLLM 4096/128/batch4: about `82.28 output tok/s`;
- vLLM 4096/1024/batch4: about `202.03 output tok/s`.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.57_dsv4_sm80_projection_gemm_backend_parity.md`
- `performance_milestones/target07_projection_gemm_backend_parity/README.md`
- `performance_milestones/target07_projection_gemm_backend_parity/summaries/real_fp8_linear_microbench.md`
- `performance_milestones/target07_projection_gemm_backend_parity/summaries/nsys_target0757_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`

Mini source areas:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM source areas:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

## vLLM Comparison Requirement

This target is allowed to be mini-owned, but the design must stay compared
against vLLM.

Specifically, review vLLM's `Fp8LinearMethod` behavior before implementation:

- on hardware/backend combinations without a usable FP8/DeepGEMM path, vLLM has
  a batch-invariant fallback that dequantizes FP8 weights to BF16 and uses
  `F.linear`;
- mini's proposed backend is the same idea with a stronger decode-serving
  assumption: the FP8 weights are static, so the BF16 dequantized weights should
  be cached once instead of recreated at every decode step.

Record the exact vLLM source lines or functions consulted in the milestone
README.  Do not claim runtime parity with vLLM unless an actual vLLM probe or
profile was run successfully.

## Scope

In scope:

- an owner-scoped cached BF16 dequantized-weight backend for `attn.q_wqb`;
- optional expansion to `attn.wo_b` and `indexer.wq_b` only after the gates
  above pass;
- real-weight microbench before/after for the selected owner;
- CUDA graph compatibility;
- explicit memory ledger and KV-cache token/page conversion;
- source-level comparison with vLLM's BF16 dequant fallback boundary.

Out of scope:

- changing the default exact BF16 behavior;
- caching every FP8 linear in the model;
- changing MoE/Marlin;
- revisiting sparse attention split-K;
- full `fp8_ds_mla` KV-cache E2E;
- activation quantization redesign unless it is needed to make the cached
  weight backend correct;
- broad graph/layout cleanup;
- full model or layer `torch.compile`.

## Implementation Guidance

Prefer named opt-in toggles so the owner-level effect is easy to isolate:

```text
MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE=1
MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE=1
MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE=1
```

The implementation may choose different names if they match existing local
style better, but the behavior must remain owner-scoped.

Suggested design:

- add a helper that caches `dequant_fp8_weight(weight, scale, out_dtype=torch.bfloat16).contiguous()`;
- guard the cache by weight pointer, scale pointer, dtype, shape, stride,
  storage offset, device, and tensor version where available;
- keep the cache on the same CUDA device as the original weight;
- allocate or rebuild the cached BF16 tensors before CUDA graph capture, not
  during decode replay;
- use the cached BF16 weight through `F.linear(x_quant, cached_weight)` after
  the existing activation fake-quant path;
- do not route large prefill `M > 16` through this path unless microbench proves
  it is also beneficial;
- make graph capture fail loudly if a supposedly cached path allocates or
  rebuilds during replay.

Important caveat:

- BF16 cached weights are 2 bytes/element, while the FP8 checkpoint weights are
  1 byte/element plus scales.  This removes per-step dequant work but increases
  steady-state HBM footprint and may increase GEMM weight bandwidth.  The
  microbench suggests the trade is promising, but the macro run must decide.

## Preallocation And Workspace Policy

This target must avoid repeated `cudaMalloc`/allocator churn from the large BF16
weight cache.

Required behavior:

- cached BF16 weight storage is created once per module/owner after weights are
  loaded and before CUDA graph capture;
- decode forward, graph capture, and graph replay must only read the cached
  BF16 weight tensor;
- if the implementation uses lazy construction, the first construction must be
  forced by an explicit prepare/warmup step before graph capture;
- if cache metadata becomes stale during CUDA graph capture or replay, raise a
  clear error instead of rebuilding in-place;
- do not allocate a fresh BF16 dequantized tensor inside every `forward`;
- record where the allocation happens in the milestone README.

Implementation options:

- simple per-module persistent tensors are acceptable for the first `q_wqb`
  PoC;
- a single owner-level contiguous arena/workspace with per-layer offsets is
  also acceptable, and is preferred if it keeps memory accounting clearer;
- either way, the README must report expected cache bytes from tensor shapes
  and measured peak memory deltas.

The cache is not a temporary scratch buffer.  For this target it should be a
persistent dequantized-weight cache: write once during preparation, then read
many times during decode.

## Memory Ledger

The milestone README must include a table like this:

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | | | | | | |
| `attn.wo_b` | | | | | | |
| `indexer.wq_b` | | | | | | |

Use measured data whenever possible:

```text
bytes_per_kv_token_per_rank =
  kv_cache_memory_bytes_per_rank_max / (num_pages * page_size)

kv_tokens_lost =
  extra_cached_weight_bytes_per_rank / bytes_per_kv_token_per_rank

kv_pages_lost =
  kv_tokens_lost / page_size
```

The benchmark JSON already reports:

- `peak_gpu_memory_allocated_bytes`;
- `peak_gpu_memory_reserved_bytes`;
- `kv_cache_memory_bytes_per_rank_max`;
- `kv_cache_memory_bytes_per_rank`;
- `num_pages`;
- `page_size`.

Record both expected bytes from tensor shapes and measured peak memory deltas.
If expected and measured differ materially, explain why before expanding the
cache.

Reference estimate for `/models/DeepSeek-V4-Flash` TP8:

| Owner | Local shape from 07.57 microbench | Extra memory estimate |
| --- | --- | ---: |
| `attn.q_wqb` | K=1024, N=32768 | `64 MiB/layer`, about `2.69 GiB/rank` for 43 layers |
| `attn.wo_b` | K=8192, N=4096 | `64 MiB/layer`, about `2.69 GiB/rank` for 43 layers |
| `indexer.wq_b` | K=1024, N=8192 | `16 MiB/C4 layer`, about `336 MiB/rank` for 21 C4 layers |

Using the prior `--num-pages 128`, `page-size 256` run as a rough reference,
KV cache was about `74 KiB/token/rank`.  That means:

- `attn.q_wqb` cache costs about `38k KV tokens/rank`, or about `148` pages;
- `attn.q_wqb + attn.wo_b` costs about `76k KV tokens/rank`, or about `296`
  pages;
- adding actual C4 `indexer.wq_b` costs about another `4.6k KV tokens/rank`, or
  about `18` pages.

These are planning estimates only.  The target must compute the final numbers
from the actual run artifacts.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_cached_bf16_projection_backend/
```

with:

- `README.md`;
- `scripts/`;
- `raw/`;
- `summaries/`.

Record the inherited 07.54/07.55 baseline, 07.56 no-promotion context, and
07.57 owner/microbench evidence.

### 2. Reproduce The Baseline Microbench

Before code changes, rerun or reuse the 07.57 real-weight microbench for:

- `attn.q_wqb.layer0`;
- `attn.wo_b.layer0` as context;
- `indexer.wq_b.layer2` as context.

The microbench must report:

- current wrapper ms;
- current intrinsic `_quantized_linear_fp8_kernel` ms;
- activation quant ms;
- fallback dequant + `F.linear` ms;
- cached BF16 `F.linear` ms;
- max absolute and relative error against current path.

### 3. Implement `attn.q_wqb` Cached BF16 Path Only

Add the first opt-in route for `attn.q_wqb`.

Do not enable the path for:

- `attn.wo_b`;
- `indexer.wq_b`;
- `attn.wo_a`;
- `wq_a/wkv`;
- shared experts;
- `lm_head`.

This keeps the first result attributable.

### 4. Validate Correctness And Graph Semantics

Required:

- focused unit or micro test against current `attn.q_wqb` output;
- text smoke with page size 256;
- CUDA graph replay preserved;
- eager decode count `0`;
- evidence that cached BF16 weight allocation/rebuild did not occur inside
  graph replay;
- no default behavior change when the new toggle is off.

Suggested smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_successor_variant> \
  --page-size 256 \
  --output performance_milestones/target07_cached_bf16_projection_backend/raw/text_smoke_qwqb.json
```

If no benchmark variant exists yet for the new toggle, add one to
`benchmark/offline/deepseek_v4_perf_matrix.py` with a clear name.

### 5. Macro Gate For `q_wqb`

Run 4096/128/batch4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_successor_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_cached_bf16_projection_backend/raw/macro_qwqb_4096x128_bs4_np128 \
  --keep-going
```

Record:

- output tok/s;
- decode tok/s;
- graph replay count;
- eager decode count;
- `peak_gpu_memory_allocated_bytes`;
- `peak_gpu_memory_reserved_bytes`;
- `kv_cache_memory_bytes_per_rank_max`;
- memory delta against baseline.

### 6. Profile And Decide Expansion

Capture a focused rank0 Nsight profile if the macro result is ambiguous or if
the throughput gain is below the 5% gate despite large microbench improvement.

Classify at least:

- `attn.q_wqb` owner time;
- `_quantized_linear_fp8_kernel` total;
- BF16 `F.linear`/cuBLAS time for the cached path;
- graph/layout cluster;
- `attn.wo_b` and `indexer.wq_b` remaining projection time;
- row-parallel all-reduce for `wo_b`.

Expansion rule:

- if `q_wqb` clears the gate, add `wo_b` next and repeat correctness, memory,
  4096/128 macro, and focused profile;
- if `q_wqb` improves focused time but macro is mostly masked by `wo_b`, add
  `wo_b` with a note that the first cut passed the profile gate but not the
  macro gate;
- if `q_wqb` misses both focused and macro gates, stop and pivot to
  kernel-internal `_quantized_linear_fp8_kernel` retuning or scale-broadcast
  optimization instead of caching more weights.

### 7. Long-Decode Validation

Run 4096/1024/batch4 only after the 4096/128 gate passes or after profile data
strongly justifies the extra run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new_successor_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_cached_bf16_projection_backend/raw/macro_qwqb_4096x1024_bs4_np128 \
  --keep-going
```

## Decision Rules

End with exactly one decision:

- `Decision: promote cached BF16 q_wqb`
  if q_wqb passes correctness, graph replay, memory ledger, and macro/profile
  gate.
- `Decision: extend cached BF16 path to wo_b`
  if q_wqb focused time improves enough and `wo_b` is now the next dominant
  same-contract owner.
- `Decision: extend cached BF16 path to indexer.wq_b`
  only after q_wqb/wo_b have passed or been clearly ruled out.
- `Decision: stop cached BF16 path`
  if memory cost is too high, focused time does not move, macro regresses, or
  graph replay breaks.
- `Decision: pivot to FP8 kernel-internal retune`
  if cached BF16 proves the repeated dequant/scale work is not the winning
  tradeoff and `_quantized_linear_fp8_kernel` remains dominant.

Do not end with "keep optimizing GEMM generally."  Name the exact owner and
backend contract.

## Stop Rules

Hard stops:

- q_wqb extra memory exceeds `3 GiB/rank` without an explained accounting
  reason;
- focused q_wqb time improves by less than `30%` and 4096/128 output tok/s
  improves by less than `5%`;
- measured peak memory delta is much larger than expected cached tensor bytes
  and cannot be explained;
- cached BF16 weights are allocated or rebuilt during decode graph replay;
- graph replay is lost or eager decode becomes nonzero;
- text smoke fails and one focused fix does not restore it;
- the implementation requires caching broad unrelated FP8 linears to show a
  gain;
- two owner expansions fail the macro/profile gate.

## Expected Output

Create:

- `performance_milestones/target07_cached_bf16_projection_backend/README.md`
- `performance_milestones/target07_cached_bf16_projection_backend/scripts/`
- `performance_milestones/target07_cached_bf16_projection_backend/raw/`
- `performance_milestones/target07_cached_bf16_projection_backend/summaries/`

The README must include:

- inherited baseline and 07.57 evidence;
- vLLM BF16 dequant fallback source comparison;
- implementation summary and exact toggles/variant name;
- correctness/text smoke;
- real-weight microbench before/after;
- 4096/128 macro result;
- 4096/1024 macro result if gate is reached;
- full memory ledger with KV-token/page conversion;
- final decision and exact next target recommendation.
