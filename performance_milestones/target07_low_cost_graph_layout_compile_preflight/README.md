# TARGET 07.56: Low-Cost Graph/Layout Compile Preflight

Date: 2026-07-02

## Result

This target ran the low-cost preflight before projection/GEMM backend parity.
One tiny opt-in PoC was implemented:

```text
MINISGL_DSV4_SM80_STATIC_SCALE_CACHE=1
```

The cache is correct and removes the focused projection-wrapper scale
`float().contiguous()` cast/copy events, but it did not meet the macro
promotion gate.  The 4096/128/batch4 output line moved from `43.0685` to
`43.2194` output tok/s, only `+0.35%` versus the required `+2%`.

Decision: no low-cost graph/layout cut; proceed to projection/GEMM parity

## Required Inputs Inspected

Prompt and milestone inputs:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md`
- `prompts/TARGET_07.55_dsv4_sm80_remaining_graph_layout_or_projection_pivot.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/README.md`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/README.md`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/summaries/remaining_graph_layout_candidate_summary.md`

Mini source areas inspected:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM source areas inspected:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/utility/noop_elimination.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/rms_quant_fusion.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`

## Inherited Baseline

Active 07.54/07.55 variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/batch4 | `87.0831` | `104.3427` | `1023` | `0` |

07.54/07.55 profile baseline:

| Bucket | Kernel s | Note |
| --- | ---: | --- |
| graph/layout cluster | `1.8271` | `graph_runtime_copy_cat_index + elementwise_graph_nodes` |
| projection/GEMM | `1.7968` | co-dominant next lever |
| `_quantized_linear_fp8_kernel` | `1.1726` | largest named projection kernel |
| remaining direct-copy kernels | `0.9456` | too diffuse for broad cleanup |
| BF16/float8 copy kernels | `0.1318` | below old 10% graph/layout gate |
| pow/mean/mul elementwise nodes | `0.5148` | under-attributed after 07.54 |

## Source Review Table

| Candidate | Source evidence | vLLM reference | Expected risk | Expected gain | Decision |
| --- | --- | --- | --- | ---: | --- |
| static scale cache | Active projection wrappers still call `scale.float().contiguous()` in `quantized_linear_fp8`, `wo_a_grouped_projection_fp8`, and `quantized_linear_fp4`; grouped FP4 MoE helpers do the same but are not active under the 07.54 Marlin-WNA16 path. `DSV4Linear.forward` and attention `wo_a` pass module-owned static scales every decode. | vLLM keeps quant/scale handling behind `QuantFP8`, scaled FP8 quant custom ops, and stable linear/custom-op boundaries. | low/medium: extra cached tensor per projection scale, guarded by env, with data_ptr/version/device/dtype/shape/stride/storage-offset invalidation. | small: focused cast/copy removal, unlikely to move full macro alone. | Implemented as opt-in PoC; correct but not promoted because macro gain was only `+0.35%`. |
| compile HC head | Mini `hc_head_fallback` is a pure PyTorch chain at `python/minisgl/kernel/deepseek_v4.py:2449`: flatten, float, rsqrt, linear, sigmoid, weighted sum, cast. Mini calls it at `python/minisgl/models/deepseek_v4.py:1386`. | vLLM uses `@torch.compile` on `hc_head` at `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:1347`. | medium: shape-specialized compile during graph capture warmup, distributed startup overhead, and possible graph/capture instability. | unknown: 07.55 pow/mean/mul bucket is sizable but not attributed specifically to HC head. | Rejected for this preflight after static-scale macro miss; do not add an unprofiled compile path before projection/GEMM parity. |
| no-op reshape/contiguous cleanup | `_flatten_linear_input` uses `x.contiguous().view` before Triton projection kernels, and `wo_a` reshapes attention output before grouped projection. These are projection-adjacent but not provably no-op across active shapes and strides. | vLLM's `NoOpEliminationPass` removes reshape/slice clutter around quant/custom-op boundaries; RMS/quant and QK norm/rope fusion passes depend on compiler-owned graph rewrites. | low if isolated, but unsafe without stride proof; broad pass is out of scope. | below gate unless stacked across multiple owners. | Rejected; no cleanup implemented. |

## Implementation

Files changed:

| File | Change |
| --- | --- |
| `python/minisgl/kernel/deepseek_v4.py` | Added `DSV4_SM80_STATIC_SCALE_CACHE_TOGGLE` to known experimental toggles. |
| `python/minisgl/models/deepseek_v4.py` | Added `_cached_projection_scale` and routed `DSV4Linear.forward` plus attention `wo_a` through it when enabled. |
| `benchmark/offline/deepseek_v4_perf_matrix.py` | Added the successor scale-cache benchmark variant. |
| `benchmark/offline/deepseek_v4_text_smoke.py` | Added the successor scale-cache smoke variant. |
| `tests/kernel/test_deepseek_v4_wrappers.py` | Added focused CUDA coverage for cache reuse/invalidation and FP8/FP4/`wo_a` output parity. |
| `scripts/microbench_static_scale_cache.py` | Added a decode-small `M <= 16` microbench for raw versus cached scale wrappers. |

Successor variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_scalecache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

The default exact BF16 path is unchanged.  The cache does not mutate checkpoint
weights or scale tensors; it stores FP32 contiguous scale tensors on the weight
owner and invalidates on data pointer, version, device, dtype, shape, stride,
or storage offset changes.

## Microbench And Focused Profile

Artifact:

- `raw/static_scale_cache_microbench.json`
- `summaries/static_scale_cache_microbench.md`
- `summaries/static_scale_cache_torch_profile.json`
- `summaries/static_scale_cache_torch_profile.md`

Decode-small FP8 wrapper microbench, `N=512`, `K=128`, A100:

| M | scale convert us | raw wrapper us | cached wrapper us | delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `9.2083` | `69.2915` | `52.7979` | `-16.4936 us` |
| 4 | `9.2920` | `70.1669` | `52.9432` | `-17.2237 us` |
| 8 | `9.3114` | `69.6797` | `53.1019` | `-16.5778 us` |
| 16 | `9.2843` | `69.7760` | `52.6165` | `-17.1595 us` |

Focused `torch.profiler` on 20 FP8 wrapper calls, `M=4`, `N=512`, `K=128`:

| Scale path | `aten::_to_copy` count | `aten::copy_` count | direct-copy CUDA kernel count | direct-copy CUDA us |
| --- | ---: | ---: | ---: | ---: |
| raw e8m0 scale | `20` | `20` | `20` | `51.0050` |
| cached FP32 scale | `0` | `0` | `0` | `0.0000` |

The focused reduction is real, but too small at full-model scale to justify
another graph/layout target.

## Validation

Compilation:

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target07_low_cost_graph_layout_compile_preflight/scripts/microbench_static_scale_cache.py
```

Unit and benchmark-variant tests:

```bash
python -m pytest -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_static_projection_scale_cache_preserves_projection_outputs -q

python -m pytest -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy -q

python -m pytest -o addopts='' \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py -q
```

Results:

| Check | Result |
| --- | --- |
| `py_compile` | pass |
| static scale CUDA test | pass |
| env policy test | pass |
| benchmark/text-smoke variant tests | `28 passed` |
| microbench | pass |

Text smoke:

| Artifact | Status | Graph |
| --- | --- | --- |
| `raw/text_smoke_scalecache.json` | pass | captured `[4, 2, 1]`, replay `9`, greedy replay `9`, eager decode `0` |

## Macro Result

4096/128/batch4:

| Variant | Output tok/s | Decode tok/s | Delta output | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 07.54/07.55 baseline | `43.0685` | `104.2028` | baseline | `127` | `0` |
| scale-cache successor | `43.2194` | `105.6231` | `+0.35%` | `127` | `0` |

Artifact:

- `raw/macro_4096x128_bs4_np128_scalecache/summary.json`

The `+2%` 4096/128 promotion threshold would require at least `43.9299`
output tok/s.  The successor did not reach it.  4096/1024 was not run because
the short macro did not pass and the focused profile reduction is too small to
be likely to change the long-decode decision.

## vLLM Reference Comparison

Static scale/cache boundary:

- Mini previously paid repeated wrapper-level scale casts in Triton projection
  wrappers even though the scales are static module metadata.
- The opt-in cache moves the scale dtype/layout conversion to a module-owner
  cache, closer to vLLM's stable quant/linear boundary discipline.
- vLLM's `QuantFP8` and scaled FP8 quant ops still represent a larger backend
  contract than this mini-side cache.  The remaining macro bottleneck is not
  solved by this staging cut.

HC head compile:

- vLLM compiles `hc_head`, but mini has no production `torch.compile` path.
- Adding a one-off compile after the static-scale macro miss would introduce
  startup/capture risk without source-attributed profile evidence that HC head
  is large enough to matter.

No-op cleanup:

- vLLM compiler passes can remove reshape/slice no-ops globally.
- Mini's active projection-adjacent reshapes/contiguous calls are local wrapper
  contracts, often needed to satisfy Triton row-major assumptions.  A mini
  compiler pass is out of scope.

## Handoff To Projection/GEMM Parity

The next target should proceed directly to projection/GEMM backend parity.
Start from the 07.55 plan and keep the 07.56 scale-cache result as context:

- attribute the `1.7968 s` projection/GEMM bucket by projection owner:
  `attn.q_proj`, `attn.q_wqb`, `attn.wo_a`, `attn.wo_b`, indexer `wq_b`,
  shared experts, and FFN projections;
- separate intrinsic `_quantized_linear_fp8_kernel` time from wrapper staging;
- compare mini FP8 wrapper/GEMM behavior against vLLM `QuantFP8`,
  `_C.scaled_fp8_quant`, quantized `ColumnParallelLinear`/`RowParallelLinear`,
  `deepseek_v4_fp8_einsum`, and the SM80 `wo_a` BMM/reference path;
- require a focused gate of at least `0.50 s` attributable projection subpath,
  then either `15%` projection/GEMM kernel-time reduction or `5%` 4096/128
  output-throughput gain with graph replay and eager decode `0`;
- do not continue general graph/layout cleanup unless a fresh profile exposes
  another single, low-risk source boundary larger than this static-scale cut.

## Final Decision

Decision: no low-cost graph/layout cut; proceed to projection/GEMM parity
