# TARGET 07.56: DSV4 SM80 Low-Cost Graph/Layout Compile Preflight

## Goal

Before starting the full projection/GEMM backend parity target, run one small
code-review-driven preflight pass for low-cost graph/layout and
`torch.compile` opportunities that may reduce projection-adjacent staging.

This target exists because TARGET 07.55 correctly rejected broad graph/layout
cleanup from profiler evidence, but code review still found a few cheap,
well-scoped candidates:

- repeated static projection scale conversions such as
  `scale.float().contiguous()` inside projection Triton wrappers;
- small pure PyTorch math functions, especially HC head, that vLLM compiles
  with `torch.compile`;
- no-op reshape/view/contiguous clutter around projection and quant boundaries
  that vLLM removes through compile passes.

This is not permission to reopen general graph/layout optimization.  It is a
short preflight.  Any accepted change must be opt-in, easy to revert, and
validated against the current 07.54/07.55 active baseline.

## Current Baseline

Use the TARGET 07.54 active opt-in variant as the baseline.  TARGET 07.55 did
not change runtime code.

Representative active variant:

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
| remaining direct-copy kernels | `0.9456` | too diffuse for a broad PoC |
| BF16/float8 copy kernels | `0.1318` | below the old 10% graph/layout gate |
| pow/mean/mul elementwise nodes | `0.5148` | under-attributed without source-boundary work |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- vLLM 4096/128/batch4: about `82.28 output tok/s`;
- vLLM 4096/1024/batch4: about `202.03 output tok/s`.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md`
- `prompts/TARGET_07.55_dsv4_sm80_remaining_graph_layout_or_projection_pivot.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/README.md`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/README.md`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/summaries/remaining_graph_layout_candidate_summary.md`

Mini source areas:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM source references:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/utility/noop_elimination.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/rms_quant_fusion.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`

## Scope

In scope:

- static FP32 contiguous scale/layout cache for projection wrappers, if it
  removes repeated runtime conversions without changing numerics;
- a small opt-in `torch.compile` experiment for pure helper functions such as
  HC head;
- a source audit for no-op reshape/view/contiguous patterns around
  projection-adjacent quant boundaries;
- one or two tiny opt-in PoCs, each with microbench and macro validation;
- a final recommendation for the next projection/GEMM backend parity target.

Out of scope:

- compiling the whole model or whole decoder layer;
- changing default exact BF16 behavior;
- broad attention metadata/indexer rewrites;
- projection/GEMM kernel replacement;
- full `fp8_ds_mla` KV-cache E2E;
- standalone `quantize_and_insert_k_cache`;
- MoE/Marlin, split-K sparse decode, NCCL, or scheduler work.

## Candidate 1: Static Projection Scale Cache

Evidence from code review:

- `python/minisgl/kernel/triton/deepseek_v4.py` repeatedly calls
  `scale.float().contiguous()` in projection wrappers such as
  `quantized_linear_fp8`, `wo_a_grouped_projection_fp8`, `quantized_linear_fp4`,
  and grouped MoE helpers.
- These scales are weight metadata, not per-token activations.
- vLLM keeps quant scale handling behind stable custom-op/linear boundaries.

Tasks:

1. Identify which projection wrappers on the active 07.54 variant still execute
   runtime `scale.float().contiguous()` or equivalent static weight/scale
   layout conversions.
2. Add a microbench or profiling counter to estimate the conversion overhead
   for decode-small `M <= 16`.
3. If justified, implement an opt-in cache, for example:

```text
MINISGL_DSV4_SM80_STATIC_SCALE_CACHE=1
```

The implementation should cache FP32 contiguous scale tensors on the module or
weight owner, with device/dtype/shape invalidation.  Do not mutate checkpoint
weights in a way that changes the default path.

Validation:

- compare cached and uncached outputs for representative FP8/FP4 projection
  wrappers;
- run text smoke;
- run 4096/128/batch4 macro if code changes.

Promotion criterion:

- at least `2%` 4096/128 output-throughput gain; or
- a clear focused profile reduction in projection-adjacent copy/cast kernels,
  with no correctness or graph replay regression.

## Candidate 2: Small Pure-Function `torch.compile`

Evidence from code review:

- vLLM compiles `hc_head` with `@torch.compile`.
- mini's `hc_head_fallback` is a pure PyTorch math chain:
  flatten, float, rsqrt, linear, sigmoid, weighted sum, cast.
- mini has no production `torch.compile` path today.

Tasks:

1. Implement only a narrow opt-in compile probe for HC head, for example:

```text
MINISGL_DSV4_SM80_COMPILE_HC_HEAD=1
```

2. Ensure compilation and warmup happen before CUDA graph capture or in a safe
   path that does not introduce decode-time recompilation.
3. Do not compile the full model, full layer, attention module, or any function
   with KV-cache side effects, NCCL, or mutable global context.

Validation:

- compare compiled and eager HC head numerics on CUDA tensors;
- run text smoke;
- confirm graph capture/replay still works and eager decode remains `0`;
- run 4096/128/batch4 macro if code changes.

Promotion criterion:

- at least `2%` 4096/128 output-throughput gain; or
- a focused profile reduction in HC head / pow-mean-mul elementwise nodes that
  is large enough to justify keeping the opt-in path.

If compile causes graph breaks, recompilation, capture failures, or large
startup overhead without macro gain, remove or disable the path and record the
failure.

## Candidate 3: No-Op Reshape/Contiguous Audit

Evidence from code review:

- vLLM has a `NoOpEliminationPass` that removes redundant reshape/slice around
  quant/custom-op boundaries.
- mini has manual `.contiguous()`, `.view()`, `.reshape()`, and dtype casts
  around projection wrappers and attention output shaping.

Tasks:

1. Audit projection-adjacent paths only:
   `DSV4Linear.forward`, `quantized_linear_ref`, `_flatten_linear_input`,
   `quantize_fp8_activation_ref`, `wo_a_grouped_projection_fp8`, and attention
   WQA/WKV/QWQB/KV/WO call sites.
2. Identify only no-op or redundant transforms that are provably redundant for
   the active 07.54 graph variant.
3. Implement at most one tiny cleanup if it is obviously safe and measurable.

Do not build a mini compiler pass in this target.

Promotion criterion:

- the cleanup removes an identifiable copy/contiguous kernel in a focused
  profile; or
- it improves 4096/128 output throughput by at least `2%`.

## Required Work Plan

### 1. Create Milestone Record

Create:

```text
performance_milestones/target07_low_cost_graph_layout_compile_preflight/
```

The README must record:

- baseline inherited from 07.54/07.55;
- code-review candidates;
- which candidates were implemented or rejected;
- microbench/profile evidence;
- macro before/after if code changed;
- final next-target recommendation.

### 2. Start With Source Review, Not Code

Build a table:

| Candidate | Source evidence | vLLM reference | Expected risk | Expected gain | Decision |
| --- | --- | --- | --- | ---: | --- |
| static scale cache | TBD | TBD | low/medium | TBD | keep/reject |
| compile HC head | TBD | TBD | medium | TBD | keep/reject |
| no-op reshape/contiguous cleanup | TBD | TBD | low | TBD | keep/reject |

Reject anything that is not low-cost or that belongs naturally to the full
projection/GEMM backend parity target.

### 3. Implement At Most Two Tiny PoCs

Allowed combinations:

- static scale cache only;
- HC head compile only;
- static scale cache plus HC head compile;
- one no-op cleanup if it is clearly safe.

Do not stack more than two behavior-changing PoCs in this target.

### 4. Validate Correctness And Graph Semantics

Required if code changes:

- focused unit or micro test for the changed helper;
- text smoke with the active/successor variant;
- graph replay count remains nonzero and eager decode count remains `0`;
- default exact BF16 behavior remains unchanged.

Suggested text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --page-size 256 \
  --output performance_milestones/target07_low_cost_graph_layout_compile_preflight/raw/text_smoke.json
```

Use the successor variant name if a new toggle/variant is added.

### 5. Macro Gate

Run 4096/128/batch4 if code changes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_low_cost_graph_layout_compile_preflight/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

Run 4096/1024/batch4 only if 4096/128 improves by at least `2%` or the profile
shows a targeted reduction likely to matter for long decode.

## Decision Rules

End with exactly one decision:

- `Decision: keep low-cost preflight cut and proceed to projection/GEMM parity`
  if a tiny PoC is correct and gives useful macro/profile improvement.
- `Decision: no low-cost graph/layout cut; proceed to projection/GEMM parity`
  if the candidates do not produce a justified implementation or fail gates.
- `Decision: blocked by compile/capture instability`
  only if `torch.compile` or graph capture fails in a way that prevents a clean
  decision; include the exact failure and disable any partial compile path.

Regardless of outcome, the next major target should be projection/GEMM backend
parity against vLLM unless this target unexpectedly delivers a large
`>=5%` 4096/128 gain and exposes another equally low-cost follow-up.

## Stop Rules

Stop after this preflight.  Do not convert this target into a broad
graph/layout or compiler project.

Hard stops:

- more than two behavior-changing PoCs would be needed;
- a proposed change requires compiling the full model/layer;
- a proposed change requires rewriting projection/GEMM kernels;
- graph replay is lost or eager decode becomes nonzero;
- text smoke fails and one focused fix does not restore it;
- 4096/128 macro gain is below `2%` and no focused profile reduction is shown.

## Expected Output

Create:

- `performance_milestones/target07_low_cost_graph_layout_compile_preflight/README.md`
- `performance_milestones/target07_low_cost_graph_layout_compile_preflight/scripts/`
- `performance_milestones/target07_low_cost_graph_layout_compile_preflight/raw/`
- `performance_milestones/target07_low_cost_graph_layout_compile_preflight/summaries/`

The README must include:

- inherited baseline;
- source-review table;
- vLLM reference comparison;
- implementation summary or rejection rationale;
- tests/smoke/macro results if code changed;
- final decision;
- concrete handoff notes for the projection/GEMM backend parity target.
