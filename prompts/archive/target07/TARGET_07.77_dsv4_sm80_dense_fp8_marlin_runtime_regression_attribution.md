# TARGET 07.77: DSV4 SM80 Dense FP8 Marlin Runtime Regression Attribution

Date: 2026-07-03

## Goal

Explain why TARGET 07.76's mini-owned dense FP8 Marlin runtime opt-in passed
correctness and memory gates but regressed the 4096/1024 macro.

This is a diagnostic target.  Do not expand owner scope, do not promote, and do
not start a new optimization lane until the regression is attributed with
owner-level evidence.

Primary question:

```text
Where did the 4096/1024 regression come from?
```

Candidate causes to distinguish:

- prepare/repack/graph-capture/TTFT overhead;
- steady-state dense Marlin GEMM slower than cached BF16 `F.linear` in the real
  model boundary;
- layout/reshape/contiguous/copy overhead around Marlin apply;
- all-reduce overlap or ordering effects even though communication bytes/counts
  were unchanged;
- benchmark fairness issue caused by per-variant env lifecycle;
- measurement noise.

## Starting Evidence From TARGET 07.76

TARGET 07.76 result:

```text
performance_milestones/target07_mini_owned_fp8_marlin_projection_runtime/README.md
```

Runtime integration succeeded:

- new preferred toggle:
  `MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1`;
- candidate variant:
  `dsv4_sm80_a100_victory_densefp8marlinproj`;
- runtime path uses `minisgl.kernel.dense_fp8_marlin`;
- no vLLM runtime dependency;
- TP8 text smoke passed;
- graph replay stayed active;
- eager decode stayed `0`;
- switched owners skipped duplicate BF16 caches;
- original FP8 weights/scales were released after successful packing;
- peak allocated memory dropped by about `806,961,152 bytes/rank`.

But macro failed:

| Workload | Baseline output tok/s | Candidate output tok/s | Delta |
| --- | ---: | ---: | ---: |
| 4096/128/batch4 `np128` | `55.3472` | `55.9099` | `+1.02%` |
| 4096/1024/batch4 `np128` | `127.4409` | `118.9051` | `-6.70%` |

4096/1024 timing evidence:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| decode tok/s | `168.4840` | `164.5180` | `-2.35%` |
| TTFT s | `5.8932` | `7.6155` | `+1.7223` |
| prefill tok/s | `3180.4989` | `3213.0861` | `+1.02%` |
| prepare s | `2.6811` | `4.4067` | `+1.7256` |
| replay | `1023` | `1023` | same |
| eager decode | `0` | `0` | same |

Communication counters were unchanged:

| Label | Count | Bytes |
| --- | ---: | ---: |
| `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `344` | `46,170,898,432` |
| `dsv4.v1_moe_reduce_once_all_reduce` | `344` | `92,341,796,864` |
| total communication | `704` | `139,602,984,960` |

Known benchmark caveat:

The current single-Engine TP8 benchmark harness applies per-variant env after
LLM construction.  A same-run baseline/candidate text-smoke attempt prepared
only the first variant and failed the candidate with a missing packed dense FP8
Marlin cache.  TARGET 07.76 therefore used separate torchrun invocations.

This target should either fix the harness lifecycle enough for fair same-run
baseline/candidate comparison, or explicitly keep separate invocations and
prove the remaining comparison is stable with repeats.

## Non-Goals

Do not do these in this target:

- promote dense FP8 Marlin into `dsv4_sm80_a100_victory`;
- expand to WQA/WKV/compress or shared experts gate/up;
- change `wo_a`;
- add INT8 MoE;
- add TVM FFI migration;
- change FP8 KV cache;
- rewrite Marlin kernels;
- perform local polishing after attribution is clear.

Small instrumentation or benchmark-harness fixes are allowed if they are needed
to measure fairly.

## Artifacts

Create:

```text
performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/
  README.md
  raw/
  scripts/
  summaries/
```

Large profiler files should live under `raw/` or be symlinked from `/tmp`.

## Implementation Plan

### 1. Stabilize The Comparison

Choose one of these routes and document it:

1. Fix/add a benchmark mode that constructs a fresh LLM/Engine per variant, so
   per-variant env is applied before `prepare_for_cuda_graph_capture()`.
2. Keep separate torchrun invocations but run at least two repetitions for the
   4096/1024 `np128` baseline and candidate to estimate noise.

Required comparison shape:

```text
TP8, page size 256, --num-pages 128
prompt_len 4096
decode_len 1024
batch_size 4
variants:
  dsv4_sm80_a100_victory
  dsv4_sm80_a100_victory_densefp8marlinproj
```

Also keep 4096/128 `np128` as the short profile shape.

### 2. Add Owner-Level Timing

Add focused timing around exactly these runtime owners:

- `attn.q_wqb`;
- `attn.wo_b` local projection before all-reduce;
- `shared_experts.down_proj` before all-reduce.

For each owner, measure both baseline and candidate:

- total owner wall time;
- pure GEMM/custom-op time if isolatable;
- input reshape/view/contiguous/copy time;
- output reshape/view/copy time;
- all-reduce time for `wo_b` and shared experts, separately from local GEMM;
- call count;
- shapes and strides of the input tensor;
- whether a `.contiguous()` branch was taken.

NVTX ranges are acceptable.  CUDA event timing is also acceptable if it does
not perturb graph replay too much.  The final README should explain the method
and its overhead.

Suggested label names:

```text
dsv4.owner.attn.q_wqb.bf16_cache_linear
dsv4.owner.attn.q_wqb.dense_fp8_marlin_apply
dsv4.owner.attn.q_wqb.dense_fp8_marlin_contiguous
dsv4.owner.attn.wo_b.bf16_cache_linear
dsv4.owner.attn.wo_b.dense_fp8_marlin_apply
dsv4.owner.attn.wo_b.row_parallel_all_reduce
dsv4.owner.shared_down.bf16_cache_linear
dsv4.owner.shared_down.dense_fp8_marlin_apply
dsv4.owner.shared_down.shared_expert_all_reduce
```

### 3. Attribute Prepare / TTFT

Break down `prepare_for_cuda_graph_capture()` and graph-capture setup:

- total model prepare time;
- dense FP8 Marlin prepare total;
- per owner family prepare total:
  - q_wqb;
  - wo_b;
  - shared-down;
- per operation where practical:
  - `pack_fp8_to_int32`;
  - `gptq_marlin_repack`;
  - scale cast/transpose/repeat/permutation;
  - FP8 exponent-bias fusion;
  - workspace allocation;
  - original tensor release;
- graph capture time separately from prepare if available.

The 07.76 4096/1024 regression had `prepare s` increase from `2.6811` to
`4.4067`.  This target must explain whether that increase is almost entirely
dense Marlin packing or includes graph capture / allocator side effects.

### 4. Profile The Short Shape

Capture a 4096/128 `np128` profile for baseline and candidate with owner-level
labels enabled.  Use existing nsys scripts if available; otherwise add a small
script under this milestone.

The profile should answer:

- Did projection/GEMM owner time decrease?
- Did layout/copy/contiguous time increase?
- Did all-reduce placement or stream ordering change?
- Are the dense Marlin kernels inside CUDA graph replay or outside it?
- Are there unexpected allocator calls during replay?

### 5. Validate Long-Shape Attribution

Run 4096/1024 `np128` again with enough instrumentation to attribute:

- output tok/s;
- decode tok/s;
- TTFT;
- prepare time;
- decode forward time;
- owner local projection time;
- all-reduce time;
- graph replay/eager counts.

Do not rely only on 4096/128 if the 4096/1024 regression has a different
profile.

## Decision Matrix

At the end, classify the regression into one primary bucket:

| Bucket | Evidence Pattern | Next Action |
| --- | --- | --- |
| prepare-only | Steady-state decode owner time improves or is neutral, but pack/prepare dominates TTFT. | Plan load-time/offline packing, persistent pack cache, or amortized serving policy; do not promote for offline macro yet. |
| Marlin steady-state slower | Candidate owner local GEMM time is worse than cached BF16 in graph replay. | Keep opt-in; investigate kernel selection/shape-specific Marlin or abandon this owner subset. |
| layout/copy overhead | GEMM is faster but contiguous/reshape/copy erases the gain. | Create a narrow layout-boundary cleanup target for the specific owner. |
| communication/ordering | Local GEMM improves but all-reduce wait/ordering worsens. | Create an all-reduce overlap/ordering target; do not expand owner scope. |
| measurement fairness/noise | Repeats or fresh-engine harness invalidate the observed regression. | Rerun 07.76 fair macro and reconsider promotion gates. |
| mixed | Multiple buckets matter. | Rank them by seconds in 4096/1024 and target the largest one only. |

## Success Criteria

This target succeeds when it produces:

- a fair or explicitly repeated baseline/candidate comparison;
- owner-level timing for q_wqb, wo_b local, and shared-down;
- prepare/TTFT breakdown;
- 4096/128 profile evidence;
- 4096/1024 attribution evidence;
- a single recommended next target based on seconds saved, not intuition.

It does not need to improve performance.

## Stop Rules

Stop after attribution is clear enough to select the next target.

Hard stop if:

- TP8 candidate no longer reproduces smoke pass;
- graph replay fails or eager decode appears;
- owner-level instrumentation perturbs timing too much to compare;
- no fair comparison can be produced after trying both fresh-engine and
  repeated separate-invocation routes.

Do not continue into implementation polishing inside this target unless the fix
is purely a measurement bug.

## Suggested README Outline

```text
# TARGET 07.77: Dense FP8 Marlin Runtime Regression Attribution

Status:

## Measurement Method
## Fairness / Repeatability
## 4096/1024 Macro Reproduction
## Prepare / TTFT Breakdown
## Owner-Level Runtime Timing
## 4096/128 Profile
## Regression Classification
## Decision
## Next Target
```

