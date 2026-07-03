# TARGET 07.76: Mini-Owned FP8 Marlin Projection Runtime

Date: 2026-07-03

Status: stop after one full TP8 decision.  The runtime bridge is now mini-owned
and works as an explicit opt-in, but it is not promoted into
`dsv4_sm80_a100_victory` because the 4096/1024 macro regressed.

## Implementation

The projection runtime path was moved from the TARGET 07.74 vLLM-helper bridge
to `minisgl.kernel.dense_fp8_marlin`:

- `MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1` is the new preferred toggle.
- `MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1` remains a legacy alias, but
  now routes to the mini-owned backend.
- The backend report name is `mini_dense_fp8_marlin_w8a16_block`.
- The Phase A owners are exactly `attn.q_wqb`, `attn.wo_b` local projection,
  and `shared_experts.down_proj`.
- Prepare happens in `prepare_for_cuda_graph_capture()` before graph capture.
  Forward raises if the packed cache is missing, so decode replay does not
  repack or allocate the projection weights.
- Switched owners skip their promoted BF16 caches and release the original FP8
  `weight` / `weight_scale_inv` after successful packing.

No Phase B owner, INT8 MoE, TVM FFI, routed MoE backend, or FP8 KV cache change
was made.

Current visible git status after this target:

```text
 M benchmark/offline/deepseek_v4_perf_matrix.py
 M benchmark/offline/deepseek_v4_text_smoke.py
 M python/minisgl/kernel/deepseek_v4.py
 M python/minisgl/models/deepseek_v4.py
 M tests/models/test_deepseek_v4_forward_fallback.py
?? performance_milestones/target07_mini_owned_fp8_marlin_projection_runtime/README.md
?? performance_milestones/target07_mini_owned_fp8_marlin_projection_runtime/scripts/run_commands.md
?? performance_milestones/target07_mini_owned_fp8_marlin_projection_runtime/summaries/decision_metrics.json
```

The `raw/` artifacts are present in the workspace and ignored by
`.gitignore:23` (`performance_milestones/**/raw/`).

## Environment And Variants

- Host: 8x A100/sm80 TP8.
- Python stack: default mini Python, torch `2.9.1+cu128`, CUDA `12.8`.
- Model: `/models/DeepSeek-V4-Flash`.
- Page size: `256`.
- Candidate variant:
  `dsv4_sm80_a100_victory_densefp8marlinproj`.
- Candidate env:
  `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1` and
  `MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1`.

The current single-Engine TP8 benchmark harness applies per-variant env after
LLM construction.  A same-run baseline/candidate text-smoke attempt therefore
prepared the first variant only and failed the candidate with a missing packed
dense FP8 Marlin cache.  Baseline and candidate TP8 evidence below were run as
separate torchrun invocations.

## TP8 Text Smoke

Raw files:

- `raw/text_smoke_baseline.json`
- `raw/text_smoke_candidate.json`

| Variant | Status | Elapsed s | Replay | Eager decode | Sane outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | pass | 1.3377 | 9 | 0 | 3/3 |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | pass | 1.3916 | 9 | 0 | 3/3 |

Candidate answers were sane:

- `2 + 2 equals 4.`
- `The sky is blue on a clear day.`
- `Hangzhou is a scenic historic and cultural city.`

Candidate prepare report on rank 0:

- dense projection cache enabled: true.
- layers cached: 43.
- backend: `mini_dense_fp8_marlin_w8a16_block`.
- total persistent bytes/rank: `412,195,248`.
- total workspace bytes/rank: `55,728`.
- total original released bytes/rank: `405,823,680`.
- duplicate BF16 cache for switched owners: false.
- `q_wqb` BF16 cache layers: `0`.
- `wo_b` BF16 cache layers: `0`.
- shared expert BF16 cache remains enabled only for gate/up; down moved to dense
  FP8 Marlin.

## Macro Results

The prompt command without `--num-pages` allocated the default large KV pool and
OOMed before a valid profile.  The valid runs use `--num-pages 128`, matching
recent `np128` target profiles.  The OOM artifact is kept under
`raw/4096x128_baseline_default_memory_oom/`.

4096/128, batch 4, decode throughput scenario, `--num-pages 128`:

| Variant | Output tok/s | Decode tok/s | TTFT s | Prefill tok/s | Elapsed s | Peak alloc bytes | Replay | Eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 55.3472 | 168.2133 | 5.9709 | 3143.0618 | 9.2507 | 47,565,656,064 | 127 | 0 |
| candidate | 55.9099 | 170.3545 | 5.9117 | 3170.6311 | 9.1576 | 46,758,694,912 | 127 | 0 |
| delta | +1.02% | +1.27% | -0.0592 | +0.88% | -1.01% | -806,961,152 | same | same |

4096/1024, batch 4, decode throughput scenario, `--num-pages 128`:

| Variant | Output tok/s | Decode tok/s | TTFT s | Prefill tok/s | Elapsed s | Peak alloc bytes | Replay | Eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 127.4409 | 168.4840 | 5.8932 | 3180.4989 | 32.1404 | 47,565,686,784 | 1023 | 0 |
| candidate | 118.9051 | 164.5180 | 7.6155 | 3213.0861 | 34.4476 | 46,758,725,632 | 1023 | 0 |
| delta | -6.70% | -2.35% | +1.7223 | +1.02% | +7.18% | -806,961,152 | same | same |

The 4096/128 gate passes.  The 4096/1024 promotion gate fails because output
throughput regressed instead of improving by at least 3%.

## Profile / Owner Attribution

Report-level owner state deltas:

| Owner | Baseline cache | Candidate cache | Layers | Persistent bytes/rank |
| --- | --- | --- | ---: | ---: |
| `attn.q_wqb` | BF16 | dense FP8 Marlin | 43 | 360,710,144 -> 183,191,696 |
| `attn.wo_b` local | BF16 | dense FP8 Marlin | 43 | 360,710,144 -> 183,191,696 |
| `shared_experts.down_proj` | BF16 | dense FP8 Marlin | 43 | 90,177,536 -> 45,811,856 |

The perf-matrix report does not expose per-owner kernel timings for these
projection GEMMs.  The visible bottleneck labels stayed decode dominated,
attention, MoE / expert GEMM, KV cache writes, and metadata construction.

4096/1024 timing evidence:

| Variant | Decode forward s | Prefill forward s | Prepare s | Prepare / elapsed |
| --- | ---: | ---: | ---: | ---: |
| baseline | 24.2872 | 5.1514 | 2.6811 | 8.34% |
| candidate | 24.8727 | 5.0991 | 4.4067 | 12.79% |

Communication counters were unchanged, so the regression is not explained by
additional all-reduce traffic:

| Label | Count | Bytes |
| --- | ---: | ---: |
| `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | 344 | 46,170,898,432 |
| `dsv4.v1_moe_reduce_once_all_reduce` | 344 | 92,341,796,864 |
| total communication | 704 | 139,602,984,960 |

The current evidence rules out graph replay loss, eager decode fallback,
communication byte/count growth, and duplicate BF16+Marlin owner memory.  The
remaining likely causes are dense projection runtime/layout overhead and the
larger prepare/first-token cost visible in the 4096/1024 run.  A follow-up
profile needs owner-level CUDA/NVTX timing before expanding the owner scope.

## Memory Lifecycle

Rank-0 KV memory in the valid `np128` runs is `2,491,495,680` bytes for
129 pages including the dummy page, so one page is `19,313,920` bytes and
represents 256 tokens.

| Item | Bytes/rank | KV pages | KV tokens |
| --- | ---: | ---: | ---: |
| Baseline BF16 switched-owner caches | 811,597,824 | 42.02 | 10,758 |
| Candidate dense FP8 Marlin persistent | 412,195,248 | 21.34 | 5,464 |
| Candidate workspace | 55,728 | 0.00 | 1 |
| Candidate original FP8 released | 405,823,680 | 21.01 | 5,379 |
| Candidate delta vs BF16 switched-owner caches | -399,402,576 | -20.68 | -5,294 |

Peak allocated memory also moved in the expected direction:

- 4096/128: `47,565,656,064 -> 46,758,694,912` bytes/rank.
- 4096/1024: `47,565,686,784 -> 46,758,725,632` bytes/rank.

No switched owner retained both BF16 and dense Marlin packed weights.

## vLLM Dependency Audit

Runtime code now imports and calls `minisgl.kernel.dense_fp8_marlin` for
projection prepare/apply.  The audit command:

```bash
rg -n "from minisgl\\.kernel import vllm_fp8_marlin|import minisgl\\.kernel\\.vllm_fp8_marlin|vllm_fp8_marlin\\.prepare|vllm_fp8_marlin\\.apply" \
  python/minisgl benchmark/offline tests
```

returned no matches.  `python/minisgl/kernel/vllm_fp8_marlin.py` still exists in
the tree, but the candidate runtime path does not depend on importing it or on a
vLLM environment.

## Decision

Promotion gates:

- TP8 page-size-256 text smoke: pass.
- Graph replay active and eager decode zero: pass.
- 4096/128 not worse than -1%: pass.
- 4096/1024 output tok/s at least +3%: fail, observed -6.70%.
- No duplicate BF16+Marlin cache for switched owners: pass.
- No vLLM runtime dependency: pass.
- Owner attribution improved or removed from top bottlenecks: not proven by the
  available perf-matrix report; macro regression stops promotion regardless.

Decision: do not promote into `dsv4_sm80_a100_victory`.  Keep the dense FP8
Marlin projection runtime behind the explicit opt-in and stop this target with
the regression documented.

## Next Target

Reprofile the opt-in path with owner-level CUDA/NVTX timing for `attn.q_wqb`,
`attn.wo_b`, and `shared_experts.down_proj`, plus layout/copy and TTFT/prepare
attribution.  Do not expand to Phase B owners until the 4096/1024 regression is
understood.
