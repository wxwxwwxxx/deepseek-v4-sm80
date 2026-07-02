# TARGET 07.66: DSV4 SM80 MoE/Shared-Expert Staging Cleanup

Date: 2026-07-02

## Summary

This target removed the largest MoE/shared-expert projection staging owners from
the exact bf16 A100 victory path by reusing the existing cached BF16
dequantized-weight projection helper for shared experts.

Implemented and promoted:

- `MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1`
- `dsv4_sm80_a100_victory_sharedbf16` audit variant
- `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1` now includes the shared expert
  cache after the correctness, owner, macro, and memory gates cleared.

The 07.64 metadata deforestation toggle remains opt-in only and was not added
to the victory bundle.  No INT8 MoE or broad MoE backend rewrite was attempted.

## Git State

- Branch: `dsv4-sglang-based`
- Starting commit during this target: `9769c9a`
- Worktree changes are scoped to DSV4 shared-expert cache plumbing, benchmark
  variants/tests, and this milestone directory.

## Source Review

| Boundary | Mini before this target | vLLM local reference | Decision |
| --- | --- | --- | --- |
| shared gate/up projection | `DSV4SharedExperts.forward` called generic `DSV4Linear.forward`, which reaches `quantized_linear_ref` and dequantizes/materializes FP8 weight work inside replay. | DeepSeek shared experts are `DeepseekV2MLP` with merged column-parallel gate/up linear. They are passed into `FusedMoE` as `shared_experts`. | Reuse mini's existing cached BF16 FP8-weight helper for this projection. |
| shared down projection | Same generic FP8 linear path, row-parallel all-reduce label `dsv4.shared_expert_all_reduce`. | vLLM shared down path is a row-parallel linear inside the shared expert module. | Reuse cached BF16 FP8-weight helper while preserving row-parallel reduce behavior and labels. |
| shared/routed finalization | Mini still combines routed/shared through fp32 staging and returns to flat hidden dtype. | vLLM runner can own shared output application/reduction ordering inside the fused-MoE runner. | Left unchanged in this target because the projection owners were clearly removable and the fp32 boundary is a separate numerics contract. |
| shared expert ordering/overlap | Mini runs shared experts in the existing runner flow with no new stream design. | `runner/shared_experts.py` supports `NO_OVERLAP`, `MK_INTERNAL_OVERLAPPED`, and `MULTI_STREAM_OVERLAPPED`. | Future target only. This target stayed with a narrow cache path. |

Relevant vLLM files reviewed:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v2.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/shared_experts.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`

## Implementation

Mini changes:

- `python/minisgl/kernel/deepseek_v4.py`
  - Added `DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE`.
  - Added it to known/experimental toggles.
  - Added it to `DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST` after the gates
    cleared.
  - Did not add it to the generic `MINISGL_DSV4_SM80_BF16_PROJECTION_CACHE`
    whitelist; generic projection-cache opt-in still means attention/indexer
    only.
- `python/minisgl/models/deepseek_v4.py`
  - `DSV4SharedExperts` now has layer-aware owner labels.
  - Added `DSV4SharedExperts.prepare_bf16_weight_cache()`.
  - `DSV4SharedExperts.forward()` uses
    `forward_fp8_cached_bf16_weight()` for shared gate/up and down projections
    when the toggle is active.
  - `DeepseekV4Model.prepare_for_cuda_graph_capture()` reports
    `shared_expert_bf16_weight_cache` and includes shared bytes in
    `projection_bf16_weight_cache_total`.
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - Added/kept `dsv4_sm80_a100_victory_sharedbf16` as an explicit audit
    variant.
- `benchmark/offline/deepseek_v4_text_smoke.py`
  - Added/kept the matching text-smoke audit variant.

The explicit audit variant remains useful for old artifact scripts.  After
promotion, `dsv4_sm80_a100_victory` and
`dsv4_sm80_a100_victory_sharedbf16` both activate the shared expert cache.

## Memory Ledger

Shared expert cache entries are BF16 dequantized copies of the FP8 weights,
created once before CUDA graph capture.

KV-equivalence uses page size 256 and the current DSV4 KV accounting:

- KV page bytes/rank: `19,313,920`
- KV token bytes/rank: `75,445`

| Cache owner | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank |
| --- | ---: | ---: | ---: | ---: |
| shared `gate_up_proj`, 43 layers, `[512, 4096]` each | `180,355,072` | `0.167969` | `2,390.55` | `9.3381` |
| shared `down_proj`, 43 layers, `[4096, 256]` each | `90,177,536` | `0.083984` | `1,195.28` | `4.6690` |
| total new shared cache | `270,532,608` | `0.251953` | `3,585.83` | `14.0071` |

Projection cache total moves from `1,434,451,968` bytes/rank
(`1.335938` GiB) to `1,704,984,576` bytes/rank (`1.587891` GiB).

This clears the memory gate: the incremental cache is about `0.252` GiB/rank,
well below `1.0` GiB/rank, and the macro gain is not noise-level.

## Correctness

Commands:

```bash
pytest -q -o addopts= \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy \
  tests/models/test_deepseek_v4_forward_fallback.py::test_shared_experts_bf16_weight_cache_matches_generic_path \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_records_shared_expert_bf16_cache \
  tests/benchmark/test_deepseek_v4_text_smoke.py::test_configure_variant_sets_shared_expert_bf16_cache
```

Result: `4 passed`.

```bash
pytest -q -o addopts= \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

Result: `32 passed`.

```bash
pytest -q -o addopts= \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy \
  tests/models/test_deepseek_v4_forward_fallback.py::test_shared_experts_bf16_weight_cache_matches_generic_path
```

Result: `2 passed`.

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target07_moe_shared_expert_staging_cleanup/scripts/classify_direct_copy_owners.py
```

Result: pass.

Text smoke after promotion:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_sharedbf16 \
  --output performance_milestones/target07_moe_shared_expert_staging_cleanup/raw/text_smoke.json
```

Result:

- overall status: `pass`
- variants: `dsv4_sm80_a100_victory` pass,
  `dsv4_sm80_a100_victory_sharedbf16` pass
- graph replay: `18`
- greedy sample replay: `18`
- eager decode: `0`
- shared cache enabled through the victory bundle: `true`
- shared cache bytes/rank: `270,532,608`
- projection cache total bytes/rank: `1,704,984,576`

## Macro Performance

Short macro command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_sharedbf16 \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_moe_shared_expert_staging_cleanup/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

Long macro command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_sharedbf16 \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_moe_shared_expert_staging_cleanup/raw/macro_4096x1024_bs4_np128 \
  --keep-going
```

The macro reports were captured with the explicit audit variant before final
bundle promotion.  After promotion, the same toggle set is active for
`dsv4_sm80_a100_victory`.

| Workload | Baseline output tok/s | New output tok/s | Delta | Baseline decode tok/s | New decode tok/s | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `59.5264` | `62.2034` | `+4.50%` | `150.2022` | `168.6592` | `+12.29%` |
| 4096/1024/batch4 | `119.4153` | `131.7707` | `+10.35%` | `149.1220` | `169.1898` | `+13.46%` |

Additional macro fields:

| Workload | TTFT mean s | Prefill tok/s | Graph replay | Greedy replay | Eager decode | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `4.972443` | `3832.2368` | `508` | `508` | `0` | `47,565,656,064` |
| 4096/1024/batch4 | `4.956396` | `3836.9788` | `4092` | `4092` | `0` | `47,565,686,784` |

## Direct-Copy Owner Profile

Profile command:

```bash
VARIANT=dsv4_sm80_a100_victory_sharedbf16 \
RUN_TAG=target0766_dsv4_sm80_a100_victory_sharedbf16_4096x128_bs4_np128 \
performance_milestones/target07_moe_shared_expert_staging_cleanup/scripts/nsys_direct_copy_owner_4096x128_bs4.sh
```

Classifier artifacts:

- `summaries/nsys_target0766_dsv4_sm80_a100_victory_sharedbf16_4096x128_bs4_np128_rank0_direct_copy_owner.json`
- `summaries/nsys_target0766_dsv4_sm80_a100_victory_sharedbf16_4096x128_bs4_np128_rank0_direct_copy_owner.md`

Rank0 profile summary:

| Metric | TARGET 07.65 promoted baseline | TARGET 07.66 shared cache |
| --- | ---: | ---: |
| total direct_copy s | `0.737039` | `0.449052` |
| total direct_copy delta | - | `-0.287987` (`-39.07%`) |
| named owner coverage | `99.97%` | `99.94%` |
| residual direct_copy s | `0.000245` | `0.000252` |
| MoE/shared staging group s | `0.379204` | `0.097361` |
| MoE/shared staging delta | - | `-0.281843` (`-74.32%`) |

Owner changes:

| Owner | 07.65 s | 07.66 s | Result |
| --- | ---: | ---: | --- |
| `dsv4.shared_experts.gate_up_proj` | `0.165751` | `0.000000` | removed from owner table |
| `dsv4.shared_experts.down_proj` | `0.119724` | `0.000000` | removed from owner table |
| `dsv4.layer*.mlp.runner.experts` | `0.053714` | `0.054546` | unchanged routed runner cost |
| `dsv4.layer*.mlp.runner.shared` | `0.031286` | `0.031721` | remaining shared runner boundary |
| `moe_shared_expert_staging.runner_finalize_to_fp32.layer*` | `0.022872` | `0.023179` | remaining finalization boundary |
| `dsv4.layer*.mlp.runner.route` | `0.021675` | `0.021900` | unchanged route boundary |
| `moe_shared_expert_staging.runner_shared_to_fp32.layer*` | `0.020026` | `0.022688` | remaining fp32 combine boundary |
| `moe_shared_expert_staging.runner_output_to_flat_dtype.layer*` | `0.011815` | `0.011906` | remaining dtype return boundary |
| `moe_shared_expert_staging.shared_hidden_to_up_dtype` | `0.007730` | `0.007866` | still required by current SwiGLU/down dtype path |

The owner gate is cleared: the target group fell by `0.281843s`, and the
largest single owner, `dsv4.shared_experts.gate_up_proj`, fell by 100%.

## Promotion Decision

Promote.

Rationale:

- Correctness passed: unit tests and TP8 text smoke pass.
- Graph replay remains active and eager decode remains `0`.
- Owner gate cleared by a large margin.
- Macro gate cleared: 4096/1024/batch4 output tok/s improved by `10.35%`.
- Memory gate cleared: incremental cache is `270,532,608` bytes/rank
  (`0.251953` GiB/rank), equivalent to about `14.01` KV pages/rank.

The promoted path remains bf16/exact.  No INT8 MoE, metadata deforestation,
attention rewrite, NCCL change, KV-cache change, or sampler change was included.

## Next Work

The shared projection owners are no longer the top MoE/shared staging issue.
The next focused targets should be:

1. Runner finalization boundaries:
   `runner_shared_to_fp32`, `runner_finalize_to_fp32`, and
   `runner_output_to_flat_dtype`.
2. A vLLM-style shared expert ordering/overlap audit if finalization cleanup
   does not move macro performance enough.
3. A separate INT8/WNA16 MoE feasibility target only if precision gates and
   backend ownership are defined independently.

Do not keep chasing metadata for this owner shape; 07.64 already showed that
metadata deforestation does not materially change the MoE/shared direct-copy
profile.
