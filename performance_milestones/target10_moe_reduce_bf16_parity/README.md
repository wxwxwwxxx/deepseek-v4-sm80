# TARGET 10.15: DSV4 SM80 MoE Reduce-Once BF16 Parity

Status: complete. The BF16 MoE reduce-once path is implemented and verified as an explicit opt-in. It is not promoted into the A100 victory bundle yet.

Decision: keep opt-in with `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1`. The change fixes the high-priority dtype/bytes mismatch found in TARGET 10.1, passes smoke, keeps decode graph replay at zero eager fallback, and gives enough evidence for TARGET 10.2 to run backend experiments on a fixed BF16 MoE-reduce dtype. Bundle promotion should wait for repeat-stable macro runs and a separate logits gather decision.

## Scope

This target only changes the final MoE reduce-once communication dtype. It does not tune PyNCCL/NCCL/custom communicator backends, does not rewrite the MoE backend, does not add FP8/INT8/activation quantization, and does not change attention or prefix-cache ownership.

Artifacts:

- Text smoke: `raw/text_smoke_routeb_lifetime_moereducebf16*.json`
- Communication A/B: `raw/comm_ab_historical_4096_128_bs4/`
- Owner timing: `raw/owner_timing_historical_4096_128_bs4/`
- Macro A/B: `raw/macro_ab_required_r01/`
- Nsight profile: `raw/nsys/`

## Dtype Flow

### Mini

| Path | Live status | Routed output | Shared output | Local combine | All-reduce input | Post-reduce output |
|---|---|---|---|---|---|---|
| Current default non-runner reduce-once | Live when `V1_MOE`/`MOE_V2` is enabled without runner; not the current promoted prefix bundle path | `experts.forward(...).float()` -> fp32 | `shared_experts.forward(...).float()` -> fp32 | fp32 | default fp32 | `y.to(flat.dtype)`, normally BF16 |
| Current promoted runner | Current promoted A100/prefix bundle path because `MINISGL_DSV4_SM80_MOE_VLLM_RUNNER` is in the bundle | `finalize_routed(...).float()` -> fp32 | `apply_shared(...).float()` -> fp32 | fp32 | default fp32 | `y.to(flat.dtype)`, normally BF16 |
| BF16 opt-in non-runner | Covered by unit test and available when non-runner reduce-once is selected | unchanged fp32 | unchanged fp32 | fp32 | BF16 if hidden dtype is BF16 and `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1` | BF16 |
| BF16 opt-in runner | Current bundle plus explicit opt-in variant | unchanged fp32 | unchanged fp32 | fp32 | BF16 if hidden dtype is BF16 and `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1` | BF16 |

Implementation references:

- Flag: `python/minisgl/kernel/deepseek_v4.py:33`, known/experimental but not in `DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST`.
- Cast helper: `python/minisgl/models/deepseek_v4.py:2187`.
- Runner reduce path: `python/minisgl/models/deepseek_v4.py:2307`.
- Non-runner reduce path: `python/minisgl/models/deepseek_v4.py:2433`.

### vLLM Source-Derived Boundary

| vLLM source point | Evidence | Inferred dtype |
|---|---|---|
| SM80 standard FusedMoE path | MegaMoE disabled for reference kernels in `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:618` | Standard FusedMoE |
| Router logits | `router_logits_dtype=torch.float32` at `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:736` | Router fp32, not hidden output |
| MoE output allocation | `output = torch.empty_like(hidden_states)` at `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/modular_kernel.py:1377` | hidden-state dtype, BF16 for this model |
| Late final reduce | `states = tensor_model_parallel_all_reduce(states)` at `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:377` | reduce combined hidden-state tensor |
| MXFP4 activations | `get_supported_act_dtypes() -> [torch.bfloat16]` at `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/mxfp4.py:66` | BF16 hidden activation path |

Conclusion: vLLM SM80 source indicates a BF16 hidden-state final MoE all-reduce, while mini previously reduced the same logical tensor as fp32.

## Env Flag And Implementation

New opt-in:

```text
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
```

The flag is known and experimental, but intentionally not part of the promoted A100 victory bundle. With the flag disabled, mini keeps the old fp32 reduce behavior. With the flag enabled, mini keeps routed/shared local computation and local fp32 combine unchanged, then casts only the final combined tensor to BF16 before `dsv4.v1_moe_reduce_once_all_reduce`.

Benchmark variants added:

- `dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16` in `benchmark/offline/deepseek_v4_perf_matrix.py`.
- Same variant in `benchmark/offline/deepseek_v4_text_smoke.py`.

Targeted tests passed:

```text
pytest -q tests/models/test_deepseek_v4_forward_fallback.py -q
pytest -q tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy -q
pytest -q tests/benchmark/test_deepseek_v4_perf_matrix.py::test_route_b_lifetime_moe_reduce_bf16_variant_extends_promoted_env tests/benchmark/test_deepseek_v4_text_smoke.py::test_route_b_lifetime_moe_reduce_bf16_variant_extends_promoted_env -q
```

## Correctness Gate

Command:

```bash
torchrun --standalone --nproc-per-node 8 benchmark/offline/deepseek_v4_text_smoke.py \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
             dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --output performance_milestones/target10_moe_reduce_bf16_parity/raw/text_smoke_routeb_lifetime_moereducebf16.json \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 32 --fail-on-warning
```

Result: both variants passed with 0 errors.

| Prompt index | fp32 reduce | BF16 reduce | Same token ids |
|---:|---|---|---|
| 0 | `2 + 2 equals 4.` in Chinese | same text | yes |
| 1 | `The sky is blue on a clear day.` | `Blue.` | no |
| 2 | Chinese Hangzhou sentence | shorter Chinese Hangzhou sentence | no |

Full logits drift was not collected in this target. The next-token/text drift was 2 of 3 prompts under deterministic smoke, but all outputs passed the existing sanity checks and no malformed, repetitive, or visibly degraded generation was observed. This is one reason to keep opt-in instead of promoting immediately.

## Communication Stats

Source: `raw/comm_ab_historical_4096_128_bs4/`, `historical_4096_128_bs4`, repeats 1, warmup repeats 1. Graph row below uses the per-case graph counters, not cumulative warmup counters.

| Variant | Total comm GiB | MoE dtype | MoE shape | MoE count | MoE GiB | Graph replay/eager |
|---|---:|---|---|---:|---:|---|
| fp32 reduce | 260.031 | `float32` | `[16384,4096] -> [16384,4096]` | 688 | 172.000 | 127/0 |
| BF16 reduce opt-in | 174.031 | `bfloat16` | `[16384,4096] -> [16384,4096]` | 688 | 86.000 | 127/0 |

The same shape/count drops from 184,683,593,728 bytes to 92,341,796,864 bytes, exactly 2x. The total communication drops by 86 GiB for this A/B run.

Other hot collectives stayed unchanged:

| Label | Dtype | Count | GiB |
|---|---|---:|---:|
| `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | BF16 | 688 | 86.000 |
| `dsv4.embedding_all_reduce` | BF16 | 16 | 2.000 |
| `dsv4.lm_head_all_gather` | fp32 | 16 | 0.031 |

## Remaining fp32 Communication Audit

After enabling BF16 MoE reduce, the remaining fp32 collective observed in `historical_4096_128_bs4` was:

| Label | Op | Dtype | Shape | Count | GiB | Decision |
|---|---|---|---|---:|---:|---|
| `dsv4.lm_head_all_gather` | all-gather | fp32 | `[4,16160] -> [32,16160]` | 16 | 0.031 | Defer. This is real vLLM parity work, but it is tiny relative to MoE and should be a separate small opt-in. |

No optional logits BF16 all-gather experiment was implemented in this target, to keep the MoE dtype change isolated.

## Short Profile

### Owner Timing

Command source: `raw/owner_timing_historical_4096_128_bs4/`, `historical_4096_128_bs4`, owner timing enabled. This run is instrumentation-heavy and should not be used as macro performance. The tool did not reset rank payload counters between the two variants in the same torchrun, so the second row is cumulative. The per-shape MoE row still confirms the extra BF16 case was recorded.

| Row | MoE owner count | MoE sum-rank ms | `[16384,4096]` count | `[16384,4096]` sum-rank ms |
|---|---:|---:|---:|---:|
| fp32 reduce aggregate | 3784 | 4681.190 | 344 | 816.956 |
| cumulative after BF16 variant | 4128 | 5117.372 | 688 | 1252.970 |
| approximate BF16 case delta | 344 | 436.182 | 344 | 436.014 |

### NCCL Kernel Family

Nsight rank0 profiles:

- `raw/nsys/fp32_reduce_historical_4096_128_bs4_rank0.sqlite`
- `raw/nsys/bf16_reduce_historical_4096_128_bs4_rank0.sqlite`

Both runs used `historical_4096_128_bs4`, repeats 1, warmup repeats 0, page size 256, graph buckets 1/2/4/8/16.

| Variant | NCCL AllReduce kernel families | NCCL AllGather |
|---|---|---|
| fp32 reduce | 258 `ncclDevKernel_AllReduce_Sum_f32_RING_LL`; 264 `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 6 `ncclDevKernel_AllGather_RING_LL` |
| BF16 reduce opt-in | 522 `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`; 0 f32 AllReduce kernels | 6 `ncclDevKernel_AllGather_RING_LL` |

This confirms the hot MoE AllReduce family changes from f32 to bf16. Baseline still had BF16 AllReduce kernels because attention and embedding were already BF16.

## Macro A/B

Source: `raw/macro_ab_required_r01/summary.json`, repeats 1, warmup repeats 0. This is a full required-scenario pass, not a repeat-stability promotion run.

| Scenario | fp32 elapsed | BF16 elapsed | Elapsed delta | fp32 decode tok/s | BF16 decode tok/s | Decode delta | fp32 E2E out tok/s | BF16 E2E out tok/s | E2E delta | BF16 graph | MoE GiB fp32 -> BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `historical_4096_128_bs4` | 9.781 | 8.090 | -17.3% | 182.5 | 182.3 | -0.1% | 52.3 | 63.3 | +20.9% | 127/0 | 86.0 -> 43.0 |
| `historical_4096_1024_bs4` | 29.904 | 29.982 | +0.3% | 182.8 | 181.8 | -0.5% | 137.0 | 136.6 | -0.3% | 1023/0 | 86.0 -> 43.0 |
| `serving_mixed_112req_wave16` | 16.157 | 15.898 | -1.6% | 275.2 | 284.4 | +3.3% | 173.3 | 176.1 | +1.6% | 441/0 | 91.7 -> 45.9 |
| `prefix_multi_112req_wave16` | 6.961 | 6.854 | -1.5% | 603.5 | 647.2 | +7.2% | 128.7 | 130.7 | +1.6% | 49/0 | 80.6 -> 40.3 |

Notes:

- All required macro scenarios passed.
- All BF16 opt-in rows stayed zero-eager under graph replay.
- The single-run macro shows no broad performance regression. `historical_4096_1024_bs4` is effectively neutral. The shorter and wave cases show small positive E2E movement, but this should be repeated before bundle promotion.
- Prefix hit-rate differed between single-run rows in the prefix scenario, so the prefix row should be interpreted as health/coverage plus no-regression evidence, not as a clean prefix-cache comparison.

## Promotion Decision

Decision: keep opt-in, do not reject, do not promote to the victory bundle yet.

Reasons to keep:

- vLLM source-derived dtype boundary is BF16 hidden-state reduce.
- Mini now supports the same BF16 reduce boundary on both runner and non-runner paths.
- Communication bytes halve exactly for the same MoE shape/count.
- Graph replay remains zero-eager.
- Nsight confirms f32 AllReduce kernels disappear from the short BF16 opt-in profile.
- Required TP8 macro scenarios pass without a clear regression.

Reasons not to bundle-promote immediately:

- Correctness smoke passed, but deterministic output drift was visible on 2 of 3 prompts and logits drift was not collected.
- Macro A/B is one full run plus the communication/profile short runs, not repeat-stable promotion evidence.
- Owner timing is instrumentation-heavy and cumulative in the two-variant run, so it is useful for ownership confirmation but not a clean timing promotion signal.

## Recommendation For TARGET 10.2

Proceed to TARGET 10.2 with MoE reduce dtype fixed to the BF16 opt-in variant for backend experiments. Compare communication stacks only after holding this dtype constant.

Suggested order:

1. Use `dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16` as the fixed mini path for initial backend A/B.
2. Keep `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1` explicit in run configs rather than silently promoting it.
3. Do not mix logits all-gather BF16 with backend routing in the first 10.2 runs. If needed, add a separate tiny logits BF16 all-gather opt-in with top-k/logit drift checks.
4. Re-run at least `historical_4096_1024_bs4` and one wave scenario when backend work changes graph capture, communicator routing, or NCCL stream behavior.
