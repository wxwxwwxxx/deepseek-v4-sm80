# TARGET 07.36: DSV4 sm80 vLLM FusedMoE Runner Adaptation

Date: 2026-07-01

Status: complete. The mini-owned vLLM-shaped exact runner cut is implemented,
validated, and measured. It does not meet the 5 percent macro or 10 percent
routed-MoE subgraph thresholds, so stop MoE wrapper work here.

## Artifacts

| Path | Contents |
| --- | --- |
| `summaries/runner_summary.json` | Small machine-readable conclusion and key metrics. |
| `summaries/nsys_runner_4096x128_bs4_summary.json` | Nsight sqlite summary for the runner 4096/128 profile. |
| `raw/moe_runner_microbench_smoke.json` | Runner-stage MoE microbench smoke. |
| `raw/moe_runner_microbench_real_shapes.json` | Runner-stage MoE microbench with DSV4-like shapes. |
| `raw/text_smoke_runner_tp8.json` | TP8 page-size-256 text smoke aggregate. |
| `raw/text_smoke_runner_tp8.<variant>.json` | Per-variant text smoke output. |
| `raw/mini_4096_128_bs4_runner_perf/` | 4096/128/batch4 profile-equivalent perf matrix. |
| `raw/mini_4096_1024_bs4_runner_perf/` | 4096/1024/batch4 official macro perf matrix. |
| `raw/mini_4096_128_bs4_runner_nsys_perf/` | Perf matrix run wrapped by Nsight Systems. |
| `raw/nsys_runner_4096x128_bs4.nsys-rep` | Short 4096/128 runner Nsight report. |
| `raw/nsys_runner_4096x128_bs4.sqlite` | Exported sqlite from the Nsight report. |

## Component Map

| vLLM component | mini component | Decision | Notes |
| --- | --- | --- | --- |
| `DeepseekV4MoE -> FusedMoE` standard sm80 path | `DSV4MoE` | adapt | Added opt-in mini-owned runner boundary. |
| `MoERunner.forward/_forward_impl` owns route, shared experts, experts, combine, reduce | `DSV4FusedMoERunner` | adapt | New runner owns route, prepare, experts, finalize, shared scheduling, and late reduce decision. |
| `FusedMoERouter.select_experts` / biased top-k | `DSV4MoEGate` | adapt by reuse | Keeps fp32 router math, correction bias, and hash routing behavior. |
| `MoEPrepareAndFinalizeNoDPEPModular.prepare` | `DSV4FusedMoERunner.prepare` | adapt | Builds `DSV4MoEExecutionPlan` and route metadata; no activation quantization. |
| `FusedMoEExpertsModular.apply` | `DSV4FusedRoutedExperts.forward` | adapt by wrapping | Calls existing exact grouped FP4 W13/SwiGLU/W2 backend. |
| `MoEPrepareAndFinalizeNoDPEPModular.finalize` | `DSV4FusedMoERunner.finalize_routed` | adapt with current-backend caveat | Current mini backend already applies route weights and sums top-k to `[M, H]`; finalize is an explicit boundary plus fp32 cast. |
| `SharedExperts` stream scheduler | `DSV4FusedMoERunner.apply_shared` | defer overlap | Scheduling decision moved into runner, but first cut remains serial for exactness and measurement. |
| `_maybe_reduce_final_output` | `DSV4FusedMoERunner.maybe_reduce_final` | adapt | Keeps one late TP all-reduce after routed + shared local sum. |
| `DeepseekV4MegaMoEExperts` | none | reject | vLLM sm80 path should not depend on it; not ported. |
| vLLM MXFP4 / FP8 cache precision lane | none | defer | Not introduced into exact default; remains TARGET 07.4 material. |
| vLLM runtime dependency | none | reject | No vLLM import or runtime dependency added. No source code copied; vLLM was used as an Apache-2.0 design reference. |

## Implementation

New toggle:

- `MINISGL_DSV4_SM80_MOE_VLLM_RUNNER=1`

The toggle is explicit and opt-in. Setting it also enables the current exact
MoE V2 route/metadata whitelist so the runner can wrap the existing grouped FP4
backend without requiring a separate vLLM dependency.

New benchmark variant:

- `v1_moe_vllm_runner_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`

Fallbacks remain available:

- no runner toggle: existing V1/V2 `DSV4MoE.forward` path;
- no grouped Triton availability: existing routed expert fallback path;
- no shared experts: runner routed-only path.

## Correctness

| Check | Result |
| --- | --- |
| New targeted runner/env pytest | 8 passed. |
| `tests/kernel/test_deepseek_v4_wrappers.py` | 30 passed, 4 external warnings. |
| `tests/models/test_deepseek_v4_forward_fallback.py` | 14 passed, 4 external warnings. |
| `tests/benchmark/test_deepseek_v4_perf_matrix.py` + `test_deepseek_v4_text_smoke.py` | 18 passed. |
| TP8 page-size-256 text smoke | pass. |

TP8 text smoke details:

- variant: `v1_moe_vllm_runner_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`
- graph replay: 7/7 decode replays
- eager decode fallback: 0
- sample outputs passed sanity checks for Chinese arithmetic, English sky-color,
  and Hangzhou prompts.

One wide mixed pytest command over all four files hit the existing global TP
state ordering issue in `test_hc_head_maintains_bf16_linear_weight_cache`.
Running the same files independently passed.

## Microbench

Artifact: `raw/moe_runner_microbench_real_shapes.json`.

| Case | V1 grouped full ms | V2 grouped full ms | runner prepare ms | runner experts ms | runner finalize ms | runner shared ms | runner total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_real | 2.0111 | 2.0180 | 0.1632 | 1.9562 | 0.0167 | 0.0 | 2.0258 |
| prefill_real | 97.9671 | 97.9203 | 0.1690 | 97.8227 | 0.0945 | 0.0 | 98.0449 |

Read: the runner boundary is neutral-to-slightly-negative for routed MoE
subgraph time. It does not clear the 10 percent routed-MoE subgraph gate.

## Macro Results

Baseline is the frozen 07.35 exact V2 result.

| Workload | Variant | Status | Decode tok/s | E2E output tok/s | E2E vs 07.35 | Graph replay | Eager fallback |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | 07.35 exact V2 | pass | 19.8442 | 10.7751 | reference | 127 | 0 |
| 4096/128/batch4 | vLLM-shaped runner | pass | 19.8299 | 10.7579 | -0.16% | 127 | 0 |
| 4096/1024/batch4 | 07.35 exact V2 | pass | 19.9037 | 17.8009 | reference | 1023 | 0 |
| 4096/1024/batch4 | vLLM-shaped runner | pass | 19.9313 | 17.8289 | +0.16% | 1023 | 0 |

The 4096/1024 runner result is far below both reference lines:

| Reference | 4096/1024 output tok/s |
| --- | ---: |
| old vLLM serving victory line | 114.07 |
| fair vLLM offline reference from TARGET 07.25 | 201.8738 |
| runner exact result | 17.8289 |

## Profile Read

Short runner Nsight profile:

- `raw/nsys_runner_4096x128_bs4.nsys-rep`
- `raw/nsys_runner_4096x128_bs4.sqlite`
- summary: `summaries/nsys_runner_4096x128_bs4_summary.json`

Top summed GPU kernel times from the exported sqlite:

| Category / kernel | Summed GPU time |
| --- | ---: |
| `_grouped_fp4_w13_kernel` | 46.781s |
| `_grouped_fp4_linear_kernel` | 31.700s |
| NCCL bf16 all-reduce | 14.946s |
| sparse attention kernels | 9.402s |
| `_indexer_bf16_logits_kernel` | 4.608s |
| NCCL f32 all-reduce | 1.946s |
| `_moe_route_sum_kernel` | 0.179s |

W13 + W2 remain the dominant measured category. Attention/cache/indexer is
still important, but it did not overtake MoE in this fresh profile.

## vLLM Runner Differences

| Area | vLLM standard runner | mini runner first cut |
| --- | --- | --- |
| Runtime dependency | vLLM internal modules | no vLLM runtime dependency |
| Precision | vLLM deployment may use MXFP4/FP8 lanes | exact bf16-direct activation policy preserved |
| Prepare | quantize/dispatch abstraction | route plan + route weights only |
| Experts | modular backend selected by quant method | existing mini grouped FP4 W13/SwiGLU/W2 backend |
| Finalize | applies top-k weights/reduce when experts do not | current backend already applies weights and route sum; finalize boundary is explicit |
| Shared experts | can run serial, MK-internal overlap, or aux-stream overlap | serial first cut, scheduling inside runner for future use |
| Reduce | late reduce unless fused output already reduced | late TP all-reduce after routed + shared |
| DP/EP/SP | broad support | no DP/EP adaptation in this target |
| MegaMoE | not used for sm80 standard path | rejected |

## Final Decision

Stop MoE wrapper work for TARGET 07.36.

Reasons:

- 4096/1024 macro gain is about +0.16 percent, below the 5 percent gate.
- DSV4-like routed-MoE subgraph gain is negative, below the 10 percent gate.
- Fresh Nsight still shows grouped FP4 W13/W2 as dominant, so the wrapper
  boundary is not the limiting MoE problem.
- Attention/cache/indexer did not clearly overtake MoE in the fresh profile.

Next target: open an exact expert-kernel backend target for W13/W2, or move to
TARGET 07.4 precision lane if the project accepts opt-in MXFP4/INT8/FP8
semantics for parity experiments. Do not continue polishing the runner wrapper
unless a later exact backend exposes a real per-route finalize/scheduling
benefit.
