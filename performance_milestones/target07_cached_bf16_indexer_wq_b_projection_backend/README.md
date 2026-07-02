# Target 07.60: Cached BF16 `indexer.wq_b` Projection Backend

Date: 2026-07-02

## Goal

Extend the promoted owner-scoped cached BF16 dequantized-weight projection
backend from `attn.q_wqb + attn.wo_b` to the remaining large same-contract
owner, `indexer.wq_b`, without changing the default path.

Inherited q_wqb+wo_b baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `49.6585` | `122.1863` | `127` | `0` |
| 4096/1024/batch4 | `98.6953` | `121.8705` | `1023` | `0` |

## Implementation

New opt-in toggle:

```text
MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE=1
```

New benchmark/text-smoke variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Runtime summary:

- `DSV4Indexer._wq_b_forward()` now uses
  `DSV4Linear.forward_fp8_cached_bf16_weight()` only when the new toggle is on.
- `DeepseekV4Model.prepare_for_cuda_graph_capture()` prepares all active
  `q_wqb`, `wo_b`, and `indexer.wq_b` BF16 weight caches before KV-cache
  allocation and before CUDA graph capture.
- Only C4/indexer layers are cached: layers
  `[2, 4, 6, ..., 42]`, 21 layers total.
- The observed local `indexer.wq_b` cached shape is `[8192, 1024]`.
- Forward uses `allow_build=False`; a missing or stale cache raises instead of
  rebuilding during decode, graph capture, or graph replay.
- `indexer.weights_proj`, `indexer.compressor`, FP8 indexer cache/logits,
  `wo_a`, `wq_a/wkv`, shared experts, MoE/Marlin, and `lm_head` were not routed
  through this toggle.

## vLLM Source Comparison

Reviewed:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`:
  `Fp8LinearMethod.apply` uses BF16 dequant plus `F.linear` for the
  batch-invariant non-block FP8 fallback.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`:
  `ReplicatedLinear`, `ColumnParallelLinear`, and `RowParallelLinear` call their
  quant method at the linear boundary.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`:
  `DeepseekV4Indexer.wq_b` is a replicated FP8 linear, followed by reshape,
  compressor, `weights_proj`, and fused indexer q/rope quant.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`:
  only C4 layers instantiate `DeepseekV4Indexer`.

Mini matches the source-level BF16 dequantized-weight linear boundary for
`indexer.wq_b`, but caches the dequantized BF16 weight once because serving
decode weights are static. This target does not claim vLLM runtime parity.

## Validation

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/scripts/microbench_indexer_wqb_cached_bf16.py \
  performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/scripts/summarize_memory_ledger.py \
  performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/scripts/summarize_projection_owner_nsys.py
```

Passed.

Focused microbench artifacts:

- `raw/indexer_wqb_cached_bf16_microbench.json`
- `summaries/indexer_wqb_cached_bf16_microbench.md`

| M | Current FP8 wrapper ms | Current intrinsic ms | Cached BF16 `F.linear` ms | Cached total projection ms | Total reduction | Max abs err | Max rel err | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.1799` | `0.1683` | `0.0173` | `0.0772` | `57.09%` | `0.000000` | `0.000000` | yes |
| 4 | `0.2570` | `0.1683` | `0.0191` | `0.0776` | `69.80%` | `0.003906` | `0.003953` | yes |
| 8 | `0.1794` | `0.1686` | `0.0175` | `0.0776` | `56.74%` | `0.031250` | `0.007692` | yes |
| 16 | `0.1794` | `0.1678` | `0.0176` | `0.0770` | `57.10%` | `0.015625` | `0.006173` | yes |

Text smoke artifact:

- `raw/text_smoke_qwqb_wob_idxwqb.json`

| Check | Value |
| --- | ---: |
| status | `pass` |
| graph replay count | `9` |
| greedy sample replay count | `9` |
| eager decode count | `0` |
| captured batch sizes | `[4, 2, 1]` |

Smoke outputs were sane:

- `2 + 2 等于 4。`
- `The sky is blue on a clear day.`
- `杭州是风景如画的历史文化名城。`

## Macro Results

Artifacts:

- `raw/macro_qwqb_wob_idxwqb_4096x128_bs4_np128/summary.json`
- `raw/macro_qwqb_wob_idxwqb_4096x1024_bs4_np128/summary.json`

| Workload | Variant | Output tok/s | Decode tok/s | Gain vs q_wqb+wo_b output | Gain vs q_wqb+wo_b decode | Graph replay | Eager decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | q_wqb + wo_b baseline | `49.6585` | `122.1863` | n/a | n/a | `127` | `0` |
| 4096/128/batch4 | q_wqb + wo_b + indexer.wq_b | `51.2962` | `132.3013` | `+3.30%` | `+8.28%` | `127` | `0` |
| 4096/1024/batch4 | q_wqb + wo_b baseline | `98.6953` | `121.8705` | n/a | n/a | `1023` | `0` |
| 4096/1024/batch4 | q_wqb + wo_b + indexer.wq_b | `105.7645` | `132.5127` | `+7.16%` | `+8.73%` | `1023` | `0` |

Both macro gates passed.

## Memory Ledger

Artifacts:

- `summaries/qwqb_wob_idxwqb_memory_ledger.json`
- `summaries/qwqb_wob_idxwqb_memory_ledger.md`

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `indexer.wq_b` | 21 | `[8192, 1024]` | 352,321,536 | `0.3281` | `4633.71` | `18.10` |
| total | 107 | mixed | 1,073,741,824 | `1.0000` | `14121.79` | `55.16` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs q_wqb+wo_b baseline | `351,272,960` |
| peak reserved delta vs q_wqb+wo_b baseline | `352,321,536` |
| peak allocated delta vs exact baseline | `1,077,149,696` |
| peak reserved delta vs exact baseline | `1,214,251,008` |

The measured incremental indexer reserved delta exactly matches the expected
cache bytes/rank. The allocated delta is slightly lower because allocator state
differs across runs.

## Fresh Post-Sequence Profile

Artifacts:

- `raw/nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0.nsys-rep`
- `raw/nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0.sqlite`
- `summaries/nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`
- `summaries/nsys_target0760_qwqb_wob_idxwqb_post_sequence_buckets.md`

Nsight run was profiling-instrumented and should not be compared directly to
non-Nsight macro throughput.

Owner-attributed projection/GEMM:

| Owner | Intrinsic/local GEMM s | Activation quant s | Copy/layout s | Communication s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `attn.q_wqb` | `0.068277` | `0.011360` | `0.000000` | `0` | cached BF16 local projection |
| `attn.wo_b` | `0.059062` | `0.011435` | `0.000000` | `0.159649` | cached BF16 local projection plus row-parallel all-reduce |
| `indexer.wq_b` | `0.050961` | `0.005291` | `0.000000` | `0` | cached BF16 local projection |
| `attn.wo_a` | `0.053534` | `0` | `0.290148` | `0` | dominant owner-local graph/layout work |
| `shared_experts.gate_up_proj` | `0.045765` | `0.012901` | `0.169527` | `0` | shared-expert projection/layout |
| `shared_experts.down_proj` | `0.029168` | `0.010882` | `0.129588` | `0` | shared-expert projection/layout |
| `mlp.routed_experts` | `0` | `0` | `0.076232` | `0` | MoE/Marlin bucket `0.281556s` inside owner detail |

The original three `_quantized_linear_fp8_kernel` owners are now handled:

- `indexer.wq_b` intrinsic dropped from `0.364997s` to `0.050961s`;
- `indexer.wq_b` copy/layout dropped from `0.012928s` to `0`;
- remaining decode projection/GEMM intrinsic bucket is `0.805080s`, with
  `0.469287s` owner-attributed and `0.335793s` unattributed.

Fresh decode-envelope bucket summary:

| Bucket | Duration s | Share of decode envelope |
| --- | ---: | ---: |
| graph/runtime/copy/cat/index | `1.141006` | `25.88%` |
| projection/GEMM | `0.805080` | `18.26%` |
| elementwise graph nodes | `0.639409` | `14.50%` |
| NCCL communication | `0.346671` | `7.86%` |
| MoE/Marlin | `0.316854` | `7.19%` |
| FP8 indexer cache/logits/topk | `0.131028` | `2.97%` |
| sparse attention decode | `0.118408` | `2.69%` |
| KV/compressor/cache store | `0.028110` | `0.64%` |

This profile says not to continue cached-weight expansion. The next meaningful
mini bottleneck is the graph/layout cluster, especially `attn.wo_a` and shared
expert projection/layout, but the proper next target should first reprofile
vLLM parity after the completed three-owner projection sequence.

## Decision

Status: pass for opt-in q_wqb + wo_b + indexer.wq_b cached BF16.

Decision: promote three-owner cached BF16 and run vLLM parity reprofile.

Rationale: focused `indexer.wq_b` projection time dropped by at least `56.74%`,
4096/128 improved by `+3.30%`, 4096/1024 improved by `+7.16%`, text smoke
passed, CUDA graph replay remained intact with eager decode `0`, and the
incremental memory cost matched the expected `0.3281 GiB/rank`.

Next target recommendation: run a vLLM parity reprofile with the new mini
three-owner cached BF16 stack as the comparison point. Use that reprofile to
choose between broad graph/layout work and a narrow `wo_a`/shared-expert
graph-layout target. Do not continue cached-weight projection work unless a new
profile identifies a fresh owner with comparable concentrated cost.
