# Target 07.59: Cached BF16 `wo_b` Projection Backend

Date: 2026-07-02

## Goal

Extend the TARGET 07.58 owner-scoped cached BF16 dequantized-weight backend from
`attn.q_wqb` to row-parallel `attn.wo_b`, while keeping the default path
unchanged and preserving CUDA graph replay.

The new path keeps the 07.58 `attn.q_wqb` cache enabled and adds only:

```text
MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE=1
```

It does not add cached BF16 weights for `indexer.wq_b`, `wo_a`, `wq_a/wkv`,
shared experts, or `lm_head`.

## Baseline

Inherited q_wqb-cached 07.58 baseline:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `47.9464` | `112.3221` | `127` | `0` |
| 4096/1024/batch4 | `92.5170` | `112.3307` | `1023` | `0` |

## Implementation

New variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Runtime summary:

- `DeepseekV4ForCausalLM.prepare_for_cuda_graph_capture()` now prepares both
  `attn.q_wqb` and `attn.wo_b` caches before KV cache allocation and before
  CUDA graph capture.
- `DSV4Linear.forward_fp8_cached_bf16_weight()` still defaults to local-only
  output for `q_wqb`; `wo_b` passes `reduce=True`.
- `wo_b` executes `quantize_fp8_activation_ref(o)`, local
  `F.linear(x_quant, cached_weight)`, then the row-parallel all-reduce with
  label `dsv4.attn.wo_b.row_parallel_projection_all_reduce`.
- Forward uses `allow_build=False`; missing or stale cached BF16 weight raises
  instead of rebuilding inside decode, graph capture, or graph replay.

Default behavior remains unchanged because both cached BF16 paths are guarded
by opt-in environment toggles.

## vLLM Source Comparison

Reviewed:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`
  `Fp8LinearMethod.apply`: batch-invariant fallback dequantizes FP8 weights to
  BF16 and calls `F.linear` when the faster FP8 path is not used.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
  `RowParallelLinear.forward`: local `quant_method.apply` result is followed by
  tensor-parallel all-reduce when `reduce_results=True`.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`:
  `wq_b` is `ColumnParallelLinear`; `wo_b` is `RowParallelLinear`.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`:
  attention output paths call `self.wo_b(...)` after `wo_a`/FP8-einsum.

Mini matches the high-level BF16 dequantized-weight boundary for `wo_b`, but
caches the BF16 weight once for repeated decode use. No vLLM runtime profile was
run in this target.

## Validation

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/microbench_wob_cached_bf16.py \
  performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/summarize_memory_ledger.py
```

Passed.

Focused microbench:

```bash
python performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/microbench_wob_cached_bf16.py \
  --warmup 10 \
  --iters 50
```

Text smoke:

```bash
performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/run_text_smoke_qwqb_wob.sh
```

Macro:

```bash
performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/run_macro_qwqb_wob_4096x128_bs4_np128.sh
performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/run_macro_qwqb_wob_4096x1024_bs4_np128.sh
```

Profile:

```bash
performance_milestones/target07_cached_bf16_wo_b_projection_backend/scripts/nsys_projection_owner_qwqb_wob_4096x128_bs4.sh
```

## Focused Microbench

Artifacts:

- `raw/wob_cached_bf16_microbench.json`
- `summaries/wob_cached_bf16_microbench.md`

Scope: local row-parallel projection only. All-reduce is not included in this
microbench.

| M | Current FP8 wrapper ms | Current intrinsic ms | Cached BF16 local `F.linear` ms | Cached total local projection ms | Local total reduction | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.1290` | `0.0913` | `0.0168` | `0.0753` | `41.6%` | yes |
| 4 | `0.1272` | `0.0914` | `0.0179` | `0.0781` | `38.6%` | yes |
| 8 | `0.1268` | `0.0915` | `0.0177` | `0.0791` | `37.6%` | yes |
| 16 | `0.1238` | `0.0916` | `0.0174` | `0.0768` | `38.0%` | yes |

Max abs error stayed at or below `0.007812`; max relative error stayed at or
below `0.007752`.

## Text Smoke

Artifact:

- `raw/text_smoke_qwqb_wob.json`

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

- `raw/macro_qwqb_wob_4096x128_bs4_np128/summary.json`
- `raw/macro_qwqb_wob_4096x1024_bs4_np128/summary.json`

| Workload | Variant | Output tok/s | Decode tok/s | Gain vs q_wqb output | Gain vs q_wqb decode | Graph replay | Eager decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | q_wqb cached BF16 baseline | `47.9464` | `112.3221` | n/a | n/a | `127` | `0` |
| 4096/128/batch4 | q_wqb + wo_b cached BF16 | `49.6585` | `122.1863` | `+3.57%` | `+8.78%` | `127` | `0` |
| 4096/1024/batch4 | q_wqb cached BF16 baseline | `92.5170` | `112.3307` | n/a | n/a | `1023` | `0` |
| 4096/1024/batch4 | q_wqb + wo_b cached BF16 | `98.6953` | `121.8705` | `+6.68%` | `+8.49%` | `1023` | `0` |

The 4096/128 output gate did not reach `+5%`, but the focused local projection
gate passed and the decode-heavy 4096/1024 macro passed the `+3%` long-decode
gate.

## Profile Breakdown

Artifacts:

- `raw/nsys_target0759_qwqb_wob_projection_owner_4096x128_bs4_np128_rank0.nsys-rep`
- `raw/nsys_target0759_qwqb_wob_projection_owner_4096x128_bs4_np128_rank0.sqlite`
- `summaries/nsys_target0759_qwqb_wob_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`
- `summaries/nsys_target0759_qwqb_wob_projection_owner_4096x128_bs4_np128_rank0_projection_owner.json`

Nsight run was profiling-instrumented and should not be compared directly to
non-Nsight macro throughput.

| Owner | Intrinsic/local GEMM s | Activation quant s | Copy/layout s | Communication s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `attn.q_wqb` | `0.068302` | `0.011431` | `0.000000` | `0` | cached BF16 local projection |
| `attn.wo_b` | `0.059160` | `0.011435` | `0.000000` | `0.161865` | cached BF16 local projection plus row-parallel all-reduce |
| `indexer.wq_b` | `0.364997` | `0.005293` | `0.012928` | `0` | largest remaining same-contract FP8 projection owner |

Compared with TARGET 07.57, `attn.wo_b` local projection moved from the FP8
path:

```text
intrinsic 0.403710s + activation 0.011370s + copy/layout 0.017131s
```

to:

```text
BF16 GEMM 0.059160s + activation 0.011435s + copy/layout 0.000000s
```

The row-parallel all-reduce stayed in place and is now the largest part of
`wo_b` itself:

```text
wo_b local projection after cache: 0.070595s
wo_b row-parallel all-reduce:      0.161865s
```

Macro communication stats also recorded the explicit all-reduce label:

```text
dsv4.attn.wo_b.row_parallel_projection_all_reduce
count: 344
bytes: 46,170,898,432
```

The remaining `_quantized_linear_fp8_kernel` owner is `indexer.wq_b`
(`0.364997s`), which is larger than `wo_b` communication in this profile.

## Memory Ledger

Artifacts:

- `summaries/qwqb_wob_memory_ledger.json`
- `summaries/qwqb_wob_memory_ledger.md`

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| total | 86 | mixed | 721,420,288 | `0.6719` | `9488.08` | `37.06` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs q_wqb-only baseline | `362,311,680` |
| peak reserved delta vs q_wqb-only baseline | `444,596,224` |
| peak allocated delta vs exact baseline | `725,876,736` |
| peak reserved delta vs exact baseline | `861,929,472` |

Expected incremental `wo_b` cache was `360,710,144 bytes/rank`; measured
prepare bytes match exactly. Peak allocated delta is close to the expected cache
bytes; reserved delta is larger because of CUDA caching allocator behavior.

## Decision

Status: pass for opt-in q_wqb + wo_b cached BF16.

The local `wo_b` gate passes, text smoke passes, CUDA graph replay is preserved,
eager decode remains `0`, memory cost is understood, and 4096/1024 improves by
`+6.68%` over the q_wqb-cached baseline.

Decision: continue cached BF16 to indexer.wq_b.

Rationale: `wo_b` all-reduce now dominates the `wo_b` owner, but the largest
remaining same-contract projection owner is `indexer.wq_b` at `0.364997s`.
After `indexer.wq_b`, revisit `wo_b` communication/overlap if profile evidence
still shows row-parallel all-reduce as the next largest blocker.
