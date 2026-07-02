# Target 07.58: Cached BF16 Projection Backend

Date: 2026-07-02

## Goal

Implement and evaluate an opt-in cached BF16 dequantized-weight projection
backend for DeepSeek V4 Flash on A100/sm80, starting with `attn.q_wqb` only.
The default path remains unchanged.

## Baseline

Promoted 07.54/07.55 baseline:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/batch4 | `87.0831` | `104.3427` | `1023` | `0` |

07.57 attributed the shared `_quantized_linear_fp8_kernel` contract:

| Owner | Decode-envelope intrinsic s | Real-weight wrapper ms | Cached BF16 `F.linear` ms |
| --- | ---: | ---: | ---: |
| `attn.q_wqb` | `0.404178` | about `0.412` | about `0.053` |
| `attn.wo_b` | `0.403710` | about `0.660` | about `0.052` |
| `indexer.wq_b` | `0.364756` | about `0.168` | about `0.019` |

## Implementation

New opt-in toggle:

```text
MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE=1
```

New benchmark/text-smoke variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Runtime summary:

- `DeepseekV4ForCausalLM.prepare_for_cuda_graph_capture()` prepares the cache
  after weights are loaded and before KV cache allocation / CUDA graph capture.
- Only `DSV4Attention.prepare_q_wqb_bf16_weight_cache()` is connected.
- `attn.q_wqb` forward checks the cache metadata and uses
  `quantize_fp8_activation_ref(q_lora)` plus `F.linear(x_quant, cached_weight)`.
- If the cache is missing or stale in forward, the path raises a clear error
  instead of rebuilding.
- `attn.wo_b`, `indexer.wq_b`, `wo_a`, shared experts, and `lm_head` are not
  connected to this BF16 weight cache.

Cache metadata guards:

- weight pointer, version, device, dtype, shape, stride, storage offset;
- scale pointer, version, device, dtype, shape, stride, storage offset;
- output dtype.

## vLLM Source Comparison

Reviewed source:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`
  lines 433-476: `Fp8LinearMethod.apply`; in batch-invariant mode, non-block
  FP8 weights are converted to BF16, scaled, and passed to `F.linear`.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py` lines
  394, 587, and 1555: linear modules call `quant_method.apply`.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
  lines 877-903: `wq_b` is `ColumnParallelLinear`, `wo_b` is
  `RowParallelLinear`.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
  lines 643-647 and 681-714: `wq_b` is lifted out before attention and `wo_b`
  remains the output projection boundary.

Mini's PoC matches vLLM's BF16 dequant fallback boundary for `q_wqb`, but caches
the dequantized BF16 weight once because serving decode weights are static.
This README does not claim vLLM runtime parity unless a vLLM run is added.

## Validation Commands

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target07_cached_bf16_projection_backend/scripts/microbench_qwqb_cached_bf16.py \
  performance_milestones/target07_cached_bf16_projection_backend/scripts/summarize_memory_ledger.py
```

```bash
python performance_milestones/target07_cached_bf16_projection_backend/scripts/microbench_qwqb_cached_bf16.py \
  --warmup 10 \
  --iters 50
```

```bash
performance_milestones/target07_cached_bf16_projection_backend/scripts/run_text_smoke_qwqb.sh
```

```bash
performance_milestones/target07_cached_bf16_projection_backend/scripts/run_macro_qwqb_4096x128_bs4_np128.sh
```

```bash
performance_milestones/target07_cached_bf16_projection_backend/scripts/run_macro_qwqb_4096x1024_bs4_np128.sh
```

## Results

Status: pass.

Focused microbench artifact:

- `raw/qwqb_cached_bf16_microbench.json`
- `summaries/qwqb_cached_bf16_microbench.md`

Real checkpoint, TP8 rank0 local `attn.q_wqb.layer0` shape is `[4096,
1024]`. This matters for both timing and memory: 07.57's early real-weight
microbench used a raw `[32768, 1024]` view, while the serving module is sharded
by tensor parallelism.

| M | current FP8 wrapper ms | current intrinsic ms | fallback dequant ms | cached BF16 `F.linear` ms | cached total ms | Total reduction vs wrapper | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.1300` | `0.0914` | `0.1982` | `0.0171` | `0.0776` | `40.3%` | yes |
| 4 | `0.1288` | `0.0914` | `0.1997` | `0.0176` | `0.0787` | `38.9%` | yes |
| 8 | `0.1295` | `0.0916` | `0.2031` | `0.0176` | `0.0792` | `38.8%` | yes |
| 16 | `0.1308` | `0.0917` | `0.1996` | `0.0179` | `0.0793` | `39.4%` | yes |

Correctness was checked against the current FP8 Triton wrapper path. Max abs
error stayed within `0.001953`, max relative error stayed within `0.007634`,
and all rows passed the `allclose` gate.

Text smoke artifact:

- `raw/text_smoke_qwqb.json`

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

Macro artifacts:

- `raw/macro_qwqb_4096x128_bs4_np128/summary.json`
- `raw/macro_qwqb_4096x1024_bs4_np128/summary.json`

| Workload | Variant | Output tok/s | Decode tok/s | Gain vs baseline output | Gain vs baseline decode | Graph replay | Eager decode | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096/128/batch4 | baseline | `43.0685` | `104.2028` | n/a | n/a | `127` | `0` | pass |
| 4096/128/batch4 | q_wqb cached BF16 | `47.9464` | `112.3221` | `+11.33%` | `+7.79%` | `127` | `0` | pass |
| 4096/1024/batch4 | baseline | `87.0831` | `104.3427` | n/a | n/a | `1023` | `0` | pass |
| 4096/1024/batch4 | q_wqb cached BF16 | `92.5170` | `112.3307` | `+6.24%` | `+7.66%` | `1023` | `0` | pass |

The q_wqb-only path passes both success gates: focused projection time drops by
about `39%` including activation quantization, and 4096/128 output tok/s rises
by `11.33%`.

## Memory Ledger

Generated with:

```bash
python performance_milestones/target07_cached_bf16_projection_backend/scripts/summarize_memory_ledger.py \
  --qwqb-report performance_milestones/target07_cached_bf16_projection_backend/raw/macro_qwqb_4096x128_bs4_np128/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json \
  --baseline-report performance_milestones/target07_graph_layout_replay_deforestation/raw/macro_4096x128_bs4_np128_actqtriton/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json
```

Artifacts:

- `summaries/qwqb_memory_ledger.json`
- `summaries/qwqb_memory_ledger.md`

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | 0 | n/a | 0 | 0 | 0 | 0 |
| `indexer.wq_b` | 0 | n/a | 0 | 0 | 0 | 0 |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs baseline | `363,565,056` |
| peak reserved delta vs baseline | `417,333,248` |

The measured peak allocated delta is close to the cached weight bytes/rank, as
expected. Cache construction happens during explicit model preparation before
CUDA graph capture; replay only reads the cached BF16 tensors.

## Decision

Promote q_wqb cached BF16 as an opt-in TARGET 07.58 backend.

The default path remains unchanged because the implementation is guarded by
`MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE=1` and only `attn.q_wqb` calls the
new cached-weight forward. Missing or stale cache state raises in forward rather
than rebuilding inside graph replay.

Next step: evaluate the same owner-scoped cached BF16 backend on `attn.wo_b` in
a separate target. The q_wqb evidence is strong enough to continue owner by
owner; `indexer.wq_b` should wait until after `wo_b` or until a fresh profile
shows it is the better target. If `wo_b` does not clear the macro/memory gate,
pivot back to retuning `_quantized_linear_fp8_kernel` scale/load/dequant
internals.
