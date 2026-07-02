# TARGET 07.62 wo_a Attention Boundary Parity

## Current Best

- Variant: `target0762_woabf16bmmcache`
- Flag under test: `MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE=1`
- Base stack: TARGET 07.60 three-owner cached BF16 stack (`attn.q_wqb`, `attn.wo_b`, `indexer.wq_b`) plus the opt-in `attn.wo_a` BF16 grouped BMM cache.
- Default behavior: unchanged. Without the flag, mini still uses the existing `wo_a_grouped_projection_fallback` path.
- Primary result: `4096/1024/batch4` improved from the 07.60 baseline `105.7645` output tok/s to `116.2553` output tok/s (`+9.92%`), with CUDA graph replay preserved and eager decode `0`.

## Implementation Summary

- Added `MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE` as a known experimental SM80 toggle in `python/minisgl/kernel/deepseek_v4.py`.
- Added an opt-in cache builder in `python/minisgl/models/deepseek_v4.py`:
  - dequantizes the FP8 `wo_a` shard once to BF16,
  - stores it as `[num_local_groups, d_per_group, o_lora_rank]`,
  - reports shape, dtype, bytes, owner, and source shape in `model_prepare_report_rank0`.
- Hooked the builder into `DeepseekV4Model.prepare_for_cuda_graph_capture()`, so the cache is ready before CUDA graph capture and replay.
- Updated `DSV4Attention.forward()` so the new path calls `torch.bmm` on `[groups, tokens, d_per_group] x [groups, d_per_group, o_lora_rank]`.
- Replay-time rebuilds are intentionally disabled. If the cache is missing or stale during forward with the flag enabled, the code raises and asks the caller to run `prepare_for_cuda_graph_capture()` before capture/replay.
- Added benchmark/smoke variants and focused tests for the new flag.

## vLLM Source Parity

- vLLM comparison files:
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- Parity target used here:
  - vLLM marks `wo_a` as BMM with per-group batching (`is_bmm`, `bmm_batch_size=n_local_groups`).
  - vLLM prepares a BF16 BMM weight cache through `_ensure_wo_a_bmm_weight()`.
  - vLLM applies that cache through `_apply_wo_a_bmm()`.
- Explicitly not included in this target:
  - vLLM fused inverse-RoPE plus FP8 `wo_a` einsum precision lane,
  - shared expert layout changes,
  - communication/overlap work,
  - broad CUDA graph cleanup.

## Memory Ledger

Source: `summaries/wo_a_memory_ledger.md`, generated from the `4096/128/batch4` macro report with page size `256`.

| Cached owner | Enabled | Layers | Cache shape/rank | Source shape/rank | Extra bytes/rank | Extra GiB/rank | KV tokens/rank | KV pages/rank |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `True` | 43 | `[4096, 1024]` | `mixed` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | `True` | 43 | `[4096, 1024]` | `mixed` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `indexer.wq_b` | `True` | 21 | `[8192, 1024]` | `mixed` | 352,321,536 | `0.3281` | `4633.71` | `18.10` |
| `attn.wo_a` | `True` | 43 | `[1, 4096, 1024]` | `[1024, 4096]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `total_cached_bf16_projection` | `n/a` | mixed | `mixed` | `mixed` | 1,434,451,968 | `1.3359` | `18865.83` | `73.69` |

| Metric | Value |
| --- | ---: |
| wo_a incremental bytes/rank | `360,710,144` |
| wo_a incremental GiB/rank | `0.3359` |
| wo_a equivalent KV tokens/rank | `4744.04` |
| wo_a equivalent KV pages/rank | `18.53` |
| KV cache bytes/rank max | `2,491,495,680` |
| Peak allocated delta vs 07.60 baseline | `360,710,144` |
| Peak reserved delta vs 07.60 baseline | `381,681,664` |

## Focused Microbench

Command:

```bash
python performance_milestones/target07_wo_a_attention_boundary_parity/scripts/microbench_wo_a_bf16_bmm_cache.py --m-values 1,4,8,16 --warmup 10 --iters 50 --tp-size 8 --tp-rank 0
```

Artifact: `summaries/wo_a_bf16_bmm_cache_microbench.md`

| M | Groups | K/group | Rank | fallback total ms | cache build ms | cached BMM total ms | BMM only ms | Speedup | Improvement | ok 5e-2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 4096 | 1024 | `0.1679` | `0.1606` | `0.0285` | `0.0167` | `5.89x` | `83.02%` | `True` |
| 4 | 1 | 4096 | 1024 | `0.1755` | `0.1608` | `0.0292` | `0.0177` | `6.01x` | `83.36%` | `True` |
| 8 | 1 | 4096 | 1024 | `0.1757` | `0.1610` | `0.0287` | `0.0173` | `6.11x` | `83.64%` | `True` |
| 16 | 1 | 4096 | 1024 | `0.1768` | `0.1610` | `0.0287` | `0.0171` | `6.15x` | `83.75%` | `True` |

The microbench gate passed: total projection time improved by more than `30%` for all required M values.

## Text Smoke

Command:

```bash
bash performance_milestones/target07_wo_a_attention_boundary_parity/scripts/run_text_smoke_wo_a.sh
```

Artifacts:

- `raw/text_smoke_wo_a_bf16_bmm_cache.json`
- `raw/text_smoke_wo_a_bf16_bmm_cache.target0762_woabf16bmmcache.json`

Result:

| Variant | Status | Graph replay | Eager decode | wo_a cached layers | wo_a cache bytes/rank |
| --- | --- | ---: | ---: | ---: | ---: |
| `target0762_woabf16bmmcache` | `pass` | 9 | 0 | 43 | 360,710,144 |

Graph replay stayed on and eager decode stayed at `0`.

## Macro

Commands:

```bash
bash performance_milestones/target07_wo_a_attention_boundary_parity/scripts/run_macro_wo_a_4096x128_bs4_np128.sh
bash performance_milestones/target07_wo_a_attention_boundary_parity/scripts/run_macro_wo_a_4096x1024_bs4_np128.sh
```

Artifacts:

- `raw/macro_wo_a_bf16_bmm_cache_4096x128_bs4_np128/summary.json`
- `raw/macro_wo_a_bf16_bmm_cache_4096x1024_bs4_np128/summary.json`

| Scenario | 07.60 baseline output tok/s | New output tok/s | Gain | Decode tok/s | Prefill tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4096/128/batch4` | `51.2962` | `53.5877` | `+4.47%` | `149.4224` | `3194.6201` | 127 | 0 |
| `4096/1024/batch4` | `105.7645` | `116.2553` | `+9.92%` | `148.8915` | `3226.6676` | 1023 | 0 |

The primary macro gate passed through `4096/1024/batch4` output throughput (`+9.92%`, threshold `+5%`).

## Owner Profile

Command:

```bash
bash performance_milestones/target07_wo_a_attention_boundary_parity/scripts/nsys_projection_owner_wo_a_4096x128_bs4.sh
```

Artifacts:

- `summaries/nsys_target0762_wo_a_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`
- `summaries/nsys_target0762_wo_a_projection_owner_4096x128_bs4_np128_rank0_projection_owner.json`
- `raw/nsys_target0762_wo_a_projection_owner_4096x128_bs4_np128_rank0.sqlite`

Comparison source for 07.60:

- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/summaries/nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`

| Owner/profile metric | 07.60 baseline | 07.62 wo_a BF16 BMM cache | Change |
| --- | ---: | ---: | ---: |
| `attn.wo_a` replay total | `0.481377s` | `0.068948s` | `-85.68%` |
| `attn.wo_a` intrinsic GEMM | `0.053534s` | `0.068948s` | `+28.79%` |
| `attn.wo_a` copy/layout | `0.290148s` | `0.000000s` | removed |
| `attn.wo_a` elementwise/scale | `0.137695s` | `0.000000s` | removed |
| `attn.wo_a` replay graph nodes | 344 | 86 | `-75.00%` |
| Decode envelope | `4.408601s` | `3.952792s` | `-10.34%` |

Interpretation: the new BF16 BMM path trades the old dequant/view/einsum wrapper work for a pure BF16 BMM owner. The GEMM bucket is slightly larger, but the previous copy/layout and scale work disappears, so total `attn.wo_a` owner drops far beyond the `15%` owner gate.

## Decision

Promote this as an opt-in candidate for the TARGET 07 stack. It passes the required stop/success gates:

- text smoke passed,
- CUDA graph replay stayed enabled,
- eager decode stayed `0`,
- focused `wo_a` microbench improved total projection time by `83%+`,
- fresh owner profile shows `attn.wo_a` owner down `85.68%`,
- `4096/1024/batch4` output throughput improved `+9.92%` over the 07.60 baseline.

The path should remain opt-in for now because it adds `0.3359 GiB/rank` of cached BF16 projection memory.

## Do Not Continue In This Target

Stop here for 07.62. Further wins likely require a separate target for one of:

- fused inverse-RoPE plus FP8 `wo_a` einsum precision lane,
- shared expert layout/cache work,
- communication overlap,
- broad CUDA graph cleanup,
- reducing the residual BMM transpose/contiguous boundary if a future owner profile shows it matters.
