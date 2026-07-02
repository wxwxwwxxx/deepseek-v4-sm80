# TARGET 07.61: Post-Cached-BF16 vLLM Parity Reprofile

Date: 2026-07-02

Status: complete. This target is evidence-only. No large optimization PoC was
implemented.

## Current Best

Current best mini stack is the TARGET 07.60 three-owner cached BF16 variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

It keeps TP8, page size `256`, `--num-pages 128`, model
`/models/DeepSeek-V4-Flash`, and CUDA graph replay for decode. The three cached
BF16 owners are `attn.q_wqb`, `attn.wo_b`, and `indexer.wq_b`.

Mini source state:

| Field | Value |
| --- | --- |
| branch | `dsv4-sglang-based` |
| commit | `d92f9c46826f6063ccc6c5febed82bac3b558b80` |
| status before 07.61 edits | clean |
| status after 07.61 edits | new `performance_milestones/target07_post_cached_bf16_vllm_parity_reprofile/` only |

Required opt-in toggles represented by the variant:

| Toggle | Value |
| --- | --- |
| `MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE` | `1` |
| `MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE` | `1` |
| `MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE` | `1` |

| Workload | Output tok/s | Decode tok/s | TTFT mean s | Graph replay | Eager decode | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4096/128/batch4 | `51.2962` | `132.3013` | `5.8816` | `127` | `0` | reused 07.60 |
| 4096/1024/batch4 | `105.7645` | `132.5127` | `5.9133` | `1023` | `0` | reused 07.60 |

Memory cost of the three cached BF16 owners remains exactly `1.0000 GiB/rank`,
or about `14121.79` KV tokens / `55.16` pages per rank.

Mini is still `7.28%` below the old `114.07 output tok/s` serving victory line
on 4096/1024/batch4.

## Artifacts

| Path | Contents |
| --- | --- |
| `raw/mini_reused_macro_qwqb_wob_idxwqb_4096x128_bs4_np128` | symlink to reused 07.60 mini 4096/128 macro |
| `raw/mini_reused_macro_qwqb_wob_idxwqb_4096x1024_bs4_np128` | symlink to reused 07.60 mini 4096/1024 macro |
| `raw/dsv4_target0761_vllm_4096x128_bs4` | symlink to fresh vLLM 4096/128 macro attempt |
| `raw/nsys_target0761_vllm_4096x128_bs4.sqlite` | fresh vLLM SQLite export, not usable for repeat-window bucket timing |
| `raw/vllm_nsys_4096x128_attempt.log` | fresh vLLM nsys attempt log |
| `summaries/nsys_target0761_vllm_4096x128_bs4_classified.md` | classifier output showing repeat-window attribution blocker |
| `summaries/parity_tables.md` | macro, bucket, owner, and decision tables |
| `summaries/parity_decision_summary.json` | machine-readable decision summary |

## vLLM Baseline

The stable vLLM macro baseline is reused from TARGET 07.43. A fresh 07.61
4096/128/batch4 run was also attempted under nsys and completed the workload.

vLLM source state:

| Field | Value |
| --- | --- |
| root | `/workspace/vllm-dsv4-docker` |
| virtualenv | `/workspace/venvs/vllm-dsv4` |
| branch | `minisgl_docker` |
| commit | `bfaea783f5192189b49ca21c2893f7266345e09c` |
| status | pre-existing 07.43 env-gated edits in `deepseek_v4_attention.py` and `sparse_attn_indexer.py`, plus untracked ncu report dirs |
| 07.61 edits | none |
| ablation envs | unset for the fresh run |

| Workload | Output tok/s | Decode tok/s | Runtime/profile note | Precision/cache note | Source |
| --- | ---: | ---: | --- | --- | --- |
| 4096/128/batch4 | `82.2824` | n/a | offline `LLM.generate`; no TTFT | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer, MXFP4/Marlin MoE | reused 07.43 control |
| 4096/128/batch4 | `80.8862` | n/a | fresh 07.61 one-repeat nsys attempt; macro pass | same | fresh 07.61 attempt |
| 4096/1024/batch4 | `202.0342` | n/a | offline `LLM.generate`; no TTFT | same | reused 07.43 control |

Fresh vLLM profile blocker: the 07.61 SQLite export succeeded, but the requested
`repeat:decode_throughput_bs8:0` NVTX window contains `0` CUDA kernels and only
one sync API. The total capture reports only `0.9681s` summed kernel time for a
`6.3299s` measured repeat. Therefore vLLM runtime bucket timing is still
unavailable; this report uses vLLM macro plus source dispatch parity.

## Macro Parity

| Framework | Workload | Output tok/s | Decode tok/s | Ratio vs mini | TTFT/prefill note | Precision/cache note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| mini | 4096/128/batch4 | `51.2962` | `132.3013` | `1.00x` | TTFT `5.8816s` | cached BF16 `q_wqb/wo_b/indexer.wq_b`; FP8 indexer cache; BF16 KV |
| vLLM | 4096/128/batch4 | `82.2824` | n/a | `1.60x` | offline macro | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer |
| vLLM fresh | 4096/128/batch4 | `80.8862` | n/a | `1.58x` | nsys attempt macro | same |
| mini | 4096/1024/batch4 | `105.7645` | `132.5127` | `1.00x` | TTFT `5.9133s` | cached BF16 `q_wqb/wo_b/indexer.wq_b`; FP8 indexer cache; BF16 KV |
| vLLM | 4096/1024/batch4 | `202.0342` | n/a | `1.91x` | offline macro | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer |

## Bucket Parity

Mini buckets are from the TARGET 07.60 rank0 4096/128 decode envelope
(`4.408601s`, 127 replayed decode steps).

| Mini bucket | Mini time s | Mini share | vLLM comparable bucket | vLLM time s | Decision note |
| --- | ---: | ---: | --- | ---: | --- |
| graph/runtime/copy/cat/index | `1.141006` | `25.88%` | graph-owned runner buffers, custom attention op, compile cleanup | n/a | largest broad mini bucket; not actionable without a named boundary |
| projection/GEMM | `0.805080` | `18.26%` | FP8 linear dispatch, Marlin, `wo_a` BMM/FP8 einsum | n/a | cached BF16 solved the prior three owners; next projection boundary is `wo_a` |
| elementwise graph nodes | `0.639409` | `14.50%` | fused q/KV RMSNorm, qnorm/RoPE/KV insert, fused inv-RoPE/quant | n/a | actionable through owner boundary, not generic elementwise cleanup |
| NCCL communication | `0.346671` | `7.86%` | graph-registered custom all-reduce | n/a | not top-two; repeat-window vLLM comm timing unavailable |
| MoE/Marlin | `0.316854` | `7.19%` | MXFP4/Marlin FusedMoE runner | n/a | not primary after mini Marlin work |
| FP8 indexer cache/logits/topk | `0.131028` | `2.97%` | FP8 indexer cache and `SparseAttnIndexer` | n/a | mini already ported the narrow FP8 indexer backend |
| sparse attention decode | `0.118408` | `2.69%` | SM80 gather/dequant plus split-K sparse decode | n/a | already near prior vLLM probe parity |

## Owner/Boundary Parity

| Owner/boundary | Mini 07.60 evidence | vLLM evidence | Next-action implication |
| --- | --- | --- | --- |
| `attn.wo_a` | replay kernel `0.481377s`; copy/layout `0.290148s`; elementwise `0.137695s`; intrinsic `0.053534s` | `deepseek_v4_attention.py` uses a SM80 per-group BF16 BMM weight cache for `wo_a`; non-reference path uses fused inverse RoPE plus FP8 einsum | choose next target |
| shared experts gate/up/down | combined replay `0.468967s`; combined non-GEMM `0.394034s` | vLLM `FusedMoE` runner can call `moe_forward_shared` and run shared experts on an aux stream or MK-owned overlap | secondary; more diffuse and no runtime timing proof |
| `attn.wo_b` all-reduce | owner comm `0.159649s`; total NCCL `0.346671s`; mini benchmark comm `704` collectives / `139.60GB` | vLLM registers CUDA graph all-reduce addresses; total profile shows few NCCL kernels but repeat timing is invalid | defer communication/overlap |
| graph/runtime/copy/cat/index | broad mini bucket `1.141006s` | vLLM graph buffers and custom op topology are clear, but per-bucket timing is missing | do not start broad graph cleanup |
| remaining projection/GEMM | `0.805080s` total; `0.469287s` owner-attributed, `0.335793s` unattributed | vLLM FP8 linear can choose Marlin or BF16 dequant/F.linear; mini already matched the old three big owners with cached BF16 | avoid blind cached-weight expansion |
| MoE/Marlin | bucket `0.316854s`; routed-expert Marlin subbucket `0.281556s` | vLLM MXFP4/Marlin FusedMoE | not top-two |

## Source Dispatch Notes

Mini source:

- `python/minisgl/models/deepseek_v4.py`: `DSV4Linear.forward_fp8_cached_bf16_weight`, cached owners, `attn.wo_a`, `attn.wo_b`, and shared experts.
- `python/minisgl/kernel/deepseek_v4.py`: `wo_a_grouped_projection_fallback` dequantizes FP8 weight to BF16 and runs grouped `einsum`.
- `python/minisgl/engine/graph.py` and `python/minisgl/attention/deepseek_v4.py`: graph replay input copy and DSV4 metadata replay boundaries.

vLLM source:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`: `DeepseekV4Attention` defines `wq_b`, `wo_a`, `wo_b`, and the reusable `topk_indices_buffer`.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`: SM80 `wo_a` BF16 BMM cache, fused inverse RoPE plus FP8 einsum, `deepseek_v4_attention` custom op, `fp8_ds_mla` KV-cache canonicalization, and FP8 indexer dispatch.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`: `Fp8LinearMethod.apply` selects Marlin FP8 scaled-mm or BF16 dequantized `F.linear` depending backend/mode.
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/shared_experts.py` and `runner/moe_runner.py`: shared expert overlap and `moe_forward_shared` structure.

## Decision

Proceed to:

```text
TARGET 07.62: DSV4 SM80 wo_a attention boundary parity
```

Rationale:

- `attn.wo_a` is the strongest single owner: `0.481377s`, or `10.92%` of the
  07.60 decode envelope.
- Its non-GEMM chain is `0.427843s`. Cutting `60%` of that chain is about
  `0.2567s` in the 127-step profile.
- Scaled to the primary 1024-token decode workload, that is about `2.07s`
  against the `38.7276s` current mini elapsed time, or roughly `5.3%` expected
  E2E gain.
- vLLM has a concrete corresponding source boundary: SM80 per-group BF16 BMM
  weight cache for `wo_a`, with an FP8 einsum alternative outside the reference
  path.

This is not a request to continue inertial cached BF16 expansion. The next
target should specifically test whether adapting the vLLM `wo_a` attention
boundary removes mini's copy/layout and elementwise chain, with memory ledger
and graph correctness gates.

## Do-Not-Continue Conditions

- Do not extend cached BF16 weights to another owner unless a fresh profile
  shows a top-two owner/bucket and at least `5%` expected primary E2E gain.
- Do not start broad graph/runtime deforestation without naming the concrete
  copied, concatenated, indexed, or reshaped boundary.
- Do not choose shared-expert layout unless a fresh owner/boundary profile shows
  it clears the `5%` primary E2E gate independently of `wo_a`.
- Do not choose communication/overlap unless fresh vLLM or mini evidence shows
  materially lower NCCL cost or overlap with at least `5%` expected primary E2E
  gain.
- Do not choose full precision/cache work unless the next target states the
  quality risk and proves a top-two bucket/owner link.
- In the next `wo_a` target, stop if graph replay is not preserved, eager decode
  becomes nonzero, or text smoke fails after one focused correctness fix.
