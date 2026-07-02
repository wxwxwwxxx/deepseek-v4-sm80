# TARGET 07.61 Parity Tables

Date: 2026-07-02

## Mini Baseline

Mini baseline source: TARGET 07.60 reused/frozen artifacts.

Variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

| Workload | Output tok/s | Decode tok/s | TTFT mean s | Page size | Num pages | TP | Graph replay | Eager decode | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096/128/batch4 | `51.2962` | `132.3013` | `5.8816` | `256` | `128` | `8` | `127` | `0` | reused 07.60 |
| 4096/1024/batch4 | `105.7645` | `132.5127` | `5.9133` | `256` | `128` | `8` | `1023` | `0` | reused 07.60 |

Mini precision/cache note: cached BF16 dequantized weights are enabled only for
`attn.q_wqb`, `attn.wo_b`, and `indexer.wq_b`. The active stack also uses the
opt-in vLLM-aligned FP8 indexer cache backend, but it does not use vLLM's full
`fp8_ds_mla` KV-cache layout.

## vLLM Baseline

| Workload | Output tok/s | Decode tok/s | TTFT/prefill note | Graph/compile note | Precision/cache note | Source |
| --- | ---: | ---: | --- | --- | --- | --- |
| 4096/128/batch4 | `82.2824` | n/a | offline `LLM.generate`, no TTFT | CUDA graph sizes `[1,2,4]` | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer, MXFP4/Marlin MoE | reused 07.43 control |
| 4096/128/batch4 | `80.8862` | n/a | fresh one-repeat nsys attempt | same; nsys repeat window had 0 kernels | same | fresh 07.61 attempt |
| 4096/1024/batch4 | `202.0342` | n/a | offline `LLM.generate`, no TTFT | CUDA graph sizes `[1,2,4]` | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer, MXFP4/Marlin MoE | reused 07.43 control |

vLLM runtime-profile caveat: the fresh 07.61 vLLM run completed and exported
SQLite, but `repeat:decode_throughput_bs8:0` contains `0` CUDA kernels and only
one sync API. Total captured kernel time is only `0.9681s` for a run whose
measured workload repeat took `6.3299s`, so the fresh vLLM SQLite is not valid
for per-bucket timing. It is valid as macro and source-path evidence.

## Macro Parity

| Framework | Workload | Output tok/s | Decode tok/s | Ratio vs mini | TTFT/prefill note | Precision/cache note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| mini | 4096/128/batch4 | `51.2962` | `132.3013` | `1.00x` | TTFT `5.8816s`; prefill `3180.24 tok/s` | three-owner cached BF16 weights plus FP8 indexer cache; BF16 KV |
| vLLM | 4096/128/batch4 | `82.2824` | n/a | `1.60x` | offline macro, no TTFT | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer, MXFP4/Marlin |
| vLLM fresh attempt | 4096/128/batch4 | `80.8862` | n/a | `1.58x` | one repeat under nsys attempt | same as vLLM control |
| mini | 4096/1024/batch4 | `105.7645` | `132.5127` | `1.00x` | TTFT `5.9133s`; prefill `3161.81 tok/s` | three-owner cached BF16 weights plus FP8 indexer cache; BF16 KV |
| vLLM | 4096/1024/batch4 | `202.0342` | n/a | `1.91x` | offline macro, no TTFT | `deepseek_v4_fp8`, `fp8_ds_mla`, FP8 indexer, MXFP4/Marlin |

Mini is still `7.28%` below the old `114.07 output tok/s` serving victory line
on the primary 4096/1024/batch4 workload.

## Bucket Parity

Mini buckets are from the 07.60 4096/128 rank0 decode envelope
(`4.408601s`, 127 replayed decode steps).

| Mini bucket | Mini time s | Mini share | vLLM comparable bucket | vLLM time s | Decision note |
| --- | ---: | ---: | --- | ---: | --- |
| graph/runtime/copy/cat/index | `1.141006` | `25.88%` | V1 graph buffers, custom attention op, compile/noop cleanup, persistent top-k workspace | n/a | largest mini bucket, but 07.41 metacopy failed macro; use only after naming a concrete boundary |
| projection/GEMM | `0.805080` | `18.26%` | `ColumnParallelLinear`/`RowParallelLinear`, Marlin FP8 linear, `wo_a` BMM or FP8 einsum | n/a | cached BF16 solved the prior three owners; next projection boundary is `wo_a`, not blind cache expansion |
| elementwise graph nodes | `0.639409` | `14.50%` | fused q/KV RMSNorm, fused qnorm/RoPE/KV insert, fused inv-RoPE/quant, compile fusions | n/a | actionable only through an owner boundary; strongest owner is `attn.wo_a` |
| NCCL communication | `0.346671` | `7.86%` | graph-registered custom all-reduce and fewer visible NCCL kernels in total capture | n/a | total vLLM capture has only `16` NCCL kernels, but repeat timing is invalid; not top-two |
| MoE/Marlin | `0.316854` | `7.19%` | MXFP4/Marlin FusedMoE runner | n/a | not top-two after mini Marlin work |
| FP8 indexer cache/logits/topk | `0.131028` | `2.97%` | FP8 indexer cache, `SparseAttnIndexer`, `fp8_paged_mqa_logits_triton` | n/a | mini already ported the narrow FP8 indexer backend |
| sparse attention decode | `0.118408` | `2.69%` | SM80 reference gather/dequant plus split-K sparse decode, or FlashMLA sparse when not reference | n/a | mini exact split-K sparse decode was already near vLLM probe parity |

## Owner/Boundary Parity

| Owner/boundary | Mini 07.60 evidence | vLLM source evidence | Next-action implication |
| --- | --- | --- | --- |
| `attn.wo_a` | replay kernel `0.481377s`; copy/layout `0.290148s`; elementwise `0.137695s`; intrinsic `0.053534s`; `344` replay graph nodes | `deepseek_v4_attention.py` has SM80 reference `wo_a` per-group BF16 BMM weight cache through `_apply_wo_a_bmm`; non-reference path uses `fused_inv_rope_fp8_quant` plus `deepseek_v4_fp8_einsum` before `wo_b` | chosen next target: narrow `wo_a`/attention projection boundary |
| shared experts gate/up/down | gate/up replay `0.273793s`; down replay `0.195174s`; combined `0.468967s`, with `0.394034s` non-GEMM wrapper/elementwise | vLLM routes shared experts through `FusedMoE` and `moe_forward_shared`; `SharedExperts` can use a separate stream or MK-owned overlap | secondary candidate, but more diffuse and lacks vLLM runtime timing proof |
| `attn.wo_b` all-reduce | owner communication `0.159649s`; total NCCL `0.346671s`; benchmark comm counters `704` collectives and `139.60GB` | vLLM logs graph address registration (`348` then `522`) and has custom all-reduce enabled; repeat-window NCCL timing is unavailable | do not pick communication before fresh overlap/timing evidence |
| graph/runtime/copy/cat/index | broad mini bucket `1.141006s`; many direct-copy/cat/index kernels | vLLM has graph-managed runner buffers and custom op boundaries, but fresh repeat profile has no kernels | broad bucket only; target must name a copied/reshaped/indexed boundary |
| remaining projection/GEMM | total projection/GEMM `0.805080s`; owner-attributed intrinsic `0.469287s`; unattributed `0.335793s` | vLLM FP8 linear dispatch can choose Marlin or BF16 dequant/F.linear depending mode; `wq_b`/`wo_b` source boundaries now mostly matched by mini cached BF16 | avoid blind cached-weight expansion |
| MoE/Marlin | bucket `0.316854s`; routed-expert owner replay `0.424127s`, Marlin subbucket `0.281556s` | vLLM uses MXFP4/Marlin FusedMoE and runner custom-op dispatch | not primary unless fresh runtime evidence moves it into top-two |

## Decision

Proceed to a named `wo_a`/attention boundary target:

```text
TARGET 07.62: DSV4 SM80 wo_a attention boundary parity
```

Why this target:

- `attn.wo_a` is a single top owner with `0.481377s` in the 127-step decode
  profile, or `10.92%` of the decode envelope.
- Its non-GEMM chain is `0.427843s`; cutting `60%` of that chain would remove
  about `0.2567s` from the 127-step profile.
- Scaled to the primary 1024-token decode workload, the same per-step cut is
  about `2.07s` against the `38.7276s` mini elapsed time, or roughly `5.3%`
  expected E2E gain.
- vLLM has a directly relevant source boundary: SM80 per-group BF16 BMM weight
  cache for `wo_a`, with the non-reference path using fused inverse RoPE plus
  FP8 einsum. This is stronger evidence than the shared-expert and comm
  candidates.

Do not choose these now:

- shared-expert layout: close in total owner time, but split across two owners
  and no reliable vLLM runtime timing proves the overlap/layout win.
- communication/overlap: `NCCL` is not top-two; a 50% cut of the current NCCL
  bucket is below the 5% primary E2E gate.
- precision/cache: vLLM's full `fp8_ds_mla` lane is real, but mini sparse
  decode and FP8 indexer are no longer top buckets; choosing it would need a
  precision-policy target and quality risk statement.
- broad graph/runtime deforestation: 07.41 and 07.55 already blocked generic
  metacopy/direct-copy work without a concrete owner boundary.

## Do-Not-Continue Conditions

- Do not extend cached BF16 weights to another owner unless a fresh profile
  shows a top-two owner/bucket and at least `5%` expected primary E2E gain.
- Do not run another broad graph/runtime cleanup without naming the concrete
  boundary and showing it is gate-sized.
- In the next `wo_a` target, stop if a focused `wo_a` BMM/cache or fused
  inv-RoPE/`wo_a` boundary does not show either a `>=15%` owner reduction or
  `>=5%` 4096/1024 output-throughput gain with graph replay preserved and eager
  decode `0`.
- Stop if the implementation requires a default precision-policy change instead
  of an owner-boundary adaptation.
- Stop after one focused correctness fix if mini text smoke fails.
