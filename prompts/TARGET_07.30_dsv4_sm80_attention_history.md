# TARGET 07.30: DSV4 SM80 Attention, Indexer, Cache History

## Status

Completed history merge for the attention/indexer/cache chain through
TARGET 07.395.

This file summarizes:

- `TARGET_07.392_dsv4_sm80_post_marlin_reprofile.md`
- `TARGET_07.393_dsv4_sm80_attention_indexer_cache_runtime_rework.md`
- `TARGET_07.394_dsv4_sm80_exact_attention_indexer_boundary_adapt.md`
- `TARGET_07.395_dsv4_sm80_bf16_sparse_decode_splitk.md`

The original prompts now live under `prompts/archive/target07/` as archival
details.  Use this file to understand why the project is moving beyond sparse
decode and toward post-splitK reprofiling, indexer/cache/runtime work, or an
opt-in precision/cache lane.

## Motivation

After Marlin WNA16, MoE was no longer the primary bottleneck.  Fresh profiles
shifted the top contributors to sparse attention, metadata/runtime/copy, and
indexer/cache.  vLLM was still much faster, but its sm80 path used a different
precision/cache policy:

- `deepseek_v4_fp8` engine quantization;
- packed `fp8_ds_mla` KV cache;
- FP8 indexer cache;
- vLLM V1 graph/runtime buffer ownership;
- MXFP4/Marlin MoE.

The attention phase therefore followed a two-lane rule:

- first adapt vLLM's design where it can preserve mini's exact bf16
  activation/cache policy;
- only open a precision/cache lane after exact bf16 boundaries have been given
  a fair chance.

## Timeline And Conclusions

### TARGET 07.392: Post-Marlin Reprofile

Artifacts:

- `performance_milestones/target07_post_marlin_reprofile/`

Key measurements:

- mini Marlin WNA16 exact, 4096/1024/batch4: `54.64 output tok/s`;
- fresh vLLM offline, 4096/1024/batch4: `201.99 output tok/s`;
- fresh vLLM offline, 4096/128/batch4: `82.08 output tok/s`.

Conclusion:

- MoE was no longer primary.
- Top mini contributors moved to sparse attention, metadata/runtime/copy, and
  indexer/cache.
- The next target had to identify the exact vLLM attention/indexer/cache path
  before implementing more local kernels.

### TARGET 07.393: Attention/Indexer/Cache Runtime Rework

Artifacts:

- `performance_milestones/target07_attention_indexer_cache_runtime/`

vLLM sm80 path identified:

```text
compute_global_topk_indices_and_lens
-> gather_dequant_two_scopes_with_mask over packed fp8_ds_mla cache
-> _dsv4_sm80_sparse_attn_decode_triton
```

Additional vLLM facts:

- backend class is `DeepseekV4FlashMLASparseBackend`, but sm80 routes through
  reference/sm80 kernels rather than sm90/sm100 FlashMLA sparse kernels;
- KV cache is packed `uint8` `fp8_ds_mla`;
- indexer cache is FP8 on sm80;
- runtime uses vLLM V1 graph dispatch and persistent buffers.

Decision:

- Do not direct-port vLLM as mini's exact default, because that would change
  cache precision/layout.
- Adapt the design in exact bf16 first.

### TARGET 07.394: Exact Global Topk/Lens

Artifacts:

- `performance_milestones/target07_exact_attention_indexer_boundary_adapt/`

Implemented:

- `MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS=1`
- `DSV4TopKTransformGlobalLensKernel`
- graph-safe propagation of `topk_lens` into DSV4 attention metadata.

Results:

- topk full transform: `0.1838 ms -> 0.0741 ms`, `59.7%` faster;
- full bf16 indexer select: `0.3323 ms -> 0.2157 ms`, `35.1%` faster;
- 4096/1024/batch4: `54.64 -> 55.05 output tok/s`.

Conclusion:

- The cut was real and should be kept.
- Macro gain was small, so topk/lens was not the main remaining gap.
- Next target: sparse decode boundary.

### TARGET 07.395: Exact BF16 Sparse Decode Split-K

Artifacts:

- `performance_milestones/target07_bf16_sparse_decode_splitk/`

Implemented:

- `MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16=1`
- decode-only bf16 two-scope gather/mask plus split-K sparse decode;
- old `dsv4_sparse_attention_two_source_bf16` path retained for A/B and
  rollback.

Results:

| Workload | Before | After | Delta |
| --- | ---: | ---: | ---: |
| sparse-only decode T=4/H=4096 | `0.5768 ms` | `0.2284 ms` | `+60.4%` |
| globaltopk + indexer + sparse | `0.7890 ms` | `0.4350 ms` | `+44.9%` |
| 4096/128/batch4 | `34.14 tok/s` | `38.94 tok/s` | `+14.05%` |
| 4096/1024/batch4 | `55.05 tok/s` | `68.81 tok/s` | `+24.99%` |

Important comparison:

- vLLM's prior gather+split-K decode probe was about `0.2258 ms`;
- mini's new exact bf16 sparse-only decode is about `0.2284 ms`.

Conclusion:

- mini-owned exact bf16 split-K effectively matches vLLM at the comparable
  decode sparse boundary.
- The remaining vLLM macro gap is therefore not mainly "we did not call the
  vLLM sparse decode kernel".
- Remaining candidates are prefill/legacy sparse work, indexer/cache/runtime
  buffer overhead, graph/memcpy/allocation behavior, and vLLM's packed
  FP8 cache/indexer precision lane.

## Current Best Exact Stack

Use this conceptual stack for the next measurement:

- Marlin WNA16 MoE expert backend;
- global topk/lens;
- bf16 split-K sparse decode;
- DSV4 decode CUDA graph replay;
- page size 256, `--num-pages 128`;
- TP8 on 8x A100.

Representative variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

## Current Decision

Do not continue topk or split-K sparse decode kernel polish.

The next step is TARGET 07.40:

- reprofile the post-splitK best exact stack;
- fix attribution so decode split-K, legacy prefill sparse, indexer/cache, and
  runtime/copy buckets are separated;
- then choose between exact indexer/cache/runtime work and the opt-in
  precision/cache lane.

## Do Not Continue Here Unless

- split-K correctness regresses;
- a new profile proves decode split-K itself is again a top-two contributor;
- a future precision/cache target needs this file as the exact bf16 baseline.
