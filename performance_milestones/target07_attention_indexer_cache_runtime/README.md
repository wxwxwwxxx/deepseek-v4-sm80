# TARGET 07.393: Attention/Indexer/Cache Runtime Rework

## Status

Evidence-only target complete.  No mini runtime behavior was changed.

Selected next route: `adapt_vllm_design` for mini's exact bf16 path, in a
narrow follow-up target.  Directly porting vLLM's attention/cache/indexer path
would change mini's default cache precision/layout, because vLLM's winning sm80
path uses `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, and FP8 indexer
cache.

## Artifacts

- Dispatch JSON:
  `summaries/dispatch_backend_report.json`
- Dispatch markdown:
  `summaries/dispatch_backend_report.md`
- Microbench summary:
  `summaries/subgraph_microbench.json`
- Candidate table:
  `summaries/candidate_decision_table.md`
- Raw mini probe:
  `raw/mini_attention_indexer_cache_microbench_t4_h4096.json`
- Raw vLLM probe:
  `raw/vllm_attention_indexer_cache_microbench_t4_h4096.json`
- Probe scripts:
  `scripts/probe_dispatch_backends.py`,
  `scripts/mini_attention_indexer_cache_microbench.py`,
  `scripts/vllm_attention_indexer_cache_microbench.py`

## Backend Findings

vLLM sm80:

- `DeepseekV4Attention` wraps `DeepseekV4MLAModules` in
  `DeepseekV4MultiHeadLatentAttentionWrapper`.
- Forward calls `torch.ops.vllm.deepseek_v4_attention`.
- The backend class is `DeepseekV4FlashMLASparseBackend`
  (`V4_FLASHMLA_SPARSE`), but FlashMLA sparse kernels support sm90/sm100 only.
  On sm80, vLLM routes through reference/sm80 kernels.
- Decode after topk is:
  `compute_global_topk_indices_and_lens` ->
  `gather_dequant_two_scopes_with_mask` over packed `fp8_ds_mla` cache ->
  `_dsv4_sm80_sparse_attn_decode_triton`.
- KV cache is `uint8` `fp8_ds_mla`, 584 bytes/token.  Indexer cache is FP8
  `uint8` on sm80.  FP4 indexer cache is SM100-only.
- Runtime uses vLLM V1 CUDA graph dispatch, persistent buffers, graph capture
  stream, pinned sampling copies, custom all-reduce graph buffer registration,
  and `AuxStreamType.Attention` overlap.

mini current canonical path:

- Exact bf16 activation/cache default plus MXFP4 Marlin WNA16 MoE.
- `DSV4CacheLayoutPolicy` is `bf16_flat` for SWA, C4, C128, indexer cache, and
  compressor state.
- Attention front/cache store uses `q_kv_norm_rope_cache_fallback` ->
  `q_kv_norm_rope_cache_bf16` when enabled.
- Sparse decode uses `dsv4_sparse_attention_two_source_bf16`, reading bf16
  SWA/C4/C128 caches directly.
- Indexer uses `indexer_select_bf16_fallback`:
  `indexer_bf16_logits` + `topk_transform_512`, bf16 indexer cache, fp32
  weights/logits.
- Graph replay still copies DSV4 attention metadata around replay.

## Paired Microbench

All current probes used T=4, history=4096, page size=256, A100/sm80.

| Probe | Mean ms | Note |
| --- | ---: | --- |
| mini attention front + bf16 SWA store | 0.253 | exact bf16 |
| mini compressed/indexer cache store | 0.976 | C4/C128 norm+RoPE store plus indexer hadamard store |
| mini replay metadata copy | 0.126 | device-side stand-in for replay metadata copies |
| mini bf16 indexer select | 0.339 | C4 length 1024, topk 512 |
| mini bf16 sparse attention | 0.578 | direct bf16 SWA+C4 cache reads |
| mini combined indexer+sparse decode | 1.311 | synthetic, excludes model projections |
| vLLM global topk/lens | 0.0566 | concrete metadata-consolidation op |
| vLLM indexer Q RoPE+quant | 0.0871 | not full SparseAttnIndexer |
| vLLM gather/dequant two scopes | 0.1129 | packed fp8 cache -> bf16 gathered KV |
| vLLM split-K sparse core | 0.1320 | core only, after gather/dequant |
| vLLM gather+split-K decode | 0.2258 | after topk known, excludes full indexer |

Important caveat: vLLM's split-K core alone is not the fair comparison.  The
fair sparse decode fragment is gather/dequant plus split-K, 0.2258 ms here.
Even with that included, it is 2.56x faster than mini's direct bf16 sparse
attention probe, but the cache layout and precision differ.

## Macro Context

From TARGET 07.392:

- mini Marlin WNA16 exact, 4096/128/bs4: 33.97 output tok/s.
- mini Marlin WNA16 exact, 4096/1024/bs4: 54.64 output tok/s.
- vLLM `deepseek_v4_fp8`, 4096/128/bs4: 82.08 output tok/s.
- vLLM `deepseek_v4_fp8`, 4096/1024/bs4: 201.99 output tok/s.
- mini rank0 4096/128 Nsight window:
  sparse attention 13.28%, metadata/runtime/copy 12.27%, indexer/cache 6.10%.

## Candidate Decision

`direct_port_vllm`: rejected for exact default.  It would port packed FP8 cache
and FP8 indexer semantics, not just a kernel.

`adapt_vllm_design`: selected.  The design-portable pieces are global topk/lens
mapping, two-scope gather/mask, split-K decode, graph-stable metadata buffers,
and attention aux-stream/buffer ownership ideas.  This clears the expected
upside threshold but is too large for a quiet 07.393 patch.

`optimize_mini_existing`: deferred.  It may be useful after the boundary is
chosen, but starting here would be speculative local polish.

`precision_cache_experiment`: recommended as a separate opt-in lane if exact
bf16 boundary adaptation cannot close enough of the gap.

## Changes Made

No runtime/model/kernel changes were made.  This target added evidence scripts,
raw probe outputs, and decision artifacts only.

## Validation

Completed:

- Python syntax checks for all new scripts.
- Dispatch report generated successfully.
- mini paired probes ran successfully on A100/sm80.
- vLLM paired probes ran successfully on A100/sm80, with full-engine-only
  boundaries recorded as blockers.
- `summaries/subgraph_microbench.json` validated with `python -m json.tool`.

Not run because there was no runtime implementation:

- TP8 text smoke.
- 4096/128 or 4096/1024 post-change macro.
- post-change Nsight.

## Next Target

Use the follow-up exact-boundary target:

`prompts/TARGET_07.394_dsv4_sm80_exact_attention_indexer_boundary_adapt.md`

Do not continue in 07.393 unless a fresh profile contradicts the current
ranking, or a vLLM engine-level probe makes full FP8 indexer/cache attribution
available without changing mini's exact default.
