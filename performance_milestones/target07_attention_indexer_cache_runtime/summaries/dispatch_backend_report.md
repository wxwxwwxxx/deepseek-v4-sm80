# TARGET 07.393 Dispatch Backend Report

## Verdict

vLLM's DeepSeek V4 sm80 path is not "just FlashMLA attention".  The observed
backend is FlashMLA Sparse metadata/cache format plus sm80 reference kernels:
packed `fp8_ds_mla` cache gather/dequant, Triton split-K sparse decode, FP8
indexer cache, and vLLM V1 CUDA graph/runtime buffer ownership.

mini's current canonical path is exact bf16 activation/cache plus MXFP4 Marlin
WNA16 MoE.  Its attention/indexer/cache boundaries are semantically close, but
the dtype/layout boundary is different enough that a direct vLLM port would be
precision-changing.

## vLLM Sm80 Backend Facts

- Model quantization reports `deepseek_v4_fp8`; MoE is MXFP4/Marlin-family, but
  the attention/cache/indexer path uses FP8 cache policy.
- `DeepseekV4Attention` wraps `DeepseekV4MLAModules` in
  `DeepseekV4MultiHeadLatentAttentionWrapper`.
- Forward calls `torch.ops.vllm.deepseek_v4_attention`, registered in
  `vllm/model_executor/layers/deepseek_v4_attention.py`.
- Attention backend class is `DeepseekV4FlashMLASparseBackend`
  (`V4_FLASHMLA_SPARSE`), but FlashMLA sparse kernels only support compute
  capability 9/10.  On A100/sm80, `use_dsv4_reference_kernels()` routes decode
  through vLLM's reference/sm80 path.
- Decode path after topk is:
  `compute_global_topk_indices_and_lens` ->
  `gather_dequant_two_scopes_with_mask` over packed `fp8_ds_mla` cache ->
  `_dsv4_sm80_sparse_attn_decode_triton`.
- Main MLA/SWA cache is `uint8` `fp8_ds_mla`, 584 bytes/token in the V4 layout.
  Compressed cache is `MLAAttentionSpec(dtype=uint8, head_size=512,
  alignment=576, model_version="deepseek_v4")`.
- Indexer backend is `DeepseekV4IndexerBackend` / `DEEPSEEK_V4_INDEXER`.
  SM80 uses FP8 indexer cache, not FP4 indexer cache.  `SparseAttnIndexer`
  uses `fp8_paged_mqa_logits_triton` and persistent/top-k decode paths.
- Runtime uses vLLM V1 `CudagraphDispatcher`, `parallel_state.graph_capture`
  on a separate stream, persistent buffers, pinned sampled-token copies, graph
  buffer registration for custom all-reduce, and `AuxStreamType.Attention` for
  KV insert/compressor overlap with indexer work.

## mini Backend Facts

- Canonical variant:
  `v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`.
- `DSV4CacheLayoutPolicy` defaults to `bf16_flat`: SWA, C4, C128, indexer
  cache, and compressor state are bf16.
- Attention front/cache store dispatch:
  `q_kv_norm_rope_cache_fallback` -> Triton `q_kv_norm_rope_cache_bf16` when
  enabled; otherwise split torch/Triton fallbacks.
- Sparse decode dispatch:
  `DSV4AttentionBackend._sparse_attention_two_source` ->
  `dsv4_sparse_attention_two_source_bf16`, reading bf16 SWA and optional bf16
  C4/C128 caches directly.
- Indexer dispatch:
  `indexer_select_bf16_fallback` -> `indexer_bf16_logits` plus
  `topk_transform_512`, with bf16 indexer cache and fp32 weights/logits.
- Cache/update dispatch:
  `compress_norm_rope_store_fallback`, `store_compressed_fallback`,
  `store_indexer_fallback`, and `copy_masked_compressed_locs`.
- Graph/runtime:
  `GraphRunner` captures `[1,2,4]`-style graphs, but DSV4 attention metadata is
  copied into graph buffers around replay.

## Paired Boundary Map

| Boundary | mini | vLLM | Portability |
| --- | --- | --- | --- |
| Attention front/cache insert | bf16 q/KV norm+RoPE and bf16 SWA/cache stores | custom op quant-inserts packed `fp8_ds_mla` cache | Direct port changes precision/layout |
| Sparse decode | one bf16-cache two-source sparse attention kernel | global topk + fp8 gather/dequant/mask + split-K decode | Adapt design for bf16 exact path; direct port belongs to FP8 cache target |
| Indexer | bf16 q/cache logits + topk transform | fused Q RoPE/FP8 quant + FP8 paged logits + persistent topk | Same role, different cache/query precision |
| Cache layout | bf16 flat SWA/C4/C128/indexer | byte-packed `fp8_ds_mla` KV and FP8 indexer cache | Central mismatch |
| Runtime/graph | mini graph replay plus attention metadata copies | CudagraphDispatcher persistent buffers, graph capture stream, aux stream overlap | Borrow buffer/metadata discipline after exact boundary work |

## Evidence Implication

The next exact-path target should adapt vLLM's boundary design, not directly
port vLLM's cache/indexer code as default.  The direct vLLM path is a valid
precision/cache experiment, but it must remain opt-in and separate from the
exact bf16 baseline.
