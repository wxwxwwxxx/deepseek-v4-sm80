# TARGET 07.41: DSV4 SM80 Exact Indexer, Cache, Runtime Work

## Goal

Implement the next exact-path optimization only if TARGET 07.40 selects
indexer/cache/runtime as the top remaining bottleneck after split-K sparse
decode.

The default precision policy must remain exact:

- bf16 activation/cache for DSV4 attention/indexer/cache state;
- Marlin WNA16 expert weights;
- no packed `fp8_ds_mla` KV cache as default;
- no FP8/FP4 indexer cache as default.

## Required Input

Do not start this target until TARGET 07.40 has written a post-splitK
bottleneck ranking.

Read first:

- `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`
- `performance_milestones/target07_post_splitk_reprofile/README.md`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`

Relevant mini paths:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

Relevant vLLM paths:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`

## Candidate Exact Cuts

Choose one cut, not all of them.

1. Exact bf16 indexer/cache boundary
   - Reduce `_indexer_bf16_logits_kernel` cost or fuse surrounding metadata work.
   - Preserve bf16 indexer cache and fp32 model-original math where required.
   - Compare against vLLM's `SparseAttnIndexer` design, while rejecting FP8
     cache/indexer pieces for the exact default.

2. Graph-stable metadata and buffer ownership
   - Reduce replay-time copies, cat/index kernels, and shape-dependent
     allocations.
   - Prefer persistent graph-owned buffers when the shape is bounded and safe.
   - Do not silently fall back during CUDA graph capture.

3. Cache store/compressor/indexer store cleanup
   - Only if TARGET 07.40 shows this bucket is top-two.
   - Keep bf16 cache layout and page size 256 semantics.

## Out Of Scope

- changing default cache/indexer dtype to FP8;
- packed `fp8_ds_mla` cache;
- MoE/Marlin work;
- more split-K sparse decode kernel polish;
- broad stream-overlap rewrites without profile evidence.

## Success Criteria

Keep the cut only if correctness passes and at least one holds:

- selected subgraph improves by at least `20%`;
- 4096/128/batch4 macro improves by at least `5%`;
- 4096/1024/batch4 macro improves by at least `5%`.

If the selected exact cut cannot clear these bars, stop and recommend either
TARGET 07.50 precision/cache or a fresh parity/profile pass.

## Expected Output

Create:

- `performance_milestones/target07_indexer_cache_runtime_exact/README.md`
- `scripts/`, `raw/`, and `summaries/` under that directory.

The README must record:

- why TARGET 07.40 selected this cut;
- exact boundary implemented;
- before/after microbench;
- TP8 text smoke;
- macro result;
- whether exact bf16 work should continue or TARGET 07.50 should begin.
