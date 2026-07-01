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
- `performance_milestones/target07_post_splitk_reprofile/summaries/target07_40_post_splitk_decision_summary.json`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/subgraph_microbench.json`
- `performance_milestones/target07_post_marlin_reprofile/README.md`

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
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/cudagraph_dispatcher.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/utils/multi_stream_utils.py`

## vLLM Comparison Checkpoint

Start by writing down what is and is not already comparable to vLLM.

Known facts from earlier targets:

- TARGET 07.40 gives strong mini-side attribution after split-K, but it does
  not provide a fully comparable vLLM per-subgraph time table.
- The existing vLLM Nsight artifact has incomplete repeat-window child-process
  attribution.  Use it only as weak supporting evidence, not as a per-kernel
  oracle.
- vLLM macro numbers remain useful:
  - 4096/128/batch4: about `82.08 output tok/s`;
  - 4096/1024/batch4: about `201.99 output tok/s`.
- vLLM code topology remains useful:
  - `SparseAttnIndexer` uses FP8 paged logits, persistent topk, and
    `topk_indices_buffer`;
  - `fused_indexer_q_rope_quant` folds Q/RoPE/quant-style work;
  - `deepseek_v4_attention` owns attention/indexer/cache buffers inside the
    attention custom-op boundary;
  - `CudagraphDispatcher`, persistent runner buffers, async output-copy streams,
    and `AuxStreamType.Attention` reduce or hide some runtime/buffer movement.

Before implementing a cut, produce a small comparison note in the milestone
README:

| Mini 07.40 bucket | vLLM analogous design | Can compare time? | Adopt/adapt/reject |
| --- | --- | --- | --- |
| runtime/copy/cat/index graph nodes | persistent buffers, custom-op boundary, graph dispatcher | usually no, unless a fresh profile exists | decide |
| elementwise math graph nodes | fused attention/indexer/cache ops and compiled graph regions | usually no | decide |
| prefill sparse + indexer | sparse prefill/indexer path, but vLLM sm80 prefill path has OOM risk | partial/code only | decide |
| bf16 indexer logits/topk/cache | FP8 paged logits + persistent topk + FP8 indexer cache | precision-changing | decide |

If a fresh vLLM node-trace or child-process-complete profile can be captured
cheaply, use it.  If not, proceed with code-topology and mini-side before/after
evidence, but explicitly state the comparability limit.

## Candidate Exact Cuts

Choose one cut, not all of them.

1. Exact bf16 indexer/cache boundary
   - Reduce `_indexer_bf16_logits_kernel` cost or fuse surrounding metadata work.
   - Preserve bf16 indexer cache and fp32 model-original math where required.
   - Compare against vLLM's `SparseAttnIndexer` design, while rejecting FP8
     cache/indexer pieces for the exact default.
   - Record whether the useful part is portable to bf16, or whether it belongs
     to TARGET 07.50 because it depends on FP8 indexer/cache layout.

2. Graph-stable metadata and buffer ownership
   - Reduce replay-time copies, cat/index kernels, and shape-dependent
     allocations.
   - Prefer persistent graph-owned buffers when the shape is bounded and safe.
   - Do not silently fall back during CUDA graph capture.
   - Compare mini buffer movement against vLLM's persistent runner buffers,
     `CudagraphDispatcher`, async copy streams, and attention custom-op
     ownership.  Adapt the ownership pattern only where mini can keep exact
     bf16 semantics.

3. Cache store/compressor/indexer store cleanup
   - Only if TARGET 07.40 shows this bucket is top-two.
   - Keep bf16 cache layout and page size 256 semantics.
   - Inspect vLLM fused cache/compressor/indexer store code, but reject direct
     default adoption if it requires packed `fp8_ds_mla` or FP8 indexer cache.

4. Legacy prefill sparse/indexer fixed-cost reduction
   - Only if the first comparison note confirms this is the most actionable
     exact cut.
   - Do not port vLLM's sm80 sparse prefill reference path as mini default if it
     materializes the known OOM-prone temporary tensors.
   - Prefer bounded-shape exact bf16 prefill/cache/indexer improvements or
     smaller metadata/copy reductions.

## Out Of Scope

- changing default cache/indexer dtype to FP8;
- packed `fp8_ds_mla` cache;
- MoE/Marlin work;
- more split-K sparse decode kernel polish;
- broad stream-overlap rewrites without profile evidence.
- direct reliance on the old vLLM Nsight repeat window as a complete
  per-subgraph timing oracle.

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
- mini-vs-vLLM comparison note, including whether time comparison was possible
  or only code/macro comparison was possible;
- exact boundary implemented;
- before/after microbench;
- TP8 text smoke;
- macro result;
- whether exact bf16 work should continue or TARGET 07.50 should begin.
