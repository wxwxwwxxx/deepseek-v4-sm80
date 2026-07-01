# TARGET 07.395: DSV4 SM80 Exact BF16 Sparse Decode Split-K

## Goal

Implement and evaluate the next exact-path attention/indexer/cache cut after
TARGET 07.394: a bf16 sparse decode boundary inspired by vLLM's sm80
gather/mask plus split-K sparse decode design.

The default mini path must remain exact:

- bf16 activation/cache for SWA, C4, C128, indexer cache, and compressor state;
- MXFP4 Marlin WNA16 expert weights;
- no packed `fp8_ds_mla` KV cache as default;
- no FP8/FP4 indexer cache as default.

This target should answer one concrete question: can mini close a meaningful
part of the sparse attention gap while preserving the existing bf16 cache
layout, or is vLLM's remaining advantage mainly tied to its packed FP8 cache
and FP8 indexer lane?

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.394_dsv4_sm80_exact_attention_indexer_boundary_adapt.md`
- `performance_milestones/target07_exact_attention_indexer_boundary_adapt/README.md`
- `performance_milestones/target07_exact_attention_indexer_boundary_adapt/summaries/microbench_globaltopk_t4_h4096_summary.json`
- `performance_milestones/target07_attention_indexer_cache_runtime/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/subgraph_microbench.json`
- `performance_milestones/target07_post_marlin_reprofile/summaries/post_marlin_reprofile_summary.json`

Mini source paths:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM reference paths:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/flashmla_sparse.py`

## Current Evidence

TARGET 07.394 successfully adapted vLLM's global topk/lens idea into mini's
exact bf16 path:

- topk full transform: `0.1838 ms -> 0.0741 ms`, `59.7%` faster;
- full bf16 indexer select: `0.3323 ms -> 0.2157 ms`, `35.1%` faster;
- combined indexer plus sparse decode: `1.0194 ms -> 0.8484 ms`, `16.8%`
  faster.

Macro moved only slightly:

- 4096/128/batch4: `33.97 -> 34.14 output tok/s`;
- 4096/1024/batch4: `54.64 -> 55.05 output tok/s`.

Therefore global topk/lens should stay as a useful opt-in exact cut, but it is
not the main gap.  The next dominant boundary is sparse decode.

From TARGET 07.393, T=4/history=4096/page size 256:

- mini direct bf16 sparse attention: about `0.578 ms`;
- vLLM packed-cache gather/dequant plus split-K decode: about `0.226 ms`;
- vLLM split-K core alone: about `0.132 ms`, but this is not a fair full
  boundary comparison because gather/dequant is required;
- mini combined indexer plus sparse decode after 07.394 remains about
  `0.848 ms`.

Important interpretation: vLLM's measured path uses packed `fp8_ds_mla` cache,
so direct code porting would change mini's default precision/layout.  This
target should adapt the boundary design, not silently adopt the precision lane.

## Scope

In scope:

- create `performance_milestones/target07_bf16_sparse_decode_splitk/`;
- re-run or reuse the 07.394 microbench as the before baseline with
  `MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS=1`;
- define the exact sparse decode boundary in mini:
  - inputs: q, SWA cache/indices/lens, optional C4 or C128 cache/indices/lens,
    attention sink, scale;
  - output: same shape and dtype as `dsv4_sparse_attention_two_source_bf16`;
  - semantics: match the current exact bf16 path within an explicit tolerance;
- implement one feature-flagged prototype first:
  - preferred cut A: bf16 two-scope gather/mask into a merged buffer, followed
    by a split-K decode core;
  - fallback cut B: split-K sparse decode that reads mini's bf16 SWA/C4/C128
    cache layout directly, if gather/mask overhead makes cut A unattractive;
- keep the old sparse attention path available for A/B testing and fallback;
- add explicit benchmark variant wiring for the new path;
- validate graph capture safety under TP8/page size 256.

Out of scope:

- changing mini's default cache precision to `fp8_ds_mla`;
- enabling FP8 or FP4 indexer cache as default;
- porting vLLM sparse prefill reference code;
- more global topk/lens polish unless required by the sparse boundary;
- MoE/Marlin work;
- broad graph/runtime rewrites before the sparse decode boundary is selected;
- making a non-exact precision lane look like an exact-path win.

## Implementation Guidance

Start with the smallest exact cut that can be measured in isolation.

Recommended order:

1. Establish before numbers.
   - Use the 07.394 global-topk variant as the baseline.
   - Record sparse-only, indexer+sparse, and macro values before code changes
     in the new milestone directory.

2. Inspect vLLM's exact sm80 boundary.
   - In `deepseek_v4_attention.py`, study the flow:
     `compute_global_topk_indices_and_lens` ->
     `gather_dequant_two_scopes_with_mask` ->
     `_dsv4_sm80_sparse_attn_decode_triton`.
   - In `cache_utils.py`, study how the gather/mask kernel constructs a merged
     `(B, topk_total, head_dim)` bf16 buffer plus invalid mask.
   - Record which parts are portable to bf16-flat mini and which are tied to
     packed FP8 cache.

3. Prototype exact bf16 gather/mask plus split-K.
   - Preserve mini's bf16 cache tensors and page-table/index semantics.
   - Do not require a new cache allocator or packed cache format.
   - Avoid `torch.cat`, eager torch masking, and replay-time shape-dependent
     allocations in the graph path.
   - If the gather buffer is persistent or graph-owned, document its ownership
     and maximum shape.

4. Add a guarded runtime path.
   - Use a feature flag such as
     `MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16=1`.
   - Add a variant name that combines the current best exact stack:
     Marlin WNA16, global topk/lens, graph replay, and bf16 split-K sparse
     decode.
   - During CUDA graph capture, do not silently fall back to eager torch or the
     legacy kernel when the opt-in split-K path is requested.  Either use the
     requested graph-safe kernel or raise clearly.

5. Correctness first, then performance.
   - Compare against the existing `dsv4_sparse_attention_two_source_bf16`
     output on synthetic SWA-only, C4, C128, empty, short-history, and mixed
     valid-length cases.
   - Run TP8 text smoke with page size 256 before macro benchmarks.

## Validation

Required after any runtime/kernel change:

- focused unit tests for the new wrapper/kernel path;
- synthetic correctness comparisons against the legacy bf16 sparse attention
  path;
- before/after microbench at T=4, history=4096, page size=256:
  - sparse-only boundary;
  - combined indexer plus sparse decode with global topk/lens enabled;
- TP8 text smoke, page size 256;
- 4096/128/batch4 macro using the Marlin WNA16 + global topk/lens exact stack;
- 4096/1024/batch4 macro if 4096/128 improves or if decode graph replay is
  touched broadly;
- Nsight or equivalent profile if macro improves by at least `5%`.

Suggested macro shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <new-marlin-globaltopk-splitk-variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target07395_splitk_4096x128_bs4 \
  --keep-going
```

Run the 4096/1024 version only after the short macro or subgraph result clears
the stop conditions below.

## Success Criteria

Keep the new path as an exact candidate only if all correctness checks pass and
at least one of these is true:

- sparse decode boundary improves by at least `20%`;
- combined indexer plus sparse decode improves by at least `20%`;
- 4096/128/batch4 macro improves by at least `5%`.

If the sparse subgraph improves but macro does not, capture a short profile and
write down the new bottleneck ranking before continuing.

## Stop Conditions

Stop this target instead of continuing local polish if:

- the first exact bf16 split-K/gather prototype does not beat the legacy sparse
  boundary by at least `20%`;
- cut A is slow because bf16 gather/mask materialization dominates, and cut B
  also fails to clear the threshold;
- the implementation requires changing mini's cache layout or dtype to packed
  FP8 to look competitive;
- macro improvement is below `5%` after a real sparse subgraph win and the new
  profile shows another bottleneck has taken over;
- graph capture requires silent fallback or shape-dependent eager allocations.

If stopped because exact bf16 cache layout cannot close the sparse/cache gap,
write a separate opt-in precision/cache target for packed `fp8_ds_mla` KV cache
and FP8/FP4 indexer experiments.  Do not change the exact default inside this
target.

## Expected Outputs

Create:

- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/scripts/`
- `performance_milestones/target07_bf16_sparse_decode_splitk/raw/`
- `performance_milestones/target07_bf16_sparse_decode_splitk/summaries/`

The README must include:

- exact implementation cut chosen: gather+split-K, direct split-K, or both;
- mini vs vLLM boundary comparison after the change;
- before/after microbench table;
- TP8 text smoke result;
- macro result and graph replay counters;
- whether this target supports continuing exact bf16 work or opening the
  opt-in precision/cache lane next.

Update after completion:

- `prompts/target.md`
- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
