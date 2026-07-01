# TARGET 07.395: Exact BF16 Sparse Decode Split-K

## Result

Implemented an opt-in exact bf16 sparse decode prototype:

- flag: `MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16=1`
- variant: `v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`
- default precision policy unchanged: mini still uses bf16 activation/cache for the sparse attention boundary, with no packed `fp8_ds_mla` KV cache and no FP8/FP4 indexer cache as default.

The first bf16 gather/mask plus split-K decode prototype cleared the stop bar:

| Boundary | Legacy bf16 | Split-K bf16 | Speedup | Delta |
| --- | ---: | ---: | ---: | ---: |
| Sparse-only decode, T=4/H=4096 | 0.5768 ms | 0.2284 ms | 2.53x | +60.4% |
| Global-topk + indexer + sparse decode, T=4/H=4096 | 0.7890 ms | 0.4350 ms | 1.81x | +44.9% |

Macro also moved substantially:

| Workload | 07.394 best exact output tok/s | 07.395 split-K output tok/s | Delta | Decode tok/s | Graph replay |
| --- | ---: | ---: | ---: | ---: | --- |
| 4096/128/bs4 | 34.1406 | 38.9379 | +14.05% | 79.5257 | 254, eager 0 |
| 4096/1024/bs4 | 55.0500 | 68.8097 | +24.99% | 80.0571 | 2046, eager 0 |

Conclusion: mini can get a real exact-bf16 sparse decode win from the vLLM-style gather/mask plus split-K boundary without adopting packed FP8 KV cache. It does not close the full vLLM gap: after this target, the fresh vLLM `deepseek_v4_fp8` line is still about 2.11x faster at 4096/128 and 2.94x faster at 4096/1024. The remaining gap should be treated as a separate cache/indexer/runtime and opt-in precision-lane question, not more global-topk polish.

## vLLM Mapping

Reference paths inspected:

- `compute_global_topk_indices_and_lens`
- `gather_dequant_two_scopes_with_mask`
- `_dsv4_sm80_sparse_attn_decode_triton`

Portable/adapted pieces:

- global topk/lens metadata consolidation from 07.394 is reused;
- two-scope gather/mask shape is adapted for mini's bf16 flat C4/C128 and SWA caches;
- split-K sparse decode is adapted as mini-owned Triton split and combine kernels;
- invalid or short-history rows are handled by a gathered mask, then the split-K combine performs log-sum-exp reduction across splits.

FP8-bound or rejected pieces:

- vLLM's packed `fp8_ds_mla` KV layout, scales, and dequantized gather are not ported;
- vLLM's FP8 indexer/cache precision lane is not introduced;
- vLLM's sm80 sparse prefill reference path is not ported as a default path.

Mini-specific choice:

- the new split-K path is decode-only (`metadata.max_seqlen_q <= 1`);
- prefill/extend keeps the legacy exact bf16 sparse kernel;
- the original `dsv4_sparse_attention_two_source_bf16` path remains available for A/B and rollback;
- when the opt-in split-K wrapper is selected but unsupported, it raises an explicit error instead of silently falling back to eager torch or the legacy sparse kernel.

One failed intermediate cut tried to use the split-K gather path for prefill as well. The 4096/128 macro failed during warmup with illegal memory access because materializing the prompt-side sparse gather is not the intended decode boundary. The decode-only gate is therefore part of the final design, not just a safety patch.

## Implementation

Primary code changes:

- `python/minisgl/kernel/triton/deepseek_v4.py`
  - `_sparse_bf16_gather_with_mask_kernel`
  - `_sparse_splitk_bf16_split_kernel`
  - `_sparse_splitk_bf16_combine_kernel`
  - `_gather_scope_bf16`
  - `sparse_attention_splitk_bf16`
- `python/minisgl/kernel/deepseek_v4.py`
  - `DSV4_SM80_SPARSE_SPLITK_BF16_TOGGLE`
  - `dsv4_sparse_attention_two_source_splitk_bf16`
- `python/minisgl/attention/deepseek_v4.py`
  - opt-in decode-only split-K dispatch before the legacy bf16 path
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - split-K bf16 macro variant
- `benchmark/offline/deepseek_v4_text_smoke.py`
  - split-K bf16 text-smoke variant

The microbench script is:

- `performance_milestones/target07_bf16_sparse_decode_splitk/scripts/mini_sparse_splitk_microbench.py`

The Nsight helper is:

- `performance_milestones/target07_bf16_sparse_decode_splitk/scripts/nsys_splitk_4096x128_bs4.sh`

## Correctness

Focused tests:

```text
pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sparse_attention_two_source_bf16_matches_reference \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sparse_attention_splitk_bf16_matches_legacy_cases \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_opt_in_kernels_match_fallbacks \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_records_marlin_wna16_globaltopk_splitk \
  tests/benchmark/test_deepseek_v4_text_smoke.py::test_configure_variant_sets_marlin_wna16_globaltopk_splitk
```

Result: `10 passed`.

Synthetic sparse attention cases covered against the legacy bf16 sparse attention kernel:

- SWA-only
- C4-like compressed pages
- C128-like compressed pages
- empty
- short history
- mixed valid lengths

The microbench correctness check also passed:

- sparse-only max abs error: `0.001953125`
- combined indexer+sparse max abs error: `0.001953125`
- both allclose at `7e-2`

TP8 text smoke:

- page size: `256`
- status: pass
- outputs sane for all 3 prompts
- captured CUDA graph sizes: `[4, 2, 1]`
- graph replay count: `9`
- greedy replay count: `9`
- `eager_decode_count=0`

## Benchmarks

Before sparse-only baseline:

- raw: `raw/before_sparse_attention_two_source_t4_h4096.json`
- legacy direct kernel: `0.5563 ms`
- legacy wrapper with length scan: `0.5758 ms`

Before global topk/lens combined baseline:

- raw: `raw/before_globaltopk_combined_t4_h4096.json`
- confirms the 07.394 global-topk/lens stack used for the before line.

After split-K microbench:

- raw: `raw/after_sparse_splitk_microbench_t4_h4096.json`
- sparse-only: `0.5768 -> 0.2284 ms`
- combined indexer+sparse: `0.7890 -> 0.4350 ms`

Macro:

- 4096/128 raw: `raw/dsv4_target07395_splitk_4096x128_bs4_np128_retry/summary.json`
- 4096/1024 raw: `raw/dsv4_target07395_splitk_4096x1024_bs4_np128/summary.json`
- both runs used TP8, page size 256, graph sizes `[1,2,4]`, and `num_pages=128`.

Text smoke:

- raw: `raw/tp8_text_smoke_splitkbf16_globaltopk_marlin_wna16.json`

## Nsight

Captured 4096/128/bs4 after the macro exceeded the 5% bar:

- raw report: `raw/nsys_target07395_splitk_4096x128_bs4_np128_rank0.nsys-rep`
- sqlite: `raw/nsys_target07395_splitk_4096x128_bs4_np128_rank0.sqlite`
- summary: `summaries/nsys_splitk_4096x128_bs4_np128_rank0_summary.json`
- macro-under-nsys summary: `summaries/macro_splitk_4096x128_bs4_np128_nsys_summary.json`

Macro under Nsight:

- output tok/s: `36.9610`
- decode tok/s: `77.5317`
- replay count: `254`
- `eager_decode_count=0`

Rank0 kernel categories from the current classifier:

| Category | Kernel time | Share |
| --- | ---: | ---: |
| sparse_attention | 4.2167 s | 32.18% |
| runtime_memcpy_allocation_kernels | 2.7816 s | 21.23% |
| indexer_cache | 1.9385 s | 14.80% |
| other | 1.3090 s | 9.99% |
| moe_route_w13_swiglu_w2_sum | 0.8944 s | 6.83% |
| hc_rmsnorm_logits_sampling | 0.8813 s | 6.73% |
| dense_linear_other | 0.7534 s | 5.75% |
| nccl | 0.3274 s | 2.50% |

Profile caveat: the summary classifier still reports the legacy CUDA `sparse_attention_kernel` as the top named kernel. That is expected to include prefill/legacy sparse work, and the current classifier is not yet precise enough to isolate all decode split-K Triton kernels inside the CUDA graph. The performance data proves the decode boundary win, but the next profile pass should improve kernel labels and separate prefill legacy sparse time from decode split-K time.

## Decision

The exact bf16 route remains viable for sparse decode boundary work. The next exact-path effort should not polish topk again. It should either:

- profile and reduce the residual sparse/prefill plus runtime/memcpy/indexer cluster with better graph-node attribution; or
- open a separate opt-in precision/cache target for packed FP8/FP4 cache and indexer experiments if the objective becomes matching vLLM's full `deepseek_v4_fp8` lane.

This target did not require a direct split-K kernel over the original bf16 cache because the first gather/mask plus split-K prototype already exceeded the 20% sparse-boundary stop condition.
