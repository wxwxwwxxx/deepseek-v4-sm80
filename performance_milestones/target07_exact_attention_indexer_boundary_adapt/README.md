# TARGET 07.394: Exact Attention/Indexer Boundary Adapt

## Status

Implemented one exact bf16 cut: opt-in global topk/lens consolidation for the
mini indexer/topk boundary.

The default precision policy is unchanged: bf16 activation/cache and MXFP4
Marlin WNA16 MoE remain the exact path.  No packed `fp8_ds_mla` KV cache and no
FP8/FP4 indexer cache are enabled by default.

## What Changed

- Added `MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS`.
- Added `DSV4TopKTransformGlobalLensKernel` beside the existing local CUDA topk
  kernel.
- The new opt-in path writes raw topk indices, page/global indices, full C4
  cache indices, and per-token `topk_lens` in one JIT CUDA launch.
- `select_indexer` now carries `topk_lens` into DSV4 attention metadata when
  available.
- Sparse decode uses the carried C4 topk lengths only when the new flag is
  enabled; otherwise it keeps the old `(indices >= 0).sum()` path.
- CUDA graph capture does not silently fall back when the new opt-in path is
  requested.  If the global-topk/lens JIT path is unavailable during capture,
  the wrapper raises.
- Added explicit benchmark variants:
  `v1_moe_vllm_runner_marlin_wna16_globaltopk_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`.

## Artifacts

- Probe script:
  `scripts/mini_global_topk_lens_microbench.py`
- Raw microbench:
  `raw/mini_global_topk_lens_microbench_t4_h4096.json`
- TP8 text smoke:
  `raw/tp8_text_smoke_globaltopk_marlin_wna16.json`
- Macro raw symlinks:
  `raw/dsv4_target07394_globaltopk_4096x128_bs4_np128`,
  `raw/dsv4_target07394_globaltopk_4096x1024_bs4_np128`
- Summaries:
  `summaries/microbench_globaltopk_t4_h4096_summary.json`,
  `summaries/macro_globaltopk_4096x128_bs4_np128_summary.json`,
  `summaries/macro_globaltopk_4096x1024_bs4_np128_summary.json`

## Microbench

Shape matches the TARGET 07.393 decode-like probe: `T=4`, history `4096`,
page size `256`, C4 topk `512`.

| Boundary | Legacy mean ms | Global topk/lens mean ms | Improvement |
| --- | ---: | ---: | ---: |
| Topk full transform | 0.1838 | 0.0741 | 59.7% |
| Full bf16 indexer select | 0.3323 | 0.2157 | 35.1% |
| Combined indexer + sparse decode | 1.0194 | 0.8484 | 16.8% |

The selected subgraph clears the 20% improvement threshold.  The combined
decode boundary improves less because sparse attention remains the dominant
piece of that combined probe.

## Validation

Completed:

- Focused tests:
  - `tests/kernel/test_deepseek_v4_wrappers.py::test_topk_transform_full_reports_lens_in_torch_fallback`
  - `tests/attention/test_deepseek_v4_backend_metadata.py::test_dsv4_indexer_select_updates_c4_sparse_metadata`
  - `tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_opt_in_kernels_match_fallbacks`
  - new benchmark variant selector tests.
- TP8 text smoke, page size 256: pass.  Graph replay captured `[4,2,1]`;
  `eager_decode_count=0`.
- 4096/128/batch4 macro: pass.
- 4096/1024/batch4 macro: pass because 4096/128 had a small positive gain.

## Macro

Before values are the TARGET 07.393/post-Marlin exact Marlin WNA16 baseline.

| Workload | Before output tok/s | After output tok/s | Delta |
| --- | ---: | ---: | ---: |
| 4096/128/bs4 | 33.9733 | 34.1406 | +0.49% |
| 4096/1024/bs4 | 54.6351 | 55.0500 | +0.76% |

Both macro runs stayed graph-safe:

- 4096/128: `eager_decode_count=0`, replay count `254`.
- 4096/1024: `eager_decode_count=0`, replay count `2046`.

Nsight was not captured for this cut because macro gain is below 5% and not a
clear profile-changing result.  The profile attribution remains the TARGET
07.392/07.393 one: sparse attention, metadata/runtime/copy, and indexer/cache
are still the top cluster.

## Conclusion

Global topk/lens consolidation is a real exact-bf16 subgraph win and should stay
as an opt-in exact path.  It does not materially move end-to-end throughput by
itself.  The evidence says not to keep polishing this local topk boundary.

The next exact-path cut should be the larger sparse decode boundary: bf16
two-scope gather/mask plus split-K sparse decode, while preserving mini's bf16
cache layout.  If that still cannot approach vLLM's packed-cache gather plus
split-K result, open a separate opt-in FP8/FP4 cache/indexer target instead of
changing the default exact path.
