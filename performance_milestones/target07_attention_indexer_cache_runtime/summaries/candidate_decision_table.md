# Candidate Decision Table

## Decision

Selected route: `adapt_vllm_design` for the exact bf16 path, implemented as a
follow-up target rather than a 07.393 runtime patch.

Reason: the measured upside clears the threshold, but the useful cut is a real
attention/indexer/cache boundary rework.  Direct vLLM cache/indexer code is
precision-changing (`fp8_ds_mla` + FP8 indexer cache), and a local mini polish
would ignore the dispatch mismatch this target was created to clarify.

| Candidate | Target Subgraph | Evidence | Expected Upside | Correctness Risk | Implementation Risk | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `direct_port_vllm` | vLLM `deepseek_v4_attention`, FP8 cache insert, FP8 indexer cache, split-K decode | vLLM T=4 gather+split-K decode is 0.226 ms vs mini bf16 sparse 0.578 ms, but vLLM uses packed `fp8_ds_mla`; standalone FP8 quant insert proxy fails on sm80 because the real path is engine C++ custom ops | Potentially large, but not exact; vLLM macro is 3.70x mini at 4096/1024 | High: changes KV/indexer cache precision and layout | High: custom ops, cache manager, graph metadata, engine context | Reject as exact default; move only to opt-in precision/cache or csrc-port target |
| `adapt_vllm_design` | Exact bf16 sparse decode boundary, global topk/lens mapping, graph-stable attention metadata | mini sparse attention wall share 13.28%; metadata/runtime/copy 12.27%; vLLM `compute_global_topk_indices_and_lens` is 0.0566 ms; vLLM gather+split-K boundary shows the current mini one-kernel bf16 boundary is not the only viable design | >=5% E2E plausible from sparse attention plus metadata cluster; >20% selected-subgraph improvement plausible | Medium: preserves bf16 exact cache but changes attention/indexer metadata path | Medium/high: new bf16 gather/mask or split-K kernel and backend integration | Selected, but too large for a quiet 07.393 patch; write follow-up exact-boundary target |
| `optimize_mini_existing` | Existing `dsv4_sparse_attention_two_source_bf16`, `compress_norm_rope_store_fallback`, metadata copies | mini compressed/indexer cache store probe is 0.976 ms and sparse is 0.578 ms, so local changes could help | Unknown until scoped; likely subgraph wins possible | Low/medium if exact bf16 preserved | Medium: easy to drift into local polish without matching vLLM boundary | Defer until after exact boundary design target defines which local kernels remain |
| `precision_cache_experiment` | `fp8_ds_mla` KV cache, FP8 indexer cache, vLLM-like cache insert/compressor | vLLM winning dispatch uses `deepseek_v4_fp8`, packed KV cache, and FP8 indexer cache; direct cache/indexer port is not exact | Large possible; macro gap remains 2.42x at 4096/128 and 3.70x at 4096/1024 | High relative to exact baseline; must be opt-in | High: cache allocator/layout, custom ops, validation against text quality | Recommend as separate lane after or parallel to exact-boundary work, not default here |
| `defer` | No implementation | Attention/indexer/cache/runtime remains top-two cluster after Marlin WNA16 | None | Low | Low | Do not defer; evidence is strong enough for a follow-up target |

## Selected Follow-Up Scope

The follow-up target should keep mini's exact bf16 default and adapt only the
parts that are design-portable:

- create a graph-safe global topk/lens mapping path comparable to vLLM
  `compute_global_topk_indices_and_lens`;
- prototype either a bf16 two-scope gather+mask feeding a split-K decode core or
  a split-K bf16 sparse attention kernel that preserves mini cache layout;
- reduce replay-time metadata copies only after the sparse/indexer boundary is
  graph-stable;
- validate with microbench before/after, TP8 text smoke, 4096/128 macro, and
  4096/1024 macro if the 4096/128 result improves.

## Files To Touch In Follow-Up

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/kernel/csrc/jit/` or `csrc/` for any mini-owned split-K/gather op
- `python/minisgl/engine/graph.py` only if metadata buffer ownership changes
- `performance_milestones/target07_attention_indexer_cache_runtime/` for before/after probes

## Stop Conditions

- If exact bf16 boundary adaptation does not improve the selected subgraph by
  at least 20% or macro by at least 5%, stop and reprofile.
- If the exact path cannot approach vLLM without adopting packed FP8 cache,
  stop and open the precision/cache target instead of continuing local polish.
