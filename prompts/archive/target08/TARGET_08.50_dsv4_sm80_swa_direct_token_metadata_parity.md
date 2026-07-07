# TARGET 08.50: DSV4 SM80 SWA Direct Token Metadata Parity

## Status

Active TARGET 08 follow-up after TARGET 08.49.

TARGET 08.49 landed a useful bounded opt-in:

```text
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
```

It caches independent-SWA page-table rows by request slot and invalidates on
request/prefix/window page-boundary changes.  Correctness, graph replay, and
capacity stayed clean.  The main owner improved substantially:

```text
dsv4.prepare.decode.attention_metadata
2968.232 ms -> 1381.333 ms on serving_mixed_112req_wave16
```

However, SWA independent still should not be promoted by default:

- fixed128 `serving_mixed_112req_wave16`: `-6.28%` vs non-SWA baseline;
- fixed128 `prefix_multi_112req_wave16`: `-10.34%` vs non-SWA baseline;
- auto Marlin+SWA same-Engine historical cases: about `-5.5%` vs auto non-SWA.

The SGLang census from TARGET 08.49 found the deeper mismatch: mini still
materializes and indexes a full SWA page table in decode, while SGLang is more
token-loc oriented.  SGLang builds the needed SWA token indices from raw
full-token locations and a full-to-SWA mapping, and stores radix/prefix SWA
component values already translated to SWA locs.

This target should attack that structural difference directly.

## Goal

Implement and validate a SGLang-aligned direct token-level SWA metadata path
for DSV4 independent SWA lifecycle on A100/sm80.

The desired direction is:

```text
avoid full SWA page-table materialization in decode
-> build graph/attention-consumed [rows, window] SWA token locs directly
-> use full-token loc windows + full-to-SWA mapping, or pretranslated SWA handles
-> use persistent workspace or helper kernel if needed
-> keep 08.49 page-table cache as fallback/ablation
-> preserve all 08.43 correctness/capacity gates
-> decide promote / keep opt-in / one more focused performance target
```

This is still an exact/BF16 lifecycle and metadata optimization target.  It is
not FP8 KV/cache, INT8 MoE, or an attention compute-kernel rewrite.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_metadata_page_table_perf_parity/README.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/sglang_source_census.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/mini_cost_split.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/parity_design.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/implementation_summary.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/macro_performance.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/owner_timing_after.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/capacity_regression_check.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/promotion_recommendation.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/README.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/e2e_overhead_attribution.md
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.49_dsv4_sm80_swa_metadata_page_table_perf_parity.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Key TARGET 08.49 facts:

- 08.49 page-table cache is useful and should remain available as fallback.
- Dirty-row cache achieved high reuse: `20608` clean rows and `896` dirty rows
  in the serving owner run.
- Remaining subowners include:
  - `active_full_to_swa_translation`: `637698 us`;
  - `cache_refresh_rows`: `616348 us`;
  - `row_construction`: `549738 us`;
  - `fill_missing`: `269457 us`;
  - `cache_select_rows`: `125744 us`;
  - `liveness_check`: `122610 us`.
- Remaining work is not more lifecycle debugging.  It is replacing or bypassing
  mini's page-table materialization path.

Important SGLang-derived facts from TARGET 08.49:

- SGLang independent SWA path is token-loc oriented.
- `full_to_swa_index_mapping[full] = swa` enables direct tensor gather from
  full-token locs to SWA-token locs.
- Radix/prefix SWA component values already store translated SWA locs.
- SGLang does not appear to use a decode dirty-row page-table cache; that was a
  mini workaround.
- True parity likely means building graph-consumed SWA token indices directly.

## Reference Source Paths

Mini:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/scheduler/cache.py
python/minisgl/scheduler/scheduler.py
python/minisgl/models/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/core/test_deepseek_v4_kvcache.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py
/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py
```

## Non-Goals

- Do not redesign SWA independent ownership/lifecycle.
- Do not re-debug the fixed 08.42, 08.47, or 08.48 correctness issues unless
  the exact gate regresses.
- Do not remove the 08.49 page-table cache; keep it as fallback and ablation.
- Do not implement FP8 KV/cache, INT8 MoE, or quantized communication.
- Do not rewrite C4/C128/SWA attention compute kernels unless a tiny interface
  helper is proven necessary by metadata evidence.
- Do not hide overhead by disabling graph capture, reducing graph buckets, or
  using eager.

## Required Work

### 1. Source And Shape Reconfirmation

Before implementation, write a short reconfirmation that answers:

- What tensor does mini ultimately need to feed SWA attention today?
- Which current mini function materializes the SWA page table?
- Which current mini function turns the page table into token locs?
- Can the target tensor be generated directly from:
  - full-token loc windows;
  - `DeepSeekV4KVCache` full-to-SWA mapping;
  - prefix/radix SWA handles that are already in SWA loc space?
- What are the exact shapes for decode buckets `[1,2,4,8,16]` and SWA window
  size?
- Which parts must be graph-captured static buffers and which can be refreshed
  as replay inputs?

Compare the answer against the SGLang source paths, especially the behavior
described in TARGET 08.49 `sglang_source_census.md`.

### 2. Design One Direct Path

Choose exactly one first implementation lane:

1. **Pure PyTorch direct token builder**
   - Build raw full-token loc windows directly from mini's existing request
     table / page table.
   - Translate through the full-to-SWA mapping with device gather.
   - Mask invalid entries to `-1`.
   - Avoid Python request-loop row materialization where possible.

2. **Pretranslated prefix SWA handles + active direct builder**
   - Store or preserve prefix SWA data in token-loc space.
   - Build only the active decode tail directly.
   - Concatenate or scatter into the final graph-consumed token-loc buffer.

3. **Persistent workspace direct builder**
   - Preallocate final `[max_rows, window]` token-loc buffers per graph bucket
     or backend.
   - Fill only live rows in place.
   - Avoid per-step `torch.full`, `torch.stack`, `torch.cat`, or large clones.

4. **Small helper kernel**
   - If PyTorch direct builder still spends too much time in gather/mask/fill,
     write a small Triton/CUDA helper to generate final SWA token locs from
     full loc windows and full-to-SWA mapping.
   - Include a microbench and a simple byte/element cost estimate before
     investing in a larger kernel.

Prefer the simplest lane that can remove page-table materialization.  Do not
stack all four lanes in one target.

### 3. Implementation Requirements

- Gate the new path behind an explicit opt-in until promoted.
- Suggested env/variant names:

```text
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_swadirect
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_swadirect
```

- 08.49 page-table cache remains available:

```text
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
```

- If both flags are enabled, define and document precedence explicitly.  The
  preferred behavior is that direct token metadata wins and page-table cache is
  used only if direct mode is unavailable.
- Preserve graph replay for buckets `[1,2,4,8,16]`.
- Preserve dummy full-token/SWA dummy-page behavior.
- Preserve 08.48 invariant: fused physical SWA cache stores use translated SWA
  locations, not full-token locations.
- Keep debug/liveness checks available under existing debug env flags, but keep
  clean performance runs free of debug overhead.
- Avoid new per-step CUDA allocations in the hot path.

### 4. Correctness Gates

Run focused tests first, including at least:

```text
python -m compileall -q python/minisgl/attention/deepseek_v4.py python/minisgl/kvcache/deepseek_v4_pool.py tests/attention/test_deepseek_v4_backend_metadata.py
pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or metadata or graph'
pytest -q tests/core/test_deepseek_v4_kvcache.py tests/core/test_dsv4_cache_option_guards.py tests/attention/test_deepseek_v4_backend_metadata.py tests/benchmark/test_deepseek_v4_text_smoke.py tests/benchmark/test_deepseek_v4_perf_matrix.py tests/engine/test_marlin_wna16_release_credit.py tests/models/test_deepseek_v4_forward_fallback.py
```

Then run TP8, page size `256`, graph buckets `[1,2,4,8,16]`.

Required full-model gates:

- fixed128 SWA independent text smoke;
- fixed128 Marlin release + SWA independent text smoke;
- explicit cap4096 Marlin release + SWA independent text smoke;
- auto-capacity Marlin release + SWA independent text smoke;
- same-Engine auto-capacity gate:
  `historical_4096_128_bs4 -> historical_4096_1024_bs4`;
- macro rows:
  - `historical_4096_128_bs4`;
  - `historical_4096_1024_bs4`;
  - `serving_mixed_112req_wave16`;
  - `prefix_multi_112req_wave16`;
  - `prefix_eviction_pressure_96req_wave16`.

Every pass row must show:

- sane text / no corrupted output;
- no CUDA illegal memory access;
- no NCCL watchdog;
- graph replay healthy;
- eager decode `0` for captured buckets;
- no SWA stale metadata, negative refcount, double free, or leaked active
  mapping.

### 5. Performance Gates

Compare at least four variants:

```text
non-SWA baseline
SWA independent without 08.49 cache
SWA independent with 08.49 page-table cache
SWA independent with 08.50 direct token metadata
```

Also compare Marlin release + SWA independent for cap4096 and auto same-Engine.

Use clean production-like throughput runs:

- no `CUDA_LAUNCH_BLOCKING=1`;
- no SWA bounds debug;
- no case-boundary debug;
- no owner timing in final throughput rows.

Owner timing may be used separately for attribution.

Promotion-oriented target:

| Scenario | 08.49 direct predecessor | Desired 08.50 result vs non-SWA |
| --- | ---: | ---: |
| fixed128 `serving_mixed_112req_wave16` | `-6.28%` | about `-3%` or better |
| fixed128 `prefix_multi_112req_wave16` | `-10.34%` | about `-5%` or better |
| fixed128 `historical_4096_1024_bs4` | `-5.15%` | about `-3%` or better |
| auto Marlin+SWA same-Engine `4096/1024` | about `-5.55%` | about `-3%` or better |

Owner target:

```text
dsv4.prepare.decode.attention_metadata on serving_mixed_112req_wave16
08.43 old SWA: 2968.232 ms
08.49 page-table cache: 1381.333 ms
08.50 target: materially below 1000 ms, ideally close to non-SWA 1217 ms or lower
```

If 08.50 improves owner timing but not macro E2E, stop and identify the new
owner instead of piling on metadata changes.

Capacity must not regress:

- auto Marlin release + SWA independent should remain near the 08.43/08.49
  range, roughly `6490-6567` pages at about `49.95 GiB/rank`;
- explicit cap4096 Marlin release + SWA independent must pass;
- fixed128 remains fixed128.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_direct_token_metadata_parity/
```

Required files:

- `README.md` with final verdict;
- `source_shape_reconfirmation.md`;
- `direct_token_design.md`;
- `implementation_summary.md`;
- `correctness_graph_soak.md`;
- `macro_performance.md`;
- `owner_timing_after.md`;
- `capacity_regression_check.md`;
- `promotion_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. Did the direct token metadata path eliminate or bypass full SWA page-table
   materialization?
2. How does the final path compare to SGLang's token-loc-oriented behavior?
3. Is the 08.49 page-table cache still needed for the optimized path?
4. Did correctness remain as clean as 08.43/08.49?
5. Did graph replay remain zero-eager for buckets `[1,2,4,8,16]`?
6. How much did E2E improve versus old SWA, 08.49 page-table cache, and non-SWA
   baseline?
7. Did capacity and Marlin release compatibility remain intact?
8. Should SWA independent stay opt-in, become a named high-capacity preset, or
   get one more focused target?

## Stop Conditions

Stop and report instead of continuing to patch if:

- direct token metadata cannot be made shape-compatible with current mini
  graph metadata without broader attention API redesign;
- correctness regresses in same-Engine auto, cap4096, prefix eviction, or graph
  replay;
- direct mode falls back to eager for captured buckets;
- capacity planning loses the SWA independent gain;
- owner timing shows the remaining gap has moved away from SWA metadata;
- a helper kernel becomes necessary but the microbench/roofline estimate shows
  little possible gain;
- the next step would be FP8/INT8/attention-compute work rather than metadata.

