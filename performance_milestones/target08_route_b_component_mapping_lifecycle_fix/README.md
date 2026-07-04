# TARGET 08.22.1 DSV4 Route B Component Mapping Lifecycle Fix

Date: 2026-07-04

## Result

Decision: **rerun TARGET 08.22 final prefix promotion gate**.

The Route B component mapping crash is fixed in the focused reproduction and
the required TP8 Route B graph scenarios.  Exact page-multiple cases `512` and
`768` no longer crash and still produce zero per-scenario Route B hit delta
under the SWA-tail guard.  Neighboring non-multiple controls `513` and `769`
recover safe hits.

## Root Cause

`CacheManager.cache_req()` built `DSV4ComponentPageHandles` for the whole
page-aligned `req.cached_len` before `RadixPrefixCache.insert_prefix()` knew
which prefix pages were already present in the radix tree.

In serving, prefill can insert a prefix and Route B can release old full/SWA
head pages while their C4/C128/indexer/state components remain owned by the
radix node.  The active request page table may still contain old full page ids
until request finish.  A later `cache_req(..., finished=True)` then tried to
rebuild component handles from those stale full page ids and failed with:

```text
RuntimeError: DSV4 component mapping is missing for active C4 full pages
```

## Fix

- Added a delayed `dsv4_component_pages_builder` path to
  `RadixPrefixCache.insert_prefix()`.
- `insert_prefix()` now walks/splits the tree first, computes `prefix_len`, and
  calls the builder only for the new page-aligned suffix
  `[prefix_len, insert_len)`.
- Existing matched prefix pages continue to use component handles stored on
  radix nodes.
- `CacheManager.cache_req()` no longer asks the KV pool to build component
  handles for already cached prefix pages, tombstoned `-1` pages, or full pages
  whose active staging mapping has already been cleared.
- For Route B, when cache insertion adopts an existing radix prefix or releases
  a full/SWA head, the active request page table is synchronized to the radix
  handle's tombstone/live-tail indices.

This target did not implement independent SWA ownership, SWA reconstruction,
decode deforest, promotion, or low-precision work.

## Exact Commands

Reproduction before the fix used a serving-style CPU lifecycle: allocate a
512-token prompt, simulate prefill `cache_req(..., finished=False)`, then
finish the same request.  Before the fix it crashed at
`make_component_page_handles()`.  After the fix it completes and leaves the
first page tombstoned and the tail page live.

Validation:

```bash
pytest -q tests/core/test_deepseek_v4_kvcache.py

pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_cache_allocate.py \
  tests/core/test_scheduler.py \
  tests/engine/test_graph_runner.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py

ruff check \
  python/minisgl/scheduler/cache.py \
  python/minisgl/kvcache/radix_cache.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py

python -m py_compile \
  python/minisgl/scheduler/cache.py \
  python/minisgl/kvcache/radix_cache.py \
  python/minisgl/kvcache/deepseek_v4_pool.py
```

Focused TP8 Route B graph:

```bash
bash performance_milestones/target08_route_b_component_mapping_lifecycle_fix/scripts/run_focused_route_b_graph.sh
```

Route B graph text smoke:

```bash
bash performance_milestones/target08_route_b_component_mapping_lifecycle_fix/scripts/run_route_b_graph_text_smoke.sh
```

## Git Status Summary

Relevant files changed by this target:

```text
M python/minisgl/kvcache/radix_cache.py
M python/minisgl/scheduler/cache.py
M tests/core/test_deepseek_v4_kvcache.py
?? performance_milestones/target08_route_b_component_mapping_lifecycle_fix/
```

The workspace also still contains pre-existing TARGET 08.22 prompt and final
gate artifacts:

```text
M benchmark/offline/deepseek_v4_perf_matrix.py
M prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md
M prompts/TARGET_08_radix_prefix_dsv4.md
M prompts/target.md
M tests/benchmark/test_deepseek_v4_perf_matrix.py
?? performance_milestones/target08_route_b_final_prefix_promotion_gate/
?? prompts/TARGET_08.22.1_dsv4_sm80_route_b_component_mapping_lifecycle_fix.md
```

## CPU Test Coverage

| check | result |
| --- | --- |
| 257 / 512 / 513 / 768 / 769 serving lifecycle | pass |
| tombstoned heads not passed to component builder | pass |
| retained prefix plus suffix partial hit | pass |
| mixed hit/miss batch lifecycle | pass |
| multi-prefix sustained reuse | pass |
| repeated hit/evict | pass |
| no stale page-table head after Route B release | pass |
| no double free / negative refcount / leak after eviction | pass |

`pytest -q tests/core/test_deepseek_v4_kvcache.py`: **21 passed**.

Required wider pytest set: **69 passed**.

## Before/After Failure Table

| scenario | before TARGET 08.22 | after 08.22.1 focused TP8 |
| --- | --- | --- |
| prefix_full_hit_257_bs4 | pass | pass |
| prefix_full_hit_512_bs4 | component mapping crash | pass |
| prefix_full_hit_513_bs4 | component mapping crash | pass |
| prefix_full_hit_768_bs4 | component mapping crash | pass |
| prefix_full_hit_769_bs4 | component mapping crash | pass |
| prefix_full_hit_513_longout_bs4 | component mapping crash | pass |
| prefix_partial_hit_769_bs8 | component mapping crash | pass |
| prefix_mixed_hit_miss_bs16 | component mapping crash | pass |
| prefix_multi_112req_wave16 | component mapping crash | pass |

## Focused TP8 Route B Graph

Summary: `summaries/focused_route_b_graph.md`.

| scenario | status | saved delta | hits/matches | avg saved/hit | replay/eager |
| --- | --- | ---: | --- | ---: | --- |
| prefix_full_hit_257_bs4 | pass | 768 | 3/4 | 256 | 6/0 |
| prefix_full_hit_512_bs4 | pass | 0 | 0/4 |  | 6/0 |
| prefix_full_hit_513_bs4 | pass | 1536 | 3/4 | 512 | 6/0 |
| prefix_full_hit_768_bs4 | pass | 0 | 0/4 |  | 6/0 |
| prefix_full_hit_769_bs4 | pass | 2304 | 3/4 | 768 | 6/0 |
| prefix_full_hit_513_longout_bs4 | pass | 1536 | 3/4 | 512 | 62/0 |
| prefix_partial_hit_769_bs8 | pass | 1792 | 7/8 | 256 | 14/0 |
| prefix_mixed_hit_miss_bs16 | pass | 6144 | 8/16 | 768 | 14/0 |
| prefix_multi_112req_wave16 | pass | 49152 | 96/112 | 512 | 49/0 |

Captured buckets stayed `[16, 8, 4, 2, 1]`, covering requested
`[1, 2, 4, 8, 16]`; eager decode count was `0` in every focused scenario.

## SWA-Tail Guard

Per-scenario deltas confirm the intended guard behavior:

| prompt length | Route B saved delta | interpretation |
| ---: | ---: | --- |
| 257 | 768 | safe one-page hit |
| 512 | 0 | exact page multiple shortened by guard |
| 513 | 1536 | safe two-page hit recovered |
| 768 | 0 | exact page multiple shortened by guard |
| 769 | 2304 | safe three-page hit recovered |

## Capacity Metrics Sanity

The focused multi-prefix run retained 68 radix pages and reported Route B
component ownership as enabled.  Rank0 final ownership counters were:

```text
live_full_pages=30
live_full_slots=7680
live_c4_slots=4352
live_c128_slots=136
live_c4_indexer_slots=4352
live_c4_state_slots=544
live_c128_state_slots=8704
live_c4_indexer_state_slots=544
available_component_pages=61
evictable_component_tokens=17408
evictable_live_full_tokens=7680
```

The CPU eviction tests additionally assert no leak after all retained prefixes
are evicted.

## Text Smoke

Route B graph text smoke status: **pass**.

Graph replay/eager from the smoke config: `5/0`; captured buckets:
`[16, 8, 4, 2, 1]`.

Outputs were short, printable, and sane for the shared-prefix Chinese city,
shared-prefix Chinese province, and English sky-color prompts.

## Final Decision

Proceed to rerun `TARGET 08.22` final prefix promotion gate.  The focused
lifecycle blocker is cleared, graph replay still covers `[1, 2, 4, 8, 16]`, and
the exact-multiple SWA-tail guard behavior is now separated from the component
mapping lifecycle bug.
