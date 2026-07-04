# TARGET 08.22.1: DSV4 Route B Component Mapping Lifecycle Fix

## Status

Active next TARGET 08 subtarget.

Run this after the first TARGET 08.22 final prefix promotion gate.  That gate
was blocked by a Route B serving correctness crash before promotion could be
evaluated.

Do not run TARGET 08.23 independent SWA ownership yet.  The 08.22 blocker is a
component mapping lifecycle bug, not proof that independent SWA ownership is
required.

## Goal

Fix the Route B component mapping lifecycle bug exposed by the final prefix
promotion gate, then run a focused rerun that proves TARGET 08.22 can be
attempted again.

The crash to fix:

```text
RuntimeError: DSV4 component mapping is missing for active C4 full pages
```

Observed stack:

```text
CacheManager.cache_req(...)
  -> DeepSeekV4KVCache.make_component_page_handles(...)
```

First failing scenario:

```text
prefix_full_hit_512_bs4
```

Other failed Route B scenarios from TARGET 08.22:

- `prefix_full_hit_513_bs4`
- `prefix_full_hit_768_bs4`
- `prefix_full_hit_769_bs4`
- `prefix_full_hit_513_longout_bs4`
- `prefix_partial_hit_769_bs8`
- `prefix_mixed_hit_miss_bs16`
- `prefix_multi_112req_wave16`
- `prefix_eviction_pressure_96req_wave16`

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate/README.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate/summaries/serving_ab.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate/summaries/swa_tail_guard_actual_impact.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate/summaries/swa_tail_guard_workload_frequency.md`
- `performance_milestones/target08_route_b_graph_deforest_serving/README.md`
- `performance_milestones/target08_independent_compressed_indexer_ownership/README.md`
- `performance_milestones/target08_compression_state_ownership/README.md`

Core code:

- `python/minisgl/scheduler/cache.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/scheduler/prefill.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `tests/core/test_deepseek_v4_kvcache.py`
- `tests/attention/test_deepseek_v4_backend_metadata.py`
- `tests/benchmark/test_deepseek_v4_perf_matrix.py`

## Current Diagnosis

The failure is probably caused by the order and scope of component handle
creation in `CacheManager.cache_req()`.

Current Route B behavior:

```text
page_indices = page_table[req.table_idx, :req.cached_len]
component_pages = kv_cache.make_component_page_handles(page_indices[:insert_len])
cached_len, new_handle = prefix_cache.insert_prefix(..., dsv4_component_pages=component_pages)
```

This builds component handles before `insert_prefix()` reports which part is
already present in the radix tree and which part is newly inserted.

That is unsafe in serving reuse cases because `page_indices[:req.cached_len]`
can contain a mixture of:

- active pages allocated for this request;
- pages copied from a matched radix handle;
- tombstoned `-1` pages for old full/SWA heads;
- full pages whose active full-to-component mapping has already been cleared
  because the component pages are owned by a radix node;
- pages that `insert_prefix()` will classify as already cached, not newly
  inserted.

The fact that `prefix_full_hit_513_bs4` and `prefix_full_hit_769_bs4` also fail
is important: those are not exact page-multiple SWA-tail guard cases.  This is
therefore not just the SWA-tail guard behaving conservatively.

## Scope

Allowed:

- refactor `CacheManager.cache_req()` and/or `RadixPrefixCache.insert_prefix()`
  so component handles are created only for pages that need new component
  ownership;
- add helper APIs to build Route B handles from active full pages plus existing
  radix component handles if that is cleaner;
- add unit probes for multi-page retained prefixes, tombstoned heads, and
  repeated insertion/reuse;
- add small diagnostic reporting for missing component mappings;
- rerun focused CPU tests, targeted TP8 scenarios, and then a reduced 08.22
  rerun.

Not allowed:

- independent SWA ownership;
- SWA KV reconstruction;
- decode metadata deforest port;
- low-precision work;
- default promotion;
- broad attention-kernel optimization;
- treating 08.22 as passed without rerunning the final promotion gate after the
  fix.

## Required Fix Direction

Before editing, write a short local design note under the milestone directory
describing the chosen fix.

The fix should satisfy these invariants:

1. Existing radix component handles remain the owner for already-cached prefix
   pages.
2. New component handles are created only from active full pages that still have
   valid full-to-component mappings.
3. Tombstoned `-1` full/SWA pages must never be passed to
   `make_component_page_handles()` as active full pages requiring mappings.
4. If a request reuses a retained prefix and inserts a suffix, the retained
   prefix uses old radix handles and the suffix uses newly allocated component
   handles.
5. `insert_prefix()` should not need component mappings for pages it will not
   insert.
6. Exact page-multiple SWA-tail guard may shorten hits, but it must not crash.
7. Non-multiple controls such as `513` and `769` should recover Route B hits
   after the lifecycle bug is fixed.

Implementation options to consider:

- change `RadixPrefixCache.insert_prefix()` to accept component handles for only
  the newly inserted suffix rather than the entire inserted length;
- or split `cache_req()` into:
  1. tree walk / insert boundary discovery;
  2. component handle creation for the new segment only;
  3. radix node creation with those new handles;
- or build a merged handle from existing radix component handles plus active
  suffix handles, but only if ownership/refcount rules stay clear.

Prefer the smallest correct change that keeps phase-1 behavior unchanged.

## Required Tests

Add or extend CPU tests covering:

- `prefix_full_hit_257` still passes;
- exact page-multiple `512` does not crash and obeys the current SWA-tail guard;
- neighboring `513` gets a safe Route B hit instead of crashing;
- `768` does not crash and obeys the guard;
- neighboring `769` gets a safe Route B hit instead of crashing;
- partial hit with a retained prefix plus long suffix;
- mixed hit/miss batch;
- multi-prefix sustained reuse;
- repeated hit/evict;
- component mappings are not required for tombstoned old full heads;
- no stale reads, double frees, negative refcounts, or leaks.

At minimum, run:

```bash
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

## Required TP8 Rerun

Run a focused Route B TP8 rerun before attempting the full 08.22 gate again.

Required scenarios:

- `prefix_full_hit_257_bs4`
- `prefix_full_hit_512_bs4`
- `prefix_full_hit_513_bs4`
- `prefix_full_hit_768_bs4`
- `prefix_full_hit_769_bs4`
- `prefix_full_hit_513_longout_bs4`
- `prefix_partial_hit_769_bs8`
- `prefix_mixed_hit_miss_bs16`
- `prefix_multi_112req_wave16`

Use:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --scenarios <focused scenarios> \
  --output-dir performance_milestones/target08_route_b_component_mapping_lifecycle_fix/raw/focused_route_b_graph \
  --keep-going
```

Also run a Route B graph text smoke if the focused scenarios pass.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_component_mapping_lifecycle_fix/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- root-cause explanation;
- before/after failure table;
- CPU test coverage;
- focused TP8 scenario table;
- graph replay/eager table;
- SWA-tail guard behavior after the fix;
- capacity metrics sanity;
- final decision:
  - rerun TARGET 08.22 final promotion gate;
  - continue debugging lifecycle;
  - or reject Route B if the lifecycle model is fundamentally wrong.

## Decision Rules

Proceed to rerun TARGET 08.22 if:

- the component mapping crash is fixed;
- `512/768` no longer crash and obey SWA-tail guard behavior;
- `513/769` recover safe Route B hits;
- focused TP8 Route B graph scenarios pass;
- graph replay still covers `[1,2,4,8,16]` with no material eager fallback;
- component/state ownership leak and double-free checks pass.

Do not proceed to TARGET 08.23 unless:

- the lifecycle bug is fixed;
- the rerun TARGET 08.22 still shows SWA-tail guard is a material serving
  bottleneck.

Stop and report blocked if:

- component handle ownership cannot be made consistent across radix insert,
  match, split, eviction, and request finish;
- fixing the crash requires broad radix-cache rewrite beyond this target;
- non-multiple controls still crash or lose hits after the lifecycle fix.
