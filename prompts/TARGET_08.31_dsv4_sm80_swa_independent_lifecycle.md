# TARGET 08.31: DSV4 SM80 SGLang-Aligned SWA Independent Lifecycle

## Status

Run this after TARGET 09.45 and before reopening TARGET 09.5.

This is a TARGET 08 follow-up, not a low-precision target.  TARGET 09.45 showed
that current mini over-retains SWA as a full 43-layer, 128-page BF16 pool.  That
makes SWA-only FP8 look useful, but SGLang-style independent SWA lifecycle may
recover most of the same memory without changing cache precision.

The goal is to prove the real SWA tail occupancy and memory headroom first.
Only after this target should TARGET 09.5 decide whether FP8 cache is still
worth implementing.

## Goal

Implement and evaluate an opt-in, SGLang-aligned independent SWA lifecycle for
DeepSeek V4 on A100/sm80:

- give SWA KV its own ownership/lifecycle instead of pinning all historical SWA
  rows through full-token page retention;
- tombstone/free out-of-window SWA independently from C4, C128, indexer, and
  compression-state components;
- preserve Route B component loc ownership, component page-table lifetime
  caching, and direct graph metadata buffer behavior;
- add runtime counters that prove how many SWA tail pages remain in real
  prefix/serving workloads;
- decide whether TARGET 09.5 should remain deferred, proceed as SWA-only FP8,
  or be rewritten as a broader source-aligned MLA/indexer FP8 plan.

## Non-Goals

- Do not implement FP8 KV/cache in this target.
- Do not implement INT8 MoE or any quantized communication path.
- Do not rewrite attention kernels unless a minimal metadata contract requires
  it.
- Do not promote prefix cache to default for all traffic.
- Do not replace the radix cache or Route B component ownership model.
- Do not add large hit-time materialization copies as the production route.
  A small oracle is allowed only if it helps prove correctness.

## Baseline

Use the post-TARGET-10 promoted prefix/communication baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL enabled by default for this preset
Default DSV4 sm80 PyNCCL max buffer size: 32M unless overridden
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Suggested opt-in surface for this target:

```bash
--enable-dsv4-swa-independent-lifecycle
```

or, if the project style makes an env gate easier:

```bash
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
```

Pick one canonical spelling before writing final docs.

## Background Evidence

TARGET 09.45 conclusion:

- current mini SWA-only FP8 appears to save about `0.576 GiB/rank` because SWA
  is allocated as a full 43-layer, 128-page BF16 pool;
- with SGLang-aligned lifecycle, long-prefix and shared-prefix workloads may
  need only about `4` to `16` SWA tail pages, not 128 historical SWA pages;
- in that model SWA-only FP8 drops to about `0.018` to `0.072 GiB/rank` for the
  long/prefix workloads that motivated prefix retention;
- lifecycle + BF16 gives a larger and lower-risk capacity win than SWA-only
  FP8, with the canonical wave16 estimate around `1.127 GiB/rank` persistent
  cache and `+1.176 GiB/rank` headroom versus current mini's promoted formula;
- TARGET 09.4's separated FP8 kernels were slower by about `+0.016` to
  `+0.018 ms` per cache boundary, so FP8 should not be the next step unless the
  memory win remains real after lifecycle fixes.

Prior TARGET 08 evidence:

- phase-1 prefix cache kept full-token pages canonical for correctness;
- Route B split C4/C128/indexer/state ownership from full pages;
- compression state and SWA ownership are distinct;
- SWA remains protected by a conservative full/SWA live-tail rule;
- `prompt_len=256 -> 0` fixed-point loss was classified as an SWA ownership
  limitation, not a compression-state limitation;
- component page-table lifetime caching is now the promoted prefix metadata
  route and must not be regressed.

## Source References

Mini:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/attention/deepseek_v4.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/common.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py`

Prior reports:

- `performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/README.md`
- `performance_milestones/target09_minimal_fp8_kv_cache_slice/README.md`
- `performance_milestones/target08_radix_prefix_dsv4/DESIGN.md`
- `performance_milestones/target08_component_loc_table_preflight/DESIGN.md`
- `performance_milestones/target08_compression_state_ownership/DESIGN.md`
- `performance_milestones/target08_route_b_lifetime_cache_promotion_gate/README.md`
- `performance_milestones/target08_post_prefix_reprofile/README.md`

## Required Work

### 1. Source Parity Map

First map SGLang's mature mechanism before designing local behavior:

- `SWATokenToKVPoolAllocator` full allocator, SWA allocator, and
  `full_to_swa_index_mapping`;
- `SWAComponent` value storage, tombstone behavior, lock/ref lifecycle, and
  validator role;
- `free_swa_out_of_window_slots()` page-aligned freeing rule;
- DSV4 pool separation among `swa_kv_pool`, C4/C128/indexer pools, and
  compression-state pools;
- how SGLang keeps C4/C128/indexer/state valid after SWA is tombstoned.

Then map mini's current lifecycle:

- full-token page table;
- Route B component page handles;
- component loc lifetime cache;
- prefix insertion/match/eviction;
- DSV4 attention metadata consumption;
- graph replay metadata buffers.

The deliverable should explicitly say which SGLang mechanisms are adapted
directly, which are intentionally simplified, and which are left out.

### 2. Ownership Design

Design the minimal independent SWA ownership model for mini:

- SWA page/slot owner separate from full-token page owner;
- full-to-SWA mapping or equivalent page-aligned tail table;
- tail allocation/free respecting `sliding_window=128` and `page_size=256`;
- tombstone state for out-of-window SWA;
- no invalidation of C4/C128/indexer/state component locs when SWA is freed;
- prefix node/cache metadata carries enough SWA component handles, version
  guards, or validity markers to avoid stale rows;
- Route B component page-table lifetime cache remains correct when SWA rows are
  tombstoned.

Important rule:

```text
C4, C128, indexer, and compression-state locations must never be derived from a
full/SWA row after that row may have been released or tombstoned.
```

### 3. Implementation

Implement behind the chosen opt-in only.

Likely touch points, to be verified from source:

- `DeepSeekV4KVCache` allocation/free paths;
- `DSV4ComponentPageHandles` or an adjacent SWA handle structure;
- `on_pages_allocated()` and `on_token_indices_freed()`;
- prefix insertion/match/lock/unlock/eviction;
- component loc table/lifetime-cache construction;
- attention metadata builders that consume SWA page/loc tables;
- graph metadata copy/replay hooks.

Use a conservative implementation first:

- no broad attention kernel rewrite;
- no runtime FP8/INT8;
- no large hit-time materialization copies;
- no dynamic CUDA allocation in graph replay.

If a small Route-A-style materialization oracle is useful, keep it CPU-only or
debug-only and document why it is not the production path.

### 4. Runtime Counters

Add or extend reports so macro output can show:

- current SWA tail pages;
- SWA pages allocated;
- SWA pages tombstoned/freed;
- SWA pages protected by active decode;
- SWA pages protected by retained prefix nodes;
- retained C4/C128/indexer/state pages;
- component-safe prefix hit length;
- saved prefill tokens;
- prefix evictions and evicted component pages;
- estimated persistent GiB/rank before and after lifecycle;
- equivalent current-page and token headroom.

Counters must distinguish source-derived estimates from runtime-proven values.

### 5. Correctness Gates

Run at least:

- DSV4 text smoke with prefix disabled, prefix enabled, and lifecycle opt-in;
- prefix hit/remap/eviction probes;
- long-prefix and tail-heavy decode probes;
- `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1`;
- graph replay verification for buckets `1,2,4,8,16`.

Correctness expectation:

- no unreadable/garbled output;
- no crash, stale component loc, negative refcount, double-free, or leak;
- no prefix hit that uses tombstoned SWA without valid reconstruction/fallback;
- no component page-table cache mismatch;
- no graph replay dynamic allocation.

Do not use broad generated-token equality across different batch slots as a
hard oracle.  TARGET 08 already established that mini does not currently
guarantee batch-slot invariance.  Use text smoke, slot-pinned comparisons, and
metadata/component invariants instead.

### 6. Performance And Memory Gates

Run the promoted baseline and lifecycle opt-in on:

- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`;
- one higher-concurrency serving case, preferably a real wave64 scenario if the
  benchmark harness supports it, otherwise a clearly labeled synthetic estimate
  or closest available workload.

For each scenario report:

- output tok/s and latency fields already emitted by the matrix;
- prefix hit rate and saved prefill tokens;
- persistent cache GiB/rank;
- SWA tail pages and tombstoned pages;
- available component pages;
- CUDA graph zero-eager status;
- no-hit overhead versus promoted baseline.

Pass target:

- recover at least about `0.5 GiB/rank` persistent cache or equivalent
  current-page/token headroom in long-prefix or prefix-wave scenarios, or
  explain why the measured workload does not exercise over-retained SWA;
- no material macro regression on no-hit and shared-prefix controls;
- graph buckets stay zero-eager.

### 7. TARGET 09.5 Decision

End the target by choosing exactly one:

1. Keep TARGET 09.5 deferred because SWA lifecycle captures most capacity value.
2. Reopen TARGET 09.5 as SWA-only FP8 because real SWA tail occupancy remains
   high enough to save at least about `0.25 GiB/rank`.
3. Rewrite TARGET 09.5 as a broader SGLang/vLLM-aligned MLA/indexer FP8 route
   because SWA-only FP8 is too small but broader source layout is still worth
   a dedicated target.
4. Stop FP8 cache work for now and return to another TARGET 09 lane.

## Stop Conditions

Stop and report rather than continuing local polishing if:

- freeing/tombstoning SWA can invalidate C4, C128, indexer, or state component
  locs;
- prefix hit validation cannot prove a component-safe fixed point;
- graph replay requires dynamic allocation or falls back to eager;
- the implementation needs a broad attention kernel rewrite before lifecycle
  correctness can be proven;
- runtime counters show little SWA memory reduction on the long/prefix
  workloads;
- macro regressions dominate the memory benefit.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_independent_lifecycle/
```

Include:

- `README.md` with the final decision and next target recommendation;
- source parity map against SGLang;
- mini ownership design notes;
- implementation summary and opt-in surface;
- correctness/verifier results;
- graph replay results;
- memory/counter tables;
- macro benchmark comparison;
- recommendation for TARGET 09.5;
- scripts, commands, and raw outputs or symlinks.

## Suggested First Prompt

Use this target as the child-thread prompt.  Start by reading this file,
`prompts/target.md`, `prompts/TARGET_08_radix_prefix_dsv4.md`,
`prompts/TARGET_09.45_dsv4_sm80_fp8_cache_roi_sglang_lifecycle.md`, and the
TARGET 09.45 report.  Then perform the source parity map before editing code.
