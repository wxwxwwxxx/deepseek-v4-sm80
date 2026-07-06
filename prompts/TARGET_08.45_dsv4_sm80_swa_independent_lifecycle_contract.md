# TARGET 08.45: DSV4 SM80 SWA Independent Lifecycle Contract

## Status

Active TARGET 08 design/contract target after TARGET 08.44.

TARGET 08.31 introduced opt-in SWA independent lifecycle.  TARGET 08.42 fixed
the Engine/KV-cache dummy full-token sentinel.  TARGET 08.43 found a stale
prefix-handle SWA double-free.  TARGET 08.44 fixed that no-weight/core
double-free path, but the full-model gate then hit CUDA illegal memory access
in the first fixed128 SWA independent macro case.

This pattern suggests the SWA independent lifecycle ownership model is not yet
fully formalized.  Do not keep adding isolated patches before writing the
contract that every radix/cache/metadata/kernel path must obey.

## Goal

Write the authoritative mini-sglang DeepSeek V4 SWA independent lifecycle
contract, aligned with SGLang where possible and adapted to mini's current
Route B component-ownership implementation.

Primary deliverable:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
```

This document must become the first reference for future TARGET 08 SWA threads.
It should define ownership, valid state transitions, release/tombstone rules,
metadata contracts, graph safety rules, and required debug/test gates.  It is
allowed to quote or paraphrase small SGLang code snippets, but the main value
should be a clear mini contract, not copied source.

This target should not fix the current CUDA illegal memory access unless a tiny
documentation-supporting assertion is unavoidable.  The implementation audit
and unified fix are separate follow-ups.

## Starting Evidence

Read first:

```text
prompts/target.md
prompts/TARGET_08_radix_prefix_dsv4.md
performance_milestones/target08_swa_independent_lifecycle/summaries/TARGET_08.31_report.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/README.md
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/README.md
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/fix_summary.md
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/full_model_long_decode_gate.md
```

Important current facts:

- 08.31 showed SWA independent lifecycle can run short fixed128 macro and has
  large auto-capacity potential with Marlin release.
- 08.42 fixed dummy full-token handling: Engine dummy token start is
  `num_tokens = planned_pages * page_size`, and DSV4 SWA translation must map
  that row to the SWA dummy page, never `-1`.
- 08.43 showed fixed128 long decode double-freed SWA page handles.
- 08.44 reproduced and fixed the stale prefix-handle double-free in the
  no-weight/core path, while preserving the real double-free guard.
- 08.44 then hit full-model CUDA illegal memory access before the first
  fixed128 `4096/128/bs4` report was emitted.

## SGLang References

Use SGLang as the primary behavioral reference, not as a source to blindly
copy.  Read at least these files:

```text
/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py
/workspace/sglang-main/python/sglang/srt/mem_cache/common.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor_v2.py
```

SGLang details that the contract must address:

- `SWATokenToKVPoolAllocator` has separate full-attention and SWA allocators.
- `full_to_swa_index_mapping` is the full-loc to SWA-loc translation boundary.
- `free()` is explicitly not idempotent.
- `free_swa()` accepts full indices, expands to pages when paged, translates
  through `full_to_swa_index_mapping`, frees SWA slots, then clears mapping.
- `free_swa_out_of_window_slots()` is page-aligned, maintains a monotonic
  `swa_evicted_seqlen`, and frees only tokens outside both the protected cache
  region and the active sliding window.
- SGLang intentionally keeps a page margin before the radix insert boundary
  unless the special drop-margin path is enabled.
- `SWAComponent` stores translated SWA component values separately from full
  attention values.  Device tombstone means the SWA component value is absent,
  while the full/base component can remain intact.
- SGLang's DSV4 sparse/prefill utilities build kernel inputs from valid
  translated SWA token ids and lengths; tombstone/invalid state must not become
  active kernel input.

Useful short source cues to verify during reading:

```text
# allocator/swa.py
# NOTE: the API is not idempotent.

# common.py
cache_protected_len must be page aligned
free tokens not in the tree cache and not in the sliding window

# swa_component.py
value becomes None while full attention value stays intact
```

## Mini References

Read current mini code paths:

```text
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/kvcache/radix_cache.py
python/minisgl/scheduler/cache.py
python/minisgl/scheduler/scheduler.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
tests/core/test_deepseek_v4_kvcache.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/benchmark/test_deepseek_v4_text_smoke.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
```

Mini behavior that the contract must classify:

- `DeepSeekV4KVCache` owns the physical SWA buffer, `_full_to_swa_page`,
  `_swa_page_refcount`, `_free_swa_pages`, and SWA dummy page.
- `DSV4SWAPageHandles` currently stores SWA page snapshots on radix nodes.
- `RadixPrefixCache.release_dsv4_swa_out_of_window()` tombstones SWA handles
  and calls the SWA release callback.
- `CacheManager.release_active_dsv4_swa_out_of_window()` actively releases old
  SWA pages during decode.
- TARGET 08.44 added active-release synchronization into radix handle snapshots.
  The contract must decide whether this is legal, only temporarily tolerated,
  or should be replaced by request-local release accounting.
- `attention/deepseek_v4.py` builds SWA page tables and SWA indices for kernels
  from prefix handles plus active full-to-SWA mapping.
- Direct SWA graph metadata is disabled under SWA independent lifecycle, but
  graph replay still consumes SWA metadata copied through the generic path.

## Required Contract Sections

The final contract document must include these sections.

### 1. Definitions

Define the vocabulary precisely:

- full token loc and full page;
- SWA token loc and SWA page;
- C4/C128/indexer/state component locs;
- active request page table;
- full-to-SWA mapping;
- SWA component value / `DSV4SWAPageHandles`;
- radix prefix node and cache handle;
- active decode tail/window;
- protected prefix/cache region;
- tombstone;
- free/release;
- evict;
- dummy full token and SWA dummy page;
- CUDA graph metadata buffer.

### 2. Ownership Model

State who owns physical SWA lifetime.

The contract must explicitly answer:

- Is `DSV4SWAPageHandles` an owning component value, a borrowed snapshot, or
  something else?
- Does active full-to-SWA mapping own a page, or is it only a translation table?
- Which operation is allowed to decrement SWA physical refcount?
- How does a page move from active request ownership to radix/prefix ownership?
- Can active decode release a page that is still represented by a prefix node?
- Is the current 08.44 "active release tombstones radix handle immediately"
  legal under the contract, or should it become request-local release metadata
  processed at finish/insert boundary?

The preferred SGLang-aligned answer should be considered first:

- SWA physical allocation is separate from full allocation.
- Prefix/radix SWA component data represents retained SWA component state and
  must have clear lifetime separate from active decode.
- Active out-of-window release should free only SWA slots that are outside both
  the protected prefix/cache region and the active sliding window.
- Free/release is not idempotent; duplicate ownership bugs should remain
  visible through guards.

If mini intentionally differs from SGLang, document why.

### 3. State Transitions

Document legal transitions with preconditions and postconditions:

- allocation of full and SWA pages;
- prefill insert into radix;
- prefix match / handle lock;
- decode step;
- active out-of-window release;
- unfinished request caching;
- finished request caching;
- prefix SWA tombstone;
- prefix full/component eviction;
- SWA-only pressure eviction;
- dummy row handling;
- graph capture and replay.

Each transition must specify:

- which mapping/table/handle may change;
- whether physical SWA refcount/free list may change;
- whether attention metadata for the current step may still reference old
  values;
- which invariants must be checked before/after the transition.

### 4. Protected Frontier Rule

Formalize the SGLang-inspired protected frontier:

```text
active_swa_free_until <= min(prefix/cache protected frontier, active window frontier)
```

The exact mini formula may differ, but the document must settle:

- whether mini has an equivalent of SGLang `cache_protected_len`;
- whether mini needs a one-page margin before the radix insert boundary;
- how `req.cached_len`, `req.device_len`, `old_handle.cached_len`, and
  `new_handle.cached_len` participate;
- when `swa_evicted_seqlen` or an equivalent monotonic per-request frontier is
  updated;
- why the free frontier is page-aligned.

This section should explicitly evaluate whether the current mini
`release_end = align_down(req.cached_len - tail_tokens, page_size)` is too
aggressive for prefix-protected pages.

### 5. Metadata And Kernel Input Contract

Define what attention kernels are allowed to receive:

- For every position covered by `swa_topk_lengths`, `swa_page_indices` must be a
  valid physical SWA token loc, not `-1`, not dummy, not free, and not
  out-of-range.
- `-1` is allowed only outside the active length or in explicitly ignored
  padding.
- Tombstoned prefix SWA pages may not appear inside active SWA attention
  lengths.
- If metadata cannot prove a SWA page is live, it must either reconstruct from
  active full-to-SWA mapping or shorten the valid length, not pass invalid locs
  to the kernel.
- CUDA graph replay may not reuse a SWA metadata buffer whose producer-side
  ownership/version has changed.

Also specify required debug gates:

```text
MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG
optional free-page membership checks
optional graph replay metadata version checks
CUDA_LAUNCH_BLOCKING attribution mode
```

### 6. Dummy And Sentinel Contract

Preserve TARGET 08.42:

- Engine dummy full token start is `planned_pages * page_size`.
- DSV4 KV-cache must map that dummy token/page to the SWA dummy page.
- The SWA dummy page is reserved, refcount-pinned, never freed, and excluded
  from release/tombstone.
- `-1` means invalid/tombstone/padding, not dummy.

### 7. Refcount And Leak Contract

Document:

- what each refcount dimension means;
- when counts are incremented and decremented;
- what `assert_no_leak()` must prove;
- what runtime counters mean and what they do not prove;
- which duplicate release cases must still raise.

### 8. Compatibility Contract

State what must not regress:

- Route B C4/C128/indexer/state component loc ownership;
- component page-table lifetime cache;
- Marlin WNA16 release + component-slot clear;
- direct C4 graph metadata buffers;
- no dynamic CUDA allocation during graph replay;
- page size `256`;
- TP8 graph buckets `[1,2,4,8,16]`.

### 9. Test And Promotion Gates

Define the required gates for any future SWA lifecycle change:

- no-weight ownership tests;
- double-free negative guard tests;
- dummy sentinel tests;
- metadata active-range bounds tests;
- graph replay/eager counters;
- fixed128 `4096/128/bs4` and `4096/1024/bs4`;
- Marlin release + SWA independent compatibility;
- serving/prefix/eviction soak only after fixed historical gates pass.

## Deliverables

Write:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
performance_milestones/target08_swa_lifecycle_contract/README.md
performance_milestones/target08_swa_lifecycle_contract/sglang_parity_notes.md
performance_milestones/target08_swa_lifecycle_contract/mini_contract_risks.md
```

The README must answer:

1. What is the final mini SWA ownership model?
2. Which SGLang mechanisms are copied, adapted, or intentionally not used?
3. Is 08.44's active-time radix-handle tombstone legal under the new contract?
4. What exact invariants must attention metadata satisfy?
5. What should TARGET 08.46 audit first?

## Stop Conditions

Stop and report rather than widening scope if:

- the contract cannot decide whether radix SWA handles are owning values or
  non-owning snapshots;
- SGLang and mini differ in a way that requires an architecture decision;
- the current implementation appears fundamentally incompatible with the chosen
  contract;
- producing the contract requires broad code changes.
