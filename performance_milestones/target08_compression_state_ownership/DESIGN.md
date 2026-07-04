# TARGET 08.21.3 Design

Date: 2026-07-04

## Current Mini State Paths

Mini defines three DSV4 compression-state pools:

| pool | layer type | ring | current loc formula |
| --- | --- | ---: | --- |
| C4 attention state | C4 attention layers | 8 | `(swa_loc // page_size) * 8 + (swa_loc % 8)` |
| C128 attention state | C128 attention layers | 128 | `(swa_loc // page_size) * 128 + (swa_loc % 128)` |
| C4 indexer state | C4 indexer layers | 8 | `(swa_loc // page_size) * 8 + (swa_loc % 8)` |

The pools are allocated in `DeepSeekV4KVCache`, but the active mini eager path
does not consume them:

- `DSV4Compressor.forward()` calls `dsv4_kernel.compress_forward_fallback()`;
- `compress_forward_fallback()` only emits rows whose complete ratio group is
  present in the current extend tensor;
- `store_compressed()` and `store_indexer()` write explicit component locs;
- no mini call site currently calls `get_state_by_state_loc()` or
  `set_state_by_state_loc()`.

SGLang does use state pools.  Its prefill path gathers prior state through
`translate_from_swa_loc_to_state_loc()`, appends current `kv_score`, writes
post-state, and compresses complete groups.  Its decode path writes the current
token state, then gathers the previous `ratio * coff` state slots.  That is the
future mini boundary this target protects.

## Runtime Needs

Suffix prefill after a prefix hit needs:

- SWA KV for the sliding window immediately before each suffix query;
- C4/C128 attention component pages for older compressed context;
- C4 indexer component pages for sparse C4 selection;
- compression state if/when mini enables a stateful compressor path.

First decode after suffix prefill needs:

- SWA KV for recent prompt/suffix tokens;
- C4/C128/indexer component pages already written during prefill;
- compression state if/when decode compression writes generated-token
  compressed rows incrementally.

The important separation is that state ownership and SWA ownership are distinct.
B2 owns state.  It does not create a separate SWA allocator.

## SGLang Comparison

Relevant SGLang concepts:

| SGLang | Mini B2 mapping |
| --- | --- |
| `CompressStatePool.translate_from_swa_loc_to_state_loc()` | same formula in phase1; Route B adds independent state-page mapping |
| `SWATokenToKVPoolAllocator.full_to_swa_index_mapping` | not yet implemented independently in mini |
| `SWAComponent` tombstone and validator | mini keeps the live-tail guard instead |
| component `value` / `lock_ref` | `DSV4ComponentPageHandles` plus slot refcounts |
| match validator fixed point | Route B validates live SWA tail plus state availability |

## Options Considered

Independent state allocator/refcount: chosen.  It matches B1 component
ownership, needs no long replay, and makes retained state pages survive full
page reuse.

Page/C128-safe retained state: insufficient alone.  The state can be retained at
page boundaries, but suffix prefill also needs SWA KV from the matched tail.

Bounded warmup reconstruction: not used.  Current mini eager compression does
not read state, and reconstructing SWA/state from tombstoned full pages would
require keeping or replaying source KV that this target deliberately avoids.

Conservative fallback guard: retained for SWA.  State pages are independent, but
the matched boundary still needs a live full/SWA tail until mini has an
independent SWA component.

## Implementation

`DSV4ComponentPageHandles` now includes:

```text
c4_state_pages
c128_state_pages
c4_indexer_state_pages
```

`DeepSeekV4KVCache` now owns per-state free lists, full-page staging maps, and
slot refcounts:

```text
_free_c4_state_pages
_free_c128_state_pages
_free_c4_indexer_state_pages
_full_to_c4_state_page
_full_to_c128_state_page
_full_to_c4_indexer_state_page
_c4_state_refcount
_c128_state_refcount
_c4_indexer_state_refcount
```

Allocation/free rules:

1. Allocating a full page allocates C4/C128/indexer component pages and the
   matching state pages.
2. Prefix insertion captures component and state page ids into the radix node.
3. Releasing old full/SWA heads with `free_components=False` clears active
   full-to-component/state maps but keeps retained component/state refs.
4. Radix eviction releases component and state handles together.
5. Integrity checks reject negative state refcounts and duplicate state pages in
   free lists.

The helper `state_locs_from_full_locs()` exposes the old and Route B state loc
translation for probes and future compressor integration.

## Fixed Point

The runtime validator accepts a matched node only when:

```text
node has a live full/SWA tail
and every node on the matched path has independent state pages or a live tail
```

This keeps suffix prefill safe while allowing state ownership to be represented
independently.  It also explains the boundary table:

| prompt length | phase1 aligned hit | Route B hit |
| ---: | ---: | ---: |
| 256 | 128 | 0 |
| 257 | 256 | 256 |

The `256 -> 0` loss is now classified as an SWA ownership limitation, not a
compression-state limitation.

## Evidence

The CPU probe covers:

- C4/C128/indexer state loc formulas;
- retained state pages after full/SWA head tombstone;
- state page reuse without stale retained refs;
- page boundary 256 and 257 fixed point;
- eviction cleanup and leak checks.

The TP8 text smoke covers readable output under
`--enable-dsv4-component-loc-ownership`.

## Follow-Up For 08.21.4

TARGET 08.21.4 should keep Route B graph/deforest guarded until direct
component loc metadata is staged for replay.  If recovering the `prompt_len=256`
hit remains important after graph work, the missing piece is independent SWA
component ownership or a bounded SWA reconstruction rule, not more compression
state ownership.
