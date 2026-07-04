# TARGET 08.21.2 Design Delta

Date: 2026-07-04

## Scope

This target implements Route B / B1 behind an explicit opt-in:

```text
--enable-dsv4-component-loc-ownership
```

The default phase-1 radix and non-radix paths remain unchanged.  The opt-in
requires `--enable-dsv4-radix-prefix-cache`, requires `window_size <= page_size`,
and disables DSV4 CUDA graph capture until TARGET 08.21.4 owns graph replay and
metadata deforest.

## Ownership Model

The KV pool now has separate component page ownership for:

| component | page size for `page_size=256` | owner |
| --- | ---: | --- |
| full/SWA token page | 256 tokens | normal KV page allocator |
| C4 compressed page | 64 component slots | component allocator |
| C128 compressed page | 2 component slots | component allocator |
| C4 indexer page | 64 component slots | component allocator |

For each newly allocated full page, the opt-in allocates independent C4, C128,
and C4-indexer component pages and stores full-page-to-component-page mappings.
The compressed/indexer stores still consume explicit loc vectors, so their write
paths stay small and do not need Route A materialization.

When a prefix node becomes retained, the radix node stores
`DSV4ComponentPageHandles` alongside the full-token value.  The scheduler then
releases old full/SWA head pages with `free_components=False`, tombstones those
full locs in the radix value with `-1`, and keeps one page-aligned full/SWA tail
for safe compression-state access.  Radix eviction releases component handles
through a KV-pool callback.

## Refcount And Free Rules

The opt-in keeps separate refcounts for full, C4, C128, and C4-indexer slots.

Free sequencing:

1. Request allocation increments full refs and component refs for every allocated
   full page.
2. Prefix insertion stores component handles for only the page-aligned retained
   region.
3. Finished request tails that are not retained are freed normally, including
   their components.
4. Retained full/SWA head pages are freed with `free_components=False`; component
   refs remain owned by the radix node.
5. Radix eviction releases live full tails with `free_components=False`, then
   releases the component handles.

Double-free protection is slot-level: component free checks fail closed if any
component slot refcount is already zero.  Component free lists are also checked
for duplicate page ids.

## Runtime Metadata Schema

The attention metadata now has optional component-owned page tables:

| field | meaning |
| --- | --- |
| `component_loc_ownership` | metadata was built from component-owned tables |
| `c4_page_table` | logical retained page -> physical C4 component page |
| `c128_page_table` | logical retained page -> physical C128 component page |
| `c4_indexer_page_table` | logical retained page -> physical indexer component page |

Eager metadata construction combines:

- component pages stored in the prefix handle;
- component pages for the active suffix still mapped from live full pages.

Sparse C4 and C128 metadata gather directly from those component page tables.
Indexer metadata receives `c4_indexer_page_table` instead of the full-token page
table.  The old `full_loc // 4` and `full_loc // 128` path remains the default
when the opt-in is off.

## Compression State Guard

Compression state ownership is deliberately not claimed in this target.  The
safe-hit rule is therefore:

```text
a matched radix node is usable only if its final full/SWA page is still live
```

If a split lands on a node whose full/SWA value is fully tombstoned, the match is
downgraded to the nearest ancestor with a live full/SWA tail, or to root.  This
is why the probe records:

| prompt length | phase-1 aligned hit | Route B safe hit |
| ---: | ---: | ---: |
| 255 | 128 | 128 |
| 256 | 128 | 0 |
| 257 | 256 | 256 |
| 258 | 256 | 256 |

This guard avoids silently depending on released full/SWA locs.  TARGET 08.21.3
should replace it with independent compression-state ownership.

## Graph Guard

When component loc ownership is enabled, the engine clears DSV4 CUDA graph
capture and logs the eager fallback.  B3/TARGET 08.21.4 must teach replay and
metadata deforest to copy/stage component-owned loc tables instead of rebuilding
compressed locs from `raw_out_loc // ratio`.

## SGLang Parity

| SGLang idea | Mini B1 mapping |
| --- | --- |
| independent component data/refcounts | C4/C128/C4-indexer component page free lists and refcounts |
| unified radix component values | `DSV4ComponentPageHandles` stored on radix nodes |
| SWA tombstone behavior | released full/SWA head pages become `-1` in radix values |
| component LRU/eviction | radix eviction calls a component release callback |
| direct component metadata | eager attention/indexer metadata consumes component page tables |

Remaining gaps are compression-state ownership and graph/deforest integration.
