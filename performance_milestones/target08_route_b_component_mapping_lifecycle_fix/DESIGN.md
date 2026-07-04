# TARGET 08.22.1 Design

## Problem

Route B component ownership moves retained C4/C128/indexer/state pages into
radix node handles, then releases old full/SWA head pages by clearing their
active full-to-component staging mappings.  The request page table may still
contain those old full page ids until the request finishes.

`CacheManager.cache_req()` built DSV4 component handles for the whole
page-aligned `req.cached_len` before `RadixPrefixCache.insert_prefix()` knew
which pages were already cached.  A later finish pass for a request whose
prefill had already inserted and tombstoned a head page therefore asked
`DeepSeekV4KVCache.make_component_page_handles()` to gather mappings for old
full pages whose component ownership already lived in the radix tree.

## Fix

Make radix insertion report the new segment before constructing Route B
component handles.

- `CacheManager.cache_req()` passes a small builder callback to
  `insert_prefix()` when DSV4 component ownership is enabled.
- `RadixPrefixCache.insert_prefix()` walks/splits the tree first and computes
  `prefix_len`.
- Only if `prefix_len < insert_len`, the callback receives
  `[prefix_len, insert_len)` and builds `DSV4ComponentPageHandles` from that
  newly inserted suffix.
- Already cached pages keep using the component handles stored on their radix
  nodes.
- Tombstoned `-1` full/SWA pages and stale request page-table entries are not
  included in the active full-page handle construction path.

## Invariants

1. Existing radix component handles remain the owner for matched prefix pages.
2. New component handles are created only for newly inserted page-aligned
   suffix pages.
3. Active full-to-component mappings are required only for those new suffix
   pages.
4. `-1` tombstones are never treated as active pages requiring mappings.
5. Route B still keeps the SWA live-tail guard; this target does not add
   independent SWA ownership or reconstruction.
6. Eviction still releases radix node component handles through the existing
   callback, and request finish still releases active full/SWA tail pages
   through the normal KV pool path.

## Expected Boundary Behavior

For `page_size=256`:

- `257` keeps a safe one-page Route B hit.
- `512` and `768` may be shortened by the SWA-tail guard, but must not crash.
- `513` and `769` should recover safe two-page/three-page Route B hits because
  the final matched radix node has a live full/SWA tail page.
