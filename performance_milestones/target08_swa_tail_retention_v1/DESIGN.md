# TARGET 08.20 DSV4 SWA Tail Retention V1 Design

Date: 2026-07-04

## Decision

V1 is a fail-closed safe subset, not a runtime component-retention
implementation.  The design work found that mini's current DSV4 cache model
cannot safely let C4/C128/indexer/compression-state data outlive released
full-token pages without either:

- a retained-component store plus materialize/remap on prefix hit; or
- independent component allocators/refcounts and attention metadata that can
  address component locations independently from `full_loc // ratio`.

Both options are larger than the intended conservative V1 and start to cross
the stop rule into a unified KV/component allocator rewrite.  The safe V1 action
is therefore to keep phase-1 full-page-owner radix prefix cache as the only
working opt-in, add a new explicit fail-closed V1 opt-in, and record the exact
ownership blocker for TARGET 08.21.

## 1. Current Mini Derivation Model

Mini stores one scheduler page table in the full-token namespace.  For DSV4,
`DeepSeekV4KVCache` derives every runtime component from those full-token
locations:

- SWA/full rows use the full loc directly in `_swa_buffer`.
- C4 rows use `full_loc // 4` at positions where `(position + 1) % 4 == 0`.
- C128 rows use `full_loc // 128` at positions where
  `(position + 1) % 128 == 0`.
- C4 indexer rows use the same C4 compressed loc namespace.
- Compression state locs are page-derived:
  `state_loc = (swa_loc // page_size) * ring_size + (swa_loc % ring_size)`.

The scheduler allocates and frees full pages through `CacheManager._allocate`
and `_free`.  `DeepSeekV4KVCache.on_pages_allocated()` increments full, C4,
C128, and C4-indexer refcounts derived from those full pages; `on_token_indices_freed()`
decrements the same derived locs.  Compression-state pools currently have no
separate refcount in mini.  Their storage is allocated per full page and their
addresses are derived from full/SWA locs.

Attention metadata also assumes this coupling.  `DSV4AttentionBackend` gathers
full locs from the request page table, converts compressed raw positions back
through the full page table, and then uses `full_loc // ratio` as the C4/C128
or indexer cache loc.  There is no metadata field today that can say "the full
token page was released, but the C4 row lives in an independent retained store".

## 2. Why Released Full Pages Would Dangle Or Collide

If an old prefix full page is returned to `CacheManager.free_slots`, a later
request can allocate the same page start.  That new request will write:

- SWA rows into the same full locs;
- C4 rows into the same `full_loc // 4` locs;
- C128 rows into the same `full_loc // 128` locs;
- C4 indexer rows into the same C4 locs;
- compression state into page-derived state locs.

Therefore a radix node that still points at the old full-token locs would
either read overwritten data or cause double free when evicted.  Keeping only a
Python-side tombstone is insufficient because current attention/indexer kernels
do not consult tombstone metadata; they consult the page table and derived
component locs.

Avoiding this requires one of these explicit ownership models:

- copy C4/C128/indexer data into a retained store, then on hit either remap
  attention metadata to that store or materialize it into newly allocated full
  pages before forward;
- add independent component allocators/refcounts so a full page can be reused
  without reusing its C4/C128/indexer/state locs;
- keep the full page alive, which is phase-1 behavior and recovers no old-page
  capacity.

## 3. V1 Data-Model Choice

V1 chooses the conservative safe subset: do not detach compressed components
from released full pages at runtime.

The rejected alternatives are:

- Independent component refcount: correct in spirit, but it requires allocator
  and attention metadata changes so C4/C128/indexer/state locs are no longer
  just `full_loc // ratio`.
- Retained-component store: safer than refcounting in the current namespace,
  but a correct version still needs radix node accounting for logical length
  versus live full/SWA tail length, split/evict behavior for retained nodes,
  and materialize/remap on hit.  This is a TARGET 08.21-sized change.
- Tail-only/head-only cache admission: this can recover capacity by caching
  less, but it is not SGLang-style component retention and would fake the
  requested capacity recovery.

The V1 opt-in is therefore fail-closed.  It must not silently fall back to an
unsafe partial implementation.

## 4. SWA Tail Page-Aligned Boundary

The SGLang reference keeps a page-aligned SWA frontier:

```text
swa_evicted_seqlen = floor((pre_len - sliding_window_size - page_margin) / page_size) * page_size
```

For mini's TARGET 08 runs:

```text
page_size = 256
sliding_window = 128
tail_pages = ceil(128 / 256) = 1
tail_tokens = 256
```

A correct mini component-retention implementation would keep one page-aligned
SWA tail per retained prefix branch and tombstone earlier SWA/full rows.  The
tail start for a retained logical prefix of length `L` would be:

```text
tail_start = max(0, align_down(L - tail_tokens, page_size))
```

This boundary is safe for SWA reads because every query needs at most the last
128 tokens, and a 256-token page contains that window.  It is not sufficient for
C4/C128/indexer reuse unless those compressed components have independent live
locations.

## 5. Safe Prefix Hit Fixed Point

The phase-1 fixed point is:

```text
hit_len = min(radix_match_len, input_len - 1)
hit_len = align_down(hit_len, page_size)
require page_size % 128 == 0
```

That works because all full pages and all derived components remain owned by
the same full-token pages.

For true component retention, the fixed point would have to be:

```text
hit_len <= logical radix match length
hit_len aligned to page_size
hit_len <= full/SWA tail materialization coverage for the active request
hit_len <= C4 retained coverage * 4
hit_len <= C128 retained coverage * 128
hit_len <= indexer retained coverage * 4
hit_len starts suffix prefill on a page/C128 boundary, or compression state
        must be retained/reconstructed for the boundary
```

In current mini, C4/C128/indexer retained coverage is not independently
addressable after old full pages are freed.  The safe fixed point therefore
collapses to the phase-1 full-page-owner model.  Releasing old full pages while
claiming the original hit length would be unsafe.

Compression state does not rescue this.  Mini allocates state pools per full
page and derives state locs from `swa_loc`; there is no independent state owner.
For page-aligned suffix prefill, mini can reconstruct new boundary state from
the suffix tokens, but that only covers future writes.  It does not preserve
old compressed/indexer rows after full-page reuse.

## 6. SGLang Parity

The SGLang model has pieces mini does not yet have:

- `SWATokenToKVPoolAllocator` owns separate full and SWA allocators and a
  full-to-SWA mapping.
- `free_swa_out_of_window_slots()` releases out-of-window SWA at a page-aligned
  `swa_evicted_seqlen`.
- `SWAComponent` stores SWA component values separately from the full component,
  tombstones old SWA by setting the SWA value to `None`, and locks only the
  sliding-window ancestors needed by a request.
- `UnifiedRadixCache` validates matches across all components and tracks
  per-component protected/evictable sizes.
- SGLang DSV4 memory pools expose separate component pools and state pools, so
  the tree can reason about component lifetimes independently.

Mini aligns with the SGLang direction conceptually, but not structurally.  The
minimal correct next slice is TARGET 08.21: introduce retained-component or
component-refcount ownership first, then add the SWA tombstone/tail policy.

## Required Runtime Behavior

- Default remains unchanged.
- `--enable-dsv4-radix-prefix-cache` remains the phase-1 rollback baseline.
- The new V1 opt-in must be explicit and fail closed until the above ownership
  model exists.
- No low-precision path, attention-kernel rewrite, CUDA graph allocator change,
  or long-distance replay is part of this target.

