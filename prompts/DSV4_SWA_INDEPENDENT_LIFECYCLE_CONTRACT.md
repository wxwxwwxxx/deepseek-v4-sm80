# DeepSeek V4 SWA Independent Lifecycle Contract

Status: authoritative contract for TARGET 08.45 and the first reference for
TARGET 08.46 audit / TARGET 08.47 unified fix / TARGET 08.48 case-boundary
fix.

Scope: mini-sglang DeepSeek V4 on the TARGET 08 Route B radix-prefix path with
component loc ownership and independent SWA lifecycle enabled. This document is
contract text only. It records current implementation risks but does not fix
them.

Primary SGLang references:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/common.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor_v2.py`

## 1. Definitions

Full token loc: the physical token-slot index in the full-attention/full-page
KV domain. Mini stores full token locs in the global request page table.

Full page: `page_size` contiguous full token locs. TARGET 08 uses `page_size =
256`. A full page is the radix prefix cache's base unit and the scheduler's
full-pool allocation/free unit.

SWA token loc: the physical token-slot index in the SWA KV domain. It is
computed as `swa_page * page_size + page_offset`. It is not interchangeable
with a full token loc once independent SWA lifecycle is enabled.

SWA page: `page_size` contiguous SWA token locs. It lives in the SWA KV buffer
owned by `DeepSeekV4KVCache`. The SWA page pool is separate from the full page
pool.

C4/C128/indexer/state component locs: physical locations in Route B component
pools. They are independent component owners and must remain valid when SWA
pages are tombstoned or freed.

Active request page table: the per-request row in `Engine.page_table`. It maps
logical sequence positions to full token locs. It is an active execution view,
not the owner of every component reachable through those full locs.

Full-to-SWA mapping: mini's `_full_to_swa_page` table, analogous to SGLang's
`full_to_swa_index_mapping`. It translates active full pages to live SWA pages.
It is a translation table and liveness witness; it is not the sole ownership
record. `-1` in this table means no live SWA mapping for that full page.

SWA component value: the retained prefix-side SWA data associated with radix
nodes. In SGLang this is `SWAComponent.value`, a tensor of translated SWA pool
indices that becomes `None` on device tombstone while the base/full value stays
intact. In mini this role is represented by `DSV4SWAPageHandles`.

`DSV4SWAPageHandles`: mini's page-level SWA component value for radix nodes. It
contains the logical length, page size, and a page vector where non-negative
entries are physical SWA pages and `-1` entries are tombstones.

Radix prefix node: a node in the radix tree storing token ids, full token locs,
Route B component handles, and optional SWA component handles for that segment.

Radix cache handle: a request's matched/inserted radix path root-to-node view.
It protects radix nodes via radix lock refs. It is not itself a physical KV
owner; the nodes and component values on its path hold the retained prefix
ownership.

Active decode window: the trailing logical positions that SWA attention may
read for the current decode step. The active SWA window is defined by
`window_size` and page alignment. Positions outside the active window may be
eligible for active-only SWA release if they are also outside protected prefix
state.

Protected prefix/cache region: logical positions represented by locked or
retained radix prefix state that active decode must not physically free. Mini's
equivalent of SGLang `cache_protected_len` must be page aligned.

Tombstone: a logical marker saying SWA component data is absent. In mini this
is `-1` in `DSV4SWAPageHandles` or `_full_to_swa_page`; in SGLang device
tombstone is `SWAComponent.value is None`. Tombstone is not a physical dummy
page and must not be passed as active kernel input.

Free/release: decrement a physical owner's refcount and possibly return the
physical page to a free list. Free/release is not idempotent. Duplicate owner
release is a correctness bug and must remain visible through guards.

Evict: remove cached prefix ownership. Full/component eviction removes radix
nodes and frees the associated full/component owners. SWA-only eviction
tombstones the SWA component value and frees only SWA pages while leaving
full/component prefix state intact.

Dummy full token: the Engine graph-padding sentinel. The authoritative dummy
full token start is `planned_pages * page_size`, passed to
`DeepSeekV4KVCache` as `dsv4_dummy_token_start`.

SWA dummy page: the reserved SWA page used for dummy graph rows. It is
refcount-pinned, excluded from the free list, never tombstoned or freed, and
may appear only in dummy rows whose outputs are ignored.

CUDA graph metadata buffer: a static capture-side metadata tensor reused by
graph replay. It is valid only for the capture shape and only after replay
copies or direct graph metadata producers have populated it for the current
batch and ownership version.

## 2. Ownership Model

Final model:

- `DeepSeekV4KVCache` owns physical SWA storage, `_swa_page_refcount`,
  `_free_swa_pages`, `_full_to_swa_page`, and the SWA dummy page.
- A non-negative `DSV4SWAPageHandles.swa_pages[i]` on a radix node is an
  owning prefix SWA component value for that node segment. It is not merely a
  borrowed snapshot. Releasing that component value may decrement SWA physical
  refcounts exactly once.
- `_full_to_swa_page` is an active translation table. It proves that an active
  full page currently has a live SWA page, but it is not by itself the owner.
- Active request full pages own active-only SWA pages until those pages are
  inserted into radix as SWA component values or freed as non-cached tail.
- Ownership transfer from active request to radix happens at radix insert /
  cache boundary. The transferred SWA page is represented by the radix node's
  `DSV4SWAPageHandles`; it must not also be freed by active release.
- Physical decrement is allowed only through the KV cache release APIs, and
  only when the caller is releasing an owner: active-only page, prefix SWA
  component value, or full prefix eviction that owns the component.
- Metadata builders, CUDA graph copy helpers, kernels, and pure tombstone
  overlay updates may not decrement physical SWA refcounts.

Authorized decrement paths:

- Active-only release: `release_swa_for_full_indices` may decrement pages that
  are outside the active SWA window and outside the protected prefix/cache
  region. It must operate on a page-aligned `[start, end)` range, not blindly
  on `[0, release_end)` when the head contains prefix-owned pages.
- Prefix SWA tombstone: radix SWA component eviction may decrement live pages
  represented by unlocked/evictable `DSV4SWAPageHandles` and then tombstone
  the component value. This is the mini analogue of SGLang
  `SWAComponent.evict_component`.
- Finish-time cache request: `cache_req(finished=True)` may free active tail
  pages that were not inserted into radix. It may also commit prefix SWA
  tombstones at the insert boundary, but it must not revisit stale live page
  snapshots that active release already physically freed.
- Full prefix eviction: radix node eviction may free full/component ownership.
  Under independent SWA lifecycle it must release the node's SWA component
  owner if and only if that component value still contains live pages.
- SWA-only pressure eviction: may release evictable SWA component values
  without deleting full/component radix state.

Non-authorized decrement paths:

- Updating `DSV4SWAPageHandles` to `-1` as a request-local overlay.
- Clearing `_full_to_swa_page` without owning the physical release.
- Attention metadata construction.
- CUDA graph capture/replay metadata copies.
- Dummy sentinel handling.

### 08.44 active-time radix-handle tombstone verdict

TARGET 08.44's active-time `tombstone_dsv4_swa_pages(req.cache_handle, pages)`
is not contract-compliant as a final lifecycle model. It is a temporary
stale-snapshot mitigation: active release decrements physical SWA pages and
then mutates radix-node SWA component snapshots so finish-time tombstone will
not double free them. That couples active decode ownership to prefix component
ownership and makes radix handles both owner and mutable borrowed snapshot.

TARGET 08.47 should replace this with one of these models:

- Request-local released-page overlay: active decode records pages released
  for this request, metadata treats those positions as absent for this request,
  and radix component values are not mutated until an owner boundary.
- Finish-boundary processing: active decode does not decrement prefix-owned
  SWA pages. At `cache_req` insert/finish, mini computes the retained SWA
  component value from the request's monotonic SWA eviction frontier and commits
  tombstones once.

Either model must keep prefix SWA component values as owning values and keep
duplicate releases visible.

## 3. State Transitions

### Allocation of full and SWA pages

Preconditions:

- Full page starts are page aligned and in range.
- `_full_to_swa_page[full_page] == -1` before SWA allocation.
- The SWA free list has enough non-dummy pages.

Postconditions:

- Full refcount increments for full token locs.
- SWA page refcount increments for allocated SWA pages.
- `_full_to_swa_page[full_page] = swa_page`.
- Component pages are allocated/cleared through Route B if enabled.
- Dummy SWA page remains pinned and is not allocated from the free list.

### Prefill insert into radix

Preconditions:

- Insert length is page aligned.
- Component and SWA handles have lengths matching the node segment.
- SWA handles are built only for pages whose SWA mapping is live, or are
  explicitly tombstoned for pages beyond the retained SWA frontier.

Postconditions:

- New radix nodes own full token locs and Route B component handles.
- New radix nodes own SWA component values for retained SWA pages.
- Active ownership for inserted pages is transferred to radix; it is not an
  additional refcount unless the implementation explicitly increments one.
- Insert may split nodes only on page boundaries for SWA component values.

### Prefix match / handle lock

Preconditions:

- Match boundary must not expose a path whose needed component state is absent.
- For SWA, the matched path must contain enough live SWA tail for the active
  window, or metadata must reconstruct from active full-to-SWA mapping.

Postconditions:

- Radix lock protects the matched prefix from full/component eviction.
- SWA component lock/protection is window-bounded. Tombstoned SWA nodes are
  skipped, not revived by lock release.

### Decode step

Preconditions:

- Active request page table contains valid full token locs for logical
  positions that will be read or written.
- All full pages in the active SWA window have live SWA mappings, unless those
  rows are dummy or the valid SWA length has been shortened.

Postconditions:

- New full/SWA pages for decode are allocated before writes.
- `store_swa` translates full output locs through live SWA mappings.
- Attention metadata for the current step is built from current ownership
  state and remains valid until the kernel consumes it.

### Active out-of-window release

Preconditions:

- Release range is page aligned.
- The range is outside the active sliding window with the required one-page
  margin.
- The range is outside the protected prefix/cache region.
- The request has a monotonic SWA eviction frontier, equivalent to SGLang
  `swa_evicted_seqlen`.

Postconditions:

- Only active-owned SWA pages are physically decremented.
- `_full_to_swa_page` is cleared for released full pages.
- Prefix radix component values are not physically released by this path.
- Request-local metadata prevents released active pages from reappearing in
  the same request's active SWA input.

### Unfinished request caching

Preconditions:

- New insert boundary is page aligned.
- Old and new handles are used only while locked.
- Active SWA eviction frontier is known.

Postconditions:

- Newly inserted pages transfer ownership to radix.
- The request locks the new handle.
- SWA pages older than the committed retained frontier are tombstoned at the
  boundary, not by stale active snapshots.
- Page table entries for matched prefix are refreshed from the radix handle
  after all owner changes.

### Finished request caching

Preconditions:

- Tail that will not be inserted is identified by `new_handle.cached_len`.
- Inserted SWA component values are current with the request's SWA eviction
  frontier.

Postconditions:

- Non-inserted active tail is freed.
- Inserted prefix owns retained SWA pages.
- Finish-time release must not decrement a SWA page already released by
  active-only release.

### Prefix SWA tombstone

Preconditions:

- Node/component is an owning SWA component value.
- Node is eligible for SWA component eviction or the insert/finish boundary is
  committing a new tombstone.
- Component lock/protection permits the tombstone.

Postconditions:

- Live pages in the component value are decremented once.
- Mini stores `-1` for tombstoned pages; SGLang stores `value = None` for a
  device tombstone.
- Full/component/base values remain intact unless this is a full node eviction.

### Prefix full/component eviction

Preconditions:

- Node is an unlocked evictable leaf, or the component-specific eviction rules
  allow internal SWA tombstone.

Postconditions:

- Full/component owners are released.
- SWA owner is released if still live.
- The radix node is removed only for full eviction; SWA-only eviction leaves
  the node and full/component values intact.

### SWA-only pressure eviction

Preconditions:

- Full and component capacity are sufficient or not the reason for eviction.
- SWA capacity has a deficit.
- Candidate SWA component value is evictable and not locked/protected.

Postconditions:

- SWA pages are freed.
- `DSV4SWAPageHandles` is tombstoned.
- Subsequent prefix match may shorten to a safe boundary or require active
  reconstruction for the current window.

### Dummy row handling

Preconditions:

- Engine fills dummy request page-table rows with `dsv4_dummy_token_start`.
- KV cache knows the same dummy full-token start.

Postconditions:

- Full dummy token translates to SWA dummy page.
- Dummy page refcount remains pinned.
- Dummy page is excluded from allocation counts, release, tombstone, and
  eviction.

### Graph capture and replay

Preconditions:

- Capture metadata for dummy rows uses valid dummy SWA locs, not `-1` inside
  active lengths.
- Replay source metadata is built for the current batch and current ownership
  version.
- Direct SWA graph metadata is disabled under independent SWA lifecycle unless
  a future versioned producer proves safety.

Postconditions:

- Replay buffers contain current valid SWA token locs for active rows.
- Padding outside active lengths is filled with `-1`.
- No dynamic CUDA allocation is introduced during graph replay.

## 4. Protected Frontier Rule

Mini must adopt the SGLang rule in spirit: active SWA release frees only pages
that are outside both the protected cache region and the active SWA window, and
all frontiers are page aligned.

Required state:

- `cache_protected_len`: page-aligned logical length of prefix/cache state that
  active release must not physically free. For mini this should be derived from
  the locked radix handle and cache insert boundary, not from `req.cached_len`
  alone.
- `swa_evicted_seqlen`: monotonic per-request frontier. It prevents repeated
  release of the same active page and makes release ranges explicit.
- `active_window_frontier`: page-aligned first position older than the active
  SWA window and the one-page margin:
  `align_down(pre_len - window_size - page_size, page_size)`.

Allowed active release range:

```text
release_start = max(req.swa_evicted_seqlen, cache_protected_len)
release_end   = max(release_start, active_window_frontier)
release [release_start, release_end)
```

If mini uses a head-only release API, the effective `active_swa_free_until`
must never cross into prefix-owned pages. A head-only `[0, release_end)` API is
therefore insufficient once `[0, cache_protected_len)` is radix-owned.

For compatibility with older mini wording:

```text
active_swa_free_until <= min(first_prefix_owned_page_start, active_window_frontier)
```

This scalar form is valid only for an active-owned head range. For a request
whose sequence head is already represented by the radix cache,
`first_prefix_owned_page_start` is usually `0`, so active head release should
release nothing. Long-running requests need the SGLang-style `[release_start,
release_end)` range above, not a head-only release.

One-page margin:

- Mini should keep SGLang's one-page margin before the radix insert boundary.
- With `page_size = 256` and `window_size = 128`, this means retaining at least
  the current SWA window plus one extra page.
- A future drop-margin mode may exist only behind an explicit opt-in and must
  have dedicated no-weight, full-model, graph, and leak gates.

Current mini risk:

- `release_end = align_down(req.cached_len - tail_tokens, page_size)` with
  `tail_tokens = ceil(window_size / page_size) * page_size` is too aggressive.
  It uses `req.cached_len` without a protected-prefix lower bound, lacks a
  monotonic `swa_evicted_seqlen`, and does not keep the SGLang one-page margin.
  It can therefore target prefix-owned pages in long decode.

`req.cached_len`, `req.device_len`, old handle, and new handle:

- `req.device_len` is the length whose trailing window is needed by the
  current forward.
- `req.cached_len` is a cache/valid length and must not be used as the sole
  active release boundary.
- `old_handle.cached_len` protects previously matched prefix state.
- `new_handle.cached_len` is the insert result boundary. It may create new
  prefix ownership, but active release must not retroactively free that new
  ownership.

## 5. Metadata And Kernel Input Contract

For each real active row `r`:

- `swa_topk_lengths[r]` is the number of SWA token locs the kernel may read.
- For every column `c < swa_topk_lengths[r]`,
  `swa_page_indices[r, c]` must be a valid physical SWA token loc:
  `0 <= loc < swa_num_tokens`.
- For real active rows, `loc` must not be `-1`, must not point to the SWA dummy
  page, must not point to a free-list page, and must not point to a page whose
  SWA refcount is zero.
- `-1` is allowed only outside the active length or in rows/columns
  explicitly ignored by the kernel.
- Tombstoned prefix SWA pages may not appear inside active SWA lengths.
- If metadata cannot prove that a SWA page is live, it must either rebuild the
  loc from active `_full_to_swa_page`, shorten the valid SWA length, or fail
  before launching the kernel.
- The sparse attention kernels must not receive invalid locs and attempt to
  rely on CUDA bounds behavior.

For graph-padded dummy rows:

- SWA dummy locs are valid only because the row is dummy and the output is
  ignored.
- Dummy locs must not be used to stand in for tombstoned real prefix pages.

Debug gates:

- `MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG` must remain available and should be
  extended to check refcount/free-list membership for active SWA pages.
- Optional graph replay metadata version checks should fail if producer-side
  SWA ownership changed after the source metadata was built.
- `CUDA_LAUNCH_BLOCKING=1` and cache sync debug modes are required attribution
  tools for CUDA illegal memory access.

## 6. Dummy And Sentinel Contract

TARGET 08.42 is part of this contract.

- Engine dummy full token start is `planned_pages * page_size`.
- Engine passes that value as `dsv4_dummy_token_start`.
- Dummy request page-table rows are filled with that full token start.
- DSV4 KV cache maps exactly that full token/page to the reserved SWA dummy
  page.
- The SWA dummy page is the last SWA page, refcount-pinned to one, never in
  `_free_swa_pages`, and ignored by release/tombstone paths.
- `-1` means invalid/tombstone/padding. It is never a dummy page.
- Translating `-1` must produce `-1`, not dummy.
- Translating the dummy full token must produce the SWA dummy page, not `-1`.

## 7. Refcount And Leak Contract

Refcount dimensions:

- Full token refcount tracks allocated full-token slots.
- SWA page refcount tracks physical non-dummy SWA page ownership plus the
  pinned dummy page.
- C4/C128/indexer/state refcounts track Route B component ownership and are
  independent from SWA.

Increment rules:

- Full and component increments happen on page allocation or explicit component
  ownership creation.
- SWA increments happen when a non-dummy SWA page is allocated or when a future
  implementation explicitly adds a second logical owner. Current mini should
  use transfer semantics for active-to-radix insert unless it also adds paired
  increments.

Decrement rules:

- Decrement only an owning value.
- Decrement once.
- Ignore negative tombstones and the SWA dummy page.
- Decrement failure on zero refcount is a hard error, not a warning.

`assert_no_leak()` must prove:

- No negative refcounts.
- All free lists are unique and in range.
- `_full_to_swa_page` never maps to dummy, free-list pages, out-of-range pages,
  or zero-refcount pages.
- Dummy page remains pinned.
- Component mappings remain valid when SWA pages are tombstoned.

Runtime counters mean:

- Allocated/free/tombstoned totals are useful trend counters.
- They do not prove that metadata did not pass a stale page to a kernel.
- They do not prove that a CUDA graph replay buffer was produced under the
  correct ownership version.

Duplicate release cases that must still raise:

- Releasing the same live `DSV4SWAPageHandles` twice.
- Releasing a stale prefix handle after active-only release if that handle was
  still an owner.
- Freeing a full page whose SWA mapping has already been physically released
  without an explicit tombstone/overlay state.

## 8. Compatibility Contract

SWA lifecycle changes must not regress:

- Route B C4/C128/indexer/state component loc ownership.
- Component page-table lifetime cache.
- Marlin WNA16 release plus component-slot clear.
- Direct C4 graph metadata buffers.
- No dynamic CUDA allocation during graph replay.
- Page size `256` for TARGET 08 DSV4 gates.
- TP8 CUDA graph buckets `[1, 2, 4, 8, 16]`.
- The double-free guard in `DeepSeekV4KVCache`.
- Dummy full-token and SWA dummy-page handling.

Independent SWA lifecycle may disable or bypass optimizations that cannot prove
their SWA ownership version, but it must fail closed rather than silently pass
invalid locs to kernels.

## 9. Case Boundary And Serving-Run Contract

TARGET 08.43 exposed an additional contract layer: a serving Engine must be
able to run multiple benchmark scenarios or request waves back-to-back without
recreating the process.  Passing a single fresh-process workload is necessary
but not sufficient for promotion.

Mini's case-boundary contract:

- At the end of a case, all finished active requests must have transferred or
  released their full/SWA/component owners through `cache_req(finished=True)`
  and related owner-boundary paths.
- No active request table slot may be reused by the next case while old active
  full/SWA/component ownership remains attached to that slot.
- Prefix cache state may intentionally survive across cases, but every retained
  radix node must remain a valid owner:
  - full/component values have valid refcounts;
  - `DSV4SWAPageHandles` entries are either live non-negative SWA pages with
    positive refcount and not on the free list, or explicit `-1` tombstones;
  - SWA-only tombstones must not invalidate full/component/base values.
- `_full_to_swa_page` is an active translation table only.  After a case, it
  may contain mappings only for live active pages or pages whose owner contract
  explicitly requires a mapping.  It must not keep stale mappings to free-list
  pages, zero-refcount pages, dummy pages, or pages released by prefix
  tombstone.
- CUDA graph capture buffers may survive across cases, but replay source
  metadata must be rebuilt for the current batch and current SWA ownership
  version.  No graph metadata path may read stale request rows or stale
  page-table columns from a previous case.
- Static-width metadata builders must use the current row's logical lengths to
  bound real active reads.  Stale page-table values outside the current
  request's active lengths may be padding only and must not become active
  kernel inputs.
- Runtime metrics and retention counters are observation points.  They may
  synchronize and expose earlier CUDA faults, but they must not be treated as
  the producer until a preceding owner/metadata/kernel path has been ruled out.
- For auto-capacity runs, Marlin WNA16 raw weight release and KV/SWA/component
  arena planning must remain compatible with case-boundary reuse.  Released raw
  expert addresses must not be read by later forwards, graph replay, integrity
  probes, or fallback paths.

SGLang parity points to keep aligned:

- `free_swa_out_of_window_slots` updates page-aligned
  `req.swa_evicted_seqlen` using `cache_protected_len`, `swa_evict_floor`, and
  a one-page margin before freeing SWA slots.
- `SWAComponent` refreshes only window-bounded ancestors at match/insert end,
  uses `swa_evicted_seqlen` during insert overlap/recovery, and can skip leaf
  creation when the inserted segment is entirely outside the retained SWA
  window.
- SGLang separates allocator state, component tombstone state, request
  lifetime, and prefix component lifetime.  Mini adaptations may differ in data
  layout, but must keep the same ownership boundaries.

TARGET 08.48 must treat this section as part of the contract.  If SGLang
review shows a stronger or safer case-boundary invariant, update this document
before making broad fixes.

## 10. Test And Promotion Gates

Any future SWA lifecycle code change must pass these gates before promotion or
large soak:

- No-weight ownership tests for allocation, active-only release, prefix insert,
  prefix tombstone, SWA-only pressure eviction, full eviction, and finish-time
  cache request.
- Negative double-free guard tests proving duplicate SWA owner release still
  raises.
- Dummy sentinel tests for Engine dummy token start, SWA dummy page mapping,
  release ignore, and graph-padded rows.
- Metadata active-range tests proving no `-1`, dummy, out-of-range,
  zero-refcount, or free-list SWA loc appears inside `swa_topk_lengths`.
- CUDA graph replay tests proving generic replay copy and any direct metadata
  path produce current SWA metadata for active rows.
- Focused fixed128 full-model gates:
  `historical_4096_128_bs4` and `historical_4096_1024_bs4`.
- Marlin release + SWA independent compatibility after fixed128 gates pass.
- Serving, prefix, and eviction soak only after historical fixed128 gates pass.
- Graph replay/eager counters must remain healthy for buckets
  `[1, 2, 4, 8, 16]`.
- Multi-case same-Engine gates:
  `historical_4096_128_bs4 -> historical_4096_1024_bs4` with auto capacity,
  Marlin release, SWA independent lifecycle, graph buckets `[1, 2, 4, 8, 16]`,
  `CUDA_LAUNCH_BLOCKING=1`, and SWA liveness debug enabled.
- A fresh-process single-case pass does not replace the multi-case gate.

## 11. TARGET 08.46 / 08.48 Audit Priorities

Audit first:

- `CacheManager.release_active_dsv4_swa_out_of_window`: replace head-only
  release with protected, page-aligned `[start, end)` release and a monotonic
  `swa_evicted_seqlen` equivalent.
- `RadixPrefixCache.tombstone_dsv4_swa_pages`: classify as temporary snapshot
  sync and remove it from the physical release path.
- `RadixPrefixCache.release_dsv4_swa_out_of_window`: add protected frontier
  and one-page margin semantics.
- `DSV4SWAPageHandles`: make owning component value semantics explicit, with
  version/epoch or owner-state checks if snapshots remain copyable.
- `DSV4AttentionBackend._build_swa_page_table_row` and
  `_make_swa_indices_from_page_table`: validate active-range SWA locs against
  refcount/free-list state under debug gates.
- CUDA graph replay metadata copy: add SWA ownership/version guard or prove
  source metadata is always built after the last SWA ownership mutation for the
  replayed step.
- Full-model sparse attention inputs: add a pre-kernel debug gate that catches
  stale/freed SWA locs before asynchronous CUDA faults surface elsewhere.
- TARGET 08.48 additionally audits case-boundary state:
  - finished request cleanup and table-slot reuse;
  - retained prefix owner validity after each case;
  - graph capture/replay buffers and source metadata after scenario changes;
  - page-table static-width readers;
  - Marlin released-weight address reuse and auto-capacity arena planning.
