# TARGET 08 DSV4 Radix Prefix Cache Design

## Source Parity Notes

### Mini Current State

- `python/minisgl/kvcache/radix_cache.py` stores a radix tree whose values are
  full-token page-table indices. It only inserts page-aligned prefixes
  (`align_down(len(input_ids), page_size)`) and it only matches page-aligned
  chunks. Node `ref_count` protects matched pages from eviction while a request
  is using them.
- `python/minisgl/scheduler/cache.py` owns physical page allocation and release.
  `allocate_paged()` allocates only pages after `req.cached_len`; a prefix hit
  copies the cached page-table entries into the new request row. `cache_req()`
  inserts the completed prompt prefix, unlocks the old handle, frees duplicate
  pages already inserted by another request, and frees only non-cached tails.
- `python/minisgl/kvcache/deepseek_v4_pool.py` already routes DSV4 ownership
  through the full-token page namespace. `on_pages_allocated()` increments
  full, C4, C128, and C4-indexer refcounts derived from the allocated full
  pages. `on_token_indices_freed()` decrements the same derived components and
  detects double frees.
- `python/minisgl/attention/deepseek_v4.py` derives all DSV4 read/write
  metadata from the request page table and logical positions: SWA reads use
  full-token slots; C4/C128/indexer slots are `full_loc // ratio` at ratio
  boundaries.

### vLLM Reference Shape

- vLLM uses block hashes rather than a radix token trie. `BlockPool` stores
  `KVCacheBlock.ref_cnt`, a hash-to-block table, and an eviction-ordered free
  queue. A cache hit calls `touch()` to increase the block refcount and remove
  it from the eviction queue when needed; freeing a request decrements refcounts
  and returns zero-ref blocks to the queue.
- `HybridKVCacheCoordinator` computes a fixed-point prefix hit across KV cache
  groups. It reduces the hit length until all groups with different block sizes
  can support the same aligned prefix. This is the important rule for mini:
  DSV4 prefix hits must stop at a length safe for full-token, SWA, C4, C128,
  and indexer cache state.
- DeepSeek V4 SWA in `sparse_swa.py` builds window metadata from `seq_lens`,
  `query_start_loc`, block table, and slot mapping. Decode/prefill SWA reads
  need the latest window slots to remain resident; they do not require a
  separate prefix-cache object if the page table still points to retained
  physical slots.

### SGLang Reference Shape

- SGLang's DSV4 pool splits SWA, C4, C128, indexer, and compress-state storage.
  Full-to-SWA and full-to-compressed translations are explicit, and C4/C128
  page sizes are derived from the model page size.
- SGLang's unified radix cache models SWA as a separate component that can be
  tombstoned independently from full-attention data. Its SWA component locks
  only the sliding window and frees out-of-window SWA slots to avoid pinning
  old SWA data unnecessarily.
- Mini does not currently have a separate SWA pool. Its DSV4 phase-1 design
  therefore keeps full-token pages as the single ownership unit and derives all
  component retention from those pages. This is less memory-efficient than
  SGLang's tombstoned SWA component, but it keeps eviction/refcount ownership
  unambiguous for the first correct opt-in.

## Phase 1 Mini Design

- Explicit opt-in only:
  `enable_dsv4_radix_prefix_cache=False` remains the default. DeepSeek V4 keeps
  the existing forced-naive path unless this flag is set.
- Use the existing radix tree for tokens and page-table values. Do not add a
  second allocator or a separate DSV4 component owner.
- Require a DSV4-safe page size for runtime opt-in. The target page size is
  `256`, which is divisible by the SWA window `128`, C4 ratio `4`, and C128
  ratio `128`. A safe hit is therefore also C4/C128 component-aligned.
- Full-token pages are canonical. Retaining a prefix means the radix tree owns
  the page-table entries for those full-token pages; the DSV4 pool's existing
  refcounts keep the derived C4/C128/indexer slots live until eviction.
- No extra KV-pool refcount is added on match. Mini's radix `lock_handle()`
  protects a cached page from eviction while a request uses it; the KV pool
  already has one allocation refcount for the retained cached page. Eviction is
  the only final release path for cached pages.
- Compression state is reconstructed by alignment in phase 1. Since runtime
  DSV4 opt-in requires `page_size % 128 == 0`, suffix prefill after a prefix
  hit starts at a C4 and C128 boundary. Current mini compression stores write
  boundary rows directly from the suffix tensors, so no partial compression
  ring state from inside the cached prefix is required. If future code needs
  online compression across non-128-aligned prefix boundaries, that must become
  a separate target.

## Alignment Rules

- Radix keys and values remain page-aligned.
- Runtime DSV4 radix prefix cache requires `page_size % 128 == 0`; target runs
  use `--page-size 256`.
- Matched prefix length is at most `align_down(input_len - 1, page_size)`,
  preserving the scheduler's existing "leave the last prompt token for prefill"
  convention.
- C4 and C4-indexer slots retained per page: `page_size / 4`.
- C128 slots retained per page: `page_size / 128`.
- SWA window correctness: page-table entries for the cached prefix stay valid,
  and DSV4 metadata gathers the last `min(seq_len, 128)` full-token slots from
  that page table.

## Metrics

`CacheManager` records cumulative prefix metrics:

- match requests, hit/miss/full/partial counts;
- prefix hit length and hit rate;
- saved prefill tokens;
- suffix prefill tokens after hit;
- inserted cached tokens;
- evictions and evicted tokens/pages;
- current retained/protected/evictable prefix tokens/pages;
- DSV4 retained full/C4/C128/indexer/compress-state slots;
- estimated retained DSV4 cache bytes.

The benchmark/text smoke reports snapshot these metrics so A/B runs can compare
prefix-disabled and prefix-enabled behavior.

## Stop Conditions For This Design

- If text or logits diverge versus prefix-disabled mode, stop and do not
  promote.
- If page size is not 128-aligned for DSV4 opt-in, reject the run instead of
  silently using an unsafe prefix hit length.
- If DSV4 refcount integrity fails after repeated hit/evict cycles, stop.
- If CUDA graph replay is disabled unexpectedly by the opt-in, stop and record
  the blocker.
