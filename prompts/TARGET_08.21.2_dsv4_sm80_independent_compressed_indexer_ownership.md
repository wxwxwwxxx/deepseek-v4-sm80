# TARGET 08.21.2: DSV4 Independent Compressed/Indexer Ownership

## Status

Planned after TARGET 08.21.1.

This is Route B / B1.  Run only after component loc table preflight proves that
direct loc metadata can reproduce phase-1 derived metadata.

## Goal

Implement the first runtime Route B slice: independent ownership for retained
C4, C128, and indexer components, without relying on released full-token pages.

The target should answer:

1. Can retained C4/C128/indexer components survive after old full/SWA head pages
   are released?
2. Can prefix hit metadata use direct component locs rather than `full_loc //
   ratio`?
3. Can this run behind an explicit opt-in while phase-1 radix prefix cache
   remains the rollback path?
4. What capacity is recovered before compression-state ownership is fully
   solved?

## Required Reading

- `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md`
- `prompts/TARGET_08.21.1_dsv4_sm80_component_loc_table_preflight.md`
- `performance_milestones/target08_component_loc_table_preflight/README.md`
- `performance_milestones/target08_component_loc_table_preflight/DESIGN.md`
- `performance_milestones/target08_prefix_cache_memory_ledger/README.md`

## Implementation Scope

Allowed:

- add a new explicit opt-in, for example
  `--enable-dsv4-component-loc-ownership`;
- add independent C4/C128/indexer component free-lists or refcounts;
- add DSV4-specific radix node/component handles if needed;
- extend cache metrics with component live/protected/evictable counts;
- release old full/SWA head pages only when retained component handles are safe;
- keep one page-aligned SWA/full tail per retained branch;
- disable graph metadata deforest for this opt-in if B3 is not ready;
- run eager-only first if graph metadata copy is not ready.

Not allowed:

- treating hit-time Route A materialization as the normal serving path;
- claiming compression-state retention is solved unless TARGET 08.21.3-level
  ownership exists;
- default promotion;
- low-precision work;
- broad attention-kernel optimization.

## Correctness And Guards

Use the shared TARGET 08.198 guarded oracle.

Additional required checks:

- component handles do not reference freed/reused full pages;
- no C4/C128/indexer stale reads after eviction pressure;
- no double free on:
  - request finish;
  - prefix insert overlap;
  - radix eviction;
  - failed allocation rollback.
- phase-1 path remains unchanged when Route B opt-in is off.

If compression state remains phase-1/full-owned in this target, the README must
state the exact safe hit-length or feature guard.  Do not silently depend on
released full/SWA locs for state.

## Required Tests

At minimum:

- full hit, partial hit, miss;
- repeated hit/evict cycle;
- multi-prefix branching;
- eviction pressure with full/SWA head reuse;
- page boundary `255/256/257/258`;
- C4 boundary;
- C128 boundary;
- indexer loc/table correctness;
- slot-pinned prefix-on/off logits and metadata;
- text smoke for shared-prefix and non-shared-prefix prompts;
- leak/refcount/integrity tests;
- capacity ledger before/after.

Graph can be guarded off with a clear B3 follow-up if eager correctness passes.

## Deliverables

Create:

```text
performance_milestones/target08_independent_compressed_indexer_ownership/
  README.md
  DESIGN_DELTA.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- opt-in name;
- ownership/refcount/free design;
- component loc table runtime schema;
- correctness results under guarded oracle;
- text smoke results;
- leak/refcount results;
- recovered-capacity table;
- graph/deforest guard status;
- decision: proceed to TARGET 08.21.3, split B1 follow-up, or reject.

## Decision Rules

Proceed to TARGET 08.21.3 if:

- retained C4/C128/indexer ownership is correct under eager guarded oracle;
- old full/SWA head pages can be released without stale component reads;
- capacity recovery is material or the remaining blocker is clearly compression
  state;
- Route B opt-in remains fail-safe and phase-1 rollback works.

Stop and report blocked if:

- component allocator/refcount design causes leaks or double frees;
- attention/indexer metadata cannot consume independent component locs after
  B0;
- correctness relies on runtime materialization copies as the normal path;
- state coupling prevents any safe retained-prefix hit.
