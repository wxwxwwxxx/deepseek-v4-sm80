# TARGET 08.21.3: DSV4 Compression-State Ownership

## Status

Planned after TARGET 08.21.2.

This is Route B / B2.  Run after C4/C128/indexer component ownership is either
implemented or its remaining state dependency is clearly isolated.

## Goal

Define and implement the compression-state ownership rule for Route B.

The target should answer:

1. Which DSV4 compression states must survive prefix retention after old
   full/SWA pages are released?
2. Can state be independently allocated/refcounted?
3. Can state be reconstructed at page/C128-safe boundaries instead of retained?
4. What exact hit-length fixed point is safe when full/SWA tail, C4, C128,
   indexer, and state availability differ?

## Required Reading

- `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md`
- `prompts/TARGET_08.21.2_dsv4_sm80_independent_compressed_indexer_ownership.md`
- `performance_milestones/target08_independent_compressed_indexer_ownership/README.md`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`

## Scope

Allowed:

- add independent state loc tables/refcounts;
- add page/C128-boundary reconstruction if proven correct;
- add a conservative fixed-point rule that shortens prefix hits when state is
  unavailable;
- keep graph disabled for this opt-in until TARGET 08.21.4 if needed;
- add focused state probes and text/logit correctness tests.

Not allowed:

- claiming state is retained while it still depends on released full/SWA locs;
- long-distance replay of thousands of tokens to rebuild SWA/state;
- low-precision changes;
- default promotion.

## Required Analysis

Document:

- current state loc formula;
- C4 attention state ring size and boundary conditions;
- C4 indexer state ring size and boundary conditions;
- C128 state ring size and boundary conditions;
- which states are read during suffix prefill after a prefix hit;
- which states are written during decode;
- what page-aligned hit lengths avoid state ambiguity.

Compare options:

- independent state allocator/refcount;
- retained state only at safe page/C128 boundaries;
- reconstruction from a bounded warmup window;
- conservative fallback to phase-1/full-owned state.

## Required Tests

At minimum:

- prefix hits ending at page boundary `256`;
- hits around C4 and C128 boundaries;
- suffix prefill after retained prefix;
- first decode after suffix prefill;
- repeated hit/evict with state reuse;
- text smoke;
- guarded slot-pinned logits/metadata comparison;
- leak/refcount checks for state pools.

## Deliverables

Create:

```text
performance_milestones/target08_compression_state_ownership/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- chosen state ownership/reconstruction rule;
- safe fixed-point hit-length rule;
- correctness table;
- text smoke results;
- state memory/capacity table;
- remaining limitations;
- decision: proceed to TARGET 08.21.4, keep state guard, or reject Route B.

## Decision Rules

Proceed to TARGET 08.21.4 if:

- state correctness is proven or safely guarded;
- Route B no longer depends on released full/SWA locs for state;
- slot-pinned same-layout correctness passes;
- text smoke is clean.

Stop and report blocked if:

- state reads cannot be identified precisely;
- state reconstruction requires long replay that erases prefix-cache value;
- state ownership needs a broader allocator rewrite than Route B B1 can support.
