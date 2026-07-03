# TARGET 08.21: DSV4 SGLang-Aligned Component Retention V2

## Status

Planned after TARGET 08.20 only if V1 succeeds.

This is the second-stage component-retention target.  It should align mini with
SGLang's mature/default SWA component behavior as much as is practical, without
adding an aggressive long-replay SWA reconstruction scheme.

## Goal

Move from a conservative V1 slice to a robust SGLang-aligned component retention
model for DSV4 prefix cache.

The target should answer:

1. Can mini support independent lifetimes for full/SWA, C4, C128, indexer, and
   compression-state components?
2. Can prefix hit matching compute a safe fixed point across those components?
3. Can tombstoned SWA branches recover only the required page-aligned SWA tail
   without retaining the whole historical SWA region?
4. Is the resulting implementation stable enough to become the preferred
   prefix-cache opt-in, while default promotion still waits for correctness and
   serving gates?

## What This Target Explicitly Avoids

Do not implement a scheme that replays thousands of tokens only to rebuild SWA.
For mini's current serving targets, replaying a large fraction of a prefix can
erase the prefix-cache benefit.  Prefer SGLang's component/tombstone/tail
retention model:

```text
keep reusable compressed components,
tombstone out-of-window SWA,
retain or restore only the page-aligned SWA tail needed by the sliding window.
```

## Source References

Primary SGLang references:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/common.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`

Mini inputs:

- TARGET 08.19 correctness report;
- TARGET 08.20 V1 report and design;
- `python/minisgl/kvcache/radix_cache.py`;
- `python/minisgl/kvcache/deepseek_v4_pool.py`;
- `python/minisgl/scheduler/cache.py`;
- `python/minisgl/attention/deepseek_v4.py`.

## Required Design

Define a mini component model with:

- component values for full/SWA tail, C4, C128, indexer, and state;
- tombstone representation for evicted SWA;
- protected vs evictable accounting per component;
- component-level lock/unlock semantics;
- page-aligned `swa_evicted_seqlen` or equivalent frontier;
- safe prefix-hit fixed point across all required components;
- metrics for recovered pages, tombstones, and retained state.

The design should explicitly document where it follows SGLang and where mini
chooses a smaller local equivalent.

## Implementation Scope

Allowed:

- replace or extend V1 opt-in internals;
- add component-level radix metadata;
- add independent component metrics and leak checks;
- add recovery for page-aligned SWA tail;
- add fixed-point prefix match validation;
- add a rollback path to phase-1 and V1 behavior.

Not allowed:

- default promotion in this target;
- low-precision KV/cache work;
- graph bucket expansion;
- CUDA graph private-pool attribution;
- broad serving scheduler redesign unrelated to component retention.

## Required Tests

Use TARGET 08.19 correctness methodology and TARGET 08.10 serving workloads.

At minimum:

- full hit, partial hit, miss;
- repeated hit/evict;
- multi-prefix branching;
- eviction pressure;
- SWA boundary around `128`;
- page boundary around `256`;
- C4/C128/indexer boundaries;
- logits comparison at suffix prefill and first decode;
- graph replay/eager counts;
- memory ledger before/after V2.

## Deliverables

Create:

```text
performance_milestones/target08_sglang_aligned_component_retention_v2/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- SGLang parity map;
- V1 to V2 delta;
- correctness table;
- component memory/capacity table;
- serving A/B table;
- known limitations;
- decision: keep V2 opt-in, reject, or use as candidate for later promotion.

## Decision Rules

Keep V2 as the preferred opt-in if:

- logits/metadata correctness passes;
- capacity recovery is materially better than V1 or phase 1;
- graph replay coverage remains intact;
- performance is neutral or better on shared-prefix workloads;
- component metrics show no leaks or double frees.

Do not promote to default if the generated-token/logit correctness blocker from
TARGET 08.19 remains unresolved.

## Stop Rules

Stop and report blocked if:

- SGLang's component model cannot be mapped without a global allocator rewrite;
- compression state recovery is ambiguous;
- tombstone or lock/refcount behavior causes leaks or double frees;
- V2 reduces TTFT/prefill benefit enough that it defeats prefix-cache value.

## Non-Goals

- Long-distance SWA replay experiments.
- Low-precision research.
- Attention or communication optimization.
- General graph/workspace memory redesign.
