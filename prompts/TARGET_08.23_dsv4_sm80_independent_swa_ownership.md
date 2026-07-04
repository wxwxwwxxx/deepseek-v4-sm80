# TARGET 08.23: DSV4 Independent SWA Ownership

## Status

Conditional follow-up after TARGET 08.22.

Do not run this target by default.  Run it only if the final prefix promotion
gate shows that Route B is materially limited by the live full/SWA tail guard.

## Goal

Design and implement a SGLang-aligned independent SWA ownership model for
DeepSeek V4 Flash in mini-sglang, so Route B prefix hits no longer require a
live full-token tail page solely for SWA attention.

The desired outcome is to remove or substantially relax the current Route B
fixed point:

```text
matched boundary is valid only if the final matched node has a live full/SWA
tail page
```

This target is about SWA KV ownership, not C4/C128/indexer/state ownership.
Those are already handled by TARGET 08.21.2 and TARGET 08.21.3.

## Required Trigger Evidence

Before starting implementation, read TARGET 08.22 results and confirm at least
one of these is true:

- exact page-multiple prompt lengths such as `512` or `768` are common enough
  to hurt Route B serving performance;
- the live full/SWA tail consumes enough capacity to matter;
- Route B cannot be promoted mainly because SWA KV is still full-token-owned.

If TARGET 08.22 shows the SWA-tail guard cost is small, do not implement this
target.  Record that decision and continue with the next post-prefix bottleneck
instead.

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate/README.md`
- `performance_milestones/target08_route_b_graph_deforest_serving/README.md`
- `performance_milestones/target08_compression_state_ownership/README.md`
- `performance_milestones/target08_independent_compressed_indexer_ownership/README.md`

Mini references:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`

SGLang references:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`

## Design Questions

Answer these before implementation:

1. Can mini add a separate SWA physical namespace without breaking current full
   KV writes?
2. Should SWA be represented by a `full_to_swa_index_mapping`, an explicit
   `swa_page_table`, or a per-radix-node `SWAComponent` value?
3. What is the exact match-validator fixed point across full, SWA, C4, C128,
   indexer, and state components?
4. Which attention metadata fields must use SWA-owned locs instead of full locs?
5. How should tombstoned full pages interact with retained SWA pages under
   eviction and branch splits?
6. Can existing SWA attention kernels consume direct SWA loc/page metadata, or
   do they require kernel changes?
7. How much memory is saved, and how much metadata/copy overhead is added?

## Scope

Allowed:

- add an explicit opt-in, for example
  `--enable-dsv4-independent-swa-ownership`;
- add SWA component allocation/refcount/free lists;
- add full-to-SWA mapping or SWA page tables;
- extend radix node component handles with SWA ownership;
- update metadata construction to consume SWA-owned locs;
- update graph metadata copy for SWA-owned locs;
- add SGLang-aligned match validation;
- keep deforest guarded if it cannot safely consume SWA-owned metadata.

Not allowed:

- long-distance replay of thousands of tokens as the normal runtime path;
- low-precision changes;
- attention-kernel tuning unrelated to SWA metadata correctness;
- default promotion in the same target;
- removing the existing Route B path before this opt-in proves itself.

## Implementation Strategy

Prefer a conservative staged route:

1. Source parity note:
   - map SGLang `SWATokenToKVPoolAllocator`;
   - map `SWAComponent` tombstone behavior;
   - map match validator behavior.
2. Metadata preflight:
   - prove direct SWA loc/page tables reproduce current full-owned SWA metadata
     while full pages remain live.
3. Eager opt-in:
   - allocate/refcount SWA component pages independently;
   - retain SWA tail/window values on radix nodes;
   - release old full head pages without losing SWA readability.
4. Correctness:
   - exact page-multiple cases `512`, `768`, and neighbors `513`, `769`;
   - repeated hit/evict;
   - multi-prefix branching;
   - slot-pinned same-layout oracle;
   - text smoke.
5. Graph integration:
   - copy/stage SWA-owned metadata for buckets `[1,2,4,8,16]`;
   - report eager fallback if graph is not ready.

## Deliverables

Create:

```text
performance_milestones/target08_independent_swa_ownership/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- trigger evidence from TARGET 08.22;
- SGLang parity map;
- SWA ownership/refcount/free design;
- fixed-point match validator;
- correctness table;
- text smoke;
- capacity and hit-length table;
- graph replay/eager table;
- comparison against Route B with SWA-tail guard;
- final decision: keep opt-in, prepare promotion gate, or reject.

## Decision Rules

Independent SWA ownership is worth keeping if:

- it restores exact page-multiple hits without stale reads;
- it passes guarded correctness and text smoke;
- it avoids leaks, double frees, and stale SWA refs under eviction;
- graph replay is preserved or the eager fallback cost is acceptable;
- serving performance/capacity improves enough to justify the added complexity.

Reject or defer if:

- SWA metadata requires broad attention-kernel rewrites before correctness can
  be proven;
- reconstruction/replay cost erases prefix-cache wins;
- the memory saved is below noise compared with the added metadata overhead;
- TARGET 08.22 showed the SWA-tail guard was not a practical bottleneck.
