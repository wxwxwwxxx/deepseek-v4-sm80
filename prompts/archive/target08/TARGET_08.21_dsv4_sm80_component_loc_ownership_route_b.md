# TARGET 08.21: DSV4 Component-Loc Ownership Route B

## Status

TARGET 08 Route B family overview.

Do not run this file as one large monolithic implementation target.  Run the
subtargets in order, starting with TARGET 08.21.1.

TARGET 08.20 deliberately rejected runtime V1 and kept
`--enable-dsv4-swa-tail-retention-v1` fail-closed.  The reason is structural:
mini currently derives C4, C128, indexer, and compression-state locations from
full-token page locations.  Releasing old full/SWA pages while retaining those
derived components would risk stale reads, component slot reuse collisions, and
double frees.

This family changes direction from "conservative V1" to **Route B**:
component-level ownership and component loc metadata.  The goal is to align mini
with the SGLang direction where components can have independent lifetimes, not
to add hit-time materialization as the main runtime path.

Route A, retained-component store plus hit-time materialization/remap, is
allowed only as an oracle or small correctness harness inside the relevant
subtarget.  Do not build a full runtime Route A unless Route B is proven
impossible.

## Route B Subtargets

Run in this order:

| Stage | Prompt | Purpose |
| --- | --- | --- |
| TARGET 08.21.1 | `prompts/TARGET_08.21.1_dsv4_sm80_component_loc_table_preflight.md` | B0 preflight: prove direct component loc tables can reproduce phase-1 derived loc metadata while full pages are still live. |
| TARGET 08.21.2 | `prompts/TARGET_08.21.2_dsv4_sm80_independent_compressed_indexer_ownership.md` | B1 implementation slice: independent C4/C128/indexer component ownership behind an opt-in, initially eager/guarded if needed. |
| TARGET 08.21.3 | `prompts/TARGET_08.21.3_dsv4_sm80_compression_state_ownership.md` | B2 state slice: decide and implement independent/reconstructed/guarded compression-state ownership. |
| TARGET 08.21.4 | `prompts/TARGET_08.21.4_dsv4_sm80_route_b_graph_deforest_serving.md` | B3 integration: graph metadata copy, deforest, serving performance, capacity ledger, and promotion decision. |

## Shared Evidence

Read before any subtarget:

- `performance_milestones/target08_swa_tail_retention_v1/README.md`
- `performance_milestones/target08_swa_tail_retention_v1/DESIGN.md`
- `performance_milestones/target08_prefix_cache_memory_ledger/README.md`
- `performance_milestones/target08_post_layer0_same_shape_decode_drift/README.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/target.md`

Important facts:

- Phase-1 prefix cache works by retaining full-token pages as the canonical
  owner.
- Current mini component locations are derived:

```text
C4 loc        = full_loc // 4
C128 loc      = full_loc // 128
indexer loc   = full_loc // 4
state loc     = derived from SWA/full page loc and ring offset
```

- TARGET 08.20 rejected runtime V1 because this derivation makes released
  full/SWA pages unsafe.
- SGLang has separate full/SWA allocators, component data on radix nodes,
  per-component lock/ref accounting, SWA tombstones, and match validators.
- mini does not currently guarantee batch-slot invariance.  Use the TARGET
  08.198 guarded oracle: slot-pinned/same-layout correctness is pass/fail;
  cross-slot generated-token equality is diagnostic only.

## Shared References

Mini:

- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/common.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`

Specific SGLang concepts to map:

- `SWATokenToKVPoolAllocator` and `full_to_swa_index_mapping`;
- `SWAComponent` tombstone and window-bounded LRU refresh;
- `ComponentData.value`, `lock_ref`, and per-component LRU lists;
- `create_match_validator()` fixed-point match validation;
- `free_swa_out_of_window_slots()`;
- DSV4 `HiSparseC4DevicePool` and compressed/indexer pool separation.

## Shared Correctness Oracle

Use the guarded oracle from TARGET 08.198:

- pass/fail:
  - slot-pinned, same-layout prefix-on versus prefix-off;
  - metadata consistency;
  - no leaks, double frees, stale reads, or state corruption;
  - text smoke with no garbled/invalid-byte or degenerate output.
- diagnostics only:
  - cross-slot generated-token equality;
  - filler-content equality;
  - identical-row equality.
- report:
  - logits max/mean diff;
  - top-k overlap;
  - top1 margin;
  - sampled ids;
  - whether `2 * max_abs >= top1_margin`.

For decode step 1+, prefer teacher-forced/fixed-token probes.  If natural
autoregressive tokens diverge after decode0, label later drift as sampler
feedback rather than a new cache owner.

## Shared Non-Goals

- Complete SGLang `UnifiedRadixCache` parity in one step.
- Default prefix-cache promotion.
- Low-precision research.
- Attention/communication optimization.
- Graph private-pool memory analysis.
- Runtime Route A materialization as the normal serving path.
