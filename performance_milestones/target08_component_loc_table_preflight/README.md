# TARGET 08.21.1 DSV4 Component Loc Table Preflight

Date: 2026-07-04

## Result

Decision: **proceed to TARGET 08.21.2**.

The B0 probe shows that direct component loc tables can exactly reproduce the
current phase-1 metadata while full pages remain live.  This target did not add
an allocator, did not free old full/SWA pages, and did not add Route A runtime
materialization.

## Exact Commands

Commands run:

```bash
ruff check --fix \
  performance_milestones/target08_component_loc_table_preflight/scripts/probe_component_loc_table.py

ruff check \
  performance_milestones/target08_component_loc_table_preflight/scripts/probe_component_loc_table.py

python performance_milestones/target08_component_loc_table_preflight/scripts/probe_component_loc_table.py

python -m py_compile \
  performance_milestones/target08_component_loc_table_preflight/scripts/probe_component_loc_table.py

rm -rf \
  performance_milestones/target08_component_loc_table_preflight/scripts/__pycache__

git diff --check
git status --short
```

Probe output:

```json
{
  "all_equal": true,
  "decision": "proceed_to_TARGET_08.21.2",
  "scenario_count": 8
}
```

## Git Status Summary

This target adds only the milestone directory:

```text
?? performance_milestones/target08_component_loc_table_preflight/
```

## Artifacts

```text
performance_milestones/target08_component_loc_table_preflight/
  README.md
  DESIGN.md
  raw/component_loc_table_probe.json
  scripts/probe_component_loc_table.py
  summaries/equality_table.csv
  summaries/equality_table.md
```

## Direct Component Loc Schema

For B0, `page_size=256` and full pages remain live:

| table | dtype | shape | meaning |
| --- | --- | --- | --- |
| `swa_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 256]` | logical token position -> SWA/full loc |
| `swa_page_table` / `full_to_swa_map` | `torch.int32` | `[metadata_rows, full_page_count]` or `[num_full_locs]` | logical page/full loc -> SWA page/loc |
| `c4_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 64]` | logical C4 slot -> physical C4 loc |
| `c128_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 2]` | logical C128 slot -> physical C128 loc |
| `c4_indexer_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 64]` | logical indexer/C4 slot -> physical indexer loc |
| `c4_indexer_page_table` | `torch.int32` | `[metadata_rows, full_page_count]` | page table consumed with `page_size//4 == 64` |
| `state_loc_placeholder` | metadata contract | same logical coverage as SWA | state loc derived from direct SWA loc; ownership deferred to B2 |

B0 identity:

```text
swa_loc   = full_loc
c4_loc    = full_loc // 4
c128_loc  = full_loc // 128
index_loc = full_loc // 4
state_loc = (swa_loc // page_size) * ring_size + (swa_loc % ring_size)
```

## Equality Table

| scenario | phase | hit style | rows | swa | c4 sparse | c128 | c4 out | c128 out | indexer out | indexer page | indexer loc | state c4 | state c128 | state indexer | result |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_hit_decode_page257 | decode | full hit | 1 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| partial_hit_suffix_prefill_256_to_258 | prefill | partial hit | 2 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| miss_style_prefill_258 | prefill | miss style | 258 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| page_boundaries_255_256_257_258_decode | decode | boundary | 4 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| c4_boundaries_decode | decode | C4 boundary | 5 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| c128_boundaries_decode | decode | C128 boundary | 5 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| swa_127_128_129_decode | decode | SWA boundary | 3 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| batched_same_layout_rows | prefill | batched same-layout | 4 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |

Raw results are in `raw/component_loc_table_probe.json`; generated CSV/Markdown
tables are in `summaries/`.

## Current Mini Coupling

| area | current `full_loc // ratio` dependency |
| --- | --- |
| KV pool ownership | `on_pages_allocated()` and `on_token_indices_freed()` derive C4/C128/indexer refcounts from expanded full pages |
| compressed write locs | `compressed_locs_from_full_locs()` filters boundary positions and returns `full_loc // ratio` |
| sparse metadata | `_make_sparse_compressed_indices()` and `_make_all_compressed_indices()` gather full endpoint locs then divide |
| indexer | `DSV4IndexerMetadata.page_table = core.page_table`; indexer uses `page_size//4` |
| graph replay | `copy_masked_compressed_locs()` recomputes C4/C128 write locs from `raw_out_loc // ratio` |
| deforest | decode metadata deforest builds page/C4/C128 indices from `ctx.page_table` |
| state | `DSV4CompressStatePool.translate_from_swa_loc_to_state_loc()` derives state loc from SWA/full loc |

## SGLang Parity Map

| SGLang concept | B0 mini mapping | next dependency |
| --- | --- | --- |
| `SWATokenToKVPoolAllocator.full_to_swa_index_mapping` | `swa_page_table` / `full_to_swa_map`, identity while full pages live | B1/B3 |
| `SWAComponent.value` tombstone model | schema can represent direct SWA locs; runtime tombstones not implemented | B3 |
| `ComponentData.value` / `lock_ref` / component LRU | not implemented in B0 | B1/B3 |
| `HiSparseC4DevicePool` compressed mapping | `c4_loc_table` / `c4_page_table` | B1 |
| `DeepSeekV4IndexerPool` separate component page size | `c4_indexer_loc_table` / `c4_indexer_page_table` | B1 |
| `CompressStatePool.translate_from_swa_loc_to_state_loc()` | placeholder state loc formula from direct SWA loc | B2 |
| `UnifiedRadixCache` match validator fixed point | not implemented in B0; call-site table below maps dependency | B3 |

## B1/B2/B3 Dependency Table

| target | call site | required change |
| --- | --- | --- |
| B1 | KV pool allocation/free | add independent C4/C128/indexer ownership/refcounts behind an opt-in |
| B1 | `DSV4AttentionBackend._build_metadata()` | build/consume direct component loc tables |
| B1 | sparse C4/C128 metadata helpers | gather direct component locs instead of full endpoints then divide |
| B1 | `DSV4IndexerMetadata` / `select_indexer*()` | pass component indexer page table; existing indexer kernels can consume it |
| B1 | compressed/indexer store wrappers | mostly unchanged, because they already consume explicit loc vectors |
| B2 | compression state pool/state boundary | choose independent state owner, reconstruction, or safe hit-length guard |
| B3 | decode metadata deforest | add component table inputs or bypass under Route B until ported |
| B3 | graph replay copy/staging | stop recomputing `raw_out_loc // ratio`; copy/stage direct write locs |
| B3 | `_empty_decode_metadata()` / replay metadata copy | allocate/copy direct component table fields |
| B3 | radix cache match/insert/evict | store component loc values and validate a component-safe fixed point |

## Route A Oracle

No Route A oracle was used.  The probe builds direct component loc tables from
live full pages and compares them to phase-1 metadata.  It does not materialize
retained component data on hit.

## Go / No-Go

Go for TARGET 08.21.2.

Reason: direct C4/C128/indexer/SWA loc tables exactly reproduce phase-1
metadata, attention sparse kernels already consume flat component loc tensors,
and indexer kernels can keep the same page-table contract if the page table is
component-owned.  Graph replay, deforest, and compression-state ownership are
real dependencies, but they map cleanly to B2/B3 rather than blocking eager B1.
