# TARGET 08.21.1 DSV4 Component Loc Table Preflight Design

Date: 2026-07-04

## Result

B0 proves that direct component loc tables can reproduce today's phase-1 DSV4
metadata while full pages remain live.  The probe does not allocate independent
component pages, does not free full/SWA pages, and does not add Route A runtime
materialization.

Decision: proceed to TARGET 08.21.2 for eager, opt-in independent
C4/C128/indexer ownership.

## Current Mini Coupling Map

Mini has one scheduler page table in the full-token namespace:

```text
ctx.page_table[req.table_idx, logical_token_pos] = full_token_loc
```

DSV4 component locs are derived from that table:

| component | current derivation | mini owner/call site |
| --- | --- | --- |
| SWA/full | `swa_loc = full_loc` | `DeepSeekV4KVCache._swa_buffer`, `store_swa()` |
| C4 cache | `c4_loc = full_loc // 4` at `(position + 1) % 4 == 0` | `compressed_locs_from_full_locs()`, `store_compressed()` |
| C128 cache | `c128_loc = full_loc // 128` at `(position + 1) % 128 == 0` | `compressed_locs_from_full_locs()`, `store_compressed()` |
| C4 indexer | `indexer_loc = full_loc // 4` | `c4_indexer_out_loc = c4_out_loc`, `store_indexer()` |
| C4 state | `(swa_loc // page_size) * 8 + (swa_loc % 8)` | `DSV4CompressStatePool.translate_from_swa_loc_to_state_loc()` |
| C128 state | `(swa_loc // page_size) * 128 + (swa_loc % 128)` | same state pool helper |
| indexer state | same as C4 state | indexer state pool helper |

Allocation/refcount is also coupled: `on_pages_allocated()` and
`on_token_indices_freed()` expand full pages, then update C4, C128, and indexer
refcounts via `full_locs // 4` and `full_locs // 128`.  That is why 08.20
rejected runtime V1: if full pages are freed, those derived component locs can
be reused underneath old radix nodes.

## Metadata Coupling

Python metadata construction currently assumes full-derived component locs:

| path | current assumption | B0 finding |
| --- | --- | --- |
| `_build_metadata()` output locs | `c4_out_loc = compressed_locs_from_full_locs(raw_out_loc, 4, positions)`, same for C128; `c4_indexer_out_loc = c4_out_loc` | replaceable by direct write-loc vectors |
| `_make_sparse_compressed_indices()` | raw C4 index -> full token endpoint -> `full // 4` | replaceable by direct C4 loc table gather |
| `_make_all_compressed_indices()` | raw C128 index -> full token endpoint -> `full // 128` | replaceable by direct C128 loc table gather |
| `DSV4IndexerMetadata` | `page_table=core.page_table`, `c4_page_size=page_size//4` | can become component indexer page table; current B0 value equals full page table |
| `select_indexer*()` | indexer fallback receives page table and component page size | no kernel rewrite if page table entries are component physical page ids |
| sparse attention | consumes `metadata.c4_sparse_page_indices` / `c128_page_indices` as flat component locs | no kernel rewrite; metadata source can change |
| graph replay copy | `copy_masked_compressed_locs()` recomputes `raw_out_loc // ratio` | B3 dependency, not eager B1 blocker |
| decode deforest | Triton helper builds page/C4/C128 indices from `ctx.page_table` | B3 dependency; eager B1 can disable/bypass deforest |

## Direct Component Loc Schema

B0 schema uses page-aligned component tables.  With `page_size=256`:

| table | dtype | shape | meaning |
| --- | --- | --- | --- |
| `swa_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 256]` | logical token position -> SWA loc. B0 identity with full loc. |
| `swa_page_table` or `full_to_swa_map` | `torch.int32` | `[metadata_rows, full_page_count]` or `[num_full_locs]` | logical page/full loc -> SWA page/loc. B0 identity. |
| `c4_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 64]` | logical C4 slot -> physical C4 loc. |
| `c4_page_table` | `torch.int32` | `[metadata_rows, full_page_count]` | logical full-page ordinal -> physical C4 component page. |
| `c128_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 2]` | logical C128 slot -> physical C128 loc. |
| `c128_page_table` | `torch.int32` | `[metadata_rows, full_page_count]` | logical full-page ordinal -> physical C128 component page. |
| `c4_indexer_loc_table` | `torch.int32` | `[metadata_rows, full_page_count * 64]` | logical indexer/C4 slot -> physical indexer loc. |
| `c4_indexer_page_table` | `torch.int32` | `[metadata_rows, full_page_count]` | indexer page table consumed with `page_size//4 == 64`. |
| `state_loc_placeholder` | metadata contract | same logical coverage as SWA | current state loc is derived from direct SWA loc; independent state ownership is deferred to B2. |

B0 populates component page tables with the current full physical page ids, so:

```text
c4_loc    = c4_page_table[logical_full_page] * 64 + c4_page_offset
c128_loc  = c128_page_table[logical_full_page] * 2 + c128_page_offset
index_loc = c4_indexer_page_table[logical_full_page] * 64 + c4_page_offset
swa_loc   = swa_page_table[logical_full_page] * 256 + token_page_offset
```

When full pages remain live, this is exactly equivalent to:

```text
c4_loc    = full_loc // 4
c128_loc  = full_loc // 128
index_loc = full_loc // 4
swa_loc   = full_loc
```

## Probe Scope

Script:

```bash
python performance_milestones/target08_component_loc_table_preflight/scripts/probe_component_loc_table.py
```

The script builds synthetic full-token page tables with non-zero, row-distinct
physical pages, then constructs:

- phase-1 metadata via the current full-derived formulas;
- direct component tables from the same live full pages;
- direct metadata by gathering the direct tables, without using `full_loc // ratio`
  at the metadata-consumer boundary.

Covered scenarios:

| scenario | coverage |
| --- | --- |
| `full_hit_decode_page257` | full hit style, first post-page decode token |
| `partial_hit_suffix_prefill_256_to_258` | cached 256-token page plus suffix prefill |
| `miss_style_prefill_258` | no cached prefix |
| `page_boundaries_255_256_257_258_decode` | page boundary 255/256/257/258 |
| `c4_boundaries_decode` | before/on/after C4 endpoints |
| `c128_boundaries_decode` | before/on/after C128 endpoints |
| `swa_127_128_129_decode` | SWA window below/equal/above 128 |
| `batched_same_layout_rows` | two rows with same logical layout and different physical pages |

Compared fields:

- `swa_page_indices`;
- `c4_sparse_page_indices`;
- `c128_page_indices`;
- `c4_out_loc`;
- `c128_out_loc`;
- `c4_indexer_out_loc`;
- `indexer_page_table`;
- `indexer_loc_table_gather`;
- C4/C128/indexer state loc derivation from SWA loc.

State is a placeholder because current mini does not pass state loc through
attention metadata and no model path currently calls the state pool helpers
outside tests.  The formula is still validated against the direct SWA loc table
so B2 has a concrete starting contract.

## SGLang Parity Map

| SGLang concept | Mini B0 equivalent | Later target |
| --- | --- | --- |
| `SWATokenToKVPoolAllocator.full_to_swa_index_mapping` | `swa_page_table` / `full_to_swa_map`, identity in B0 | B1/B3 |
| `SWAComponent.value` and tombstone `None` | per-node SWA loc values are not implemented; schema can carry direct SWA locs | B3 |
| `ComponentData.value`, `lock_ref`, component LRU | no runtime component ownership in B0 | B1/B3 |
| `UnifiedRadixCache.create_match_validator()` fixed point | B0 only proves loc representability | B3 |
| `HiSparseC4DevicePool.translate_loc_to_hisparse_device()` | `c4_loc_table` / `c4_page_table` | B1 |
| `DeepSeekV4IndexerPool` with `page_size//4` pages | `c4_indexer_loc_table` / `c4_indexer_page_table` | B1 |
| `CompressStatePool.translate_from_swa_loc_to_state_loc()` | state placeholder formula from direct SWA loc | B2 |

## B1/B2/B3 Call-Site Dependencies

| target | call site | change required |
| --- | --- | --- |
| B1 | `DeepSeekV4KVCache.on_pages_allocated()` / `on_token_indices_freed()` | introduce independent C4/C128/indexer ownership/refcounts behind an opt-in; stop deriving component ownership from released full pages |
| B1 | `DSV4AttentionBackend._build_metadata()` | accept/build `DSV4ComponentLocTables`; source C4/C128/indexer locs from direct tables |
| B1 | `_make_sparse_compressed_indices()` / `_make_all_compressed_indices()` | gather direct component loc tables instead of endpoint full locs then divide |
| B1 | `DSV4IndexerMetadata` and `select_indexer*()` | pass component indexer page table/loc table; existing logits/topk kernels can consume page ids if they are component page ids |
| B1 | `store_compressed()` / `store_indexer()` | mostly unchanged; they already consume explicit loc vectors |
| B2 | `DSV4CompressStatePool` and compressor state boundary | choose independent state owner, boundary reconstruction, or safe hit-length guard |
| B3 | `decode_metadata_deforest_fallback()` and Triton deforest helper | add direct component table inputs or bypass deforest under Route B until ported |
| B3 | `copy_masked_compressed_locs()` and graph hook | copy/stage direct C4/C128/indexer write locs instead of recomputing `raw_out_loc // ratio` |
| B3 | `_empty_decode_metadata()` / `_copy_metadata_for_replay()` / `copy_decode_metadata_for_replay()` | allocate and copy direct component table/page fields |
| B3 | radix cache insertion/match/eviction | store component loc values and validate fixed point across full/SWA/C4/C128/indexer/state |

## Route A Oracle

Route A retained-store materialization was not used.  The probe uses only live
full pages and direct component loc tables built from those live pages.  This
keeps B0 focused on metadata representability rather than runtime materialize
or remap behavior.

## Decision

Proceed to TARGET 08.21.2.

Reason:

- direct C4/C128/indexer/SWA loc tables exactly reproduce phase-1 metadata in
  all B0 probe scenarios;
- sparse attention kernels consume flat component loc tensors and do not require
  a broad rewrite for eager B1;
- indexer logits/topk paths can use a component page table with the existing
  page-table contract;
- graph replay and deforest assumptions are real, but they are B3 integration
  dependencies rather than eager B1 blockers;
- compression state ownership remains deliberately deferred to B2.
