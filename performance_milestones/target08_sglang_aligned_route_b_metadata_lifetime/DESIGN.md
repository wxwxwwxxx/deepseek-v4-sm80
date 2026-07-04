# TARGET 08.27 Design: SGLang-Aligned Route B Metadata Lifetime

## Scope

This milestone targets one narrow Route B decode bottleneck in mini-sglang:
`python/minisgl/attention/deepseek_v4.py::_make_component_page_tables`.
The default path remains unchanged. The new path is gated by:

- `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1`
- optional oracle: `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1`

The benchmark variant that enables it is
`dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime`.

## SGLang And vLLM Parity Map

| System | File | Mature behavior | Mini implication |
| --- | --- | --- | --- |
| SGLang | `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py` | Decode can begin with `DSV4RawDecodeMetadata(req_pool_indices, seq_lens, out_cache_loc)`. When `SGLANG_PREP_IN_CUDA_GRAPH` is enabled, raw metadata is copied into graph buffers and expanded in graph prep. | Do not keep rebuilding derived decode metadata from request objects if a stable request mapping table can own the lifetime. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py` | Full decode `page_table` is derived from stable `req_to_token[req_pool_indices_repeated, :max_seq_len:page_size] // page_size`. | Candidate A should mirror a stable row table keyed by request slot, not a new ad hoc per-step dirty-row architecture. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py` | `DSV4AttnMetadata.copy_` copies tensor fields whose graph kernels read by address, while `refresh_for_breakable_cuda_graph_replay_` reference-assigns non-address-captured metadata such as page tables/FlashMLA metadata. | Mini must distinguish captured-address buffers from ordinary Python metadata references. For current mini graphs, component page-table destination addresses are captured, so this milestone reduces source construction only. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py` | `copy_metadata` enforces field coverage; nested metadata has explicit copy/reference rules. | Any mini opt-in must keep an oracle path and complete field comparison. The verify env rebuilds the old tables and checks equality. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` | `CudaGraphBufferRegistry` owns stable graph-resident buffers for `ForwardBatch` fields and fills them from eager batches. Backend-private buffers stay backend-owned. | Mini's component table cache belongs in the attention backend, while graph replay still copies into captured metadata buffers. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py` | Prefix cache nodes store component data, locks, protected/ejectable accounting, tombstone handling, and safe match boundaries. | Mini already has Route B component ownership in radix/cache manager; this milestone should consume those stable handles rather than changing scheduler/cache ownership. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py` | `ComponentData` carries component value, lock refs, host value, and callbacks for match/insert/evict lifecycle. | Component rows should be invalidated by request/cache-handle lifecycle signals, not by guessing from tensor addresses alone. |
| SGLang | `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py` | SWA component values can be tombstoned independently from full cache values; insert/match validation enforces window safety. | Do not change SWA ownership in this milestone. Route B component page tables only read existing component handles/mappings. |
| vLLM | `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py` | Decode attention reads metadata from forward context; DSV4 SWA/C4/C128 metadata is generated against block tables and slot mappings. | Keep metadata generation tied to stable block/request tables, not to transient request object scans. |
| vLLM | `/workspace/vllm-dsv4-docker/vllm/v1/core/block_pool.py` | Prefix-cached block IDs stay allocated and block tables are effectively append-only while blocks are live. | Mini should use stable per-request rows and update only on slot reuse/prefix/page growth. |
| vLLM | `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_coordinator.py` | Hybrid KV cache groups are coordinated around stable group/block allocation contracts. | Avoid a large scheduler/KV rewrite for this target; use the existing mini component ownership contract. |

## Mini Field Lifecycle Table

| Mini field | Current owner/lifetime | Graph boundary | 08.27 change |
| --- | --- | --- | --- |
| `table_indices` / `req.table_idx` | Scheduler request table slot; rebuilt per batch. | Copied into captured graph input metadata. | Unchanged. Used as stable row selector. |
| full `page_table` | Global context `page_table`; request rows persist while table slot is live. | Still built/copied every decode step. | Unchanged for this milestone. |
| `c4_page_table` | Previously rebuilt from `cache_handle` plus active full-page mappings every decode step. | Captured graph reads backend-owned destination addresses. | New opt-in keeps persistent backend rows by `table_idx`; decode selects rows into the old source tensor shape. |
| `c128_page_table` | Same as C4. | Same as C4. | Same row-cache behavior. |
| `c4_indexer_page_table` | Same as C4. | Same as C4. | Same row-cache behavior. |
| `c4/c128/swa` sparse/index tensors | Built by metadata routines; some C4 tensors can be written directly into graph buffers by TARGET 08.26. | Existing direct-C4 path writes C4 sparse index buffers into captured graph metadata. | Unchanged; new variant composes with direct C4. |
| `raw_out_loc`, `c4_out_loc`, `c128_out_loc` | Per-token decode write locations. | Address-captured graph fields; copied or direct component-table path. | Unchanged. |
| `seq_lens`, `positions`, top-k lengths | Per-token/per-request batch metadata. | Address-captured graph fields. | Unchanged. |
| `req.cache_handle` component pages | Radix/cache-manager Route B component lifecycle. | Not directly captured. | Read only when a row signature changes. |
| full-to-component page mappings | `DeepSeekV4KVCache` active-page mapping. | Not directly captured. | Read only when a request row refreshes. |

## Captured Address vs Reference Assign

SGLang can reference-assign selected metadata in
`refresh_for_breakable_cuda_graph_replay_` because the graph replay contract knows
which fields are captured by address and which are looked up through refreshed
Python metadata.

Mini currently captures kernels against `self.capture.core_metadata` buffers. For
component page tables, replay still copies into `dst_core.c4_page_table`,
`dst_core.c128_page_table`, and `dst_core.c4_indexer_page_table`. Replacing those
with a Python reference assignment would not update captured kernel addresses.

Therefore the safe 08.27 boundary is:

- Persistent source rows: yes.
- Per-step `index_select` source tensor with old shape: yes.
- Copy into captured graph destination buffers: unchanged.
- Reference-assign component page tables into a captured graph: deferred to a
  later graph metadata contract target.

## Candidate Choice

### A: Persistent request/component mapping table

Chosen. It is the closest mini equivalent of SGLang's stable `req_to_token`
lifetime:

- rows are keyed by `req.table_idx`;
- row signatures include request uid, table slot, cache-handle cached length,
  radix node UUID, and logical page count;
- row contents are rebuilt by the old Route B builder only when the signature
  changes;
- decode steps inside the same page reuse rows.

This attacks the measured 08.26 hot spot directly: component page-table owner
time in Route B direct C4 was `3341.1692 ms`.

### B: Raw metadata plus graph prep refresh contract

Deferred. SGLang's raw metadata path is attractive, but a correct mini port
would require a broader graph-prep contract for full/page/component metadata and
clear captured-address rules across attention metadata. That is larger than this
milestone and would risk scheduler/graph churn.

### C: Minimal dirty-row cache

Not chosen as the architecture. The implementation has local row signatures, but
only as the invalidation mechanism inside Candidate A. It does not introduce a
separate dirty-row subsystem or scheduler rewrite.

## Why This Is Not From Scratch

The implementation follows mature ownership patterns:

- SGLang's `req_to_token` style stable request rows motivate `table_idx` keyed
  persistent component rows.
- SGLang's CUDA graph buffer registry motivates keeping graph-resident buffers
  separate from backend-private derived buffers.
- SGLang component cache files motivate respecting component locks/handles and
  tombstone semantics instead of taking over SWA/component ownership.
- vLLM's block pool and KV cache coordinator reinforce the stable-block-table
  model for cached prefixes.

The old mini builder remains the source of truth. The opt-in path calls the same
row construction routine on refresh, and `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1`
compares cached output against the uncached builder.

## Opt-In And Rollback

Default behavior is unchanged. Rollback is one env/variant change:

- Disable `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE`.
- Use `dsv4_sm80_a100_victory_directgraphmetadata_c4` for the TARGET 08.26
  direct-C4 path.
- Use `dsv4_sm80_a100_victory` with `--enable-dsv4-component-loc-ownership` for
  the Route B graph baseline.

Safety probes:

- `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1` rebuilds the
  uncached oracle and raises on mismatch.
- The cache reallocates and clears signatures on row-count, width, C4/C128 group
  shape changes.
- Table-slot reuse is detected by `req.uid` and `req.table_idx`.
- Prefix handle movement is detected by cached length and radix node UUID.
- Active-page growth is detected by logical page count.

## Known Boundaries

- The captured graph destination buffers are still copied every replay; this is
  why replay component page-table copy time is almost unchanged.
- Full `page_table` construction is unchanged.
- SWA ownership, low precision, attention kernels, MoE, and NCCL behavior are
  untouched.
- Broader eviction stress and multi-workload promotion remain for the later
  gate.
