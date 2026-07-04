# TARGET 08.25 Direct Graph Metadata Buffer Design

## Scope

This milestone follows TARGET 08.24.  The 08.24 component-aware metadata formula
was safe, but it still built large eager source tensors and then copied or staged
them into CUDA graph replay buffers.  This design keeps Route B opt-in and writes
selected decode metadata directly into the final graph-consumed buffers.

Out of scope:

- independent SWA ownership;
- SWA KV reconstruction;
- lower precision metadata;
- attention kernel algorithm changes;
- default Route B promotion;
- scheduler-scale dirty-row tracking.

## Graph Metadata Buffer Lifecycle

The captured graph owns a static `DSV4AttentionMetadata` instance built during
capture.  Its `core_metadata` tensors are allocated in
`DSV4AttentionBackend._make_capture_metadata` with bucket-shaped dimensions
derived from `max_seq_len`, page size, SWA window, C4 `index_topk`, and C128
alignment.

On eager decode, `_build_metadata` creates a source `DSV4AttentionMetadata` for
the active padded batch.  Before graph replay, `prepare_for_replay` calls
`_copy_metadata_for_replay(capture, source, padded_size)`, which updates the
capture buffers for the active rows.  The captured graph then reads those static
capture buffers directly.

Captured graph direct readers include:

- scalar/vector rows: `raw_out_loc`, `positions`, `seq_lens`,
  `req_seq_lens`, `extend_lens`, `req_table_indices`;
- prefix/bucket rows: `cu_seqlens_q`, full `page_table`;
- SWA indices: `swa_page_indices`, `swa_topk_lengths`;
- C4 sparse metadata: `c4_sparse_raw_indices`,
  `c4_sparse_page_indices`, `c4_sparse_full_indices`,
  `c4_sparse_topk_lengths`, `c4_topk_lengths_*`;
- C128 metadata: `c128_raw_indices`, `c128_page_indices`,
  `c128_full_indices`, `c128_topk_lengths_clamp1`;
- Route B component page tables: `c4_page_table`, `c128_page_table`,
  `c4_indexer_page_table`;
- write locations: `c4_out_loc`, `c128_out_loc`, `c4_indexer_out_loc`.

Fields still copied from source metadata:

- scalar/vector rows and lengths;
- `cu_seqlens_q`;
- full `page_table`;
- Route B component page tables;
- component write locations.

Fields generated directly to graph buffers with the 08.25 opt-in:

- `swa_page_indices`;
- `c4_sparse_raw_indices`;
- `c4_sparse_page_indices`;
- `c4_sparse_full_indices`;
- `c128_raw_indices`;
- `c128_page_indices`;
- `c128_full_indices`.

Inputs for direct generation:

- `positions`;
- `req_table_indices`;
- full page table from the global request table;
- Route B `c4_page_table` and `c128_page_table`;
- page size, SWA window, and C4 `index_topk`.

## Direct Generation Semantics

SWA:

```text
swa_index[row, offset] = full_page_table[req_table_idx, position - offset]
```

Invalid negative positions, out-of-table positions, or tombstoned full pages
write `-1`.

C4 sparse:

```text
c4_len = (position + 1) // 4
c4_start = max(c4_len - index_topk, 0)
raw = c4_start + offset
full_pos = raw * 4 + 3
page_loc = c4_page_table[row, raw // c4_component_page_size] * c4_component_page_size
         + raw % c4_component_page_size
```

C128:

```text
c128_len = (position + 1) // 128
raw = offset
full_pos = raw * 128 + 127
page_loc = c128_page_table[row, raw // c128_component_page_size] * c128_component_page_size
         + raw % c128_component_page_size
```

C4/C128 page locs never derive component ownership from full loc arithmetic.
Missing component pages write `-1`.  Full index fields independently write `-1`
when the full page table is tombstoned.

## Source Elision

`_should_elide_index_source_for_graph` only enables source elision when:

- the batch is decode;
- Route B component ownership is active;
- CUDA graph capture exists and the padded bucket was captured;
- `MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1`;
- the SM80 Triton helper is enabled.

When enabled, `_build_metadata` replaces large eager matrices for direct fields
with `(rows, 1)` int32 placeholders filled with `-1`.  `_copy_metadata_for_replay`
skips those fields in the fused replay-copy kernel and invokes
`direct_decode_index_metadata_for_replay` against the destination graph buffers.

If direct generation fails after source elision, replay raises.  If source
elision did not happen, replay can fall back to copying the existing source
oracle fields.

## Stable Row Attempt

This milestone stops short of scheduler-wide dirty-row tracking.  It preserves
the existing per-step component page-table copy and write-loc update behavior.
Owner timing shows these remaining copied rows are much smaller than the removed
index matrices, but eliminating them correctly would need request-slot lifetime
bookkeeping across prefix hit, eviction, and graph bucket changes.

## Instrumentation

Owner timing has three relevant byte/call families:

- `dsv4.metadata_build.*` for eager source construction;
- `dsv4.replay_metadata_copy.*` for source-to-graph staging;
- `dsv4.direct_graph_metadata.*` for bytes written directly to graph buffers.

The profile run is intentionally separate from throughput runs because owner
timing slows large-wave decode.

## Decision

The implementation satisfies correctness and graph replay constraints, but it
does not meet the large-wave performance threshold.  Direct generation should
remain an explicit experimental diagnostic path.
