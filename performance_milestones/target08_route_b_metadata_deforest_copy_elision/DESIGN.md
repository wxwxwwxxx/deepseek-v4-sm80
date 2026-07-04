# TARGET 08.24 Route B metadata deforest/copy elision design

## Scope

This milestone targets DSV4 SM80 Route B radix/SWA prefix-cache decode metadata
overhead.  It keeps Route B opt-in, keeps the 08.22 ownership model, and does
not change SWA ownership, SWA KV reconstruction, numeric precision, or attention
kernel algorithms.

The working hypothesis from 08.22 was that Route B was losing output throughput
mainly in decode metadata preparation and graph staging/copy, not in the SWA-tail
guard.  This change therefore makes the guarded metadata path component-aware
and adds field-level copy counters so the gate can prove or reject that
hypothesis.

## Attribution model

Owner timing now emits two counter families when `MINISGL_DSV4_OWNER_TIMING=1`:

- `dsv4.metadata_build.bytes` and `dsv4.metadata_build.calls`
- `dsv4.replay_metadata_copy.bytes` and `dsv4.replay_metadata_copy.calls`

Each counter carries `phase`, `rows`, `field`, and `stable` metadata.  The
stability buckets used by the summary are:

- `per-token`: `raw_out_loc`, `positions`, `seq_lens`, `req_table_indices`,
  SWA lengths/indices, C4 sparse lengths/indices, write locations.
- `per-request`: request sequence lengths, extend lengths, full page table, and
  Route B component page tables.
- `per-prefix-hit`: C128 raw/component/full indices, because these grow with
  retained prefix length rather than only with decode batch size.
- `per-bucket`: `cu_seqlens_q`, because graph replay copies a bucket-shaped
  prefix of the static buffer.

The milestone summary script aggregates these counters into:

- `summaries/metadata_build_bytes.md`
- `summaries/metadata_build_calls.md`
- `summaries/replay_copy_bytes.md`
- `summaries/replay_copy_calls.md`

This makes each gate run answer which fields copy every step, which fields are
stable enough to move toward request/hit/bucket reuse, and whether the largest
cost is still metadata/copy.

## Component-aware deforest

The old decode deforest helper was safe for phase1 because compressed component
locations were derived from full KV locations with arithmetic such as
`full_loc // 4` and `full_loc // 128`.  Route B invalidates that assumption:
full/SWA pages and C4/C128/indexer component pages can have independent physical
ownership.  Under eviction, a full page can be tombstoned while component pages
remain live, or a component page can be missing while the full page is still
present.

The new opt-in path consumes Route B metadata directly:

```text
component_raw = floor((position + 1) / ratio) - 1
logical_page = component_raw // component_page_size
offset = component_raw % component_page_size
component_loc = component_page_table[row, logical_page] * component_page_size + offset
```

For decode index metadata:

- full and SWA indices still read `ctx_page_table` and produce `-1` when the
  full/SWA page is tombstoned.
- C4/C128 component indices read `c4_page_table` and `c128_page_table`.
- Missing component pages produce `-1`; they never fall back to `full_loc //
  ratio`.

For replay write locations:

- C4, C128, and C4-indexer write locs are generated from `c4_page_table`,
  `c128_page_table`, and `c4_indexer_page_table`.
- Non-boundary decode rows and tombstoned/missing component pages are written as
  `-1`.
- The helper returns `False` on missing, non-CUDA, non-contiguous, wrong dtype,
  or under-sized inputs, so Python falls back to the existing safe path outside
  graph capture.

## Copy elision shape

This is intentionally conservative.  It does not introduce a new lifetime owner
or make default promotion decisions.  Instead it removes the Route B guard from
decode deforest only when component tables are available and valid, and it
reduces graph replay staging by replacing repeated Python compact/copy work with
one Triton pass for component write locations.

The expected movement is:

- fewer temporary `clone`/`cat`/indexed tensors in decode metadata build,
  because `page_table`, SWA indices, C4 sparse indices, and C128 indices are
  emitted by one component-aware kernel;
- fewer Python-side graph staging synchronizations for write locs, because the
  Route B replay path no longer needs a `mask.sum().item()` sized compact copy;
- clearer evidence for any remaining large copies, because the owner timing
  counters report bytes and calls by field.

The first implementation still copies active rows into graph buffers before
replay.  Stable fields are measured and classified here; deeper reuse of
per-request/per-hit rows is left behind the gate, because it needs more lifetime
bookkeeping than this milestone should add.

## Correctness guardrails

The Route B path is fail-safe by construction:

- component deforest is opt-in through
  `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`;
- component tables are required to be CUDA, contiguous, int32, two-dimensional,
  and row-aligned with the decode batch;
- tombstoned full/SWA pages and missing component pages produce `-1`;
- no stale component read is synthesized from full KV location arithmetic;
- replay write-loc generation validates all destination buffers before using the
  Triton helper and falls back to the previous path when safe to do so.

Focused kernel tests cover a tombstoned full page with live/missing component
pages and verify that component-aware deforest matches the oracle.  A second
kernel test verifies replay write loc generation from component page tables.

## Gate

The milestone gate compares four modes:

- `prefix_off`
- `phase1_prefix_on`
- `route_b_graph_baseline`
- `route_b_metadata_deforest`

The Route B deforest mode must keep text/correctness/graph passing, preserve the
08.22 saved-prefill behavior, avoid new eager fallback for graph buckets
`[1, 2, 4, 8, 16]`, and either reduce large-wave decode prepare overhead by at
least 50% or recover output throughput to at least 0.90x phase1.
