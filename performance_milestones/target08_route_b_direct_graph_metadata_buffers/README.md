# TARGET 08.25 DSV4 Route B Direct Graph Metadata Buffers

Date: 2026-07-04

## Result

Decision: **keep_experimental**.

The implementation adds an explicit opt-in:

```bash
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
```

Under Route B component ownership and CUDA graph replay, decode metadata for
`swa_page_indices`, `c4_sparse_*`, and `c128_*` can now be generated directly
into the captured graph replay buffers.  The eager Route B metadata path remains
the oracle/fallback/debug path, and this opt-in is not part of the A100 victory
bundle.

## What Changed

- Added `direct_decode_index_metadata_for_replay`, a component-aware Triton
  helper that writes graph-consumed SWA, C4 sparse, and C128 index buffers.
- Added replay-copy skip flags so direct fields are not copied from eager source
  tensors into graph buffers.
- Elided eager source construction for direct fields during decode graph replay
  by building tiny `-1` placeholders instead of large source matrices.
- Kept fail-closed behavior: if a source field was elided and direct generation
  fails, replay raises instead of reading stale graph data.
- Added owner-timing counters for `dsv4.direct_graph_metadata.*`.
- Added benchmark/text-smoke variants:
  `dsv4_sm80_a100_victory_directgraphmetadata`.

## Component Formula

Route B component page indices do not use `full_loc // 4` or `full_loc // 128`.
They use the component page table:

```text
logical_page = raw_index // component_page_size
offset = raw_index % component_page_size
component_loc = component_page_table[row, logical_page] * component_page_size + offset
```

Missing or tombstoned component pages write `-1`.  Full/SWA indices still read
the full page table; tombstoned full pages also produce `-1`.

## Gate Summary

| run | status | output tok/s | decode tok/s | decode prepare s | decode forward s | TTFT s | graph replay/eager | saved prefill |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 prefix on large | pass | 169.6261 | 269.7954 | 0.9370 | 9.9631 | 0.7682 | 441/0 | 0 |
| Route B graph baseline large | pass | 134.4667 | 262.3821 | 4.4707 | 10.2446 | 0.8235 | 441/0 | 0 |
| Route B direct C4 large | pass | 136.7244 | 260.4445 | 4.2564 | 10.3208 | 0.7985 | 441/0 | 0 |
| Route B direct SWA+C4+C128 large | pass | 128.4799 | 251.1565 | 3.8700 | 10.7025 | 0.9685 | 441/0 | 0 |

Large-wave direct SWA+C4+C128 reduced decode prepare by **13.4%** versus the
Route B graph baseline, below the 40% target.  It reached **0.757x** phase1
output throughput, below the 0.90x target, and regressed throughput versus the
Route B graph baseline.  Therefore this path is not promoted.

The C4-only probe did prove the high-value first cut: decode prepare fell from
0.6912s to 0.6417s on `decode_ladder_bs16`, C4 eager source/copy bytes dropped
to placeholder/zero, and graph replay stayed 63/0.

## Correctness And Safety

| check | result |
| --- | --- |
| focused kernel/unit tests | pass, 5 selected tests |
| broader metadata/cache/benchmark tests | pass, 84 tests |
| text smoke direct opt-in | pass, captured `[16,8,4,2,1]`, replay/eager 9/0 |
| prefix-hit direct-only sanity | pass, saved prefill 1536, hit rate 0.75, replay/eager 62/0 |
| eviction-pressure direct-only sanity | pass, 5 evictions, replay/eager 6/0 |

Raw logs are under `raw/`.  Summaries are under `summaries/`.

## Stop Decision

The direct graph-buffer path removed the intended eager materialization and
replay-copy bytes, but the broader large-wave profile is no longer dominated by
those copies.  Further wins likely require stable-row/dirty-row lifetime tracking
for page tables and prefix-hit rows, or attention/MoE side changes.  That would
exceed this milestone's boundary and risks becoming a scheduler rewrite, so this
milestone stops here with a safe experimental opt-in.
