# TARGET 08.26: DSV4 Route B Remaining Gap Attribution Reset

## Status

Run this after TARGET 08.25.

TARGET 08.24 and TARGET 08.25 both produced safe experimental paths, but neither
met the performance gate:

- TARGET 08.24 component-aware deforest was correct but slower because it still
  materialized large source metadata tensors and staged them into graph buffers.
- TARGET 08.25 direct graph metadata buffers removed the intended source/copy
  bytes, but large-wave throughput still did not improve enough.

The next step is an attribution reset, not another speculative metadata
optimization.

## Goal

Identify the real owner of the remaining Route B versus phase-1 prefix-cache
gap after direct graph-buffer metadata generation.

This target should answer which path should be optimized next:

- stable-row / dirty-row metadata lifetime tracking;
- component page-table construction/copy;
- graph replay/runtime prepare;
- attention, MoE, communication, or other decode forward work;
- independent SWA ownership;
- or global post-prefix reprofile in TARGET 08.30.

## Background

TARGET 08.25 large-wave result:

| mode | output tok/s | decode tok/s | decode prepare s | decode forward s | graph replay/eager |
| --- | ---: | ---: | ---: | ---: | --- |
| phase1 prefix on | 169.6261 | 269.7954 | 0.9370 | 9.9631 | 441/0 |
| Route B graph baseline | 134.4667 | 262.3821 | 4.4707 | 10.2446 | 441/0 |
| Route B direct C4 | 136.7244 | 260.4445 | 4.2564 | 10.3208 | 441/0 |
| Route B direct SWA+C4+C128 | 128.4799 | 251.1565 | 3.8700 | 10.7025 | 441/0 |

Interpretation:

- C4 direct generation is safe and gives a small local win.
- Full SWA+C4+C128 direct generation removes the intended metadata source/copy
  bytes but regresses output throughput.
- The remaining large-wave gap is no longer explained by those copied index
  matrices alone.

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.24_dsv4_sm80_route_b_metadata_deforest_copy_elision.md`
- `prompts/TARGET_08.25_dsv4_sm80_route_b_direct_graph_metadata_buffers.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/README.md`
- `performance_milestones/target08_route_b_direct_graph_metadata_buffers/README.md`
- `performance_milestones/target08_route_b_direct_graph_metadata_buffers/DESIGN.md`
- `performance_milestones/target08_route_b_direct_graph_metadata_buffers/summaries/large_wave_ab.md`
- `performance_milestones/target08_route_b_direct_graph_metadata_buffers/summaries/metadata_counters.md`
- `performance_milestones/target08_route_b_direct_graph_metadata_buffers/summaries/probe_c4_effect.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/README.md`

Core code references:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

## Scope

Allowed:

- add or refine timing counters, NVTX labels, and summary scripts;
- run focused macro/profile comparisons;
- run nsys when it helps separate CPU/runtime, CUDA graph replay, kernel, and
  NCCL owners;
- add tiny diagnostic switches if needed to isolate one owner;
- keep 08.24/08.25 experimental opt-ins as comparison variants.

Not allowed:

- implementing stable-row/dirty-row lifetime tracking beyond a tiny diagnostic
  probe;
- independent SWA ownership;
- low-precision work;
- attention/MoE/communication optimization;
- default promotion of 08.24 or 08.25 paths;
- large scheduler refactors.

## Required Comparisons

Use separate `torchrun` invocations per variant.

Compare at least:

- phase1 prefix on;
- Route B graph baseline;
- Route B direct C4;
- Route B direct SWA+C4+C128;
- optionally prefix off as a non-prefix control.

Use the promoted base path:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Primary scenarios:

- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`;
- `decode_ladder_bs16`;
- `prefix_eviction_pressure_96req_wave16`.

If runtime is limited, run `serving_mixed_112req_wave16` first and add the other
three only after the instrumentation is readable.

## Attribution Questions

Answer these with tables:

1. How much of the Route B gap is in decode prepare versus decode forward?
2. Within decode prepare, how much belongs to:
   - `_make_component_page_tables`;
   - full `page_table` construction;
   - component page-table copy/staging;
   - direct graph metadata kernels;
   - fallback metadata construction;
   - graph input binding/copy;
   - Python/runtime overhead not accounted by CUDA timers?
3. Within decode forward, which owners changed between phase1 and Route B:
   - attention;
   - indexer;
   - compressor/write-loc stores;
   - MoE;
   - communication/all-reduce/all-gather;
   - graph replay overhead;
   - other wrapper calls?
4. Does C4-only direct generation still show a stable win in unprofiled repeat
   runs?
5. Does full SWA+C4+C128 direct generation slow decode forward because of extra
   kernels, memory pressure, changed graph capture, or measurement noise?
6. Are per-request page tables or per-prefix-hit C128 rows updated every token
   often enough to justify stable-row/dirty-row tracking?
7. Is SWA-tail guard still too small to matter in these workloads?

## Instrumentation Expectations

Prefer small, targeted counters:

- host wall time around metadata build and replay prepare;
- CUDA-event timing for component page-table build/copy;
- direct metadata kernel time by field group;
- graph replay copy bytes/calls by field;
- NCCL byte/count summaries;
- top kernel/NVTX summaries from nsys if available.

Keep profiling overhead separate from throughput runs.  Do not use owner-timing
runs as final throughput evidence.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_remaining_gap_attribution_reset/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands and variants;
- git status summary;
- throughput table;
- prepare versus forward attribution table;
- decode-prepare owner table;
- decode-forward owner table;
- graph replay/eager table;
- SWA-tail impact recap;
- recommendation for the next target.

## Decision Rules

Recommend a stable-row/dirty-row target only if:

- per-request or per-prefix-hit metadata update/copy is a top remaining owner;
- a small diagnostic shows request-slot reuse would avoid meaningful work;
- the design can be implemented without a scheduler rewrite.

Recommend moving to TARGET 08.30 if:

- metadata/runtime is no longer a dominant owner;
- the remaining gap is split across attention/MoE/communication/graph runtime;
- Route B is good enough as a preferred opt-in and needs global post-prefix
  reprofile.

Recommend TARGET 10 attention/communication work if:

- decode forward, attention, MoE, or communication dominates after prepare is
  separated;
- there is a clear `>=2%` E2E opportunity.

Recommend returning to TARGET 08.23 independent SWA ownership only if:

- SWA-tail guard or SWA retention is a measured top capacity/hit-rate limiter;
- the measured impact exceeds the metadata/runtime gap.

## Stop Rules

Stop and report instead of optimizing if:

- the owner ranking is unstable across repeat runs;
- profiling overhead prevents credible comparison;
- a proposed fix requires broad scheduler or attention rewrites;
- no single owner has enough weight to justify a focused follow-up target.

