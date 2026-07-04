# TARGET 08.27 Route B Metadata Lifetime

This milestone implements an experimental, opt-in Route B component page-table
lifetime cache aligned with SGLang's stable request-row metadata model.

## What Changed

- Added `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1`.
- Added `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1` as an
  oracle check against the old uncached builder.
- Added benchmark/text-smoke variant
  `dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime`.
- Kept the old Route B path as default and rollback.

## Correctness And Graph Replay

Artifacts:

- `raw/text_smoke_routeb_lifetime_verify.json`
- `raw/verify_serving_route_b_lifetime/`
- `summaries/correctness.md`

Results:

| check | status | graph replay/eager | note |
| --- | --- | --- | --- |
| Text smoke with cache verifier | pass | not reported by text smoke | outputs: `杭州西湖位于杭州市。`, `浙江省。`, `Blue.` |
| Full `serving_mixed_112req_wave16` with cache verifier | pass | 441/0 | no stale component-row mismatch |
| Unprofiled throughput repeats | pass | 441/0 | all three repeats full graph replay |

## Throughput

Runtime shape:

- `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- `page_size=256`
- `--num-pages 128`
- `--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16`
- workload: `serving_mixed_112req_wave16`

The first three rows are the frozen TARGET 08.26 comparison set; the final row
is the current TARGET 08.27 opt-in.

| mode | runs | output tok/s mean | stdev | decode tok/s mean | decode prepare s | decode forward s | graph replay/eager |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | 3 | 169.7381 | 0.8408 | 269.4540 | 0.9403 | 9.9757 | 441/0 |
| Route B graph baseline | 3 | 136.2373 | 0.4446 | 266.4154 | 4.4798 | 10.0897 | 441/0 |
| Route B direct C4 | 3 | 138.1281 | 0.7047 | 265.3675 | 4.2067 | 10.1297 | 441/0 |
| Route B direct C4 + lifetime cache | 3 | 162.4726 | 0.5952 | 268.5946 | 1.1416 | 10.0077 | 441/0 |

Delta:

- vs Route B graph baseline: `+19.26%` output tok/s.
- vs Route B direct C4: `+17.62%` output tok/s.
- vs phase1 prefix on: `-4.28%` output tok/s.

## Decode Prepare vs Forward

| mode | decode prepare s | decode forward s | prepare share of prepare+forward |
| --- | --- | --- | --- |
| phase1 prefix on | 0.9403 | 9.9757 | 8.61% |
| Route B graph baseline | 4.4798 | 10.0897 | 30.75% |
| Route B direct C4 | 4.2067 | 10.1297 | 29.34% |
| Route B direct C4 + lifetime cache | 1.1416 | 10.0077 | 10.24% |

Decode prepare is no longer the dominant Route B gap. It is still about
`0.20 s` above phase1, but the big per-step component table rebuild has been
removed.

## Component Page-Table Owner Timing

Owner timing includes profiling overhead and is for attribution only.

| mode | decode prepare s | host attention metadata ms | component table build ms | replay component table copy ms | direct C4 index buffers ms |
| --- | --- | --- | --- | --- | --- |
| Route B direct C4 | 4.4941 | 4304.6365 | 3341.1692 | 103.2981 | 71.1383 |
| Route B direct C4 + lifetime cache | 1.5189 | 1330.1929 | 354.4121 | 103.0957 | 66.3969 |

Row cache counters from the profile:

| counter | value |
| --- | --- |
| dirty component rows | 112 |
| clean component row reuses | 2576 |
| total decoded rows | 2688 |

The replay copy into captured graph addresses is intentionally unchanged in this
milestone; this result confirms the improvement comes from avoiding source table
rebuilds.

## Repro

Run all current milestone probes:

```bash
performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/scripts/run_route_b_metadata_lifetime.sh
```

Regenerate summaries from existing raw artifacts:

```bash
python performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/scripts/summarize_route_b_metadata_lifetime.py
```

Additional local checks run:

```bash
python -m py_compile \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py

pytest -q tests/benchmark/test_deepseek_v4_perf_matrix.py
```

`pytest` result: `25 passed`.

## Decision

Promote to a later gate as an experimental opt-in. Do not make it default yet.

Reasons:

- Correctness smoke and full workload oracle checks pass.
- Graph replay is stable at `441/0`.
- Route B direct C4 throughput improves from `138.1281` to `162.4726` tok/s.
- Component page-table build owner time drops from `3341.1692 ms` to
  `354.4121 ms`.
- Remaining gap to phase1 is now small enough to split into follow-ups:
  captured-destination/reference-assign contract, full page-table lifetime, and
  broader eviction/mixed-workload promotion tests.
