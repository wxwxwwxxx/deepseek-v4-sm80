# Correctness And Safety

| check | result | raw artifact |
| --- | --- | --- |
| focused kernel wrappers | pass, 5 tests | `raw/pytest_route_b_direct_graph_metadata_correctness.log` plus rerun console |
| broader metadata/cache/benchmark tests | pass, 84 tests | `raw/pytest_route_b_direct_graph_metadata_correctness.log` |
| text smoke | pass | `raw/text_smoke_route_b_direct_graph_metadata.json` |
| prefix-hit direct-only | pass | `raw/prefix_hit_route_b_direct_graph_metadata_direct_only/` |
| eviction pressure direct-only | pass | `raw/eviction_pressure_route_b_direct_graph_metadata/` |

Text smoke graph status:

| captured buckets | replay | eager |
| --- | ---: | ---: |
| `[16,8,4,2,1]` | 9 | 0 |

Prefix-hit fresh direct-only sanity:

| scenario | saved prefill | hit rate | replay/eager |
| --- | ---: | ---: | ---: |
| `prefix_full_hit_513_longout_bs4` | 1536 | 0.75 | 62/0 |

Eviction-pressure fresh direct-only sanity:

| scenario | status | evictions | evicted tokens | retained pages | replay/eager |
| --- | --- | ---: | ---: | ---: | ---: |
| `prefix_eviction_pressure_96req_wave16` | pass | 5 | 40960 | 32 | 6/0 |

The direct path writes `-1` for missing component pages and tombstoned full
pages.  The eviction run exercised Route B component ownership with prefix
evictions and completed without stale-read, double-free, leak, or lifecycle
errors.
