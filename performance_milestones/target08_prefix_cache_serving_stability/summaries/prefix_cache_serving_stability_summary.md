# TARGET 08.10 Prefix Cache Serving Stability Summary

## Correctness
| scenario | off/on | outputs match | requests | full | partial | miss | evict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefix_full_hit_257_bs4 | pass/pass | no | 4 | 3 | 0 | 1 | 0 |
| prefix_partial_hit_769_bs8 | pass/pass | no | 8 | 0 | 7 | 1 | 0 |
| prefix_mixed_hit_miss_bs16 | pass/pass | no | 16 | 8 | 0 | 8 | 0 |
| prefix_multi_112req_wave16 | pass/pass | no | 112 | 96 | 0 | 16 | 0 |
| prefix_eviction_pressure_96req_wave16 | pass/pass | yes | 96 | 0 | 0 | 96 | 5 |

## Serving Workloads
| mode | scenario | hit rate | saved | TTFT s | prefill s | decode s | out tok/s | replay/eager |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefix_off | prefix_full_hit_257_bs4 | 0.000 | 0 | 2.569 | 5.273 | 0.135 | 2.898 | 6/0 |
| prefix_on | prefix_full_hit_257_bs4 | 0.750 | 768 | 1.628 | 3.526 | 0.135 | 4.257 | 6/0 |
| prefix_off | prefix_partial_hit_769_bs8 | 0.000 | 0 | 3.654 | 3.889 | 0.309 | 13.590 | 14/0 |
| prefix_on | prefix_partial_hit_769_bs8 | 0.875 | 1792 | 3.592 | 4.113 | 0.306 | 13.819 | 14/0 |
| prefix_off | prefix_mixed_hit_miss_bs16 | 0.000 | 0 | 4.632 | 6.424 | 0.316 | 17.589 | 14/0 |
| prefix_on | prefix_mixed_hit_miss_bs16 | 0.500 | 6144 | 2.931 | 3.141 | 0.312 | 34.108 | 14/0 |
| prefix_off | prefix_multi_112req_wave16 | 0.000 | 0 | 2.460 | 14.746 | 1.315 | 47.769 | 49/0 |
| prefix_on | prefix_multi_112req_wave16 | 0.857 | 49152 | 0.976 | 6.194 | 1.309 | 107.009 | 49/0 |
| prefix_off | prefix_eviction_pressure_96req_wave16 | 0.000 | 0 | 2.304 | 11.947 | 0.162 | 13.640 | 6/0 |
| prefix_on | prefix_eviction_pressure_96req_wave16 | 0.000 | 0 | 1.901 | 9.508 | 0.164 | 16.467 | 6/0 |

## Long Text Smoke
| mode | status | text | tokens | hit rate | saved | pages | retained GiB | replay/eager |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefix_off | pass | OK | [[11932]] | 0.000 | 0 | 0 | 0.000 | 2/0 |
| prefix_on | pass | OK | [[11932]] | 0.500 | 768 | 3 | 0.054 | 2/0 |

long text outputs match: yes; texts match: yes

## Memory Retention
| mode | scenario | pages | tokens | retained GiB | full | C4 | C128 | indexer | evicted tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefix_on | prefix_full_hit_257_bs4 | 1 | 256 | 0.018 | 256 | 64 | 2 | 64 | 0 |
| prefix_on | prefix_partial_hit_769_bs8 | 16 | 4096 | 0.288 | 4096 | 1024 | 32 | 1024 | 0 |
| prefix_on | prefix_mixed_hit_miss_bs16 | 40 | 10240 | 0.719 | 10240 | 2560 | 80 | 2560 | 0 |
| prefix_on | prefix_multi_112req_wave16 | 56 | 14336 | 1.007 | 14336 | 3584 | 112 | 3584 | 0 |
| prefix_on | prefix_eviction_pressure_96req_wave16 | 112 | 28672 | 2.015 | 28672 | 7168 | 224 | 7168 | 34816 |

## Decision Inputs
| check | value |
| --- | --- |
| all_reports_passed | yes |
| off_on_outputs_match | no |
| long_text_smoke_outputs_match | yes |
| long_text_smoke_texts_match | yes |
| long_text_smoke_prefix_hit_requests | 1 |
| full_hits_observed | yes |
| partial_hits_observed | yes |
| misses_observed | yes |
| evictions_observed | yes |
| prefix_on_total_eager_decode | 0 |
| prefix_on_total_graph_replay | 89 |
| prefix_on_total_saved_prefill_tokens | 57856 |
| max_retained_prefix_pages | 112 |
| max_retained_memory_gib | 2.015 |
