| scenario | phase1 saved | Route B saved | delta saved | phase1 hits | Route B hits | TTFT delta s |
| --- | --- | --- | --- | --- | --- | --- |
| decode_ladder_bs16 | 0 | 0 | 0 | 0 | 0 | -0.0085 |
| serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 0 | 0.0345 |
| prefix_full_hit_257_bs4 | 768 | 768 | 0 | 3 | 3 | 0.0101 |
| prefix_full_hit_512_bs4 | 768 | 0 | -768 | 3 | 0 | 0.1318 |
| prefix_full_hit_513_bs4 | 1536 | 1536 | 0 | 3 | 3 | 0.0090 |
| prefix_full_hit_768_bs4 | 1536 | 0 | -1536 | 3 | 0 | 0.3122 |
| prefix_full_hit_769_bs4 | 2304 | 2304 | 0 | 3 | 3 | 0.0090 |
| prefix_full_hit_513_longout_bs4 | 1536 | 1536 | 0 | 3 | 3 | 0.0048 |
| prefix_partial_hit_769_bs8 | 1792 | 1792 | 0 | 7 | 7 | 0.0413 |
| prefix_mixed_hit_miss_bs16 | 6144 | 6144 | 0 | 8 | 8 | 0.0832 |
| prefix_multi_112req_wave16 | 49152 | 49152 | 0 | 96 | 96 | 0.0977 |
| prefix_eviction_pressure_96req_wave16 | 0 | 0 | 0 | 0 | 0 | 0.1871 |
