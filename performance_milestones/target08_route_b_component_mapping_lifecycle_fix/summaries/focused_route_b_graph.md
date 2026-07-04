| scenario | status | saved delta | hits/matches | avg saved/hit | replay/eager | captured | error |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| prefix_full_hit_257_bs4 | pass | 768 | 3/4 | 256 | 6/0 | [16, 8, 4, 2, 1] |  |
| prefix_full_hit_512_bs4 | pass | 0 | 0/4 |  | 6/0 | [16, 8, 4, 2, 1] |  |
| prefix_full_hit_513_bs4 | pass | 1536 | 3/4 | 512 | 6/0 | [16, 8, 4, 2, 1] |  |
| prefix_full_hit_768_bs4 | pass | 0 | 0/4 |  | 6/0 | [16, 8, 4, 2, 1] |  |
| prefix_full_hit_769_bs4 | pass | 2304 | 3/4 | 768 | 6/0 | [16, 8, 4, 2, 1] |  |
| prefix_full_hit_513_longout_bs4 | pass | 1536 | 3/4 | 512 | 62/0 | [16, 8, 4, 2, 1] |  |
| prefix_partial_hit_769_bs8 | pass | 1792 | 7/8 | 256 | 14/0 | [16, 8, 4, 2, 1] |  |
| prefix_mixed_hit_miss_bs16 | pass | 6144 | 8/16 | 768 | 14/0 | [16, 8, 4, 2, 1] |  |
| prefix_multi_112req_wave16 | pass | 49152 | 96/112 | 512 | 49/0 | [16, 8, 4, 2, 1] |  |
