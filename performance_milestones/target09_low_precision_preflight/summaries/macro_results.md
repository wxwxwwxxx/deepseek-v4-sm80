| scenario | status | elapsed_s | E2E tok/s | decode tok/s | prefill tok/s | graph replay/eager | prefix hit | saved prefill | peak alloc | comm bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | pass | 9.646 | 53.080 | 189.291 | 3187.1 | 127/0 | 0.0% | 0 | 44.31 GiB | 93.43 GB |
| historical_4096_1024_bs4 | pass | 28.778 | 142.329 | 191.808 | 3825.4 | 1023/0 | 0.0% | 0 | 44.31 GiB | 93.43 GB |
| serving_mixed_112req_wave16 | pass | 15.230 | 183.843 | 301.657 | 4477.6 | 441/0 | 0.0% | 0 | 41.56 GiB | 100.08 GB |
| prefix_multi_112req_wave16 | pass | 6.596 | 135.848 | 680.745 | 4210.1 | 49/0 | 41.4% | 49152 | 42.89 GiB | 88.04 GB |
