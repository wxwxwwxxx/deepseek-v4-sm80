# Throughput By Workload

| group | scenario | runs | output tok/s | stdev | decode prepare s | decode forward s | graph replay/eager | saved prefill | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving_mixed | serving_mixed_112req_wave16 | 3 | 163.7220 | 1.2721 | 1.1359 | 9.8927 | 441/0 | 0 | 0 |
| prefix_multi | prefix_multi_112req_wave16 | 3 | 105.4163 | 7.6928 | 0.2868 | 1.9164 | 49/0 | 49152 | 0 |
| prefix_eviction | prefix_eviction_pressure_96req_wave16 | 2 | 13.0260 | 0.0873 | 0.1537 | 0.1917 | 6/0 | 0 | 3 |
| decode_ladder | decode_ladder_bs16 | 1 | 98.3116 | 0.0000 | 0.1639 | 1.6786 | 63/0 | 0 | 0 |
