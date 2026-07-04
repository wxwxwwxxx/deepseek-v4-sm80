# Prefix And Eviction Metrics

| run | scenario | hits | saved prefill | evictions | evicted tokens | retained pages |
| --- | --- | --- | --- | --- | --- | --- |
| decode_ladder_lifetime | decode_ladder_bs16 | 0 | 0 | 0 | 0 | 0 |
| prefix_eviction_r01_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| prefix_eviction_r02_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| prefix_multi_r01_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| prefix_multi_r02_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| prefix_multi_r03_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| profile_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| profile_prefix_multi_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| profile_serving_mixed_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| serving_mixed_r01_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| serving_mixed_r02_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| serving_mixed_r03_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| verify_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| verify_serving_mixed_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
