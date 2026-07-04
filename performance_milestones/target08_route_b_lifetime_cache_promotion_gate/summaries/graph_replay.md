# Graph Replay

| run | scenario | captured bs | requested bs | replay/eager | verifier |
| --- | --- | --- | --- | --- | --- |
| decode_ladder_lifetime | decode_ladder_bs16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 63/0 | False |
| prefix_eviction_r01_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | False |
| prefix_eviction_r02_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | False |
| prefix_multi_r01_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| prefix_multi_r02_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| prefix_multi_r03_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| profile_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | False |
| profile_prefix_multi_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| profile_serving_mixed_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| serving_mixed_r01_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| serving_mixed_r02_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| serving_mixed_r03_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| verify_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | True |
| verify_serving_mixed_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | True |
