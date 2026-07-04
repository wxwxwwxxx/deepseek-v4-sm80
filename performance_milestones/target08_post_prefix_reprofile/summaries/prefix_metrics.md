# Prefix Metrics

| kind | run | scenario | hits | misses | saved prefill | evictions | evicted tokens | retained pages | retained MiB | SWA MiB | available comp pages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro | macro_promoted_decode_ladder_bs16_r01 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| macro | macro_promoted_decode_ladder_bs16_r02 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| macro | macro_promoted_decode_ladder_bs16_r03 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| macro | macro_promoted_historical_4096_1024_bs4_r01 | historical_4096_1024_bs4 | 0 | 4 | 0 | 2 | 9728 | 114 | 2099.7876 | 1225.5000 | 15 |
| macro | macro_promoted_historical_4096_1024_bs4_r02 | historical_4096_1024_bs4 | 0 | 4 | 0 | 2 | 9728 | 114 | 2099.7876 | 1225.5000 | 15 |
| macro | macro_promoted_historical_4096_1024_bs4_r03 | historical_4096_1024_bs4 | 0 | 4 | 0 | 2 | 9728 | 114 | 2099.7876 | 1225.5000 | 15 |
| macro | macro_promoted_historical_4096_128_bs4_r01 | historical_4096_128_bs4 | 0 | 4 | 0 | 1 | 4096 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_historical_4096_128_bs4_r02 | historical_4096_128_bs4 | 0 | 4 | 0 | 1 | 4096 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_historical_4096_128_bs4_r03 | historical_4096_128_bs4 | 0 | 4 | 0 | 1 | 4096 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r01 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r02 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r03 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_multi_112req_wave16_r01 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| macro | macro_promoted_prefix_multi_112req_wave16_r02 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| macro | macro_promoted_prefix_multi_112req_wave16_r03 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| macro | macro_promoted_serving_mixed_112req_wave16_r01 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| macro | macro_promoted_serving_mixed_112req_wave16_r02 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| macro | macro_promoted_serving_mixed_112req_wave16_r03 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| profile | profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| profile | profile_promoted_4096_128 | historical_4096_128_bs4 | 0 | 4 | 0 | 0 | 0 | 64 | 1178.8281 | 688.0000 | 65 |
| profile | profile_promoted_prefix_eviction_pressure_96req_wave16 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| profile | profile_promoted_prefix_multi_112req_wave16 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| profile | profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| verify | verify_promoted_prefix_eviction | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| verify | verify_promoted_serving_mixed | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
