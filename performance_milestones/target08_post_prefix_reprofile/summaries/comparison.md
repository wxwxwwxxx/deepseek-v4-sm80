# Historical Comparison

| source | variant | scenario | out tok/s | decode prep s | decode fwd s | graph | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TARGET 07.79 non-prefix | target07_control | historical_4096_1024_bs4 | 131.7561 |  |  | 0 eager | from prompts/target.md |
| TARGET 07.79 non-prefix | target07_control | historical_4096_128_bs4 | 62.3925 |  |  | 0 eager | from prompts/target.md |
| old vLLM baseline | vLLM old serving line | 4096_1024_bs4_serving_line | 114.0700 |  |  |  | historical old serving victory line |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | serving_mixed_112req_wave16 | 163.7220 | 1.1359 | 9.8927 | 441/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | prefix_multi_112req_wave16 | 105.4163 | 0.2868 | 1.9164 | 49/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | prefix_eviction_pressure_96req_wave16 | 13.0260 | 0.1537 | 0.1917 | 6/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | decode_ladder_bs16 | 98.3116 | 0.1639 | 1.6786 | 63/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.30 current | promoted_prefix | decode_ladder_bs16 | 130.7605 | 0.1577 | 1.3665 | 63/0 | CV=0.0011 |
| TARGET 08.30 current | promoted_prefix | historical_4096_1024_bs4 | 137.1625 | 2.2245 | 22.2977 | 1023/0 | CV=0.0014 |
| TARGET 08.30 current | promoted_prefix | historical_4096_128_bs4 | 62.8933 | 0.2814 | 2.7684 | 127/0 | CV=0.0022 |
| TARGET 08.30 current | promoted_prefix | prefix_eviction_pressure_96req_wave16 | 13.0827 | 0.1464 | 0.1714 | 6/0 | CV=0.0019 |
| TARGET 08.30 current | promoted_prefix | prefix_multi_112req_wave16 | 110.1417 | 0.2779 | 1.4750 | 49/0 | CV=0.0033 |
| TARGET 08.30 current | promoted_prefix | serving_mixed_112req_wave16 | 163.3985 | 1.0919 | 9.7624 | 441/0 | CV=0.0035 |
| TARGET 08.30 current | target07_control | decode_ladder_bs16 | 135.5353 | 0.1295 | 1.3297 | 63/0 | CV=0.0222 |
| TARGET 08.30 current | target07_control | historical_4096_1024_bs4 | 139.8415 | 1.9793 | 22.0498 | 1023/0 | CV=0.0008 |
| TARGET 08.30 current | target07_control | historical_4096_128_bs4 | 63.7732 | 0.2525 | 2.7258 | 127/0 | CV=0.0015 |
| TARGET 08.30 current | target07_control | prefix_eviction_pressure_96req_wave16 | 15.0270 | 0.0175 | 0.1453 | 6/0 | CV=0.0041 |
| TARGET 08.30 current | target07_control | prefix_multi_112req_wave16 | 51.0507 | 0.1324 | 1.1799 | 49/0 | CV=0.0031 |
| TARGET 08.30 current | target07_control | serving_mixed_112req_wave16 | 178.3004 | 0.9130 | 9.2552 | 441/0 | CV=0.0010 |
