# Graph Coverage

| kind | run | variant | scenario | requested | captured | replay/eager | replay by padded | eager by bs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro | macro_promoted_decode_ladder_bs16_r01 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_promoted_decode_ladder_bs16_r02 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_promoted_decode_ladder_bs16_r03 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_promoted_historical_4096_1024_bs4_r01 | promoted_prefix | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_promoted_historical_4096_1024_bs4_r02 | promoted_prefix | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_promoted_historical_4096_1024_bs4_r03 | promoted_prefix | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_promoted_historical_4096_128_bs4_r01 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_promoted_historical_4096_128_bs4_r02 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_promoted_historical_4096_128_bs4_r03 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r01 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r02 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r03 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_promoted_prefix_multi_112req_wave16_r01 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_promoted_prefix_multi_112req_wave16_r02 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_promoted_prefix_multi_112req_wave16_r03 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_promoted_serving_mixed_112req_wave16_r01 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_promoted_serving_mixed_112req_wave16_r02 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_promoted_serving_mixed_112req_wave16_r03 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_control_decode_ladder_bs16_r01 | target07_control | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_control_decode_ladder_bs16_r02 | target07_control | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_control_decode_ladder_bs16_r03 | target07_control | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_control_historical_4096_1024_bs4_r01 | target07_control | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_control_historical_4096_1024_bs4_r02 | target07_control | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_control_historical_4096_1024_bs4_r03 | target07_control | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_control_historical_4096_128_bs4_r01 | target07_control | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_control_historical_4096_128_bs4_r02 | target07_control | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_control_historical_4096_128_bs4_r03 | target07_control | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_control_prefix_eviction_pressure_96req_wave16_r01 | target07_control | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_control_prefix_eviction_pressure_96req_wave16_r02 | target07_control | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_control_prefix_eviction_pressure_96req_wave16_r03 | target07_control | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_control_prefix_multi_112req_wave16_r01 | target07_control | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_control_prefix_multi_112req_wave16_r02 | target07_control | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_control_prefix_multi_112req_wave16_r03 | target07_control | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_control_serving_mixed_112req_wave16_r01 | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_control_serving_mixed_112req_wave16_r02 | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_control_serving_mixed_112req_wave16_r03 | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| profile | profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| profile | profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| profile | profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| profile | profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| profile | profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| profile | profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| verify | verify_promoted_prefix_eviction | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| verify | verify_promoted_serving_mixed | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
