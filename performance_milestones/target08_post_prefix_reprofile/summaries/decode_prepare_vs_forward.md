# Decode Prepare vs Forward

| variant | scenario | decode prepare s | decode forward s | prepare share | prefill forward s | prefill prepare s |
| --- | --- | --- | --- | --- | --- | --- |
| promoted_prefix | decode_ladder_bs16 | 0.1577 | 1.3665 | 0.1034 | 1.3516 | 0.0977 |
| promoted_prefix | historical_4096_1024_bs4 | 2.2245 | 22.2977 | 0.0907 | 4.2998 | 0.7325 |
| promoted_prefix | historical_4096_128_bs4 | 0.2814 | 2.7684 | 0.0923 | 4.2998 | 0.7266 |
| promoted_prefix | prefix_eviction_pressure_96req_wave16 | 0.1464 | 0.1714 | 0.4606 | 10.8178 | 2.2478 |
| promoted_prefix | prefix_multi_112req_wave16 | 0.2779 | 1.4750 | 0.1585 | 4.5400 | 0.8640 |
| promoted_prefix | serving_mixed_112req_wave16 | 1.0919 | 9.7624 | 0.1006 | 4.7832 | 0.6850 |
| target07_control | decode_ladder_bs16 | 0.1295 | 1.3297 | 0.0888 | 1.3874 | 0.0712 |
| target07_control | historical_4096_1024_bs4 | 1.9793 | 22.0498 | 0.0824 | 4.2955 | 0.7034 |
| target07_control | historical_4096_128_bs4 | 0.2525 | 2.7258 | 0.0848 | 4.2913 | 0.7184 |
| target07_control | prefix_eviction_pressure_96req_wave16 | 0.0175 | 0.1453 | 0.1075 | 10.4995 | 2.0154 |
| target07_control | prefix_multi_112req_wave16 | 0.1324 | 1.1799 | 0.1009 | 13.5254 | 2.5801 |
| target07_control | serving_mixed_112req_wave16 | 0.9130 | 9.2552 | 0.0898 | 4.7665 | 0.5424 |
