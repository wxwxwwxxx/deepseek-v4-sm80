# Workload Throughput

| variant | scenario | runs | out tok/s | stdev | CV | TTFT ms | TPOT/ITL ms | prefill fwd s | decode prep s | decode fwd s | graph | saved | evict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| promoted_prefix | decode_ladder_bs16 | 3 | 130.7605 | 0.1441 | 0.0011 | 1452.4240 | 30.5793 | 1.3516 | 0.1577 | 1.3665 | 63/0 | 0 | 0 |
| promoted_prefix | historical_4096_1024_bs4 | 3 | 137.1625 | 0.1960 | 0.0014 | 5048.6030 | 24.2519 | 4.2998 | 2.2245 | 22.2977 | 1023/0 | 0 | 2 |
| promoted_prefix | historical_4096_128_bs4 | 3 | 62.8933 | 0.1391 | 0.0022 | 5041.8038 | 24.3931 | 4.2998 | 0.2814 | 2.7684 | 127/0 | 0 | 1 |
| promoted_prefix | prefix_eviction_pressure_96req_wave16 | 3 | 13.0827 | 0.0250 | 0.0019 | 2273.9510 | 171.6111 | 10.8178 | 0.1464 | 0.1714 | 6/0 | 0 | 3 |
| promoted_prefix | prefix_multi_112req_wave16 | 3 | 110.1417 | 0.3680 | 0.0033 | 854.9569 | 43.8182 | 4.5400 | 0.2779 | 1.4750 | 49/0 | 49152 | 0 |
| promoted_prefix | serving_mixed_112req_wave16 | 3 | 163.3985 | 0.5651 | 0.0035 | 796.7467 | 31.8282 | 4.7832 | 1.0919 | 9.7624 | 441/0 | 0 | 0 |
| target07_control | decode_ladder_bs16 | 3 | 135.5353 | 3.0061 | 0.0222 | 1461.2713 | 26.9086 | 1.3874 | 0.1295 | 1.3297 | 63/0 | 0 | 0 |
| target07_control | historical_4096_1024_bs4 | 3 | 139.8415 | 0.1143 | 0.0008 | 5001.6229 | 23.7401 | 4.2955 | 1.9793 | 22.0498 | 1023/0 | 0 | 0 |
| target07_control | historical_4096_128_bs4 | 3 | 63.7732 | 0.0965 | 0.0015 | 5012.5199 | 23.7412 | 4.2913 | 0.2525 | 2.7258 | 127/0 | 0 | 0 |
| target07_control | prefix_eviction_pressure_96req_wave16 | 3 | 15.0270 | 0.0612 | 0.0041 | 2088.8871 | 40.2401 | 10.4995 | 0.0175 | 0.1453 | 6/0 | 0 | 0 |
| target07_control | prefix_multi_112req_wave16 | 3 | 51.0507 | 0.1588 | 0.0031 | 2304.1904 | 28.9489 | 13.5254 | 0.1324 | 1.1799 | 49/0 | 0 | 0 |
| target07_control | serving_mixed_112req_wave16 | 3 | 178.3004 | 0.1715 | 0.0010 | 760.6074 | 26.8178 | 4.7665 | 0.9130 | 9.2552 | 441/0 | 0 | 0 |
