# Memory And Capacity Ledger

| variant | scenario | peak alloc GiB | peak reserved GiB | KV GiB/rank | graph delta GiB | retained pages | retained tokens | retained MiB | SWA MiB | C4 MiB | C128 MiB | available comp pages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| promoted_prefix | decode_ladder_bs16 | 41.4668 | 42.1797 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 129 |
| promoted_prefix | historical_4096_1024_bs4 | 44.3071 | 45.8359 | 2.3204 | 19.0137 | 114 | 29184 | 2099.7876 | 1225.5000 | 149.6250 | 4.4531 | 15 |
| promoted_prefix | historical_4096_128_bs4 | 44.3070 | 45.8359 | 2.3204 | 19.0137 | 112 | 28672 | 2062.9492 | 1204.0000 | 147.0000 | 4.3750 | 17 |
| promoted_prefix | prefix_eviction_pressure_96req_wave16 | 42.6890 | 43.6777 | 2.3204 | 19.0137 | 112 | 28672 | 2062.9492 | 1204.0000 | 147.0000 | 4.3750 | 17 |
| promoted_prefix | prefix_multi_112req_wave16 | 42.8853 | 43.9180 | 2.3204 | 19.0137 | 16 | 4096 | 294.7070 | 172.0000 | 21.0000 | 0.6250 | 113 |
| promoted_prefix | serving_mixed_112req_wave16 | 41.5562 | 42.4648 | 2.3204 | 19.0137 | 14 | 3584 | 257.8687 | 150.5000 | 18.3750 | 0.5469 | 115 |
| target07_control | decode_ladder_bs16 | 41.4668 | 42.1543 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | historical_4096_1024_bs4 | 44.3041 | 46.4570 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | historical_4096_128_bs4 | 44.3040 | 46.4570 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | prefix_eviction_pressure_96req_wave16 | 42.6885 | 43.9805 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | prefix_multi_112req_wave16 | 42.8850 | 44.2656 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | serving_mixed_112req_wave16 | 41.5561 | 42.4453 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
