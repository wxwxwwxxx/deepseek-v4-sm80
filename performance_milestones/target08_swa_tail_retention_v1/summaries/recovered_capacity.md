| case | phase-1 full/SWA pages | V1 full/SWA pages | phase-1 C4/C128/indexer slots | phase-1 state slots C4/C128/indexer | V1 C4/C128/indexer slots | V1 state slots C4/C128/indexer | recovered pages | recovered tokens | recovered GiB/rank | decision |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| short shared prefix | 1 | 1 | 64/2/64 | 8/128/8 | 64/2/64 | 8/128/8 | 0 | 0 | 0.000 | reject_runtime_retention |
| 1024-token prefix | 4 | 4 | 256/8/256 | 32/512/32 | 256/8/256 | 32/512/32 | 0 | 0 | 0.000 | reject_runtime_retention |
| 4096-token prefix | 16 | 16 | 1024/32/1024 | 128/2048/128 | 1024/32/1024 | 128/2048/128 | 0 | 0 | 0.000 | reject_runtime_retention |
| multi-prefix mixed | 40 | 40 | 2560/80/2560 | 320/5120/320 | 2560/80/2560 | 320/5120/320 | 0 | 0 | 0.000 | reject_runtime_retention |
| 08.10 sustained workload | 56 | 56 | 3584/112/3584 | 448/7168/448 | 3584/112/3584 | 448/7168/448 | 0 | 0 | 0.000 | reject_runtime_retention |
| eviction pressure | 112 | 112 | 7168/224/7168 | 896/14336/896 | 7168/224/7168 | 896/14336/896 | 0 | 0 | 0.000 | reject_runtime_retention |
