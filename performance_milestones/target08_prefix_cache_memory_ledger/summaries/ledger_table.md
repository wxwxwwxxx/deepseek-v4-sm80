| case | pages | tokens | full/SWA slots | C4 slots | C128 slots | indexer slots | C4 state | C128 state | idx state | GiB/rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 1 | 256 | 256 | 64 | 2 | 64 | 8 | 128 | 8 | 0.018 |
| 1024-token prefix | 4 | 1024 | 1024 | 256 | 8 | 256 | 32 | 512 | 32 | 0.072 |
| 4096-token prefix | 16 | 4096 | 4096 | 1024 | 32 | 1024 | 128 | 2048 | 128 | 0.288 |
| multi-prefix mixed | 40 | 10240 | 10240 | 2560 | 80 | 2560 | 320 | 5120 | 320 | 0.719 |
| 08.10 sustained workload | 56 | 14336 | 14336 | 3584 | 112 | 3584 | 448 | 7168 | 448 | 1.007 |
| eviction pressure | 112 | 28672 | 28672 | 7168 | 224 | 7168 | 896 | 14336 | 896 | 2.015 |
