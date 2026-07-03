| case | SWA/full GiB | C4 GiB | C128 GiB | indexer BF16 GiB | indexer FP8 extra GiB | compress-state GiB | total GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 0.010 | 0.001 | 0.000 | 0.000 | 0.000 | 0.006 | 0.018 |
| 1024-token prefix | 0.042 | 0.005 | 0.000 | 0.001 | 0.001 | 0.023 | 0.072 |
| 4096-token prefix | 0.168 | 0.021 | 0.001 | 0.005 | 0.003 | 0.091 | 0.288 |
| multi-prefix mixed | 0.420 | 0.051 | 0.002 | 0.013 | 0.007 | 0.227 | 0.719 |
| 08.10 sustained workload | 0.588 | 0.072 | 0.002 | 0.018 | 0.009 | 0.318 | 1.007 |
| eviction pressure | 1.176 | 0.144 | 0.004 | 0.036 | 0.019 | 0.637 | 2.015 |
