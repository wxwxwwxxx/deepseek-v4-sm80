| case | recoverable full pages upper | recoverable tokens upper | SWA-only saved GiB | SWA+state saved GiB | SWA+state saved eq KV pages | compressed kept GiB |
| --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 0 | 0 | 0.000 | 0.000 | 0.0 | 0.002 |
| 1024-token prefix | 3 | 768 | 0.031 | 0.049 | 2.7 | 0.007 |
| 4096-token prefix | 15 | 3840 | 0.157 | 0.243 | 13.4 | 0.029 |
| multi-prefix mixed | 39 | 9984 | 0.409 | 0.631 | 34.8 | 0.072 |
| 08.10 sustained workload | 55 | 14080 | 0.577 | 0.890 | 49.1 | 0.101 |
| eviction pressure | 111 | 28416 | 1.165 | 1.796 | 99.1 | 0.202 |
