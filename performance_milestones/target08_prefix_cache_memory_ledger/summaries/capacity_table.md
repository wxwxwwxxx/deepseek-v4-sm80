| case | KV pool pages used | 4096 prompts eq | 4096+1024 req eq | remaining KV pages | remaining 4096+1024 reqs | free GiB if extra |
| --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 0.8% | 0.06 | 0.05 | 127 | 6.35 | 36.430 |
| 1024-token prefix | 3.1% | 0.25 | 0.20 | 124 | 6.20 | 36.376 |
| 4096-token prefix | 12.5% | 1.00 | 0.80 | 112 | 5.60 | 36.160 |
| multi-prefix mixed | 31.2% | 2.50 | 2.00 | 88 | 4.40 | 35.729 |
| 08.10 sustained workload | 43.8% | 3.50 | 2.80 | 72 | 3.60 | 35.441 |
| eviction pressure | 87.5% | 7.00 | 5.60 | 16 | 0.80 | 34.434 |
