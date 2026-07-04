# Throughput Repeat

Unprofiled `serving_mixed_112req_wave16` runs. Owner timing is disabled here.

| mode | runs | output tok/s mean | output tok/s stdev | decode tok/s mean | decode prepare s mean | decode forward s mean | graph replay/eager |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | 3 | 169.7381 | 0.8408 | 269.4540 | 0.9403 | 9.9757 | 441/0 |
| Route B graph baseline | 3 | 136.2373 | 0.4446 | 266.4154 | 4.4798 | 10.0897 | 441/0 |
| Route B direct C4 | 3 | 138.1281 | 0.7047 | 265.3675 | 4.2067 | 10.1297 | 441/0 |
| Route B direct SWA+C4+C128 | 3 | 141.4511 | 1.2289 | 268.9379 | 3.8731 | 9.9964 | 441/0 |
