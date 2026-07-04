# Throughput Repeat

Unprofiled `serving_mixed_112req_wave16`; owner timing disabled. The first three rows are the TARGET 08.26 frozen comparison set, and the final row is the TARGET 08.27 opt-in run produced in this milestone.

| mode | runs | output tok/s mean | stdev | decode tok/s mean | decode prepare s | decode forward s | graph replay/eager | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | 3 | 169.7381 | 0.8408 | 269.4540 | 0.9403 | 9.9757 | 441/0 | 08.26 frozen baseline |
| Route B graph baseline | 3 | 136.2373 | 0.4446 | 266.4154 | 4.4798 | 10.0897 | 441/0 | 08.26 frozen baseline |
| Route B direct C4 | 3 | 138.1281 | 0.7047 | 265.3675 | 4.2067 | 10.1297 | 441/0 | 08.26 frozen baseline |
| Route B direct C4 + lifetime cache | 3 | 162.4726 | 0.5952 | 268.5946 | 1.1416 | 10.0077 | 441/0 | 08.27 current opt-in |
