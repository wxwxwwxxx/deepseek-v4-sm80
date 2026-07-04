# Prepare Versus Forward Attribution

| mode | output tok/s | decode tok/s | decode prepare s | decode forward s | prepare delta vs phase1 s | forward delta vs phase1 s |
| --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | 169.7381 | 269.4540 | 0.9403 | 9.9757 | 0.0000 | 0.0000 |
| Route B graph baseline | 136.2373 | 266.4154 | 4.4798 | 10.0897 | 3.5395 | 0.1140 |
| Route B direct C4 | 138.1281 | 265.3675 | 4.2067 | 10.1297 | 3.2664 | 0.1540 |
| Route B direct SWA+C4+C128 | 141.4511 | 268.9379 | 3.8731 | 9.9964 | 2.9328 | 0.0206 |

Direct C4 mean output gain vs Route B baseline: 1.8908 tok/s; decode-prepare reduction: 0.2731 s; direct-C4 output stdev: 0.7047 tok/s.
