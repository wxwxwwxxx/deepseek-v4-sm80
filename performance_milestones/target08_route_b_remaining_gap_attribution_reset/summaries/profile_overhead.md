# Profile Overhead

Owner timing was run as a separate profiled pass and is not used as final throughput evidence.

| mode | unprofiled output tok/s mean | owner-timing output tok/s | delta |
| --- | --- | --- | --- |
| phase1 prefix on | 169.7381 | 123.1169 | -27.47% |
| Route B graph baseline | 136.2373 | 102.2208 | -24.97% |
| Route B direct C4 | 138.1281 | 101.9809 | -26.17% |
| Route B direct SWA+C4+C128 | 141.4511 | 103.5551 | -26.79% |

Interpretation: profile runs are useful for owner ranking, but their throughput and decode phase totals are instrumented. The throughput recommendation should use `throughput_repeat.md`.
