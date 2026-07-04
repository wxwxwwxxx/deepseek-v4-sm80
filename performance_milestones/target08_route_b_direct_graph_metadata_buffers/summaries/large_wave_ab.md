# Large-Wave A/B

Scenario: `serving_mixed_112req_wave16`, page size 256, 128 pages, CUDA graph
buckets `[1,2,4,8,16]`.

| mode | status | output tok/s | decode tok/s | decode prepare s | decode forward s | TTFT s | graph replay/eager | saved prefill |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 prefix on | pass | 169.6261 | 269.7954 | 0.9370 | 9.9631 | 0.7682 | 441/0 | 0 |
| Route B graph baseline | pass | 134.4667 | 262.3821 | 4.4707 | 10.2446 | 0.8235 | 441/0 | 0 |
| Route B direct C4 | pass | 136.7244 | 260.4445 | 4.2564 | 10.3208 | 0.7985 | 441/0 | 0 |
| Route B direct SWA+C4+C128 | pass | 128.4799 | 251.1565 | 3.8700 | 10.7025 | 0.9685 | 441/0 | 0 |

Derived:

| comparison | value |
| --- | ---: |
| C4-only decode-prepare reduction vs Route B baseline | 4.79% |
| SWA+C4+C128 decode-prepare reduction vs Route B baseline | 13.43% |
| SWA+C4+C128 output throughput vs Route B baseline | 0.955x |
| SWA+C4+C128 output throughput vs phase1 prefix on | 0.757x |

Decision: performance threshold not met.
