# Prepare Owner Profile

Owner-timing profile for `serving_mixed_112req_wave16`. These numbers include profiling overhead and are for attribution, not throughput.

| mode | prepare s | forward s | host attention metadata ms | component tables ms | full page table ms | C4 sparse ms | C128 ms | replay component tables ms | direct index ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Route B direct C4 | 4.4941 | 14.8697 | 4304.6365 | 3341.1692 | 69.8436 | 409.5431 | 421.9036 | 103.2981 | 71.1383 |
| Route B direct C4 + lifetime cache | 1.5189 | 14.3617 | 1330.1929 | 354.4121 | 69.8970 | 410.3621 | 429.7783 | 103.0957 | 66.3969 |
