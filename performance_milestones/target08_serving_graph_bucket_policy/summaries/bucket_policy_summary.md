# TARGET 08.05 Bucket Policy Summary

Recommended bucket set: `[1, 2, 4, 8, 16]`.

## Runs

| bucket | mode | status | captured | capture GiB | capture s | replay | eager | mean output tok/s | mean decode tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1, 2, 4, 8, 16]` | prefix_off | pass | `[16, 8, 4, 2, 1]` | 19.04 | 17.43 | 1684 | 0 | 122.33 | 213.25 |
| `[1, 2, 4, 8, 16]` | prefix_on_shared | pass | `[16, 8, 4, 2, 1]` | 19.04 | 15.17 | 30 | 0 | 42.66 | 187.54 |
| `[1, 2, 4, 8]` | prefix_off | pass | `[8, 4, 2, 1]` | 18.96 | 15.02 | 1564 | 120 | 82.00 | 146.56 |
| `[1, 2, 4, 8]` | prefix_on_shared | pass | `[8, 4, 2, 1]` | 18.96 | 14.97 | 30 | 0 | 28.29 | 187.58 |
| `[1, 2, 4]` | prefix_off | pass | `[4, 2, 1]` | 18.90 | 14.39 | 1485 | 199 | 59.08 | 96.79 |
| `[1, 2, 4]` | prefix_on_shared | pass | `[4, 2, 1]` | 18.90 | 14.00 | 15 | 15 | 16.86 | 40.74 |

## Coverage

| bucket | mode | actual decode bs coverage |
| --- | --- | --- |
| `[1, 2, 4, 8, 16]` | prefix_off | bs1: r143/e0/tok143/6.4%<br>bs2: r128/e0/tok256/6.6%<br>bs4: r1214/e0/tok4856/73.8%<br>bs7: r15/e0/tok105/1.0%<br>bs8: r64/e0/tok512/4.1%<br>bs16: r120/e0/tok1920/8.2% |
| `[1, 2, 4, 8, 16]` | prefix_on_shared | bs1: r15/e0/tok15/41.1%<br>bs7: r15/e0/tok105/58.9% |
| `[1, 2, 4, 8]` | prefix_off | bs1: r143/e0/tok143/4.4%<br>bs2: r128/e0/tok256/4.5%<br>bs4: r1214/e0/tok4856/50.8%<br>bs7: r15/e0/tok105/0.7%<br>bs8: r64/e0/tok512/2.8%<br>bs16: r0/e120/tok1920/36.8% |
| `[1, 2, 4, 8]` | prefix_on_shared | bs1: r15/e0/tok15/41.3%<br>bs7: r15/e0/tok105/58.7% |
| `[1, 2, 4]` | prefix_off | bs1: r143/e0/tok143/3.2%<br>bs2: r128/e0/tok256/3.3%<br>bs4: r1214/e0/tok4856/36.3%<br>bs7: r0/e15/tok105/3.4%<br>bs8: r0/e64/tok512/19.4%<br>bs16: r0/e120/tok1920/34.5% |
| `[1, 2, 4]` | prefix_on_shared | bs1: r15/e0/tok15/8.9%<br>bs7: r0/e15/tok105/91.1% |

## Workloads

| bucket | mode | scenario | status | output tok/s | decode tok/s | TTFT s | TPOT s | replay | eager | peak alloc GiB | peak reserved GiB |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1, 2, 4, 8, 16]` | prefix_off | `historical_4096_1024_bs4` | pass | 125.78 | 166.61 | 5.98 | 0.0260 | 1023 | 0 | 44.30 | 46.46 |
| `[1, 2, 4, 8, 16]` | prefix_off | `historical_4096_128_bs4` | pass | 62.00 | 167.10 | 4.96 | 0.0259 | 127 | 0 | 44.30 | 46.46 |
| `[1, 2, 4, 8, 16]` | prefix_off | `shared_prompt_reuse_bs8` | pass | 42.30 | 185.81 | 1.74 | 0.0269 | 30 | 0 | 42.57 | 46.46 |
| `[1, 2, 4, 8, 16]` | prefix_off | `decode_ladder_bs16` | pass | 194.72 | 272.32 | 0.50 | 0.0291 | 63 | 0 | 41.47 | 46.46 |
| `[1, 2, 4, 8, 16]` | prefix_off | `serving_mixed_112req_wave16` | pass | 186.87 | 274.41 | 0.60 | 0.0289 | 441 | 0 | 41.56 | 46.46 |
| `[1, 2, 4, 8, 16]` | prefix_on_shared | `shared_prompt_reuse_bs8` | pass | 42.66 | 187.54 | 0.82 | 0.0285 | 30 | 0 | 41.32 | 41.96 |
| `[1, 2, 4, 8]` | prefix_off | `historical_4096_1024_bs4` | pass | 120.46 | 166.60 | 7.46 | 0.0259 | 1023 | 0 | 44.30 | 46.43 |
| `[1, 2, 4, 8]` | prefix_off | `historical_4096_128_bs4` | pass | 62.05 | 167.28 | 4.97 | 0.0258 | 127 | 0 | 44.30 | 46.43 |
| `[1, 2, 4, 8]` | prefix_off | `shared_prompt_reuse_bs8` | pass | 42.63 | 187.72 | 1.74 | 0.0265 | 30 | 0 | 42.57 | 46.43 |
| `[1, 2, 4, 8]` | prefix_off | `decode_ladder_bs16` | pass | 93.16 | 105.19 | 0.50 | 0.1428 | 48 | 15 | 41.46 | 46.43 |
| `[1, 2, 4, 8]` | prefix_off | `serving_mixed_112req_wave16` | pass | 91.70 | 106.01 | 0.60 | 0.1415 | 336 | 105 | 41.55 | 46.43 |
| `[1, 2, 4, 8]` | prefix_on_shared | `shared_prompt_reuse_bs8` | pass | 28.29 | 187.58 | 0.75 | 0.0309 | 30 | 0 | 41.31 | 41.94 |
| `[1, 2, 4]` | prefix_off | `historical_4096_1024_bs4` | pass | 120.88 | 168.45 | 7.60 | 0.0257 | 1023 | 0 | 44.30 | 46.43 |
| `[1, 2, 4]` | prefix_off | `historical_4096_128_bs4` | pass | 62.31 | 169.39 | 4.96 | 0.0256 | 127 | 0 | 44.30 | 46.43 |
| `[1, 2, 4]` | prefix_off | `shared_prompt_reuse_bs8` | pass | 17.77 | 40.93 | 3.40 | 0.1601 | 15 | 15 | 42.56 | 46.43 |
| `[1, 2, 4]` | prefix_off | `decode_ladder_bs16` | pass | 31.51 | 33.25 | 1.00 | 0.3996 | 40 | 23 | 41.46 | 46.43 |
| `[1, 2, 4]` | prefix_off | `serving_mixed_112req_wave16` | pass | 62.94 | 71.95 | 0.87 | 0.1835 | 280 | 161 | 41.55 | 46.43 |
| `[1, 2, 4]` | prefix_on_shared | `shared_prompt_reuse_bs8` | pass | 16.86 | 40.74 | 3.06 | 0.1614 | 15 | 15 | 41.31 | 41.94 |
