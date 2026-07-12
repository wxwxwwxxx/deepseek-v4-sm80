# DeepSeek V4 SM80 Performance

This page summarizes the public performance and capacity baseline for DSV4 on
SM80. The M64 grid was refreshed on the release candidate; the M128 and M256
rows retain the `v0.0.0` baseline measurements.

## Test Platform

| Item | Configuration |
| --- | --- |
| GPU | 8x NVIDIA A100-SXM4-80GB, TP8 |
| Model | DeepSeek V4 Flash |
| Precision | BF16 compute with model-defined FP32/FP8/FP4 state |
| Runtime | CUDA 12.8, NCCL 2.26-2.27 |
| Page size | 256 tokens |
| Prefill chunk | 8,192 tokens |
| Communication | PyNCCL threshold32m |

Performance rows are closed, single-wave offline workloads: all requests fit
simultaneously, and each request produces 1,024 output tokens. Results should
be treated as reference measurements rather than guarantees for other sm80
systems.

## Throughput

| Graph max M | Active M | Prompt/request | Requests/s | Output tok/s | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 1K | 0.1502 | 153.76 | 1,162.43 | 193.37 |
| 64 | 4 | 4K | 0.1408 | 144.21 | 3,384.38 | 191.95 |
| 64 | 4 | 16K | 0.0884 | 90.52 | 3,215.23 | 191.33 |
| 64 | 16 | 1K | 0.4559 | 466.83 | 4,195.70 | 579.20 |
| 64 | 16 | 4K | 0.3240 | 331.78 | 3,923.75 | 572.52 |
| 64 | 16 | 16K | 0.1432 | 146.64 | 3,543.58 | 562.98 |
| 64 | 64 | 1K | 0.8908 | 912.14 | 5,051.73 | 1,254.57 |
| 64 | 64 | 4K | 0.4993 | 511.30 | 4,111.43 | 1,230.79 |
| 64 | 64 | 16K | Does not fit | - | - | - |
| 128 | 128 | 1K | 1.0356 | 1,060.42 | 5,236.73 | 1,512.68 |
| 256 | 256 | 1K | 1.1941 | 1,222.75 | 5,397.09 | 1,827.28 |

## CUDA Graph And KV Capacity

Values are per rank. Larger graph coverage improves decode coverage at the
cost of startup time and KV-cache capacity.

| Maximum captured M | Physical graph memory | KV tokens |
| ---: | ---: | ---: |
| 64 | 1.36 GiB | 771,328 |
| 128 | 2.14 GiB | 737,024 |
| 256 | 3.56 GiB | 668,416 |

The default M256 configuration favors throughput. M64 and M128 are useful when
the workload benefits more from KV capacity than from high-M graph replay.

## Long Context

| Workload | Result |
| --- | --- |
| One 512K total sequence | Passed with 64 prefill chunks |
| Aggregate 512K across batch size 4 | Planner-runnable |
| Four independent 512K sequences | Does not fit the validated capacity |
| One exact 1M total sequence | Passed with 128 prefill chunks |

The exact 1M smoke used 1,048,568 prompt tokens plus eight decode tokens.
Long-context results are capability validation rather than a latency claim.

## Notes

- CUDA graph decode replay, radix prefix caching, independent SWA lifetime,
  chunked prefill, Marlin WNA16 MoE, and PyNCCL were enabled.
- Chinese, English, code, arithmetic, and exact-instruction text smoke passed.

The full measurement methodology and capacity ledger remain in
[`prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md`](prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md).
