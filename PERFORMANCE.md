# DeepSeek V4 SM80 Performance

This page summarizes the public performance and capacity baseline for DSV4 on
SM80. The graph64 capacity row was refreshed after the fused WQA/WKV allocation
order amendment. The graph128/256 capacity rows retain the BF16 pretranspose
correction measurements, and the M128/M256 throughput rows retain `v0.0.0`
measurements.

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
| 64 | 4 | 1K | 0.1604 | 164.27 | 2,331.43 | 192.83 |
| 64 | 4 | 4K | 0.1400 | 143.35 | 3,307.03 | 191.53 |
| 64 | 4 | 16K | 0.0904 | 92.53 | 3,387.68 | 190.29 |
| 64 | 16 | 1K | 0.4531 | 463.99 | 4,220.60 | 573.84 |
| 64 | 16 | 4K | 0.3218 | 329.51 | 3,939.09 | 564.67 |
| 64 | 16 | 16K | 0.1426 | 146.05 | 3,534.32 | 558.15 |
| 64 | 64 | 1K | 0.8776 | 898.67 | 5,066.35 | 1,228.84 |
| 64 | 64 | 4K | 0.4952 | 507.10 | 4,113.78 | 1,208.01 |
| 64 | 64 | 16K | Does not fit | - | - | - |
| 128 | 128 | 1K | 1.0356 | 1,060.42 | 5,236.73 | 1,512.68 |
| 256 | 256 | 1K | 1.1941 | 1,222.75 | 5,397.09 | 1,827.28 |

## CUDA Graph And KV Capacity

Values are per rank. Larger graph coverage improves decode coverage at the
cost of startup time and KV-cache capacity.

| Maximum captured M | Physical graph memory | KV tokens |
| ---: | ---: | ---: |
| 64 | 1.40 GiB | 848,384 |
| 128 | 2.17 GiB | 794,112 |
| 256 | 3.61 GiB | 725,504 |

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
- Optimized uses cached BF16 projection weights through ordinary `F.linear`;
  no duplicate pretransposed BF16 weight cache is retained.
- Chinese, English, code, arithmetic, and exact-instruction text smoke passed.

The full measurement methodology and capacity ledger remain in
[`prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md`](prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md).
