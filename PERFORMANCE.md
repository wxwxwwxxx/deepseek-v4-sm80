# DeepSeek V4 SM80 Performance

This page summarizes the performance and capacity measured on one DGX A100
system with eight 80GB GPUs.

## Test Platform

| Item | Configuration |
| --- | --- |
| GPU | 8x NVIDIA A100-SXM4-80GB, TP8 |
| Model | DeepSeek V4 Flash |
| Precision | BF16 compute with model-defined FP32/FP8/FP4 state |
| Runtime | CUDA 12.8, NCCL 2.26-2.27 |
| Page size | 256 tokens |
| Total prefill-forward budget | 8,192 tokens |
| Communication | PyNCCL threshold32m |

Performance rows are current, closed, single-wave offline workloads: all
requests fit simultaneously, and each request produces 1,024 output tokens.
Every published configuration keeps maximum running requests equal to maximum
captured M. The P1K M128 and M256 rows are medians of three complete macro
repeats; the remaining rows are single-run reference measurements. Results are
not guarantees for other sm80 systems.

## Throughput

| Configuration | Max running / graph M | Active M | Prompt/request | Requests/s | Output tok/s | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_context_m8` | 8 | 8 | 1K | 0.2698 | 276.30 | 1,987.01 | 362.60 |
| `long_context_m8` | 8 | 8 | 4K | 0.2110 | 216.08 | 2,556.49 | 360.00 |
| `low_m64` | 64 | 64 | 1K | 0.9708 | 994.12 | 3,028.56 | 1,745.31 |
| `low_m64` | 64 | 64 | 4K | 0.4713 | 482.59 | 3,024.79 | 1,740.50 |
| `default_m128` | 128 | 128 | 1K | 1.2784 | 1,309.05 | 4,880.63 | 2,487.78 |
| `default_m128` | 128 | 128 | 4K | 0.5044 | 516.55 | 3,394.67 | 2,444.44 |
| `high_m256` | 256 | 256 | 1K | 1.2979 | 1,329.03 | 4,449.96 | 3,143.95 |
| `high_m256` | 256 | 256 | 4K | 0.4890 | 500.73 | 3,589.89 | 3,043.86 |

## CUDA Graph And KV Capacity

Values are per rank. Larger graph coverage improves decode coverage at the
cost of startup time and KV-cache capacity.

| Configuration | Max running / graph M | Physical graph memory | KV tokens |
| --- | ---: | ---: | ---: |
| `long_context_m8` | 8 | 1.07 GiB | 6,411,008 |
| `low_m64` | 64 | 2.29 GiB | 5,485,824 |
| `default_m128` | 128 | 3.60 GiB | 4,453,376 |
| `high_m256` | 256 | 6.48 GiB | 2,387,968 |

Graph-supported decode uses a 32K primary graph and automatically falls back to
a 1M-wide graph for longer materialized contexts; context width does not force
a supported M to eager execution. The default covers decode batches through
M=128. M64 trades some graph coverage for additional KV capacity, while M256
trades capacity for high-concurrency graph replay. M8 is the promoted
long-context configuration.

## Long Context

| Recipe/workload | TTFT | Prefill tok/s | Decode tok/s | Peak allocated/rank | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| M8, 512-Ki prompt + 8 output, BS8 | 5,562.27 s | 758.97 | 179.71 | 71.94 GiB | Passed: 512 resident BS8 prefill forwards, 7 decode M8 graph replays |
| M8, exact 1-Mi total/request, BS4 | 9,869.62 s | 425.97 | 78.32 | 72.31 GiB | Passed: 512 resident BS4 prefill forwards, 7 decode M4 graph replays |

Long-context numbers are single-run capability smokes on the stated platform,
not latency guarantees. Prompts differed at token zero, saved prefill tokens
were zero, and the total prefill-forward budget was 8,192 tokens.

## Notes

- CUDA graph decode replay, radix prefix caching, independent SWA lifetime,
  chunked prefill, Marlin WNA16 MoE, and PyNCCL were enabled.
- Chinese, English, code, arithmetic, and exact-instruction text smoke passed.
