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
| Prefill chunk | 8,192 tokens |
| Communication | PyNCCL threshold32m |

Performance rows are current, closed, single-wave offline workloads: all
requests fit simultaneously, and each request produces 1,024 output tokens.
Every published configuration keeps maximum running requests equal to maximum
captured M. Results are single-run reference measurements rather than
guarantees for other sm80 systems.

## Throughput

| Configuration | Max running / graph M | Active M | Prompt/request | Requests/s | Output tok/s | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `default_m128` | 128 | 128 | 1K | 0.9997 | 1,023.67 | 4,686.70 | 1,506.39 |
| `low_m64` | 64 | 4 | 4K | 0.1153 | 118.02 | 2,498.31 | 160.36 |

## CUDA Graph And KV Capacity

Values are per rank. Larger graph coverage improves decode coverage at the
cost of startup time and KV-cache capacity.

| Configuration | Max running / graph M | Physical graph memory | KV tokens |
| --- | ---: | ---: | ---: |
| `long_context_m4` | 4 | 0.87 GiB | 930,816 |
| `low_m64` | 64 | 1.52 GiB | 811,008 |
| `default_m128` | 128 | 2.33 GiB | 682,240 |
| `high_m256` | 256 | 3.58 GiB | 424,704 |

The default covers decode batches through M=128. M64 trades some graph coverage
for additional KV capacity, while M256 trades capacity for high-concurrency
graph replay. M4 is a capability-oriented configuration for 512K contexts.
Effective context length is bounded by this KV capacity even though the model
configuration permits up to 1M tokens.

## Long Context

| Workload | TTFT | Prefill tok/s | Decode tok/s | Peak allocated/rank | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 512K prompt + 8 output, BS1 | 790.68 s | 677.82 | 32.29 | 72.04 GiB | Passed with 64 prefill chunks |

Long-context numbers are single-run reference measurements.

## Notes

- CUDA graph decode replay, radix prefix caching, independent SWA lifetime,
  chunked prefill, Marlin WNA16 MoE, and PyNCCL were enabled.
- Chinese, English, code, arithmetic, and exact-instruction text smoke passed.
