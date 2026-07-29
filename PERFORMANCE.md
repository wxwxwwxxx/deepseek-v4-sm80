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
captured M. Results are single-run reference measurements rather than
guarantees for other sm80 systems.

## Throughput

| Configuration | Max running / graph M | Active M | Prompt/request | Requests/s | Output tok/s | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_context_m8` | 8 | 8 | 1K | 0.2503 | 256.29 | 3,225.57 | 305.16 |
| `long_context_m8` | 8 | 8 | 4K | 0.1936 | 198.25 | 3,348.49 | 284.98 |
| `low_m64` | 64 | 64 | 1K | 0.7155 | 732.64 | 4,353.23 | 984.60 |
| `low_m64` | 64 | 64 | 4K | 0.3876 | 396.93 | 3,555.71 | 865.93 |
| `default_m128` | 128 | 128 | 1K | 0.8295 | 849.44 | 4,409.16 | 1,189.50 |
| `default_m128` | 128 | 128 | 4K | 0.4187 | 428.75 | 3,586.80 | 1,016.41 |
| `high_m256` | 256 | 256 | 1K | 0.9086 | 930.37 | 4,508.60 | 1,341.18 |
| `high_m256` | 256 | 254 | 4K | 0.4489 | 459.66 | 3,665.04 | 1,170.26 |

## CUDA Graph And KV Capacity

Values are per rank. Larger graph coverage improves decode coverage at the
cost of startup time and KV-cache capacity.

| Configuration | Max running / graph M | Physical graph memory | KV tokens |
| --- | ---: | ---: | ---: |
| `long_context_m8` | 8 | 0.95 GiB | 5,523,200 |
| `low_m64` | 64 | 1.57 GiB | 4,758,016 |
| `default_m128` | 128 | 2.50 GiB | 3,904,256 |
| `high_m256` | 256 | 4.41 GiB | 2,196,992 |

The default covers decode batches through M=128. M64 trades some graph coverage
for additional KV capacity, while M256 trades capacity for high-concurrency
graph replay. M8 is the promoted long-context configuration. Effective context
length is bounded by this KV capacity even though the model configuration
permits up to 1M tokens.

## Long Context

| Recipe/workload | TTFT | Prefill tok/s | Decode tok/s | Peak allocated/rank | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| M8, 512-Ki prompt + 8 output, BS8 | 5,132.41 s | 835.63 | 168.48 | 71.95 GiB | Passed: 512 resident BS8 prefill forwards, decode M8 |
| M8, exact 1-Mi total/request, BS4 | 10,346.82 s | 410.42 | 73.04 | 72.32 GiB | Passed: 512 resident BS4 prefill forwards, decode M4 |

Long-context numbers are single-run capability smokes on the stated platform,
not latency guarantees. Prompts differed at token zero, saved prefill tokens
were zero, and the total prefill-forward budget was 8,192 tokens.

## Notes

- CUDA graph decode replay, radix prefix caching, independent SWA lifetime,
  chunked prefill, Marlin WNA16 MoE, and PyNCCL were enabled.
- Chinese, English, code, arithmetic, and exact-instruction text smoke passed.
