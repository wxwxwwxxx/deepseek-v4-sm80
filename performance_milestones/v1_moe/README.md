# V1 MoE Performance Milestone

## Scope

This milestone records the first end-to-end evidence for the V1 exact grouped
MoE path:

- `MINISGL_DSV4_SM80_V1_MOE=1`
- TP=8 on A100 sm80
- page size = 256
- model path: `/models/DeepSeek-V4-Flash`

V1 enables the v0 bf16 bundle plus `MINISGL_DSV4_SM80_MOE_ROUTE=1`. It is still
an exact/bf16-direct route, not the approximate INT8 Tensor Core path.

## Artifact Layout

| Path | Form | Notes |
| --- | --- | --- |
| `raw/dsv4_v1_moe_e2e_gate` | symlink | Full raw E2E gate directory from `/tmp`. |
| `raw/dsv4_v1_moe_4096x1024_bs4` | symlink | Full raw 4096/1024/batch4 comparison directory from `/tmp`. |
| `raw/dsv4_nsys_mini_v1_4096x128_bs4` | symlink | Benchmark JSON directory for the short nsys run. |
| `raw/nsys_mini_v1_moe_4096x128_bs4.nsys-rep` | symlink | Large Nsight Systems report, about 1.3G. |
| `raw/nsys_mini_v1_moe_4096x128_bs4.sqlite` | symlink | Large exported sqlite, about 3.6G. |
| `summaries/dsv4_v1_moe_e2e_gate` | copied | `run_config.json`, `summary.json`, `matrix.jsonl`. |
| `summaries/dsv4_v1_moe_4096x1024_bs4` | copied | `run_config.json`, `summary.json`, `matrix.jsonl`. |
| `summaries/dsv4_nsys_mini_v1_4096x128_bs4` | copied | short nsys benchmark config/summary/matrix. |
| `summaries/nsys_mini_v1_moe_4096x128_bs4` | copied | `nsys stats` CSV summaries. |

## E2E Gate Result

Artifact: `summaries/dsv4_v1_moe_e2e_gate`.

| Scenario | Metric | v0_bf16 | v1_moe | Ratio |
| --- | ---: | ---: | ---: | ---: |
| decode_throughput_bs8 | decode tok/s | 2.88 | 22.53 | 7.83x |
| decode_throughput_bs8 | E2E output tok/s | 2.72 | 20.95 | 7.71x |
| decode_throughput_bs8 | TTFT | 13.16s | 1.92s | 6.84x lower |
| decode_throughput_bs8 | TPOT | 2.78s | 0.357s | 7.78x lower |
| mixed_prefill_decode_bs4 | decode tok/s | 1.49 | 7.20 | 4.82x |
| mixed_prefill_decode_bs4 | E2E output tok/s | 1.23 | 5.53 | 4.50x |
| mixed_prefill_decode_bs4 | TTFT | 14.07s | 3.84s | 3.67x lower |
| mixed_prefill_decode_bs4 | TPOT | 1.82s | 0.348s | 5.22x lower |

Key counter evidence:

- `moe_route_dispatch_bf16_grouped` none-skip count goes to zero in V1.
- `dequant_fp4_weight` is removed from the V1 hot path.
- `quantized_linear_ref` and `silu_and_mul_clamp_fallback` calls drop sharply.

## 4096/1024/Batch4 Result

Artifact: `summaries/dsv4_v1_moe_4096x1024_bs4`.

This is the closest local comparison point to the old vLLM-based framework
serving benchmark: 4 requests, 4096 input tokens each, 1024 output tokens each.

| Metric | v0_bf16 | v1_moe | Ratio |
| --- | ---: | ---: | ---: |
| elapsed | 2188.35s | 389.80s | 5.61x faster |
| output throughput | 1.87 tok/s | 10.51 tok/s | 5.61x |
| decode throughput | 1.90 tok/s | 11.25 tok/s | 5.94x |
| TTFT | 27.21s | 24.26s | 1.12x lower |
| TPOT | 2112.55ms | 357.32ms | 5.91x lower |
| prefill throughput | 621.71 tok/s | 695.41 tok/s | 1.12x |

Major V0 -> V1 counter deltas:

- `dequant_fp4_weight`: `19,373,928 -> 0`
- `quantized_linear_ref`: `21,659,496 -> 2,285,568`
- `silu_and_mul_clamp_fallback`: `6,810,232 -> 352,256`
- `moe_route_dispatch_bf16_grouped` none-skips: `352,256 -> 0`

## Old Framework Gap

Old vLLM-based framework baseline provided by the user:

- 4 successful requests
- input: 4096 tokens/request
- output: 1024 tokens/request
- duration: 35.91s
- output throughput: 114.07 tok/s
- total token throughput: 570.78 tok/s
- mean TTFT: 123.21ms
- mean TPOT: 15.68ms

Compared with `v1_moe` 4096/1024/batch4:

| Metric | old vLLM-based framework | mini V1 MoE | Gap |
| --- | ---: | ---: | ---: |
| duration | 35.91s | 389.80s | mini slower by 10.85x |
| output throughput | 114.07 tok/s | 10.51 tok/s | mini slower by 10.86x |
| total token throughput | 570.78 tok/s | 52.54 tok/s | mini slower by 10.86x |
| TPOT | 15.68ms | 357.32ms | mini slower by 22.79x |
| TTFT | 123.21ms | 24.26s | mini slower by 196.87x |

Conclusion: V1 MoE closes a very large part of the V0 gap, but the exact
4096/1024 workload still has an order-of-magnitude throughput gap against the
old framework.

## Nsight Systems Short Run

Artifacts:

- `raw/nsys_mini_v1_moe_4096x128_bs4.nsys-rep`
- `raw/nsys_mini_v1_moe_4096x128_bs4.sqlite`
- `summaries/nsys_mini_v1_moe_4096x128_bs4`
- `summaries/dsv4_nsys_mini_v1_4096x128_bs4`

Short nsys benchmark workload:

- 4096 input tokens/request
- 128 output tokens/request
- batch size 4
- `v1_moe`

Benchmark under nsys:

| Metric | Value |
| --- | ---: |
| elapsed | 101.08s |
| TTFT | 28.29s |
| TPOT | 573.07ms |
| decode throughput | 7.02 tok/s |
| output throughput | 5.07 tok/s |

The profiler adds visible overhead, so these numbers should be used for
structure and ranking, not final throughput.

Kernel time in the formal workload window, summed across all 8 GPUs:

| Category | GPU time | Share |
| --- | ---: | ---: |
| NCCL bf16 all-reduce | 166.04s | 26.8% |
| NCCL f32 all-reduce | 96.02s | 15.5% |
| MoE grouped FP4 w13 | 113.45s | 18.3% |
| MoE grouped FP4 w2 | 74.84s | 12.1% |
| PyTorch elementwise/copy/reduce small kernels | 114.14s | 18.4% |
| Sparse attention | 32.89s | 5.3% |
| Indexer bf16 logits | 8.05s | 1.3% |
| cuBLAS/CUTLASS/GEMM | 9.28s | 1.5% |

Important trace facts:

- Formal workload window: about 102s.
- Kernel instances in the workload window: about 26.47M.
- `cudaLaunchKernel` calls in the workload window: about 25.58M.
- No CUDA Graph replay evidence: `graphNodeId` is null for captured kernel
  events and no graph runtime APIs appear in the workload window.
- Per forward step, rough single-rank kernel-time estimate:
  - NCCL all-reduce: about 256ms
  - grouped MoE FP4: about 184ms
  - PyTorch small kernels: about 111ms
  - sparse attention: about 32ms

## Current Interpretation

V1 confirms that the original diagnosis was correct: the V0 decode path was
dominated by MoE/FP4 expert handling. The grouped MoE route removes the largest
FP4 dequant/fallback loop and gives 4.5x-7.8x E2E improvements on the local
gate workloads.

The remaining gap now appears to be split across:

1. TP communication boundaries. NCCL all-reduce is the largest nsys category.
2. MoE grouped kernel cost. V1 is good enough to prove the path, but it is not
   the final fused expert pipeline.
3. PyTorch small-kernel fragmentation. Tens of millions of small kernels remain,
   and there is no CUDA Graph capture evidence.
4. Long-prefill/TTFT work. 4096-token TTFT improves only slightly from V0 to V1,
   so the next bottleneck is not only routed MoE.
5. Sparse attention is visible but not the first target in this profile.

## Recommended Next Steps

1. Capture the old vLLM-based framework with the same 4096/128/batch4 nsys
   workload and compare NCCL count, kernel count, CUDA Graph use, and top MoE
   kernels.
2. Audit all all-reduce call sites and compare against old framework behavior.
   The V1 reduce-once MoE change helped, but the profile still shows many per-
   layer collectives.
3. Continue MoE V2 exact fusion before approximate INT8: fuse or pipeline w13,
   activation/routed-weight combine, and w2 more tightly.
4. Reduce PyTorch small-kernel count around gate/indexer/shared experts/HC and
   test CUDA Graph capture for stable decode shapes.
5. Only after the paired vLLM profile, decide whether to partially port old
   framework implementation details or just mirror the design.
