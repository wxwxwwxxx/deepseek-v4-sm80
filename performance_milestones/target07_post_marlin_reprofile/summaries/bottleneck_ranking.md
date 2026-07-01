# Bottleneck Ranking

| Rank | Candidate | Mini window s | Wall share | Amdahl max gain | Example gain | Note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `sparse_attention` | 2.109879 | 13.28% | 15.31% | 10.25% | Top mini kernel category; vLLM has different packed-cache/split attention boundary, so exact parity estimate is risky. |
| 2 | `metadata_runtime_copy_visible` | 1.949199 | 12.27% | 13.99% | 6.54% | Visible copy/launch/graph/memcpy overhead only; sync wait is excluded because it mostly waits for GPU work already counted. |
| 3 | `indexer_cache` | 0.968744 | 6.10% | 6.49% | 3.80% | Mini bf16 indexer/cache selection is second kernel category; vLLM uses FP8 indexer cache and fused q/RoPE/quant. |
| 4 | `hc_rmsnorm_logits_sampling` | 0.439664 | 2.77% | 2.85% | 1.40% | Not a top contributor after HC/RMSNorm/sampling graph work. |
| 5 | `dense_linear_other` | 0.370444 | 2.33% | 2.39% | 1.18% | Residual dense projection/GEMM bucket. |
| 6 | `moe_route_w13_swiglu_w2_sum` | 0.317591 | 2.00% | 2.04% | 1.01% | Whole visible MoE bucket including Marlin WNA16 and activation/route pieces. |
| 7 | `marlin_wna16_expert_gemm_only` | 0.234230 | 1.47% | 1.50% | 0.74% | Specific Marlin WNA16 expert kernel from top-kernel table; below 10% workload window. |
| 8 | `nccl` | 0.163036 | 1.03% | 1.04% | 0.52% | NCCL kernels are visible but not currently dominant in rank0 window. |
