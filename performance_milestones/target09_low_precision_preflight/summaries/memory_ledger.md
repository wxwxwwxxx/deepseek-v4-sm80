| item | value | evidence |
| --- | --- | --- |
| KV/cache pages per rank | 2.32 GiB | runtime-reported |
| KV/cache bytes/page | 18.56 MiB | derived from runtime bytes / num_pages |
| BF16 projection weight caches | 1.59 GiB | runtime-reported model_prepare |
| Dense FP8 Marlin projection cache | disabled | runtime-reported model_prepare |
| Allocated before graph capture | 23.24 GiB | runtime-reported |
| Allocated after graph capture | 41.06 GiB | runtime-reported |
| CUDA graph free-memory delta | 18.78 GiB | runtime-reported |
| CUDA graph allocated delta | 17.82 GiB | derived from allocated after-before |
| Loaded model/static residual estimate | 19.33 GiB | capture_memory_allocated_before_graph - reported KV/cache pages - reported BF16 projection weight caches; includes sharded model weights plus base runtime/comm buffers. |
| Context/page capacity | 128 pages x 256 tokens = 32768 tokens | runtime config |

Projection/cache weights:

| cache | enabled | layers | bytes | toggle/backend |
| --- | --- | --- | --- | --- |
| q_wqb_bf16_weight_cache | True | 43 | 0.34 GiB | MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE |
| wo_b_bf16_weight_cache | True | 43 | 0.34 GiB | MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE |
| indexer_wq_b_bf16_weight_cache | True | 21 | 0.33 GiB | MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE |
| wo_a_bf16_bmm_cache | True | 43 | 0.34 GiB | MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE |
| shared_expert_bf16_weight_cache | True | 43 | 0.25 GiB | MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE |
| projection_bf16_weight_cache_total | n/a | n/a | 1.59 GiB |  |
| dense_fp8_marlin_projection_cache | False | 0 | 0.00 GiB | MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION |

Prefix/cache final state by scenario:

| scenario | hit | saved prefill | retained pages | retained memory | available component pages | live full pages | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | 0.0% | 0 | 64 | 1.15 GiB | 65 | 4 | 0 |
| historical_4096_1024_bs4 | 0.0% | 0 | 124 | 2.23 GiB | 5 | 11 | 1 |
| serving_mixed_112req_wave16 | 0.0% | 0 | 106 | 1.91 GiB | 23 | 23 | 3 |
| prefix_multi_112req_wave16 | 41.4% | 49152 | 87 | 1.56 GiB | 42 | 28 | 4 |
