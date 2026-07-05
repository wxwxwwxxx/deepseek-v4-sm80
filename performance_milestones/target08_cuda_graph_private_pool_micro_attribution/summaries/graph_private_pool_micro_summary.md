# TARGET 08.32 Graph Private-Pool Micro Summary

- cases: `63`
- successes: `63`
- failures: `0`

## Largest Measured Free-Memory Deltas

| case | ok | bs | N | variant | free GiB | alloc GiB | reserved GiB | capture s | explicit MiB | projected GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| indexer_topk_only_n16_bs16 | yes | 16 | 16 | out_of_place | 0.111 | 0.008 | 0.088 | 0.0118 | 2.3 | n/a |
| indexer_topk_only_n21_bs16 | yes | 16 | 21 | out_of_place | 0.111 | 0.008 | 0.088 | 0.0136 | 2.3 | 0.111 |
| indexer_topk_only_n2_bs16 | yes | 16 | 2 | out_of_place | 0.107 | 0.008 | 0.088 | 0.0062 | 2.3 | n/a |
| indexer_topk_only_n4_bs16 | yes | 16 | 4 | out_of_place | 0.107 | 0.008 | 0.088 | 0.0073 | 2.3 | n/a |
| indexer_topk_only_n8_bs16 | yes | 16 | 8 | out_of_place | 0.107 | 0.008 | 0.088 | 0.0159 | 2.3 | n/a |
| c4_indexer_topk_bs16 | yes | 16 | 1 | out_of_place | 0.105 | 0.008 | 0.086 | 0.0056 | 2.3 | 2.215 |
| indexer_topk_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.105 | 0.008 | 0.086 | 0.0056 | 2.3 | 2.215 |
| attention_only_n43_bs16 | yes | 16 | 43 | out_of_place | 0.088 | 0.008 | 0.064 | 0.0277 | 40.4 | 0.088 |
| attention_only_n16_bs16 | yes | 16 | 16 | out_of_place | 0.084 | 0.008 | 0.064 | 0.0187 | 40.4 | n/a |
| attention_only_n4_bs16 | yes | 16 | 4 | out_of_place | 0.084 | 0.008 | 0.064 | 0.0141 | 40.4 | n/a |

## Largest Simple Projections

| case | ok | bs | N | variant | free GiB | alloc GiB | reserved GiB | capture s | explicit MiB | projected GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c4_indexer_topk_bs16 | yes | 16 | 1 | out_of_place | 0.105 | 0.008 | 0.086 | 0.0056 | 2.3 | 2.215 |
| indexer_topk_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.105 | 0.008 | 0.086 | 0.0056 | 2.3 | 2.215 |
| moe_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.045 | 0.008 | 0.025 | 0.0099 | 12.1 | 1.932 |
| attention_mlp_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0100 | 42.1 | 1.848 |
| attention_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0117 | 40.4 | 1.848 |
| bf16_matmul_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0103 | 32.1 | 1.848 |
| projection_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0099 | 36.1 | 1.848 |
| repeated_bf16_matmul_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0098 | 32.1 | 1.848 |
| swa_attention_bs1 | yes | 1 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0138 | 32.0 | 1.848 |
| swa_attention_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0137 | 32.1 | 1.848 |

## Controls

| case | ok | bs | N | variant | free GiB | alloc GiB | reserved GiB | capture s | explicit MiB | projected GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_staging_bs16 | yes | 16 | 1 | out_of_place | 0.021 | 0.000 | 0.002 | 0.0125 | 0.0 | 0.021 |
| elementwise_bs16 | yes | 16 | 1 | out_of_place | 0.023 | 0.000 | 0.004 | 0.0116 | 0.1 | 0.023 |
| elementwise_bs16_prealloc | yes | 16 | 1 | prealloc | 0.023 | 0.000 | 0.004 | 0.0112 | 0.3 | 0.023 |
| empty_graph | yes | 16 | 1 | out_of_place | 0.021 | 0.000 | 0.002 | 0.0103 | 0.0 | 0.021 |

## Communication Controls

| case | ok | bs | N | variant | free GiB | alloc GiB | reserved GiB | capture s | explicit MiB | projected GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comm_all_reduce_bf16_32mib_tp8 | yes | n/a | 1 | bf16 | 0.002 | 0.000 | 0.002 | 0.0037 | 32.0 | 0.002 |
| comm_all_reduce_fp32_32mib_tp8 | yes | n/a | 1 | fp32 | 0.002 | 0.000 | 0.002 | 0.0036 | 32.0 | 0.002 |

## DSV4 Subgraphs

| case | ok | bs | N | variant | free GiB | alloc GiB | reserved GiB | capture s | explicit MiB | projected GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c128_attention_bs1 | yes | 1 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0169 | 32.3 | 0.859 |
| c128_attention_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.009 | 0.023 | 0.0161 | 32.4 | 0.859 |
| c4_indexer_topk_bs1 | yes | 1 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0047 | 2.0 | 0.902 |
| c4_indexer_topk_bs16 | yes | 16 | 1 | out_of_place | 0.105 | 0.008 | 0.086 | 0.0056 | 2.3 | 2.215 |
| c4_sparse_attention_bs1 | yes | 1 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0120 | 40.0 | 0.902 |
| c4_sparse_attention_bs16 | yes | 16 | 1 | out_of_place | 0.082 | 0.008 | 0.062 | 0.0122 | 40.2 | 1.723 |
| metadata_deforest_bs1 | yes | 1 | 1 | out_of_place | 0.023 | 0.000 | 0.004 | 0.0054 | 0.0 | 0.023 |
| metadata_deforest_bs16 | yes | 16 | 1 | out_of_place | 0.023 | 0.000 | 0.004 | 0.0054 | 0.0 | 0.023 |
| qkv_norm_rope_cache_store_bs1 | yes | 1 | 1 | out_of_place | 0.023 | 0.000 | 0.004 | 0.0050 | 32.0 | 1.008 |
| qkv_norm_rope_cache_store_bs16 | yes | 16 | 1 | out_of_place | 0.023 | 0.000 | 0.004 | 0.0052 | 32.1 | 1.008 |
| swa_attention_bs1 | yes | 1 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0138 | 32.0 | 1.848 |
| swa_attention_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0137 | 32.1 | 1.848 |

## One-Layer / Repeated-Layer Scaling

| case | ok | bs | N | variant | free GiB | alloc GiB | reserved GiB | capture s | explicit MiB | projected GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp_n16_bs16 | yes | 16 | 16 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0164 | 42.1 | n/a |
| attention_mlp_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0100 | 42.1 | 1.848 |
| attention_mlp_n2_bs16 | yes | 16 | 2 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0154 | 42.1 | n/a |
| attention_mlp_n43_bs16 | yes | 16 | 43 | out_of_place | 0.049 | 0.008 | 0.023 | 0.0209 | 42.1 | 0.049 |
| attention_mlp_n4_bs16 | yes | 16 | 4 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0107 | 42.1 | n/a |
| attention_mlp_n8_bs16 | yes | 16 | 8 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0161 | 42.1 | n/a |
| attention_only_n16_bs16 | yes | 16 | 16 | out_of_place | 0.084 | 0.008 | 0.064 | 0.0187 | 40.4 | n/a |
| attention_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0117 | 40.4 | 1.848 |
| attention_only_n2_bs16 | yes | 16 | 2 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0191 | 40.4 | n/a |
| attention_only_n43_bs16 | yes | 16 | 43 | out_of_place | 0.088 | 0.008 | 0.064 | 0.0277 | 40.4 | 0.088 |
| attention_only_n4_bs16 | yes | 16 | 4 | out_of_place | 0.084 | 0.008 | 0.064 | 0.0141 | 40.4 | n/a |
| attention_only_n8_bs16 | yes | 16 | 8 | out_of_place | 0.084 | 0.008 | 0.064 | 0.0148 | 40.4 | n/a |
| bf16_matmul_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0103 | 32.1 | 1.848 |
| bf16_matmul_bs16_prealloc | yes | 16 | 1 | prealloc | 0.041 | 0.008 | 0.021 | 0.0094 | 32.4 | 1.764 |
| indexer_topk_only_n16_bs16 | yes | 16 | 16 | out_of_place | 0.111 | 0.008 | 0.088 | 0.0118 | 2.3 | n/a |
| indexer_topk_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.105 | 0.008 | 0.086 | 0.0056 | 2.3 | 2.215 |
| indexer_topk_only_n21_bs16 | yes | 16 | 21 | out_of_place | 0.111 | 0.008 | 0.088 | 0.0136 | 2.3 | 0.111 |
| indexer_topk_only_n2_bs16 | yes | 16 | 2 | out_of_place | 0.107 | 0.008 | 0.088 | 0.0062 | 2.3 | n/a |
| indexer_topk_only_n4_bs16 | yes | 16 | 4 | out_of_place | 0.107 | 0.008 | 0.088 | 0.0073 | 2.3 | n/a |
| indexer_topk_only_n8_bs16 | yes | 16 | 8 | out_of_place | 0.107 | 0.008 | 0.088 | 0.0159 | 2.3 | n/a |
| moe_only_n16_bs16 | yes | 16 | 16 | out_of_place | 0.049 | 0.008 | 0.025 | 0.0162 | 12.1 | n/a |
| moe_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.045 | 0.008 | 0.025 | 0.0099 | 12.1 | 1.932 |
| moe_only_n2_bs16 | yes | 16 | 2 | out_of_place | 0.045 | 0.008 | 0.025 | 0.0106 | 12.1 | n/a |
| moe_only_n43_bs16 | yes | 16 | 43 | out_of_place | 0.055 | 0.008 | 0.025 | 0.0295 | 12.1 | 0.055 |
| moe_only_n4_bs16 | yes | 16 | 4 | out_of_place | 0.045 | 0.008 | 0.025 | 0.0137 | 12.1 | n/a |
| moe_only_n8_bs16 | yes | 16 | 8 | out_of_place | 0.045 | 0.008 | 0.025 | 0.0207 | 12.1 | n/a |
| projection_only_n16_bs16 | yes | 16 | 16 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0170 | 36.1 | n/a |
| projection_only_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0099 | 36.1 | 1.848 |
| projection_only_n2_bs16 | yes | 16 | 2 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0129 | 36.1 | n/a |
| projection_only_n43_bs16 | yes | 16 | 43 | out_of_place | 0.047 | 0.008 | 0.023 | 0.0217 | 36.1 | 0.047 |
| projection_only_n4_bs16 | yes | 16 | 4 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0101 | 36.1 | n/a |
| projection_only_n8_bs16 | yes | 16 | 8 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0146 | 36.1 | n/a |
| repeated_bf16_matmul_n16_bs16 | yes | 16 | 16 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0099 | 32.1 | n/a |
| repeated_bf16_matmul_n16_bs16_prealloc | yes | 16 | 16 | prealloc | 0.041 | 0.008 | 0.021 | 0.0109 | 32.4 | n/a |
| repeated_bf16_matmul_n1_bs16 | yes | 16 | 1 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0098 | 32.1 | 1.848 |
| repeated_bf16_matmul_n1_bs16_prealloc | yes | 16 | 1 | prealloc | 0.041 | 0.008 | 0.021 | 0.0095 | 32.4 | 1.764 |
| repeated_bf16_matmul_n2_bs16 | yes | 16 | 2 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0184 | 32.1 | n/a |
| repeated_bf16_matmul_n2_bs16_prealloc | yes | 16 | 2 | prealloc | 0.041 | 0.008 | 0.021 | 0.0095 | 32.4 | n/a |
| repeated_bf16_matmul_n43_bs16 | yes | 16 | 43 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0101 | 32.1 | 0.043 |
| repeated_bf16_matmul_n43_bs16_keep_all | yes | 16 | 43 | keep_all | 0.047 | 0.008 | 0.027 | 0.0106 | 32.1 | 0.047 |
| repeated_bf16_matmul_n43_bs16_prealloc | yes | 16 | 43 | prealloc | 0.041 | 0.008 | 0.021 | 0.0099 | 32.4 | 0.041 |
| repeated_bf16_matmul_n4_bs16 | yes | 16 | 4 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0098 | 32.1 | n/a |
| repeated_bf16_matmul_n4_bs16_prealloc | yes | 16 | 4 | prealloc | 0.041 | 0.008 | 0.021 | 0.0160 | 32.4 | n/a |
| repeated_bf16_matmul_n8_bs16 | yes | 16 | 8 | out_of_place | 0.043 | 0.008 | 0.023 | 0.0196 | 32.1 | n/a |
| repeated_bf16_matmul_n8_bs16_prealloc | yes | 16 | 8 | prealloc | 0.041 | 0.008 | 0.021 | 0.0097 | 32.4 | n/a |
