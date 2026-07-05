| scenario | label | op | dtype | shape | out_shape | count | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| historical_4096_128_bs4 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 8 | 1.07 GB |
| historical_4096_128_bs4 | dsv4.lm_head_all_gather | all_gather | float32 | 4x16160 | 32x16160 | 8 | 0.02 GB |
| historical_4096_128_bs4 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| historical_4096_1024_bs4 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| historical_4096_1024_bs4 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 8 | 1.07 GB |
| historical_4096_1024_bs4 | dsv4.lm_head_all_gather | all_gather | float32 | 4x16160 | 32x16160 | 8 | 0.02 GB |
| historical_4096_1024_bs4 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| serving_mixed_112req_wave16 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 2496x4096 | 2496x4096 | 2408 | 49.24 GB |
| serving_mixed_112req_wave16 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 2496x4096 | 2496x4096 | 56 | 1.15 GB |
| serving_mixed_112req_wave16 | dsv4.lm_head_all_gather | all_gather | float32 | 16x16160 | 128x16160 | 56 | 0.46 GB |
| serving_mixed_112req_wave16 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 2496x4096 | 2496x4096 | 2408 | 49.24 GB |
| prefix_multi_112req_wave16 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 1024x4096 | 1024x4096 | 2064 | 17.31 GB |
| prefix_multi_112req_wave16 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 9216x4096 | 9216x4096 | 344 | 25.97 GB |
| prefix_multi_112req_wave16 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 1024x4096 | 1024x4096 | 48 | 0.40 GB |
| prefix_multi_112req_wave16 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 9216x4096 | 9216x4096 | 8 | 0.60 GB |
| prefix_multi_112req_wave16 | dsv4.lm_head_all_gather | all_gather | float32 | 16x16160 | 128x16160 | 56 | 0.46 GB |
| prefix_multi_112req_wave16 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 1024x4096 | 1024x4096 | 2064 | 17.31 GB |
| prefix_multi_112req_wave16 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 9216x4096 | 9216x4096 | 344 | 25.97 GB |
