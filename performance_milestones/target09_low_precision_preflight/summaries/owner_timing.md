Owner timing was collected in separate one-scenario runs. Category values are max-rank label-sum signals, not wall-time percentages, because some owner labels are nested.

| scenario | category | max-rank total signal | captured signal | labels | top label | top label total |
| --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | communication | 3866.3 ms | 3.263 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 1924.1 ms |
| historical_4096_128_bs4 | metadata/runtime | 2277.3 ms | 0.000 ms | 29 | dsv4.prepare.prefill.attention_metadata | 778.6 ms |
| historical_4096_128_bs4 | projection/GEMM | 662.2 ms | 7.586 ms | 112 | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 209.0 ms |
| historical_4096_128_bs4 | MoE | 296.0 ms | 4.774 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 67.6 ms |
| historical_4096_128_bs4 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| historical_4096_128_bs4 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| historical_4096_1024_bs4 | metadata/runtime | 8096.7 ms | 0.000 ms | 29 | dsv4.prepare.decode.attention_metadata | 2847.9 ms |
| historical_4096_1024_bs4 | communication | 4232.7 ms | 3.181 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2346.9 ms |
| historical_4096_1024_bs4 | projection/GEMM | 641.5 ms | 7.583 ms | 112 | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 205.5 ms |
| historical_4096_1024_bs4 | MoE | 283.9 ms | 4.760 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 66.9 ms |
| historical_4096_1024_bs4 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| historical_4096_1024_bs4 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| serving_mixed_112req_wave16 | communication | 4179.8 ms | 15.033 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2421.2 ms |
| serving_mixed_112req_wave16 | metadata/runtime | 4116.0 ms | 0.000 ms | 29 | dsv4.prepare.decode.attention_metadata | 1349.7 ms |
| serving_mixed_112req_wave16 | projection/GEMM | 1877.5 ms | 37.386 ms | 112 | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 667.3 ms |
| serving_mixed_112req_wave16 | MoE | 357.8 ms | 23.457 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 82.5 ms |
| serving_mixed_112req_wave16 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| serving_mixed_112req_wave16 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| prefix_multi_112req_wave16 | communication | 5216.4 ms | 3.402 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3179.4 ms |
| prefix_multi_112req_wave16 | projection/GEMM | 2864.5 ms | 7.678 ms | 112 | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 1169.0 ms |
| prefix_multi_112req_wave16 | metadata/runtime | 2353.8 ms | 0.000 ms | 29 | dsv4.prepare.prefill.attention_metadata | 847.4 ms |
| prefix_multi_112req_wave16 | MoE | 326.5 ms | 4.826 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 78.7 ms |
| prefix_multi_112req_wave16 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| prefix_multi_112req_wave16 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |

Top labels:

| scenario | scope | category | label | max-rank total | captured | count |
| --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 1924.1 ms | 1.471 ms | 3784 |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 802.9 ms | 1.528 ms | 3784 |
| historical_4096_128_bs4 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 778.6 ms | 0.000 ms | 8 |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 687.2 ms | 0.059 ms | 88 |
| historical_4096_128_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 462.2 ms | 0.000 ms | 1024 |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 452.1 ms | 0.206 ms | 88 |
| historical_4096_128_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 391.7 ms | 0.000 ms | 8 |
| historical_4096_128_bs4 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 351.3 ms | 0.000 ms | 1016 |
| historical_4096_128_bs4 | cuda | projection/GEMM | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 209.0 ms | 0.022 ms | 88 |
| historical_4096_128_bs4 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 93.0 ms | 1.540 ms | 3784 |
| historical_4096_128_bs4 | cuda | projection/GEMM | dsv4.owner.attn.wo_b.bf16_cache_local_total | 86.8 ms | 1.521 ms | 3784 |
| historical_4096_128_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_write_locs | 72.4 ms | 0.000 ms | 1024 |
| historical_4096_1024_bs4 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 2847.9 ms | 0.000 ms | 8184 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2346.9 ms | 1.465 ms | 3784 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 1114.7 ms | 0.000 ms | 8192 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.direct_graph_metadata.decode.index_buffers | 993.7 ms | 0.000 ms | 8184 |
| historical_4096_1024_bs4 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 782.7 ms | 0.000 ms | 8 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 725.4 ms | 1.430 ms | 3784 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 686.6 ms | 0.060 ms | 88 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_write_locs | 581.5 ms | 0.000 ms | 8192 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 473.8 ms | 0.226 ms | 88 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 392.2 ms | 0.000 ms | 8 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_component_page_tables | 230.7 ms | 0.000 ms | 8192 |
| historical_4096_1024_bs4 | cuda | projection/GEMM | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 205.5 ms | 0.022 ms | 88 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2421.2 ms | 6.836 ms | 5848 |
| serving_mixed_112req_wave16 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 1349.7 ms | 0.000 ms | 3528 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 854.5 ms | 6.771 ms | 5848 |
| serving_mixed_112req_wave16 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 692.6 ms | 0.000 ms | 56 |
| serving_mixed_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 667.3 ms | 7.651 ms | 5848 |
| serving_mixed_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_activation_quantize | 592.8 ms | 1.594 ms | 5848 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 542.1 ms | 0.338 ms | 136 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 438.0 ms | 0.000 ms | 3584 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 412.8 ms | 0.000 ms | 56 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 362.0 ms | 1.088 ms | 136 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_component_page_tables | 350.5 ms | 0.000 ms | 3584 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_write_locs | 253.5 ms | 0.000 ms | 3584 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3179.4 ms | 1.506 ms | 5848 |
| prefix_multi_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 1169.0 ms | 1.558 ms | 5848 |
| prefix_multi_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_activation_quantize | 1115.6 ms | 0.326 ms | 5848 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 868.3 ms | 1.497 ms | 5848 |
| prefix_multi_112req_wave16 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 847.4 ms | 0.000 ms | 56 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 704.6 ms | 0.086 ms | 136 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 464.1 ms | 0.313 ms | 136 |
| prefix_multi_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 366.1 ms | 0.000 ms | 56 |
| prefix_multi_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 350.3 ms | 0.000 ms | 448 |
| prefix_multi_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_component_page_tables | 325.7 ms | 0.000 ms | 448 |
| prefix_multi_112req_wave16 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 312.9 ms | 0.000 ms | 392 |
| prefix_multi_112req_wave16 | cuda | projection/GEMM | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 214.1 ms | 0.023 ms | 136 |
