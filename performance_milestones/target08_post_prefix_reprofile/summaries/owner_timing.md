# Owner Timing

| run | variant | scenario | section | label | max-rank ms | count |
| --- | --- | --- | --- | --- | --- | --- |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3351.6042 | 5848 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3336.2684 | 5504 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3200.1859 | 5848 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3128.1046 | 5848 |
| profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3004.8750 | 3784 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2486.3387 | 3784 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 2220.1558 | 128 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 2172.3463 | 48 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | host | dsv4.prepare.decode.attention_metadata | 1976.1841 | 3528 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 1633.5723 | 5848 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 1567.6785 | 5848 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 1498.2598 | 5504 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1492.4249 | 136 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1483.0363 | 88 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1469.3135 | 136 |
| profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1374.6500 | 88 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | host | dsv4.prepare.decode.attention_metadata | 1336.5405 | 3528 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1314.0829 | 136 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 1154.5177 | 48 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 1111.9479 | 56 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 909.4580 | 3584 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 889.3899 | 5504 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 876.8931 | 56 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 858.6540 | 5848 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 855.3292 | 96 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 846.0992 | 3584 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 810.6878 | 3784 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | host | dsv4.prepare.prefill.attention_metadata | 787.2578 | 8 |
| profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 728.5740 | 3784 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 700.4669 | 56 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.shared_down.bf16_cache_local_total | 509.0395 | 5504 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 463.4095 | 5848 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 458.1059 | 5848 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.metadata.decode.make_c128_indices | 457.3876 | 1024 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 432.0505 | 3584 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 416.4905 | 56 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 395.2982 | 8 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 381.3584 | 56 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 361.2777 | 448 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_component_page_tables | 356.1025 | 3584 |

## Component Row Counters

| run | scenario | label | count |
| --- | --- | --- | --- |
| profile_promoted_4096_128 | historical_4096_128_bs4 | dsv4.component_page_table_cache.rows/decode/dirty | 4 |
| profile_promoted_4096_128 | historical_4096_128_bs4 | dsv4.component_page_table_cache.rows/decode/clean | 504 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 16 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 224 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 64 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 32 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 32 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 16 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | prefix_eviction_pressure_96req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 96 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | prefix_eviction_pressure_96req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 0 |
| profile_promoted_prefix_multi_112req_wave16 | prefix_multi_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 112 |
| profile_promoted_prefix_multi_112req_wave16 | prefix_multi_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 672 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 112 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 1568 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 448 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 224 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 224 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 112 |
