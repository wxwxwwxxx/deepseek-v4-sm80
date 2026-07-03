| Scenario | Prefill eager/graph | Decode eager/graph | Real bs | Padded bs | Graph source |
| --- | --- | --- | --- | --- | --- |
| identical_prompts_batch | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| single_target_alone | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| target_in_batch_slot0 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| target_in_batch_slot1 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| target_in_batch_slot2 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| target_in_batch_slot3 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| target_table_row_after_0_dummy | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| target_table_row_after_2_dummy | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| target_table_row_after_3_dummy | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| target_physical_page_none | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| target_physical_page_one_page | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| target_physical_page_mixed_pages | pass max=0 | pass max=0 | 1 | 1 | cuda_graph_replay |
| swa_boundary_127_128_129_bs3 | pass max=0 | FAIL max=2.18697 | 3 | 4 | cuda_graph_replay |
| page_boundary_255_256_257_258 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| c4_c128_boundary_lengths | pass max=0 | pass max=0 | 8 | 8 | cuda_graph_replay |
