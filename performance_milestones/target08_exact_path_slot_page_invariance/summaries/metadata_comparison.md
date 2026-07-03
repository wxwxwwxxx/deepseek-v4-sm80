| Scenario | Decode bs | Padded | Source | table_idx rows | device_lens |
| --- | --- | --- | --- | --- | --- |
| identical_prompts_batch | 4 | 4 | eager | [15, 14, 13, 12] | [258, 258, 258, 258] |
| single_target_alone | 1 | 1 | eager | [12] | [258] |
| target_in_batch_slot0 | 4 | 4 | eager | [12, 13, 14, 15] | [258, 258, 258, 258] |
| target_in_batch_slot1 | 4 | 4 | eager | [15, 14, 13, 12] | [258, 258, 258, 258] |
| target_in_batch_slot2 | 4 | 4 | eager | [12, 13, 14, 15] | [258, 258, 258, 258] |
| target_in_batch_slot3 | 4 | 4 | eager | [15, 14, 13, 12] | [258, 258, 258, 258] |
| target_physical_page_none | 1 | 1 | eager | [14] | [258] |
| target_physical_page_one_page | 1 | 1 | eager | [12] | [258] |
| target_physical_page_mixed_pages | 1 | 1 | eager | [14] | [258] |
| swa_boundary_127_128_129_bs3 | 3 | 3 | eager | [14, 12, 13] | [128, 129, 130] |
