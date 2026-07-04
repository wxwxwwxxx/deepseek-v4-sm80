| Scenario | Decode bs | Padded | Source | table_idx rows | device_lens |
| --- | --- | --- | --- | --- | --- |
| identical_prompts_batch | 4 | 4 | eager | [15, 14, 13, 12] | [258, 258, 258, 258] |
| single_target_alone | 1 | 1 | eager | [15] | [258] |
| target_in_batch_slot0 | 4 | 4 | eager | [12, 13, 14, 15] | [258, 258, 258, 258] |
