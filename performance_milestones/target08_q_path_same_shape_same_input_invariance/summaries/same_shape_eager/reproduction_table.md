| Scenario | Prompt lens | Target row | Prefill bs | Decode source | Decode padded | Target table_idx |
| --- | --- | --- | --- | --- | --- | --- |
| single_target_alone | 257 | 0 | 1 | eager | 1 | 15 |
| identical_prompts_batch | 257,257,257,257 | 0 | 4 | eager | 4 | 15 |
| target_slot0_fixed_fillers | 257,257,257,257 | 0 | 4 | eager | 4 | 12 |
| target_slot1_fixed_fillers | 257,257,257,257 | 1 | 4 | eager | 4 | 14 |
| target_slot2_fixed_fillers | 257,257,257,257 | 2 | 4 | eager | 4 | 14 |
| target_slot3_fixed_fillers | 257,257,257,257 | 3 | 4 | eager | 4 | 12 |
| target_slot0_altA_fillers | 257,257,257,257 | 0 | 4 | eager | 4 | 12 |
| target_slot0_altB_fillers | 257,257,257,257 | 0 | 4 | eager | 4 | 15 |
