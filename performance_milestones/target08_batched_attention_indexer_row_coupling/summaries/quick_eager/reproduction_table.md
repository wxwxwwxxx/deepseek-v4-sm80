| Scenario | Prompt lens | Labels | Prefill bs | Decode source | Decode padded | Target table_idx | Target first physical locs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_target_alone | 257 | target | 1 | eager | 1 | 15 | [0, 1, 2, 3] |
| identical_prompts_batch | 257,257,257,257 | target_slot0,target_slot1,target_slot2,target_slot3 | 4 | eager | 4 | 15 | [512, 513, 514, 515] |
| target_in_batch_slot0 | 257,257,257,257 | target,filler1,filler2,filler3 | 4 | eager | 4 | 12 | [2560, 2561, 2562, 2563] |
