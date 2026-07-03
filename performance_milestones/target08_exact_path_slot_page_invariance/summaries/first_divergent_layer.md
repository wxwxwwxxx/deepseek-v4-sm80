| Lens | Scenario | Phase | First checkpoint | Result |
| --- | --- | --- | --- | --- |
| identical rows | identical_prompts_batch | prefill | layer3.attention_output | FAIL max=0.101562 |
| single vs slots | target_in_batch_slot0 | prefill | layer0.attention_output | FAIL max=0.0625 |
