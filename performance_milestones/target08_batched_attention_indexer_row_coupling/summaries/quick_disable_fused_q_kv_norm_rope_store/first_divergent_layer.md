| Lens | Scenario | Phase | First checkpoint | Result |
| --- | --- | --- | --- | --- |
| identical rows | identical_prompts_batch | prefill | layer0.attention_backend.swa_selected_full_indices | FAIL max=1536 |
| single vs slots | target_in_batch_slot0 | prefill | layer0.q_after_q_norm_rope | FAIL max=0.0351562 |
