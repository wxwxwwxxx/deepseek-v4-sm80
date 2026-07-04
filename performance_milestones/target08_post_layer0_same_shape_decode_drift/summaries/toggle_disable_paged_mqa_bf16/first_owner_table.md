| Phase | Group | Owner bucket | First checkpoint | Worst pair | Max abs | Mean abs |
| --- | --- | --- | --- | --- | --- | --- |
| prefill | target-slot | later-layer attention/indexer | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.000812054 | 1.26883e-05 |
| prefill | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000976562 | 4.592e-07 |
| prefill | identical-row | later-layer attention/indexer | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 6.81877e-05 | 1.06543e-06 |
| decode0 | target-slot | later-layer attention/indexer | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.000867605 | 1.35563e-05 |
| decode0 | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000244141 | 1.52737e-07 |
| decode0 | identical-row | later-layer attention/indexer | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.000104427 | 1.63168e-06 |
| decode1 | target-slot | sampler feedback | decode0.sampled_token_ids | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 0.129613 |
| decode1 | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 6.10352e-05 | 1.67638e-08 |
| decode1 | identical-row | sampler feedback | decode0.sampled_token_ids | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 0.10798 |
