| Phase | Group | Checkpoint | Worst pair | Max abs | Mean abs | Exact |
| --- | --- | --- | --- | --- | --- | --- |
| prefill | target-slot | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| prefill | identical-row | layer0.attention_input | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| prefill | identical-row | layer0.wqa_output | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| prefill | identical-row | layer0.q_lora_after_norm | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| prefill | identical-row | layer0.q_wqb_output | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| prefill | identical-row | layer0.q_after_q_norm_rope | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| prefill | identical-row | layer0.final_attention_output | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| prefill | filler-content | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| prefill | identical-row | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| prefill | target-slot | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000976562 | 4.592e-07 | no |
| prefill | identical-row | layer1.attention_backend.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.000812054 | 1.26883e-05 | no |
| prefill | filler-content | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0922782 | 0.0205868 | no |
| prefill | identical-row | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 6.81877e-05 | 1.06543e-06 | no |
| prefill | target-slot | final_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.125 | 0.0110386 | no |
| prefill | filler-content | final_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.078125 | 0.0105856 | no |
| prefill | identical-row | final_norm | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.125 | 0.00901788 | no |
| prefill | target-slot | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 1.20849 | 0.139527 | no |
| prefill | filler-content | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.00624 | 0.136297 | no |
| prefill | identical-row | lm_head_logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.887962 | 0.136481 | no |
| decode0 | target-slot | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode0 | identical-row | layer0.attention_input | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode0 | identical-row | layer0.wqa_output | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode0 | identical-row | layer0.q_lora_after_norm | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode0 | identical-row | layer0.q_wqb_output | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode0 | identical-row | layer0.q_after_q_norm_rope | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode0 | identical-row | layer0.final_attention_output | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode0 | filler-content | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode0 | identical-row | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode0 | target-slot | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000244141 | 1.52737e-07 | no |
| decode0 | identical-row | layer1.attention_backend.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.000867605 | 1.35563e-05 | no |
| decode0 | filler-content | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0834758 | 0.0169597 | no |
| decode0 | identical-row | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.000104427 | 1.63168e-06 | no |
| decode0 | target-slot | final_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.09375 | 0.0104806 | no |
| decode0 | filler-content | final_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.125 | 0.00998252 | no |
| decode0 | identical-row | final_norm | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.1875 | 0.0111864 | no |
| decode0 | target-slot | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 0.129613 | no |
| decode0 | filler-content | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.9766 | 0.128717 | no |
| decode0 | identical-row | lm_head_logits | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 0.10798 | no |
| decode1 | target-slot | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.0908203 | 0.017193 | no |
| decode1 | filler-content | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.attention_input | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.130371 | 0.0277975 | no |
| decode1 | target-slot | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.185791 | 0.0411628 | no |
| decode1 | filler-content | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.wqa_output | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.30957 | 0.0699366 | no |
| decode1 | target-slot | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.0838623 | 0.0192671 | no |
| decode1 | filler-content | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.q_lora_after_norm | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.146484 | 0.0333066 | no |
| decode1 | target-slot | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.226562 | 0.0161078 | no |
| decode1 | filler-content | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.q_wqb_output | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.298828 | 0.0273285 | no |
| decode1 | target-slot | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 3.70312 | 0.439972 | no |
| decode1 | filler-content | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.q_after_q_norm_rope | identical_prompts_batch[0] vs identical_prompts_batch[2] | 8.07812 | 0.727525 | no |
| decode1 | target-slot | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 3.23047 | 0.742051 | no |
| decode1 | filler-content | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.final_attention_output | identical_prompts_batch[0] vs identical_prompts_batch[2] | 6.65625 | 1.22831 | no |
| decode1 | target-slot | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode1 | filler-content | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode1 | identical-row | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode1 | target-slot | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 6.66406 | 0.173475 | no |
| decode1 | filler-content | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 6.10352e-05 | 1.67638e-08 | no |
| decode1 | identical-row | layer1.attention_backend.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[2] | 6.22266 | 0.213824 | no |
| decode1 | target-slot | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.00861 | 0.167323 | no |
| decode1 | filler-content | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0707439 | 0.0127097 | no |
| decode1 | identical-row | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[1] | 1.69928 | 0.763619 | no |
| decode1 | target-slot | final_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.90625 | 0.0600416 | no |
| decode1 | filler-content | final_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0859375 | 0.0077449 | no |
| decode1 | identical-row | final_norm | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.59375 | 0.0749249 | no |
| decode1 | target-slot | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 8.30493 | 0.960316 | no |
| decode1 | filler-content | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.633387 | 0.105347 | no |
| decode1 | identical-row | lm_head_logits | identical_prompts_batch[0] vs identical_prompts_batch[1] | 8.74046 | 1.12454 | no |
