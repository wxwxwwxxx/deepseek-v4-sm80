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
| prefill | target-slot | layer1.attention_backend.merged_attention_output_before_wo | n/a | n/a | n/a | no |
| prefill | filler-content | layer1.attention_backend.merged_attention_output_before_wo | n/a | n/a | n/a | no |
| prefill | identical-row | layer1.attention_backend.merged_attention_output_before_wo | n/a | n/a | n/a | no |
| prefill | target-slot | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.00195312 | 1.37112e-05 | no |
| prefill | identical-row | layer1.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| prefill | target-slot | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| prefill | filler-content | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0488601 | 0.0221593 | no |
| prefill | identical-row | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.00263977 | 4.20865e-05 | no |
| prefill | target-slot | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 5.126e-05 | 8.00937e-07 | no |
| prefill | filler-content | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0684707 | 0.0148649 | no |
| prefill | identical-row | layer4.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.0676441 | 0.0126874 | no |
| prefill | target-slot | final_norm | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.101562 | 0.00892203 | no |
| prefill | filler-content | final_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0625 | 0.00989212 | no |
| prefill | identical-row | final_norm | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.125 | 0.00917622 | no |
| prefill | target-slot | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.915958 | 0.115318 | no |
| prefill | filler-content | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.0259 | 0.128459 | no |
| prefill | identical-row | lm_head_logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 1.26655 | 0.153578 | no |
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
| decode0 | filler-content | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000976562 | 3.82657e-06 | no |
| decode0 | identical-row | layer1.attention_backend.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000976562 | 3.82657e-06 | no |
| decode0 | identical-row | layer1.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0 | 0 | yes |
| decode0 | target-slot | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0 | 0 | yes |
| decode0 | filler-content | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.044976 | 0.010066 | no |
| decode0 | identical-row | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.000768423 | 1.24611e-05 | no |
| decode0 | target-slot | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 9.4533e-05 | 1.47708e-06 | no |
| decode0 | filler-content | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.121267 | 0.0206324 | no |
| decode0 | identical-row | layer4.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.105725 | 0.0143418 | no |
| decode0 | target-slot | final_norm | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.0703125 | 0.0060586 | no |
| decode0 | filler-content | final_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0625 | 0.00936352 | no |
| decode0 | identical-row | final_norm | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.125 | 0.00934473 | no |
| decode0 | target-slot | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.695354 | 0.110111 | no |
| decode0 | filler-content | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.11319 | 0.11425 | no |
| decode0 | identical-row | lm_head_logits | identical_prompts_batch[0] vs identical_prompts_batch[2] | 1.07875 | 0.130227 | no |
| decode1 | target-slot | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.0908203 | 0.017193 | no |
| decode1 | filler-content | layer0.attention_input | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.attention_input | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.130371 | 0.0277975 | no |
| decode1 | target-slot | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.185791 | 0.0411628 | no |
| decode1 | filler-content | layer0.wqa_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.wqa_output | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.30957 | 0.0699366 | no |
| decode1 | target-slot | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.0838623 | 0.0192671 | no |
| decode1 | filler-content | layer0.q_lora_after_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.q_lora_after_norm | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.146484 | 0.0333066 | no |
| decode1 | target-slot | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.226562 | 0.0161078 | no |
| decode1 | filler-content | layer0.q_wqb_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.q_wqb_output | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.298828 | 0.0273285 | no |
| decode1 | target-slot | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 3.70312 | 0.439972 | no |
| decode1 | filler-content | layer0.q_after_q_norm_rope | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.q_after_q_norm_rope | identical_prompts_batch[0] vs identical_prompts_batch[3] | 8.07812 | 0.727525 | no |
| decode1 | target-slot | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 3.23047 | 0.742051 | no |
| decode1 | filler-content | layer0.final_attention_output | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0 | 0 | yes |
| decode1 | identical-row | layer0.final_attention_output | identical_prompts_batch[0] vs identical_prompts_batch[3] | 6.65625 | 1.22831 | no |
| decode1 | target-slot | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode1 | filler-content | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode1 | identical-row | decode0.sampled_token_ids | n/a | n/a | n/a | no |
| decode1 | target-slot | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 6.67188 | 0.17345 | no |
| decode1 | filler-content | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.00195312 | 8.15152e-06 | no |
| decode1 | identical-row | layer1.attention_backend.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[3] | 6.22266 | 0.213991 | no |
| decode1 | target-slot | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 6.67188 | 0.17345 | no |
| decode1 | filler-content | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.00195312 | 8.15152e-06 | no |
| decode1 | identical-row | layer1.merged_attention_output_before_wo | identical_prompts_batch[0] vs identical_prompts_batch[3] | 6.22266 | 0.213991 | no |
| decode1 | target-slot | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.934793 | 0.170846 | no |
| decode1 | filler-content | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.0459849 | 0.012157 | no |
| decode1 | identical-row | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 1.67758 | 0.815099 | no |
| decode1 | target-slot | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.759139 | 0.269718 | no |
| decode1 | filler-content | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.159181 | 0.0195639 | no |
| decode1 | identical-row | layer4.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 1.04147 | 0.527193 | no |
| decode1 | target-slot | final_norm | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.84375 | 0.0620893 | no |
| decode1 | filler-content | final_norm | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.125 | 0.00954653 | no |
| decode1 | identical-row | final_norm | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.53125 | 0.0729388 | no |
| decode1 | target-slot | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 8.11427 | 0.98344 | no |
| decode1 | filler-content | lm_head_logits | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.91345 | 0.140528 | no |
| decode1 | identical-row | lm_head_logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 7.86494 | 0.928054 | no |
