| Phase | Group | Owner bucket | First checkpoint | Worst pair | Max abs | Mean abs |
| --- | --- | --- | --- | --- | --- | --- |
| prefill | target-slot | later-layer attention/indexer | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 5.126e-05 | 8.00937e-07 |
| prefill | filler-content | later-layer attention/indexer | layer1.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.00195312 | 1.37112e-05 |
| prefill | identical-row | later-layer attention/indexer | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.00263977 | 4.20865e-05 |
| decode0 | target-slot | later-layer attention/indexer | layer4.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 9.4533e-05 | 1.47708e-06 |
| decode0 | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000976562 | 3.82657e-06 |
| decode0 | identical-row | later-layer attention/indexer | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.000768423 | 1.24611e-05 |
| decode1 | target-slot | sampler feedback | decode0.sampled_token_ids | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.695354 | 0.110111 |
| decode1 | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.00195312 | 8.15152e-06 |
| decode1 | identical-row | sampler feedback | decode0.sampled_token_ids | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.959872 | 0.117708 |
