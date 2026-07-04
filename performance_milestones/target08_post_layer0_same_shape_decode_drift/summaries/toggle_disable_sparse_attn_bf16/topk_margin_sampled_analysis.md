| Phase | Group | Pair | Logit max abs | Logit mean abs | Top1 ids | Top1 changed | Sampled ids | Sampled changed | Left top1 margin | 2*max_abs >= margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.592521 | 0.0987744 | 32->32 | no | 32->32 | no | 1.14012 | yes |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | n/a | n/a | n/a | no | n/a | no | n/a | no |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.915958 | 0.115318 | 32->32 | no | 32->32 | no | 1.14012 | yes |
| prefill | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.0259 | 0.128459 | 32->32 | no | 32->32 | no | 1.14012 | yes |
| prefill | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altB_fillers[0] | n/a | n/a | n/a | no | n/a | no | n/a | no |
| prefill | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.829313 | 0.12448 | 32->32 | no | 32->32 | no | 1.14288 | yes |
| prefill | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 1.11412 | 0.127519 | 32->32 | no | 32->32 | no | 1.14288 | yes |
| prefill | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[3] | 1.26655 | 0.153578 | 32->32 | no | 32->32 | no | 1.14288 | yes |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.487177 | 0.0790657 | 603->603 | no | 603->603 | no | 0.184565 | yes |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | n/a | n/a | n/a | no | n/a | no | n/a | no |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.695354 | 0.110111 | 603->223 | yes | 603->223 | yes | 0.184565 | yes |
| decode0 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.11319 | 0.11425 | 603->603 | no | 603->603 | no | 0.184565 | yes |
| decode0 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altB_fillers[0] | n/a | n/a | n/a | no | n/a | no | n/a | no |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[1] | 1.03891 | 0.126653 | 603->603 | no | 603->603 | no | 0.0567665 | yes |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 1.07875 | 0.130227 | 603->603 | no | 603->603 | no | 0.0567665 | yes |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.959872 | 0.117708 | 603->322 | yes | 603->322 | yes | 0.0567665 | yes |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 0.716138 | 0.104881 | 327->223 | yes | 327->223 | yes | 0.162966 | yes |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | n/a | n/a | n/a | no | n/a | no | n/a | no |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 8.11427 | 0.98344 | 327->223 | yes | 327->223 | yes | 0.162966 | yes |
| decode1 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.91345 | 0.140528 | 327->327 | no | 327->327 | no | 0.162966 | yes |
| decode1 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altB_fillers[0] | n/a | n/a | n/a | no | n/a | no | n/a | no |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.995834 | 0.133272 | 327->327 | no | 327->327 | no | 0.0372295 | yes |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.974686 | 0.152267 | 327->327 | no | 327->327 | no | 0.0372295 | yes |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[3] | 7.86494 | 0.928054 | 327->1018 | yes | 327->1018 | yes | 0.0372295 | yes |
