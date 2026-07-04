| Phase | Group | Pair | Logit max abs | Logit mean abs | Top1 ids | Top1 changed | Sampled ids | Sampled changed | Left top1 margin | 2*max_abs >= margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.1501 | 0.137505 | 32->32 | no | 32->32 | no | 0.929865 | yes |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | 1.22151 | 0.163528 | 32->32 | no | 32->32 | no | 0.929865 | yes |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 1.20849 | 0.139527 | 32->32 | no | 32->32 | no | 0.929865 | yes |
| prefill | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.00624 | 0.136297 | 32->32 | no | 32->32 | no | 0.929865 | yes |
| prefill | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altB_fillers[0] | 0.845127 | 0.156562 | 32->32 | no | 32->32 | no | 0.929865 | yes |
| prefill | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.884253 | 0.127772 | 32->32 | no | 32->32 | no | 1.07734 | yes |
| prefill | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.883242 | 0.090831 | 32->32 | no | 32->32 | no | 1.07734 | yes |
| prefill | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.887962 | 0.136481 | 32->32 | no | 32->32 | no | 1.07734 | yes |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 0.129613 | 223->603 | yes | 223->603 | yes | 0.156403 | yes |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | 0.982476 | 0.130267 | 223->603 | yes | 223->603 | yes | 0.156403 | yes |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.85173 | 0.0973762 | 223->223 | no | 223->223 | no | 0.156403 | yes |
| decode0 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.9766 | 0.128717 | 223->223 | no | 223->223 | no | 0.156403 | yes |
| decode0 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altB_fillers[0] | 0.793823 | 0.107561 | 223->223 | no | 223->223 | no | 0.156403 | yes |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[1] | 0.911032 | 0.159881 | 322->223 | yes | 322->223 | yes | 0.116428 | yes |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 0.10798 | 322->603 | yes | 322->603 | yes | 0.116428 | yes |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.780247 | 0.112801 | 322->603 | yes | 322->603 | yes | 0.116428 | yes |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 8.30493 | 0.960316 | 223->327 | yes | 223->327 | yes | 2.06961 | yes |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | 8.1021 | 0.942231 | 223->327 | yes | 223->327 | yes | 2.06961 | yes |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.779362 | 0.109116 | 223->223 | no | 223->223 | no | 2.06961 | no |
| decode1 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.633387 | 0.105347 | 223->223 | no | 223->223 | no | 2.06961 | no |
| decode1 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altB_fillers[0] | 0.977363 | 0.119098 | 223->223 | no | 223->223 | no | 2.06961 | no |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[1] | 8.74046 | 1.12454 | 1018->223 | yes | 1018->223 | yes | 0.704569 | yes |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 7.86887 | 0.931572 | 1018->223 | yes | 1018->223 | yes | 0.704569 | yes |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[3] | 8.03972 | 0.926484 | 1018->223 | yes | 1018->223 | yes | 0.704569 | yes |
