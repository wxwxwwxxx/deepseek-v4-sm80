| Phase | Group | Worst pair | Logit max abs | Top1 ids | Top10 same | Sampled ids | Sampled same | Left top1 margin | 2*max_abs >= margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | shape_change | single_target_alone[0] vs target_slot0_fixed_fillers[0] | 0.848237 | 32->32 | no | 32->32 | yes | 1.11026 | yes |
| prefill | same_shape_position | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | 1.22151 | 32->32 | no | 32->32 | yes | 0.929865 | yes |
| prefill | same_shape_filler | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.00624 | 32->32 | no | 32->32 | yes | 0.929865 | yes |
| prefill | identical_rows | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.887962 | 32->32 | no | 32->32 | yes | 1.07734 | yes |
| decode | shape_change | single_target_alone[0] vs target_slot0_fixed_fillers[0] | 0.756046 | 223->223 | no | 223->223 | yes | 0.149835 | yes |
| decode | same_shape_position | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 223->603 | no | 223->603 | no | 0.156403 | yes |
| decode | same_shape_filler | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.9766 | 223->223 | no | 223->223 | yes | 0.156403 | yes |
| decode | identical_rows | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 322->603 | no | 322->603 | no | 0.116428 | yes |
