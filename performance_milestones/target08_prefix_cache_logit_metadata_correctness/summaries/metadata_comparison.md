| Scenario | Expected cached | Actual cached | cached_len | suffix range | semantic metadata | physical counts | prefix page reuse | earliest metadata mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | [256] | [256] | yes | yes | yes | yes | yes | none |
| single_partial_hit_769_c128 | [256] | [256] | yes | yes | yes | yes | yes | none |
| identical_prompts_batch_slots | [256, 256, 256, 256] | [256, 256, 256, 256] | yes | yes | yes | yes | yes | none |
| mixed_hit_miss_batch | [256, 0, 256, 0, 256, 0, 256, 0] | [256, 0, 256, 0, 256, 0, 256, 0] | yes | yes | yes | yes | yes | none |
| swa_boundary_127_128_129_no_hit | [0, 0, 0] | [0, 0, 0] | yes | yes | yes | yes | yes | none |
| c4_boundary_partial261 | [256] | [256] | yes | yes | yes | yes | yes | none |
| page_boundary_255_256_257_258 | [0, 0, 256, 256] | [0, 0, 256, 256] | yes | yes | yes | yes | yes | none |
