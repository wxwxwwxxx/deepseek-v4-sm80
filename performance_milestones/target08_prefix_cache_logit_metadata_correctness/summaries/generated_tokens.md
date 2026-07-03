| Scenario | off/on match | on graph/eager match | off tokens | on tokens | on eager tokens |
| --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | no | yes | [[89, 223]] | [[89, 269]] | [[89, 269]] |
| single_partial_hit_769_c128 | no | yes | [[294, 710]] | [[11, 223]] | [[11, 223]] |
| identical_prompts_batch_slots | no | yes | [[344, 928], [84, 223]] +2 rows | [[344, 928], [344, 928]] +2 rows | [[344, 928], [344, 928]] +2 rows |
| mixed_hit_miss_batch | no | yes | [[740, 446], [80, 201]] +6 rows | [[740, 446], [80, 201]] +6 rows | [[740, 446], [80, 201]] +6 rows |
| swa_boundary_127_128_129_no_hit | yes | yes | [[271, 988], [223, 223]] +1 rows | [[271, 988], [223, 223]] +1 rows | [[271, 988], [223, 223]] +1 rows |
| c4_boundary_partial261 | yes | yes | [[223, 223]] | [[223, 223]] | [[223, 223]] |
| page_boundary_255_256_257_258 | no | yes | [[223, 993], [223, 223]] +2 rows | [[223, 223], [223, 223]] +2 rows | [[223, 223], [223, 223]] +2 rows |
