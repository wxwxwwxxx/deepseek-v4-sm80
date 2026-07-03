| Scenario | Coverage | Warm lens | Probe lens | Expected cached_len | Graph |
| --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | single-request full hit, page boundary around 256 | [257] | [257] | [256] | True |
| single_partial_hit_769_c128 | single-request partial hit, C128 boundary around 128 | [257] | [769] | [256] | True |
| identical_prompts_batch_slots | identical prompts in batch slots, single-request full hit | [257] | [257, 257, 257, 257] | [256, 256, 256, 256] | True |
| mixed_hit_miss_batch | mixed hit/miss batch | [257] | [257, 257, 257, 257, 257, 257, 257, 257] | [256, 0, 256, 0, 256, 0, 256, 0] | True |
| swa_boundary_127_128_129_no_hit | SWA boundary around 128, prefix-disabled equivalent miss path | [] | [127, 128, 129] | [0, 0, 0] | True |
| c4_boundary_partial261 | C4 boundary around 4 | [257] | [261] | [256] | True |
| page_boundary_255_256_257_258 | page boundary around 256, mixed hit/miss batch | [258] | [255, 256, 257, 258] | [0, 0, 256, 256] | True |
