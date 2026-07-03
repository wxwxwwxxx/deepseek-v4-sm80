| Scenario | suffix prefill off/on | decode off/on | decode graph/eager | prefill top10 | decode top10 |
| --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | FAIL max=0.472356 | FAIL max=0.590127 | pass max=0 | no | no |
| single_partial_hit_769_c128 | FAIL max=1.72707 | FAIL max=10.0014 | pass max=0 | no | no |
| identical_prompts_batch_slots | FAIL max=4.86287 | FAIL max=9.56691 | pass max=0 | no | no |
| mixed_hit_miss_batch | FAIL max=4.37167 | FAIL max=10.4436 | pass max=0 | no | no |
| swa_boundary_127_128_129_no_hit | pass max=0 | FAIL max=0.245584 | FAIL max=1.9489 | yes | no |
| c4_boundary_partial261 | FAIL max=1.47294 | FAIL max=1.82026 | pass max=0 | no | no |
| page_boundary_255_256_257_258 | FAIL max=4.24537 | FAIL max=7.13711 | pass max=0 | no | no |
