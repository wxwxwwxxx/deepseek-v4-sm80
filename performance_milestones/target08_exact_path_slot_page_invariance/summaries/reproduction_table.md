| Scenario | Prompt lens | Labels | Prefill bs | Decode source | Decode padded | Target table_idx | Target first physical locs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| identical_prompts_batch | 257,257,257,257 | target_slot0,target_slot1,target_slot2,target_slot3 | 4 | eager | 4 | 15 | [0, 1, 2, 3] |
| single_target_alone | 257 | target | 1 | eager | 1 | 12 | [2048, 2049, 2050, 2051] |
| target_in_batch_slot0 | 257,257,257,257 | target,filler1,filler2,filler3 | 4 | eager | 4 | 12 | [2560, 2561, 2562, 2563] |
| target_in_batch_slot1 | 257,257,257,257 | filler0,target,filler2,filler3 | 4 | eager | 4 | 14 | [5120, 5121, 5122, 5123] |
| target_in_batch_slot2 | 257,257,257,257 | filler0,filler1,target,filler3 | 4 | eager | 4 | 14 | [7680, 7681, 7682, 7683] |
| target_in_batch_slot3 | 257,257,257,257 | filler0,filler1,filler2,target | 4 | eager | 4 | 12 | [10240, 10241, 10242, 10243] |
| target_table_row_after_0_dummy | 257 | target | 1 | eager | 1 | 12 | [10752, 10753, 10754, 10755] |
| target_table_row_after_2_dummy | 257 | target | 1 | eager | 1 | 13 | [12288, 12289, 12290, 12291] |
| target_table_row_after_3_dummy | 257 | target | 1 | eager | 1 | 14 | [14336, 14337, 14338, 14339] |
| target_physical_page_none | 257 | target | 1 | eager | 1 | 14 | [14848, 14849, 14850, 14851] |
| target_physical_page_one_page | 257 | target | 1 | eager | 1 | 12 | [16384, 16385, 16386, 16387] |
| target_physical_page_mixed_pages | 257 | target | 1 | eager | 1 | 14 | [18688, 18689, 18690, 18691] |
| swa_boundary_127_128_129_bs3 | 127,128,129 | len127,len128,len129 | 3 | eager | 3 | 14 | [19200, 19201, 19202, 19203] |
| page_boundary_255_256_257_258 | 255,256,257,258 | len255,len256,len257,len258 | 4 | eager | 4 | 13 | [19968, 19969, 19970, 19971] |
| c4_c128_boundary_lengths | 3,4,5,127,128,129,255,256 | len3,len4,len5,len127,len128,len129,len255,len256 | 8 | eager | 8 | 15 | [21760, 21761, 21762] |
