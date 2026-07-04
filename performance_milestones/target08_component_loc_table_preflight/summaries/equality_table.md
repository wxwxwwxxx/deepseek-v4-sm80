| scenario | phase | hit style | rows | swa | c4 sparse | c128 | c4 out | c128 out | indexer out | indexer page | indexer loc | state c4 | state c128 | state indexer | result |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_hit_decode_page257 | decode | full hit | 1 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| partial_hit_suffix_prefill_256_to_258 | prefill | partial hit | 2 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| miss_style_prefill_258 | prefill | miss style | 258 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| page_boundaries_255_256_257_258_decode | decode | boundary | 4 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| c4_boundaries_decode | decode | C4 boundary | 5 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| c128_boundaries_decode | decode | C128 boundary | 5 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| swa_127_128_129_decode | decode | SWA boundary | 3 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| batched_same_layout_rows | prefill | batched same-layout | 4 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
