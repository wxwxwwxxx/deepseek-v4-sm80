# C4-First Probe Effect

Scenario: `decode_ladder_bs16`, page size 256, 128 pages, CUDA graph buckets
`[1,2,4,8,16]`, owner timing enabled.

| mode | status | output tok/s | decode tok/s | decode prepare s | decode forward s | TTFT s | graph replay/eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Route B graph baseline | pass | 60.9866 | 164.9115 | 0.6912 | 2.3285 | 3.4913 | 63/0 |
| Route B direct C4 | pass | 80.0069 | 182.9806 | 0.6417 | 2.0986 | 2.2110 | 63/0 |

Owner counters confirm the intended first cut:

| field | baseline build bytes | baseline copy bytes | direct build bytes | direct copy bytes | direct graph bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c4_sparse_raw_indices` | 786432 | 786432 | 1536 | 0 | 786432 |
| `c4_sparse_page_indices` | 786432 | 786432 | 1536 | 0 | 786432 |
| `c4_sparse_full_indices` | 786432 | 786432 | 1536 | 0 | 786432 |

C4 direct-to-dst graph buffers worked and reduced materialization/copy.  The
benefit did not scale enough in the large-wave gate after extending to all index
groups.
