# Metadata Counter Summary

Owner-timing profile runs aggregate counters from rank 0 decode steps.

## Probe Baseline

| field | build bytes | replay copy bytes | direct graph bytes |
| --- | ---: | ---: | ---: |
| `swa_page_indices` | 196608 | 196608 | 0 |
| `c4_sparse_raw_indices` | 786432 | 786432 | 0 |
| `c4_sparse_page_indices` | 786432 | 786432 | 0 |
| `c4_sparse_full_indices` | 786432 | 786432 | 0 |
| `c128_raw_indices` | 98304 | 98304 | 0 |
| `c128_page_indices` | 98304 | 98304 | 0 |
| `c128_full_indices` | 98304 | 98304 | 0 |
| total decode metadata | 2878776 | 2882040 | 0 |

## Probe Direct C4

| field | build bytes | replay copy bytes | direct graph bytes |
| --- | ---: | ---: | ---: |
| `swa_page_indices` | 196608 | 196608 | 0 |
| `c4_sparse_raw_indices` | 1536 | 0 | 786432 |
| `c4_sparse_page_indices` | 1536 | 0 | 786432 |
| `c4_sparse_full_indices` | 1536 | 0 | 786432 |
| `c128_raw_indices` | 98304 | 98304 | 0 |
| `c128_page_indices` | 98304 | 98304 | 0 |
| `c128_full_indices` | 98304 | 98304 | 0 |
| total decode metadata | 524088 | 522744 | 2359296 |

## Large Direct SWA+C4+C128 Profile

| field | build bytes | replay copy bytes | direct graph bytes |
| --- | ---: | ---: | ---: |
| `swa_page_indices` | 10752 | 0 | 1376256 |
| `c4_sparse_raw_indices` | 10752 | 0 | 5505024 |
| `c4_sparse_page_indices` | 10752 | 0 | 5505024 |
| `c4_sparse_full_indices` | 10752 | 0 | 5505024 |
| `c128_raw_indices` | 10752 | 0 | 688128 |
| `c128_page_indices` | 10752 | 0 | 688128 |
| `c128_full_indices` | 10752 | 0 | 688128 |
| `page_table` | 21056 | 21056 | 0 |
| `c4_page_table` | 21056 | 21056 | 0 |
| `c128_page_table` | 21056 | 21056 | 0 |
| `c4_indexer_page_table` | 21056 | 21056 | 0 |
| total decode metadata | 312200 | 259784 | 19955712 |

Interpretation: direct graph generation removed the intended eager source/copy
bytes for SWA/C4/C128 index matrices.  Remaining performance loss is therefore
not explained by those replay copies alone.
