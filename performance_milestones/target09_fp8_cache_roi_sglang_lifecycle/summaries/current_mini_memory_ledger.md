| component | bytes/page | MiB/page | GiB at 128 pages | evidence |
| --- | --- | --- | --- | --- |
| swa_bf16 | 11,272,192 | 10.75 | 1.344 | source-derived |
| c4_bf16 | 1,376,256 | 1.31 | 0.164 | source-derived |
| c128_bf16 | 40,960 | 0.04 | 0.005 | source-derived |
| c4_indexer_bf16 | 344,064 | 0.33 | 0.041 | source-derived |
| c4_indexer_fp8_side_cache | 177,408 | 0.17 | 0.021 | runtime-active in promoted bundle, source-derived size |
| c4_state_bf16 | 688,128 | 0.66 | 0.082 | source-derived |
| c4_indexer_state_bf16 | 172,032 | 0.16 | 0.021 | source-derived |
| c128_state_bf16 | 5,242,880 | 5.00 | 0.625 | source-derived |
| total_with_existing_fp8_side | 19,313,920 | 18.42 | 2.302 | source-derived formula; matches promoted indexer FP8 side mode |
| total_without_fp8_side_reference | 19,136,512 | 18.25 | 2.281 | source-derived formula |

Runtime report from TARGET 09.0: 2,491,495,680 B / 2.320 GiB per rank. That equals 129.0 pages at the promoted component formula, so the README uses the user-requested 128-page formula for ROI and keeps the 129-page runtime pool as separate runtime-proven context.
