# Metadata Update Pressure

Call counts from owner-timing profile runs. Per-request page tables and per-prefix-hit C128 rows are expected to be stable candidates if their calls track decode replay steps.

| mode | counter | field | stable | value |
| --- | --- | --- | --- | --- |
| phase1 prefix on | dsv4.metadata_build.calls | c128_full_indices | per-prefix-hit | 441 |
| phase1 prefix on | dsv4.metadata_build.calls | c128_page_indices | per-prefix-hit | 441 |
| phase1 prefix on | dsv4.metadata_build.calls | c128_raw_indices | per-prefix-hit | 441 |
| phase1 prefix on | dsv4.metadata_build.calls | page_table | per-request | 441 |
| phase1 prefix on | dsv4.replay_metadata_copy.calls | c128_full_indices | per-prefix-hit;fallback | 441 |
| phase1 prefix on | dsv4.replay_metadata_copy.calls | c128_page_indices | per-prefix-hit;fallback | 441 |
| phase1 prefix on | dsv4.replay_metadata_copy.calls | c128_raw_indices | per-prefix-hit;fallback | 441 |
| phase1 prefix on | dsv4.replay_metadata_copy.calls | page_table | per-request;fallback | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | c128_full_indices | per-prefix-hit | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | c128_page_indices | per-prefix-hit | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | c128_page_table | per-request | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | c128_raw_indices | per-prefix-hit | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | c4_indexer_page_table | per-request | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | c4_page_table | per-request | 441 |
| Route B graph baseline | dsv4.metadata_build.calls | page_table | per-request | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | c128_full_indices | per-prefix-hit;fallback | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | c128_page_indices | per-prefix-hit;fallback | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | c128_page_table | per-request | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | c128_raw_indices | per-prefix-hit;fallback | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | c4_indexer_page_table | per-request | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | c4_page_table | per-request | 441 |
| Route B graph baseline | dsv4.replay_metadata_copy.calls | page_table | per-request;fallback | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | c128_full_indices | per-prefix-hit | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | c128_page_indices | per-prefix-hit | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | c128_page_table | per-request | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | c128_raw_indices | per-prefix-hit | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | c4_indexer_page_table | per-request | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | c4_page_table | per-request | 441 |
| Route B direct C4 | dsv4.metadata_build.calls | page_table | per-request | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | c128_full_indices | per-prefix-hit;fallback | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | c128_page_indices | per-prefix-hit;fallback | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | c128_page_table | per-request | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | c128_raw_indices | per-prefix-hit;fallback | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | c4_indexer_page_table | per-request | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | c4_page_table | per-request | 441 |
| Route B direct C4 | dsv4.replay_metadata_copy.calls | page_table | per-request;fallback | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | c128_full_indices | per-prefix-hit | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | c128_page_indices | per-prefix-hit | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | c128_page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | c128_raw_indices | per-prefix-hit | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | c4_indexer_page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | c4_page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.metadata_build.calls | page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.replay_metadata_copy.calls | c128_page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.replay_metadata_copy.calls | c4_indexer_page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.replay_metadata_copy.calls | c4_page_table | per-request | 441 |
| Route B direct SWA+C4+C128 | dsv4.replay_metadata_copy.calls | page_table | per-request;fallback | 441 |
