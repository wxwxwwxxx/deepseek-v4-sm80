# Metadata Update Pressure

The old Route B direct-C4 path rebuilt each component page-table field once per decode replay step. The new opt-in still selects/copies graph source rows every step, but actual row rebuilds are request-slot refreshes.

| mode | counter | field | stable | value | source |
| --- | --- | --- | --- | --- | --- |
| Route B direct C4 | dsv4.metadata_build.calls | c128_page_table | per-request | 441 | 08.26 direct C4 |
| Route B direct C4 | dsv4.metadata_build.calls | c4_indexer_page_table | per-request | 441 | 08.26 direct C4 |
| Route B direct C4 | dsv4.metadata_build.calls | c4_page_table | per-request | 441 | 08.26 direct C4 |
| Route B direct C4 + lifetime cache | dsv4.component_page_table_cache.rows | dirty_rows | request-slot refresh | 112 | 08.27 profile rank0 |
| Route B direct C4 + lifetime cache | dsv4.component_page_table_cache.rows | clean_rows | request-slot reuse | 2576 | 08.27 profile rank0 |
