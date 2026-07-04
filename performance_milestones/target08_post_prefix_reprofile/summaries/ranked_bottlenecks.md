# Ranked Bottlenecks

| rank | bucket | evidence | seconds/ms | share | interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | decode forward | serving_mixed phase total | 9.7624 | 0.5697 | dominant remaining E2E bucket; owner timing points to comm/attention work inside it |
| 2 | communication / all-reduce owners | owner timing profile, attribution only | 6489.2434 | 0.3787 | wo_b row-parallel, MoE reduce-once, and embedding all-reduce are top owner labels |
| 3 | prefill forward / TTFT base cost | serving_mixed phase total | 4.7832 | 0.2791 | not helped unless workload has real prefix hits |
| 4 | decode prepare / prefix metadata runtime | serving_mixed phase total plus owner timing | 1.0919 | 0.0637 | post-lifetime-cache tax; compare against 08.28 and owner rows |
| 5 | component page-table lifetime cache owner | owner timing profile, attribution only | 356.1025 | 0.0208 | metadata owner is now small relative to decode forward |
