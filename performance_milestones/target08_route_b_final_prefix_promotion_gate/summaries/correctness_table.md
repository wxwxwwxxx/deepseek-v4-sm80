| check | result | evidence |
| --- | --- | --- |
| focused unit tests | pass | KV ownership, metadata graph copy, option guards, graph exact-bs guard |
| serving reports | fail | Route B perf_matrix reports hit a serving correctness blocker |
| Route B text smoke | not_run | not run because Route B serving gate stopped on the ownership blocker |
| slot-pinned guarded oracle | pass | B1/B2 CPU ownership and B3 direct-table graph-copy oracles pass; cross-slot generated equality remains diagnostic per TARGET 08.198 |
| stale read/double-free/leak | pass | component/state no-stale-reuse, repeated eviction, pool assert_no_leak |
