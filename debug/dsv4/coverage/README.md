# DeepSeek V4 release runtime coverage

This harness traces every TP worker in a real optimized DeepSeek V4 run. It is
intended to find release-path code that deserves a static call-site audit; an
uncovered line alone is not proof that the line is dead.

Run from the repository root:

```bash
bash debug/dsv4/coverage/audit_release_path.sh
bash debug/dsv4/coverage/run_lifecycle_coverage.sh
bash debug/dsv4/coverage/run_tp8_release_coverage.sh
```

The static audit rejects selectors and replay fallbacks that are incompatible
with the single release path. The lifecycle harness is CPU/no-weight and covers
radix ownership, eviction, abort/reuse, chunked prefill, and metadata contracts.
The TP8 harness covers the integrated optimized model and CUDA Graph path.
Their output directories contain a `.coverage` database, `coverage.json`, and
a text report; TP8 output also contains per-worker data, torchrun logs, and the
smoke result.

Interpret uncovered code in this order:

1. Error handling and environment validation are expected to remain uncovered.
2. Prefix hits, eviction, abort, chunked prefill, and uncommon graph buckets
   require dedicated workloads before judging their lifecycle code.
3. A function is a removal candidate only when it is uncovered in the relevant
   workload and has no release call site or is guarded by a state the release
   configuration cannot enter.

`coverage.py` is already provided by the development dependency on
`pytest-cov`; no runtime dependency is added to Mini-SGLang.

## 2026-07-29 release-path audit

The first TP8 run used the `low_m64` recipe, graph buckets `M=1,2,4`, three
short prefills, and graph-only decode. CUDA graph capture succeeded for every
bucket, in-graph metadata preparation was active, and eager decode count was
zero.

The selected core files reported 60% aggregate line/branch coverage. The most
useful per-file signals were:

- model: 85%
- engine: 75%
- graph runner: 78%
- attention backend: 61%
- release kernel wrapper: 54%
- DeepSeek V4 KV pool: 51%

Every top-level model, attention, and release-kernel execution entry had an
entered body. The audit removed three zero-call remnants: the non-prepacked MoE
dispatch, an unused shared-expert down-cache prepare method, and obsolete
indexer capture-width experiment branches.

Do not use this short run to remove uncovered radix/cache lifecycle code. A
prefix-hit/eviction/abort/chunked-prefill matrix is required first. The old
host-copy graph replay chain and constructor modes with component ownership or
independent SWA disabled remain bounded cleanup candidates because the release
configuration uses in-graph metadata, component ownership, and independent SWA.

The expanded no-weight lifecycle baseline ran 98 tests and reported:

- radix cache: 84%
- CacheManager: 82%
- DeepSeek V4 KV pool: 73%
- attention backend: 72%
- scheduler prefill: 80%
- selected lifecycle files in aggregate: 76%

After the release-path closure, the narrowed matrix ran 95 tests with the same
76% selected aggregate coverage. Unsupported-topology tests were removed while
prefix hit/miss, eviction, abort/reuse, stale handles, capacity pressure,
chunked prefill, and graph metadata contracts remained covered.

The final TP8 graph smoke captured `M=1,2,4`, reported
`prep_metadata_in_graph=true`, used zero eager decode steps, and passed all
three text checks. A separate `--disable-cuda-graph` TP8 smoke reported seven
eager decode steps and produced the same three short answers. The eager run
also caught and closed an over-pruned C128 metadata branch: explicit eager
decode now uses the final one-surface C128 page-index producer without
restoring the removed graph host-copy chain.
