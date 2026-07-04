# TARGET 08.28: DSV4 Route B Lifetime Cache Promotion Gate

## Status

Run this after TARGET 08.27.

TARGET 08.27 implemented an experimental Route B component page-table lifetime
cache:

```text
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
```

The result is strong enough to justify a promotion/robustness gate, but not
enough to make it default immediately.  This target should validate the opt-in
across broader prefix-cache serving workloads, especially table-slot reuse,
prefix-hit changes, and eviction pressure.

Do not use this target to invent another metadata mechanism.  The job is to
prove whether the TARGET 08.27 SGLang-aligned lifetime cache is robust enough
to become part of the preferred Route B opt-in bundle.

## Goal

Decide whether the TARGET 08.27 component page-table lifetime cache can be
promoted from an experimental opt-in to the preferred Route B prefix-cache
runtime configuration.

The target should answer:

1. Does the lifetime cache remain correct under prefix reuse, mixed misses,
   eviction pressure, and decode-only controls?
2. Do table-slot reuse, prefix-handle movement, active-page growth, and
   component eviction invalidate rows correctly?
3. Does throughput remain close to phase1 prefix-on and clearly ahead of Route B
   direct C4 across the broader serving suite?
4. If promotion is not safe, is the blocker a small lifecycle bug, measurement
   noise, or evidence that the design needs a deeper graph/metadata contract?

## Required Reading

Project route:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.27_dsv4_sm80_sglang_aligned_route_b_metadata_lifetime.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/README.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/DESIGN.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/summaries/throughput_repeat.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/summaries/prepare_owner_profile.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/summaries/metadata_update_pressure.md`

Core code references:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`
- `tests/benchmark/test_deepseek_v4_perf_matrix.py`

SGLang references remain useful only for lifecycle sanity checks:

- `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `/workspace/sglang-main/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`

## Starting Evidence

TARGET 08.27 `serving_mixed_112req_wave16`, page size 256, TP8, graph buckets
`[1,2,4,8,16]`:

| mode | output tok/s | decode prepare s | decode forward s | replay/eager |
| --- | ---: | ---: | ---: | --- |
| phase1 prefix on | 169.7381 | 0.9403 | 9.9757 | 441/0 |
| Route B graph baseline | 136.2373 | 4.4798 | 10.0897 | 441/0 |
| Route B direct C4 | 138.1281 | 4.2067 | 10.1297 | 441/0 |
| Route B direct C4 + lifetime cache | 162.4726 | 1.1416 | 10.0077 | 441/0 |

The 08.27 owner profile showed:

- component page-table build time dropped from `3341.1692 ms` to `354.4121 ms`;
- dirty component rows: `112`;
- clean component row reuses: `2576`;
- graph replay remained `441/0`;
- verifier mode passed the full `serving_mixed_112req_wave16` workload.

Remaining known boundaries:

- graph destination component page-table buffers are still copied every replay;
- full `page_table` construction is unchanged;
- broader eviction and mixed-workload promotion have not been gated yet;
- SWA ownership, low precision, attention, MoE, and NCCL are out of scope.

## Scope

Allowed:

- run correctness, verifier, text smoke, and throughput gates;
- add targeted tests or assertions for table-slot reuse, stale row detection,
  prefix-handle movement, and eviction lifecycle;
- fix small lifecycle bugs in the TARGET 08.27 lifetime cache;
- improve benchmark variants, scripts, and summaries for the promotion gate;
- if all gates pass, add or document a preferred Route B benchmark variant that
  enables direct C4 graph metadata plus the lifetime cache.

Not allowed:

- making the prefix cache or Route B lifetime cache default for all users;
- introducing a new dirty-row subsystem;
- porting SGLang raw-metadata graph prep;
- replacing graph metadata copy with reference assignment;
- independent SWA ownership;
- low precision, attention, MoE, or communication optimization;
- broad scheduler/radix/KV-cache rewrites.

## Required Variants

Use separate `torchrun` invocations per variant.  Do not reuse an Engine in the
same Python process.

Compare at least:

- phase1 prefix on;
- Route B graph baseline;
- Route B direct C4;
- Route B direct C4 + lifetime cache;
- Route B direct C4 + lifetime cache + verifier, for correctness only.

Use the standard runtime shape:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

The lifetime-cache variant should enable:

```text
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
```

The verifier variant should additionally enable:

```text
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1
```

## Required Workloads

Run in this order:

1. TP8 text smoke with verifier on.
2. `serving_mixed_112req_wave16` with verifier on once.
3. `serving_mixed_112req_wave16` verifier off, at least 3 repeats.
4. `prefix_multi_112req_wave16` verifier off, at least 3 repeats.
5. `prefix_eviction_pressure_96req_wave16` with verifier on once, then verifier
   off for at least 2 repeats if the verifier pass is not too slow.
6. `decode_ladder_bs16` as a non-prefix/decode-control run.

Optional if runtime is available:

- repeat `prefix_multi_112req_wave16` with verifier on;
- add a small targeted table-slot reuse micro scenario if existing workloads do
  not exercise slot reuse enough;
- run a short nsys profile only if throughput regresses or prepare attribution
  becomes ambiguous.

## Correctness And Lifecycle Checks

Report these explicitly:

- text smoke output and status;
- verifier mismatch count/status;
- graph replay/eager count by scenario;
- prefix metrics: hit requests, saved prefill tokens, evictions, evicted tokens;
- component row dirty/clean counters;
- any table-slot reuse counters or targeted tests added in this target;
- cache integrity checks after eviction pressure;
- whether failures, if any, are correctness, graph replay, OOM/capacity, or
  performance-only failures.

Cross-slot generated-token equality remains diagnostic only.  Do not block on
batch-slot invariance unless text quality, metadata verifier, or cache integrity
fails.

## Performance Gates

For `serving_mixed_112req_wave16`:

- output tok/s should remain within `3%` of TARGET 08.27's `162.4726` mean;
- decode prepare should remain near `1.14 s` and must not return to the
  `4 s` Route B direct-C4 range;
- graph replay/eager should remain `441/0`.

For broader prefix workloads:

- lifetime cache must stay materially faster than Route B direct C4 on workloads
  where decode prepare was previously dominant;
- no scenario may show a repeated, unexplained `>5%` throughput regression
  versus Route B direct C4;
- verifier-on correctness passes are more important than verifier-on speed.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_lifetime_cache_promotion_gate/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands and environment variables;
- git status summary;
- correctness/text-smoke table;
- workload-by-workload throughput table;
- phase1/Route B/direct-C4/lifetime-cache comparison;
- verifier-on results;
- graph replay/eager table;
- prefix metrics and eviction metrics;
- component row dirty/clean counters;
- any small bug fixes or tests added;
- final decision: promote, keep experimental, split fix target, or reject.

## Decision Rules

Promote to preferred Route B opt-in bundle if:

- all verifier/text/cache-integrity checks pass;
- graph replay remains fully covered for selected buckets;
- eviction pressure does not produce stale rows or component lifecycle errors;
- `serving_mixed_112req_wave16` remains close to TARGET 08.27 performance;
- broader prefix workloads do not show repeated unexplained regressions;
- the code remains a small SGLang-aligned lifetime cache, not a new subsystem.

Keep experimental if:

- correctness passes but one workload needs more repeats;
- performance is positive but not stable enough to promote;
- an additional small targeted lifecycle test is needed.

Split a fix target if:

- a stale-row mismatch has a clear small root cause;
- table-slot reuse or eviction invalidation is incomplete but local to the
  lifetime cache.

Reject the opt-in if:

- verifier catches stale component rows under ordinary serving workloads;
- graph replay becomes unstable;
- fixing correctness requires broad scheduler/KV-cache rewrites;
- performance regresses toward the old Route B direct-C4 baseline.

## Stop Rules

Stop and report instead of optimizing if:

- failures point to SWA ownership, low precision, attention, MoE, or NCCL;
- captured-address/reference-assignment work becomes necessary;
- repeated throughput runs disagree enough that no promotion decision is
  credible;
- the target drifts from promotion/robustness validation into new mechanism
  design.
