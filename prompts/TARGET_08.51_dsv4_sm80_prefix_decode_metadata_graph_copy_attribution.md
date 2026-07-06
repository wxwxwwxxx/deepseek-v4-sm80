# TARGET 08.51: DSV4 SM80 Prefix Decode Metadata / Graph-Copy Attribution

## Status

Active TARGET 08 follow-up after TARGET 08.50.

TARGET 08.49 and TARGET 08.50 substantially narrowed the SWA-specific metadata
problem:

- TARGET 08.49 added an opt-in independent-SWA page-table row cache:
  `MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1`.
- TARGET 08.50 added an opt-in SGLang-style direct token metadata path:
  `MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1`.
- 08.50 confirmed decode no longer materializes the full SWA page table when
  direct mode is enabled.
- Correctness, graph replay, capacity, Marlin release compatibility, and the
  same-Engine auto gate remained clean.

However, default promotion is still not justified:

- fixed128 `serving_mixed_112req_wave16`: direct SWA is still `-6.01%` vs
  non-SWA;
- fixed128 `prefix_multi_112req_wave16`: direct SWA is still `-9.39%` vs
  non-SWA;
- fixed128 `historical_4096_1024_bs4`: direct SWA is still `-5.49%` vs
  non-SWA;
- direct mode improved over the 08.49 cache by only `+0.28%` on serving and
  `+1.06%` on prefix_multi.

The next target must not keep polishing the old SWA page-table path.  The
remaining gap appears to belong to the wider decode metadata / graph-copy /
prefix-scheduler surface.

## Goal

Identify the dominant remaining owner of the `6-10%` SWA independent E2E gap
after TARGET 08.50, then recommend the next implementation target.

This is primarily an attribution reset:

```text
08.49/08.50 SWA-specific fixes are done enough
-> split remaining decode metadata and graph-copy owners
-> compare non-SWA / old SWA / 08.49 cache / 08.50 direct
-> include prefix scheduler/cache/free owners
-> identify one dominant next owner
-> write the next focused implementation plan
```

Implementing a small instrumentation-only patch is in scope.  Implementing a
tiny low-risk fix is allowed only if the dominant owner is already obvious and
the fix does not weaken correctness gates.  Broad optimization work belongs in
the follow-up target.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_direct_token_metadata_parity/README.md
performance_milestones/target08_swa_direct_token_metadata_parity/owner_timing_after.md
performance_milestones/target08_swa_direct_token_metadata_parity/macro_performance.md
performance_milestones/target08_swa_direct_token_metadata_parity/promotion_recommendation.md
performance_milestones/target08_swa_direct_token_metadata_parity/source_shape_reconfirmation.md
performance_milestones/target08_swa_direct_token_metadata_parity/direct_token_design.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/README.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/owner_timing_after.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/README.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/e2e_overhead_attribution.md
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.50_dsv4_sm80_swa_direct_token_metadata_parity.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Key TARGET 08.50 owner data:

```text
dsv4.prepare.decode.attention_metadata
post-fix old SWA:              2968.232 ms
TARGET 08.49 page-table cache: 1381.333 ms
TARGET 08.50 direct metadata:  1308.225 ms
non-SWA reference:             1217.490 ms
```

Top host owners in the 08.50 fixed128 serving run:

```text
dsv4.prepare.decode.attention_metadata      1308.225 ms
dsv4.scheduler.prefix.cache_req             1114.383 ms
dsv4.prepare.prefill.attention_metadata      761.558 ms
dsv4.kvcache.pages.on_freed                  452.577 ms
dsv4.kvcache.swa.release_full_pages          105.053 ms
dsv4.kvcache.swa.release_handles              76.423 ms
dsv4.scheduler.page.allocate_paged            41.485 ms
dsv4.prepare.decode.positions                 31.637 ms
dsv4.prepare.decode.input_tuple               29.226 ms
dsv4.metadata.build.table_indices             29.076 ms
```

Important caution:

- TARGET 08.50 CUDA-event owner timing with graph replay segfaulted.  Do not
  rely on CUDA-event owner timing inside replay unless the instrumentation is
  first proven safe.  Prefer host owner timing, NVTX/nsys outside fragile graph
  event code, or synthetic/micro attribution.

## Reference Source Paths

Mini:

```text
python/minisgl/scheduler/scheduler.py
python/minisgl/scheduler/cache.py
python/minisgl/kvcache/radix_cache.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/graph.py
benchmark/offline/deepseek_v4_perf_matrix.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/core/test_deepseek_v4_kvcache.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
```

SGLang / reference behavior, use selectively:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py
```

## Non-Goals

- Do not keep optimizing 08.49 dirty-row SWA page-table cache unless fresh
  evidence shows it is dominant again.
- Do not keep optimizing 08.50 direct token SWA metadata locally unless fresh
  evidence shows it is dominant.
- Do not redesign SWA independent lifecycle correctness.
- Do not implement FP8 KV/cache, INT8 MoE, or quantized communication.
- Do not rewrite attention compute kernels.
- Do not promote SWA independent by default in this target.
- Do not hide overhead by disabling prefix cache, graph capture, graph buckets,
  Marlin release checks, or same-Engine gates.

## Required Work

### 1. Establish Comparable Baselines

Use the same fixed TP8 shape unless a source-specific probe explicitly needs a
smaller run:

```text
--tensor-parallel-size 8
--page-size 256
--num-pages 128
--allow-dsv4-cuda-graph
--cuda-graph-bs 1 2 4 8 16
--cuda-graph-capture-greedy-sample
```

Compare at least:

```text
non-SWA prefix Route B baseline
old SWA independent without 08.49/08.50
08.49 page-table cache
08.50 direct token metadata
```

Scenarios:

```text
serving_mixed_112req_wave16
prefix_multi_112req_wave16
historical_4096_1024_bs4
prefix_eviction_pressure_96req_wave16
```

Include Marlin release + SWA independent for:

```text
auto same-Engine historical_4096_128_bs4 -> historical_4096_1024_bs4
explicit cap4096 macro, if runtime budget allows
```

### 2. Decode Metadata / Graph-Copy Split

Add or refine opt-in host timing labels that split
`dsv4.prepare.decode.attention_metadata` into the smallest useful owners.

At minimum split:

- final SWA token-loc source construction;
- final SWA token-loc graph/replay input copy;
- `swa_topk_lengths` construction/copy;
- C4 sparse/index metadata source construction;
- C4 graph/replay copy;
- C128 sparse/index metadata source construction;
- C128 graph/replay copy;
- component page-table or component source generation still present outside SWA;
- `table_indices` / row-selection work;
- metadata object assembly / validation / dtype/device normalization;
- debug/liveness paths, if they are accidentally active.

The output should make clear whether the remaining `attention_metadata` gap is:

1. still SWA-specific final tensor/copy;
2. C4/C128/indexer metadata;
3. component ownership/table work;
4. generic graph metadata copy;
5. Python object/host overhead;
6. not in attention metadata anymore.

### 3. Prefix Scheduler / Cache Owner Split

`dsv4.scheduler.prefix.cache_req` and `dsv4.kvcache.pages.on_freed` are large
in TARGET 08.50.  Split them enough to know whether the prefix-heavy gap comes
from:

- radix match / insert / split / lock;
- component handle collection;
- SWA tombstone or protected-frontier update;
- page/free-list updates;
- component page-table lifetime-cache update;
- Python container/list/dict work;
- cross-rank imbalance or one slow rank;
- release/free work that can be deferred, batched, or moved out of the hot
  serving path.

Do not change lifecycle semantics in this target; attribution first.

### 4. Graph-Copy And Replay Evidence

Collect evidence for replay metadata copy volume and frequency:

- bytes copied per decode step and per scenario;
- per metadata group: SWA, C4, C128, component, sampler if relevant;
- per graph bucket `[1,2,4,8,16]`;
- difference between non-SWA, 08.49 cache, and 08.50 direct;
- whether copy bytes correlate with the E2E gap.

If safe, use NVTX ranges and nsys for one short representative run.  Avoid
fragile CUDA event owner timing in graph replay unless first fixed/proven.

Recommended short trace if used:

```text
serving_mixed_112req_wave16
fixed128
non-SWA vs 08.50 direct
one repeat, no warmup repeats unless needed
```

### 5. Microbench / Partial Probes

If the full TP8 runs do not isolate the owner, add no-weight or partial probes:

- metadata builder microbench for fixed rows/window/buckets;
- graph input-copy microbench for the same metadata tensor sizes;
- scheduler prefix/cache microbench with synthetic radix/component handles;
- page-free bookkeeping microbench under SWA independent vs non-SWA.

These probes should be used to explain the full-model result, not replace it.

### 6. Correctness Guard

This is an attribution target, but any instrumentation or small patch must keep
the known gates clean:

- focused unit tests around attention metadata, kvcache, and perf-matrix
  variants;
- fixed128 SWA independent text smoke;
- 08.50 direct token metadata text smoke if the direct path is touched;
- same-Engine auto Marlin release + SWA independent gate if runtime-affecting
  code changes are made;
- graph replay remains healthy with eager decode `0` for captured buckets.

If only instrumentation/reporting code is added and no runtime logic changes,
record the test subset used and why it is sufficient.

## Deliverables

Write results under:

```text
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/
```

Required files:

- `README.md` with final attribution verdict;
- `baseline_matrix.md`;
- `decode_metadata_split.md`;
- `prefix_scheduler_cache_split.md`;
- `graph_copy_replay_ledger.md`;
- `microbench_or_partial_probes.md`;
- `correctness_guard.md`;
- `next_target_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. What is the dominant remaining owner of the SWA independent E2E gap after
   08.50?
2. Is the remaining gap still SWA-specific, or has it moved to generic
   prefix/metadata/graph-copy/scheduler overhead?
3. How much of `dsv4.prepare.decode.attention_metadata` is final SWA token-loc
   source/copy versus C4/C128/component metadata?
4. How much of the prefix-heavy gap is scheduler/cache/free bookkeeping?
5. Does graph replay copy volume explain the measured macro deltas?
6. Are 08.49 cache and/or 08.50 direct still useful opt-ins?
7. What exact implementation target should run next?

## Stop Conditions

Stop and report instead of continuing to patch if:

- the attribution identifies one dominant owner suitable for a focused target;
- the owner clearly moved away from SWA metadata;
- instrumentation overhead makes timing invalid and a safer probe is needed;
- CUDA graph replay or correctness regresses;
- the next step would require lifecycle redesign, low precision, attention
  compute-kernel work, or broad scheduler rewrite;
- no single owner dominates and the right next step is a repeatable profile
  methodology target rather than implementation.

