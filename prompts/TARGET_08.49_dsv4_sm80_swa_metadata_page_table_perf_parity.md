# TARGET 08.49: DSV4 SM80 SWA Metadata/Page-Table Performance Parity

## Status

Active TARGET 08 follow-up after TARGET 08.43 post-fix promotion soak.

TARGET 08.43 rerun showed that the SWA independent lifecycle correctness
blockers are fixed after TARGET 08.48:

- focused regression tests passed;
- fixed128, explicit cap4096, and auto-capacity paths passed;
- Marlin WNA16 release + SWA independent passed;
- the previously blocking same-Engine auto-capacity sequence
  `historical_4096_128_bs4 -> historical_4096_1024_bs4` passed with healthy
  graph replay and eager decode `0`;
- no CUDA illegal memory access, corrupted text, NCCL watchdog, or graph
  fallback was observed.

The remaining blocker is performance.  SWA independent remains opt-in because
fixed-capacity serving/prefix E2E throughput regresses by about `16-18%`.
Owner timing points primarily to decode attention metadata / SWA page-table
construction rather than SWA release/free bookkeeping.

This target should align mini's behavior with SGLang where practical, then land
the smallest performance fix that reduces the metadata/page-table overhead
without reopening lifecycle correctness.

## Goal

Reduce SWA independent decode metadata/page-table overhead enough that SWA
independent lifecycle can be reconsidered for promotion as a high-capacity
prefix/serving preset.

The desired path is:

```text
SGLang source behavior census
-> mini source/runtime census
-> exact cost split inside prepare.decode.attention_metadata
-> parity design for stable-row / dirty-row / direct graph metadata behavior
-> one bounded opt-in implementation
-> correctness soak
-> clean performance rerun
-> promote / keep opt-in / write next focused target
```

Do not skip the SGLang comparison.  If SGLang already has a mature mechanism
for this class of metadata lifetime or SWA page-table reuse, prefer adapting
that design over inventing a mini-only mechanism.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_independent_post_fix_promotion_soak/README.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/e2e_overhead_attribution.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/macro_serving_performance.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/serving_capacity_ledger.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/swa_tail_runtime_counters.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/promotion_decision.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/target09_5_recommendation.md
performance_milestones/target08_marlin_swa_auto_cross_case_lifecycle_fix/README.md
performance_milestones/target08_marlin_swa_auto_cross_case_lifecycle_fix/root_cause.md
performance_milestones/target08_marlin_swa_auto_cross_case_lifecycle_fix/fix_summary.md
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.43_dsv4_sm80_swa_independent_post_fix_promotion_soak.md
prompts/TARGET_08.48_dsv4_sm80_marlin_swa_auto_cross_case_lifecycle_fix.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Important TARGET 08.43 rerun facts:

- fixed128 `serving_mixed_112req_wave16` E2E delta:
  SWA independent `-15.9%`;
- fixed128 `prefix_multi_112req_wave16` E2E delta:
  SWA independent `-18.1%`;
- fixed128 `historical_4096_1024_bs4` E2E delta:
  SWA independent `-12.0%`;
- fixed128 serving owner timing:
  `dsv4.prepare.decode.attention_metadata`
  `1217.49 ms -> 2968.23 ms`, delta `+1750.74 ms`;
- SWA release/free bookkeeping is much smaller:
  release full pages about `84.65 ms`, release handles about `55.55 ms`,
  allocate pages about `3.69 ms`;
- capacity gain is strong:
  auto Marlin release + SWA independent planned `6490` pages / `1661440`
  tokens at the same about `49.96 GiB/rank` KV budget where Marlin release
  alone planned `2777` pages / `710912` tokens.

## Reference Source Paths

Mini:

```text
python/minisgl/scheduler/scheduler.py
python/minisgl/scheduler/cache.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
tests/core/test_deepseek_v4_kvcache.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
```

SGLang source tree:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py
/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py
```

If the local SGLang tree changes or is missing, clearly mark which conclusions
are source-derived from the available tree and which are not verified at
runtime.

## Non-Goals

- Do not redesign the SWA independent lifecycle contract.
- Do not re-debug the fixed 08.42 dummy-token issue, 08.47 ownership issue, or
  08.48 fused SWA-store address-space issue unless an exact regression appears.
- Do not implement FP8 KV/cache or INT8 MoE.
- Do not rewrite C4/C128/SWA attention kernels unless metadata evidence proves
  a tiny kernel-side interface change is required.
- Do not promote SWA independent by default unless performance and correctness
  gates both pass.
- Do not hide overhead by disabling prefix cache, disabling graph capture,
  reducing graph bucket coverage, or switching to eager.

## Required Work

### 1. SGLang Metadata Behavior Census

Read the SGLang source paths above and write a concise census answering:

- How does SGLang represent SWA indices under radix/unified cache?
- Are SWA indices stored already translated to SWA-pool locations, or derived
  from full-token locations at hit time?
- Which metadata/page-table buffers are rebuilt every decode step, and which
  are kept stable across request lifetime, radix hit lifetime, or graph replay?
- Does SGLang use dirty-row, request-slot, ownership-version, UUID, or similar
  invalidation boundaries for SWA/prefix metadata?
- Does SGLang directly generate graph-consumed metadata buffers, or does it use
  staging buffers and copies?
- Are debug/liveness checks in the hot decode path, or only in validation/debug
  paths?
- Are there helper kernels such as metadata kernels, paged metadata builders,
  or compact index builders that mini can adapt?

The output should include a mini-vs-SGLang table with exact file/function
references and a recommended adaptation strategy.

### 2. Mini Cost Split

Refine owner timing inside
`dsv4.prepare.decode.attention_metadata` before changing behavior.

Split at least these owners when SWA independent is enabled:

- active request full-to-SWA translation;
- prefix/radix SWA handle merge;
- SWA page-table row construction;
- C4/C128/component page-table work not caused by SWA;
- graph metadata staging/copy;
- ownership-version/liveness checks;
- any CPU tensor allocation, clone, `torch.cat`, `torch.unique`, `torch.isin`,
  or host-to-device copy in the hot path.

Use opt-in timing labels so the clean production path is not slowed down.  Run
the split on at least:

```text
fixed128 baseline
fixed128 SWA independent
fixed128 Marlin release + SWA independent
auto Marlin release + SWA independent
```

Primary scenario:

```text
serving_mixed_112req_wave16
```

Also include:

```text
prefix_multi_112req_wave16
historical_4096_1024_bs4
```

### 3. Parity Design

Based on SGLang and mini evidence, choose one bounded implementation lane.

Candidate lanes:

1. **Stable-row / dirty-row cache**
   - Cache SWA page-table rows for request slots when the relevant
     `swa_ownership_version`, prefix handle identity, protected frontier, and
     active decode window have not changed.
   - Update only dirty rows.
   - This should mirror SGLang-style lifetime boundaries if such a mechanism
     exists in the source census.

2. **Pretranslated SWA component handles**
   - Store or preserve prefix SWA handles as compact SWA physical locations
     rather than repeatedly translating from full-token locations in decode
     metadata.
   - Preserve the TARGET 08.48 invariant that all physical SWA cache writes
     receive SWA locations, not full-token locations.

3. **Direct graph-consumed SWA metadata buffers**
   - Revisit direct SWA graph metadata only if the source census and current
     ownership-version guard prove it can be made stale-safe.
   - The default should stay conservative: direct C4 buffers may remain enabled,
     direct SWA buffers remain disabled unless this target proves parity.

4. **Persistent workspace / no-allocation metadata builder**
   - Preallocate metadata scratch/page-table buffers and rewrite the hot path
     to fill them in place.
   - Remove hot-path allocation, clone, concat, and avoidable D2D/H2D copies.

5. **Metadata helper kernel**
   - If the cost split shows a small tensor transformation dominates and SGLang
     has an equivalent helper kernel, adapt it or write a minimal mini-owned
     version.
   - Use a microbench and a roofline-style byte/element estimate before doing
     a larger kernel port.

Do not implement all lanes.  Pick the highest-confidence lane, explain why,
and keep it easy to disable with an opt-in flag or temporary benchmark variant.

### 4. Implementation

Implementation requirements:

- Preserve the existing SWA independent lifecycle contract.
- Keep fallback behavior fail-closed when metadata becomes stale.
- Do not weaken liveness checks; keep them available under debug env flags.
- Avoid Python-side CUDA synchronization in the production path.
- Avoid new per-step `cudaMalloc` or unbounded tensor allocation.
- Keep graph replay valid for buckets `[1,2,4,8,16]`.
- Add focused tests for dirty-row invalidation or metadata-cache invalidation
  if a cache is introduced.
- Add benchmark variant/env coverage only as narrowly as needed for the new
  opt-in.

Recommended temporary names if an opt-in is needed:

```text
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_metacache
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_metacache
```

Use different names if the chosen design is not a cache.

### 5. Correctness Gates

Run focused tests first.  At minimum include the suites used by the 08.43 soak:

```text
tests/core/test_deepseek_v4_kvcache.py
tests/core/test_dsv4_cache_option_guards.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/benchmark/test_deepseek_v4_text_smoke.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
tests/engine/test_marlin_wna16_release_credit.py
tests/models/test_deepseek_v4_forward_fallback.py
```

Then run TP8, page size `256`, CUDA graph buckets `[1,2,4,8,16]`.

Required correctness gates:

- fixed128 SWA independent text smoke;
- fixed128 Marlin release + SWA independent text smoke;
- explicit cap4096 Marlin release + SWA independent text smoke;
- auto-capacity Marlin release + SWA independent text smoke;
- same-Engine auto-capacity gate:
  `historical_4096_128_bs4 -> historical_4096_1024_bs4`;
- macro rows for:
  `historical_4096_128_bs4`,
  `historical_4096_1024_bs4`,
  `serving_mixed_112req_wave16`,
  `prefix_multi_112req_wave16`,
  `prefix_eviction_pressure_96req_wave16`.

All pass rows must have:

- no corrupted text;
- no CUDA illegal memory access;
- no NCCL watchdog;
- graph replay healthy;
- eager decode `0` for captured buckets;
- no stale SWA metadata, negative SWA refcount, double free, or leaked active
  mapping.

### 6. Performance Gates

Use clean production-like runs for final performance.  Do not include
`CUDA_LAUNCH_BLOCKING=1`, SWA bounds debug, case-boundary debug, or owner timing
debug in the final throughput comparison.

Primary fixed128 targets:

| Scenario | 08.43 SWA overhead | Desired next result |
| --- | ---: | ---: |
| `serving_mixed_112req_wave16` | `-15.9%` | reduce to about `-6%` or better |
| `prefix_multi_112req_wave16` | `-18.1%` | reduce to about `-8%` or better |
| `historical_4096_1024_bs4` | `-12.0%` | reduce to about `-5%` or better |

Owner-timing target:

```text
dsv4.prepare.decode.attention_metadata delta
from about +1.75s on serving_mixed_112req_wave16
to about +0.6s or less
```

These are promotion-oriented goals, not excuses to overfit.  If the first
bounded fix achieves a clear partial gain but not the full target, stop and
write the next focused target rather than stacking risky changes.

Capacity must not regress materially:

- auto Marlin release + SWA independent should remain near the 08.43 result:
  about `6490` pages / `1661440` tokens at about `49.96 GiB/rank`;
- explicit cap4096 Marlin release + SWA independent must remain stable;
- fixed128 planning remains fixed128.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_metadata_page_table_perf_parity/
```

Required files:

- `README.md` with final verdict;
- `sglang_source_census.md`;
- `mini_cost_split.md`;
- `parity_design.md`;
- `implementation_summary.md`;
- `correctness_graph_soak.md`;
- `macro_performance.md`;
- `owner_timing_after.md`;
- `capacity_regression_check.md`;
- `promotion_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. What exactly does SGLang do differently for DSV4 SWA metadata/page-table
   lifetime or graph metadata?
2. Which mini sub-owner explains the `prepare.decode.attention_metadata`
   overhead?
3. Which design lane was implemented, and why was it chosen over the others?
4. Did correctness remain as clean as TARGET 08.43?
5. Did graph replay remain zero-eager for buckets `[1,2,4,8,16]`?
6. How much did serving/prefix/historical E2E overhead improve?
7. Did the capacity gain remain intact?
8. Should SWA independent remain opt-in, be promoted as a high-capacity preset,
   or get one more focused performance target?

## Stop Conditions

Stop and report instead of continuing to patch if:

- the SGLang census shows mini's architecture differs enough that direct parity
  needs a separate design target;
- refined timing shows the main owner is not metadata/page-table construction;
- a correctness failure appears in same-Engine auto, prefix eviction, or graph
  replay;
- graph replay falls back to eager for captured buckets;
- capacity planning loses most of the SWA independent gain;
- the bounded implementation does not move the main owner after one clean
  attempt;
- the next fix would require broad lifecycle redesign, FP8/INT8 work, or
  attention-kernel rewrite.

