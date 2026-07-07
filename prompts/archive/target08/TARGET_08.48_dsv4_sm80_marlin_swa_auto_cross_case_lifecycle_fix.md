# TARGET 08.48: DSV4 SM80 Marlin+SWA Auto-Capacity Cross-Case Lifecycle Fix

## Status

Queued TARGET 08 correctness target after TARGET 08.43 promotion soak blocked.

TARGET 08.47 fixed the request-level SWA owner-boundary contract and passed
fixed128 full-model gates.  TARGET 08.43 then found a new blocker only in the
same-Engine, auto-capacity, Marlin release + SWA independent multi-case
sequence:

```text
historical_4096_128_bs4 -> historical_4096_1024_bs4
--num-pages 0
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent
CUDA_LAUNCH_BLOCKING=1
MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG=1
```

The first case passes, then the next case fails with CUDA illegal memory
access.  The same `historical_4096_1024_bs4` case passes when run alone in a
fresh process.  This target must treat the bug as a case-boundary / serving-run
lifecycle contract issue until proven otherwise.

## Goal

Make Marlin WNA16 release + SWA independent lifecycle safe across consecutive
benchmark cases in the same Engine/process while preserving:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
```

The desired end state:

```text
same Engine sequence
-> active request cleanup complete
-> retained prefix owners valid
-> SWA refcount/free-list/full-to-SWA mappings valid
-> graph replay metadata rebuilt or version-checked
-> page-table static-width reads bounded by current logical lengths
-> Marlin released raw expert addresses not read
-> auto-capacity two-case gate passes
-> TARGET 08.43 promotion soak can be rerun
```

## Required Inputs

Read first:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.43_dsv4_sm80_swa_independent_post_fix_promotion_soak.md
prompts/TARGET_08.47_dsv4_sm80_swa_contract_unified_fix.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/README.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/correctness_graph_soak.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/swa_tail_runtime_counters.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/promotion_decision.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/raw/README.md
performance_milestones/target08_swa_contract_unified_fix/README.md
performance_milestones/target08_swa_contract_unified_fix/contract_fixes.md
performance_milestones/target08_swa_contract_unified_fix/metadata_graph_fix_summary.md
```

Also review the SGLang SWA implementation before making broad lifecycle
changes:

```text
/workspace/sglang-main/python/sglang/srt/mem_cache/common.py
/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
```

Important SGLang concepts to compare:

- `free_swa_out_of_window_slots`;
- `req.cache_protected_len`;
- `req.swa_evicted_seqlen`;
- one-page SWA eviction margin;
- `SWAComponent` match/insert-end window-bounded LRU refresh;
- `SWAComponent.update_component_on_insert_overlap`;
- `SWAComponent.should_skip_leaf_creation`;
- `SWAComponent.prepare_for_caching_req`;
- `SWAComponent.free_out_of_window_slots`;
- allocator `clear`, `free`, `free_swa`, and full-to-SWA mapping semantics.

## Non-Goals

- Do not promote SWA independent lifecycle in this target.
- Do not implement FP8 KV cache, INT8 MoE, or quantized communication.
- Do not hide the failure by running one scenario per process as the final
  answer.  Fresh-process runs are controls only.
- Do not mark the target fixed by skipping `prefix_metrics_snapshot()` or
  moving SWA counters to CPU.  `runtime_swa_counters()` is an observation point,
  not the proven producer.
- Do not disable CUDA graph, Marlin release, or SWA independent lifecycle as
  the final fix.  These are attribution toggles only.
- Do not weaken double-free, dummy-page, component-clear, or SWA liveness
  guards.

## Required Work

### 1. Reproduce And Classify The Failure

Reproduce the current blocker:

```bash
CUDA_LAUNCH_BLOCKING=1 \
MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent \
  --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 \
  --page-size 256 \
  --num-pages 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --repeats 1 \
  --warmup-repeats 0 \
  --output-dir /tmp/dsv4_0848_repro_cross_case \
  --keep-going
```

Confirm negative controls:

- `historical_4096_1024_bs4` alone in a fresh process passes;
- the two-case sequence under fixed128 or cap4096, if runtime permits;
- the two-case sequence with graph disabled;
- the two-case sequence without Marlin release;
- the two-case sequence without SWA independent lifecycle.

Use controls to identify whether the trigger is:

- same-Engine case boundary;
- graph replay/capture state;
- auto-capacity memory pressure;
- Marlin release address reuse;
- SWA independent lifecycle state;
- prefix cache retained owner state;
- page-table/static-width stale rows.

### 2. Add Case-Boundary Instrumentation

Add opt-in debug instrumentation rather than broad permanent overhead.
Suggested env:

```text
MINISGL_DSV4_CASE_BOUNDARY_DEBUG=1
```

The debug mode should be able to snapshot and validate, at minimum:

- before each case;
- after warmup;
- after each repeat;
- after final request cleanup;
- before `prefix_metrics_snapshot()`;
- after `prefix_metrics_snapshot()`.

Snapshot/validation content:

- active request count, decode manager state, prefill manager pending state;
- table-manager free slots and active table rows;
- prefix cache retained/protected/evictable pages;
- per-node or aggregate DSV4 SWA handle liveness:
  non-negative pages in range, refcount-positive, not free-list, not dummy;
- `_full_to_swa_page` entries:
  no stale mapping to free-list/zero-refcount/dummy/out-of-range pages;
- SWA ownership version before/after case and before graph replay;
- graph replay capture status and source metadata version;
- whether direct SWA graph metadata is disabled;
- Marlin release cache integrity and whether raw expert weights remain
  inaccessible by fallback paths.

If a debug sync finds CUDA illegal memory access, report the earliest stage.
Do not stop at the later `runtime_swa_counters()` observation unless every
earlier stage is clean.

### 3. Prove Or Disprove Candidate Producers

Work through candidates in order.  Keep evidence in the milestone.

#### A. Case Cleanup / Active Request Lifetime

Verify that after the first case:

- all requests finished or aborted;
- `cache_req(finished=True)` ran exactly once per finished request;
- all table slots were returned to the table manager;
- active-only full/SWA/component owners were transferred or released;
- `Req.swa_evicted_seqlen` is no longer needed after request cleanup and is
  not reused by the next request object.

#### B. Prefix Retained Owner State

Verify retained radix prefix state after the first case:

- retained full/component owners are valid;
- retained `DSV4SWAPageHandles` are owning values, not stale snapshots;
- tombstoned SWA handles do not contain hidden physical owners;
- prefix SWA-only release and full prefix eviction bump SWA ownership version;
- prefix match in the second case cannot expose stale SWA pages inside the
  active window.

#### C. Page Table / Static Width Metadata

Verify that second-case metadata builders never read old first-case table
columns as active inputs:

- active logical lengths bound every real row;
- stale page-table columns beyond active lengths are padding only;
- graph capture/replay buffers for padded rows use dummy locs only for dummy
  rows;
- static-width C4/C128/indexer/SWA metadata paths have current lengths.

#### D. CUDA Graph Replay State

Verify graph state across cases:

- replay source metadata is rebuilt for the current batch;
- SWA ownership version guard covers cross-case owner changes;
- direct C4 graph metadata cannot carry stale locs across cases;
- direct SWA graph metadata remains disabled in independent lifecycle;
- graph capture buffers are not directly aliased to mutable request-local
  tensors that are freed or reused between cases.

If graph-disabled control passes and graph-enabled fails, fix graph metadata
lifetime rather than disabling graph as final behavior.

#### E. Marlin Release / Auto-Capacity Arena

Verify that Marlin WNA16 raw expert release remains safe across cases:

- raw routed expert weights were released before KV allocation only if Marlin
  packed caches are complete;
- fallback/grouped-FP4 paths cannot read released raw weights;
- integrity probes do not read released raw addresses;
- component-slot clear remains enabled on page allocation;
- auto-capacity KV/SWA/component arena does not reuse released raw expert
  addresses in a way that any later model path reads.

Use existing debug envs where useful:

```text
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG=1
MINISGL_DSV4_MARLIN_WNA16_GUARD_INTEGRITY_DEBUG=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_THEN_FREE=1
```

Quarantine/poison modes are attribution tools only; the final fix should
preserve the capacity benefit unless evidence proves a small guard/reserve is
required.

### 4. Update Contract If Needed

Before implementing a broad fix, update:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
```

if the SGLang comparison or the new evidence changes any of these:

- case-boundary cleanup requirements;
- prefix retained SWA owner validity;
- graph replay metadata versioning;
- page-table static-width read contract;
- Marlin release compatibility requirements.

Do not make a code change that knowingly violates the contract.

### 5. Implement The Smallest Contract-Compliant Fix

Preferred fix areas:

- case-boundary cleanup / validation hooks in scheduler or benchmark harness;
- prefix retained SWA owner validation and repair at owner boundaries;
- graph metadata rebuild/version guard extension;
- page-table static-width active-length guards;
- Marlin release integrity guard if and only if evidence proves old-address
  reads.

Avoid:

- per-case process restart as final fix;
- broad attention kernel rewrites;
- disabling prefix cache retention;
- globally clearing prefix cache between cases unless a source-compatible
  serving reset API is added and explicitly selected as a benchmark-only
  option;
- turning CUDA errors into warnings.

## Required Test Gates

Run focused tests first:

```bash
python -m pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/engine/test_marlin_wna16_release_credit.py
```

Add or update no-weight tests for any new contract rule:

- case-boundary active request cleanup;
- retained prefix SWA liveness after finished requests;
- stale page-table/static-width metadata rejection;
- graph replay stale metadata across case boundary;
- Marlin release old-address/fallback guard if relevant.

Then run the required two-case gate:

```bash
CUDA_LAUNCH_BLOCKING=1 \
MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent \
  --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 \
  --page-size 256 \
  --num-pages 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --repeats 1 \
  --warmup-repeats 0 \
  --output-dir /tmp/dsv4_0848_fixed_cross_case \
  --keep-going
```

After that passes, run a narrow replay of the 08.43 blocker path:

- auto Marlin release + SWA independent;
- fixed128 if touched;
- cap4096 if touched;
- at least one serving/prefix/eviction smoke if runtime allows.

Do not rerun the full 08.43 promotion soak in this target unless the fix is
small and the required gates finish early.  Otherwise write a clear rerun
recommendation.

## Deliverables

Write results under:

```text
performance_milestones/target08_marlin_swa_auto_cross_case_lifecycle_fix/
```

Required files:

- `README.md` with final verdict;
- `repro_matrix.md`;
- `sglang_case_boundary_parity.md`;
- `case_boundary_contract_update.md`;
- `instrumentation.md`;
- `root_cause.md`;
- `fix_summary.md`;
- `focused_tests.md`;
- `two_case_gate.md`;
- `rerun_0843_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. Does the original two-case auto Marlin+SWA sequence reproduce?
2. Which control isolates the trigger: graph, auto capacity, Marlin release,
   SWA independent lifecycle, prefix retention, or page-table/static-width
   state?
3. What was the earliest failing stage before the later
   `runtime_swa_counters()` observation point?
4. What contract rule was missing or violated?
5. What was changed, and why is it contract-compliant?
6. Does the two-case gate pass with graph replay and zero avoidable eager
   fallback?
7. Can TARGET 08.43 promotion soak be rerun next?

## Stop Conditions

Stop and report rather than expanding scope if:

- the failure cannot be reproduced with the documented two-case sequence;
- every attribution control passes but the original still fails without an
  earlier failing stage;
- a proposed fix requires weakening double-free, liveness, dummy-page, or
  Marlin release guards;
- the only passing path requires disabling SWA independent lifecycle, Marlin
  release, prefix cache, or CUDA graph;
- text corruption or NCCL watchdog remains after the candidate fix;
- SGLang parity reveals a different owner model that requires a larger design
  update than this target can safely implement.
