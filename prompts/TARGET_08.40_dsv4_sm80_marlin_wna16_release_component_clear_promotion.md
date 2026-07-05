# TARGET 08.40: DSV4 SM80 Marlin WNA16 Release Component-Clear Promotion

## Status

Active TARGET 08 release-route promotion target after TARGET 08.39.

TARGET 08.39 effectively solved the correctness mystery: unguarded early
Marlin WNA16 raw-expert release becomes safe when newly allocated DSV4
component KV slots are initialized on page allocation.  The failure is most
consistent with an uninitialized component-cache read after the CUDA allocator
reuses old raw expert-weight storage for C4/C128/indexer component owners.

This target should turn that fix from a successful diagnostic into a production
release preset and regression-protected default behavior for the Marlin WNA16
release path.

## Goal

Root out the raw-expert release correctness bug and promote a clean high-memory
capacity path:

```text
Marlin WNA16 prebuild
-> release original routed expert weights before KV allocation
-> apply release capacity credit
-> initialize newly allocated DSV4 component KV slots
-> allow KV/component tensors to reuse old raw expert ranges safely
-> pass correctness, graph replay, macro, prefix, and serving gates
```

The desired final state is **not** a guard/quarantine workaround.  The desired
final state is an unguarded release path where KV/component allocations can use
the recovered `~17.13 GiB/rank` raw-expert memory without text corruption.

## Starting Evidence

Read these reports first:

```text
performance_milestones/target08_marlin_wna16_old_address_root_cause/README.md
performance_milestones/target08_marlin_wna16_old_address_root_cause/trap_mode_results.md
performance_milestones/target08_marlin_wna16_old_address_root_cause/stage_layer_bisection.md
performance_milestones/target08_marlin_wna16_old_address_root_cause/capacity_ledger.md
performance_milestones/target08_marlin_wna16_old_address_root_cause/fix_summary.md
performance_milestones/target08_marlin_wna16_safe_release_arena_capacity/README.md
```

Key TARGET 08.39 conclusions:

- unsafe unguarded release with no clear still produces BOS-token flood;
- `clear=component` passes;
- `clear=full` fails, so full/SWA KV alone is not the root owner;
- `clear=state` fails, so compress-state score buffers alone are not enough;
- `CUDA_LAUNCH_BLOCKING=1 + clear=none` still fails, reducing the likelihood of
  a stream-lifetime race;
- fixed eager release passes;
- fixed graph release passes with zero eager decode replay;
- fixed auto-capacity release passes and plans `2,779` pages at page size
  `256`;
- guard bytes are `0` in the fixed path, and owner ledgers still prove
  KV/component buffers reuse old raw expert ranges.

The intended fix currently lives around:

```text
python/minisgl/kvcache/deepseek_v4_pool.py
benchmark/offline/deepseek_v4_text_smoke.py
benchmark/offline/deepseek_v4_perf_matrix.py
tests/core/test_deepseek_v4_kvcache.py
```

## Non-Goals

- Do not re-open broad root-cause exploration unless a promotion gate fails.
- Do not promote the TARGET 08.38 safe-arena guard as the main solution.
- Do not blanket-zero the full KV arena unless a measured gate proves component
  clear is insufficient.
- Do not add low-precision changes, INT8 MoE, or FP8 KV cache here.
- Do not hide correctness failure by reducing page count, disabling graph, or
  falling back to a non-Marlin expert backend.

## Required Work

### 1. Source Review And Production Semantics

Audit the 08.39 changes and make the production semantics explicit.

Required checks:

- `MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC` defaults to `component` when
  `MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1`;
- the default remains disabled when raw-expert release is not enabled;
- `component` clears all newly allocated C4/C128/indexer component slots that
  can be read before overwrite:
  - `kvcache.dsv4.c4_buffer`;
  - `kvcache.dsv4.c128_buffer`;
  - `kvcache.dsv4.c4_indexer_buffer`;
  - `kvcache.dsv4.c4_indexer_fp8_paged_cache`, if present;
  - auxiliary indexer FP8 value/scale buffers, if present;
- both non-Route-B and component-ownership allocation paths are covered;
- prefix-cache hits, misses, eviction, and newly allocated pages call the same
  page-allocation clear hook;
- release capacity credit is applied only for the intended Marlin WNA16 release
  path and is visible in the capacity report;
- `safe_arena` remains a diagnostic variant and is not used as the promoted
  release path.

Decide whether explicit `MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=none`
should remain a debug-only escape hatch.  If it remains allowed, the report
must state that this is unsafe with raw-expert release and should not be used
in production.  If practical, require a debug env such as
`MINISGL_DSV4_ALLOW_UNSAFE_RELEASE_NO_CLEAR=1` before accepting that unsafe
combination.

### 2. Regression Tests

Add or tighten tests so this bug cannot silently return.

Required test coverage:

- unit test that allocated-page component clear resets live component buffers
  and does not clear unrelated component slots;
- unit test that release-enabled default clear mode is `component`;
- unit test that release-disabled default clear mode remains empty;
- unit test or benchmark-config test that
  `dsv4_sm80_a100_victory_marlin_release` expands to:
  - `MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1`;
  - `MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1`;
  - `MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc`;
  - `MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1`;
  - `MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component`;
- same config test for the prefix Route B release variant;
- test that diagnostic envs are not accidentally preserved across unrelated
  variants unless explicitly requested;
- if feasible, a small CPU/GPU-local poisoned component-cache test that fails
  without component clear and passes with it.

Run at least:

```text
pytest -q tests/core/test_deepseek_v4_kvcache.py
pytest -q tests/benchmark/test_deepseek_v4_text_smoke.py tests/benchmark/test_deepseek_v4_perf_matrix.py
pytest -q tests/engine/test_marlin_wna16_release_credit.py
python -m py_compile \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/engine/graph.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_marlin_wna16_guard_census.py
```

If any listed test file does not exist or has changed name, use the local
equivalent and record that in the report.

### 3. Correctness Gates

Run TP8 text smokes at page size `256`:

- prebuild-only baseline;
- promoted unguarded release variant;
- prefix Route B lifetime release variant;
- optional unsafe `clear=none` control, expected to fail or warn;
- optional safe-arena diagnostic control, expected to pass but not promoted.

Required pass conditions:

- no BOS/token-0 flood;
- no obvious corrupted text or invalid Unicode;
- no NaN/Inf logit explosion;
- graph variant has zero eager decode replay for captured buckets;
- model prepare report shows raw expert release occurred;
- capacity report shows release credit applied.

### 4. Performance And Capacity Gates

Measure whether page-allocation component clear is acceptable.

Required macro shapes:

```text
4096 input / 128 output / batch 4 / TP8
4096 input / 1024 output / batch 4 / TP8
```

Compare at least:

- `dsv4_sm80_a100_victory`;
- `dsv4_sm80_a100_victory_marlin_prebuild`;
- `dsv4_sm80_a100_victory_marlin_release`;
- `dsv4_sm80_a100_victory_marlin_release_safe_arena` as diagnostic only.

For prefix-sensitive work, also compare:

- `dsv4_sm80_a100_victory_prefix_routeb_lifetime`;
- `dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release`;

Record:

- output token throughput;
- TTFT / TPOT / ITL if available;
- graph replay counters;
- planned pages/tokens;
- raw expert released bytes/rank;
- guard/reserved bytes/rank;
- max allocated/reserved memory;
- owner clear count and estimated bytes cleared on page allocation.

If component clear causes a visible regression, attribute it:

- page-allocation frequency;
- number of component slots cleared;
- C4/C128/indexer buffer clear cost;
- prefix cache hit/miss behavior;
- eviction/reallocation behavior.

Do not optimize the clear kernel in this target unless the measured overhead is
a blocker.  If it is a blocker, propose a focused follow-up.

### 5. Serving And Prefix Cache Gate

Because release capacity is mainly useful for larger serving capacity, include
one serving-style pass when short correctness and macro gates pass.

Use existing TARGET 08 serving scenarios if available, including:

- no-hit serving mixed workload;
- shared-prefix workload;
- prefix Route B lifetime preset;
- graph buckets `[1,2,4,8,16]` unless a test explicitly requires another set.

This gate should answer:

1. Does release+component-clear preserve no-hit throughput?
2. Does prefix shared-hit workload still benefit from radix prefix cache?
3. Does higher auto-planned page count increase usable capacity without text
   corruption?
4. Does page-allocation clear introduce overhead during bursty allocation or
   eviction?

### 6. Cleanup And Default/Preset Decision

After validation, decide one of:

1. **Promote release preset**
   - `dsv4_sm80_a100_victory_marlin_release` becomes the preferred high-memory
     DSV4 sm80 release preset;
   - prefix release variant is also valid if prefix gates pass;
   - safe-arena remains diagnostic.
2. **Keep release opt-in**
   - if performance overhead or serving risk remains;
   - document exact blocker and next target.
3. **Reject release for now**
   - only if correctness fails after the component-clear fix.

Clean up or clearly label debug-only tooling:

- poison-then-free;
- KV-as-sentinel;
- guard census;
- layer-filter release;
- safe-arena guard.

Do not remove useful diagnostic tools if they are still needed, but keep them
out of default presets and document their envs as debug-only.

## Deliverables

Write results under:

```text
performance_milestones/target08_marlin_wna16_release_component_clear_promotion/
```

Required files:

- `README.md` with verdict and promotion decision;
- `source_semantics_review.md`;
- `correctness_gate.md`;
- `capacity_ledger.md`;
- `macro_performance.md`;
- `serving_prefix_gate.md` if serving gate is run;
- `page_alloc_clear_overhead.md`;
- `cleanup_and_defaults.md`;
- raw logs / JSON under `raw/`.

The README must answer:

1. Is the raw-expert release bug considered fixed, not merely guarded?
2. Does the promoted path use `0` guard/reserved bytes?
3. Does KV/component reuse old raw expert ranges safely?
4. How many pages/tokens are planned at page size `256`?
5. What is the performance cost of component clear?
6. Is the release path promoted, kept opt-in, or blocked?

## Stop Conditions

Stop once one of these is true:

- release+component-clear passes correctness, macro, prefix/serving gates, and
  a promotion decision is recorded;
- a correctness regression appears and is attributed to a concrete owner;
- component-clear overhead is a clear blocker and a focused optimization target
  is proposed;
- after two failed attempts to stabilize the fix, no new evidence appears.  In
  that case, keep the safe release path opt-in and write the blocker clearly.
