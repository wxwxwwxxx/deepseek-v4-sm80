# TARGET 08.43: DSV4 SM80 SWA Independent Post-Fix Promotion Soak

## Status

Active TARGET 08 follow-up after TARGET 08.42.

TARGET 08.31 implemented opt-in, SGLang-aligned SWA independent lifecycle for
DeepSeek V4 on A100/sm80.  TARGET 08.41 then found that fixed-128 serving was
correctness-clean, but large-capacity Marlin WNA16 release + SWA independent
serving crashed with CUDA illegal memory access.  TARGET 08.42 fixed that
blocker by aligning the Engine/KV-cache dummy full-page token contract.

This target is the post-fix promotion soak.  It should not re-implement SWA
lifecycle, re-debug the old dummy-token issue, or broaden into FP8 KV/cache.
It should decide whether SWA independent lifecycle is ready to promote, should
remain opt-in because of overhead, or needs one focused follow-up.

## Goal

Verify the 08.42 fix under serving-like pressure and make a promotion decision:

```text
SWA independent lifecycle
-> dummy full-token contract remains correct
-> Marlin WNA16 release + component-slot clear remains compatible
-> fixed and auto capacity paths are correctness-clean
-> graph replay covers serving buckets
-> memory/capacity value is quantified
-> remaining E2E overhead is attributed
-> promotion / opt-in / blocker decision is recorded
```

The desired end state is one of:

1. promote SWA independent lifecycle as a named high-capacity prefix/serving
   preset;
2. keep it opt-in with a documented overhead tradeoff and next optimization
   target;
3. block promotion only if a new correctness, graph replay, or large-capacity
   stability issue appears.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_independent_lifecycle/summaries/TARGET_08.31_report.md
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/README.md
performance_milestones/target08_swa_large_capacity_serving_correctness/README.md
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
prompts/TARGET_08.41_dsv4_sm80_swa_independent_lifecycle_promotion_soak.md
prompts/TARGET_08.42_dsv4_sm80_swa_large_capacity_serving_correctness.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Important TARGET 08.31 conclusions:

- SWA KV has independent page lifecycle and can tombstone/free out-of-window
  tail pages without invalidating C4/C128/indexer/state/component locations.
- Feature gates:
  - CLI: `--enable-dsv4-swa-independent-lifecycle`
  - env: `MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1`
  - requires radix prefix cache and DSV4 component loc ownership.
- Runtime counters showed small live SWA tails:
  - historical: about `4` live SWA pages;
  - serving mixed: about `18`;
  - prefix multi: about `26`.
- Marlin release auto-capacity path showed large potential:
  - release baseline: `2776` pages / `710656` tokens;
  - release + SWA independent: `6636` pages / `1698816` tokens.

Important TARGET 08.41 conclusions:

- Fixed `--num-pages 128` SWA independent serving/prefix/eviction passed with
  graph buckets `[1,2,4,8,16]`.
- Large-capacity Marlin release + SWA independent crashed at auto capacity and
  explicit `--num-pages 4096`; this blocked promotion.
- E2E overhead was mainly attributed to decode prepare / attention metadata:
  serving wave16 elapsed `+1.844s`, decode prepare `+1.703s`, scheduler
  `+0.193s`, while decode forward improved `-0.134s`.

Important TARGET 08.42 conclusions:

- The large-capacity crash is fixed.
- Root cause: Engine planned `num_tokens = planned_pages * page_size`, but
  allocated `planned_pages + 1` KV pages for the dummy request row.  The
  Engine dummy row used token start `num_tokens`, while DSV4 KV-cache code had
  interpreted `allocated_pages * page_size` as the dummy full-token sentinel.
  The mismatch let the Engine dummy row look like a real full page, translate
  to SWA loc/page `-1`, and corrupt device memory in graph-padded serving.
- Fix: Engine passes `dsv4_dummy_token_start=num_tokens` into
  `create_kvcache_pool()`, and `DeepSeekV4KVCache` stores/uses that sentinel
  in full-to-SWA translation and SWA page derivation.
- No-weight repros verified the old invalid producer and the fixed dummy-page
  mapping across page counts.
- Full-model confirmation passed:
  - text smoke fixed 128 and auto;
  - serving macro cap4096: `4096` pages / `1048576` tokens, replay `441`,
    eager `0`;
  - serving macro auto: about `6495` pages / `1662720` tokens, replay `441`,
    eager `0`;
  - no CUDA illegal memory access or NCCL watchdog.
- Focused tests passed: `77 passed`.

## Non-Goals

- Do not implement FP8 KV/cache in this target.
- Do not implement INT8 MoE or quantized communication.
- Do not rewrite attention kernels unless a tiny correctness fix is needed.
- Do not re-open the SWA ownership design unless correctness fails.
- Do not hide a performance regression by switching to eager or reducing graph
  bucket coverage.
- Do not bypass TARGET 08.40 component-slot clear semantics.

## Required Work

### 1. Post-Fix Source Sanity

Review the current code and confirm these contracts:

- Engine and DSV4 KV cache agree on the dummy full-token sentinel.
- The dummy full token maps to the SWA dummy page, never to `-1`.
- The SWA independent lifecycle remains off by default unless a preset/CLI/env
  enables it.
- It still requires radix prefix cache and DSV4 component loc ownership.
- It does not invalidate C4/C128/indexer/compression-state/component locs when
  SWA rows are tombstoned or freed.
- TARGET 08.40 component-slot clear still runs on page allocation when Marlin
  WNA16 raw expert release is enabled.
- Direct C4 graph metadata buffers remain valid.  Direct SWA graph metadata is
  either disabled in independent mode or explicitly proven safe.
- Existing focused tests for dummy mapping, SWA page translation, component
  clear, and prefix/eviction still pass.

If any of these checks fail, fix the smallest contract bug before running the
long soak.

### 2. Correctness And Graph Soak

Run TP8 with page size `256`.

Required graph buckets:

```text
--cuda-graph-bs 1 2 4 8 16
```

Required variant families:

- promoted prefix Route B lifetime baseline;
- prefix Route B lifetime + SWA independent;
- prefix Route B lifetime + Marlin WNA16 release;
- prefix Route B lifetime + Marlin WNA16 release + SWA independent.

Required capacity modes:

- fixed capacity: `--num-pages 128`;
- explicit large capacity: `--num-pages 4096`;
- auto capacity: `--num-pages 0`.

Required gates:

- text smoke passes with sane output;
- no garbled text, token-0 flood, NaN/Inf logit explosion, crash, or NCCL
  watchdog;
- graph replay is zero-eager for captured serving buckets;
- prefix hit/remap/eviction probes do not use tombstoned SWA rows;
- dummy full-token rows always map to SWA dummy rows;
- no SWA negative refcount, double-free, stale handle, or leaked active
  mapping;
- Marlin release path still reports component-slot clear and `0` guard bytes.

If a workload uses a decode batch size outside captured buckets, either add the
bucket or label it clearly; avoidable eager fallback is not promotion-ready.

### 3. Serving And Capacity Ledger

Run fixed-capacity and auto-capacity paths separately.

Fixed-capacity path:

```text
--num-pages 128
--page-size 256
```

Large and auto-capacity paths:

```text
--num-pages 4096
--num-pages 0
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent
```

Workloads:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4` if runtime budget allows;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`;
- `prefix_eviction_pressure_96req_wave16` or closest available eviction
  pressure workload;
- one higher-concurrency serving case, such as wave64, if the harness supports
  it without excessive runtime.

Report, per workload and variant:

- planned full pages, SWA pages, and total KV GiB/rank;
- planned tokens;
- live SWA current/tail pages;
- SWA allocated/freed/tombstoned totals;
- live full/C4/C128/indexer/state pages or slots;
- prefix hit rate and saved prefill tokens;
- retained prefix pages;
- evictions;
- graph replay/eager counters;
- output tok/s, TTFT/TPOT/ITL where available;
- max allocated/reserved memory.

The capacity ledger must clearly separate:

- fixed `--num-pages 128` safe-floor planning;
- explicit `--num-pages 4096` large-capacity stability;
- auto-capacity Marlin release page/token gain;
- runtime-proven live SWA tail reduction.

### 4. E2E Overhead Attribution

Re-run the 08.41 overhead attribution after the dummy-token fix.  The prior
owner was decode prepare / attention metadata; verify whether that is still
true.

Split overhead into:

- scheduler page allocation/free;
- SWA tombstone/free bookkeeping;
- prefix hit/remap path;
- SWA page-table construction;
- attention metadata preparation;
- graph metadata copy/replay preparation;
- decode forward;
- prefill forward;
- sampling/postprocess if relevant.

Produce a table comparing baseline vs SWA independent for:

- historical `4096/128/bs4`;
- serving mixed wave16;
- prefix multi wave16;
- auto-capacity Marlin release path.

Stop optimizing if overhead is already within noise.  If one owner explains
most of the E2E regression, write a focused next target instead of polishing
many small local paths.

### 5. Promotion Decision

End with one decision:

1. **Promote**
   - SWA independent lifecycle becomes a named high-capacity prefix/serving
     preset.
   - It may remain opt-in if prefix cache itself remains opt-in for no-hit
     traffic.
2. **Keep Opt-In**
   - Correctness and capacity are good, but E2E overhead is too high or soak
     coverage is not strong enough for default promotion.
   - Name the exact blocker and next target.
3. **Blocked**
   - Correctness, graph replay, or Marlin release compatibility fails.
   - Identify the failing owner and write the shortest fix target.

Recommended acceptance bar:

- correctness and graph replay pass for buckets `[1,2,4,8,16]`;
- fixed 128, cap4096, and auto-capacity paths pass;
- Marlin WNA16 release + SWA independent remains clean;
- no correctness regression under prefix hit and eviction pressure;
- E2E regression is under about `3%` on key serving/prefix workloads, or a
  clear capacity-vs-latency tradeoff is documented;
- auto-capacity page/token gain remains close to the 08.31/08.42 result unless
  a safer planner intentionally reduces it.

### 6. TARGET 09.5 Decision

Revisit TARGET 09.5 only after the soak verdict:

- If SWA independent is promoted or kept as a stable high-capacity opt-in, keep
  TARGET 09.5 narrow: SWA-only FP8 on top of independent lifecycle.
- If BF16 SWA independent lifecycle already recovers most practical memory and
  FP8 ROI is tiny, keep TARGET 09.5 deferred.
- Do not broaden to MLA/indexer FP8 in this target.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_independent_post_fix_promotion_soak/
```

Required files:

- `README.md` with final verdict;
- `post_fix_source_sanity.md`;
- `correctness_graph_soak.md`;
- `serving_capacity_ledger.md`;
- `swa_tail_runtime_counters.md`;
- `macro_serving_performance.md`;
- `e2e_overhead_attribution.md`;
- `promotion_decision.md`;
- `target09_5_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. Did the 08.42 dummy-token fix stay correct under fixed, cap4096, and auto
   capacity?
2. Is SWA independent lifecycle correctness-clean under prefix/serving/eviction
   pressure?
3. Does it remain compatible with Marlin WNA16 release + component clear?
4. Does graph replay cover buckets `[1,2,4,8,16]` without avoidable eager
   fallback?
5. What is the real fixed-128, cap4096, and auto-capacity memory/token benefit?
6. What is the current E2E overhead owner, and is it acceptable?
7. Should SWA independent lifecycle be promoted, kept opt-in, or blocked?
8. What exactly should happen to TARGET 09.5 next?

## Stop Conditions

Stop and report rather than continuing to polish if:

- CUDA illegal memory access, NCCL watchdog, or corrupted text reappears;
- graph replay falls back to eager after adding the obvious missing bucket;
- a correctness failure appears under prefix hit/eviction pressure;
- Marlin release + SWA independent breaks the TARGET 08.40 component clear
  safety contract;
- one overhead owner clearly explains the regression and requires a dedicated
  optimization target;
- runtime SWA counters show no meaningful reduction in the workloads that were
  supposed to benefit.
