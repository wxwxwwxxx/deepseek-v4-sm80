# TARGET 08.41: DSV4 SM80 SWA Independent Lifecycle Promotion Soak

## Status

Active TARGET 08 follow-up after TARGET 08.31.

TARGET 08.31 implemented opt-in, SGLang-aligned SWA independent lifecycle for
DeepSeek V4 on A100/sm80.  Correctness, component ownership, Route B page-table
lifetime caching, graph replay, and Marlin WNA16 release compatibility were
validated.  Runtime counters proved that live SWA tails are much smaller than
the old conservative 128-page SWA retention model.

This target is the promotion/soak pass.  It should not re-implement SWA
lifecycle.  It should decide whether the current implementation is ready to
promote, should remain opt-in with a known overhead blocker, or needs one
focused fix first.

## Goal

Stabilize and evaluate the SWA independent lifecycle under serving-like
pressure:

```text
SGLang-aligned SWA independent lifecycle
-> full/C4/C128/indexer/state ownership stays valid
-> Marlin WNA16 release + component-slot clear stays compatible
-> graph replay works for serving buckets
-> runtime SWA tail counters prove real memory/capacity value
-> E2E overhead is attributed and bounded
-> promotion decision is recorded
```

The desired output is a clear decision:

1. promote SWA independent lifecycle as a default or named high-capacity
   prefix/serving preset;
2. keep it opt-in and propose one focused overhead/correctness follow-up;
3. reject/defer promotion if serving correctness or graph replay is unstable.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_independent_lifecycle/summaries/TARGET_08.31_report.md
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Important TARGET 08.31 conclusions:

- SWA KV now has independent page lifecycle and can tombstone/free
  out-of-window tail pages without invalidating C4/C128/indexer/state/component
  locations.
- Feature gates:
  - CLI: `--enable-dsv4-swa-independent-lifecycle`
  - env: `MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1`
  - requires radix prefix cache and DSV4 component loc ownership.
- Marlin WNA16 raw expert release + component-slot clear from TARGET 08.40
  remains compatible.
- Unit/integration gate passed: `106 passed`.
- Fixed `--num-pages 128` safe-floor planning gives no persistent GiB win in
  macro/serving runs because the planner conservatively keeps `128` SWA pages.
- Runtime counters still show small live SWA tails:
  - historical: live SWA pages about `4`;
  - serving mixed: about `18`;
  - prefix multi: about `26`.
- Marlin release auto-capacity path is the strongest capacity result:
  - baseline release: `2776` pages / `710656` tokens;
  - release + SWA independent: `6636` pages / `1698816` tokens;
  - effectively same KV memory budget, about `2.39x` pages.
- Macro decode throughput was near noise, but E2E output throughput regressed:
  - historical `4096/128/bs4`: about `-3.81%`;
  - serving mixed: about `-3.11%`;
  - prefix multi: about `-8.96%`.
- Prefix multi in 08.31 used graph buckets `[1,2,4]`; its decode batch size
  reached `16`, so `replay=0/eager=49` there was expected and must be re-run
  with graph bucket `16` included before promotion.

## Non-Goals

- Do not implement FP8 KV/cache in this target.
- Do not implement INT8 MoE or quantized communication.
- Do not rewrite the attention kernels unless an extremely small metadata fix
  is needed for graph replay stability.
- Do not re-open the entire SWA ownership design unless correctness fails.
- Do not promote prefix cache as a universal default for all no-hit traffic.
- Do not bypass TARGET 08.40 component-slot clear semantics.

## Required Work

### 1. Source And Semantics Audit

Review the 08.31 implementation and confirm the production contract:

- SWA independent lifecycle is off by default unless a preset/CLI/env enables
  it.
- It requires radix prefix cache and DSV4 component loc ownership.
- It cannot invalidate C4/C128/indexer/compression-state/component locs when
  SWA rows are tombstoned or freed.
- `on_pages_allocated()` still applies TARGET 08.40 component-slot clear for
  Marlin WNA16 raw expert release.
- Prefix node SWA handles, active request SWA mappings, and dummy SWA page
  semantics are documented.
- Direct C4 graph metadata buffers remain valid.  Direct SWA graph metadata is
  intentionally disabled or separately justified in independent mode.
- Unsafe low-SWA-pool debug modes from 08.31 are not exposed as default serving
  behavior.

Deliver a short source review.  If any semantic gap is found, fix it before
running the long soak.

### 2. Correctness And Graph Soak

Run TP8 text and macro correctness with page size `256`.

Required graph buckets:

```text
--cuda-graph-bs 1 2 4 8 16
```

Required variants:

- promoted prefix Route B lifetime baseline;
- prefix Route B lifetime + SWA independent;
- prefix Route B lifetime + Marlin release;
- prefix Route B lifetime + Marlin release + SWA independent.

Required gates:

- text smoke passes with sane outputs;
- no garbled text, BOS/token-0 flood, crash, NaN/Inf logit explosion;
- graph replay is zero-eager for captured serving buckets;
- prefix hit/remap/eviction probes do not use tombstoned SWA rows;
- component page-table lifetime verification passes if enabled;
- no SWA negative refcount, double-free, stale handle, or leaked active mapping;
- Marlin release path still reports component-slot clear and `0` guard bytes.

If a workload uses a decode batch size outside captured buckets, either add
the bucket or label it clearly; do not treat avoidable eager fallback as a
promotion-ready result.

### 3. Serving And Capacity Soak

Run fixed-capacity and auto-capacity paths separately.

Fixed-capacity path:

```text
--num-pages 128
--page-size 256
```

Auto-capacity path:

```text
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
- one higher-concurrency case such as wave64 if the harness supports it.

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

The capacity ledger must distinguish:

- fixed `--num-pages 128` safe-floor planning;
- runtime-proven live SWA tail reduction;
- Marlin release auto-capacity page/token gain.

### 4. E2E Regression Attribution

The 08.31 implementation had acceptable decode throughput but E2E output
regressed in short offline macro runs.  Attribute this before promotion.

Use owner timing, lightweight NVTX/CPU timers, or existing benchmark fields to
split overhead into:

- scheduler page allocation/free;
- SWA tombstone/free bookkeeping;
- prefix hit/remap path;
- SWA page-table construction;
- attention metadata preparation;
- graph metadata copy/replay preparation;
- decode forward;
- prefill forward;
- sampling / postprocess if relevant.

Produce a table comparing baseline vs SWA independent for:

- historical `4096/128/bs4`;
- serving mixed wave16;
- prefix multi wave16;
- auto-capacity Marlin release path if available.

Stop optimizing if overhead is already within noise after proper graph bucket
coverage.  If a single overhead owner explains most of the E2E regression,
propose a focused follow-up rather than broad local polishing.

### 5. Promotion Decision

End with one decision:

1. **Promote**
   - SWA independent lifecycle becomes a named serving/high-capacity preset.
   - It may remain opt-in for no-hit traffic if prefix cache remains opt-in.
2. **Keep Opt-In**
   - Correctness and capacity are good, but E2E overhead or soak coverage is not
     strong enough for default promotion.
   - Name the exact blocker and next target.
3. **Blocked**
   - Correctness, graph replay, or Marlin release compatibility fails.
   - Identify the failing owner and write the shortest fix target.

Recommended acceptance bar:

- correctness and graph replay pass for buckets `[1,2,4,8,16]`;
- Marlin release + SWA independent auto-capacity passes;
- no correctness regression under prefix hit and eviction pressure;
- E2E regression is under about `3%` on key serving/prefix workloads, or a
  clear capacity/serving tradeoff is documented;
- auto-capacity page/token gain remains close to the 08.31 result unless a
  safer planner intentionally reduces it.

### 6. TARGET 09.5 Decision

Revisit TARGET 09.5 only after the soak verdict:

- If SWA independent is promoted or kept opt-in with stable counters, keep
  TARGET 09.5 narrow: SWA-only FP8 on top of independent lifecycle.
- If SWA independent recovers enough memory and FP8 ROI is tiny, keep TARGET
  09.5 deferred.
- Do not broaden to MLA/indexer FP8 in this target.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/
```

Required files:

- `README.md` with final verdict;
- `source_semantics_review.md`;
- `correctness_graph_soak.md`;
- `serving_capacity_ledger.md`;
- `swa_tail_runtime_counters.md`;
- `macro_serving_performance.md`;
- `e2e_overhead_attribution.md`;
- `promotion_decision.md`;
- `target09_5_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. Is SWA independent lifecycle correctness-clean under prefix/serving pressure?
2. Does it remain compatible with Marlin WNA16 release + component clear?
3. Does graph replay cover buckets `[1,2,4,8,16]` without avoidable eager
   fallback?
4. What is the real fixed-128 and auto-capacity memory/token benefit?
5. What caused the 08.31 E2E regression, and is it acceptable?
6. Should SWA independent lifecycle be promoted, kept opt-in, or blocked?
7. What exactly should happen to TARGET 09.5 next?

## Stop Conditions

Stop and report rather than continuing to polish if:

- graph replay still falls back to eager after adding the obvious missing
  bucket;
- a correctness failure appears under prefix hit/eviction pressure;
- Marlin release + SWA independent breaks the TARGET 08.40 component clear
  safety contract;
- one overhead owner clearly explains the regression and requires a dedicated
  optimization target;
- runtime SWA counters show no meaningful reduction in the workloads that were
  supposed to benefit.
