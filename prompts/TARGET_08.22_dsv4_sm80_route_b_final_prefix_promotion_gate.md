# TARGET 08.22: DSV4 Route B Final Prefix Promotion Gate

## Status

Active next TARGET 08 subtarget.

Run this after TARGET 08.21.4.  TARGET 08.21.4 showed that Route B is
graph-capable and is a preferred opt-in candidate, but it did not run the full
serving/correctness gate needed for promotion.

## Goal

Decide whether `--enable-dsv4-component-loc-ownership` should become the
preferred DSV4 prefix-cache opt-in, remain experimental, or require independent
SWA ownership before promotion.

This target should run the complete serving gate.  Do not implement independent
SWA ownership here.  The current Route B SWA-tail guard should remain enabled
and measured.

## Background

Completed Route B milestones:

- TARGET 08.21.1 proved direct component loc tables can reproduce phase-1
  metadata while full pages stay live.
- TARGET 08.21.2 implemented independent C4/C128/indexer component ownership.
- TARGET 08.21.3 implemented independent C4/C128/indexer compression-state
  ownership.
- TARGET 08.21.4 restored Route B graph replay for buckets `[1,2,4,8,16]`,
  kept decode metadata deforest guarded off, and quantified the remaining
  SWA-tail guard.

Important TARGET 08.21.4 conclusion:

```text
Route B is graph-capable as a preferred opt-in candidate; prepare a final
prefix promotion gate rather than promoting by default in 08.21.4.
```

For `page_size=256`, the SWA-tail guard is mostly a risk at exact page-multiple
prompt lengths:

| prompt len | phase-1 hit | Route B hit |
| ---: | ---: | ---: |
| 256 | 0 | 0 |
| 257 | 256 | 256 |
| 512 | 256 | 0 |
| 513 | 512 | 512 |
| 768 | 512 | 0 |
| 769 | 768 | 768 |

Do not assume this is a blocker until a serving workload shows those exact
page-multiple cases are common enough to matter.

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md`
- `performance_milestones/target08_component_loc_table_preflight/README.md`
- `performance_milestones/target08_independent_compressed_indexer_ownership/README.md`
- `performance_milestones/target08_compression_state_ownership/README.md`
- `performance_milestones/target08_route_b_graph_deforest_serving/README.md`
- `prompts/TARGET_08.10_dsv4_sm80_prefix_cache_serving_stability_promotion_gate.md`
- `prompts/TARGET_08.198_dsv4_sm80_post_layer0_same_shape_decode_drift.md`

Core code references:

- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/engine/graph.py`

## Scope

Allowed:

- add or refine benchmark scenarios needed for a fair final prefix gate;
- add summary scripts for multi-run comparisons;
- fix small Route B correctness/graph-copy bugs found by the gate;
- improve reporting for exact page-multiple prompt lengths and SWA-tail guard
  shortening;
- decide whether Route B should become the preferred opt-in.

Not allowed:

- independent SWA ownership;
- SWA KV reconstruction;
- low-precision work;
- attention-kernel or communication optimization;
- default promotion without a separate explicit decision and clean evidence;
- decode deforest port unless it is a very small local bug fix.  If deforest is
  still guarded, measure and report the cost.

## Required Runs

Use separate `torchrun` invocations per variant when needed.  The engine has
CUDA lifecycle assumptions, so avoid destroying and recreating multiple engines
inside one Python process for final gate evidence.

Compare at least:

- prefix off;
- phase-1 prefix on;
- Route B graph:
  - `--enable-dsv4-radix-prefix-cache`
  - `--enable-dsv4-component-loc-ownership`
  - `--allow-dsv4-cuda-graph`
  - `--cuda-graph-bs 1 2 4 8 16`

Use the promoted path:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
```

Serving workloads should include:

- the full TARGET 08.10 shared-prefix suite, including multi-request waves;
- mixed hit/miss workloads;
- eviction pressure;
- short-output and longer-output variants;
- exact page-multiple prompt lengths such as `512` and `768`;
- neighboring non-multiple controls such as `513` and `769`;
- at least one non-shared-prefix control.

If runtime cost is high, keep a compact smoke matrix first, then run the full
suite only on the candidates that pass.

## Correctness Gate

Use the TARGET 08.198 guarded oracle:

- pass/fail:
  - slot-pinned, same-layout prefix-on versus prefix-off;
  - metadata consistency;
  - no stale reads, double frees, leaks, or cache-state corruption;
  - text smoke with no garbled/invalid-byte output, degeneracy, or crash.
- diagnostics only:
  - cross-slot generated-token equality;
  - filler-content equality;
  - identical-row equality.

Required checks:

- TP8 text smoke for Route B graph;
- shared-prefix Chinese and English prompts;
- exact page-multiple and non-multiple pairs;
- repeated hit/evict;
- eviction pressure;
- graph replay/eager counts for buckets `[1,2,4,8,16]`;
- prefix cache metrics: hit requests, saved prefill tokens, retained pages,
  retained component/state slots, evictions.

## Performance And Capacity Gate

Report:

- TTFT;
- TPOT / ITL where available;
- output token throughput;
- prefill-forward throughput;
- graph replay count and eager decode count;
- captured buckets;
- hit rate and saved prefill tokens;
- retained full/SWA pages;
- retained C4/C128/indexer slots;
- retained C4/C128/indexer state slots;
- recovered pages/tokens/GiB per rank versus phase-1;
- exact page-multiple frequency and total hit shortening;
- deforest guard status and measured overhead if visible.

The summary should clearly compare:

```text
prefix_off vs phase1_prefix_on vs route_b_graph
```

## Deliverables

Create:

```text
performance_milestones/target08_route_b_final_prefix_promotion_gate/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- correctness table;
- text smoke outputs;
- full serving A/B table;
- graph replay/eager table;
- capacity ledger;
- SWA-tail guard shortening table;
- exact page-multiple workload frequency;
- deforest guard cost;
- final decision:
  - preferred opt-in;
  - keep experimental;
  - or proceed to TARGET 08.23 independent SWA ownership.

## Decision Rules

Route B can become the preferred DSV4 prefix-cache opt-in if:

- guarded correctness passes;
- TP8 text smoke is clean;
- graph replay covers `[1,2,4,8,16]` with no material eager decode fallback;
- Route B performance is close to phase-1 prefix-on while recovering meaningful
  component/full-page capacity;
- SWA-tail guard shortening is rare or has small serving impact;
- phase-1 rollback remains intact.

Proceed to TARGET 08.23 independent SWA ownership if:

- exact page-multiple prompt lengths are common enough to materially reduce
  Route B hit rate or TTFT benefit;
- Route B capacity recovery is limited by the live full/SWA tail;
- or promotion is blocked mainly by SWA-tail guard behavior.

Stop and report blocked if:

- Route B graph metadata copy shows stale component reads;
- eager decode fallback dominates the Route B graph path;
- text smoke or slot-pinned correctness fails;
- deforest guard overhead erases the prefix-cache win.
