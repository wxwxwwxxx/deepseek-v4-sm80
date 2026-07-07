# TARGET 08.42: DSV4 SM80 SWA Large-Capacity Serving Correctness

## Status

Active TARGET 08 correctness target after TARGET 08.41.

TARGET 08.41 proved that SWA independent lifecycle is correctness-clean for the
fixed `--num-pages 128` path, including prefix/serving/eviction pressure and
CUDA graph buckets `[1,2,4,8,16]`.  It also proved that SWA independent
lifecycle has real large-capacity value when combined with TARGET 08.40 Marlin
WNA16 raw-expert release:

```text
Marlin release baseline auto capacity:          2776 pages / 710656 tokens
Marlin release + SWA independent auto capacity: 6489-6636 pages / ~1.66M-1.70M tokens
```

However, the high-capacity serving path is not correctness-clean:

- auto-capacity Marlin release + SWA independent passes text smoke and the
  first historical macro case, then crashes in serving with CUDA illegal memory
  access;
- explicit `--num-pages 4096` also crashes in serving at about `32.02 GiB/rank`
  KV, with enough graph-capture headroom;
- the stack often surfaces while reading `_swa_page_refcount`, but that is
  likely only the first synchronized observation of an earlier illegal device
  access.

This target should identify and fix that large-capacity SWA serving correctness
bug before any promotion or FP8 cache work continues.

## Goal

Make Marlin release + SWA independent lifecycle serving-correct at large
capacity:

```text
large page count
-> SWA independent full-to-SWA mapping
-> prefix hit / no-hit serving flows
-> graph buckets [1,2,4,8,16]
-> no CUDA illegal memory access
-> no stale SWA handle/refcount/free-list corruption
-> no component loc invalidation
```

The primary output is a concrete root cause and fix for the large-capacity
illegal memory access.  Performance overhead is secondary and should not be
optimized before correctness is stable.

## Speed Rule: No-Weight First

Do not begin by repeatedly loading the full DeepSeek V4 weights.

The investigation should first build no-weight or partial repros that exercise
SWA lifecycle and metadata behavior without model weight loading:

- instantiate `DeepSeekV4KVCache` / radix cache / scheduler cache structures
  directly when possible;
- use synthetic request/page/prefix states to allocate, free, tombstone, and
  remap SWA pages;
- construct synthetic full-to-SWA mappings and SWA page tables at large page
  counts;
- invoke attention metadata builders or graph metadata preparation with fake
  tensors where possible;
- use tiny synthetic CUDA tensors to check bounds, refcount, free-list, dummy
  page, and table-index invariants;
- only load the full model for final text smoke, macro reproduction, and
  promotion-grade verification.

If a full-model run is needed for a candidate boundary, keep it short and
targeted.  Do not run broad macro matrices until a narrowed hypothesis exists.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/README.md
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/correctness_graph_soak.md
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/serving_capacity_ledger.md
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/swa_tail_runtime_counters.md
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/promotion_decision.md
performance_milestones/target08_swa_independent_lifecycle_promotion_soak/target09_5_recommendation.md
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
prompts/TARGET_08.41_dsv4_sm80_swa_independent_lifecycle_promotion_soak.md
```

Key TARGET 08.41 facts:

- fixed-128 SWA independent serving is clean;
- graph buckets `[1,2,4,8,16]` replay with `0` eager for fixed-128;
- Marlin release + SWA independent text smoke passes at auto capacity;
- Marlin release + SWA independent serving crashes at auto capacity and at
  explicit `--num-pages 4096`;
- cap4096 has enough graph memory headroom, so this is not merely near-OOM;
- suspected area: large-capacity SWA mapping/page-table/refcount/free-list
  interaction under serving pressure, with Marlin release/component ownership
  enabled;
- E2E overhead owner is decode prepare / attention metadata, but overhead work
  must wait until correctness is fixed.

## Non-Goals

- Do not implement FP8 KV/cache or INT8 MoE.
- Do not optimize SWA metadata overhead before the illegal memory access is
  fixed.
- Do not promote SWA independent lifecycle.
- Do not broaden to MLA/indexer FP8.
- Do not hide the bug by lowering capacity back to 128 pages.
- Do not bypass TARGET 08.40 component-slot clear semantics.
- Do not make unsafe no-clear Marlin release paths production-visible.

## Required Investigation

### 1. No-Weight Repro Ladder

Build a fast repro ladder before full model runs.

Suggested page-count ladder:

```text
128, 512, 1024, 2048, 4096, 6489, 6636
```

Synthetic/no-weight probes should cover:

- SWA page allocation/free/tombstone at each page count;
- `_swa_page_refcount` bounds and dtype/device invariants;
- full-page to SWA-page mapping construction and teardown;
- dummy SWA page substitution for invalid/full dummy rows;
- SWA page-table row construction for serving batch shapes;
- prefix hit/no-hit/eviction sequences that mimic:
  - `serving_mixed_112req_wave16`;
  - `prefix_multi_112req_wave16`;
  - `prefix_eviction_pressure_96req_wave16`;
- Route B component loc ownership staying valid while SWA rows tombstone/free;
- component page-table lifetime cache interaction with changing SWA tables;
- graph metadata preparation and replay buffer update behavior, if it can be
  exercised without full weights.

Every no-weight probe should be able to run quickly and should report the first
page count / sequence where an invariant fails.

### 2. Toggle Bisection

Narrow which condition is required for the crash.

Run focused probes or short full-model smokes for:

- SWA independent without Marlin release;
- Marlin release without SWA independent;
- Marlin release + SWA independent with fixed 128 pages;
- Marlin release + SWA independent with explicit caps:
  - 512;
  - 1024;
  - 2048;
  - 4096;
- auto capacity;
- graph off vs graph on;
- graph buckets `[1,2,4,8,16]` vs minimal buckets;
- prefix disabled / no-hit serving / prefix-hit serving / eviction pressure;
- direct C4 graph metadata buffers on/off if relevant;
- component page-table lifetime cache on/off if relevant.

The target should identify the smallest toggle set and page count that
reproduces the illegal memory access.

### 3. Device-Side Bounds And Invariant Checks

Add debug-only checks around likely owners.

Candidate invariants:

- every SWA page index used by metadata is either in `[0, swa_capacity_pages)`
  or exactly the dummy SWA page;
- no full-page index is accidentally used as an SWA page index;
- every SWA page-table row has expected shape and stride for the current batch;
- tombstoned SWA handles never reappear in graph metadata unless mapped to
  dummy;
- freed SWA pages are not still present in active request mappings;
- refcount increments/decrements never go negative and never exceed plausible
  active ownership count;
- free-list entries are unique and within range;
- prefix node SWA handles and active request SWA mappings agree after hit,
  eviction, and append;
- component locs are never derived from released/tombstoned SWA rows.

Where possible, implement cheap Python/Torch assertions in no-weight probes.
For GPU table builders, add debug kernels or device-side sentinel checks only
when needed.  If using `CUDA_LAUNCH_BLOCKING=1` or compute-sanitizer, use the
smallest repro.

### 4. Full-Model Confirmation

Only after the repro is narrowed, run full-model checks.

Minimum full-model gates after a candidate fix:

- text smoke:
  - Marlin release + SWA independent auto capacity;
  - fixed 128 SWA independent;
- cap ladder:
  - at least the first failing cap from the repro;
  - `4096`;
  - auto capacity;
- serving macro:
  - `serving_mixed_112req_wave16`;
  - `prefix_multi_112req_wave16`;
  - `prefix_eviction_pressure_96req_wave16` or closest available;
- graph buckets `[1,2,4,8,16]`;
- page size `256`;
- TP8.

Required pass criteria:

- no CUDA illegal memory access;
- no NCCL watchdog abort;
- graph replay has `0` avoidable eager decode calls;
- no SWA negative refcount, double-free, leaked active mapping, stale handle,
  or out-of-range page-table row;
- Marlin release still reports component clear and `0` guard/reserved bytes;
- capacity remains meaningfully above baseline Marlin release pages.

### 5. Fix Strategy

Prefer the smallest evidence-backed fix:

- if page index type/width overflows at large page count, fix dtype/shape at
  the producer and add a regression test;
- if full-page indices leak into SWA page-table rows, separate index spaces and
  add invariant checks;
- if dummy SWA page substitution is incomplete, fix the substitution boundary;
- if graph metadata caches stale SWA rows, invalidate/update only SWA metadata
  without touching stable C4 component metadata;
- if prefix eviction leaves stale SWA handles, fix tombstone/free lifecycle and
  refcount handling;
- if CUDA graph replay captures dynamic SWA table addresses incorrectly, make
  the SWA table graph input/update explicit or disable only the unsafe direct
  path.

Do not apply a broad workaround that disables SWA independent lifecycle or
falls back to 128 pages unless the report labels it as temporary and proposes
the real fix.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_large_capacity_serving_correctness/
```

Required files:

- `README.md` with verdict and next action;
- `no_weight_repro_ladder.md`;
- `toggle_bisection.md`;
- `bounds_invariant_checks.md`;
- `root_cause.md`;
- `fix_summary.md` if code changes are made;
- `full_model_confirmation.md`;
- `capacity_after_fix.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. Was the illegal memory access reproduced without loading weights?
2. What is the minimum page count / toggle set that reproduces it?
3. Which owner is responsible: SWA mapping, page table, dummy page, refcount,
   prefix eviction, graph metadata, or another owner?
4. What exact fix was made?
5. Does Marlin release + SWA independent pass serving at cap4096 and auto
   capacity?
6. How much capacity remains after the fix?
7. Is TARGET 08.41 promotion soak ready to rerun, or is another correctness
   target needed?

## Stop Conditions

Stop and report if:

- a no-weight repro identifies a concrete invariant violation and a fix is
  implemented;
- a full-model-only crash remains after no-weight probes pass, with the narrow
  full-model owner identified;
- cap4096 and auto capacity pass after the fix and the 08.41 soak can be
  rerun;
- after two focused bisection rounds, no new evidence appears.  In that case,
  document all negative results and propose the next diagnostic hook rather
  than running more broad full-model matrices.
