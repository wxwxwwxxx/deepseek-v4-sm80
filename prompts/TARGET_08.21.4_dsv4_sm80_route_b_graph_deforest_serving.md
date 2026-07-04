# TARGET 08.21.4: DSV4 Route B Graph/Deforest/Serving Integration

## Status

Planned after TARGET 08.21.3.

This is Route B / B3.  Run after eager Route B component ownership and state
correctness are proven or guarded.

## Goal

Integrate Route B with graph metadata copy, decode metadata deforest, and
serving-style benchmarks.

The target should answer:

1. Can Route B preserve CUDA graph replay for the selected serving buckets
   `[1,2,4,8,16]`?
2. Can decode metadata deforest produce component loc metadata directly, or must
   it be guarded off?
3. What is the real capacity recovery, TTFT, prefill-forward, output throughput,
   and graph/eager profile versus phase-1 prefix cache?
4. Is Route B ready to become the preferred prefix-cache opt-in?

## Required Reading

- `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md`
- `prompts/TARGET_08.21.1_dsv4_sm80_component_loc_table_preflight.md`
- `prompts/TARGET_08.21.2_dsv4_sm80_independent_compressed_indexer_ownership.md`
- `prompts/TARGET_08.21.3_dsv4_sm80_compression_state_ownership.md`
- previous Route B milestone READMEs.

## Scope

Allowed:

- update graph metadata copy for component loc fields;
- update decode metadata deforest to emit component loc metadata;
- or guard deforest off for Route B with measured cost;
- run serving workload benchmarks;
- update capacity ledger and promotion decision.

Not allowed:

- changing low-precision behavior;
- broad attention-kernel optimization unrelated to component loc metadata;
- graph private-pool attribution beyond reporting;
- default promotion without a separate final gate.

## Required Tests

Correctness:

- guarded slot-pinned prefix-on/off logits and metadata;
- text smoke;
- full hit, partial hit, miss;
- repeated hit/evict;
- eviction pressure;
- graph replay and eager fallback counts;
- graph bucket exact-bs guard behavior.

Performance/capacity:

- phase-1 prefix off;
- phase-1 prefix on;
- Route B eager if graph is guarded;
- Route B graph if supported;
- TARGET 08.10 shared-prefix workloads;
- serving bucket distribution `[1,2,4,8,16]`;
- retained full/SWA pages;
- retained C4/C128/indexer/state slots;
- recovered pages/tokens/GiB per rank;
- TTFT, prefill-forward, output throughput.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_graph_deforest_serving/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- graph metadata field map;
- deforest support/guard decision;
- correctness table;
- text smoke results;
- graph replay/eager counts;
- capacity recovery table;
- performance A/B table;
- decision: keep Route B opt-in, prepare final prefix promotion gate, or split
  another Route B follow-up.

## Decision Rules

Route B may become the preferred opt-in if:

- guarded correctness passes;
- text smoke is clean;
- graph replay coverage is preserved or eager fallback cost is acceptable and
  documented;
- capacity recovery is material;
- TTFT/prefill benefit remains useful versus phase-1 prefix cache;
- no leaks, double frees, or stale refs.

Stop and report blocked if:

- graph metadata copy cannot safely represent component locs;
- deforest cannot be guarded without major performance regression;
- serving performance is dominated by metadata/materialization overhead;
- capacity recovery is below noise after all required guards.
