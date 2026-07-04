# TARGET 08.24: DSV4 Route B Metadata Deforest And Copy Elision

## Status

Run this after TARGET 08.22 rerun.

TARGET 08.22 rerun selected Route B as the preferred prefix-cache opt-in:

```text
performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/
Decision: Route_B_preferred_opt_in
```

Do not run TARGET 08.23 independent SWA ownership first.  The 08.22 rerun
showed the SWA-tail guard has a small workload impact in the measured suite,
while Route B still pays a visible decode metadata overhead because component
ownership keeps decode metadata deforest guarded off.

## Goal

Reduce Route B decode metadata preparation and graph-staging overhead without
changing the Route B ownership model or precision path.

The desired result is a Route B graph path that keeps the correctness and
capacity benefits from independent C4/C128/indexer/state ownership while
recovering most of the output-throughput gap versus phase-1 prefix cache.

## Background

TARGET 08.22 rerun summary:

| mode | mean TTFT s | mean output tok/s | saved prefill | graph replay/eager |
| --- | ---: | ---: | ---: | --- |
| prefix_off | 1.0961 | 50.8501 | 0 | 679/0 |
| phase1_prefix_on | 0.6946 | 65.4797 | 65536 | 679/0 |
| route_b_graph | 0.7706 | 53.4904 | 63232 | 679/0 |

Route B recovered `0.9648x` of phase-1 saved prefill tokens and passed
correctness/text/graph checks, but output throughput stayed at `0.8169x`
phase-1.  The strongest evidence points to guarded decode metadata deforest:

```text
performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/summaries/deforest_guard_cost.md
```

Representative decode prepare deltas:

| scenario | phase1 s | Route B s | delta s |
| --- | ---: | ---: | ---: |
| decode_ladder_bs16 | 0.1320 | 0.6281 | +0.4961 |
| serving_mixed_112req_wave16 | 0.9008 | 4.4587 | +3.5578 |
| prefix_multi_112req_wave16 | 0.1316 | 1.2133 | +1.0817 |
| prefix_eviction_pressure_96req_wave16 | 0.0178 | 0.1500 | +0.1322 |

The old deforest path is unsafe under Route B because it derives component
metadata from full-token locations, for example `full_loc // 4` and
`full_loc // 128`.  Route B deliberately allows old full/SWA pages to be
tombstoned while retaining independent C4/C128/indexer/state components, so the
deforest path must consume component-owned page tables and direct component
locations instead of inferring them from full pages.

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.21.4_dsv4_sm80_route_b_graph_deforest_serving.md`
- `prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/README.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/summaries/deforest_guard_cost.md`
- `performance_milestones/target08_route_b_graph_deforest_serving/README.md`

Mini code references:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/engine/graph.py`

vLLM/SGLang references for comparison, not blind copying:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`

## Scope

Allowed:

- add fine-grained timers/counters for Route B metadata construction, graph
  metadata copy, and per-field bytes/call counts;
- extend the decode metadata deforest fallback/Triton helper so it can consume
  Route B component page tables;
- eliminate per-step copies for metadata that is stable across decode steps;
- preallocate graph metadata buffers per bucket and update only active/dirty
  rows;
- fuse multiple small metadata copies into one or a few copy kernels;
- keep the new path behind a guarded opt-in until correctness and performance
  are proven.

Not allowed:

- independent SWA ownership or SWA KV reconstruction;
- low-precision KV/cache/projection changes;
- attention kernel algorithm tuning unrelated to metadata staging;
- changing the promoted TARGET 07 non-prefix path;
- default promotion in this target without a separate explicit gate.

## Implementation Plan

### Phase 0: Attribute The Overhead

Before changing behavior, produce an attribution note that answers:

- how many times per decode step each Route B metadata builder/copy path runs;
- which fields are copied into graph replay buffers;
- approximate bytes copied per field and per scenario;
- which fields are stable per request, per prefix hit, per graph bucket, or per
  decode token;
- which operations allocate or materialize intermediate tensors.

At minimum, classify these fields:

- base decode fields: `raw_out_loc`, `positions`, `page_table`;
- SWA fields: `swa_page_indices`, SWA lengths;
- C4 fields: `c4_sparse_*`, `c4_topk_*`, `c4_page_table`;
- C128 fields: `c128_*`, `c128_page_table`;
- indexer fields: `c4_indexer_out_loc`, `c4_indexer_page_table`;
- direct component locs: `c4_out_loc`, `c128_out_loc`,
  `c4_indexer_out_loc`.

### Phase 1: Component-Aware Deforest Preflight

Create a CPU or eager oracle that proves a component-aware deforest formula can
reproduce current Route B eager metadata without using `full_loc // ratio`.

The preflight must include:

- full hit, partial hit, miss, and eviction pressure;
- exact page multiples `512`, `768`, `1024`;
- neighboring controls `513`, `769`, `1025`;
- mixed hit/miss and multi-prefix workloads;
- tombstoned full/SWA pages with live C4/C128/indexer components.

### Phase 2: Kernelize The Safe Path

Extend or add kernels so Route B can build decode metadata from component-owned
inputs:

- consume `c4_page_table`, `c128_page_table`, and
  `c4_indexer_page_table`;
- produce C4/C128/indexer page indices and top-k length metadata directly;
- generate direct component output locs where possible instead of copying
  compact loc vectors from eager output;
- reject or fall back safely if a required component mapping is missing.

Keep the old non-Route-B deforest path unchanged unless a shared helper can be
used without behavior risk.

### Phase 3: Copy Elision And Buffer Reuse

Reduce graph staging overhead:

- preallocate metadata buffers per graph bucket;
- copy only active rows, not the full padded bucket table, where possible;
- avoid copying component page tables on every decode step if the request's
  component mapping did not change;
- replace chains of tiny copies with a packed copy/update kernel;
- avoid `clone`, `cat`, and temporary materialization in the hot decode path.

### Phase 4: Correctness And Serving Gate

Compare at least three variants:

- phase-1 prefix on;
- Route B graph baseline from the 08.22 rerun;
- Route B graph with metadata deforest/copy-elision opt-in.

Use the TARGET 08.198 guarded oracle:

- pass/fail: slot-pinned same-layout correctness, metadata consistency, no
  stale reads, no double frees, no leaks, text smoke without garbled output;
- diagnostic only: cross-slot generated-token equality and filler-content
  equality.

Required scenarios:

- `decode_ladder_bs16`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`;
- `prefix_eviction_pressure_96req_wave16`;
- exact page multiples and neighboring controls from 08.22;
- TP8 text smoke with graph buckets `[1,2,4,8,16]`.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_metadata_deforest_copy_elision/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- overhead attribution table before optimization;
- component-aware metadata formula and safety proof;
- field stability table: per-token, per-request, per-hit, per-bucket;
- copy-elision changes and buffer lifetime notes;
- correctness table;
- text-smoke table;
- graph replay/eager table;
- serving A/B table against phase-1 and 08.22 Route B baseline;
- final decision: keep opt-in, promote as Route B default, or reject.

## Success Criteria

Treat this target as successful if:

- correctness/text/graph gates stay green;
- Route B saved-prefill ratio remains close to 08.22 rerun;
- Route B decode prepare overhead drops by at least `50%` in the large-wave
  scenarios, or output throughput recovers to at least `0.90x` of phase-1 in
  the same benchmark family;
- no new CUDA graph eager fallbacks appear for buckets `[1,2,4,8,16]`;
- no new leak, stale read, or lifecycle bug appears under eviction pressure.

## Stop Rules

Stop and report instead of continuing to polish if:

- the overhead is dominated by attention compute rather than metadata/copy;
- component-aware deforest requires broad attention kernel rewrites;
- graph replay breaks or correctness becomes ambiguous;
- copy-elision gives less than `10%` decode prepare reduction after the main
  obvious copies are removed;
- SWA ownership becomes the only remaining blocker.  In that case, return to
  TARGET 08.23 with concrete trigger evidence.

