# TARGET 08.27: DSV4 SGLang-Aligned Route B Metadata Lifetime

## Status

Run this after TARGET 08.26.

TARGET 08.26 re-ranked the Route B remaining gap and found that the dominant
owner is still decode prepare, especially component page-table construction and
metadata updates that repeat on every graph replay step.

Do not treat "stable-row / dirty-row" as the design up front.  The main rule for
this target is:

> Do not reinvent a runtime mechanism when SGLang or vLLM already has a mature
> design that can be adapted cleanly.

The first deliverable is therefore a source-parity design note.  Only after that
should the target implement the smallest useful opt-in.

## Goal

Reduce Route B decode metadata lifetime overhead by aligning mini-sglang with
SGLang's mature DSV4 metadata and CUDA graph preparation patterns.

The target should answer:

1. Which SGLang mechanism corresponds to mini's Route B metadata hotspot?
2. Can mini avoid rebuilding component page tables every decode step by using a
   persistent request/component mapping table, graph-resident raw metadata, or
   SGLang-style graph metadata refresh contract?
3. If a small implementation is feasible, does it reduce Route B decode prepare
   time without breaking correctness, graph replay, or Route B ownership rules?

## Required Reading

Project route:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.24_dsv4_sm80_route_b_metadata_deforest_copy_elision.md`
- `prompts/TARGET_08.25_dsv4_sm80_route_b_direct_graph_metadata_buffers.md`
- `prompts/TARGET_08.26_dsv4_sm80_route_b_remaining_gap_attribution_reset.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/README.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/README.md`
- `performance_milestones/target08_route_b_direct_graph_metadata_buffers/README.md`
- `performance_milestones/target08_route_b_remaining_gap_attribution_reset/README.md`

Mini implementation:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

SGLang references, read before designing:

- `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py`
- `/workspace/sglang-main/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- `/workspace/sglang-main/python/sglang/srt/environ.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`

vLLM may be used as a secondary reference if SGLang does not cover a boundary:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_coordinator.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/block_pool.py`

## Known Starting Evidence

TARGET 08.26 unprofiled `serving_mixed_112req_wave16` repeats:

| mode | output tok/s | decode prepare s | decode forward s | replay/eager |
| --- | ---: | ---: | ---: | --- |
| phase1 prefix on | 169.7381 +/- 0.8408 | 0.9403 | 9.9757 | 441/0 |
| Route B baseline | 136.2373 +/- 0.4446 | 4.4798 | 10.0897 | 441/0 |
| Route B direct C4 | 138.1281 +/- 0.7047 | 4.2067 | 10.1297 | 441/0 |
| Route B direct SWA+C4+C128 | 141.4511 +/- 1.2289 | 3.8731 | 9.9964 | 441/0 |

TARGET 08.26 owner timing showed:

- the remaining Route B direct C4 gap is mostly decode prepare;
- component page-table construction is the largest measured decode-prepare
  owner;
- `page_table`, `c4_page_table`, `c128_page_table`,
  `c4_indexer_page_table`, and derived C128/C4 buffers are rebuilt or copied on
  all 441 decode replay steps;
- decode forward, graph replay coverage, and NCCL counters are not the next
  owner.

Mini hotspot:

- `DeepseekV4Attention._make_component_page_tables(...)` reconstructs C4/C128
  component page tables from Route B cache handles and active full pages every
  decode step.

SGLang patterns observed during pre-review:

- `SGLANG_PREP_IN_CUDA_GRAPH` defaults to true.
- DSV4 decode first creates lightweight raw metadata
  (`req_pool_indices`, `seq_lens`, `out_cache_loc`).
- SGLang can upgrade raw decode metadata to full DSV4 attention metadata inside
  the graph path.
- core page tables are derived from a stable `req_to_token` mapping instead of
  rebuilding per-request rows from prefix handles every token.
- CUDA graph metadata has an explicit copy-versus-reference assignment
  contract for captured tensor addresses.
- `CudaGraphBufferRegistry` provides a reusable stable-buffer/padding/grouped
  copy pattern for ForwardBatch fields.
- unified radix components model component ownership, tombstones, and component
  lifecycle at the cache layer.

## Scope

Allowed:

- write a source-parity design note before implementation;
- add lightweight probes for request-slot stability, dirty-row opportunity, and
  per-step table rebuild counts;
- implement one minimal opt-in if the SGLang-aligned design is clear;
- introduce a persistent component mapping table or component-page table pool if
  it is small and follows SGLang's `req_to_token` style;
- prototype raw-metadata / graph-prep behavior if it can be isolated behind an
  opt-in;
- keep 08.24 and 08.25 opt-ins as comparison variants;
- add focused counters and summary scripts.

Not allowed:

- broad scheduler rewrites;
- independent SWA ownership;
- low-precision work;
- attention, MoE, or communication optimization;
- replacing Route B ownership rules without a correctness oracle;
- promoting a new default path before repeated serving gates pass;
- inventing a new component metadata architecture before comparing SGLang and
  vLLM source behavior.

## Design Tasks

### 1. Build the SGLang parity map

Create a concise table that maps:

- mini `table_indices`, `page_table`, `c4_page_table`, `c128_page_table`,
  `c4_indexer_page_table`, SWA/C4/C128 derived indices;
- SGLang `req_pool_indices`, `req_to_token`, `DSV4RawDecodeMetadata`,
  `DSV4AttnMetadata`, `PagedIndexerMetadata`, graph metadata refresh fields;
- Route B component-loc ownership structures in mini;
- any vLLM equivalent if SGLang does not have one.

For every mini field, mark:

- stable for the request lifetime;
- changes only on new request/prefix hit/eviction;
- changes every decode token;
- must keep captured tensor address;
- safe to reference-assign per replay;
- generated inside graph or outside graph.

### 2. Choose the implementation route

Pick the smallest SGLang-aligned route that can attack the measured owner.

Candidate A: persistent component mapping table.

- Add a request/table-index keyed component page-table mapping analogous to
  SGLang's stable `req_to_token` mapping.
- Update rows only when a request is admitted, a prefix hit is installed, an
  active full page gains a component page, or a component is evicted/freed.
- Decode metadata should slice/gather this persistent mapping instead of
  reconstructing rows from cache handles every token.

Candidate B: raw metadata plus graph-prep contract.

- Keep per-step metadata small outside graph, following SGLang
  `DSV4RawDecodeMetadata`.
- Move the expensive, deterministic metadata preparation into captured graph
  buffers or captured graph work when feasible.
- Preserve captured tensor addresses for fields consumed by captured kernels,
  and reference-assign only fields proven safe by source comparison.

Candidate C: minimal dirty-row cache.

- Use only if A or B is too invasive for this target.
- Cache graph metadata rows by stable request/table slot and mark dirty on
  admission, prefix hit, component mapping change, eviction, or bucket change.
- This is a fallback/probe, not the preferred architecture.

The README must explain why the selected route is closer to SGLang/vLLM than a
from-scratch design.

### 3. Implement behind an opt-in

Keep any runtime change guarded by a new explicit environment flag or CLI flag.
The old Route B path must remain available as the oracle and rollback.

Prefer one focused implementation.  Do not combine multiple speculative
mechanisms unless the first one passes and the second is required to explain
the remaining owner.

### 4. Verify correctness and graph behavior

Run the existing guarded Route B correctness style:

- slot-pinned/same-layout prefix on/off checks;
- metadata consistency checks against eager Route B oracle;
- TP8 text smoke with page size 256;
- graph replay/eager counters for `[1,2,4,8,16]`;
- no stale component reads after prefix hit, eviction, or active full-page
  mapping updates.

Cross-slot generated-token equality is diagnostic only.  Do not block on batch
slot invariance unless text quality or metadata correctness fails.

## Required Comparisons

Use separate `torchrun` invocations per variant.

Compare at least:

- phase1 prefix on;
- Route B graph baseline;
- Route B direct C4 from TARGET 08.25/08.26;
- new 08.27 opt-in;
- optionally Route B direct SWA+C4+C128 as diagnostic.

Use:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Primary workload:

- `serving_mixed_112req_wave16`

Add these only after the primary workload is correct and readable:

- `prefix_multi_112req_wave16`
- `prefix_eviction_pressure_96req_wave16`
- `decode_ladder_bs16`

## Success Criteria

The target may recommend promotion to a later gate if:

- correctness and text smoke pass;
- graph replay remains zero-eager for selected buckets;
- component page-table construction/rebuild work drops by at least `50%` versus
  Route B baseline or Route B direct C4;
- `serving_mixed_112req_wave16` output throughput improves by at least `5%`
  over Route B direct C4, or decode prepare time moves close enough to phase1
  that the next bottleneck is clearly elsewhere;
- the implementation is clearly aligned with SGLang/vLLM source behavior.

Keep as experimental if:

- the design is correct but the macro gain is below the gate;
- it only helps one metadata field group;
- it needs another focused follow-up to remove the remaining owner.

Reject or stop if:

- the fix requires broad scheduler/lifecycle rewrites;
- source comparison shows SGLang solves this through a mechanism that is too
  different to port safely in this target;
- graph capture becomes unstable;
- stale component rows appear under prefix hit or eviction;
- the measured owner disappears under repeat runs.

## Deliverables

Create:

```text
performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

`DESIGN.md` must include:

- SGLang/vLLM parity map;
- selected route A/B/C and why;
- field lifetime table;
- captured-address versus reference-assign contract;
- dirty/update events if any persistent rows are introduced;
- rollback plan.

`README.md` must include:

- exact commands;
- git status summary;
- correctness/text/graph results;
- throughput table;
- decode prepare versus forward table;
- component page-table owner timing before/after;
- graph replay/eager table;
- final decision: promote to a gate, keep experimental, split a follow-up, or
  reject.

## Stop Rules

Stop and report instead of optimizing if:

- SGLang parity mapping is not complete enough to justify an implementation;
- the target drifts into low precision, SWA ownership, attention, MoE, or NCCL;
- implementation exceeds a small metadata-lifetime change;
- less than `2%` E2E opportunity remains after attribution;
- repeated runs disagree on whether decode prepare is still the dominant owner.
