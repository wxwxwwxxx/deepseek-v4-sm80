# TARGET 08.25: DSV4 Route B Direct Graph Metadata Buffers

## Status

Run this after TARGET 08.24.

TARGET 08.24 proved the component-aware formula is correct, but the runtime
opt-in stayed experimental because performance regressed:

```text
performance_milestones/target08_route_b_metadata_deforest_copy_elision/
Decision: keep_experimental
```

The next target is not another attempt to tune the same temporary-metadata
path.  The target is to generate Route B decode metadata directly into the
CUDA-graph replay buffers that the captured graph consumes.

## Goal

Remove Route B's hot decode pattern:

```text
build eager source metadata tensors -> copy/stage them into graph buffers
```

Replace it, where safe, with:

```text
component-owned inputs + active rows -> final graph replay metadata buffers
```

The desired result is a Route B graph path that preserves TARGET 08.22
correctness/capacity behavior while materially reducing decode prepare time and
the large `c4_sparse_*`, `swa_page_indices`, and `c128_*` staging copies.

## Background

TARGET 08.24 result:

| mode | mean TTFT s | output tok/s | decode prepare s | saved prefill | graph replay/eager |
| --- | ---: | ---: | ---: | ---: | --- |
| phase1_prefix_on | 0.6981 | 64.8296 | 1.4436 | 183040 | 679/0 |
| route_b_graph_baseline | 0.7909 | 52.5981 | 7.3987 | 165376 | 679/0 |
| route_b_metadata_deforest | 0.7733 | 47.4662 | 13.2645 | 165376 | 679/0 |

The component-aware helper was safe, but it did not remove the dominant memory
traffic.  The largest measured metadata families remained:

- `c4_sparse_raw_indices`;
- `c4_sparse_page_indices`;
- `c4_sparse_full_indices`;
- `swa_page_indices`;
- `c128_raw_indices`;
- `c128_page_indices`;
- `c128_full_indices`.

The old 08.24 opt-in mostly changed how these tensors are generated.  It still
materialized source tensors and then staged them into graph buffers.  It also
added component-table write-loc generation, which made long-output and
large-wave scenarios worse.

Therefore this target should focus on final-buffer generation and row reuse,
not on expanding the 08.24 temporary-output helper.

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.24_dsv4_sm80_route_b_metadata_deforest_copy_elision.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/README.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/DESIGN.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/summaries/metadata_build_bytes.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/summaries/replay_copy_bytes.md`
- `performance_milestones/target08_route_b_metadata_deforest_copy_elision/summaries/deforest_effect.md`
- `performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/README.md`

Mini code references:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`

External references for behavior comparison:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`

## Scope

Allowed:

- add graph-buffer-specific metadata generation kernels;
- extend graph replay buffer ownership/lifetime bookkeeping;
- update only active or dirty rows in graph metadata buffers;
- reuse stable per-request/per-hit rows across decode steps;
- generate `c4_sparse_*`, `swa_page_indices`, `c128_*`, and component write
  locs directly into destination graph buffers;
- keep the new path behind a Route B opt-in until a gate proves it.

Not allowed:

- independent SWA ownership or SWA reconstruction;
- low-precision KV/cache/projection work;
- attention algorithm tuning unrelated to metadata staging;
- default promotion without a separate gate;
- treating cross-slot generated-token equality as pass/fail.

## Design Requirements

### Final-Buffer Generation

The new path should write directly into the captured graph's metadata buffers:

- no large eager `src_core` tensor is needed for fields that can be generated
  from compact inputs;
- destination tensors are the authoritative replay inputs;
- missing/tombstoned component pages write `-1` and never fall back to
  `full_loc // ratio`;
- graph replay remains valid for buckets `[1,2,4,8,16]`.

Prioritize direct generation for the largest fields first:

1. `c4_sparse_raw_indices`
2. `c4_sparse_page_indices`
3. `c4_sparse_full_indices`
4. `swa_page_indices`
5. `c128_raw_indices`
6. `c128_page_indices`
7. `c128_full_indices`

### Stable Row Reuse

Classify metadata by lifetime and avoid refreshing stable rows every token:

- per-token: positions, raw out locs, sequence lengths, current SWA/C4 lengths;
- per-request: page tables and component page tables;
- per-prefix-hit: C128 prefix metadata and retained component mapping;
- per-bucket: static graph buffers and `cu_seqlens_q` shape.

At minimum, investigate whether per-request page tables and component page
tables can be installed once when a request enters a graph bucket, then updated
only when a request slot changes.

### Source Metadata Compatibility

Do not remove the existing eager Route B metadata path.  Keep it as:

- an oracle for tests;
- a fallback when direct graph-buffer generation is unavailable;
- a debug path for comparing direct destination buffers against source metadata.

The 08.24 `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1` path must remain
experimental unless this target explicitly supersedes it with better evidence.

## Implementation Plan

### Phase 0: Row Lifetime And Buffer Map

Produce a short design note that maps:

- where each graph replay metadata buffer is allocated;
- which buffers are read by captured attention/indexer/compressor code;
- which fields currently require source metadata copies;
- which fields can be generated from `positions`, `raw_out_loc`,
  `req_table_indices`, full page tables, and component page tables;
- which fields must remain eager fallback for now.

### Phase 1: Direct Destination Kernel Prototype

Implement a narrow prototype for one high-impact group first.

Recommended first group:

```text
c4_sparse_raw_indices
c4_sparse_page_indices
c4_sparse_full_indices
```

The prototype should:

- write directly to `dst_core.c4_sparse_*`;
- consume component-owned page tables under Route B;
- keep full-index output correct for tombstoned full pages;
- avoid allocating source C4 sparse tensors in the graph-replay hot path when
  the direct path is enabled.

Only after the C4 prototype passes correctness and shows lower copy/build
overhead should the target extend to SWA or C128.

### Phase 2: Expand To SWA And C128

If Phase 1 is beneficial, extend direct generation to:

- `swa_page_indices`;
- `c128_raw_indices`;
- `c128_page_indices`;
- `c128_full_indices`;
- component write locs, only if this does not add net overhead.

C128 is per-prefix-hit sized; prefer row reuse or dirty-row update over
regenerating the whole C128 table every decode token.

### Phase 3: Dirty Row / Slot Reuse

Add the minimum graph metadata state needed to skip stable-row updates:

- detect unchanged request-to-slot mapping;
- mark slots dirty when a new request enters, a prefix hit changes, or a graph
  bucket row is reassigned;
- update only dirty rows for per-request/per-hit metadata;
- always update per-token metadata for active rows.

Do not build a large new scheduler abstraction unless a small local owner is
insufficient.

### Phase 4: Gate

Compare:

- phase1 prefix on;
- Route B graph baseline from TARGET 08.24;
- Route B direct graph metadata opt-in;
- optionally the 08.24 component-aware deforest opt-in as a negative control.

Required scenarios:

- `decode_ladder_bs16`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`;
- `prefix_eviction_pressure_96req_wave16`;
- exact page multiples and neighbors from TARGET 08.22;
- TP8 text smoke with graph buckets `[1,2,4,8,16]`.

Use separate `torchrun` invocations per variant.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_direct_graph_metadata_buffers/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- final-buffer generation design;
- row lifetime and dirty-row policy;
- byte/call reduction table for metadata build and replay copy;
- correctness table against the eager Route B oracle;
- text-smoke table;
- graph replay/eager table;
- serving A/B table;
- final decision: keep experimental, make Route B default opt-in path, or
  reject.

## Success Criteria

Treat this target as successful if:

- correctness/text/graph gates stay green;
- Route B saved-prefill behavior does not regress versus TARGET 08.24 baseline;
- graph replay remains `0` eager for buckets `[1,2,4,8,16]`;
- Route B decode prepare time drops at least `40%` versus TARGET 08.24 Route B
  baseline in large-wave scenarios, or output throughput reaches at least
  `0.90x` of phase1;
- no new stale read, double free, leak, or component lifecycle bug appears.

## Stop Rules

Stop and report instead of broadening the target if:

- direct generation for the C4 sparse group cannot beat the baseline;
- correctness depends on reconstructing SWA ownership;
- the implementation needs broad attention-kernel rewrites;
- dirty-row tracking becomes a scheduler rewrite;
- measured overhead is dominated by attention/MoE compute rather than
  metadata generation/copy.

