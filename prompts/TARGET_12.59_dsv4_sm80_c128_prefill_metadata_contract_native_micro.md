# TARGET 12.59: DSV4 SM80 C128 Prefill Metadata Contract And Native Micro

## Status

Run after TARGET 12.58. This is the current immediate target.

TARGET 12.58 promoted the bounded FP8 indexer path from TARGET 12.57 and
established the true release-default long-context envelope:

```text
524288 / 1 / bs1 / max_extend_tokens=8192: pass
1048576 / 1 / bs1 / max_extend_tokens=8192:
  fail while preparing chunk 90
  89 chunks / 729088 tokens committed
  first owner: C128 all-indices component metadata
```

At the failure boundary:

```text
rows:                8192
C128 width:          5760
one int32 surface:   180 MiB
one int64 surface:   360 MiB
failed allocation:   torch.full_like(locs, -1), int64, 360 MiB
```

The request was admitted with enough full/component pages, and independent SWA
retention stayed at two pages through the 512k pass. This is not a general KV
capacity or SWA lifecycle failure. It is a metadata/attention ABI problem.

## Purpose

Define the minimum C128 metadata contract for release eager prefill, compare it
with SGLang and vLLM, and build a no-weight native micro baseline that directly
produces the final int32 C128 component locations without full-matrix int64
intermediates.

This target must answer:

1. Which of `c128_raw_indices`, `c128_page_indices`, and
   `c128_full_indices` are actually consumed by release eager prefill?
2. Which fields are required only by decode graph capture, fallback/oracle,
   debug, or prefix validation?
3. Can release eager prefill follow a SGLang-like one-surface contract:
   `c128_page_indices + c128_lengths`?
4. Can a mini-owned Triton/CUDA helper generate that surface exactly with peak
   temporary memory bounded by the final int32 output plus small fixed state?
5. Is one complete int32 surface sufficient for the 1M integration target, or
   will query-row tiling/direct page-table consumption also be required?

Do not run the full 1M model in this target. The expensive integration and
promotion soak belongs to TARGET 12.595.

## Non-Goals

- Do not modify or re-enable MTP.
- Do not change BF16/FP8 precision policy.
- Do not change SWA independent or Route-B component ownership.
- Do not change `max_extend_tokens=8192`.
- Do not expand CUDA graph buckets.
- Do not optimize the FP8 indexer in this target.
- Do not rewrite C128 attention before proving the one-surface ABI is
  insufficient.
- Do not load full model weights for source census or the primary microbench.
- Do not remove fallback/debug semantics without an explicit lazy oracle path.

## References To Read First

Evidence:

```text
performance_milestones/target12_post_indexer_long_context_envelope/README.md
performance_milestones/target12_release_fallback_census_native_backend_gate/README.md
prompts/TARGET_12.58_dsv4_sm80_post_indexer_long_context_envelope.md
```

Mini source:

```text
python/minisgl/attention/deepseek_v4.py
  DeepSeekV4AttentionBackend._build_metadata
  _make_all_compressed_indices
  _compressed_raw_to_component_locs
  _compressed_raw_to_full_locs
  _sparse_attention_two_source
  _fallback_attention
  _context_metadata_for_queries

python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/engine/graph.py
```

SGLang reference:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
```

vLLM reference:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py
```

The references are design oracles, not code to copy blindly. Mini's Route-B
component page table differs from SGLang/vLLM cache ownership, so preserve the
mini component mapping while matching the mature final-metadata contract.

## Known Source Evidence To Verify

Current mini appears to do this for eager prefill:

```text
_make_all_compressed_indices
  -> materialize raw int32 [rows, width]
  -> materialize full int32 through a full-location gather
  -> convert raw/page intermediates to int64 matrices
  -> materialize component page int32
  -> return raw + page + full
```

The release two-source C128 attention fast path appears to consume only:

```text
c128_page_indices int32
c128_topk_lengths_clamp1 int32
c128_cache
```

`c128_raw_indices` and `c128_full_indices` appear to remain useful for some
decode graph, fallback, context-oracle, debug, and replay-copy paths. The
target must produce a phase/backend consumer table before changing allocation.

SGLang normally materializes one int32 `c128_page_indices` surface from page
metadata. It does not require mini's eager-prefill raw/page/full plus int64
intermediate family. Use this as the first baseline.

## Work Items

### 1. Build A Field Consumer And Lifetime Census

For each field, identify producer, consumer, phase, backend, dtype, shape, and
required lifetime:

```text
c128_topk_lengths_clamp1
c128_raw_indices
c128_page_indices
c128_full_indices
c128_page_table
c128_out_loc
```

At minimum distinguish:

```text
release eager prefill fast attention
release decode CUDA graph
eager decode
torch/reference attention fallback
debug/owner validation
prefix-cache validation/materialization
test-only oracle
```

The report must say whether each field is:

```text
mandatory final input
derivable in kernel
lazy fallback-only
debug-only
fixed-address graph input
safe empty placeholder in eager prefill
```

Do not infer this only from dataclass membership. Trace actual reads and
dispatch guards.

### 2. Compare Mini, SGLang, And vLLM Contracts

Produce a compact parity table:

```text
framework
prefill C128 metadata inputs
materialized outputs
dtype
query-row dimension
cache ownership/layout
attention backend consumer
workspace/lifetime policy
long-context bounding or tiling
```

Answer explicitly:

- Does SGLang build one C128 page-index surface or multiple equivalent
  surfaces for the release prefill path?
- Does vLLM precompute per-query C128 indices, gather one row per request, or
  derive locations inside its prefill path?
- Which behavior can be adapted without changing mini's Route-B ownership?
- Can mini preserve decode graph raw/full fields while eliding them only for
  eager prefill?

Prefer adapting the narrow mature mechanism over inventing a new global cache
contract.

### 3. Write The C128 Eager-Prefill Contract

Record the contract in the report and concise source comments/tests:

```text
For a supported release eager-prefill fast path:
  input:
    component_page_table [rows, component_pages] int32
    c128_lengths [rows] int32
    component_page_size
  mandatory output:
    c128_page_indices [rows, aligned_width] int32
    invalid tail = -1
  no mandatory eager-prefill output:
    c128_raw_indices full matrix
    c128_full_indices full matrix
  forbidden implementation detail:
    full [rows, width] int64 intermediates
```

Fallback/debug code may request raw/full explicitly, but that must be a lazy,
named oracle path. It must not silently run in the true release fast path.

Keep decode graph behavior unchanged in this target unless a focused test
proves the same helper can be adopted without changing fixed-address capture
contracts.

### 4. Implement A Native One-Surface Micro Helper

Implement or adapt a Triton/CUDA helper that directly writes final component
locations:

```text
logical raw index j
logical component page = j // component_page_size
offset = j % component_page_size
physical component page = component_page_table[row, logical component page]
output = physical component page * component_page_size + offset
valid iff j < c128_length and physical page is valid
invalid output = -1
```

Requirements:

- output dtype is int32;
- no full-size int64 tensor;
- no Python loop per row or per C128 token;
- no `.tolist()`, `.item()`, or CPU synchronization in the hot helper;
- support non-uniform lengths and invalid page-table entries;
- preserve alignment/padding expected by current attention;
- expose a backend marker for focused tests/diagnostics;
- keep the current torch construction as an explicit oracle.

The helper may remain behind an explicit integration toggle or be called only
by tests in this target. Do not promote it into release metadata construction
until TARGET 12.595.

### 5. No-Weight Microbench And Memory Ledger

Create a reusable focused harness under an appropriate benchmark/debug/test
directory. Do not put durable harness code only under `/tmp`.

Cover at least:

```text
rows: 1, 16, 1024, 8192
width: 512, 2048, 4096, 5760, 8192
uniform and ragged lengths
component page size used by DSV4 page_size=256 / C128
valid, invalid, and partially missing page-table rows
```

For small/medium shapes, compare the full tensor exactly with the torch oracle.
For the largest shapes, use a bounded/chunked oracle or sampled exact rows so
the oracle itself does not recreate the known OOM.

Record:

```text
runtime median/range after warmup
output bytes
peak allocated/reserved delta
driver free-memory delta
temporary bytes beyond final output
kernel launch count
exact mismatch count
```

The primary pass gate is:

```text
temporary full-matrix int64 bytes = 0
native output exact to oracle
peak live growth approximately final int32 output + small fixed overhead
```

### 6. Decide Whether One Surface Is Enough

At the 1M target shape, account for:

```text
8192 * 8192 * 4 bytes = 256 MiB final page-index surface
```

Combine this with TARGET 12.58's observed free/reserved memory. Decide:

```text
ONE_SURFACE_INTEGRATION_READY
ONE_SURFACE_PLUS_PREALLOCATED_WORKSPACE_REQUIRED
QUERY_ROW_TILING_REQUIRED
ATTENTION_PAGE_TABLE_ABI_REQUIRED
```

Do not claim one-surface is sufficient merely because the first 360 MiB
allocation disappears. Account for all simultaneously live final tensors.

If query tiling or a direct page-table attention ABI is required, provide the
smallest focused TARGET 12.595 design; do not begin a broad attention rewrite
inside this target.

### 7. Validation

Run syntax/lint and focused tests. At minimum cover:

```text
exact native versus oracle component locations
ragged C128 lengths
invalid `-1` page entries
C128 boundary lengths around 127/128/129
multiple rows with different component page tables
large-shape allocation contract
existing DSV4 wrapper/metadata tests
```

Do not require full weights. A very small partial-model probe is allowed only
if source and micro evidence cannot prove the consumer contract.

## Termination Conditions

Stop when one of these is true:

1. The consumer contract is proven, the native one-surface helper is exact,
   and TARGET 12.595 has a clear integration boundary.
2. Source parity proves the final C128 all-query surface is unnecessary and a
   direct page-table attention ABI is the smaller correct design; document the
   replacement target.
3. The helper cannot support Route-B ownership without changing cache
   correctness; identify the precise contract blocker.
4. The remaining issue requires full-model integration or a 1M soak; defer it
   to TARGET 12.595 rather than running it here.

Do not optimize the FP8 indexer, Marlin, graph buckets, or unrelated metadata
after the C128 micro decision is complete.

## Output

Write:

```text
performance_milestones/target12_c128_prefill_metadata_contract_native_micro/README.md
```

Required sections:

- decision and exact next integration route;
- commit/dirty-state summary;
- mini/SGLang/vLLM parity table;
- field consumer/lifetime table;
- written eager-prefill C128 contract;
- native helper design and changed files;
- microbench correctness/performance/memory table;
- 1M final-surface memory ledger;
- explicit TARGET 12.595 recommendation.

