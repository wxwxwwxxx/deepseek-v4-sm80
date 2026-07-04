# TARGET 08.20: DSV4 SGLang-Style SWA Tail Retention V1

## Status

Active next TARGET 08 subtarget.

TARGET 08.18 recommends GO for component retention with guardrails.  This target
is the first conservative implementation slice, but TARGET 08.19 found a DSV4
exact-path slot/page-location blocker.  TARGET 08.195 fixed a real compressor
cross-request pooling bug, TARGET 08.196 narrowed the remaining batched drift,
and TARGET 08.197 classified the layer0 q-path drift as GEMM shape numeric
drift rather than q_norm/RoPE row-coupling.  Start this target only after TARGET
08.198 fixes or guards the remaining post-layer0 same-shape decode drift, or
provides a stable oracle.  This target should remain behind a new opt-in and
must not replace phase-1 full-page-owner prefix cache by default.

TARGET 08.198 provided the required guard: mini does not currently guarantee
batch-slot invariance.  For this target, cross-slot/filler-content/identical-row
generated-token equality is diagnostic only, not a pass/fail oracle.  Use a
slot-pinned, same-layout prefix-on versus prefix-off comparison plus text smoke
that rejects obvious correctness failures such as garbled or invalid-byte text,
crashes, leaks, metadata corruption, or cache-state misuse.

## Motivation

TARGET 08.18 showed that phase-1 full-page prefix retention consumes logical KV
capacity quickly:

- sustained TARGET 08.10 workload: `56 / 128` pages retained, `1.007 GiB/rank`;
- eviction pressure: `112 / 128` pages retained, `2.015 GiB/rank`;
- the largest byte owners are full/SWA rows and compression state.

The goal is not to chase the aggressive "replay thousands of tokens to rebuild
SWA" idea.  That may be useful for very long contexts, but mini's serving target
does not require it yet.  Prefer the mature SGLang direction: tombstone
out-of-window SWA, keep a page-aligned SWA tail, and preserve compressed
components safely.

## Goal

Build a minimal, correctness-first opt-in that separates old full/SWA page
retention from reusable DSV4 compressed prefix components.

The V1 target should answer:

1. Can mini recover meaningful logical KV capacity without changing the
   promoted non-prefix path?
2. Can one page-aligned SWA tail per retained prefix branch be kept while older
   SWA/full pages are released or tombstoned?
3. What minimal component ownership model is needed for C4, C128, indexer, and
   compression state to survive after old full pages are no longer canonical?
4. Does the opt-in preserve slot-pinned/same-layout logits and metadata, text
   smoke quality, and TARGET 08.05 graph replay coverage?

## Source References

Mini:

- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/attention/deepseek_v4.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/common.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`

Key SGLang ideas to map:

- `SWAComponent` tombstones old SWA while full/compressed data can remain;
- `swa_evicted_seqlen` is page-aligned and defines the SWA eviction frontier;
- `free_swa_out_of_window_slots()` releases out-of-window SWA;
- `alloc_extend_swa_tail()` allocates SWA only for the tail in selected paths;
- component match must find a safe fixed point across all required components.

## Required Design Step

Before editing runtime behavior, write a short design note under the milestone
directory explaining the chosen V1 data model.

The note must explicitly answer:

- In current mini, C4/C128/indexer slots are derived from full locations.  If a
  full page is released and reused, how will retained compressed components
  avoid collision?
- Will V1 copy retained compressed components into a separate component store,
  add independent component refcounts, or choose a smaller safe subset?
- How is compression state retained, reconstructed, or deliberately not
  optimized in V1?
- What is the exact safe hit length rule when SWA tail, C4, C128, indexer, and
  compression state have different availability?

If no narrow implementation can answer these without a broad allocator rewrite,
stop with a design-level blocker and do not fake capacity recovery.

## Implementation Scope

Allowed:

- add a new explicit opt-in for V1 component retention;
- add component-level metrics and leak checks;
- add page-aligned SWA tail/tombstone metadata;
- add a minimal retained-component store or refcount path if required;
- keep phase-1 full-page-owner prefix cache as rollback;
- add targeted correctness and capacity tests.

Not allowed:

- default promotion;
- low-precision KV/cache;
- CUDA graph allocator redesign;
- graph private-pool attribution;
- PyNCCL or communication overlap changes;
- aggressive long replay of thousands of tokens to rebuild SWA.

## Required Tests

Correctness:

- full hit, partial hit, miss;
- page boundary `256`;
- SWA boundary `128`;
- C4 and C128 boundaries;
- mixed hit/miss batch;
- repeated hit/evict cycle;
- eviction pressure;
- guarded logits/metadata comparison:
  - main pass/fail: slot-pinned, same-layout prefix-on versus prefix-off;
  - diagnostics only: cross-slot, filler-content, and identical-row
    generated-token equality;
  - report max/mean logit diffs, top-k, top1 margin, and sampled ids;
  - treat sampled-token changes as unstable but expected when
    `2 * max_abs >= top1_margin`;
  - for decode step 1+, use teacher-forced/fixed-token probes where possible,
    or label natural autoregressive mismatch as sampler feedback once decode0
    sampled ids differ.
- text smoke:
  - shared-prefix and non-shared-prefix prompts;
  - no obvious garbled or invalid-byte artifacts;
  - no empty/degenerate output unless EOS is expected;
  - no crash, hang, or cache-state assertion failure.

Capacity and performance:

- retained full/SWA pages before and after V1;
- retained C4/C128/indexer/state slots;
- recovered logical KV pages/tokens;
- TTFT and prefill-forward on TARGET 08.10 shared-prefix workloads;
- graph replay/eager counts with `[1,2,4,8,16]`.

## Deliverables

Create:

```text
performance_milestones/target08_swa_tail_retention_v1/
  README.md
  DESIGN.md
  raw/
  scripts/
  summaries/
```

The README must include:

- SGLang source parity summary;
- mini V1 design and opt-in name;
- guarded correctness results;
- text-smoke results;
- recovered-capacity table;
- performance A/B versus phase-1 prefix cache;
- graph replay coverage;
- decision: keep V1 opt-in, reject, or prepare TARGET 08.21.

## Decision Rules

Keep V1 as an opt-in if:

- slot-pinned/same-layout logits and metadata pass for tested boundaries;
- text smoke shows no obvious garbled output, degenerate output, crash, or
  state misuse;
- no leaks or double frees;
- graph replay remains covered;
- recovered pages are material for sustained or eviction-pressure workloads.

Proceed to TARGET 08.21 only if:

- V1 proves the component split is correct;
- capacity savings are meaningful;
- the remaining limitations are clearly due to conservative V1 scope rather
  than fundamental mismatch with mini's architecture.

## Stop Rules

Stop and report blocked if:

- compressed components cannot outlive released full pages safely;
- compression state ownership cannot be made unambiguous;
- slot-pinned/same-layout logits diverge from phase-1 prefix cache or
  prefix-disabled mode beyond the guarded oracle;
- text smoke shows garbled output, degenerate output, or repeated invalid-byte
  artifacts;
- recovered capacity is below the noise floor;
- implementation starts turning into a global KV allocator rewrite.

## Non-Goals

- Complete SGLang unified cache parity.
- Default prefix-cache promotion.
- Low-precision research.
- Attention-kernel optimization.
