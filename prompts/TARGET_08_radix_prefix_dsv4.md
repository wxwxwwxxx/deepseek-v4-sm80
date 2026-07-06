# TARGET 08: DSV4 Radix Prefix Cache Roadmap

## Status

Closed as the current prefix-cache baseline, with active capacity follow-up
children:

```text
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
prompts/TARGET_08.41_dsv4_sm80_swa_independent_lifecycle_promotion_soak.md
prompts/TARGET_08.42_dsv4_sm80_swa_large_capacity_serving_correctness.md
prompts/TARGET_08.32_dsv4_sm80_cuda_graph_private_pool_micro_attribution.md
prompts/TARGET_08.33_dsv4_sm80_indexer_capture_static_width_audit.md
prompts/TARGET_08.34_dsv4_sm80_moe_marlin_wna16_cache_lifecycle.md
prompts/TARGET_08.35_dsv4_sm80_marlin_wna16_release_preset_promotion.md
prompts/TARGET_08.36_dsv4_sm80_marlin_wna16_release_correctness_attribution.md
prompts/TARGET_08.37_dsv4_sm80_marlin_wna16_release_storage_reuse_owner.md
prompts/TARGET_08.38_dsv4_sm80_marlin_wna16_safe_release_arena_capacity.md
prompts/TARGET_08.39_dsv4_sm80_marlin_wna16_old_address_root_cause.md
prompts/TARGET_08.40_dsv4_sm80_marlin_wna16_release_component_clear_promotion.md
```

TARGET 08.31 is about SGLang-aligned SWA lifecycle and memory ownership.  It is
not a low-precision target, but it should run before reopening TARGET 09.5
because it changes the real memory denominator for SWA-only FP8 cache.

TARGET 08.41 is the promotion/soak pass after TARGET 08.31.  TARGET 08.31
implemented opt-in SWA independent lifecycle and proved large auto-capacity
potential with Marlin release, but fixed-128 serving uses a conservative SWA
floor and short offline E2E throughput regressed.  TARGET 08.41 should run
serving/prefix soak with graph buckets `[1,2,4,8,16]`, attribute overhead, and
decide whether SWA independent lifecycle should be promoted or remain opt-in.

TARGET 08.42 is the correctness target after TARGET 08.41.  TARGET 08.41
showed fixed-128 SWA independent serving is clean, but Marlin release + SWA
independent crashes with CUDA illegal memory access at large capacity, including
an explicit `--num-pages 4096` cap.  TARGET 08.42 should use no-weight and
partial repros first, then full-model confirmation, to fix the large-capacity
SWA serving bug before any promotion or FP8 cache work continues.

TARGET 08.32 is about CUDA graph private-pool memory attribution.  It should
avoid full model weight loading at first and instead use synthetic/partial
decode graph probes to explain the `~19 GiB/rank` graph capture cost.

TARGET 08.33 is the focused follow-up after TARGET 08.32.  TARGET 08.32 ruled
out many synthetic owners but did not explain the full-model cost.  TARGET
08.33 audits the real DSV4 C4 indexer logits capture width, especially whether
`page_table.shape[1] * page_size` accidentally over-expands a mini token-slot
table by `256x`.

TARGET 08.34 is the focused follow-up after TARGET 08.33.  TARGET 08.33
falsified the indexer-width hypothesis, but its stage ledger showed the large
memory jump happens during warmup `model.forward()` before the actual
`torch.cuda.graph` block.  TARGET 08.34 audits whether MoE Marlin WNA16 expert
weight repack is lazily creating about `17-18 GiB/rank` of persistent backend
state, and whether that cache should be prebuilt and accounted before KV
capacity planning.

TARGET 08.35 is the promotion gate after TARGET 08.34.  TARGET 08.34 proved
that prebuild fixes lifecycle/accounting and that releasing original routed FP4
expert weights can recover about `17.13 GiB/rank`.  TARGET 08.35 should turn
prebuild+release into a named high-memory-efficiency preset, not a loose manual
opt-in, and prove fail-closed backend semantics plus correctness/performance
gates.

TARGET 08.36 is the correctness follow-up after TARGET 08.35 rejected
promotion.  TARGET 08.35 landed the preset naming, release ledger, and
fail-closed behavior, but TP8 text smoke showed stable corrupted output only
after raw routed FP4 expert weights/scales were released.  TARGET 08.36 should
attribute that blocker before any release preset is promoted.

TARGET 08.37 continues the release route after TARGET 08.36.  TARGET 08.36
ruled out prebuild, prepacked branch, normal attribute deletion, graph replay
as primary cause, and sampled Marlin cache corruption.  Its strongest evidence
points to early physical release of large expert-weight storages interacting
with later KV/cache/warmup/graph/attention/indexer allocations.  TARGET 08.37
should identify the concrete storage-reuse owner or safe release boundary.

TARGET 08.38 is the repair target after TARGET 08.37.  TARGET 08.37 found that
the immediate-release failure is triggered when DSV4 KV/component pools reuse
released raw expert-weight ranges.  TARGET 08.38 should implement a safe
release arena / capacity-planning policy that keeps live KV/component buffers
off unsafe released ranges while preserving meaningful KV headroom.

TARGET 08.39 is the root-cause target after TARGET 08.38.  TARGET 08.38 proved
that a `3.1875 GiB/rank` guard arena can make `before_kv_alloc` release useful
and correctness-clean, but it did not identify the owner that makes unguarded
reuse unsafe.  TARGET 08.39 should map that guard to source expert ranges,
poison old expert regions, bisect stages/layers/kernels, and try to fix the
underlying stale-address, uninitialized-read, stream-lifetime, or OOB issue so
KV/component tensors can safely use the recovered raw-expert capacity.

TARGET 08.40 is the promotion target after TARGET 08.39.  TARGET 08.39 found
that the release bug is an uninitialized DSV4 component-cache read after
allocator reuse of old raw expert ranges, and that component-slot clear on page
allocation makes unguarded release pass.  TARGET 08.40 should productionize
that fix, add regression coverage, run macro/prefix/serving gates, and decide
whether to promote the release preset.

Milestone tag:

```text
dsv4-sm80-prefix-routeb-lifetime-baseline
```

Promoted prefix preset:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
```

Runtime shape for this baseline:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--page-size 256
--num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Prefix cache remains an explicit opt-in for serving because no-hit workloads
still pay overhead.  It should nevertheless be treated as an important feature
baseline for future optimization, especially when touching scheduler, cache,
metadata, graph buffers, eviction, or DSV4 component ownership.

## Outcome

TARGET 08 added a working DSV4 radix prefix cache and evolved it from a simple
full-page owner into a Route B component-ownership path with SGLang-aligned
metadata lifetime caching.

The final TARGET 08.30 reprofile showed:

- text smoke/verifier passed for the promoted prefix preset;
- CUDA graph replay stayed zero-eager for the measured buckets;
- shared-prefix workload `prefix_multi_112req_wave16` improved from
  `51.0507` to `110.1417` output tok/s and saved `49152` prefill tokens;
- historical no-hit workloads stayed near the TARGET 07 control, for example
  `4096/1024/bs4` was `137.1625` versus `139.8415` output tok/s;
- no-hit serving-mixed workload showed visible opt-in overhead:
  `163.3985` versus `178.3004` output tok/s;
- the main remaining bottleneck moved away from prefix metadata and back toward
  decode forward plus communication/all-reduce owners.

This is enough to close TARGET 08 as a functional/performance milestone.  Do
not keep polishing prefix metadata unless a fresh profile shows it is again a
top bottleneck.

## Historical Evolution

### Phase 1: Conservative Radix Prefix Cache

The first implementation added:

- explicit `--enable-dsv4-radix-prefix-cache`;
- `page_size % 128 == 0` guard, with target runs using page size `256`;
- full-token pages as the canonical prefix ownership unit;
- hit/miss/eviction metrics and shared-prefix smoke/perf gates.

It proved prefill reuse and TTFT wins, but retained full DSV4 pages for cached
prefixes.  That made the design simple and correct, but expensive in retained
SWA/full-page memory.

Key artifact:

```text
performance_milestones/target08_radix_prefix_dsv4/
```

### Phase 2: Serving Graph And Memory Baseline

TARGET 08.05-08.10 established the serving test shape:

- serving-style workload suite;
- graph buckets `[1,2,4,8,16]`;
- CUDA graph memory attribution;
- BF16 cache graph-memory audit;
- controlled prefix-cache serving gate.

The important decision was to keep prefix cache opt-in until correctness and
ownership became cleaner.

### Phase 3: Correctness Boundary

TARGET 08.19-08.198 investigated prefix-on/off correctness.

The key conclusion was subtle: mini does not currently guarantee batch-slot
invariance.  Identical logical prompts can produce tiny logit drift across
different slots/page layouts, and that drift may change sampled tokens when
logit margins are small.  Prefix correctness probes should therefore use:

- text smoke that rejects obvious corruption, invalid text, crashes, leaks, and
  cache-state damage;
- slot-pinned or same-layout comparisons when comparing logits;
- generated-token equality only as a diagnostic, not as a broad oracle.

One real bug was fixed in this phase: compressor cross-request pooling / state
coupling.  Later residual drift was classified as shape/layout numeric drift,
not direct evidence that prefix metadata was corrupt.

### Phase 4: Route B Component Ownership

The first SWA/component-retention V1 idea was rejected and left fail-closed.
It was unsafe because several DSV4 component locations were still derived from
released full-token pages.

Route B became the main design:

- keep the radix tree as the prefix structure;
- give DSV4 compressed/indexer/component pages independent ownership;
- stop deriving C4/C128/indexer locations from full-token pages after those
  pages may be released;
- keep SWA/full tail conservative until separate evidence justifies deeper SWA
  ownership work.

TARGET 08.21.1-08.21.4 split Route B into bounded steps: table preflight,
independent compressed/indexer ownership, compression-state ownership, and graph
deforest/serving integration.

### Phase 5: Metadata Deforest And Lifetime Cache

Route B initially recovered correctness but exposed decode-prepare overhead.
Several direct metadata experiments were tried:

- component-aware metadata deforest;
- direct graph metadata buffers;
- remaining-gap attribution reset.

The useful result was not a broad direct-generation rewrite.  The winning idea
was SGLang-aligned metadata lifetime: component page tables are stable for a
request/table slot and should not be rebuilt on every decode replay step.

TARGET 08.27 added request-slot keyed component page-table lifetime caching.
TARGET 08.28 promoted it after verifier/text/eviction/prefix_multi/decode
controls.  TARGET 08.29 cleaned it into the promoted preset.

### Phase 6: Post-Prefix Reprofile

TARGET 08.30 reprofiled the promoted prefix preset and found:

- prefix cache is valuable on shared-prefix workloads;
- prefix metadata/runtime is no longer the first bottleneck;
- no-hit prefix overhead exists and justifies keeping prefix cache opt-in;
- next evidence-based work should focus on decode-forward communication and
  all-reduce owners before broad low-precision work.

Key artifact:

```text
performance_milestones/target08_post_prefix_reprofile/
```

## Current Default Baselines

Non-prefix exact-ish baseline:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
--page-size 256
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Prefix baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Use the prefix baseline for future work that touches:

- radix/prefix cache;
- DSV4 component ownership;
- eviction;
- scheduler cache allocation/free;
- attention metadata;
- graph metadata buffers;
- graph bucket policy.

For unrelated compute-kernel work, a fast non-prefix smoke or macro is usually
enough during development.  Run the full prefix matrix before release-style
promotion.

## Archive

Completed TARGET 08 execution prompts live in:

```text
prompts/archive/target08/
```

New Codex threads should not read the full archive by default.  Start from:

1. `prompts/target.md`
2. this roadmap
3. the active future target prompt, currently TARGET 08.42, TARGET 08.40, or
   a TARGET 09 child
4. archived TARGET 08 prompts only when exact old commands or stop rules are
   needed

The archive contains implementation history, not active todos.

## Unfinished Or Deferred Items

### Independent SWA Ownership

Status: active follow-up as TARGET 08.42 after TARGET 08.41.

SWA KV is still effectively protected by the conservative full/SWA tail rule.
This costs retained memory and can reduce useful prefix capacity, but TARGET
08.30 did not show it as the active bottleneck for the measured `--num-pages
128` workloads.

TARGET 09.45 reopened this item from a capacity/low-precision ROI angle.  It
found that current mini's full 128-page SWA BF16 pool makes SWA-only FP8 appear
more valuable than it may be after SGLang-aligned lifecycle.  The next step is
therefore to prove independent SWA lifecycle and runtime SWA tail occupancy
before deciding whether TARGET 09.5 should implement FP8 cache.

TARGET 08.31 result: opt-in SWA independent lifecycle was implemented and
validated against SGLang's allocator/component model.  SWA KV now has separate
page lifecycle and can tombstone/free out-of-window tail pages without
invalidating C4/C128/indexer/compression-state/component locations.  Unit and
integration gates passed, graph replay remained valid for the tested buckets,
and Marlin WNA16 release + component-slot clear stayed compatible.  Runtime
counters showed live SWA tails of only about `4`, `18`, and `26` pages in the
fresh fixed-128 historical/serving/prefix runs.  The Marlin release auto
capacity path improved from `2776` pages to `6636` pages at roughly the same
KV memory budget.  However, fixed-128 macro/serving safe-floor planning still
keeps `128` SWA pages, and short offline E2E throughput regressed by about
`3%` to `9%`.  TARGET 08.41 should run the promotion soak and overhead
attribution before any default promotion.

TARGET 08.41 result: SWA independent lifecycle should not be promoted yet.
Fixed `--num-pages 128` serving/prefix/eviction runs are correctness-clean with
graph buckets `[1,2,4,8,16]`, but Marlin release + SWA independent
auto-capacity crashes under serving with CUDA illegal memory access.  The same
failure appears with explicit `--num-pages 4096`, so it is not merely an
auto-planner near-OOM artifact.  E2E overhead was attributed mainly to decode
prepare / attention metadata, but correctness takes priority.  TARGET 08.42
should fix the large-capacity SWA serving crash, using no-weight/partial repros
before full model runs.

Active prompt:

```text
prompts/TARGET_08.42_dsv4_sm80_swa_large_capacity_serving_correctness.md
```

Previous prompts:

```text
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
prompts/TARGET_08.41_dsv4_sm80_swa_independent_lifecycle_promotion_soak.md
```

Historical prompt:

```text
prompts/archive/target08/TARGET_08.23_dsv4_sm80_independent_swa_ownership.md
```

TARGET 08.31 should still respect the old guardrails:

- do not invalidate C4/C128/indexer/state locs when freeing SWA;
- avoid large hit-time materialization copies as the production design;
- keep Route B component loc ownership and lifetime-cache verification;
- preserve CUDA graph replay.

### Prefix Cache Default Promotion

Status: not promoted to default.

Prefix cache should stay explicit opt-in until no-hit overhead is either lower
or acceptable for the target serving product.  The promoted preset is suitable
as a benchmark and feature baseline, not necessarily as the default runtime
for all traffic.

### CUDA Graph Memory Pool

Status: active follow-up as TARGET 08.40 after the broad TARGET 08.32 probe,
the focused TARGET 08.33 indexer audit, TARGET 08.34 MoE lifecycle
attribution, TARGET 08.35 release-preset promotion gate, and TARGET 08.36
release-correctness attribution, and TARGET 08.37 storage-reuse owner
attribution, TARGET 08.38 safe-arena capacity validation, and TARGET 08.39
old-address root-cause attribution.

Graph capture delta is large and repeatable.  TARGET 08.06/08.07 found it is
not primarily caused by BF16 caches, metadata, bucket count, greedy sample,
`max_seq_len`, or `num_pages`.  Treat this as future runtime/capacity work, not
as a prefix-cache blocker.

TARGET 08.32 reopened this from a capacity/headroom angle without repeating the
old full-model A/B matrix.  It used no-weight and synthetic partial-model graph
probes.  Its important negative result was that simple graph overhead, BF16
matmul/cuBLAS workspace, synthetic SWA/C4/C128 attention, synthetic C4
indexer/topk, metadata helpers, and NCCL controls did not explain the
multi-GiB cost:

```text
prompts/TARGET_08.32_dsv4_sm80_cuda_graph_private_pool_micro_attribution.md
```

The first focused target was:

```text
prompts/TARGET_08.33_dsv4_sm80_indexer_capture_static_width_audit.md
```

TARGET 08.33 audited real full-model indexer logits call-site shapes and
falsified the indexer-width hypothesis.  It found the real C4/indexer width is
`128 * 64 = 8192`, which is expected and explains only about `0.010 GiB/rank`.

The more important TARGET 08.33 result was the stage ledger: the large
`~18 GiB/rank` movement appears after warmup `model.forward()`, while the
actual `torch.cuda.graph` capture block adds almost nothing.  The next focused
target is therefore:

```text
prompts/TARGET_08.34_dsv4_sm80_moe_marlin_wna16_cache_lifecycle.md
```

TARGET 08.34 audited the default `marlin_wna16` MoE backend and confirmed that
lazy routed-expert repack fully explains the warmup jump.  Prebuild moved the
large cost before KV capacity planning, and release of original routed FP4
expert weights recovered about `17.13 GiB/rank`, equivalent to about `400` DSV4
KV pages or `102k` tokens per rank at page size `256`.

The next focused target is:

```text
prompts/TARGET_08.35_dsv4_sm80_marlin_wna16_release_preset_promotion.md
```

TARGET 08.35 should make prebuild+release a named high-memory-efficiency
`marlin_wna16` preset rather than a loose opt-in, while proving correctness,
graph replay, macro performance, KV capacity accounting, and fail-closed
backend semantics.

TARGET 08.35 result: preset naming, env expansion, two-stage prebuild/release,
memory reporting, and fail-closed backend semantics landed, but promotion was
rejected because TP8 text smoke produced corrupted text only for the release
variant.  Baseline and prebuild-only text smoke passed; release recovered
`17.1328 GiB/rank` but failed correctness.  The next focused target is:

```text
prompts/TARGET_08.36_dsv4_sm80_marlin_wna16_release_correctness_attribution.md
```

TARGET 08.36 should not run broad macros until text sanity is fixed or the
release path is rejected.  It should isolate whether the failure belongs to
CUDA graph replay, decode progression, MoE packed-cache lifetime, raw tensor
storage release, or a runtime branch change after raw attributes are removed.

TARGET 08.36 result: release remains blocked, but the release route is still
worth investigating.  Baseline and prebuild-only pass.  Release fails in both
graph and eager/no-graph modes, so graph replay is not the primary owner.
`force-prepacked-with-raw-present`, `keep-hidden-ref`, and
`release-after-capture` pass; `weights-only` fails while `scales-only` passes;
partial release shows a failure threshold between about `3.1875` and
`6.3750 GiB/rank`.  Sampled Marlin packed-cache tensors and MoE micro-parity
remain stable.  The first observed full-model symptom is around
`layer2.indexer_select.logits`, but the likely root is early physical release
of large expert-weight storage and subsequent allocator reuse.  The next target
is:

```text
prompts/TARGET_08.37_dsv4_sm80_marlin_wna16_release_storage_reuse_owner.md
```

TARGET 08.37 should build a freed-range ledger, owner-tagged post-release
allocation ledger, release timing ladder, poison/quarantine probes, and a
layer2 indexer/attention owner probe to identify the concrete unsafe owner or
the earliest safe release boundary.

TARGET 08.37 result: the unsafe owner is the DSV4 KV/component allocation phase
after immediate model-prepare release.  Owner ledgers show `after_kv_alloc`
overlaps in `kvcache.dsv4.c4_buffer`, `c4_indexer_buffer`,
`c4_indexer_fp8_paged_cache`, `c128_buffer`, and per-layer
`compress_state` / `indexer_state` buffers.  Releasing after KV allocation
passes eager and graph smokes but is capacity-neutral.  Freed-block quarantine
passes, showing the issue follows allocator ownership of released ranges, not
raw weight contents.  The next target is:

```text
prompts/TARGET_08.38_dsv4_sm80_marlin_wna16_safe_release_arena_capacity.md
```

TARGET 08.38 should repair the release route by separating capacity accounting
from unsafe address reuse.  It should plan with a Marlin release credit while
using an arena/guard/allocation-order policy to keep live KV/component buffers
off unsafe released expert-weight ranges.

TARGET 08.38 result: a `before_kv_alloc` release with a `3.1875 GiB/rank`
deterministic guard arena passed short text smokes, graph replay, and the
historical 4096x128 / 4096x1024 macro shapes.  Auto-planned capacity improved
from `1,826` to `2,602` pages at page size `256`, a gain of `198,656` planned
tokens.  The guard is 32 tensors and, in the rank-0 safe-arena record, maps to
layers `0-7` and four raw expert components per layer.  This is not yet proof
that layers `0-7` are semantically special; it may reflect release-ledger or
allocator order.  The next target is:

```text
prompts/TARGET_08.39_dsv4_sm80_marlin_wna16_old_address_root_cause.md
```

TARGET 08.39 should stop treating the guard as the final answer.  It should
use old expert address traps, NaN/byte poison, KV-as-sentinel probes,
stage/layer bisection, and stream-lifetime controls to find the root cause.
The preferred success condition is unguarded early release passing while
KV/component tensors safely use the formerly raw expert-weight ranges.

TARGET 08.39 result: the release bug was attributed to uninitialized DSV4
component-cache reads after allocator reuse of old raw expert-weight storage.
`clear=component` passes, while `clear=none`, `clear=full`, and `clear=state`
do not.  `CUDA_LAUNCH_BLOCKING=1 + clear=none` still fails, making a
stream-lifetime race unlikely.  Fixed unguarded release passes eager and graph
text smokes, records `0` guard bytes, still lets KV/component owners overlap
old raw expert ranges, and auto-plans `2,779` pages at page size `256`.  The
next target is:

```text
prompts/TARGET_08.40_dsv4_sm80_marlin_wna16_release_component_clear_promotion.md
```

TARGET 08.40 should productionize the component-slot clear fix, add regression
tests, run macro/prefix/serving performance gates, measure page-allocation
clear overhead, and decide whether to promote the Marlin WNA16 raw-expert
release preset.

Old reports:

```text
performance_milestones/target08_cuda_graph_memory_attribution/README.md
performance_milestones/target08_bf16_cache_graph_memory_attribution/README.md
performance_milestones/target08_cuda_graph_private_pool_micro_attribution/README.md
performance_milestones/target08_indexer_capture_static_width_audit/README.md
performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/README.md
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
performance_milestones/target08_marlin_wna16_release_correctness_attribution/README.md
performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/README.md
performance_milestones/target08_marlin_wna16_safe_release_arena_capacity/README.md
performance_milestones/target08_marlin_wna16_old_address_root_cause/README.md
```

### Broader Serving Benchmark

Status: future release gate.

Before claiming serving readiness, run a broader pass with:

- multiple request rates and concurrency limits;
- shared-prefix and no-prefix mixes;
- short and long decode lengths;
- queueing latency, TTFT, ITL/TPOT, output throughput;
- GPU utilization and graph replay coverage;
- KV/prefix retained memory and eviction behavior.

## Relationship To Radix Cache

Prefix cache and radix cache are conceptually separate: prefix cache is the
feature, radix tree is one implementation.

In current mini-sglang DSV4 code they are intentionally bound:

- enabling DSV4 prefix cache requires `cache_type='radix'`;
- disabling DSV4 prefix cache falls back to the no-op `naive` cache;
- Route B component loc ownership requires `--enable-dsv4-radix-prefix-cache`;
- page size must remain 128-aligned, with page size `256` used for this project.

Future hash/block-prefix managers could decouple the concepts, but that is not
part of TARGET 08.

## References

Mini:

- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_coordinator.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/block_pool.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

Old branch, use carefully:

- `git show dsv4:python/minisgl/kvcache/deepseek_pool.py`
- `git show dsv4:tests/core/test_deepseek_prefix_cache.py`

## Non-Goals For Closed TARGET 08

- Adding FP8 KV cache or INT8 MoE.
- Tuning PyNCCL or all-reduce overlap.
- Rewriting attention kernels.
- Continuing broad prefix-cache polishing without fresh profile evidence.
- Promoting prefix cache to default for all traffic.
