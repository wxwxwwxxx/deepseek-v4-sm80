# TARGET 08: DSV4 Radix Prefix Cache Roadmap

## Status

Closed as the current prefix-cache baseline, with active capacity follow-up
children:

```text
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
prompts/TARGET_08.32_dsv4_sm80_cuda_graph_private_pool_micro_attribution.md
prompts/TARGET_08.33_dsv4_sm80_indexer_capture_static_width_audit.md
prompts/TARGET_08.34_dsv4_sm80_moe_marlin_wna16_cache_lifecycle.md
```

TARGET 08.31 is about SGLang-aligned SWA lifecycle and memory ownership.  It is
not a low-precision target, but it should run before reopening TARGET 09.5
because it changes the real memory denominator for SWA-only FP8 cache.

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
3. the active future target prompt, currently TARGET 08.31, TARGET 08.34, or
   a TARGET 09 child
4. archived TARGET 08 prompts only when exact old commands or stop rules are
   needed

The archive contains implementation history, not active todos.

## Unfinished Or Deferred Items

### Independent SWA Ownership

Status: active follow-up as TARGET 08.31.

SWA KV is still effectively protected by the conservative full/SWA tail rule.
This costs retained memory and can reduce useful prefix capacity, but TARGET
08.30 did not show it as the active bottleneck for the measured `--num-pages
128` workloads.

TARGET 09.45 reopened this item from a capacity/low-precision ROI angle.  It
found that current mini's full 128-page SWA BF16 pool makes SWA-only FP8 appear
more valuable than it may be after SGLang-aligned lifecycle.  The next step is
therefore to prove independent SWA lifecycle and runtime SWA tail occupancy
before deciding whether TARGET 09.5 should implement FP8 cache.

Active prompt:

```text
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md
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

Status: active follow-up as TARGET 08.34 after the broad TARGET 08.32 probe and
the focused TARGET 08.33 indexer audit.

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

TARGET 08.34 should audit whether the default `marlin_wna16` MoE backend lazily
repacked and retained all routed expert weights on first forward.  Rough memory
math puts raw FP4 expert weights plus scales at about `0.398 GiB/layer/rank`,
or `~17.1 GiB/rank` for 43 layers, which is close to the observed warmup jump.
If confirmed, prebuild/account the cache before KV capacity planning and
evaluate opt-in original-weight release.

Old reports:

```text
performance_milestones/target08_cuda_graph_memory_attribution/README.md
performance_milestones/target08_bf16_cache_graph_memory_attribution/README.md
performance_milestones/target08_cuda_graph_private_pool_micro_attribution/README.md
performance_milestones/target08_indexer_capture_static_width_audit/README.md
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
