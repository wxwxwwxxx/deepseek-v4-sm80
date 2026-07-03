# TARGET 08: DSV4 Radix/SWA Prefix Cache

## Status

Active next target.

TARGET 07 is closed.  The promoted non-prefix path is stable enough to start
prefix-cache work:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
```

Start with fixed or capped page counts such as `--num-pages 128`.  Do not treat
automatic `memory_ratio=0.9` graph-mode capacity as a serving default yet:
TARGET 07.79 showed that it can choose a very large KV pool and OOM during graph
capture.

## Goal

Add correct, measurable radix prefix cache support for DeepSeek V4 Flash in
mini-sglang.

This target is complete when repeated requests with shared prefixes skip
cached-prefix prefill, produce the same decode results as prefix-disabled mode,
and release all DSV4 cache components correctly under eviction.

Primary value:

- reduce TTFT and prefill work for shared-prefix workloads;
- preserve decode graph replay and the promoted TARGET 07 path;
- establish a DSV4-aware cache ownership model before future low-precision or
  capacity work.

## Required vLLM/SGLang Alignment

Do not implement this from a blank page.  First inspect and map the relevant
cache-state designs.

Mini references:

- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`

vLLM references:

- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_coordinator.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/block_pool.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

SGLang references:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`

Old branch references, use carefully:

- `git show dsv4:python/minisgl/kvcache/deepseek_pool.py`
- `git show dsv4:tests/core/test_deepseek_prefix_cache.py`
- `git show dsv4:prompts/PLAN_tier3_correctness_milestone.md`
- `git show dsv4:prompts/PLAN_tier4_tilelang_sm80_milestone.md`

Useful vLLM ideas to compare against:

- hybrid KV cache coordination across groups with different block sizes;
- block-hash based prefix matching and eviction;
- `DeepseekV4SWACache` block sizing and alignment;
- `SlidingWindowMLASpec`;
- per-layer sparse/SWA metadata builders;
- fixed-point prefix hit length that all cache groups can support.

## DSV4 Cache Model

A prefix hit must represent more than ordinary full-token KV pages.

Track or reconstruct these components consistently:

- full token KV slots;
- SWA-visible slots/window boundary;
- C4 attention compressed slots;
- C128 attention compressed slots;
- C4 indexer slots;
- compression state or enough warmup information to rebuild it;
- logical token length;
- page-aligned prefix length;
- owner/refcount state needed for eviction.

Recommended design rule:

- make full-token pages the canonical prefix ownership unit;
- derive C4/C128/indexer component ownership from the same token range where
  possible;
- make prefix-cache pin/unpin use the same DSV4 component refcount machinery
  that normal allocation/free uses;
- avoid double-free by routing all final token release through one DSV4-aware
  owner path.

Compression state is the hardest part.  Phase 1 may choose correctness-first
reconstruction around the prefix boundary, for example by recomputing the small
ring/warmup state from cached tokens, instead of persisting every internal state
object in the radix tree.

## Plan

### Phase 0: Source Parity And Design Note

Produce a short note under:

```text
performance_milestones/target08_radix_prefix_dsv4/
```

It must answer:

- how mini currently matches prefix pages;
- how mini currently frees DSV4 full/C4/C128/indexer components;
- how vLLM coordinates SWA and compressed cache groups;
- what exact prefix state mini will store vs reconstruct;
- what lengths must be page-aligned or component-aligned.

Do not implement before this note exists.

### Phase 1: Minimal Explicit Opt-In

Add an explicit opt-in for DSV4 radix prefix cache.  Default behavior must stay
unchanged until correctness and eviction tests pass.

The minimal path should support:

- full prefix hit;
- partial prefix hit at safe page boundaries;
- prefix miss fallback;
- suffix prefill after a hit;
- decode after suffix prefill;
- eviction after requests finish.

### Phase 2: Correctness Tests

Add tests or smoke scripts for:

- full prefix hit;
- partial prefix hit;
- prefix miss;
- prefix eviction;
- multi-request shared system prompt;
- SWA boundary below, at, and above `window_size=128`;
- C4/C128 component boundary near compression ratios;
- repeated hit/evict cycles;
- logits or generated-token comparison against radix-disabled mode.

The first correctness gate may use small prompt/decode lengths, but it must use
the same DSV4 configuration style as normal usage, especially page size `256`.

### Phase 3: Metrics

Expose at least:

- prefix hit length;
- prefix hit rate;
- saved prefill tokens;
- retained prefix pages;
- retained or pinned DSV4 component slots;
- evicted prefix tokens/pages;
- suffix prefill tokens after hit;
- memory retained by prefix cache.

### Phase 4: Performance Gate

Run a shared-prefix benchmark that includes:

- repeated shared system prompt;
- at least one no-hit control;
- prefix cache on/off comparison;
- TTFT and prefill-forward delta;
- decode throughput sanity check;
- graph replay/eager decode status;
- memory-retention and eviction ledger.

Do not expect single-token decode throughput to improve.  The target is prefill
reuse and TTFT reduction on shared-prefix workloads.

## Done Criteria

- DSV4 radix prefix cache can be enabled explicitly.
- Shared-prefix requests skip cached-prefix prefill.
- Outputs match prefix-disabled mode for the tested prompts.
- SWA boundary behavior is correct around `128` tokens.
- C4/C128/indexer component ownership survives hit, suffix prefill, decode, and
  eviction.
- Eviction frees all DSV4 cache components without leaks or double-free.
- Metrics report hit rate, saved prefill tokens, retained memory, and evictions.
- A milestone README records the design, commands, correctness results,
  performance result, and remaining risks.

## Stop Rules

Stop and report blocked if:

- prefix hits corrupt generated text or logits versus radix-disabled mode;
- DSV4 component refcounting cannot be made unambiguous;
- SWA boundary reuse is not understood;
- graph replay is unexpectedly disabled by the feature;
- eviction leaks or double-frees any full/C4/C128/indexer component;
- automatic KV sizing causes graph-capture OOM during the target.

Stop after a correct opt-in and benchmark report.  Do not broaden into FP8 KV
cache, INT8 MoE, PyNCCL, or attention-kernel optimization inside TARGET 08.

## Non-Goals

- Changing the promoted TARGET 07 default path before correctness passes.
- Implementing a new eviction policy beyond correctness and basic memory
  control.
- Adding FP8 KV cache or low-precision cache changes.
- Optimizing C4A/C128A sparse attention kernels.
- Tuning NCCL or PyNCCL.
- Treating prefix cache as a replacement for decode kernel optimization.
