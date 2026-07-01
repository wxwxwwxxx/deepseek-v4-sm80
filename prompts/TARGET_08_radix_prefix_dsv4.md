# TARGET 08: DSV4 Radix Prefix Cache v2

## Goal

Add radix prefix cache support for DSV4 after the non-prefix DSV4 path is correct and benchmarked.

This target is complete when repeated requests with shared prefixes skip cached-prefix prefill, produce the same decode results as prefix-disabled mode, and release all DSV4 cache components correctly under eviction.

## Primary References

- Current local radix implementation:
  - `python/minisgl/kvcache/radix_cache.py`
  - `python/minisgl/scheduler/cache.py`
  - `python/minisgl/scheduler/scheduler.py`
- DSV4 memory pool reference:
  - `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
  - `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- SGLang SWA/unified cache ideas:
  - `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
  - `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`

## Old dsv4 Branch References

Use these carefully:

- `git show dsv4:python/minisgl/kvcache/deepseek_pool.py`
- `git show dsv4:tests/core/test_deepseek_prefix_cache.py`
- `git show dsv4:prompts/PLAN_tier3_correctness_milestone.md`
- `git show dsv4:prompts/PLAN_tier4_tilelang_sm80_milestone.md`

The old branch may contain useful prefix-cache tests, but this target must integrate with the new DSV4 KV pool from TARGET 03.

## Plan

1. Design a DSV4 prefix entry.
   - A prefix hit must represent more than ordinary token slots.
   - Store or reference:
     - full token slots
     - SWA slots
     - C4 attention slots
     - C128 attention slots
     - C4 indexer slots
     - compress state
     - logical token length
   - Include enough metadata to rebuild TARGET 04 attention metadata after a hit.

2. Add DSV4-aware radix matching.
   - Reuse token-prefix matching from current radix cache.
   - On hit, set cached length and only prefill the suffix.
   - On miss, behave like the no-prefix path.
   - Partial hit must be supported.

3. Handle SWA correctly.
   - Prefix cache must not free SWA-relevant slots too early.
   - Attention metadata after a prefix hit must still expose the latest `window_size=128` tokens correctly.
   - Eviction must update all DSV4 cache components together.

4. Add reference counting and eviction.
   - Prefix entries should own or retain references to all DSV4 cache components.
   - Evicting a prefix must free all associated component slots.
   - Active requests must pin prefix entries they use.

5. Add metrics.
   - Prefix hit length.
   - Prefix hit rate.
   - Saved prefill tokens.
   - Evicted prefix tokens.
   - DSV4 component memory retained by prefix cache.

6. Add correctness tests.
   - Full prefix hit.
   - Partial prefix hit.
   - Prefix miss.
   - Prefix eviction.
   - Multi-request shared system prompt.
   - SWA boundary around lengths below, equal to, and above 128.
   - Compare logits/decode tokens with radix disabled.

7. Add performance test.
   - Repeated shared system prompt should show lower TTFT.
   - Decode throughput should not be expected to improve much.
   - Memory usage should increase or remain retained longer due to cached prefixes.

## Done Criteria

- DSV4 radix prefix cache can be enabled explicitly.
- Shared-prefix requests skip cached-prefix prefill.
- Outputs match prefix-disabled mode.
- Eviction frees all DSV4 cache components without leaks.
- Metrics expose hit rate, saved prefill tokens, and retained memory.

## Non-Goals

- Implementing prefix cache before TARGET 03 and TARGET 04 are stable.
- Assuming radix prefix improves single-token decode kernel speed.
- Optimizing eviction policy beyond correctness and basic memory control.
