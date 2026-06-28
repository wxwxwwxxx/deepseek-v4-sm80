# TARGET 03: DSV4 KV Cache v1 Without Radix Prefix

## Goal

Implement a DSV4-specific KV cache pool that can support SWA, C4, C128, indexer cache, and compress state, while keeping radix prefix cache disabled for the first working version.

This target is complete when DSV4 prefill/decode can allocate, write, read, and free all required cache components without slot leaks.

## Primary References

- Main memory pool reference: `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- Compress state reference: `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- SGLang KV mixin reference: `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- Current local KV code:
  - `python/minisgl/kvcache/base.py`
  - `python/minisgl/kvcache/mha_pool.py`
  - `python/minisgl/kvcache/naive_cache.py`
  - `python/minisgl/kvcache/radix_cache.py`
  - `python/minisgl/scheduler/cache.py`

## Old dsv4 Branch References

Use these for allocation ideas and tests:

- `git show dsv4:python/minisgl/kvcache/deepseek_pool.py`
- `git show dsv4:tests/core/test_deepseek_prefix_cache.py`
- `git show dsv4:tests/core/test_cache_allocate.py`

Do not inherit old branch prefix behavior in this target. Prefix cache stays disabled here.

## Plan

1. Add a DSV4 KV pool type.
   - The pool must not assume ordinary MHA `k_cache` and `v_cache`.
   - Represent DSV4's latent KV as one logical cache where appropriate.
   - Include separate storage for:
     - full token slots
     - SWA cache
     - C4 attention cache
     - C128 attention cache
     - C4 indexer cache
     - compress state

2. Implement layer-to-cache mapping.
   - Use `compress_ratios` from config.
   - Ratio `0` layers use normal/SWA behavior.
   - Ratio `4` layers use C4 paths.
   - Ratio `128` layers use C128 paths.
   - Store enough mapping metadata for TARGET 04 attention metadata construction.

3. Implement allocation and release.
   - Start with naive contiguous or page-based allocation consistent with current mini-sglang scheduler.
   - Maintain mappings from logical request tokens to DSV4 cache slots.
   - Ensure completed requests release all DSV4 cache components.

4. Add dtype/layout policy.
   - Default v1 storage: bf16.
   - Reserve interfaces for fp8 packed FlashMLA layout.
   - Do not require fp8 kernels to complete this target.

5. Wire pool creation.
   - DSV4 model config should cause scheduler/model runner to create the DSV4 KV pool.
   - Existing models must still create `MHAKVCache`.
   - Disable radix prefix automatically for DSV4 in this target.

6. Add cache tests.
   - Allocation/free for one request.
   - Allocation/free for multiple sequential requests.
   - Layer mapping by `compress_ratios`.
   - No slot leak after request completion.

## Done Criteria

- DSV4 has a dedicated KV pool selected by model type.
- All required cache components can allocate and free.
- DSV4 cache dtype defaults to bf16 on sm80.
- Radix prefix is explicitly disabled for DSV4 v1.
- Tests prove allocation, release, and ratio mapping.

## Non-Goals

- Prefix reuse.
- FlashMLA packed fp8 cache.
- Sparse top-k attention optimization.
- Final memory compaction or eviction policy.
