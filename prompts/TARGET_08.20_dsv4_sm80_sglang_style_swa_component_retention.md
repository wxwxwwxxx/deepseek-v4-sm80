# TARGET 08.20: DSV4 SGLang-Style SWA/Component Retention Efficiency

## Status

Optional.  Run only if TARGET 08.18 recommends "go".

## Goal

Reduce DSV4 prefix-cache memory retention if the phase-1 full-page-owner design
is proven to be a real capacity or latency limiter.

This target should adapt the useful parts of SGLang's independent SWA/component
retention design while preserving mini's DSV4 correctness and eviction
integrity.

## Preconditions

Do not start unless TARGET 08.18 provides:

- quantified retained-memory pressure;
- theoretical savings from component-level retention;
- a concrete reason that phase-1 full-page ownership is not enough;
- a go decision.

## Source References

Mini:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/swa_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/pure_swa_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_coordinator.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`

## Plan

1. Reconfirm the ownership model.
   - Identify which data must stay resident for full prefix reuse.
   - Identify which SWA-only data can be tombstoned or retained separately.
   - Identify which C4/C128/indexer/compression states can be derived or pinned.

2. Build a minimal component-retention design.
   - Avoid a broad allocator rewrite if possible.
   - Keep page-size and 128-alignment safety constraints.
   - Make eviction/refcount ownership explicit.
   - Preserve phase-1 full-page-owner path as rollback.

3. Implement behind a separate opt-in.
   - Do not replace phase-1 prefix cache by default.
   - The opt-in must fail closed on unsupported alignment or state ambiguity.

4. Add correctness tests.
   - Full hit, partial hit, miss.
   - SWA boundary around `128`.
   - Multi-prefix eviction pressure.
   - Repeated hit/evict cycles.
   - Component refcount leak/double-free checks.
   - Text or logits compared to prefix-disabled and phase-1 prefix cache.

5. Measure capacity and performance.
   - Retained memory versus phase 1.
   - Usable KV pages/tokens recovered.
   - TTFT/prefill-forward under shared-prefix workloads.
   - Eviction stability under sustained serving workloads.

## Deliverables

Create:

```text
performance_milestones/target08_sglang_style_swa_component_retention/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- design comparison to phase 1 and SGLang;
- exact ownership/refcount model;
- correctness results;
- capacity savings table;
- performance A/B;
- promote/keep-opt-in/reject decision.

## Promotion Rules

Promote only if:

- correctness matches prefix-disabled and phase-1 prefix cache;
- no leaks or double-free;
- memory savings are significant under the workloads that triggered this
  target;
- performance is neutral or better;
- complexity is documented and rollback remains available.

## Stop Rules

Stop and reject the target if:

- component ownership becomes ambiguous;
- compression state cannot be reconstructed or retained safely;
- SWA boundary correctness fails;
- memory savings are much smaller than TARGET 08.18 estimated;
- graph replay or serving stability regresses.

## Non-Goals

- FP8 KV cache or low-precision research.
- Attention-kernel optimization.
- PyNCCL or communication overlap.
- General cache/workspace manager redesign beyond what is required for this
  component-retention experiment.
