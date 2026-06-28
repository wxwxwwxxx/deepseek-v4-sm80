# TARGET 05: DSV4 Kernel Wrappers and sm80 Adaptation Layer

## Goal

Create a centralized DSV4 kernel wrapper layer in mini-sglang so all high-performance fused operations have stable interfaces, explicit sm80 fallbacks, and clear TODOs for kernels that must be ported later.

This target is complete when model and attention code call DSV4 kernels only through wrappers, and sm80 never accidentally enters sm90/sm100-only paths.

## Primary References

- SGLang DSV4 JIT wrappers: `/workspace/sglang-main/python/sglang/srt/jit_kernel/dsv4`
- SGLang DSV4 attention backend: `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- SGLang hook/default args: `/workspace/sglang-main/python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- SGLang server args for DSV4 defaults: `/workspace/sglang-main/python/sglang/srt/server_args.py`
- FlashMLA import/interface reference inside sglang-main.

## Old dsv4 Branch References

Useful local prior work:

- `git show dsv4:python/minisgl/kernel/deepseek_v4.py`
- `git show dsv4:python/minisgl/attention/deepseek_tilelang.py`
- `git show dsv4:tests/kernel/test_deepseek_tilelang.py`
- `git show dsv4:benchmark/offline/deepseek_v4_perf_matrix.py`
- `git show dsv4:prompts/PERF_RESULTS.md`

The old branch can guide sm80 fallback behavior and benchmark lessons, but the wrapper API should follow sglang-main.

## Plan

1. Add a centralized DSV4 kernel wrapper module.
   - Suggested location: `python/minisgl/kernel/deepseek_v4.py`.
   - All DSV4 fused operations must be called through this module.
   - Model code should not directly import optional third-party kernels.

2. Define wrappers for required fused operations.
   - fused q norm + rope
   - fused q indexer rope + hadamard quant
   - fused k norm + rope + cache store
   - fused store cache
   - compressed cache construction
   - C4 top-k transform
   - FlashMLA with KV cache
   - sparse prefill attention
   - bf16/fp8/fp4 GEMM wrappers
   - MHC / hash routing helpers if needed by MoE

3. Add capability detection.
   - Detect CUDA capability.
   - On sm80, default to safe fallback for:
     - DeepGEMM-only paths
     - sm100 fp4 indexer
     - sm100-only fp8 `wo_a` GEMM
   - If `sgl_kernel` exists but only has sm100 binary, report that clearly.

4. Implement fallbacks.
   - bf16 torch fallback for correctness.
   - Deterministic top-k fallback for indexer.
   - Explicit `NotImplementedError` only when no correctness fallback exists.
   - Error messages must name the wrapper and the missing capability.

5. Add interface documentation in code.
   - For every wrapper, record:
     - sglang-main source function
     - expected input shapes
     - expected output shapes
     - current sm80 behavior
     - later high-performance replacement target

6. Add guard tests.
   - sm80 should not enter sm100-only paths.
   - Missing optional kernel should produce clear fallback or clear error.
   - Wrappers should preserve expected shape and dtype under fallback.

## Done Criteria

- DSV4 model/attention code has no scattered direct dependency on optional fused kernels.
- sm80 capability gates are explicit and tested.
- Wrappers exist for all known DSV4 fused-kernel boundaries.
- Unsupported paths fail clearly instead of crashing deep inside an import.
- Later kernel porting can happen by replacing wrapper internals.

## Non-Goals

- Port every fused kernel in this target.
- Match final sglang throughput.
- Implement radix prefix cache.
