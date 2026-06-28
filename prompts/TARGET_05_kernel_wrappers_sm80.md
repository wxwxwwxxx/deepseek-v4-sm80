# TARGET 05: DSV4 Kernel Wrappers and sm80 Adaptation Layer

## Goal

Audit mini-sglang against sglang-main's DeepSeek V4 fused-kernel boundaries, then create a centralized DSV4 kernel wrapper layer so all high-performance fused operations have stable interfaces, explicit sm80 fallbacks, and clear TODOs for kernels that must be ported later.

This target is complete when the wrapper module contains a code-level kernel inventory matching the known sglang-main DSV4 fusion points, model and attention code call DSV4 kernels only through wrappers, and sm80 never accidentally enters sm90/sm100-only paths.

## Primary References

- SGLang DSV4 JIT wrappers: `/workspace/sglang-main/python/sglang/jit_kernel/dsv4`
- SGLang DSV4 model fused call sites: `/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py`
- SGLang DSV4 attention backend: `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- SGLang DSV4 attention helpers: `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4`
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

0. Audit sglang-main DSV4 fused-kernel boundaries.
   - Start from `sglang.jit_kernel.dsv4.__all__`.
   - Cross-check call sites in `models/deepseek_v4.py`, `layers/attention/deepseek_v4_backend.py`, and `layers/attention/dsv4/*`.
   - Cross-check current mini-sglang direct/fallback implementations in:
     - `python/minisgl/models/deepseek_v4.py`
     - `python/minisgl/attention/deepseek_v4.py`
     - `python/minisgl/kvcache/deepseek_v4_pool.py`
   - Produce a code-level inventory table in the wrapper module, not just prose in the prompt.

1. Add a centralized DSV4 kernel wrapper module.
   - Suggested location: `python/minisgl/kernel/deepseek_v4.py`.
   - All DSV4 fused operations must be called through this module.
   - Model code should not directly import optional third-party kernels.
   - Existing correctness fallbacks in model/attention code should also be routed through this module when they represent a future fused-kernel boundary.

2. Define wrappers for required fused operations.
   - fused rope inplace
   - fused q norm + rope
   - fused q indexer rope + hadamard quant
   - fused q indexer rope + hadamard fp4 quant
   - fused k norm + rope + cache store
   - fused norm + rope inplace
   - fused store cache
   - compressed cache construction and compressor plan creation
   - compressed forward and compressed norm + rope + store
   - C4 top-k transform
   - top-k v2 planning
   - FlashMLA with KV cache
   - sparse prefill attention
   - indexer paged-MQA logits
   - bf16/fp8/fp4 GEMM wrappers
   - `wo_a` grouped projection / fp8 einsum
   - MHC pre/post/head helpers
   - hash routing helpers
   - MoE pre-dispatch and silu/mul/post-quant helpers

3. Add capability detection.
   - Detect CUDA capability.
   - Detect optional dependency availability for DeepGEMM, FlashMLA/sgl_kernel, FlashInfer, Marlin, TileLang, and Triton kernels.
   - On sm80, default to safe fallback for:
     - DeepGEMM-only paths
     - sm100 fp4 indexer
     - sm100-only fp8 `wo_a` GEMM
     - top-k v2 paths that exceed sm80 shared-memory limits
     - FlashMLA paths that require packed fp8 cache layouts not yet implemented locally
   - If `sgl_kernel` exists but only has sm100 binary, report that clearly.

4. Implement fallbacks.
   - bf16 torch fallback for correctness.
   - Deterministic top-k fallback for indexer.
   - fp8/fp4 dequant + torch matmul fallback for correctness.
   - bf16 KV-cache fallback when packed fp8/fp4 cache layout is unavailable.
   - Explicit `NotImplementedError` only when no correctness fallback exists.
   - Error messages must name the wrapper and the missing capability.

5. Add interface documentation in code.
   - Add a small dataclass or constant table such as `DSV4_KERNEL_INVENTORY`.
   - For every wrapper, record:
     - sglang-main source function
     - expected input shapes
     - expected output shapes
     - current mini-sglang call site
     - current sm80 behavior
     - optional dependencies
     - status: `native`, `fallback`, `unsupported`, or `todo`
     - later high-performance replacement target

6. Add guard tests.
   - sm80 should not enter sm100-only paths.
   - Missing optional kernel should produce clear fallback or clear error.
   - Wrappers should preserve expected shape and dtype under fallback.
   - The inventory should contain entries for all known sglang-main DSV4 fused-kernel exports or document why an export is out of scope.

## Done Criteria

- DSV4 model/attention code has no scattered direct dependency on optional fused kernels.
- The wrapper module includes a kernel inventory that maps sglang-main fused functions to mini-sglang wrapper names.
- The inventory reports sm80 status, optional dependency status, fallback behavior, and later port target for every known DSV4 fused-kernel boundary.
- sm80 capability gates are explicit and tested.
- Wrappers exist for all known DSV4 fused-kernel boundaries.
- Unsupported paths fail clearly instead of crashing deep inside an import.
- Later kernel porting can happen by replacing wrapper internals.

## Non-Goals

- Port every fused kernel in this target.
- Match final sglang throughput.
- Implement radix prefix cache.
