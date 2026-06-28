# P0 HIGH: compress_norm_rope_store_fallback bf16 Plan

## Why This Is High Priority

Compressed KV store feeds C4/C128 attention and indexer cache. The current
wrapper only stores tensors, so the first task is making the boundary capable
of expressing norm plus optional RoPE plus store.

## Policy

- Keep compressed activations bf16.
- Do not pack compressed cache to fp8/fp4 in the first path.
- Try upstream `compress_norm_rope_store` or `fused_norm_rope_inplace` if its
  plan/cache contract can be adapted on sm80; otherwise write a local kernel.

## Current State

The current wrapper receives only `kvcache`, `layer_id`, `kv`, and `loc`. It
does not receive positions, norm weight, eps, or rope config.

## Typical Workloads

- Ratio 4 C4 attention and C4 indexer cache.
- Ratio 128 compressed attention.
- Decode small rows and prefill compressed rows after `compress_forward`.

## Implementation Plan

1. Extend the wrapper boundary.
   - Add positions, norm weight, norm eps, rotary dim/base, and cache type.
   - Preserve store-only fallback when those arguments are absent.

2. Try upstream first.
   - Check SGLang JIT `compress_norm_rope_store` on sm80.
   - Use it only if plan tensors and cache layout can be matched without large
     staging buffers.

3. Implement local bf16 fused kernel if needed.
   - Normalize compressed row.
   - Apply RoPE when required.
   - Scatter bf16 into C4, C128, or indexer cache.

4. Split C4 and C128 kernels if ratio-specific branches become expensive.

## Validation

- Parity against separate norm, RoPE, and store operations.
- Microbench store-only, norm+store, and norm+RoPE+store.
- Test C4, C128, and indexer cache cases.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `compress_norm_rope_store_fallback` with correctness, microbench, decision,
and artifact paths.
