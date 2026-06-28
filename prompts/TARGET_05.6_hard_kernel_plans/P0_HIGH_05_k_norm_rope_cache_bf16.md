# P0 HIGH: k_norm_rope_cache_fallback bf16 Plan

## Why This Is High Priority

KV norm, RoPE, and cache write are part of the attention input path. The
current wrapper boundary is too narrow, so cleaning this fallback starts with
fixing the interface.

## Policy

- Store bf16 KV first.
- Do not add activation/KV quantization in the first path.
- Prefer upstream `fused_k_norm_rope_flashmla` only if its cache contract can
  be satisfied on sm80. Otherwise implement a local flat-cache bf16 path.

## Current State

Mini currently applies KV norm before this wrapper, the wrapper only applies
RoPE, and cache writes happen in the attention backend. That split prevents a
true fused replacement.

## Typical Workloads

- `kv[tokens, head_dim]`.
- Tail RoPE dim on the last part of the row.
- Decode small token count; prefill contiguous token ranges.

## Implementation Plan

1. Refactor the wrapper interface.
   - Add norm weight, norm eps, output locations, and target cache.
   - Keep old RoPE-only behavior as a fallback compatibility lane.

2. Try upstream fused kernel only after the interface can express it.
   - Check whether upstream FlashMLA cache write assumptions match mini.
   - If the cache layout does not match, do not force it.

3. Implement local bf16 flat-cache kernel.
   - Load KV, compute RMSNorm in fp32, apply RoPE tail, store bf16 to cache.
   - Specialize for `head_dim=128` and common rope dim first.

4. Leave fp8 KV packing for later.

## Validation

- Parity against separate norm, RoPE, and cache store operations.
- Microbench decode and prefill token counts.
- Include non-contiguous output locations.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `k_norm_rope_cache_fallback` with correctness, microbench, decision, and
artifact paths.
