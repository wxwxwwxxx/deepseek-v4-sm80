# P0 HIGH: wo_a_grouped_projection_fallback bf16 Weight-Dequant Plan

## Why This Is High Priority

`wo_a_grouped_projection_fallback` is on the attention output path. It currently
dequants fp8 weights and runs `torch.einsum`; removing it helps clear another
main-path torch fallback.

## Policy

- Keep activation/output input `o` in bf16.
- Dequant fp8 weights inside the kernel to bf16 tiles.
- Use bf16 tensor core MMA where the shape is large enough.
- Prefer upstream DeepGEMM or sgl-kernel `fp8_einsum` only if it loads and
  explicitly works on sm80. Otherwise write a local grouped projection.

## Current State

Upstream uses DeepGEMM `fp8_einsum`. Local DeepGEMM currently fails to load, so
mini uses full dequant plus torch einsum.

## Typical Workloads

- `o[tokens, groups, d_per_group]`.
- fp8 weight viewed as `wo_a[groups, o_lora_rank, d_per_group]`.
- Output `[tokens, groups * o_lora_rank]`.

## Implementation Plan

1. Check upstream first.
   - Re-test DeepGEMM load status.
   - Check for any installed sm80 `fp8_einsum` equivalent.
   - If unavailable, continue with local Triton.

2. Implement dequant-on-load grouped projection.
   - Program tile over token block, group, and rank tile.
   - Load fp8 weight plus scale, convert to bf16, and use bf16 dot.
   - Avoid materializing the full dequanted weight.

3. Compare direct Triton against torch fallback.
   - Include full dequant cost in fallback timing.
   - Measure decode and prefill token counts.

4. Keep fp8 activation quant out of this path for now.

## Validation

- Parity against `wo_a_grouped_projection_fallback`.
- Microbench token counts 1, 8, 64, 512.
- Record real DSV4 group/rank dimensions.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `wo_a_grouped_projection_fallback` with correctness, microbench, decision,
and artifact paths.
