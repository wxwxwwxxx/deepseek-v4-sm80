# P2 LOW: silu_and_mul_post_quant Deferred Plan

## Why This Is Low Priority

Plain bf16 SWIGLU already has an opt-in Triton kernel. Post-quant activation is
not useful until a downstream grouped quantized GEMM consumes the quantized
activation directly.

## Policy

- First implementation keeps activations bf16.
- Do not add activation fp8/fp4 quantization while clearing initial fallbacks.
- Revisit post-quant only after grouped GEMM has a quantized activation
  contract.

## Current State

`silu_and_mul_masked_post_quant` and `silu_and_mul_contig_post_quant` remain
unsupported. Adding them now would likely add memory traffic without removing a
larger fallback.

## Typical Workloads

- Expert activation after w1/w3.
- Optional routed weights.
- Output consumed by w2 projection.

## Implementation Plan

1. Wait for grouped GEMM design.
   - Define activation quant group size, scale dtype, and layout.

2. Implement contiguous post-quant before masked post-quant.
   - Prefer route grouping that creates contiguous expert chunks.

3. Compare against bf16 activation.
   - The only acceptable win is full activation-plus-w2 improvement.

## Validation

- Quant-dequant tolerance against bf16 activation.
- Microbench activation-only and activation-plus-consumer.
- Oracle/top-k gate if approximation affects model outputs.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `silu_and_mul_*_post_quant` with correctness, microbench, decision, and
artifact paths.
