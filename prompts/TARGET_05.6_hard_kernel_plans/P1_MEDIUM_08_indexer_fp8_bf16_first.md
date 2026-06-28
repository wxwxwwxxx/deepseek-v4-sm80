# P1 MEDIUM: fused_q_indexer_rope_hadamard_quant bf16-First Plan

## Why This Is Medium Priority

The indexer path matters for C4 sparse selection, but fp8 activation quant is
not required for the first fallback-cleaning pass. The useful first target is a
bf16-direct indexer pipeline.

## Policy

- Keep indexer Q activations bf16 first.
- Do not quantize indexer activations until bf16 logits/top-k are working.
- Prefer upstream fused indexer kernels if they are actually present and sm80
  compatible. If upstream only provides fp8 quant without a downstream fp8
  logits consumer, do not use it as the first path.

## Current State

Local `torch.ops.sgl_kernel` exposes no DSV4 indexer ops. Upstream fuses RoPE,
Hadamard, and fp8 quantization, but mini lacks the downstream fp8 paged logits
kernel.

## Typical Workloads

- Active token rows.
- Indexer head dim smaller than main attention.
- Output feeds sparse score/top-k construction.

## Implementation Plan

1. Build bf16-direct RoPE+Hadamard first.
   - Keep Q bf16/fp32.
   - Produce logits or transformed Q consumed by a bf16 score path.

2. Reuse packed metadata from attention/top-k plans.
   - Avoid per-row Python lists before optimizing quantization.

3. Try upstream only where it fits.
   - If upstream JIT can run sm80 and a bf16 or dequantable path exists, use it.
   - Otherwise implement local Triton.

4. Add fp8 activation quant later.
   - Only after there is an fp8 logits consumer.
   - Require top-k/oracle evidence.

## Validation

- Parity against torch indexer fallback for logits/top-k.
- Top-k stability on fixed seeds.
- Microbench RoPE+Hadamard and full indexer path.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `fused_q_indexer_rope_hadamard_quant` with correctness, microbench,
decision, and artifact paths.
