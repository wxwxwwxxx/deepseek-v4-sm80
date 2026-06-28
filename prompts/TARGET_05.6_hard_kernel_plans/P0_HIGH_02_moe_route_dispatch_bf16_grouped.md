# P0 HIGH: moe_gate/hash_topk/mega_moe_pre_dispatch bf16 Grouped Plan

## Why This Is High Priority

MoE routing plus expert dispatch is on every routed layer. Even a good expert
GEMM cannot help if tokens still flow through Python expert loops and many
small fallback calls.

## Policy

- Keep activations bf16 through routing and dispatch.
- If expert weights are fp4/fp8, dequant or upcast weights inside the grouped
  GEMM consumer, then use bf16 tensor cores.
- Prefer upstream route kernels first. Use SGLang JIT route helpers only if they
  compile and run on sm80. Avoid dense invalid-route graph fallbacks as default.

## Current State

`moe_gate_fallback` uses torch scoring/top-k, and routed experts are selected
with Python loops over expert ids. Upstream has `hash_topk`,
`mask_topk_ids`, and `mega_moe_pre_dispatch`, but local sgl-kernel DSV4 ops are
not exposed.

## Typical Workloads

- Decode: small token count, `topk` routes per token.
- Prefill: larger token count, more route reuse.
- Hash layers use token-id table lookup; non-hash layers use router scores.

## Implementation Plan

1. Split the work into route metadata and expert compute.
   - Route metadata: sorted token ids, expert ids, per-expert counts, offsets,
     routed weights.
   - Expert compute: grouped dequant-on-load bf16 tensor core GEMM.

2. Try upstream route helpers first.
   - Attempt SGLang JIT route kernels in isolation.
   - Keep them only if they are sm80-compatible and do exact route grouping.

3. Implement exact local route grouping if upstream is blocked.
   - No dense invalid-route computation.
   - No Python loop over experts in the hot path.
   - Preserve deterministic accumulation order.

4. Connect to `quantized_linear_ref` replacement.
   - Dispatch grouped hidden states into expert chunks.
   - Reuse the P0 quantized-weight dequant-on-load bf16 tensor core plan.

## Validation

- Route metadata parity with current torch fallback.
- Full routed expert output parity.
- Microbench routing alone and route+GEMM together.
- No E2E requirement at this stage.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `moe_gate/hash_topk/mega_moe_pre_dispatch` with correctness, microbench,
decision, and artifact paths.
