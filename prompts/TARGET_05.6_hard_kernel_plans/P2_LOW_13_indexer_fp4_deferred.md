# P2 LOW: fused_q_indexer_rope_hadamard_fp4_quant Deferred Plan

## Why This Is Low Priority

The fp4 indexer is approximate and depends on downstream fp8/fp4 paged logits.
It should wait until the bf16 or fp8 indexer path is working and measurable.

## Policy

- Do not quantize activations for the initial fallback-cleaning path.
- Prefer upstream fp4 indexer only if it is proven sm80-compatible.
- If local work is needed, treat fp4 as an approximation requiring oracle/top-k
  evidence.

## Current State

The upstream fp4 path is sm100/DeepGEMM-oriented. Local DeepGEMM fails to load,
and A100 has no native fp4 tensor-core path.

## Typical Workloads

- Same indexer query rows as bf16/fp8 indexer.
- Packed fp4 Q plus scale.
- Downstream paged logits must consume fp4 Q efficiently.

## Implementation Plan

1. Do not start until bf16/fp8 indexer is working.

2. Try upstream only if environment changes.
   - Re-check DeepGEMM and sgl-kernel DSV4 ops.
   - Confirm sm80 support, not just import success.

3. If local work is justified, start with pack/dequant.
   - Measure fp4 pack cost separately.
   - Compare against bf16-direct indexer.

4. Require oracle/top-k gate before promotion.

## Validation

- Pack/dequant round-trip tests.
- Top-k agreement against bf16 indexer.
- Microbench pack, logits, and full path separately.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `fused_q_indexer_rope_hadamard_fp4_quant` with correctness, microbench,
decision, and artifact paths.
