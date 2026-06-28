# P1 MEDIUM: FlashMLA Adapter Upstream-First Plan

## Why This Is Medium Priority

FlashMLA may become the preferred attention backend if mini can provide the
expected cache layout. It should not block the first bf16 flat-cache attention
replacement.

## Policy

- Prefer upstream FlashMLA/FlashInfer over local kernels when their sm80 path is
  usable.
- Keep activations and cache bf16 for the first adapter.
- Do not introduce fp8 packed cache until a bf16 adapter proves the metadata
  contract.

## Current State

`sgl_kernel.flash_mla` imports, and FlashInfer exposes MLA wrappers. Mini lacks
the packed page metadata and ckv/kpe cache layout required by these APIs.

## Typical Workloads

- Decode with paged cache.
- Sparse prefill with combined compressed and SWA rows.
- Long context where attention candidate handling is visible.

## Implementation Plan

1. Write the exact upstream contract.
   - Cache shapes and dtypes.
   - Page table and indptr/indices layout.
   - q split into nope/rope components.

2. Build a zero-copy adapter if possible.
   - If zero-copy is impossible, measure staging cost.
   - Keep adapter allocation out of the hot path if possible.

3. Run isolated decode parity.
   - Compare upstream backend output against `paged_mqa_attention_fallback`.

4. Only then evaluate sparse prefill.

## Validation

- Metadata shape tests.
- Attention parity against fallback.
- Microbench with adapter cost included.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `flash_mla_with_kvcache` / `flash_mla_sparse_prefill` with correctness,
microbench, decision, and artifact paths.
