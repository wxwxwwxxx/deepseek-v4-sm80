# P0 HIGH: paged_mqa_attention_fallback bf16 Plan

## Why This Is High Priority

The attention fallback is structurally expensive because it uses Python ragged
indices and small torch operations per query. Clearing it is required before
the DSV4 attention path can be considered mostly fallback-free.

## Policy

- Keep q and KV activations bf16 first.
- Do not introduce fp8 KV cache for the first replacement.
- Prefer upstream FlashInfer or FlashMLA if the required cache layout can be
  adapted cheaply on sm80. If not, implement a flat-cache bf16 kernel.

## Current State

Mini has bf16 flat cache buffers and per-row context index tensors. FlashInfer
MLA wrappers exist in the environment, but they require paged ckv/kpe cache
layout and packed metadata.

## Typical Workloads

- Decode: active tokens `1..batch`, local heads, `head_dim=128`.
- SWA-only candidate count bounded by `window_size`.
- C4/C128 layers combine compressed candidates and SWA candidates.

## Implementation Plan

1. Normalize metadata first.
   - Replace Python list input with `indptr`, `indices`, and row lengths.
   - Preserve duplicate removal and ordering exactly.

2. Try upstream backends with an adapter.
   - Build a FlashInfer/FlashMLA metadata adapter prototype.
   - Include adapter allocation/copy cost in the microbench.
   - Keep it only if it is cheaper than a local kernel.

3. Implement local bf16 flat-cache decode if upstream layout is not viable.
   - One program per query/head or query/head block.
   - Load candidate rows by flat cache index.
   - Compute scores in fp32, apply attention sink, reduce values to bf16.

4. Add fp8 KV dequant only after the bf16 path is stable and useful.

## Validation

- Parity against `paged_mqa_attention_fallback` for empty, single, SWA-only,
  C4+SWA, C128+SWA, and duplicate candidate rows.
- Microbench candidate counts 32, 128, 640, 1024.
- No E2E requirement until other major fallbacks are removed.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `paged_mqa_attention_fallback` with correctness, microbench, decision, and
artifact paths.
