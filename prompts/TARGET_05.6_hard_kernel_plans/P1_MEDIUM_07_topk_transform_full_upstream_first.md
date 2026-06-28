# P1 MEDIUM: topk_transform_512 Full Upstream-First Plan

## Why This Is Medium Priority

Mini currently only pads/materializes existing indices to width 512. That is
not the full upstream score/page-table top-k transform. Full top-k matters when
the indexer starts producing score rows instead of final sparse indices.

## Policy

- Keep all score and indexer activations bf16/fp32 as appropriate.
- Do not quantize activations for this path.
- Prefer upstream `topk_transform_512` / v2 first if it compiles and runs on
  sm80. Only write local Triton if upstream JIT is blocked or sm90/sm100-only.

## Current State

The current padding-only opt-in is neutral in microbench. Upstream consumes
scores, sequence lengths, page tables, and page size.

## Typical Workloads

- C4 sparse selection width 512.
- Rows aligned to 64/512.
- Page-table translation from raw compressed positions to page/full indices.

## Implementation Plan

1. Decide when full semantics are needed.
   - If metadata keeps producing final indices, padding-only is enough.
   - If bf16 indexer logits are added, implement full transform.

2. Try upstream JIT.
   - Validate compile and runtime on A100.
   - Avoid any path requiring sm90/sm100-only shared memory assumptions.

3. Add a new full-transform wrapper signature.
   - Inputs: scores, seq_lens, page_table, page_size.
   - Outputs: raw indices plus page/full indices.

4. Implement local top-k only if upstream is blocked.

## Validation

- Top-k parity on fixed scores and edge lengths.
- Page-table translation parity.
- Microbench realistic score-row shapes.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `topk_transform_512_fallback` / v2 with correctness, microbench, decision,
and artifact paths.
