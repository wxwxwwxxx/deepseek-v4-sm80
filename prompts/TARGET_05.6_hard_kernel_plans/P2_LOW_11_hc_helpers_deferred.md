# P2 LOW: hc_pre/post/head_fallback Deferred Plan

## Why This Is Low Priority

HC helpers are important, but they should wait until attention and MoE fallbacks
are reduced. Otherwise optimization effort may chase a secondary bottleneck.

## Policy

- Keep activations bf16 unless the existing math requires fp32 intermediates.
- Prefer upstream MHC/TileLang helpers if they work on sm80.
- Write local kernels only after caller-level profiling proves HC is visible.

## Current State

HC helpers are torch fallbacks. Upstream fast paths use MHC, TileLang, AITER, or
DeepGEMM depending on platform. Local DeepGEMM is unavailable.

## Typical Workloads

- `x[tokens, hc_mult, hidden]`.
- Small `hc_mult`, large hidden.
- `hc_pre` includes prenorm linear, split, sigmoid, softmax, and Sinkhorn.

## Implementation Plan

1. Profile HC after P0 work.
   - Measure `hc_pre`, `hc_post`, and `hc_head` separately.

2. Try upstream MHC helpers first.
   - Keep only sm80-compatible exact paths.

3. If local work is needed, start with `hc_post`.
   - It is mostly elementwise residual mixing and should be easier than
     Sinkhorn.

4. Defer approximations.
   - Sinkhorn or mixing approximations require oracle evidence.

## Validation

- Parity against HC fallback helpers.
- Per-helper microbench.
- DSV4 forward fallback tests with toggles on.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `hc_pre/post/head_fallback` with correctness, microbench, decision, and
artifact paths.
