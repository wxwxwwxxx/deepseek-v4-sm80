# P1 MEDIUM: linear_bf16_fp32_fallback Upstream-First Plan

## Why This Is Medium Priority

This helper may matter for HC/router paths, but it is not the first fallback to
clear unless profiling shows it is visible.

## Policy

- Inputs remain bf16; output remains fp32.
- Prefer torch/cuBLAS or upstream kernels before local matmul work.
- Do not write a custom matmul unless real shapes show torch is inadequate.

## Current State

The fallback uses `F.linear(x.float(), weight.float())`. Upstream may route this
through torch `mm`, AITER, or DeepGEMM. DeepGEMM is not currently usable in the
local environment.

## Typical Workloads

- HC prenorm linear with flattened `hc_mult * hidden`.
- Possible router/helper matmuls in later paths.

## Implementation Plan

1. Measure real caller shapes.
   - HC pre/head token counts.
   - Router shapes if this helper is reused.

2. Try upstream/native first.
   - `torch.mm`/cuBLAS with fp32 output.
   - Any installed sm80 upstream helper.

3. Specialize only if needed.
   - If custom Triton is justified, specialize to stable HC dimensions.
   - Keep fallback for all other shapes.

## Validation

- Parity against `linear_bf16_fp32_fallback`.
- Microbench real HC/router shapes.
- Caller-level microbench before promotion.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `linear_bf16_fp32_fallback` with correctness, microbench, decision, and
artifact paths.

## Result 2026-06-29

Implemented an upstream-first opt-in path behind
`MINISGL_DSV4_SM80_LINEAR_BF16_FP32`:

- `linear_bf16_fp32_fallback` now uses `torch.mm(..., out_dtype=torch.float32)`
  only when the toggle is enabled, the device is sm80 CUDA, and both inputs are
  bf16.
- DSV4 HC pre/head callers keep the original fp32 HC weights, but lazily cache
  bf16 copies for the opt-in path. The cache is invalidated when the source
  tensor changes.
- fp32 HC-weight fallback semantics remain the default.

Microbench artifact:
`/tmp/dsv4_linear_bf16_fp32_upstream_microbench_20260629.json`.

Observed speedup versus current fp32-weight fallback:

- HC pre, `K=16384,N=24`: 1.06-1.12x for small `M<=128`, 14.32x for `M=2048`.
- HC head, `K=16384,N=4`: 1.00-1.08x for small `M<=128`, 5.71x for `M=2048`.

Decision: keep opt-in. The path is useful evidence for large prefill HC work,
but it rounds fp32 HC weights to bf16 and therefore still needs caller-level
correctness/E2E gates before default promotion.
