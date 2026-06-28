# P0 HIGH: quantized_linear_ref bf16 Weight-Dequant Plan

## Why This Is First

`quantized_linear_ref` sits on the main DSV4 path: attention projections,
shared experts, and routed experts all depend on fp8/fp4 weights. Clearing this
fallback removes repeated Python-level dequant plus torch linear work from the
largest compute surface.

## Policy

- Keep activations in bf16 first. Do not add activation fp8/fp4 quantization in
  the first working path.
- If weights are fp8 or fp4, dequantize or upcast weights inside the kernel and
  feed bf16 values into sm80 tensor cores.
- Prefer upstream existing operators first. Try torch/cuBLAS, FlashInfer,
  sgl-kernel, or other installed upstream kernels before writing local Triton or
  CUDA. Define a new operator only after the upstream path is missing,
  unavailable, or clearly not sm80-compatible.

## Current State

The fallback dequants fp8/fp4 weights and calls `F.linear`. It is correct but
does extra memory traffic. A100 sm80 does not have native fp8/fp4 tensor cores,
so the practical target is weight dequant-on-load followed by bf16 tensor core
MMA.

## Typical Workloads

- Dense fp8 projections: q, k/v, output, shared experts.
- Routed MoE fp4 experts: w1/w3 and w2.
- Decode: `m=1..batch`.
- Prefill: larger `m`, same weight shapes.

## Implementation Plan

1. Build an isolated shape matrix.
   - Measure `m=1,4,8,16,64,256`.
   - Include real DSV4 fp8 projection shapes and fp4 expert shapes.
   - Report fallback time, dequant time, and matmul time separately.

2. Try upstream options first.
   - Check torch native `mm`/`_scaled_mm` support for bf16 output from dequanted
     weights.
   - Check installed FlashInfer or sgl-kernel GEMM helpers.
   - Do not use DeepGEMM unless it loads and explicitly supports sm80 in the
     local environment.

3. Implement local dequant-on-load only if needed.
   - First kernel lane: fp8/fp4 weight -> bf16 tile -> bf16 tensor core MMA.
   - Accumulate fp32 or bf16 according to tolerance.
   - Keep activation input bf16.

4. Integrate with MoE route work.
   - Single-call linear optimization is useful, but grouped expert GEMM is the
     larger win.
   - Share dequant tile logic with `P0_HIGH_02_moe_route_dispatch`.

## Validation

- Numeric parity against `quantized_linear_ref`.
- Isolated microbench for each target shape.
- No E2E requirement until the major torch fallbacks are cleaned; record only
  per-kernel and caller-level microbench.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `quantized_linear_ref` with correctness, microbench, decision, and artifact
paths.
