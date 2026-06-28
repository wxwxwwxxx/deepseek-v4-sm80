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

## 实现结论

是的，这次实验的结论可以简化为：

当前 `quantized_linear_ref` 仍然应该默认使用 PyTorch fallback。

我加的 Triton kernel 已经能跑、能 opt-in，但不适合直接默认启用。原因是：

- `fp8` dense projection：Triton dequant-on-load 比当前 fallback 慢，大约 `0.62x-0.82x`。
- `fp4 w13`：也略慢，大约 `0.87x-0.90x`。
- `fp4 w2` 小 batch decode：有收益，大约 `1.50x`，但范围窄。
- 大 `m` 已经自动 fallback，避免数值/性能风险。

所以当前策略是：保留 PyTorch fallback 作为默认正确路径，Triton 路径只作为实验开关和后续 grouped MoE 的参考。

`torch.compile` 我不建议现在加。

原因是这个 fallback 的核心问题不是 Python 调度，而是“每次调用都 materialize dequant weight，然后再 cuBLAS matmul”。`torch.compile` 很难把 `fp8/fp4 weight dequant + F.linear` 真正融合成 dequant-on-load GEMM，尤其还涉及 float8/int8 packed weight、scale expansion、动态 shape。它可能减少一点 Python overhead，但不太可能改变主要瓶颈，反而会引入 compile latency、graph break 和调试复杂度。

更值得做的下一步是：

1. 继续用当前 PyTorch fallback 作为默认。
2. 后续在 MoE grouped route 上做真正的 grouped fp4 dequant-on-load GEMM。
3. 如果要优化 decode 小 batch，可以单独围绕 `fp4 w2` 小 m 路径继续实验，而不是全局替换。