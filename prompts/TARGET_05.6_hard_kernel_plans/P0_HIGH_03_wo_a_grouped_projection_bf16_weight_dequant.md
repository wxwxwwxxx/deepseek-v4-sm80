# P0 HIGH: wo_a_grouped_projection_fallback bf16 Weight-Dequant Plan

## Why This Is High Priority

`wo_a_grouped_projection_fallback` is on the attention output path. It currently
dequants fp8 weights and runs `torch.einsum`; removing it helps clear another
main-path torch fallback.

## Policy

- Keep activation/output input `o` in bf16.
- Dequant fp8 weights inside the kernel to bf16 tiles.
- Use bf16 tensor core MMA where the shape is large enough.
- Prefer upstream DeepGEMM or sgl-kernel `fp8_einsum` only if it loads and
  explicitly works on sm80. Otherwise write a local grouped projection.

## Current State

Upstream uses DeepGEMM `fp8_einsum`. Local DeepGEMM currently fails to load, so
mini uses full dequant plus torch einsum.

## Typical Workloads

- `o[tokens, groups, d_per_group]`.
- fp8 weight viewed as `wo_a[groups, o_lora_rank, d_per_group]`.
- Output `[tokens, groups * o_lora_rank]`.

## Implementation Plan

1. Check upstream first.
   - Re-test DeepGEMM load status.
   - Check for any installed sm80 `fp8_einsum` equivalent.
   - If unavailable, continue with local Triton.

2. Implement dequant-on-load grouped projection.
   - Program tile over token block, group, and rank tile.
   - Load fp8 weight plus scale, convert to bf16, and use bf16 dot.
   - Avoid materializing the full dequanted weight.

3. Compare direct Triton against torch fallback.
   - Include full dequant cost in fallback timing.
   - Measure decode and prefill token counts.

4. Keep fp8 activation quant out of this path for now.

## Validation

- Parity against `wo_a_grouped_projection_fallback`.
- Microbench token counts 1, 8, 64, 512.
- Record real DSV4 group/rank dimensions.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `wo_a_grouped_projection_fallback` with correctness, microbench, decision,
and artifact paths.

## 实现结论

本阶段已经落地一个 opt-in、decode-gated 的 `bf16-direct` 路径：

- 新增本地 Triton `wo_a_grouped_projection_fp8`，输入保持 bf16，fp8 e4m3 权重按 tile 反量化并直接参与 bf16 Tensor Core dot，不再在小 token decode 路径中 materialize 完整 bf16 `wo_a`。
- `wo_a_grouped_projection_fallback` 在 `MINISGL_DSV4_SM80_WO_A_BF16=1` 时尝试新路径；默认关闭，且 `tokens > 16` 会自动回退到原 dequant+einsum。
- `MINISGL_DSV4_SM80_WO_A_FP8` 暂未实现；本阶段没有引入 fp8 activation quant。

外部 fast path 复测结论：

- DeepGEMM 仍因缺少 `libcudart.so.13` 无法加载。
- 已安装 `sgl_kernel` 可 import，但没有可用的 `fp8_einsum` 等价符号。
- 上游 SGLang 也会在 sm80 上禁用 `SGLANG_OPT_FP8_WO_A_GEMM`，因此本地 Triton 是当前可控路线。

验证结果：

- `pytest -q -o addopts='' tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_opt_in_kernels_match_fallbacks` 通过。
- 目标 smoke 通过：`tests/kernel/test_deepseek_v4_wrappers.py`、`tests/attention/test_deepseek_v4_backend_metadata.py`、`tests/core/test_deepseek_v4_kvcache.py`、`tests/models/test_deepseek_v4_forward_fallback.py`，共 20 passed。
- microbench artifact: `/tmp/dsv4_wo_a_grouped_projection_bf16_weight_dequant_microbench_20260628.json`。

当前策略仍是默认关闭。真实 DSV4 尺寸 G=8/R=1024/D=4096 下，tokens 1/8 约 1.21x；tokens 64/512 的直接 Triton dequant-on-load 会反复读取权重，因此最终实现对大 token 自动回退，避免 prefill 退化。
