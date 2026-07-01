# TARGET 07.4: DSV4 sm80 Precision Lanes

## Goal

Evaluate precision tradeoffs after the bf16-direct exact path has a strong
baseline. This target decides whether fp8/fp4 activation quantization or INT8
Tensor Core expert compute is worth adding to mini-sglang's DeepSeek V4 sm80
path.

The default promoted path remains exact unless an opt-in precision lane passes
explicit quality and performance gates.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- sm80 kernel R&D record:
  `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md`
- exact MoE V2 target:
  `prompts/TARGET_07.3_dsv4_sm80_moe_v2_exact.md`
- mini DSV4 wrappers: `python/minisgl/kernel/deepseek_v4.py`
- mini DSV4 model: `python/minisgl/models/deepseek_v4.py`
- vLLM DSV4 model:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- vLLM fused MoE:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`

## Precision Roadmap

1. bf16-direct baseline.
   - Main sm80 kernels should use bf16 Tensor Core paths where possible.
   - Do not quantize activations in the default exact path.
   - Keep model-original fp32 computations as fp32 unless there is a separate
     tested TF32 experiment.
   - TF32 is allowed only as an explicit opt-in measurement for fp32
     matmul-like work; record correctness impact before keeping it.

2. fp8/fp4 activation quantization.
   - Test fp8-act and fp4-act as separate opt-in variants.
   - Use vLLM's precision implementation as a priority reference when the lane
     is comparable.
   - Measure whether activation quantization reduces memory bandwidth or
     improves Tensor Core utilization enough to justify quality risk.

3. INT8 Tensor Core path.
   - Focus first on MoE expert compute if exact V2 remains compute-bound.
   - Keep INT8 behind an explicit toggle such as
     `MINISGL_DSV4_SM80_MOE_INT8_TC=1`.
   - Compare against the best exact bf16-direct path, not against a fallback.

## Plan

1. Freeze the bf16-direct baseline.
   - Use the best post-TARGET 07.2/07.3 exact variant as the oracle.
   - Save 4096/128 nsys, 4096/1024 macro benchmark, TP8 text smoke, and wrapper
     parity artifacts before running quantized lanes.

2. Build fp8/fp4 activation experiments.
   - Add opt-in variants for fp8-act and fp4-act only where a fused consumer can
     use them.
   - Do not add post-quant kernels that only increase launch count.
   - Compare vLLM's corresponding quantized activation implementation before
     writing local code.

3. Build INT8 Tensor Core MoE experiment.
   - Keep a narrow first target: routed expert W13/W2 or the exact V2 MoE plan's
     dominant GEMM-like segment.
   - Preserve exact V2 route metadata, workspace, and reduce boundaries where
     possible.
   - Add explicit oracle checks for route output and final layer output.

4. Decide promotion or rejection.
   - Promote nothing by default in this target unless quality and performance
     gates pass.
   - Record rejected lanes with evidence so later threads do not repeat the same
     experiment.

## Quality Gates

- Wrapper parity against the bf16-direct baseline for each changed kernel.
- Logits/top-k comparison on fixed synthetic inputs.
- TP8 page-size-256 text smoke on simple Chinese and English prompts.
- No obvious乱码, prompt echo regression, or repeated-symbol degeneration.
- For TF32 experiments, report max/mean error and text-smoke impact separately.

## Performance Gates

- Each quantized activation lane must improve at least one target workload
  without regressing the 4096/1024/batch4 macro benchmark.
- INT8 MoE must show a meaningful MoE microbench win and at least 1.3x E2E
  improvement over the best exact path before further investment.
- Any promoted opt-in lane must be measured on the official 4096/1024/batch4
  victory line.

## Non-Goals

- Do not change the default exact path's activation precision.
- Do not silently downcast model-original fp32 work.
- Do not copy vLLM quantized behavior blindly when mini's accuracy target is
  different.
- Do not introduce DeepGEMM or sm90/sm100-only dependencies for sm80.
