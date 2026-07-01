# TARGET 07.3: DSV4 sm80 MoE Exact V2

## Goal

Build the next exact grouped MoE path after TARGET 07.1 and TARGET 07.2 clarify
how much gap remains after fair benchmarking, communication repair, and decode
CUDA graph.

This target is complete when an exact MoE V2 variant measurably improves the
4096/1024/batch4 workload without correctness regression and has enough
microbench/profile evidence to decide whether INT8 Tensor Core MoE should be
opened as a separate opt-in target.

Precision policy for this target: optimize the exact `bf16-direct` path first.
Do not add activation quantization to MoE V2. Keep model-original fp32 work at
its original precision unless a separate TF32 experiment is explicitly opened.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- sm80 kernel R&D record:
  `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md`
- precision-lane target:
  `prompts/TARGET_07.4_dsv4_sm80_precision_lanes.md`
- mini V1 milestone: `performance_milestones/v1_moe/README.md`
- mini DSV4 model: `python/minisgl/models/deepseek_v4.py`
- mini DSV4 wrappers: `python/minisgl/kernel/deepseek_v4.py`

vLLM references:

- DSV4 MoE wrapper:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- Fused MoE layer:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- Modular fused MoE:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- Routed expert capture support:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`

## Current V1 State

V1 enables `MINISGL_DSV4_SM80_V1_MOE=1`, which activates the v0 bf16 bundle plus
`MINISGL_DSV4_SM80_MOE_ROUTE=1`.

V1 wins:

- removes `dequant_fp4_weight` from the hot path;
- removes grouped optional-none skips;
- makes grouped MoE the default path for the V1 variant;
- rank-local routed/shared sum happens before one TP all-reduce.

Remaining profile evidence:

- grouped FP4 W13 and W2 still account for about 188s summed GPU time in the
  short nsys workload;
- PyTorch small kernels around MoE/shared experts remain large;
- communication cost may hide or amplify MoE cost, so this target should start
  after TARGET 07.2 unless the fair profile says MoE is still dominant.

## Plan

1. Create a mini-side MoE execution plan.
   - Add an internal `DSV4MoEPlan` or equivalent local structure for route
     metadata, token/expert layout, workspace ownership, and finalize/reduce
     policy.
   - Keep the public model behavior unchanged.
   - Keep V1 as fallback for the V2 variant.

2. Match vLLM's useful abstractions without adding vLLM runtime dependency.
   - Compare vLLM `FusedMoE` prepare/fused-experts/finalize boundaries against
     mini V1.
   - Port/adapt only the parts that reduce kernel count, workspace churn, or
     reduce-boundary ambiguity.
   - Preserve Apache-2.0 attribution if code is copied.

3. Optimize exact grouped FP4 compute.
   - Reduce intermediate writes between W13, activation, route weighting, and
     W2 where possible.
   - Reuse persistent workspaces instead of allocating or materializing per
     step.
   - Use table/LUT-based FP4/E8M0 scale decode where applicable; avoid exp-like
     math in hot kernels.
   - Keep output numerics exact relative to the current V1 path within existing
     tolerances.
   - Do not introduce fp8/fp4 activation quantization in this target.

4. Reduce MoE-adjacent small kernels.
   - Fuse route metadata transforms when they show up in nsys.
   - Avoid PyTorch elementwise/copy/reduce around gate, shared expert, and
     final route sum when a small local kernel can replace multiple launches.

5. Keep INT8 separate.
   - Document expected INT8 Tensor Core opportunity and accuracy risks.
   - Do not implement INT8 inside this exact target; use TARGET 07.4 or a later
     opt-in target for that lane.

## Done Criteria

- V2 exact MoE passes wrapper parity, DSV4 forward tests, and TP8 text smoke.
- V2 has microbench evidence for route metadata, W13, activation/finalize, and
  W2 changes.
- V2 has 4096/128 nsys and 4096/1024 macro benchmark results.
- V2 improves exact 4096/1024 output throughput versus the best post-07.2 exact
  variant.
- TARGET 05.5 R&D matrix and TARGET 07 master doc are updated with results.

## Non-Goals

- Do not make INT8 the default path.
- Do not add activation quantization to the exact MoE V2 path.
- Do not silently downcast model-original fp32 work.
- Do not depend on DeepGEMM or sm90/sm100-only kernels.
- Do not remove V1 or torch fallback paths.
- Do not optimize MoE before TARGET 07.1/07.2 evidence confirms it is still a
  top bottleneck.
