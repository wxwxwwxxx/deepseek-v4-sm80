# TARGET 07.3: DSV4 sm80 MoE Exact V2

## Goal

Build the next exact grouped MoE path after TARGET 07.1, TARGET 07.2, and
TARGET 07.25 show that MoE remains a top contributor to the mini-vs-vLLM gap.

This target is complete when an exact MoE V2 variant measurably improves the
4096/1024/batch4 workload without correctness regression and has enough
microbench/profile evidence to decide whether INT8 Tensor Core MoE should be
opened as a separate opt-in target.

Precision policy for this target: optimize the exact `bf16-direct` path first.
Do not add activation quantization to MoE V2. Keep model-original fp32 work at
its original precision unless a separate TF32 experiment is explicitly opened.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- subgraph parity target:
  `prompts/TARGET_07.25_dsv4_sm80_vllm_subgraph_parity.md`
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
- communication, graph, and attention-boundary work in TARGET 07.2 improved the
  best exact path but did not close the vLLM gap;
- this target should start only after TARGET 07.25 maps mini/vLLM subgraphs and
  confirms MoE is the right next implementation target.

## TARGET 07.25 Evidence To Carry Forward

The current post-07.2 best exact variant reaches about 25.3 output tok/s on
4096/1024/batch4. TARGET 07.25 ranks MoE routed experts and the MoE execution
boundary as the top remaining bottleneck.

Important measurements from
`performance_milestones/target07_subgraph_parity/README.md`:

- mini rank0 4096/128 profile: `_grouped_fp4_w13_kernel` plus
  `_grouped_fp4_linear_kernel` take about 47.25s of 76.45s summed kernel time;
- mini standalone routed MoE probe: about 2.27ms for T=4 and 98.23ms for
  T=4096;
- vLLM's exact comparable MoE boundary could not be safely benchmarked
  standalone because it depends on engine `FusedMoE`, transformed MXFP4 weights,
  static forward context, router, and shared-expert state;
- vLLM disables `DeepseekV4MegaMoEExperts` on sm80, so MegaMoE is not the
  A100 route;
- vLLM's MXFP4/FP8 precision semantics are a deferred precision-lane topic, not
  part of this exact default target.

Expected outcome for this target: a serious MoE V2 cut should either improve
4096/1024 exact throughput by at least 1.3x over the 25.3 tok/s post-07.2
baseline, reduce MoE W13/W2 summed kernel time by at least 2x, or prove with a
fresh profile that another subgraph has become the new top bottleneck.

## Plan

1. Create a mini-side MoE execution plan and baseline harness.
   - Add an internal `DSV4MoEPlan` or equivalent local structure for route
     metadata, token/expert layout, workspace ownership, and finalize/reduce
     policy.
   - Keep the public model behavior unchanged.
   - Keep V1 as fallback for the V2 variant.
   - Before major edits, record the exact post-07.2 baseline command, variant,
     text smoke artifact, 4096/1024 macro result, and 4096/128 profile used for
     comparison.

2. Match vLLM's useful abstractions without adding vLLM runtime dependency.
   - Compare vLLM `FusedMoE` prepare/fused-experts/finalize boundaries against
     mini V1.
   - Use TARGET 07.25 paired MoE subgraph microbench results to choose the
     first implementation cut.
   - Port/adapt only the parts that reduce kernel count, workspace churn, or
     reduce-boundary ambiguity.
   - Preserve Apache-2.0 attribution if code is copied.

3. First implementation cut: execution boundary and workspace.
   - Consolidate route metadata, dispatch state, expert-token layout,
     intermediate buffers, finalize, and reduce policy into the MoE plan.
   - Reduce per-step allocation/materialization and PyTorch small-kernel
     participation around route metadata and route sum.
   - Keep routed and shared expert outputs rank-local until the intended single
     TP reduce boundary.
   - Measure before adding deeper kernel math changes.

4. Second implementation cut: exact grouped FP4 compute, only if still top.
   - Reduce intermediate writes between W13, activation, route weighting, and
     W2 where possible.
   - Reuse persistent workspaces instead of allocating or materializing per
     step.
   - Use table/LUT-based FP4/E8M0 scale decode where applicable; avoid exp-like
     math in hot kernels.
   - Keep output numerics exact relative to the current V1 path within existing
     tolerances.
   - Do not introduce fp8/fp4 activation quantization in this target.

5. Shared-expert scheduling and overlap, only with evidence.
   - Study vLLM `SharedExperts` aux-stream overlap and runner integration.
   - Add mini-side overlap only if shared experts or route/shared combine remain
     visible in profile after the main routed-expert cut.
   - Record stream dependencies and wall-time gain, not just summed kernel time.

6. Reduce MoE-adjacent small kernels.
   - Fuse route metadata transforms when they show up in nsys.
   - Avoid PyTorch elementwise/copy/reduce around gate, shared expert, and
     final route sum when a small local kernel can replace multiple launches.
   - Do not chase small kernels outside MoE unless TARGET 07.35 re-ranks them as
     a top bottleneck.

7. Keep INT8 separate.
   - Document expected INT8 Tensor Core opportunity and accuracy risks.
   - Do not implement INT8 inside this exact target; use TARGET 07.4 or a later
     opt-in target for that lane.

## Stop Conditions

Stop TARGET 07.3 and move to TARGET 07.35 re-parity when any of these is true:

- exact 4096/1024/batch4 output throughput exceeds 114.07 tok/s and TP8
  page-size-256 text smoke passes;
- MoE routed W13/W2 is no longer a top-two contributor in the fresh 4096/128
  profile;
- a serious MoE V2 cut improves 4096/1024 output throughput by at least 1.3x or
  reduces MoE W13/W2 summed kernel time by at least 2x, meaning the next move
  should be chosen from a refreshed ranking;
- two consecutive MoE implementation cuts each produce less than 5% macro
  throughput gain and less than 10% routed-MoE subgraph improvement;
- the remaining proposed work is mainly attention/cache, communication,
  precision, or generic graph cleanup rather than MoE.

Only continue within TARGET 07.3 after a successful MoE cut if a fresh profile
shows MoE is still the dominant bottleneck and the next MoE change has a clear
expected E2E gain of at least 10%.

## Done Criteria

- V2 exact MoE passes wrapper parity, DSV4 forward tests, and TP8 text smoke.
- V2 has microbench evidence for route metadata, W13, activation/finalize, and
  W2 changes.
- V2 has 4096/128 nsys and 4096/1024 macro benchmark results.
- V2 improves exact 4096/1024 output throughput versus the best post-07.2 exact
  variant.
- The V2 implementation choices are traceable to TARGET 07.25 subgraph
  comparison results.
- TARGET 05.5 R&D matrix and TARGET 07 master doc are updated with results.
- TARGET 07.35 is selected as the next step unless the 114.07 tok/s win line is
  already passed and the remaining work is only stabilization.

## Non-Goals

- Do not make INT8 the default path.
- Do not add activation quantization to the exact MoE V2 path.
- Do not silently downcast model-original fp32 work.
- Do not depend on DeepGEMM or sm90/sm100-only kernels.
- Do not remove V1 or torch fallback paths.
- Do not optimize MoE before TARGET 07.25 confirms it is still a top
  bottleneck.
