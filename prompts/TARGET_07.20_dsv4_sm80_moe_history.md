# TARGET 07.20: DSV4 SM80 MoE History

## Status

Completed history merge for the MoE optimization chain.

This file summarizes:

- `TARGET_07.3_dsv4_sm80_moe_v2_exact.md`
- `TARGET_07.35_dsv4_sm80_post_moe_reparity.md`
- `TARGET_07.36_dsv4_sm80_vllm_fused_moe_runner_adapt.md`
- `TARGET_07.37_dsv4_sm80_moe_backend_identification.md`
- `TARGET_07.38_dsv4_sm80_moe_exact_backend_adapt.md`
- `TARGET_07.39_dsv4_sm80_marlin_custom_op_bridge.md`
- `TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port.md`

The original prompts now live under `prompts/archive/target07/` as archival
details.  Use this file for project history and decision context.

## Motivation

After the foundation phase, MoE routed expert execution was the largest clear
gap.  The early mini path spent too much time in FP4 expert handling and
fragmented per-expert execution.  On A100/sm80 there is no native FP4 Tensor
Core path, so the practical question became:

- can mini use a vLLM-style exact W4A16/MXFP4 expert backend;
- can it do so without depending on vLLM at runtime;
- does the resulting MoE path remain the top bottleneck after integration?

## Timeline And Conclusions

### TARGET 07.3: Exact MoE V2

Artifacts:

- `performance_milestones/target07_moe_v2/`

Conclusion:

- Introduced a MoE execution plan, per-layer workspace, and bf16-output SwiGLU
  cut.
- Correctness passed, but 4096/1024/batch4 decode throughput barely moved.
- This showed that wrapper-level MoE cleanup was not enough.

### TARGET 07.35: Post-MoE Re-Parity

Artifacts:

- `performance_milestones/target07_post_moe_reparity/`

Conclusion:

- Re-profiled after MoE V2.
- Decided not to keep polishing peripheral MoE structure.
- Selected vLLM FusedMoE runner adaptation as the next evidence target.

### TARGET 07.36: vLLM FusedMoE Runner Shape

Artifacts:

- `performance_milestones/target07_vllm_fused_moe_runner/`

Conclusion:

- Implemented a mini-owned vLLM-shaped FusedMoE runner boundary.
- Correctness and TP8 text smoke passed.
- Macro gain was only about `+0.16%`.
- The real gap was not the Python/runner shape; it was the expert backend.

### TARGET 07.37: MoE Backend Identification

Artifacts:

- `performance_milestones/target07_moe_backend_identification/`

Conclusion:

- vLLM DeepSeek V4 sm80 MoE backend was identified as Marlin-family
  MXFP4/W4A16 with bf16/fp16 activations.
- This was not an activation precision lane; it was an exact expert backend
  choice for the model's MXFP4 weights.

### TARGET 07.38: Direct Backend Adapt Attempt

Artifacts:

- `performance_milestones/target07_moe_exact_backend_adapt/`

Conclusion:

- Direct adaptation stopped at a precise blocker: mini did not own the required
  Marlin custom-op surface, including `gptq_marlin_repack` and
  `_moe_C::moe_wna16_marlin_gemm`.
- The opt-in guard was made explicit so the Marlin path could not silently fall
  back to grouped FP4.

### TARGET 07.39: vLLM Marlin Bridge Feasibility

Artifacts:

- `performance_milestones/target07_marlin_custom_op_bridge/`

Conclusion:

- The locally installed vLLM compiled ops could be imported and called.
- Synthetic DSV4-like MoE probes passed and showed large speedups over mini's
  old grouped FP4 path.
- The bridge stayed probe-only.  The right next step was a mini-owned csrc port,
  not a runtime dependency on vLLM.

### TARGET 07.391: mini-owned Marlin WNA16 csrc Port

Artifacts:

- `performance_milestones/target07_marlin_wna16_csrc_port/`

Conclusion:

- mini now has a mini-owned `marlin_wna16` opt-in backend.
- It vendors the narrow Marlin WNA16 source surface, transforms/caches MXFP4
  expert weights, and runs without a vLLM runtime dependency.
- TP8 text smoke passed.
- 4096/1024/batch4 reached about `54.47 output tok/s` in the csrc-port
  milestone and `54.64 output tok/s` in the following post-Marlin reprofile.
- MoE stopped being the primary bottleneck.  Whole visible MoE was about `2%`
  wall share in the post-Marlin profile; the Marlin expert kernel itself was
  about `1.47%`.

## Current MoE Decision

Keep Marlin WNA16 as the strongest exact mini MoE backend:

```bash
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
```

Do not spend more TARGET 07 effort on MoE unless a fresh profile places MoE
back in the top two contributors.

## Lessons

- vLLM's backend identification was more valuable than local MoE wrapper
  experimentation.
- Directly depending on vLLM runtime is not necessary when a narrow csrc port is
  feasible.
- Exact weight-only quantized expert compute can be a major win without changing
  mini's default activation/cache precision policy.

## Do Not Continue Here Unless

- Marlin WNA16 correctness regresses;
- a new workload makes MoE a top-two bottleneck again;
- a future precision-lane target explicitly compares INT8 Tensor Core MoE
  against the Marlin WNA16 exact baseline.
