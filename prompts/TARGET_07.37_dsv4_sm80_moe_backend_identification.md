# TARGET 07.37: DSV4 sm80 MoE Backend Identification

## Goal

Identify the old vLLM DeepSeek V4 sm80 MoE expert backend and decide whether
mini-sglang should adapt an exact W4A16 backend, move to TARGET 07.4 precision
lanes, or leave MoE and target attention/cache next.

This is a decision and evidence target. Do not implement a large backend port
here. The point is to avoid another long thread that builds the wrong backend.

## Start Point

TARGET 07.36 is complete. It adapted the vLLM-shaped FusedMoE runner boundary
into mini, but did not improve performance:

- 4096/1024/batch4 exact runner: about `17.8289` output tok/s;
- runner macro gain over 07.35: about `+0.16%`;
- DSV4-like routed-MoE microbench did not improve;
- fresh 4096/128 Nsight still shows grouped FP4 W13 and W2 as dominant:
  `_grouped_fp4_w13_kernel` about `46.781s` and
  `_grouped_fp4_linear_kernel` about `31.700s`.

Therefore, do not continue MoE wrapper/runner cleanup. The next question is
which expert backend vLLM actually uses on sm80 and whether its advantage is an
exact backend/layout win or a precision-lane win.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- Runner milestone:
  `performance_milestones/target07_vllm_fused_moe_runner/README.md`
- Post-MoE re-parity:
  `performance_milestones/target07_post_moe_reparity/README.md`
- Subgraph parity:
  `performance_milestones/target07_subgraph_parity/README.md`
- mini runner/model: `python/minisgl/models/deepseek_v4.py`
- mini MoE kernels: `python/minisgl/kernel/triton/deepseek_v4.py`
- vLLM source root: `/workspace/vllm-dsv4-docker`
- vLLM MXFP4 selector:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
- vLLM FusedMoE backends:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- vLLM MXFP4/Marlin utilities:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/utils/`

## Plan

1. Create `performance_milestones/target07_moe_backend_identification/`.
   - Store small summaries under `summaries/`.
   - Store scripts/probes under `scripts/`.
   - Store large or external artifacts under `raw/` as symlinks when appropriate.
   - Write a README with backend map, measurements, and final decision.

2. Identify the vLLM sm80 backend.
   - Inspect vLLM backend selection for MXFP4 MoE: priority order, explicit env
     overrides, supported device capability, activation dtype, and weight format.
   - Determine whether DeepSeek V4 Flash uses GPT-OSS MXFP4, generic MXFP4,
     Marlin, Batched Marlin, Triton, FlashInfer CUTLASS/TRTLLM, or fallback
     emulation on A100/sm80.
   - Prefer an executable probe in the vLLM virtualenv that prints selected
     backend, experts class, quant config, activation format, and rejected
     backends with reasons.
   - If the probe cannot run, record the exact blocker and provide a static source
     conclusion from `is_supported_config` checks and env/backend selection code.

3. Map backend semantics.
   - Record weight layout, scale layout, transform-after-loading path, workspace
     shape, route metadata shape, activation dtype, output dtype, and where
     top-k weights are applied.
   - Classify the backend as one of:
     - `exact_candidate`: W4A16 or equivalent, bf16 activations, no activation
       quantization required for the main speed path;
     - `precision_lane`: requires MXFP8/FP8/MXFP4 activation quantization or other
       non-default precision to explain the speed;
     - `defer_or_reject`: sm90/sm100-only, unavailable dependency/cubin, or too
       large to port as a narrow mini backend.

4. Run only lightweight comparison experiments.
   - Reuse existing mini runner and current grouped FP4 microbench as the mini
     baseline.
   - If feasible, run a vLLM-side synthetic backend microbench at DSV4-like
     shapes: T=4 and T=4096, topk=6, experts=256, hidden=4096,
     local intermediate=256.
   - If vLLM cannot run the backend standalone, record why and use macro/profile
     evidence plus code-path evidence.
   - Do not implement a mini backend port in this target.

5. Make the next-target decision.
   - If there is a feasible exact W4A16 backend with expected MoE microbench win
     of at least `1.5x`, open TARGET 07.38.
   - If the best vLLM backend advantage depends on activation quantization or
     MXFP8/FP8/MXFP4 activation semantics, move to TARGET 07.4.
   - If no vLLM backend is portable and exact W13/W2 remains dominant, write an
     exact local expert-kernel plan instead of copying vLLM.
   - If fresh data shows attention/cache/indexer is now above MoE, open a
     dedicated attention/cache/indexer target.

## Stop Conditions

Stop as soon as the backend is classified and the next target is selected. Do
not keep expanding probes after they answer the decision.

Hard stops:

- selected vLLM backend and precision semantics are known;
- probe is blocked after one focused environment fix attempt, and static source
  evidence is sufficient to classify the backend;
- no candidate backend has at least `1.5x` expected W13/W2 improvement;
- backend requires default precision-policy changes, meaning TARGET 07.4 is the
  correct next target.

## Done Criteria

- README exists under `performance_milestones/target07_moe_backend_identification/`.
- vLLM backend map lists selected, supported, rejected, and deferred backends.
- mini current best backend is compared against the vLLM candidate at the
  semantic and measurement level.
- Final decision is exactly one of:
  - start TARGET 07.38 exact expert backend adaptation;
  - move to TARGET 07.4 precision lanes;
  - open exact local expert-kernel backend target;
  - open attention/cache/indexer target.

## Non-Goals

- Do not port backend kernels here.
- Do not change mini default precision.
- Do not add vLLM as a runtime dependency.
- Do not introduce INT8, FP8 activation, MXFP8 activation, or MXFP4 activation
  into mini in this target.
- Do not continue MoE runner/wrapper cleanup.
