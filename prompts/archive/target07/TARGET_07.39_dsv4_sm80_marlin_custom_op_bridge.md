# TARGET 07.39: DSV4 sm80 Marlin Custom-Op Bridge Feasibility

## Goal

Determine whether mini-sglang can use the locally installed vLLM DeepSeek V4
SM80 Marlin MXFP4 W4A16 custom ops as an experimental bridge, before opening any
larger mini-owned CUDA-extension port.

This target follows TARGET 07.38, which rejected direct Marlin adaptation
because mini does not own the required custom-op surface. The purpose here is
not to promote a vLLM runtime dependency. It is to answer whether the compiled
vLLM ops already present in `/workspace/venvs/vllm-dsv4` can execute the DSV4
MoE expert shape and whether the result is fast enough to justify a narrow csrc
port target.

## Non-Negotiable Precision Policy

- Default mini remains exact bf16-direct with current grouped FP4 experts.
- Do not introduce activation quantization, INT8, MXFP8/FP8 cache, or precision
  lane behavior.
- The bridge may import vLLM compiled ops only in milestone probe scripts or an
  explicitly named experimental backend.
- Do not make vLLM a mini runtime dependency or promoted path.
- Do not silently fall back to grouped FP4 when a Marlin bridge was requested.

## Inputs

Read first:

- `performance_milestones/target07_moe_exact_backend_adapt/README.md`
- `performance_milestones/target07_moe_exact_backend_adapt/summaries/marlin_feasibility_audit.json`
- `performance_milestones/target07_moe_backend_identification/README.md`
- `performance_milestones/target07_vllm_fused_moe_runner/README.md`
- `prompts/TARGET_07.38_dsv4_sm80_moe_exact_backend_adapt.md`
- mini MoE code:
  - `python/minisgl/models/deepseek_v4.py`
  - `python/minisgl/kernel/deepseek_v4.py`
  - `python/minisgl/kernel/triton/deepseek_v4.py`
- vLLM Marlin references:
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`
  - `/workspace/vllm-dsv4-docker/csrc/quantization/marlin/gptq_marlin_repack.cu`
  - `/workspace/vllm-dsv4-docker/csrc/moe/marlin_moe_wna16/ops.cu`
  - `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py`

## Plan

1. Create `performance_milestones/target07_marlin_custom_op_bridge/`.
   Record README, scripts, summaries, raw probe output, and command lines.

2. Write a minimal probe using `/workspace/venvs/vllm-dsv4/bin/python`.
   Verify import/registration and direct calls for:
   - `vllm._custom_ops.gptq_marlin_repack`
   - `vllm._custom_ops.moe_wna16_marlin_gemm`

3. Validate layout compatibility.
   Record:
   - mini current packed MXFP4 `w13`/`w2` layout;
   - vLLM raw MXFP4 layout expected by `prepare_moe_mxfp4_layer_for_marlin`;
   - Marlin WNA16 packed qweight/scale shapes after transform;
   - top-k weight application position;
   - route metadata compatibility:
     `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`;
   - workspace/intermediate cache contract.

4. Run synthetic DSV4-like Marlin MoE calls.
   Required shape:
   - hidden = 4096
   - local intermediate = 256
   - experts = 256
   - topk = 6
   - tokens = 4 first, then tokens = 4096 if the small case passes

5. Compare against current mini grouped FP4 for `decode_real` and
   `prefill_real`. Record total time and stage information where possible.

6. Decide:
   - if bridge runs and routed-MoE is clearly faster, open a vendor/narrow csrc
     port target;
   - if bridge runs but does not improve enough, stop Marlin and move to local
     exact W4A16 kernel plan;
   - if bridge fails on ABI/import/layout/signature/route/workspace/dtype or
     numerics, move to local exact W4A16 kernel plan;
   - if the win requires activation quantization or another precision lane,
     move to TARGET 07.4.

## Done Criteria

- Import/compile probe result recorded.
- `gptq_marlin_repack` and `moe_wna16_marlin_gemm` direct call results
  recorded, or precise blocker recorded.
- Synthetic DSV4-like Marlin call result for T=4 and, if possible, T=4096
  recorded.
- Current grouped FP4 comparison for decode/prefill real shapes recorded.
- Silent fallback status recorded.
- vLLM runtime dependency status recorded.
- README final decision names the next target.

## Stop Conditions

Stop this target when bridge feasibility is answered. Do not begin a csrc port
inside this target unless a separate prompt/target is opened after a positive
bridge result.
