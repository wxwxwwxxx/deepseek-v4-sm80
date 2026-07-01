# TARGET 07.391: DSV4 SM80 Marlin WNA16 Csrc Port

## Goal

Vendor or reimplement the minimum mini-owned Marlin WNA16 MoE custom-op surface
needed to replace the current grouped FP4 expert backend for DeepSeek V4 SM80,
without making vLLM a mini runtime dependency.

TARGET 07.39 proved that the locally installed vLLM Marlin MXFP4 W4A16 custom
ops can be imported and called on the current A100 SM80 environment, and that
synthetic DSV4-like routed MoE is substantially faster than mini's grouped FP4
path. This target turns that evidence into a mini-owned compiled surface.

## Required Inputs

Read first:

- `performance_milestones/target07_marlin_custom_op_bridge/README.md`
- `performance_milestones/target07_moe_exact_backend_adapt/README.md`
- `performance_milestones/target07_moe_backend_identification/README.md`
- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- mini csrc/build path:
  - `python/minisgl/kernel/utils.py`
  - `python/minisgl/kernel/pynccl.py`
  - `python/minisgl/kernel/csrc/`
- vLLM source references:
  - `/workspace/vllm-dsv4-docker/csrc/quantization/marlin/`
  - `/workspace/vllm-dsv4-docker/csrc/moe/marlin_moe_wna16/`

## Scope

Implement only the exact W4A16 path needed by DSV4 SM80:

- MXFP4 weights
- E8M0 scales
- bf16/fp16 activations
- no activation quantization
- no INT8 Tensor Core path
- no MXFP8/FP8 cache
- no vLLM runtime dependency

Prefer a minimal surface:

- `gptq_marlin_repack` or equivalent repack support;
- `moe_wna16_marlin_gemm` or equivalent W4A16 expert GEMM;
- MXFP4 E8M0 scale transform compatible with vLLM
  `prepare_moe_mxfp4_layer_for_marlin`;
- route metadata adapter compatible with the Marlin WNA16 contract;
- cached transformed weights, so request-time execution does not repack;
- explicit unsupported errors for shape, dtype, layout, or build misses.

## Work Plan

1. Create `performance_milestones/target07_marlin_wna16_csrc_port/` with
   `README.md`, `scripts/`, `summaries/`, and `raw/`.

2. Audit the Marlin source dependency graph and record the exact files needed
   for a mini-owned build. Preserve Apache-2.0 attribution for copied or adapted
   vLLM/Marlin source.

3. Build a minimal mini extension surface. The first acceptable cut can be an
   optional PyTorch CUDA extension if it is mini-owned and does not import vLLM
   at runtime; follow-up cleanup may migrate it into mini's existing csrc build
   style if that is mechanically better.

4. Reproduce the TARGET 07.39 synthetic benchmark with mini-owned ops:

   - hidden = `4096`
   - local intermediate = `256`
   - experts = `256`
   - top-k = `6`
   - tokens = `4`
   - tokens = `4096`

   Compare against both the 07.39 vLLM bridge numbers and mini grouped FP4.
   Performance should not materially regress from the vLLM bridge.

5. Integrate a formal backend selector, for example:

   ```bash
   MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
   ```

   Default remains grouped FP4. If the Marlin backend cannot run, it must raise
   an explicit unsupported error. It must not silently fall back to grouped FP4.

6. Add correctness smoke and microbench coverage. Run macro benchmark and Nsight
   only after the synthetic port is executable and close to the 07.39 bridge.

## Stop Conditions

Stop this target and record the precise blocker if either condition is met:

- the csrc port cannot compile or link after one concentrated dependency
  narrowing pass;
- the mini-owned synthetic kernel runs but cannot approach the 07.39 bridge
  performance without substantial unrelated optimization.

Do not open TARGET 07.4 unless the evidence shows the remaining win requires
activation quantization, FP8 activation/cache behavior, INT8 Tensor Core
semantics, or another precision-lane change.

## Done Criteria

- Mini-owned Marlin WNA16 custom-op surface exists or a precise blocker is
  recorded.
- vLLM is not a mini runtime dependency.
- Weight transform/cache behavior is implemented or the missing piece is named.
- Backend selector integration is explicit and no-silent-fallback safe.
- Synthetic T=4 and T=4096 results are recorded, or the compile/link/runtime
  blocker preventing them is recorded.
- README summarizes:
  - ported vLLM/Marlin components;
  - mini/vLLM interface and layout differences;
  - synthetic and real-model performance;
  - correctness results;
  - whether the backend can proceed to end-to-end macro optimization;
  - if failed, the smallest viable remediation path.
