# TARGET 07.37 - DeepSeek V4 SM80 MoE Backend Identification

Date: 2026-07-01

## Conclusion

vLLM DeepSeek V4 on the tested SM80/A100 path selects the MXFP4 **Marlin**
MoE expert backend:

- Selector: `DeepseekV4FP8Config` -> `Mxfp4MoEMethod` -> `select_mxfp4_moe_backend`
- Selected backend: `MARLIN`
- Experts class: `vllm.model_executor.layers.fused_moe.fused_marlin_moe.MarlinExperts`
- Quant semantics: MXFP4 weights, unquantized bf16/fp16 activations (`W4A16`)
- Activation format: `Standard`
- Weight key: `kMxfp4Static`
- Activation key: `None`
- Probe device: `NVIDIA A100-SXM4-80GB`, CUDA, capability `(8, 0)`

This is an **exact_candidate**, not a precision lane. The old vLLM SM80 MoE
benefit is therefore most likely expert backend/layout/kernel quality rather
than default activation quantization.

Next target: **start TARGET 07.38 exact expert backend adaptation**.

## Artifacts

- Script: `performance_milestones/target07_moe_backend_identification/scripts/probe_vllm_mxfp4_moe_backend.py`
- Raw probe JSON: `performance_milestones/target07_moe_backend_identification/raw/vllm_mxfp4_backend_probe_sm80.json`
- Compact summary JSON: `performance_milestones/target07_moe_backend_identification/summaries/backend_identification_summary.json`

Probe command:

```bash
/workspace/venvs/vllm-dsv4/bin/python \
  /workspace/mini-sglang/performance_milestones/target07_moe_backend_identification/scripts/probe_vllm_mxfp4_moe_backend.py \
  --output /workspace/mini-sglang/performance_milestones/target07_moe_backend_identification/raw/vllm_mxfp4_backend_probe_sm80.json \
  --summary-output /workspace/mini-sglang/performance_milestones/target07_moe_backend_identification/summaries/backend_identification_summary.json \
  --pretty
```

The probe ran successfully after one focused serialization fix. No vLLM-side
backend blocker remains for this identification target. No microbench was run:
the selector/support matrix already answered the target question, and the stop
condition was reached.

## vLLM Backend Selection Map

Actual DeepSeek V4 path:

1. `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
   maps `FusedMoE` layers to `Mxfp4MoEMethod` through
   `DeepseekV4FP8Config.get_quant_method`.
2. `select_mxfp4_moe_backend` chooses an activation format from the MoE
   parallel config. For TP8, non-EP, non-NIXL/non-DeepEP-LL this is `Standard`.
3. Auto priority for this path is:
   `FLASHINFER_TRTLLM_MXFP4_MXFP8`,
   `DEEPGEMM_MXFP4`,
   `MARLIN`,
   `BATCHED_MARLIN`.
4. Each backend is tested with:
   `weight_key=kMxfp4Static`,
   `activation_key=_backend_activation_key(backend)`,
   `activation_format=Standard`,
   and `experts_cls.is_supported_config(...)`.
5. On A100/SM80, FlashInfer TRTLLM MXFP8 and DeepGEMM are rejected by device
   support. `MARLIN` is the first supported backend.

Relevant env snapshot from the probe:

- `VLLM_MXFP4_USE_MARLIN`: unset
- `VLLM_MARLIN_INPUT_DTYPE`: unset
- `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16`: unset
- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8`: unset
- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS`: unset
- `VLLM_HAS_FLASHINFER_CUBIN`: unset
- `VLLM_BATCH_INVARIANT`: unset/false

So the result is the natural `auto` selector, not an env override.

## Selected, Supported, Rejected, Deferred

| Backend | Status on SM80 TP8 Standard | Class | Category | Reason |
| --- | --- | --- | --- | --- |
| `MARLIN` | selected | `MarlinExperts` | `exact_candidate` | CUDA capability >= 7.5, supports `kMxfp4Static` with activation key `None`, Standard activation format. |
| `BATCHED_MARLIN` | rejected for this config | `BatchedMarlinExperts` | `exact_candidate` only for batched activation format | Requires `BatchedExperts` activation format, not the TP8 standard path. |
| `DEEPGEMM_MXFP4` | rejected | `DeepGemmFP4Experts` | `precision_lane` and SM100-only | Uses FP8 dynamic 128 activation key; current device rejected. |
| `FLASHINFER_TRTLLM_MXFP4_MXFP8` | rejected | TRTLLM MXFP4 experts | `precision_lane` and SM100-only | Requires MXFP8 activation lane and SM100-family FlashInfer TRTLLM support. |
| `FLASHINFER_TRTLLM_MXFP4_BF16` | rejected when forced | TRTLLM MXFP4 experts | exact but `defer_or_reject` | BF16 activation variant, but SM100-family only. Not in actual DeepSeek V4 auto priority. |
| `FLASHINFER_CUTLASS_MXFP4_BF16` | rejected when forced | `FlashInferExperts` | exact but `defer_or_reject` | BF16 activation variant, but SM90+ CUTLASS/FlashInfer path, not SM80. |
| `FLASHINFER_CUTLASS_MXFP4_MXFP8` | rejected when forced | `FlashInferExperts` | `precision_lane` and SM100-only | MXFP8 activation lane; current device rejected. |
| `TRITON` | rejected when forced | GPT-OSS Triton experts | `defer_or_reject` | The available Triton kernel path is not SM80; probe also saw a local `triton_kernels` import mismatch in the GPT-OSS reference selector. |
| `TRITON_UNFUSED` | rejected when forced | GPT-OSS Triton experts | `defer_or_reject` | Same SM80 rejection. |
| `AITER` | rejected when forced | `AiterExperts` | `defer_or_reject` | ROCm backend, not CUDA SM80. |
| `XPU` | rejected when forced | `XPUExpertsMXFp4` | `defer_or_reject` | XPU backend, not CUDA SM80. |
| `EMULATION` | supported only when explicitly forced | `OCP_MXQuantizationEmulationTritonExperts` | `defer_or_reject` | Not in actual DeepSeek V4 auto priority, not a performance backend, and forced quant config is `mxfp4` activation plus `mxfp4` weights (`W4A4` emulation), not the default exact lane. |

The compact summary reports `supported_for_standard_tp8 = ["MARLIN", "EMULATION"]`.
Only `MARLIN` is promoted as a usable candidate here; `EMULATION` is deliberately
deferred/rejected because it is forced, non-default precision semantics, and not
the old serving backend likely responsible for the gap.

## Backend Semantics

### Weight and scale layout before kernel conversion

`Mxfp4MoEMethod.create_weights` registers:

- `w13_weight`: `[num_experts, 2 * intermediate_size_per_partition, hidden_size // 2]`, `uint8`
- `w13_weight_scale`: `[num_experts, 2 * intermediate_size_per_partition, hidden_size // 32]`, `uint8`
- `w2_weight`: `[num_experts, hidden_size, intermediate_size_per_partition // 2]`, `uint8`
- `w2_weight_scale`: `[num_experts, hidden_size, intermediate_size_per_partition // 32]`, `uint8`

For the TP8 DSV4 shape used by the probe:

- `hidden_size = 4096`
- `intermediate_size_per_partition = 256`
- Marlin rounding leaves the shape unchanged: `hidden=4096`, `intermediate=256`

### Weight transform after loading

`Mxfp4MoEMethod.process_weights_after_loading` calls `_setup_kernel`, which calls
`convert_weight_to_mxfp4_moe_kernel_format`. For `MARLIN` and `BATCHED_MARLIN`
this dispatches to `prepare_moe_mxfp4_layer_for_marlin`.

That means the runtime backend is not using the raw HuggingFace MXFP4 layout
directly. It repacks the per-expert weights/scales into Marlin's WNA16 format
before inference.

### Activation and output dtype

The selected quant config is:

```json
{
  "quant_dtype": null,
  "weight_quant_dtype": "mxfp4",
  "is_quantized_activation": false,
  "use_mxfp4_w4a16": true
}
```

`MarlinExpertsBase` asserts W4A16-style quant configs and `fused_marlin_moe`
asserts the incoming hidden states are `torch.float16` or `torch.bfloat16`.
For this DSV4 path the probe uses bf16 input activations. `VLLM_MARLIN_INPUT_DTYPE`
was unset, so Marlin does not quantize activations through the optional int8/fp8
input path.

Output is allocated as the same dtype as the hidden states in the Marlin path
and reduced to `[M, K]` before the modular finalizer. `MarlinExperts` returns
`TopKWeightAndReduceNoOP`, so finalize does not apply a second top-k reduction.

### Route metadata and workspace shape

For `MarlinExperts`:

- Activation format: `Standard`, namely `[num_tokens, hidden_dim]`
- Route inputs: `topk_ids [M, topk]`, `topk_weights [M, topk]`
- Route metadata is built by `moe_align_block_size(...)`, producing
  `sorted_token_ids`, `expert_ids`, and `num_tokens_post_padded`
- Workspace shapes requested by `MarlinExperts.workspace_shapes`:
  - `workspace13`: `(M * topk, max(N, K))`
  - `workspace2`: `(M * topk * max(2 * N, K),)`
  - `output`: `(M, K)`
- The apply method swaps the two workspaces when calling `fused_marlin_moe`, to
  preserve the expected output allocation behavior.

### Top-k weights

Marlin passes `topk_weights` directly into `ops.moe_wna16_marlin_gemm`.

- If `apply_router_weight_on_input=True`, top-k weights are multiplied in the
  W13 GEMM.
- In the default path, `apply_router_weight_on_input=False`, so top-k weights
  are multiplied in the W2 GEMM.
- The routed outputs are then summed from `[M, topk, K]` to `[M, K]` inside
  `fused_marlin_moe`, before the NoOP finalizer.

This matches the important mini semantic that route weights are applied before
the returned routed MoE output, but not the implementation/layout.

## Mini vs vLLM Candidate Delta

Current mini exact MoE path:

- `DSV4FusedRoutedExperts.forward` calls
  `dsv4_kernel.moe_route_dispatch_bf16_grouped`.
- The grouped path calls mini Triton `grouped_fp4_moe`.
- The dominant kernels remain `_grouped_fp4_w13_kernel` and
  `_grouped_fp4_linear_kernel`, with route weighting in the fused compute path
  and `_moe_route_sum_kernel` reducing routed outputs.

vLLM selected candidate:

- Repacked Marlin WNA16 expert GEMMs through `ops.moe_wna16_marlin_gemm`
- Marlin-specific packed weight and scale layout after load
- Marlin route metadata from `moe_align_block_size`
- Marlin workspace contract and output/no-op finalize behavior
- Same broad default precision class: MXFP4 weights, bf16/fp16 activation

So mini is not missing a runner wrapper. It is missing the exact expert backend
implementation/layout used by vLLM's SM80 path.

## Static Source Anchors

- DeepSeek V4 quant method: `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:66`, `:104`, `:736`
- Actual MXFP4 priority and activation keys:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:214`, `:232`, `:425`
- MXFP4 weight registration and post-load setup:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/mxfp4.py:505`, `:599`, `:697`
- Marlin transform dispatch:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:973`
- Quant config semantics:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:1169`
- Marlin support, workspace, apply, and top-k handling:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py:546`, `:581`, `:591`, `:672`, `:685`, `:717`
- Mini grouped FP4 kernels:
  `python/minisgl/models/deepseek_v4.py:819`,
  `python/minisgl/kernel/deepseek_v4.py:2083`,
  `python/minisgl/kernel/triton/deepseek_v4.py:994`,
  `python/minisgl/kernel/triton/deepseek_v4.py:1169`,
  `python/minisgl/kernel/triton/deepseek_v4.py:1278`,
  `python/minisgl/kernel/triton/deepseek_v4.py:1301`

## Decision

Proceed to **TARGET 07.38 exact expert backend adaptation**.

Rationale:

- The selected vLLM SM80 backend is `MARLIN`, an exact W4A16 candidate.
- It does not require default activation quantization, INT8, MXFP8, FP8 cache,
  or a vLLM runtime dependency.
- The fresh 07.36 Nsight evidence still points at grouped FP4 W13/W2 as the
  dominant cost, so an exact expert backend adaptation directly targets the
  largest remaining gap.
- Do not move to TARGET 07.4 precision lanes yet. The selected old-vLLM-style
  MoE benefit does not require MXFP8/FP8/MXFP4 activation by default.
- Do not move to attention/cache/indexer yet. MoE has a concrete exact backend
  candidate and remains dominant.
- Open an exact local expert-kernel target only if 07.38 shows that Marlin
  cannot be adapted through a narrow surface or fails to give a meaningful
  routed-MoE speedup.
