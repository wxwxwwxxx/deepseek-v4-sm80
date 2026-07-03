# vLLM Quantized-Linear Backend Parity

Date: 2026-07-02

Primary reference checkout: `/workspace/vllm-dsv4-docker`.

## Source Facts

| Fact | Source |
| --- | --- |
| DeepSeek V4 uses `DeepseekV4FP8Config`, which inherits `Fp8Config`, sets `is_scale_e8m0=True`, routes regular linear layers through the standard FP8 linear method, and maps checkpoint `.scale` tensors to `.weight_scale_inv`. | `vllm/model_executor/models/deepseek_v4.py` |
| `Fp8LinearMethod` selects an FP8 scaled-mm kernel through `init_fp8_linear_kernel`; for block-quant DeepSeek V4 weights on A100, the selected class is `MarlinFP8ScaledMMLinearKernel`. | `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/kernels/linear/__init__.py` |
| `MarlinFP8ScaledMMLinearKernel` processes block FP8 weights with `process_fp8_weight_block_strategy`, then calls `prepare_fp8_layer_for_marlin`. | `vllm/model_executor/kernels/linear/scaled_mm/marlin.py` |
| `prepare_fp8_layer_for_marlin` creates a persistent Marlin workspace, packs FP8 weights with `pack_fp8_to_int32`, repacks with `gptq_marlin_repack`, expands block/tensor scales into Marlin scale layout, permutes scales, and fuses FP8 exponent bias into scales. | `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` |
| vLLM Marlin FP8 on SM80 is W8A16 weight-only. Supplying an 8-bit activation dtype raises `Marlin W8A8 is not supported.` | `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` |
| `FBGEMMFp8LinearMethod` uses Marlin on GPUs without native FP8 support and deletes `input_scale_ub`; activations are not quantized on its Marlin route. | `vllm/model_executor/layers/quantization/fbgemm_fp8.py` |
| `torch._scaled_mm` FP8 paths require newer native FP8 hardware in vLLM's torch backend and are not an SM80 DeepSeek V4 projection route here. | `vllm/model_executor/kernels/linear/scaled_mm/pytorch.py`, availability smoke |
| vLLM generic INT8 scaled-mm selects `CutlassInt8ScaledMMLinearKernel` on CUDA and dynamically quantizes activations with `scaled_int8_quant`. vLLM's DeepSeek V4 source does not present this as an existing DSV4 projection backend. | `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`, `vllm/model_executor/layers/quantization/online/int8.py` |

## Candidate Table

| Lane | Backend candidate | Quant/dequant strategy | Activation policy | Weight/scale layout | A100 availability | Real DSV4 projection candidate | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | vLLM FP8 Marlin W8A16 block linear | One-time vLLM block processing, Marlin repack, scale expansion/permutation, exponent-bias fusion; replay uses `marlin_gemm`. | BF16 input, no activation quantization. | Checkpoint `float8_e4m3fn` weight plus `float8_e8m0fnu` block `weight_scale_inv` with `[ceil(N/128), ceil(K/128)]`. | Available; selected `MarlinFP8ScaledMMLinearKernel`. | Yes for dense DSV4 projection owners. `wo_a` grouped two-launch diagnostic is not attractive. | Standalone gate passes on `q_wqb`, `wo_b local`, and shared expert down. |
| B | vLLM FBGEMM FP8 Marlin-compatible route | Diagnostic load-time conversion from DSV4 block FP8 to BF16, then per-channel FP8 requantization and Marlin repack. | BF16 input; FBGEMM Marlin path deletes input scale upper bound. | Per-channel FP8 `weight_scale`, not native DSV4 block-scale checkpoint layout. | Available; token FP8 selection also routes to Marlin on A100. | Not as-is; requires extra conversion/requantization and has slightly worse quality. | Standalone timing passes on several owners, but Lane A is preferred. |
| C | `torch._scaled_mm` / FP8 W8A8 | Requires activation FP8 quantization and scaled-mm support. | FP8 activation quantization. | Per-tensor/channel scaled-mm contracts. | Unavailable in this A100 smoke: `torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`. | No. | Reject for SM80 DSV4 projection. |
| D | INT8 W8A8 Cutlass projection probe | Load-time BF16 dequant and per-channel INT8 weight quantization; replay includes dynamic `scaled_int8_quant` plus `cutlass_scaled_mm`. | Dynamic INT8 activation quantization per replay. | INT8 weight with per-channel scales. | Available; selected `CutlassInt8ScaledMMLinearKernel`. | Diagnostic only; not an existing DSV4 projection backend and needs separate quality target. | Fails standalone performance gate. |
| E | Custom mini kernel R&D | Not entered in this target. | TBD. | TBD. | Not applicable. | Only if vLLM baseline fails or integration profile leaves clear headroom. | Defer; Lane A passes standalone gate. |

## Availability Summary

Artifact: `../raw/backend_availability.json`.

| Probe | Result |
| --- | --- |
| Runner | `/workspace/venvs/vllm-dsv4/bin/python` |
| Device | `NVIDIA A100-SXM4-80GB`, capability `[8, 0]` |
| mini env vLLM import | unavailable: `ModuleNotFoundError: No module named 'vllm'` |
| vLLM env import | available: `/workspace/vllm-dsv4-docker/vllm/__init__.py` |
| Custom ops | `gptq_marlin_repack`, `marlin_gemm`, `scaled_int8_quant`, and `cutlass_scaled_mm` present |
| DeepSeek V4 block FP8 kernel selection | `MarlinFP8ScaledMMLinearKernel` |
| FBGEMM/token FP8 kernel selection | `MarlinFP8ScaledMMLinearKernel` |
| Marlin W8A8 | rejected by vLLM helper |
| `torch._scaled_mm` FP8 | unsupported on A100 in this environment |
| INT8 W8A8 | `CutlassInt8ScaledMMLinearKernel` selected; activation quant op available |
