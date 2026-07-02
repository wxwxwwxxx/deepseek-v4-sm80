# Projection/GEMM Owner Attribution: nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Decode forward ranges: `127`; decode envelope s: `4.408601`
- DSV4 owner NVTX ranges found: `9016`
- Decode projection/GEMM intrinsic bucket from kernel names: `0.805080` s, `795` graph nodes
- Owner-attributed projection/GEMM intrinsic: `0.469287` s; unattributed intrinsic: `0.335793` s

## Owner Table

| Owner | Kernel s | Runtime/copy s | Graph nodes | Top kernels | Backend contract | Keep/Pivot |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `attn.q_proj_wqa_wkv` | `0.090055` | `0.029333` | 215 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0782s, `void at::native::elementwise_kernel<(int)128,...` 0.0165s, `_fp8_activation_quantize_kernel` 0.0128s | FP8 WQA/WKV projection; active fwqakvcache path may dequantize cached BF16 weights and run F.linear. | keep if fused WQA/WKV owns >=0.50s; otherwise use as context. |
| `attn.q_wqb` | `0.068277` | `0.011360` | 86 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0683s, `_fp8_activation_quantize_kernel` 0.0114s | DSV4Linear ColumnParallel FP8: quantize_fp8_activation_ref/_fp8_activation_quantize_kernel + _quantized_linear_fp8_kernel. | keep if >=0.50s; compare against vLLM lifted wq_b ColumnParallelLinear. |
| `attn.wo_b` | `0.059062` | `0.011435` | 129 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(nccl...` 0.1596s, `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0591s, `_fp8_activation_quantize_kernel` 0.0114s | DSV4Linear RowParallel FP8: _quantized_linear_fp8_kernel plus row-parallel all-reduce. | keep if >=0.50s; compare against vLLM RowParallelLinear quant path. |
| `attn.wo_a` | `0.053534` | `0.427843` | 344 | `void at::native::unrolled_elementwise_kernel<...` 0.1506s, `void at::native::vectorized_elementwise_kerne...` 0.1377s, `void at::native::elementwise_kernel<(int)128,...` 0.0901s | Grouped output projection: wo_a_grouped_projection_fp8 when enabled, otherwise dequant/einsum fallback. | keep if >=0.50s; compare against vLLM SM80 wo_a BMM/reference and fp8_einsum boundary. |
| `indexer.wq_b` | `0.050961` | `0.005291` | 42 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0510s, `_fp8_activation_quantize_kernel` 0.0053s | Indexer query projection DSV4Linear FP8: _quantized_linear_fp8_kernel plus activation quant. | keep if >=0.50s; otherwise context for FP8 indexer cache path. |
| `shared_experts.gate_up_proj` | `0.045765` | `0.228028` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0880s, `void at::native::elementwise_kernel<(int)128,...` 0.0561s, `void at::native::vectorized_elementwise_kerne...` 0.0456s | Shared expert FP8 gate/up projection through DSV4Linear. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `shared_experts.down_proj` | `0.029168` | `0.166006` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0671s, `void at::native::elementwise_kernel<(int)128,...` 0.0388s, `void cutlass::Kernel2<cutlass_80_wmma_tensoro...` 0.0292s | Shared expert FP8 down projection through DSV4Linear plus optional all-reduce. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `indexer.compressor` | `0.026734` | `0.062095` | 252 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0196s, `void at::native::<unnamed>::CatArrayBatchedCo...` 0.0158s, `void at::native::<unnamed>::cunn_SpatialSoftM...` 0.0115s | Indexer compressor projection/norm/cache-adjacent work. | context; not the primary projection owner unless it dominates. |
| `lm_head` | `0.026720` | `0.044728` | 5 | `void at::native::unrolled_elementwise_kernel<...` 0.0441s, `void gemmSN_TN_kernel<float, (int)128, (int)1...` 0.0267s, `ncclDevKernel_AllGather_RING_LL(ncclDevKernel...` 0.0055s | Vocab-parallel output linear: BF16/FP32 F.linear plus all-gather. | context for decode envelope; not a projection backend PoC unless dominant. |
| `indexer.weights_proj` | `0.019012` | `0.000000` | 42 | `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` 0.0121s, `void cublasLt::splitKreduce_kernel<(int)32, (...` 0.0069s | Indexer weights/logits projection: BF16 F.linear/sgemm plus scale multiply. | keep only if BF16 projection dominates. |
| `mlp.routed_experts` | `0.000000` | `0.142570` | 688 | `void marlin_moe_wna16::Marlin<(long)112589990...` 0.1453s, `void marlin_moe_wna16::Marlin<(long)112589990...` 0.1010s, `void at::native::elementwise_kernel<(int)128,...` 0.0457s | Routed expert backend, usually Marlin WNA16 in the active variant. | context; out of scope unless projection attribution shows shared/routed FFN dominates. |

## Owner Details

### `attn.q_proj_wqa_wkv`

- NVTX ranges: `602`; capture graph nodes: `1118`; replay graph nodes: `215`
- Replay kernel total: `0.131518` s; intrinsic GEMM: `0.090055` s; activation quant: `0.012800` s; copy/layout: `0.016533` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 10922 | `0.090055` | 86 |
| `wrapper_copy_layout` | 5461 | `0.016533` | 43 |
| `activation_quant` | 5461 | `0.012800` | 43 |
| `sampling_logits_norm` | 5461 | `0.012130` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.078234` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 5461 | `0.016533` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012800` | 43 |
| `_rms_norm_bf16_kernel` | 5461 | `0.012130` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.011821` | 43 |

### `attn.q_wqb`

- NVTX ranges: `301`; capture graph nodes: `516`; replay graph nodes: `86`
- Replay kernel total: `0.079637` s; intrinsic GEMM: `0.068277` s; activation quant: `0.011360` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5461 | `0.068277` | 43 |
| `activation_quant` | 5461 | `0.011360` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5461 | `0.068277` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011360` | 43 |

### `attn.wo_b`

- NVTX ranges: `301`; capture graph nodes: `1032`; replay graph nodes: `129`
- Replay kernel total: `0.230145` s; intrinsic GEMM: `0.059062` s; activation quant: `0.011435` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `communication` | 5461 | `0.159649` | 43 |
| `intrinsic_gemm` | 5461 | `0.059062` | 43 |
| `activation_quant` | 5461 | `0.011435` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(un...` | 5461 | `0.159649` | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5461 | `0.059062` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011435` | 43 |

### `attn.wo_a`

- NVTX ranges: `301`; capture graph nodes: `1978`; replay graph nodes: `344`
- Replay kernel total: `0.481377` s; intrinsic GEMM: `0.053534` s; activation quant: `0.000000` s; copy/layout: `0.290148` s; elementwise/scale: `0.137695` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.290148` | 215 |
| `elementwise_scale_math` | 5461 | `0.137695` | 43 |
| `intrinsic_gemm` | 10922 | `0.053534` | 86 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.150623` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.137695` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.090150` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.049375` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.040373` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.013161` | 43 |

### `indexer.wq_b`

- NVTX ranges: `147`; capture graph nodes: `252`; replay graph nodes: `42`
- Replay kernel total: `0.056252` s; intrinsic GEMM: `0.050961` s; activation quant: `0.005291` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 2667 | `0.050961` | 21 |
| `activation_quant` | 2667 | `0.005291` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050961` | 21 |
| `_fp8_activation_quantize_kernel` | 2667 | `0.005291` | 21 |

### `shared_experts.gate_up_proj`

- NVTX ranges: `301`; capture graph nodes: `2236`; replay graph nodes: `387`
- Replay kernel total: `0.273793` s; intrinsic GEMM: `0.045765` s; activation quant: `0.012901` s; copy/layout: `0.169527` s; elementwise/scale: `0.045601` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.169527` | 215 |
| `intrinsic_gemm` | 10922 | `0.045765` | 86 |
| `elementwise_scale_math` | 5461 | `0.045601` | 43 |
| `activation_quant` | 5461 | `0.012901` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.087968` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.056129` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.045601` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.031187` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.025429` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.014578` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012901` | 43 |

### `shared_experts.down_proj`

- NVTX ranges: `301`; capture graph nodes: `2322`; replay graph nodes: `387`
- Replay kernel total: `0.195174` s; intrinsic GEMM: `0.029168` s; activation quant: `0.010882` s; copy/layout: `0.129588` s; elementwise/scale: `0.025536` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 32766 | `0.129588` | 258 |
| `intrinsic_gemm` | 5461 | `0.029168` | 43 |
| `elementwise_scale_math` | 5461 | `0.025536` | 43 |
| `activation_quant` | 5461 | `0.010882` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.067091` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.038756` | 86 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_...` | 5461 | `0.029168` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.025536` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 10922 | `0.023741` | 86 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.010882` | 43 |

### `indexer.compressor`

- NVTX ranges: `147`; capture graph nodes: `504`; replay graph nodes: `252`
- Replay kernel total: `0.100330` s; intrinsic GEMM: `0.026734` s; activation quant: `0.000000` s; copy/layout: `0.035545` s; elementwise/scale: `0.026551` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 13335 | `0.035545` | 105 |
| `intrinsic_gemm` | 5334 | `0.026734` | 42 |
| `elementwise_scale_math` | 10668 | `0.026551` | 84 |
| `sampling_logits_norm` | 2667 | `0.011501` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 2667 | `0.019649` | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>...` | 5334 | `0.015780` | 42 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, ...` | 2667 | `0.011501` | 21 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 2667 | `0.011479` | 21 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 2667 | `0.010399` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.007084` | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::BU...` | 2667 | `0.005531` | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 2667 | `0.005414` | 21 |

### `lm_head`

- NVTX ranges: `7`; capture graph nodes: `34`; replay graph nodes: `5`
- Replay kernel total: `0.076904` s; intrinsic GEMM: `0.026720` s; activation quant: `0.000000` s; copy/layout: `0.044728` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 381 | `0.044728` | 3 |
| `intrinsic_gemm` | 127 | `0.026720` | 1 |
| `communication` | 127 | `0.005456` | 1 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 254 | `0.044058` | 2 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)...` | 127 | `0.026720` | 1 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned lo...` | 127 | `0.005456` | 1 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 127 | `0.000671` | 1 |

### `indexer.weights_proj`

- NVTX ranges: `147`; capture graph nodes: `252`; replay graph nodes: `42`
- Replay kernel total: `0.019012` s; intrinsic GEMM: `0.019012` s; activation quant: `0.000000` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5334 | `0.019012` | 42 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` | 2667 | `0.012065` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.006947` | 21 |

### `mlp.routed_experts`

- NVTX ranges: `301`; capture graph nodes: `4386`; replay graph nodes: `688`
- Replay kernel total: `0.424127` s; intrinsic GEMM: `0.000000` s; activation quant: `0.000000` s; copy/layout: `0.076232` s; elementwise/scale: `0.066339` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `moe_marlin` | 27305 | `0.281556` | 215 |
| `wrapper_copy_layout` | 32766 | `0.076232` | 258 |
| `elementwise_scale_math` | 27305 | `0.066339` | 215 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.145324` | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.101020` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 10922 | `0.045686` | 86 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 5461 | `0.028318` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Fi...` | 16383 | `0.022228` | 129 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 10922 | `0.018842` | 86 |
| `_moe_route_fill_kernel` | 5461 | `0.016763` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 5461 | `0.010450` | 43 |
