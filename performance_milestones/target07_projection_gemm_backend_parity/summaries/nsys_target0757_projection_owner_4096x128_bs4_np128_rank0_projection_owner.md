# Projection/GEMM Owner Attribution: nsys_target0757_projection_owner_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Decode forward ranges: `127`; decode envelope s: `5.436976`
- DSV4 owner NVTX ranges found: `9016`
- Decode projection/GEMM intrinsic bucket from kernel names: `1.796818` s, `795` graph nodes
- Owner-attributed projection/GEMM intrinsic: `1.462901` s; unattributed intrinsic: `0.333917` s

## Owner Table

| Owner | Kernel s | Runtime/copy s | Graph nodes | Top kernels | Backend contract | Keep/Pivot |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `attn.q_wqb` | `0.404178` | `0.029766` | 129 | `_quantized_linear_fp8_kernel` 0.4042s, `void at::native::unrolled_elementwise_kernel<...` 0.0184s, `_fp8_activation_quantize_kernel` 0.0113s | DSV4Linear ColumnParallel FP8: quantize_fp8_activation_ref/_fp8_activation_quantize_kernel + _quantized_linear_fp8_kernel. | keep if >=0.50s; compare against vLLM lifted wq_b ColumnParallelLinear. |
| `attn.wo_b` | `0.403710` | `0.028501` | 172 | `_quantized_linear_fp8_kernel` 0.4037s, `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(nccl...` 0.1692s, `void at::native::unrolled_elementwise_kernel<...` 0.0171s | DSV4Linear RowParallel FP8: _quantized_linear_fp8_kernel plus row-parallel all-reduce. | large owner, but verify intrinsic vs staging |
| `indexer.wq_b` | `0.364756` | `0.018079` | 63 | `_quantized_linear_fp8_kernel` 0.3648s, `void at::native::unrolled_elementwise_kernel<...` 0.0129s, `_fp8_activation_quantize_kernel` 0.0052s | Indexer query projection DSV4Linear FP8: _quantized_linear_fp8_kernel plus activation quant. | keep if >=0.50s; otherwise context for FP8 indexer cache path. |
| `attn.q_proj_wqa_wkv` | `0.089755` | `0.029276` | 215 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0778s, `void at::native::elementwise_kernel<(int)128,...` 0.0165s, `_fp8_activation_quantize_kernel` 0.0128s | FP8 WQA/WKV projection; active fwqakvcache path may dequantize cached BF16 weights and run F.linear. | keep if fused WQA/WKV owns >=0.50s; otherwise use as context. |
| `attn.wo_a` | `0.053440` | `0.427810` | 344 | `void at::native::unrolled_elementwise_kernel<...` 0.1511s, `void at::native::vectorized_elementwise_kerne...` 0.1370s, `void at::native::elementwise_kernel<(int)128,...` 0.0904s | Grouped output projection: wo_a_grouped_projection_fp8 when enabled, otherwise dequant/einsum fallback. | keep if >=0.50s; compare against vLLM SM80 wo_a BMM/reference and fp8_einsum boundary. |
| `shared_experts.gate_up_proj` | `0.045668` | `0.228309` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0888s, `void at::native::elementwise_kernel<(int)128,...` 0.0562s, `void at::native::vectorized_elementwise_kerne...` 0.0453s | Shared expert FP8 gate/up projection through DSV4Linear. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `shared_experts.down_proj` | `0.029229` | `0.166127` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0673s, `void at::native::elementwise_kernel<(int)128,...` 0.0387s, `void cutlass::Kernel2<cutlass_80_wmma_tensoro...` 0.0292s | Shared expert FP8 down projection through DSV4Linear plus optional all-reduce. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `indexer.compressor` | `0.026728` | `0.062478` | 252 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0196s, `void at::native::<unnamed>::CatArrayBatchedCo...` 0.0158s, `void at::native::reduce_kernel<(int)128, (int...` 0.0119s | Indexer compressor projection/norm/cache-adjacent work. | context; not the primary projection owner unless it dominates. |
| `lm_head` | `0.026654` | `0.044760` | 5 | `void at::native::unrolled_elementwise_kernel<...` 0.0441s, `void gemmSN_TN_kernel<float, (int)128, (int)1...` 0.0267s, `ncclDevKernel_AllGather_RING_LL(ncclDevKernel...` 0.0053s | Vocab-parallel output linear: BF16/FP32 F.linear plus all-gather. | context for decode envelope; not a projection backend PoC unless dominant. |
| `indexer.weights_proj` | `0.018783` | `0.000000` | 42 | `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` 0.0119s, `void cublasLt::splitKreduce_kernel<(int)32, (...` 0.0069s | Indexer weights/logits projection: BF16 F.linear/sgemm plus scale multiply. | keep only if BF16 projection dominates. |
| `mlp.routed_experts` | `0.000000` | `0.143422` | 688 | `void marlin_moe_wna16::Marlin<(long)112589990...` 0.1469s, `void marlin_moe_wna16::Marlin<(long)112589990...` 0.1028s, `void at::native::elementwise_kernel<(int)128,...` 0.0462s | Routed expert backend, usually Marlin WNA16 in the active variant. | context; out of scope unless projection attribution shows shared/routed FFN dominates. |

## Owner Details

### `attn.q_wqb`

- NVTX ranges: `301`; capture graph nodes: `774`; replay graph nodes: `129`
- Replay kernel total: `0.433945` s; intrinsic GEMM: `0.404178` s; activation quant: `0.011343` s; copy/layout: `0.018423` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5461 | `0.404178` | 43 |
| `wrapper_copy_layout` | 5461 | `0.018423` | 43 |
| `activation_quant` | 5461 | `0.011343` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 5461 | `0.404178` | 43 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 5461 | `0.018423` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011343` | 43 |

### `attn.wo_b`

- NVTX ranges: `301`; capture graph nodes: `1290`; replay graph nodes: `172`
- Replay kernel total: `0.601383` s; intrinsic GEMM: `0.403710` s; activation quant: `0.011370` s; copy/layout: `0.017131` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5461 | `0.403710` | 43 |
| `communication` | 5461 | `0.169172` | 43 |
| `wrapper_copy_layout` | 5461 | `0.017131` | 43 |
| `activation_quant` | 5461 | `0.011370` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 5461 | `0.403710` | 43 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(un...` | 5461 | `0.169172` | 43 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 5461 | `0.017131` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011370` | 43 |

### `indexer.wq_b`

- NVTX ranges: `147`; capture graph nodes: `378`; replay graph nodes: `63`
- Replay kernel total: `0.382835` s; intrinsic GEMM: `0.364756` s; activation quant: `0.005213` s; copy/layout: `0.012866` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 2667 | `0.364756` | 21 |
| `wrapper_copy_layout` | 2667 | `0.012866` | 21 |
| `activation_quant` | 2667 | `0.005213` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 2667 | `0.364756` | 21 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 2667 | `0.012866` | 21 |
| `_fp8_activation_quantize_kernel` | 2667 | `0.005213` | 21 |

### `attn.q_proj_wqa_wkv`

- NVTX ranges: `602`; capture graph nodes: `1118`; replay graph nodes: `215`
- Replay kernel total: `0.131261` s; intrinsic GEMM: `0.089755` s; activation quant: `0.012791` s; copy/layout: `0.016486` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 10922 | `0.089755` | 86 |
| `wrapper_copy_layout` | 5461 | `0.016486` | 43 |
| `activation_quant` | 5461 | `0.012791` | 43 |
| `sampling_logits_norm` | 5461 | `0.012230` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.077784` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 5461 | `0.016486` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012791` | 43 |
| `_rms_norm_bf16_kernel` | 5461 | `0.012230` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.011971` | 43 |

### `attn.wo_a`

- NVTX ranges: `301`; capture graph nodes: `1978`; replay graph nodes: `344`
- Replay kernel total: `0.481250` s; intrinsic GEMM: `0.053440` s; activation quant: `0.000000` s; copy/layout: `0.290843` s; elementwise/scale: `0.136967` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.290843` | 215 |
| `elementwise_scale_math` | 5461 | `0.136967` | 43 |
| `intrinsic_gemm` | 10922 | `0.053440` | 86 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.151071` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.136967` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.090361` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.049411` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.040307` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.013133` | 43 |

### `shared_experts.gate_up_proj`

- NVTX ranges: `301`; capture graph nodes: `2236`; replay graph nodes: `387`
- Replay kernel total: `0.273977` s; intrinsic GEMM: `0.045668` s; activation quant: `0.012787` s; copy/layout: `0.170270` s; elementwise/scale: `0.045253` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.170270` | 215 |
| `intrinsic_gemm` | 10922 | `0.045668` | 86 |
| `elementwise_scale_math` | 5461 | `0.045253` | 43 |
| `activation_quant` | 5461 | `0.012787` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.088799` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.056237` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.045253` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.031180` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.025233` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.014488` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012787` | 43 |

### `shared_experts.down_proj`

- NVTX ranges: `301`; capture graph nodes: `2322`; replay graph nodes: `387`
- Replay kernel total: `0.195356` s; intrinsic GEMM: `0.029229` s; activation quant: `0.010880` s; copy/layout: `0.129601` s; elementwise/scale: `0.025647` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 32766 | `0.129601` | 258 |
| `intrinsic_gemm` | 5461 | `0.029229` | 43 |
| `elementwise_scale_math` | 5461 | `0.025647` | 43 |
| `activation_quant` | 5461 | `0.010880` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.067314` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.038749` | 86 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_...` | 5461 | `0.029229` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.025647` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 10922 | `0.023538` | 86 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.010880` | 43 |

### `indexer.compressor`

- NVTX ranges: `147`; capture graph nodes: `504`; replay graph nodes: `252`
- Replay kernel total: `0.100720` s; intrinsic GEMM: `0.026728` s; activation quant: `0.000000` s; copy/layout: `0.035582` s; elementwise/scale: `0.026896` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 13335 | `0.035582` | 105 |
| `elementwise_scale_math` | 10668 | `0.026896` | 84 |
| `intrinsic_gemm` | 5334 | `0.026728` | 42 |
| `sampling_logits_norm` | 2667 | `0.011514` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 2667 | `0.019647` | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>...` | 5334 | `0.015816` | 42 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 2667 | `0.011862` | 21 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, ...` | 2667 | `0.011514` | 21 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 2667 | `0.010382` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.007081` | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::BU...` | 2667 | `0.005512` | 21 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char...` | 2667 | `0.005430` | 21 |

### `lm_head`

- NVTX ranges: `7`; capture graph nodes: `34`; replay graph nodes: `5`
- Replay kernel total: `0.076688` s; intrinsic GEMM: `0.026654` s; activation quant: `0.000000` s; copy/layout: `0.044760` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 381 | `0.044760` | 3 |
| `intrinsic_gemm` | 127 | `0.026654` | 1 |
| `communication` | 127 | `0.005274` | 1 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 254 | `0.044058` | 2 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)...` | 127 | `0.026654` | 1 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned lo...` | 127 | `0.005274` | 1 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 127 | `0.000702` | 1 |

### `indexer.weights_proj`

- NVTX ranges: `147`; capture graph nodes: `252`; replay graph nodes: `42`
- Replay kernel total: `0.018783` s; intrinsic GEMM: `0.018783` s; activation quant: `0.000000` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5334 | `0.018783` | 42 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` | 2667 | `0.011860` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.006923` | 21 |

### `mlp.routed_experts`

- NVTX ranges: `301`; capture graph nodes: `4386`; replay graph nodes: `688`
- Replay kernel total: `0.428274` s; intrinsic GEMM: `0.000000` s; activation quant: `0.000000` s; copy/layout: `0.076640` s; elementwise/scale: `0.066782` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `moe_marlin` | 27305 | `0.284852` | 215 |
| `wrapper_copy_layout` | 32766 | `0.076640` | 258 |
| `elementwise_scale_math` | 27305 | `0.066782` | 215 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.146922` | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.102846` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 10922 | `0.046204` | 86 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 5461 | `0.028663` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Fi...` | 16383 | `0.022168` | 129 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 10922 | `0.018838` | 86 |
| `_moe_route_fill_kernel` | 5461 | `0.016873` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 5461 | `0.010566` | 43 |
