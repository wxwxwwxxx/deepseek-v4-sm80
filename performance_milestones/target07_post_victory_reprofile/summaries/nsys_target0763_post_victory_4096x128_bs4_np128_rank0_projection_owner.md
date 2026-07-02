# Projection/GEMM Owner Attribution: nsys_target0763_post_victory_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Decode forward ranges: `127`; decode envelope s: `3.942934`
- DSV4 owner NVTX ranges found: `9016`
- Decode projection/GEMM intrinsic bucket from kernel names: `0.812100` s, `795` graph nodes
- Owner-attributed projection/GEMM intrinsic: `0.483219` s; unattributed intrinsic: `0.328881` s

## Owner Table

| Owner | Kernel s | Runtime/copy s | Graph nodes | Top kernels | Backend contract | Keep/Pivot |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `attn.q_proj_wqa_wkv` | `0.089719` | `0.029319` | 215 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0779s, `void at::native::elementwise_kernel<(int)128,...` 0.0165s, `_fp8_activation_quantize_kernel` 0.0128s | FP8 WQA/WKV projection; active fwqakvcache path may dequantize cached BF16 weights and run F.linear. | keep if fused WQA/WKV owns >=0.50s; otherwise use as context. |
| `attn.wo_a` | `0.068947` | `0.000000` | 86 | `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_s...` 0.0572s, `void cublasLt::splitKreduce_kernel<(int)32, (...` 0.0118s | Grouped output projection: wo_a_grouped_projection_fp8 when enabled, otherwise dequant/einsum fallback. | keep if >=0.50s; compare against vLLM SM80 wo_a BMM/reference and fp8_einsum boundary. |
| `attn.q_wqb` | `0.068381` | `0.011437` | 86 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0684s, `_fp8_activation_quantize_kernel` 0.0114s | DSV4Linear ColumnParallel FP8: quantize_fp8_activation_ref/_fp8_activation_quantize_kernel + _quantized_linear_fp8_kernel. | keep if >=0.50s; compare against vLLM lifted wq_b ColumnParallelLinear. |
| `attn.wo_b` | `0.057773` | `0.011413` | 129 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(nccl...` 0.1600s, `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0578s, `_fp8_activation_quantize_kernel` 0.0114s | DSV4Linear RowParallel FP8: _quantized_linear_fp8_kernel plus row-parallel all-reduce. | keep if >=0.50s; compare against vLLM RowParallelLinear quant path. |
| `indexer.wq_b` | `0.050902` | `0.005280` | 42 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0509s, `_fp8_activation_quantize_kernel` 0.0053s | Indexer query projection DSV4Linear FP8: _quantized_linear_fp8_kernel plus activation quant. | keep if >=0.50s; otherwise context for FP8 indexer cache path. |
| `shared_experts.gate_up_proj` | `0.045904` | `0.227054` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0872s, `void at::native::elementwise_kernel<(int)128,...` 0.0562s, `void at::native::vectorized_elementwise_kerne...` 0.0456s | Shared expert FP8 gate/up projection through DSV4Linear. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `shared_experts.down_proj` | `0.029315` | `0.166097` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0670s, `void at::native::elementwise_kernel<(int)128,...` 0.0389s, `void cutlass::Kernel2<cutlass_80_wmma_tensoro...` 0.0293s | Shared expert FP8 down projection through DSV4Linear plus optional all-reduce. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `indexer.compressor` | `0.026739` | `0.062209` | 252 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0197s, `void at::native::<unnamed>::CatArrayBatchedCo...` 0.0158s, `void at::native::<unnamed>::cunn_SpatialSoftM...` 0.0115s | Indexer compressor projection/norm/cache-adjacent work. | context; not the primary projection owner unless it dominates. |
| `lm_head` | `0.026498` | `0.044722` | 5 | `void at::native::unrolled_elementwise_kernel<...` 0.0441s, `void gemmSN_TN_kernel<float, (int)128, (int)1...` 0.0265s, `ncclDevKernel_AllGather_RING_LL(ncclDevKernel...` 0.0058s | Vocab-parallel output linear: BF16/FP32 F.linear plus all-gather. | context for decode envelope; not a projection backend PoC unless dominant. |
| `indexer.weights_proj` | `0.019041` | `0.000000` | 42 | `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` 0.0121s, `void cublasLt::splitKreduce_kernel<(int)32, (...` 0.0070s | Indexer weights/logits projection: BF16 F.linear/sgemm plus scale multiply. | keep only if BF16 projection dominates. |
| `mlp.routed_experts` | `0.000000` | `0.143309` | 688 | `void marlin_moe_wna16::Marlin<(long)112589990...` 0.1322s, `void marlin_moe_wna16::Marlin<(long)112589990...` 0.0978s, `void at::native::elementwise_kernel<(int)128,...` 0.0465s | Routed expert backend, usually Marlin WNA16 in the active variant. | context; out of scope unless projection attribution shows shared/routed FFN dominates. |

## Owner Details

### `attn.q_proj_wqa_wkv`

- NVTX ranges: `602`; capture graph nodes: `1118`; replay graph nodes: `215`
- Replay kernel total: `0.131247` s; intrinsic GEMM: `0.089719` s; activation quant: `0.012791` s; copy/layout: `0.016528` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 10922 | `0.089719` | 86 |
| `wrapper_copy_layout` | 5461 | `0.016528` | 43 |
| `activation_quant` | 5461 | `0.012791` | 43 |
| `sampling_logits_norm` | 5461 | `0.012209` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.077925` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 5461 | `0.016528` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012791` | 43 |
| `_rms_norm_bf16_kernel` | 5461 | `0.012209` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.011794` | 43 |

### `attn.wo_a`

- NVTX ranges: `301`; capture graph nodes: `516`; replay graph nodes: `86`
- Replay kernel total: `0.068947` s; intrinsic GEMM: `0.068947` s; activation quant: `0.000000` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 10922 | `0.068947` | 86 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.057154` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.011793` | 43 |

### `attn.q_wqb`

- NVTX ranges: `301`; capture graph nodes: `516`; replay graph nodes: `86`
- Replay kernel total: `0.079818` s; intrinsic GEMM: `0.068381` s; activation quant: `0.011437` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5461 | `0.068381` | 43 |
| `activation_quant` | 5461 | `0.011437` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5461 | `0.068381` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011437` | 43 |

### `attn.wo_b`

- NVTX ranges: `301`; capture graph nodes: `1032`; replay graph nodes: `129`
- Replay kernel total: `0.229213` s; intrinsic GEMM: `0.057773` s; activation quant: `0.011413` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `communication` | 5461 | `0.160028` | 43 |
| `intrinsic_gemm` | 5461 | `0.057773` | 43 |
| `activation_quant` | 5461 | `0.011413` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(un...` | 5461 | `0.160028` | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5461 | `0.057773` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011413` | 43 |

### `indexer.wq_b`

- NVTX ranges: `147`; capture graph nodes: `252`; replay graph nodes: `42`
- Replay kernel total: `0.056181` s; intrinsic GEMM: `0.050902` s; activation quant: `0.005280` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 2667 | `0.050902` | 21 |
| `activation_quant` | 2667 | `0.005280` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050902` | 21 |
| `_fp8_activation_quantize_kernel` | 2667 | `0.005280` | 21 |

### `shared_experts.gate_up_proj`

- NVTX ranges: `301`; capture graph nodes: `2236`; replay graph nodes: `387`
- Replay kernel total: `0.272958` s; intrinsic GEMM: `0.045904` s; activation quant: `0.012657` s; copy/layout: `0.168813` s; elementwise/scale: `0.045585` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.168813` | 215 |
| `intrinsic_gemm` | 10922 | `0.045904` | 86 |
| `elementwise_scale_math` | 5461 | `0.045585` | 43 |
| `activation_quant` | 5461 | `0.012657` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.087174` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.056216` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.045585` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.031357` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.025423` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.014547` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012657` | 43 |

### `shared_experts.down_proj`

- NVTX ranges: `301`; capture graph nodes: `2322`; replay graph nodes: `387`
- Replay kernel total: `0.195412` s; intrinsic GEMM: `0.029315` s; activation quant: `0.010929` s; copy/layout: `0.129690` s; elementwise/scale: `0.025478` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 32766 | `0.129690` | 258 |
| `intrinsic_gemm` | 5461 | `0.029315` | 43 |
| `elementwise_scale_math` | 5461 | `0.025478` | 43 |
| `activation_quant` | 5461 | `0.010929` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.067008` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.038918` | 86 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_...` | 5461 | `0.029315` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.025478` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 10922 | `0.023764` | 86 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.010929` | 43 |

### `indexer.compressor`

- NVTX ranges: `147`; capture graph nodes: `504`; replay graph nodes: `252`
- Replay kernel total: `0.100438` s; intrinsic GEMM: `0.026739` s; activation quant: `0.000000` s; copy/layout: `0.035737` s; elementwise/scale: `0.026472` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 13335 | `0.035737` | 105 |
| `intrinsic_gemm` | 5334 | `0.026739` | 42 |
| `elementwise_scale_math` | 10668 | `0.026472` | 84 |
| `sampling_logits_norm` | 2667 | `0.011489` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 2667 | `0.019666` | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>...` | 5334 | `0.015793` | 42 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, ...` | 2667 | `0.011489` | 21 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 2667 | `0.011457` | 21 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 2667 | `0.010456` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.007073` | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::BU...` | 2667 | `0.005492` | 21 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char...` | 2667 | `0.005466` | 21 |

### `lm_head`

- NVTX ranges: `7`; capture graph nodes: `34`; replay graph nodes: `5`
- Replay kernel total: `0.077000` s; intrinsic GEMM: `0.026498` s; activation quant: `0.000000` s; copy/layout: `0.044722` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 381 | `0.044722` | 3 |
| `intrinsic_gemm` | 127 | `0.026498` | 1 |
| `communication` | 127 | `0.005780` | 1 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 254 | `0.044056` | 2 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)...` | 127 | `0.026498` | 1 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned lo...` | 127 | `0.005780` | 1 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 127 | `0.000665` | 1 |

### `indexer.weights_proj`

- NVTX ranges: `147`; capture graph nodes: `252`; replay graph nodes: `42`
- Replay kernel total: `0.019041` s; intrinsic GEMM: `0.019041` s; activation quant: `0.000000` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5334 | `0.019041` | 42 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` | 2667 | `0.012069` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.006972` | 21 |

### `mlp.routed_experts`

- NVTX ranges: `301`; capture graph nodes: `4386`; replay graph nodes: `688`
- Replay kernel total: `0.408459` s; intrinsic GEMM: `0.000000` s; activation quant: `0.000000` s; copy/layout: `0.076930` s; elementwise/scale: `0.066379` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `moe_marlin` | 27305 | `0.265151` | 215 |
| `wrapper_copy_layout` | 32766 | `0.076930` | 258 |
| `elementwise_scale_math` | 27305 | `0.066379` | 215 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.132209` | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.097777` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 10922 | `0.046454` | 86 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 5461 | `0.028349` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Fi...` | 16383 | `0.022202` | 129 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 10922 | `0.018876` | 86 |
| `_moe_route_fill_kernel` | 5461 | `0.017027` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 5461 | `0.010440` | 43 |
