# Projection/GEMM Owner Attribution: nsys_target0759_qwqb_wob_projection_owner_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Decode forward ranges: `127`; decode envelope s: `4.732020`
- DSV4 owner NVTX ranges found: `9016`
- Decode projection/GEMM intrinsic bucket from kernel names: `1.117767` s, `795` graph nodes
- Owner-attributed projection/GEMM intrinsic: `0.782388` s; unattributed intrinsic: `0.335379` s

## Owner Table

| Owner | Kernel s | Runtime/copy s | Graph nodes | Top kernels | Backend contract | Keep/Pivot |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `indexer.wq_b` | `0.364997` | `0.018221` | 63 | `_quantized_linear_fp8_kernel` 0.3650s, `void at::native::unrolled_elementwise_kernel<...` 0.0129s, `_fp8_activation_quantize_kernel` 0.0053s | Indexer query projection DSV4Linear FP8: _quantized_linear_fp8_kernel plus activation quant. | keep if >=0.50s; otherwise context for FP8 indexer cache path. |
| `attn.q_proj_wqa_wkv` | `0.089500` | `0.029357` | 215 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0777s, `void at::native::elementwise_kernel<(int)128,...` 0.0165s, `_fp8_activation_quantize_kernel` 0.0128s | FP8 WQA/WKV projection; active fwqakvcache path may dequantize cached BF16 weights and run F.linear. | keep if fused WQA/WKV owns >=0.50s; otherwise use as context. |
| `attn.q_wqb` | `0.068302` | `0.011431` | 86 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0683s, `_fp8_activation_quantize_kernel` 0.0114s | DSV4Linear ColumnParallel FP8: quantize_fp8_activation_ref/_fp8_activation_quantize_kernel + _quantized_linear_fp8_kernel. | keep if >=0.50s; compare against vLLM lifted wq_b ColumnParallelLinear. |
| `attn.wo_b` | `0.059160` | `0.011435` | 129 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(nccl...` 0.1619s, `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_l...` 0.0592s, `_fp8_activation_quantize_kernel` 0.0114s | DSV4Linear RowParallel FP8: _quantized_linear_fp8_kernel plus row-parallel all-reduce. | keep if >=0.50s; compare against vLLM RowParallelLinear quant path. |
| `attn.wo_a` | `0.053404` | `0.428135` | 344 | `void at::native::unrolled_elementwise_kernel<...` 0.1508s, `void at::native::vectorized_elementwise_kerne...` 0.1375s, `void at::native::elementwise_kernel<(int)128,...` 0.0903s | Grouped output projection: wo_a_grouped_projection_fp8 when enabled, otherwise dequant/einsum fallback. | keep if >=0.50s; compare against vLLM SM80 wo_a BMM/reference and fp8_einsum boundary. |
| `shared_experts.gate_up_proj` | `0.045700` | `0.227534` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0879s, `void at::native::elementwise_kernel<(int)128,...` 0.0562s, `void at::native::vectorized_elementwise_kerne...` 0.0452s | Shared expert FP8 gate/up projection through DSV4Linear. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `shared_experts.down_proj` | `0.029315` | `0.166204` | 387 | `void at::native::unrolled_elementwise_kernel<...` 0.0670s, `void at::native::elementwise_kernel<(int)128,...` 0.0389s, `void cutlass::Kernel2<cutlass_80_wmma_tensoro...` 0.0293s | Shared expert FP8 down projection through DSV4Linear plus optional all-reduce. | keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated. |
| `lm_head` | `0.026723` | `0.044689` | 5 | `void at::native::unrolled_elementwise_kernel<...` 0.0441s, `void gemmSN_TN_kernel<float, (int)128, (int)1...` 0.0267s, `ncclDevKernel_AllGather_RING_LL(ncclDevKernel...` 0.0053s | Vocab-parallel output linear: BF16/FP32 F.linear plus all-gather. | context for decode envelope; not a projection backend PoC unless dominant. |
| `indexer.compressor` | `0.026491` | `0.062733` | 252 | `void cutlass::Kernel2<cutlass_80_tensorop_s16...` 0.0194s, `void at::native::<unnamed>::CatArrayBatchedCo...` 0.0157s, `void at::native::reduce_kernel<(int)128, (int...` 0.0122s | Indexer compressor projection/norm/cache-adjacent work. | context; not the primary projection owner unless it dominates. |
| `indexer.weights_proj` | `0.018796` | `0.000000` | 42 | `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` 0.0119s, `void cublasLt::splitKreduce_kernel<(int)32, (...` 0.0069s | Indexer weights/logits projection: BF16 F.linear/sgemm plus scale multiply. | keep only if BF16 projection dominates. |
| `mlp.routed_experts` | `0.000000` | `0.142922` | 688 | `void marlin_moe_wna16::Marlin<(long)112589990...` 0.1454s, `void marlin_moe_wna16::Marlin<(long)112589990...` 0.0999s, `void at::native::elementwise_kernel<(int)128,...` 0.0461s | Routed expert backend, usually Marlin WNA16 in the active variant. | context; out of scope unless projection attribution shows shared/routed FFN dominates. |

## Owner Details

### `indexer.wq_b`

- NVTX ranges: `147`; capture graph nodes: `378`; replay graph nodes: `63`
- Replay kernel total: `0.383218` s; intrinsic GEMM: `0.364997` s; activation quant: `0.005293` s; copy/layout: `0.012928` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 2667 | `0.364997` | 21 |
| `wrapper_copy_layout` | 2667 | `0.012928` | 21 |
| `activation_quant` | 2667 | `0.005293` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 2667 | `0.364997` | 21 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 2667 | `0.012928` | 21 |
| `_fp8_activation_quantize_kernel` | 2667 | `0.005293` | 21 |

### `attn.q_proj_wqa_wkv`

- NVTX ranges: `602`; capture graph nodes: `1118`; replay graph nodes: `215`
- Replay kernel total: `0.131061` s; intrinsic GEMM: `0.089500` s; activation quant: `0.012813` s; copy/layout: `0.016544` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 10922 | `0.089500` | 86 |
| `wrapper_copy_layout` | 5461 | `0.016544` | 43 |
| `activation_quant` | 5461 | `0.012813` | 43 |
| `sampling_logits_norm` | 5461 | `0.012204` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.077715` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 5461 | `0.016544` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012813` | 43 |
| `_rms_norm_bf16_kernel` | 5461 | `0.012204` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.011785` | 43 |

### `attn.q_wqb`

- NVTX ranges: `301`; capture graph nodes: `516`; replay graph nodes: `86`
- Replay kernel total: `0.079733` s; intrinsic GEMM: `0.068302` s; activation quant: `0.011431` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5461 | `0.068302` | 43 |
| `activation_quant` | 5461 | `0.011431` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5461 | `0.068302` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011431` | 43 |

### `attn.wo_b`

- NVTX ranges: `301`; capture graph nodes: `1032`; replay graph nodes: `129`
- Replay kernel total: `0.232461` s; intrinsic GEMM: `0.059160` s; activation quant: `0.011435` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `communication` | 5461 | `0.161865` | 43 |
| `intrinsic_gemm` | 5461 | `0.059160` | 43 |
| `activation_quant` | 5461 | `0.011435` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(un...` | 5461 | `0.161865` | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5461 | `0.059160` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.011435` | 43 |

### `attn.wo_a`

- NVTX ranges: `301`; capture graph nodes: `1978`; replay graph nodes: `344`
- Replay kernel total: `0.481539` s; intrinsic GEMM: `0.053404` s; activation quant: `0.000000` s; copy/layout: `0.290625` s; elementwise/scale: `0.137511` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.290625` | 215 |
| `elementwise_scale_math` | 5461 | `0.137511` | 43 |
| `intrinsic_gemm` | 10922 | `0.053404` | 86 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.150790` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.137511` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.090280` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.049554` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.040248` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.013156` | 43 |

### `shared_experts.gate_up_proj`

- NVTX ranges: `301`; capture graph nodes: `2236`; replay graph nodes: `387`
- Replay kernel total: `0.273235` s; intrinsic GEMM: `0.045700` s; activation quant: `0.012744` s; copy/layout: `0.169613` s; elementwise/scale: `0.045177` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 27305 | `0.169613` | 215 |
| `intrinsic_gemm` | 10922 | `0.045700` | 86 |
| `elementwise_scale_math` | 5461 | `0.045177` | 43 |
| `activation_quant` | 5461 | `0.012744` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.087923` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.056190` | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.045177` | 43 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 5461 | `0.031198` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 5461 | `0.025501` | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 5461 | `0.014502` | 43 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.012744` | 43 |

### `shared_experts.down_proj`

- NVTX ranges: `301`; capture graph nodes: `2322`; replay graph nodes: `387`
- Replay kernel total: `0.195519` s; intrinsic GEMM: `0.029315` s; activation quant: `0.010910` s; copy/layout: `0.129618` s; elementwise/scale: `0.025676` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 32766 | `0.129618` | 258 |
| `intrinsic_gemm` | 5461 | `0.029315` | 43 |
| `elementwise_scale_math` | 5461 | `0.025676` | 43 |
| `activation_quant` | 5461 | `0.010910` | 43 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 10922 | `0.066958` | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 10922 | `0.038938` | 86 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_...` | 5461 | `0.029315` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bi...` | 5461 | `0.025676` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bf...` | 10922 | `0.023723` | 86 |
| `_fp8_activation_quantize_kernel` | 5461 | `0.010910` | 43 |

### `lm_head`

- NVTX ranges: `7`; capture graph nodes: `34`; replay graph nodes: `5`
- Replay kernel total: `0.076710` s; intrinsic GEMM: `0.026723` s; activation quant: `0.000000` s; copy/layout: `0.044689` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 381 | `0.044689` | 3 |
| `intrinsic_gemm` | 127 | `0.026723` | 1 |
| `communication` | 127 | `0.005298` | 1 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 254 | `0.044061` | 2 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)...` | 127 | `0.026723` | 1 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned lo...` | 127 | `0.005298` | 1 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native...` | 127 | `0.000628` | 1 |

### `indexer.compressor`

- NVTX ranges: `147`; capture graph nodes: `504`; replay graph nodes: `252`
- Replay kernel total: `0.100709` s; intrinsic GEMM: `0.026491` s; activation quant: `0.000000` s; copy/layout: `0.035604` s; elementwise/scale: `0.027129` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `wrapper_copy_layout` | 13335 | `0.035604` | 105 |
| `elementwise_scale_math` | 10668 | `0.027129` | 84 |
| `intrinsic_gemm` | 5334 | `0.026491` | 42 |
| `sampling_logits_norm` | 2667 | `0.011486` | 21 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_...` | 2667 | `0.019449` | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>...` | 5334 | `0.015730` | 42 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 2667 | `0.012181` | 21 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, ...` | 2667 | `0.011486` | 21 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_...` | 2667 | `0.010414` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.007042` | 21 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char...` | 2667 | `0.005467` | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::BU...` | 2667 | `0.005458` | 21 |

### `indexer.weights_proj`

- NVTX ranges: `147`; capture graph nodes: `252`; replay graph nodes: `42`
- Replay kernel total: `0.018796` s; intrinsic GEMM: `0.018796` s; activation quant: `0.000000` s; copy/layout: `0.000000` s; elementwise/scale: `0.000000` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `intrinsic_gemm` | 5334 | `0.018796` | 42 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` | 2667 | `0.011882` | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv...` | 2667 | `0.006914` | 21 |

### `mlp.routed_experts`

- NVTX ranges: `301`; capture graph nodes: `4386`; replay graph nodes: `688`
- Replay kernel total: `0.423377` s; intrinsic GEMM: `0.000000` s; activation quant: `0.000000` s; copy/layout: `0.076611` s; elementwise/scale: `0.066311` s

| Category | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `moe_marlin` | 27305 | `0.280455` | 215 |
| `wrapper_copy_layout` | 32766 | `0.076611` | 258 |
| `elementwise_scale_math` | 27305 | `0.066311` | 215 |

| Top kernel | Count | Duration s | Graph nodes |
| --- | ---: | ---: | ---: |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.145433` | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953...` | 5461 | `0.099878` | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native...` | 10922 | `0.046123` | 86 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp...` | 5461 | `0.028309` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Fi...` | 16383 | `0.022220` | 129 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 10922 | `0.018840` | 86 |
| `_moe_route_fill_kernel` | 5461 | `0.017020` | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<u...` | 5461 | `0.010454` | 43 |
