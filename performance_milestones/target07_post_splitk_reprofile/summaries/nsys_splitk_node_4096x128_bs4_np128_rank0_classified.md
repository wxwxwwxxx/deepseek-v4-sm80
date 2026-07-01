# Post-SplitK Nsight Classification: nsys_target0740_splitk_node_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`6.595733`, decode_envelope_s=`7.052103`
- Tables:
  - `CUPTI_ACTIVITY_KIND_KERNEL`: `True`
  - `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`: `False`
  - `CUPTI_ACTIVITY_KIND_RUNTIME`: `True`
  - `CUPTI_ACTIVITY_KIND_MEMCPY`: `True`
  - `CUPTI_ACTIVITY_KIND_MEMSET`: `True`
  - `NVTX_EVENTS`: `True`

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | n/a | `n/a` | ranges=None |
| kernel | 3332639 | `25.748209` |  |
| runtime | 764219 | `44.252920` |  |
| memcpy | 276564 | `10.036158` | bytes=174439565792 |
| memset | 340 | `0.000594` |  |
| sync | 119062 | `28.674748` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 86 | `4.216931` | 16.38% | 0 | 0 |
| `decode_splitk_gather_split_combine` | 43690 | `0.239113` | 0.93% | 43180 | 170 |
| `indexer_logits_topk_cache` | 53506 | `2.398652` | 9.32% | 52578 | 207 |
| `runtime_copy_cat_index_kernels` | 1470140 | `6.544544` | 25.42% | 1210818 | 4767 |
| `fp8_projection_gemm` | 27499 | `2.378030` | 9.24% | 27178 | 107 |
| `moe_marlin_route` | 111112 | `1.412856` | 5.49% | 87376 | 344 |
| `hc_rmsnorm_logits_sampling` | 89311 | `1.264328` | 4.91% | 87630 | 345 |
| `dense_linear_other` | 177636 | `1.998751` | 7.76% | 174752 | 688 |
| `nccl_communication` | 22792 | `0.974960` | 3.79% | 22352 | 88 |
| `elementwise_math_other` | 1313594 | `4.219086` | 16.39% | 1278890 | 5035 |
| `other` | 23273 | `0.100958` | 0.39% | 22098 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 125530 | `28.773365` | 65.02% | 0 | 0 |
| `cuda_graph_launch_runtime` | 794 | `7.505158` | 16.96% | 0 | 0 |
| `kernel_launch_runtime` | 357721 | `2.891434` | 6.53% | 0 | 0 |
| `memcpy_runtime` | 233100 | `2.001038` | 4.52% | 0 | 0 |
| `allocation_runtime` | 1345 | `1.747460` | 3.95% | 0 | 0 |
| `module_runtime` | 76 | `0.226100` | 0.51% | 0 | 0 |
| `other` | 45653 | `1.108365` | 2.50% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 82 | `4.132187` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 350945 | `2.534475` | 343916 | 1354 |
| `_quantized_linear_fp8_kernel` | 27499 | `2.378030` | 27178 | 107 |
| `_indexer_bf16_logits_kernel` | 5376 | `2.010787` | 5334 | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 349668 | `1.483951` | 343154 | 1351 |
| `_hc_split_pre_kernel` | 22274 | `0.852443` | 21844 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 162882 | `0.563736` | 160274 | 631 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11137 | `0.523912` | 10922 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 166709 | `0.511020` | 163322 | 643 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 60997 | `0.499445` | 59690 | 235 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 33280 | `0.479363` | 33020 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 172 | `0.467184` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | `0.439514` | 11176 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 43691 | `0.417951` | 43434 | 171 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 166796 | `0.385464` | 163576 | 644 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 77745 | `0.297246` | 54864 | 216 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 100532 | `0.296816` | 98552 | 388 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 11008 | `0.295565` | 10922 | 43 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 72261 | `0.285732` | 70866 | 279 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44112 | `0.279881` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 83398 | `0.277461` | 81788 | 322 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 27930 | `0.260425` | 27432 | 108 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.245095` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 27930 | `0.236880` | 27432 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 424 | `0.222175` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 113834 | `21.168272` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 525 | `7.567268` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 254 | `7.165956` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 348500 | `2.827395` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 233100 | `2.001038` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `0.878633` | 0 | 0 |
| `cudaMalloc_v3020` | 778 | `0.620566` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.322956` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.250262` | 0 | 0 |
| `cudaFree_v3020` | 340 | `0.247223` | 0 | 0 |
| `cuModuleLoadData` | 72 | `0.226098` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.175371` | 0 | 0 |
| `cuMemRelease` | 389 | `0.130733` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.119688` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.118409` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.073774` | 0 | 0 |
| `cuMemCreate` | 147 | `0.069649` | 0 | 0 |
| `cuLaunchKernelEx` | 8449 | `0.059372` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.058473` | 0 | 0 |
| `cuMemMap` | 194 | `0.034906` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 32247 | `0.020622` | 0 | 0 |
| `cudaGetDeviceProperties_v2_v12000` | 9 | `0.014773` | 0 | 0 |
| `cudaGraphDestroy_v10000` | 3 | `0.014577` | 0 | 0 |
| `cuKernelGetFunction` | 773 | `0.012335` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 2258 | `0.008929` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `14.141072` | ranges=1 |
| kernel | 1561348 | `12.156898` |  |
| runtime | 101646 | `12.558485` |  |
| memcpy | 59810 | `0.105932` | bytes=2713484844 |
| memset | 121 | `0.000197` |  |
| sync | 2668 | `9.161460` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.104355` | 17.31% | 0 | 0 |
| `decode_splitk_gather_split_combine` | 21590 | `0.118015` | 0.97% | 21590 | 170 |
| `indexer_logits_topk_cache` | 26516 | `1.197332` | 9.85% | 26289 | 207 |
| `runtime_copy_cat_index_kernels` | 650904 | `2.752281` | 22.64% | 605409 | 4767 |
| `fp8_projection_gemm` | 13589 | `1.172015` | 9.64% | 13589 | 107 |
| `moe_marlin_route` | 44032 | `0.583477` | 4.80% | 43688 | 344 |
| `hc_rmsnorm_logits_sampling` | 44180 | `0.627945` | 5.17% | 43815 | 345 |
| `dense_linear_other` | 87956 | `0.990731` | 8.15% | 87376 | 688 |
| `nccl_communication` | 11264 | `0.477909` | 3.93% | 11176 | 88 |
| `elementwise_math_other` | 649747 | `2.082712` | 17.13% | 639445 | 5035 |
| `other` | 11527 | `0.050127` | 0.41% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 4063 | `9.169383` | 73.01% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `2.725380` | 21.70% | 0 | 0 |
| `kernel_launch_runtime` | 57922 | `0.354202` | 2.82% | 0 | 0 |
| `memcpy_runtime` | 37839 | `0.307500` | 2.45% | 0 | 0 |
| `other` | 1695 | `0.002020` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.066028` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 173439 | `1.251775` | 171958 | 1354 |
| `_quantized_linear_fp8_kernel` | 13589 | `1.172015` | 13589 | 107 |
| `_indexer_bf16_logits_kernel` | 2688 | `1.005248` | 2667 | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 173226 | `0.734258` | 171577 | 1351 |
| `_hc_split_pre_kernel` | 11008 | `0.423419` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 80768 | `0.280192` | 80137 | 631 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.259476` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 82431 | `0.251975` | 81661 | 643 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30166 | `0.245875` | 29845 | 235 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.237509` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.233116` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.212781` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.207573` | 21717 | 171 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 82432 | `0.190134` | 81788 | 644 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.148090` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 49705 | `0.146339` | 49276 | 388 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 35712 | `0.140277` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 41216 | `0.136802` | 40894 | 322 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13824 | `0.129865` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13824 | `0.117297` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.110759` | 0 | 0 |
| `_hc_post_kernel` | 11008 | `0.107053` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.102214` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.096739` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | `5.314182` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 257 | `3.844900` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `2.725380` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 57020 | `0.347347` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 37839 | `0.307500` | 0 | 0 |
| `cuLaunchKernelEx` | 902 | `0.006856` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.003820` | 0 | 0 |
| `cudaEventQuery_v3020` | 972 | `0.003277` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001644` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1102 | `0.001103` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000925` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000606` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000367` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 296 | `0.000187` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000184` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000082` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000075` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000048` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.808262` | ranges=1 |
| kernel | 12816 | `5.741996` |  |
| runtime | 14963 | `5.507176` |  |
| memcpy | 171 | `0.003039` | bytes=2342650204 |
| memset | 21 | `0.000043` |  |
| sync | 670 | `5.420513` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.104355` | 36.65% | 0 | 0 |
| `indexer_logits_topk_cache` | 227 | `0.984503` | 17.15% | 0 | 0 |
| `runtime_copy_cat_index_kernels` | 5509 | `0.799860` | 13.93% | 0 | 0 |
| `moe_marlin_route` | 344 | `0.262743` | 4.58% | 0 | 0 |
| `hc_rmsnorm_logits_sampling` | 365 | `0.445436` | 7.76% | 0 | 0 |
| `dense_linear_other` | 580 | `0.368348` | 6.41% | 0 | 0 |
| `nccl_communication` | 88 | `0.152162` | 2.65% | 0 | 0 |
| `elementwise_math_other` | 5451 | `0.598169` | 10.42% | 0 | 0 |
| `other` | 209 | `0.026420` | 0.46% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1287 | `5.423378` | 98.48% | 0 | 0 |
| `kernel_launch_runtime` | 12816 | `0.080863` | 1.47% | 0 | 0 |
| `memcpy_runtime` | 171 | `0.002235` | 0.04% | 0 | 0 |
| `other` | 689 | `0.000701` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.066028` | 0 | 0 |
| `_indexer_bf16_logits_kernel` | 21 | `0.921839` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 1481 | `0.398812` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.352814` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 1649 | `0.289390` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.233116` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 631 | `0.152638` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.110759` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106601` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.098866` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.096739` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.083613` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076117` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069791` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 770 | `0.054168` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053490` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.053258` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 644 | `0.047403` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 429 | `0.044626` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 322 | `0.043382` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 43 | `0.038823` | 0 | 0 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.038327` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 279 | `0.036931` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 44 | `0.030726` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 279 | `0.030579` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `5.306555` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.113232` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 11914 | `0.074007` | 0 | 0 |
| `cuLaunchKernelEx` | 902 | `0.006856` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 171 | `0.002235` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001644` | 0 | 0 |
| `cudaEventQuery_v3020` | 322 | `0.000803` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000520` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000364` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000207` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 294 | `0.000186` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 198 | `0.000185` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 88 | `0.000180` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000080` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000075` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000048` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `7.052103` | ranges=1 |
| kernel | 1515775 | `6.356570` |  |
| runtime | 21387 | `6.609160` |  |
| memcpy | 27315 | `0.041585` | bytes=343979728 |
| memset | 0 | `0.000000` |  |
| sync | 1924 | `3.739837` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_splitk_gather_split_combine` | 21590 | `0.118015` | 1.86% | 21590 | 170 |
| `indexer_logits_topk_cache` | 26289 | `0.212829` | 3.35% | 26289 | 207 |
| `runtime_copy_cat_index_kernels` | 612813 | `1.894902` | 29.81% | 605409 | 4767 |
| `fp8_projection_gemm` | 13589 | `1.172015` | 18.44% | 13589 | 107 |
| `moe_marlin_route` | 43688 | `0.320734` | 5.05% | 43688 | 344 |
| `hc_rmsnorm_logits_sampling` | 43815 | `0.182509` | 2.87% | 43815 | 345 |
| `dense_linear_other` | 87376 | `0.622383` | 9.79% | 87376 | 688 |
| `nccl_communication` | 11176 | `0.325747` | 5.12% | 11176 | 88 |
| `elementwise_math_other` | 644138 | `1.483772` | 23.34% | 639445 | 5035 |
| `other` | 11301 | `0.023665` | 0.37% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2681 | `3.744750` | 56.66% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `2.725380` | 41.24% | 0 | 0 |
| `kernel_launch_runtime` | 12349 | `0.083773` | 1.27% | 0 | 0 |
| `memcpy_runtime` | 5344 | `0.054350` | 0.82% | 0 | 0 |
| `other` | 886 | `0.000906` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 13589 | `1.172015` | 13589 | 107 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 171958 | `0.852963` | 171958 | 1354 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 171577 | `0.444868` | 171577 | 1351 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.237509` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 29845 | `0.230126` | 29845 | 235 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.207573` | 21717 | 171 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 81661 | `0.197807` | 81661 | 643 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.160611` | 5461 | 43 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.159522` | 5588 | 44 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.148090` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 81788 | `0.142731` | 81788 | 644 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 80137 | `0.127553` | 80137 | 631 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 35433 | `0.109698` | 35433 | 279 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.102214` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 49276 | `0.101713` | 49276 | 388 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 40894 | `0.093421` | 40894 | 322 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::<unnamed>::pow_tensor_tensor_kernel(at...` | 35433 | `0.090125` | 35433 | 279 |
| `_indexer_bf16_logits_kernel` | 2667 | `0.083409` | 2667 | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27432 | `0.078053` | 27432 | 216 |
| `_hc_split_pre_kernel` | 10922 | `0.070605` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::log2_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::o...` | 35433 | `0.068492` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (ins...` | 35433 | `0.067052` | 35433 | 279 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13716 | `0.063806` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.060053` | 24384 | 192 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 35433 | `0.057358` | 35433 | 279 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `3.731117` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `2.725380` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 12349 | `0.083773` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 5344 | `0.054350` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 913 | `0.007104` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.003213` | 0 | 0 |
| `cudaEventQuery_v3020` | 632 | `0.002402` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000914` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000904` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2 | `0.000002` | 0 | 0 |

