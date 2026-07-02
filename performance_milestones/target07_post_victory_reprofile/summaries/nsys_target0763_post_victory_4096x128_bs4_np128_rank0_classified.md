# Post-SplitK Nsight Classification: nsys_target0763_post_victory_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`3.513694`, decode_envelope_s=`3.942934`
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
| kernel | 980044 | `9.058767` |  |
| runtime | 608437 | `27.507168` |  |
| memcpy | 203294 | `12.192556` | bytes=171698135588 |
| memset | 219 | `0.000399` |  |
| sync | 116390 | `18.968346` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 103633 | `1.204027` | 13.29% | 100965 | 795 |
| `graph_runtime_copy_cat_index` | 444377 | `2.258605` | 24.93% | 240665 | 1895 |
| `elementwise_graph_nodes` | 208116 | `0.773287` | 8.54% | 196977 | 1551 |
| `moe_marlin` | 67080 | `0.815727` | 9.00% | 43688 | 344 |
| `nccl_communication` | 11528 | `0.570243` | 6.29% | 11176 | 88 |
| `sampling_logits` | 45131 | `0.636398` | 7.03% | 43815 | 345 |
| `fp8_indexer` | 21379 | `0.257553` | 2.84% | 20828 | 164 |
| `sparse_attention_decode` | 22100 | `0.121512` | 1.34% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.121776` | 23.42% | 0 | 0 |
| `kv_compressor_cache_store` | 8362 | `0.068625` | 0.76% | 8128 | 64 |
| `unknown` | 48295 | `0.231014` | 2.55% | 46482 | 366 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 121473 | `19.058800` | 69.29% | 0 | 0 |
| `cuda_graph_launch_runtime` | 667 | `1.381154` | 5.02% | 0 | 0 |
| `kernel_launch_runtime` | 261261 | `2.473687` | 8.99% | 0 | 0 |
| `memcpy_runtime` | 195031 | `1.728131` | 6.28% | 0 | 0 |
| `allocation_runtime` | 1429 | `1.709328` | 6.21% | 0 | 0 |
| `module_runtime` | 74 | `0.227513` | 0.83% | 0 | 0 |
| `other` | 28502 | `0.928555` | 3.38% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.075377` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 73921 | `0.695106` | 71247 | 561 |
| `_hc_split_pre_kernel` | 11266 | `0.429891` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5633 | `0.323992` | 5461 | 43 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44072 | `0.276452` | 0 | 0 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.242532` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5764 | `0.240254` | 5588 | 44 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16770 | `0.237219` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235409` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 50140 | `0.211387` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 36549 | `0.179916` | 35433 | 279 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16470 | `0.170318` | 16256 | 128 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128...` | 22446 | `0.156253` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22400 | `0.139867` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5547 | `0.134453` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 14106 | `0.130511` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 11094 | `0.128414` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 14106 | `0.120069` | 13716 | 108 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 25241 | `0.113134` | 24384 | 192 |
| `_hc_post_kernel` | 11266 | `0.107163` | 10922 | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 23134 | `0.105075` | 21971 | 173 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 36679 | `0.102013` | 35433 | 279 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5590 | `0.099638` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097759` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089657` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084739` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084462` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077147` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 11008 | `0.072719` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33396 | `0.069173` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5764 | `0.060918` | 5588 | 44 |
| `_moe_route_fill_kernel` | 11266 | `0.060853` | 10922 | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11266 | `0.060636` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5200 | `0.060427` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 17030 | `0.059918` | 16510 | 130 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24726 | `0.059729` | 24384 | 192 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat...` | 22016 | `0.059346` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5633 | `0.059272` | 5461 | 43 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5547 | `0.058228` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16857 | `0.052567` | 16510 | 130 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 112789 | `16.737216` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 251656 | `2.407795` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 267 | `2.300346` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 195031 | `1.728131` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.217009` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `0.892991` | 0 | 0 |
| `cudaMalloc_v3020` | 861 | `0.590322` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.287959` | 0 | 0 |
| `cuModuleLoadData` | 70 | `0.227510` | 0 | 0 |
| `cudaFree_v3020` | 341 | `0.225116` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.140408` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.121544` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.104357` | 0 | 0 |
| `cuMemCreate` | 147 | `0.082868` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.072451` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.069623` | 0 | 0 |
| `cuMemRelease` | 389 | `0.067199` | 0 | 0 |
| `cuLaunchKernelEx` | 9005 | `0.062190` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.034830` | 0 | 0 |
| `cuMemMap` | 194 | `0.029074` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 1992 | `0.019416` | 0 | 0 |
| `cudaGetDeviceProperties_v2_v12000` | 9 | `0.015593` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 16805 | `0.011063` | 0 | 0 |
| `cuKernelGetFunction` | 601 | `0.008995` | 0 | 0 |
| `cudaGraphDestroy_v10000` | 3 | `0.007264` | 0 | 0 |
| `cudaEventRecord_v3020` | 2117 | `0.005935` | 0 | 0 |
| `cudaEventQuery_v3020` | 1667 | `0.005131` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 1393 | `0.004668` | 0 | 0 |
| `cudaStreamCreateWithFlags_v5000` | 268 | `0.004114` | 0 | 0 |
| `cuLaunchKernel` | 600 | `0.003702` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 3155 | `0.003657` | 0 | 0 |
| `cudaStreamCreateWithPriority_v5050` | 128 | `0.003320` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 1539 | `0.002453` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 219 | `0.001513` | 0 | 0 |
| `cudaStreamDestroy_v5050` | 268 | `0.001427` | 0 | 0 |
| `cuMemAddressReserve` | 194 | `0.001343` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 683 | `0.001124` | 0 | 0 |
| `cudaStreamEndCapture_v10000` | 3 | `0.001000` | 0 | 0 |
| `cudaMemGetInfo_v3020` | 8 | `0.000848` | 0 | 0 |
| `cuGetProcAddress_v2` | 1761 | `0.000842` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `10.461228` | ranges=1 |
| kernel | 785561 | `7.683257` |  |
| runtime | 94976 | `8.105740` |  |
| memcpy | 46454 | `0.101256` | bytes=2685565852 |
| memset | 121 | `0.000213` |  |
| sync | 2657 | `5.982925` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 101545 | `1.184404` | 15.42% | 100965 | 795 |
| `graph_runtime_copy_cat_index` | 282776 | `1.259741` | 16.40% | 240665 | 1895 |
| `elementwise_graph_nodes` | 203688 | `0.750836` | 9.77% | 196977 | 1551 |
| `moe_marlin` | 44032 | `0.565865` | 7.36% | 43688 | 344 |
| `nccl_communication` | 11264 | `0.500140` | 6.51% | 11176 | 88 |
| `sampling_logits` | 44180 | `0.631772` | 8.22% | 43815 | 345 |
| `fp8_indexer` | 20992 | `0.254509` | 3.31% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.118446` | 1.54% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.121776` | 27.62% | 0 | 0 |
| `kv_compressor_cache_store` | 8212 | `0.067919` | 0.88% | 8128 | 64 |
| `unknown` | 47239 | `0.227850` | 2.97% | 46482 | 366 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 4055 | `5.990077` | 73.90% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.217009` | 15.01% | 0 | 0 |
| `kernel_launch_runtime` | 51247 | `0.369334` | 4.56% | 0 | 0 |
| `memcpy_runtime` | 37818 | `0.291030` | 3.59% | 0 | 0 |
| `allocation_runtime` | 23 | `0.015158` | 0.19% | 0 | 0 |
| `module_runtime` | 15 | `0.221247` | 2.73% | 0 | 0 |
| `other` | 1691 | `0.001885` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.075377` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 71828 | `0.675252` | 71247 | 561 |
| `_hc_split_pre_kernel` | 11008 | `0.427893` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.264638` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235409` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.232933` | 16510 | 130 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.229684` | 5588 | 44 |
| `_fp8_activation_quantize_kernel` | 35712 | `0.177404` | 35433 | 279 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.167958` | 16256 | 128 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.132209` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13824 | `0.129789` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.126154` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13824 | `0.117915` | 13716 | 108 |
| `_hc_post_kernel` | 11008 | `0.106343` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 24555 | `0.104099` | 24384 | 192 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.097777` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097759` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22144 | `0.096693` | 21971 | 173 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 35732 | `0.096683` | 35433 | 279 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089657` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27648 | `0.087749` | 27432 | 216 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084739` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084462` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077147` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33394 | `0.069169` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5632 | `0.059888` | 5588 | 44 |
| `_moe_route_fill_kernel` | 11008 | `0.059805` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5120 | `0.059374` | 5080 | 40 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11008 | `0.059097` | 10922 | 86 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5504 | `0.058661` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.058567` | 24384 | 192 |
| `_rms_norm_bf16_kernel` | 16640 | `0.058522` | 16510 | 130 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.057154` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16597 | `0.051455` | 16510 | 130 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050902` | 2667 | 21 |
| `_rotary_tail_kernel` | 10880 | `0.048545` | 10795 | 85 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20972 | `0.047183` | 19177 | 151 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046399` | 0 | 0 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044617` | 5461 | 43 |
| `_indexer_fp8_quantize_fold_kernel` | 2688 | `0.041675` | 2667 | 21 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | `3.681347` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 257 | `2.299167` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.217009` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 50045 | `0.360022` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 37818 | `0.291030` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.221247` | 0 | 0 |
| `cudaMalloc_v3020` | 15 | `0.012633` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009312` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.003460` | 0 | 0 |
| `cudaEventQuery_v3020` | 961 | `0.003252` | 0 | 0 |
| `cudaHostAlloc_v3020` | 8 | `0.002525` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001403` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1117 | `0.000999` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000783` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000635` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000380` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 92 | `0.000198` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 276 | `0.000121` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000086` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000076` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000048` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000005` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.151027` | ranges=1 |
| kernel | 6141 | `4.256206` |  |
| runtime | 8260 | `4.064618` |  |
| memcpy | 150 | `0.002961` | bytes=2320630108 |
| memset | 21 | `0.000046` |  |
| sync | 656 | `3.785400` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 580 | `0.372304` | 8.75% | 0 | 0 |
| `graph_runtime_copy_cat_index` | 2125 | `0.343805` | 8.08% | 0 | 0 |
| `elementwise_graph_nodes` | 1860 | `0.252072` | 5.92% | 0 | 0 |
| `moe_marlin` | 344 | `0.265349` | 6.23% | 0 | 0 |
| `nccl_communication` | 88 | `0.160126` | 3.76% | 0 | 0 |
| `sampling_logits` | 365 | `0.449904` | 10.57% | 0 | 0 |
| `fp8_indexer` | 164 | `0.123062` | 2.89% | 0 | 0 |
| `prefill_sparse_attention` | 43 | `2.121776` | 49.85% | 0 | 0 |
| `kv_compressor_cache_store` | 84 | `0.039856` | 0.94% | 0 | 0 |
| `unknown` | 488 | `0.127952` | 3.01% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1274 | `3.787822` | 93.19% | 0 | 0 |
| `kernel_launch_runtime` | 6141 | `0.043525` | 1.07% | 0 | 0 |
| `memcpy_runtime` | 150 | `0.001527` | 0.04% | 0 | 0 |
| `allocation_runtime` | 7 | `0.009867` | 0.24% | 0 | 0 |
| `module_runtime` | 15 | `0.221247` | 5.44% | 0 | 0 |
| `other` | 673 | `0.000631` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.075377` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.357228` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 581 | `0.295785` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235409` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106565` | 0 | 0 |
| `_fp8_activation_quantize_kernel` | 279 | `0.101429` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.097939` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097759` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089657` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084462` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077147` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069825` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 21 | `0.065345` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.062149` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053530` | 0 | 0 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046399` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 43 | `0.039386` | 0 | 0 |
| `_indexer_fp8_quantize_fold_kernel` | 21 | `0.034177` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 44 | `0.030931` | 0 | 0 |
| `_rotary_tail_kernel` | 85 | `0.026207` | 0 | 0 |
| `_moe_route_fill_kernel` | 86 | `0.025405` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn` | 43 | `0.022161` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 43 | `0.019426` | 0 | 0 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 40 | `0.015087` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 299 | `0.014665` | 0 | 0 |
| `_rms_norm_bf16_kernel` | 130 | `0.014267` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 86 | `0.012643` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 171 | `0.011057` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 86 | `0.009983` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 216 | `0.008957` | 0 | 0 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 62 | `0.008543` | 0 | 0 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 21 | `0.007389` | 0 | 0 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 84 | `0.006008` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 103 | `0.005477` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_kernel(at::TensorIteratorBase &)::[lambda() (instance ...` | 43 | `0.005155` | 0 | 0 |
| `_moe_route_count_kernel` | 86 | `0.004331` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<fl...` | 62 | `0.002038` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 150 | `0.001877` | 0 | 0 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` | 67 | `0.001673` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c1...` | 1 | `0.001117` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `3.673346` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.221247` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.111440` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 4939 | `0.034214` | 0 | 0 |
| `cudaMalloc_v3020` | 7 | `0.009867` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009312` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 150 | `0.001527` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001403` | 0 | 0 |
| `cudaEventQuery_v3020` | 308 | `0.000594` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000395` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000375` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 205 | `0.000210` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000187` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000173` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 270 | `0.000119` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000083` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000076` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000048` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000005` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `3.942934` | ranges=1 |
| kernel | 746663 | `3.357065` |  |
| runtime | 21395 | `3.547246` |  |
| memcpy | 13980 | `0.023252` | bytes=338080832 |
| memset | 0 | `0.000000` |  |
| sync | 1928 | `2.196462` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 100965 | `0.812100` | 24.19% | 100965 | 795 |
| `graph_runtime_copy_cat_index` | 248069 | `0.846795` | 25.22% | 240665 | 1895 |
| `elementwise_graph_nodes` | 201670 | `0.497965` | 14.83% | 196977 | 1551 |
| `moe_marlin` | 43688 | `0.300516` | 8.95% | 43688 | 344 |
| `nccl_communication` | 11176 | `0.340015` | 10.13% | 11176 | 88 |
| `sampling_logits` | 43815 | `0.181868` | 5.42% | 43815 | 345 |
| `fp8_indexer` | 20828 | `0.131447` | 3.92% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.118446` | 3.53% | 21590 | 170 |
| `kv_compressor_cache_store` | 8128 | `0.028062` | 0.84% | 8128 | 64 |
| `unknown` | 46734 | `0.099853` | 2.97% | 46482 | 366 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2685 | `2.201012` | 62.05% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.217009` | 34.31% | 0 | 0 |
| `kernel_launch_runtime` | 12349 | `0.080379` | 2.27% | 0 | 0 |
| `memcpy_runtime` | 5344 | `0.048083` | 1.36% | 0 | 0 |
| `other` | 890 | `0.000763` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 71247 | `0.379467` | 71247 | 561 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.232933` | 16510 | 130 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.167958` | 16256 | 128 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.167535` | 5588 | 44 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.166699` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.132209` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.126154` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.097777` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 21971 | `0.095799` | 21971 | 173 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 24384 | `0.093041` | 24384 | 192 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 35433 | `0.082018` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27432 | `0.078792` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 35433 | `0.075976` | 35433 | 279 |
| `_hc_split_pre_kernel` | 10922 | `0.070665` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13716 | `0.064385` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.058567` | 24384 | 192 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.057154` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050902` | 2667 | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16510 | `0.050550` | 16510 | 130 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 10922 | `0.046454` | 10922 | 86 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20815 | `0.046350` | 19177 | 151 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044617` | 5461 | 43 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5080 | `0.044287` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 16510 | `0.044256` | 16510 | 130 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.040171` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28439 | `0.038690` | 27305 | 215 |
| `_hc_post_kernel` | 10922 | `0.036518` | 10922 | 86 |
| `_moe_route_fill_kernel` | 10922 | `0.034400` | 10922 | 86 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10668 | `0.034110` | 10668 | 84 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.033657` | 5461 | 43 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at:...` | 5080 | `0.030488` | 5080 | 40 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` | 19304 | `0.030398` | 19304 | 152 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.029315` | 5461 | 43 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5588 | `0.028956` | 5588 | 44 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cub...` | 127 | `0.026498` | 127 | 1 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 5334 | `0.025462` | 5334 | 42 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2667 | `0.024263` | 2667 | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 10541 | `0.024226` | 10541 | 83 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::...` | 13716 | `0.023633` | 13716 | 108 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13716 | `0.023225` | 13716 | 108 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `2.187331` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.217009` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 12349 | `0.080379` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 5344 | `0.048083` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 913 | `0.007362` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.002961` | 0 | 0 |
| `cudaEventQuery_v3020` | 636 | `0.002584` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000773` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000761` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 6 | `0.000002` | 0 | 0 |

