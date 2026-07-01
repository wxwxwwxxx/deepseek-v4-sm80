# Post-SplitK Nsight Classification: nsys_target0754_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`5.017663`, decode_envelope_s=`5.474079`
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
| kernel | 1027408 | `10.536108` |  |
| runtime | 611537 | `25.992737` |  |
| memcpy | 203294 | `8.766202` | bytes=171698653748 |
| memset | 219 | `0.000373` |  |
| sync | 116363 | `16.325398` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 486194 | `2.613317` | 24.80% | 281559 | 2217 |
| `elementwise_graph_nodes` | 213706 | `0.918395` | 8.72% | 202438 | 1594 |
| `fp8_activation_quant_poc` | 36549 | `0.179876` | 1.71% | 35433 | 279 |
| `fp8_indexer` | 21379 | `0.257245` | 2.44% | 20828 | 164 |
| `sparse_attention_decode` | 22100 | `0.121000` | 1.15% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.116624` | 20.09% | 0 | 0 |
| `kv_compressor_cache_store` | 8362 | `0.068682` | 0.65% | 8128 | 64 |
| `projection_gemm` | 103590 | `2.217516` | 21.05% | 100965 | 795 |
| `moe_marlin` | 67080 | `0.834276` | 7.92% | 43688 | 344 |
| `nccl_communication` | 11528 | `0.519933` | 4.93% | 11176 | 88 |
| `sampling_logits` | 45131 | `0.638484` | 6.06% | 43815 | 345 |
| `unknown` | 11746 | `0.050759` | 0.48% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 121446 | `16.418736` | 63.17% | 0 | 0 |
| `cuda_graph_launch_runtime` | 667 | `2.076438` | 7.99% | 0 | 0 |
| `kernel_launch_runtime` | 263322 | `2.509458` | 9.65% | 0 | 0 |
| `memcpy_runtime` | 195031 | `1.745920` | 6.72% | 0 | 0 |
| `allocation_runtime` | 1351 | `1.868197` | 7.19% | 0 | 0 |
| `module_runtime` | 80 | `0.235306` | 0.91% | 0 | 0 |
| `other` | 29640 | `1.138683` | 4.38% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.070209` | 0 | 0 |
| `_quantized_linear_fp8_kernel` | 13910 | `1.206385` | 13589 | 107 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 99011 | `0.901394` | 95758 | 754 |
| `_hc_split_pre_kernel` | 11266 | `0.430014` | 10922 | 86 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44072 | `0.279404` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5633 | `0.269085` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30831 | `0.253176` | 29845 | 235 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5764 | `0.245568` | 5588 | 44 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.245008` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16770 | `0.241437` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235083` | 0 | 0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21974 | `0.211645` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 50097 | `0.211464` | 27432 | 216 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 34314 | `0.197609` | 32893 | 259 |
| `_fp8_activation_quantize_kernel` | 36549 | `0.179876` | 35433 | 279 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128...` | 22446 | `0.156448` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 42269 | `0.153058` | 40894 | 322 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5547 | `0.147591` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22400 | `0.140978` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 14106 | `0.131333` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 14106 | `0.120548` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.111969` | 0 | 0 |
| `_hc_post_kernel` | 11266 | `0.109176` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5590 | `0.102904` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097810` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084861` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084490` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076985` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 11008 | `0.072768` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33396 | `0.069366` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5764 | `0.061301` | 5588 | 44 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24683 | `0.061046` | 24384 | 192 |
| `_moe_route_fill_kernel` | 11266 | `0.060618` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5200 | `0.060451` | 5080 | 40 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11266 | `0.060442` | 10922 | 86 |
| `_rms_norm_bf16_kernel` | 17030 | `0.059912` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat...` | 22016 | `0.059660` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5633 | `0.059234` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::float8_copy_kernel_cuda(at::Ten...` | 11094 | `0.052566` | 0 | 0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16857 | `0.052547` | 16510 | 130 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)3, (int)128...` | 172 | `0.049741` | 0 | 0 |
| `_rotary_tail_kernel` | 11093 | `0.048820` | 10795 | 85 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 21257 | `0.047991` | 19177 | 151 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046416` | 0 | 0 |
| `_sparse_splitk_bf16_split_kernel` | 5590 | `0.045785` | 5461 | 43 |
| `_indexer_fp8_quantize_fold_kernel` | 2751 | `0.041729` | 2667 | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 29517 | `0.041012` | 27305 | 215 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10836 | `0.040631` | 10668 | 84 |
| `_sparse_bf16_gather_with_mask_kernel` | 10920 | `0.040577` | 10668 | 84 |
| `_sparse_splitk_bf16_combine_kernel` | 5590 | `0.034638` | 5461 | 43 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 112789 | `13.240706` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 267 | `3.154083` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 252903 | `2.431474` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.853767` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 195031 | `1.745920` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `0.985785` | 0 | 0 |
| `cudaMalloc_v3020` | 781 | `0.600606` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.341327` | 0 | 0 |
| `cudaFree_v3020` | 343 | `0.281068` | 0 | 0 |
| `cuModuleLoadData` | 76 | `0.235303` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.190261` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.162477` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.133139` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.106756` | 0 | 0 |
| `cuMemRelease` | 389 | `0.104291` | 0 | 0 |
| `cuMemCreate` | 147 | `0.090581` | 0 | 0 |
| `cuLaunchKernelEx` | 9647 | `0.072467` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.058812` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.047912` | 0 | 0 |
| `cuMemMap` | 194 | `0.037010` | 0 | 0 |
| `cudaGetDeviceProperties_v2_v12000` | 9 | `0.017070` | 0 | 0 |
| `cuKernelGetFunction` | 773 | `0.016403` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 17900 | `0.011757` | 0 | 0 |
| `cudaGraphDestroy_v10000` | 3 | `0.011719` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 1942 | `0.010085` | 0 | 0 |
| `cudaEventRecord_v3020` | 2117 | `0.006725` | 0 | 0 |
| `cudaEventQuery_v3020` | 1640 | `0.005611` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 1393 | `0.005580` | 0 | 0 |
| `cuLaunchKernel` | 772 | `0.005517` | 0 | 0 |
| `cudaStreamCreateWithFlags_v5000` | 268 | `0.005142` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 3075 | `0.003600` | 0 | 0 |
| `cudaStreamCreateWithPriority_v5050` | 128 | `0.003150` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 1539 | `0.002613` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 219 | `0.001813` | 0 | 0 |
| `cuMemAddressReserve` | 194 | `0.001731` | 0 | 0 |
| `cudaStreamDestroy_v5050` | 268 | `0.001549` | 0 | 0 |
| `cudaStreamEndCapture_v10000` | 3 | `0.001530` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 683 | `0.001218` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000872` | 0 | 0 |
| `cudaMemGetInfo_v3020` | 8 | `0.000839` | 0 | 0 |
| `cudaGraphAddEventRecordNode_v11010` | 264 | `0.000689` | 0 | 0 |
| `cuMemAddressFree` | 194 | `0.000646` | 0 | 0 |
| `cudaEventDestroy_v3020` | 626 | `0.000640` | 0 | 0 |
| `cuGetProcAddress_v2` | 1761 | `0.000434` | 0 | 0 |
| `cudaStreamUpdateCaptureDependencies_v11030` | 531 | `0.000336` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 616 | `0.000302` | 0 | 0 |
| `cudaGraphAddDependencies_v10000` | 264 | `0.000253` | 0 | 0 |
| `cuMemRetainAllocationHandle` | 194 | `0.000235` | 0 | 0 |
| `cudaGraphRetainUserObject_v11030` | 264 | `0.000227` | 0 | 0 |
| `cudaUserObjectCreate_v11030` | 264 | `0.000140` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `12.045910` | ranges=1 |
| kernel | 832816 | `9.173280` |  |
| runtime | 95801 | `9.654687` |  |
| memcpy | 46454 | `0.098659` | bytes=2686084012 |
| memset | 121 | `0.000200` |  |
| sync | 2632 | `6.832578` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 324420 | `1.610654` | 17.56% | 281559 | 2217 |
| `elementwise_graph_nodes` | 209299 | `0.897079` | 9.78% | 202438 | 1594 |
| `fp8_activation_quant_poc` | 35712 | `0.177355` | 1.93% | 35433 | 279 |
| `fp8_indexer` | 20992 | `0.254195` | 2.77% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.117934` | 1.29% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.116624` | 23.07% | 0 | 0 |
| `kv_compressor_cache_store` | 8212 | `0.067968` | 0.74% | 8128 | 64 |
| `projection_gemm` | 101545 | `2.169174` | 23.65% | 100965 | 795 |
| `moe_marlin` | 44032 | `0.581862` | 6.34% | 43688 | 344 |
| `nccl_communication` | 11264 | `0.496521` | 5.41% | 11176 | 88 |
| `sampling_logits` | 44180 | `0.633819` | 6.91% | 43815 | 345 |
| `unknown` | 11527 | `0.050093` | 0.55% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 4030 | `6.840978` | 70.86% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.853767` | 19.20% | 0 | 0 |
| `kernel_launch_runtime` | 52147 | `0.410592` | 4.25% | 0 | 0 |
| `memcpy_runtime` | 37818 | `0.313524` | 3.25% | 0 | 0 |
| `allocation_runtime` | 23 | `0.016975` | 0.18% | 0 | 0 |
| `module_runtime` | 15 | `0.216938` | 2.25% | 0 | 0 |
| `other` | 1641 | `0.001913` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.070209` | 0 | 0 |
| `_quantized_linear_fp8_kernel` | 13589 | `1.172625` | 13589 | 107 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 96639 | `0.881340` | 95758 | 754 |
| `_hc_split_pre_kernel` | 11008 | `0.428012` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.264863` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30166 | `0.245415` | 29845 | 235 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.237089` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235083` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.226557` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.208847` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 33366 | `0.190001` | 32893 | 259 |
| `_fp8_activation_quantize_kernel` | 35712 | `0.177355` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 41343 | `0.148330` | 40894 | 322 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.145324` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13824 | `0.130606` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13824 | `0.118356` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.111969` | 0 | 0 |
| `_hc_post_kernel` | 11008 | `0.108330` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.101005` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097810` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27648 | `0.087586` | 27432 | 216 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084861` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084490` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076985` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33394 | `0.069361` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5632 | `0.060253` | 5588 | 44 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.059992` | 24384 | 192 |
| `_moe_route_fill_kernel` | 11008 | `0.059565` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5120 | `0.059394` | 5080 | 40 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11008 | `0.058865` | 10922 | 86 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5504 | `0.058617` | 5461 | 43 |
| `_rms_norm_bf16_kernel` | 16640 | `0.058511` | 16510 | 130 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16597 | `0.051430` | 16510 | 130 |
| `_rotary_tail_kernel` | 10880 | `0.048180` | 10795 | 85 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20972 | `0.047081` | 19177 | 151 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046416` | 0 | 0 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044530` | 5461 | 43 |
| `_indexer_fp8_quantize_fold_kernel` | 2688 | `0.041500` | 2667 | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10752 | `0.040181` | 10668 | 84 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039765` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28677 | `0.039196` | 27305 | 215 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 5396 | `0.034018` | 5334 | 42 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.033638` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` | 19456 | `0.032670` | 19304 | 152 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2688 | `0.031497` | 2667 | 21 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at:...` | 5120 | `0.031315` | 5080 | 40 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 5504 | `0.030903` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 10644 | `0.029755` | 10541 | 83 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.029236` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 11008 | `0.028808` | 10922 | 86 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | `3.677131` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 257 | `3.152887` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.853767` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 50945 | `0.400774` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 37818 | `0.313524` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.216938` | 0 | 0 |
| `cudaMalloc_v3020` | 15 | `0.014243` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009818` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.004129` | 0 | 0 |
| `cudaEventQuery_v3020` | 936 | `0.003651` | 0 | 0 |
| `cudaHostAlloc_v3020` | 8 | `0.002732` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001628` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1117 | `0.000968` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000872` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000682` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000384` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 92 | `0.000209` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 226 | `0.000144` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000087` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000081` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000037` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000002` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.163017` | ranges=1 |
| kernel | 7041 | `4.259001` |  |
| runtime | 9089 | `4.066705` |  |
| memcpy | 150 | `0.002966` | bytes=2320630108 |
| memset | 21 | `0.000043` |  |
| sync | 633 | `3.780823` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 2875 | `0.353822` | 8.31% | 0 | 0 |
| `elementwise_graph_nodes` | 2010 | `0.256673` | 6.03% | 0 | 0 |
| `fp8_activation_quant_poc` | 279 | `0.101456` | 2.38% | 0 | 0 |
| `fp8_indexer` | 164 | `0.123080` | 2.89% | 0 | 0 |
| `prefill_sparse_attention` | 43 | `2.116624` | 49.70% | 0 | 0 |
| `kv_compressor_cache_store` | 84 | `0.039861` | 0.94% | 0 | 0 |
| `projection_gemm` | 580 | `0.372379` | 8.74% | 0 | 0 |
| `moe_marlin` | 344 | `0.264889` | 6.22% | 0 | 0 |
| `nccl_communication` | 88 | `0.153692` | 3.61% | 0 | 0 |
| `sampling_logits` | 365 | `0.450008` | 10.57% | 0 | 0 |
| `unknown` | 209 | `0.026517` | 0.62% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1251 | `3.783748` | 93.04% | 0 | 0 |
| `kernel_launch_runtime` | 7041 | `0.052719` | 1.30% | 0 | 0 |
| `memcpy_runtime` | 150 | `0.001638` | 0.04% | 0 | 0 |
| `allocation_runtime` | 7 | `0.011039` | 0.27% | 0 | 0 |
| `module_runtime` | 15 | `0.216938` | 5.33% | 0 | 0 |
| `other` | 625 | `0.000624` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.070209` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.357261` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 881 | `0.300912` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235083` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.111969` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106541` | 0 | 0 |
| `_fp8_activation_quantize_kernel` | 279 | `0.101456` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.098024` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097810` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084490` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076985` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069905` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 21 | `0.065361` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.055629` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053586` | 0 | 0 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046416` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 43 | `0.039392` | 0 | 0 |
| `_indexer_fp8_quantize_fold_kernel` | 21 | `0.034189` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 44 | `0.030969` | 0 | 0 |
| `_rotary_tail_kernel` | 85 | `0.026202` | 0 | 0 |
| `_moe_route_fill_kernel` | 86 | `0.025411` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 43 | `0.019423` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 449 | `0.016516` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 321 | `0.015646` | 0 | 0 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 40 | `0.015078` | 0 | 0 |
| `_rms_norm_bf16_kernel` | 130 | `0.014263` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 86 | `0.012654` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 86 | `0.009970` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 216 | `0.008957` | 0 | 0 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 62 | `0.008538` | 0 | 0 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 21 | `0.007388` | 0 | 0 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 84 | `0.005998` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 103 | `0.005484` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_kernel(at::TensorIteratorBase &)::[lambda() (instance ...` | 43 | `0.005133` | 0 | 0 |
| `_moe_route_count_kernel` | 86 | `0.004205` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 473 | `0.003917` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<fl...` | 62 | `0.002024` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 150 | `0.001874` | 0 | 0 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` | 67 | `0.001675` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c1...` | 1 | `0.001117` | 0 | 0 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at:...` | 40 | `0.000927` | 0 | 0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 87 | `0.000904` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::softplus_kernel(at::TensorIteratorBase &, const c10::Scalar...` | 43 | `0.000888` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 21 | `0.000731` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 129 | `0.000664` | 0 | 0 |
| `_silu_and_mul_clamp_kernel` | 43 | `0.000598` | 0 | 0 |
| `void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)8, void at::native::_cuda_scatter_gather_internal_kernel<(bool)0, at::...` | 43 | `0.000596` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::o...` | 43 | `0.000579` | 0 | 0 |
| `_compress_norm_rope_store_bf16_kernel` | 41 | `0.000470` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 217 | `0.000394` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `3.668821` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.216938` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.111215` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 5839 | `0.042901` | 0 | 0 |
| `cudaMalloc_v3020` | 7 | `0.011039` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009818` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 150 | `0.001638` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001628` | 0 | 0 |
| `cudaEventQuery_v3020` | 285 | `0.000877` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000549` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000377` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000196` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 205 | `0.000186` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000178` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 222 | `0.000141` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000085` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000081` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000037` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000002` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.474079` | ranges=1 |
| kernel | 793018 | `4.844093` |  |
| runtime | 21391 | `5.059623` |  |
| memcpy | 13980 | `0.023481` | bytes=338598992 |
| memset | 0 | `0.000000` |  |
| sync | 1926 | `3.050934` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 288963 | `1.187490` | 24.51% | 281559 | 2217 |
| `elementwise_graph_nodes` | 207131 | `0.639607` | 13.20% | 202438 | 1594 |
| `fp8_activation_quant_poc` | 35433 | `0.075900` | 1.57% | 35433 | 279 |
| `fp8_indexer` | 20828 | `0.131115` | 2.71% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.117934` | 2.43% | 21590 | 170 |
| `kv_compressor_cache_store` | 8128 | `0.028107` | 0.58% | 8128 | 64 |
| `projection_gemm` | 100965 | `1.796796` | 37.09% | 100965 | 795 |
| `moe_marlin` | 43688 | `0.316973` | 6.54% | 43688 | 344 |
| `nccl_communication` | 11176 | `0.342830` | 7.08% | 11176 | 88 |
| `sampling_logits` | 43815 | `0.183811` | 3.79% | 43815 | 345 |
| `unknown` | 11301 | `0.023531` | 0.49% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2683 | `3.056208` | 60.40% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.853767` | 36.64% | 0 | 0 |
| `kernel_launch_runtime` | 12349 | `0.093403` | 1.85% | 0 | 0 |
| `memcpy_runtime` | 5344 | `0.055487` | 1.10% | 0 | 0 |
| `other` | 888 | `0.000759` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 13589 | `1.172625` | 13589 | 107 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 95758 | `0.580428` | 95758 | 754 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.237089` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 29845 | `0.229770` | 29845 | 235 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.208847` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 32893 | `0.186084` | 32893 | 259 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.170928` | 5588 | 44 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.166839` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.145324` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 40894 | `0.131814` | 40894 | 322 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.101005` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27432 | `0.078628` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 35433 | `0.075900` | 35433 | 279 |
| `_hc_split_pre_kernel` | 10922 | `0.070750` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13716 | `0.064769` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.059992` | 24384 | 192 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16510 | `0.050525` | 16510 | 130 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20815 | `0.046231` | 19177 | 151 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 10922 | `0.046211` | 10922 | 86 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044530` | 5461 | 43 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5080 | `0.044316` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 16510 | `0.044248` | 16510 | 130 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039765` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28439 | `0.038704` | 27305 | 215 |
| `_hc_post_kernel` | 10922 | `0.038425` | 10922 | 86 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10668 | `0.034183` | 10668 | 84 |
| `_moe_route_fill_kernel` | 10922 | `0.034154` | 10922 | 86 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.033638` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` | 19304 | `0.032295` | 19304 | 152 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at:...` | 5080 | `0.030388` | 5080 | 40 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5588 | `0.029284` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.029236` | 5461 | 43 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cub...` | 127 | `0.026627` | 127 | 1 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 5334 | `0.025479` | 5334 | 42 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::...` | 13716 | `0.024822` | 13716 | 108 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 10541 | `0.024272` | 10541 | 83 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2667 | `0.024109` | 2667 | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13716 | `0.024065` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<fl...` | 5334 | `0.022586` | 5334 | 42 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 13843 | `0.022586` | 13843 | 109 |
| `_rotary_tail_kernel` | 10795 | `0.021978` | 10795 | 85 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 11677 | `0.021049` | 10795 | 85 |
| `_moe_route_offsets_kernel` | 10922 | `0.019847` | 10922 | 86 |
| `_indexer_fp8_paged_logits_kernel` | 2667 | `0.019501` | 2667 | 21 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5461 | `0.019225` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 10922 | `0.018838` | 10922 | 86 |
| `_moe_route_count_kernel` | 10922 | `0.016644` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::softplus_kernel(at::TensorIteratorBase &, const c10::Scalar...` | 5461 | `0.015915` | 5461 | 43 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<fl...` | 5461 | `0.014761` | 5461 | 43 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` | 5715 | `0.011887` | 5715 | 45 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `3.041503` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.853767` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 12349 | `0.093403` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 5344 | `0.055487` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 913 | `0.007700` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.003454` | 0 | 0 |
| `cudaEventQuery_v3020` | 634 | `0.002691` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000859` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000755` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 4 | `0.000004` | 0 | 0 |

