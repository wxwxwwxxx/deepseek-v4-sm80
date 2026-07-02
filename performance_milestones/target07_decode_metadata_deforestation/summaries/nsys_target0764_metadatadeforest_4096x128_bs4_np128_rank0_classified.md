# Post-SplitK Nsight Classification: nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`3.520566`, decode_envelope_s=`4.010320`
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
| kernel | 971662 | `8.990742` |  |
| runtime | 598335 | `29.477339` |  |
| memcpy | 201897 | `13.684610` | bytes=171697025608 |
| memset | 219 | `0.000387` |  |
| sync | 115986 | `20.387695` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 103633 | `1.204510` | 13.40% | 100965 | 795 |
| `graph_runtime_copy_cat_index` | 439678 | `2.252446` | 25.05% | 240665 | 1895 |
| `elementwise_graph_nodes` | 204306 | `0.764671` | 8.51% | 196977 | 1551 |
| `moe_marlin` | 67080 | `0.818279` | 9.10% | 43688 | 344 |
| `nccl_communication` | 11528 | `0.506401` | 5.63% | 11176 | 88 |
| `sampling_logits` | 45131 | `0.637056` | 7.09% | 43815 | 345 |
| `fp8_indexer` | 21379 | `0.257485` | 2.86% | 20828 | 164 |
| `sparse_attention_decode` | 22100 | `0.121450` | 1.35% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.128245` | 23.67% | 0 | 0 |
| `kv_compressor_cache_store` | 8362 | `0.068614` | 0.76% | 8128 | 64 |
| `unknown` | 48422 | `0.231585` | 2.58% | 46482 | 366 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 121069 | `20.487116` | 69.50% | 0 | 0 |
| `cuda_graph_launch_runtime` | 667 | `1.378934` | 4.68% | 0 | 0 |
| `kernel_launch_runtime` | 252879 | `2.520204` | 8.55% | 0 | 0 |
| `memcpy_runtime` | 193634 | `1.820262` | 6.18% | 0 | 0 |
| `allocation_runtime` | 1429 | `1.903366` | 6.46% | 0 | 0 |
| `module_runtime` | 201 | `0.246260` | 0.84% | 0 | 0 |
| `other` | 28456 | `1.121196` | 3.80% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.081836` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 73921 | `0.695466` | 71247 | 561 |
| `_hc_split_pre_kernel` | 11266 | `0.430732` | 10922 | 86 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44072 | `0.278995` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5633 | `0.269340` | 5461 | 43 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.245077` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16770 | `0.237241` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235312` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5764 | `0.230288` | 5588 | 44 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 50140 | `0.212387` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 36549 | `0.179939` | 35433 | 279 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16470 | `0.170345` | 16256 | 128 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128...` | 22446 | `0.156376` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22019 | `0.140242` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5547 | `0.134399` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 14106 | `0.130534` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 11094 | `0.128407` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 14106 | `0.120027` | 13716 | 108 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 25241 | `0.113159` | 24384 | 192 |
| `_hc_post_kernel` | 11266 | `0.106981` | 10922 | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 23134 | `0.105093` | 21971 | 173 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 36679 | `0.102214` | 35433 | 279 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5590 | `0.099791` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097934` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089651` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084725` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084559` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077236` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 11008 | `0.072752` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 32253 | `0.066955` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5764 | `0.060948` | 5588 | 44 |
| `_moe_route_fill_kernel` | 11266 | `0.060837` | 10922 | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11266 | `0.060641` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5200 | `0.060409` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 17030 | `0.059925` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat...` | 22016 | `0.059898` | 0 | 0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24726 | `0.059724` | 24384 | 192 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5633 | `0.059257` | 5461 | 43 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5547 | `0.058227` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::float8_copy_kernel_cuda(at::Ten...` | 11094 | `0.052554` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 112408 | `18.183891` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 243147 | `2.449244` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 267 | `2.281529` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 193634 | `1.820262` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.233787` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `1.007613` | 0 | 0 |
| `cudaMalloc_v3020` | 861 | `0.619311` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.353273` | 0 | 0 |
| `cudaFree_v3020` | 341 | `0.275927` | 0 | 0 |
| `cuModuleLoadData` | 197 | `0.246257` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.161017` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.134236` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.120083` | 0 | 0 |
| `cuMemRelease` | 389 | `0.098730` | 0 | 0 |
| `cuMemCreate` | 147 | `0.094702` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.088948` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.072637` | 0 | 0 |
| `cuLaunchKernelEx` | 9132 | `0.067177` | 0 | 0 |
| `cuMemMap` | 194 | `0.036095` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 1946 | `0.024814` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.017482` | 0 | 0 |
| `cudaGetDeviceProperties_v2_v12000` | 9 | `0.015341` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 16805 | `0.011509` | 0 | 0 |
| `cuKernelGetFunction` | 601 | `0.009098` | 0 | 0 |
| `cudaGraphDestroy_v10000` | 3 | `0.007069` | 0 | 0 |
| `cudaEventRecord_v3020` | 2117 | `0.005978` | 0 | 0 |
| `cudaEventQuery_v3020` | 1644 | `0.005138` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 1393 | `0.004903` | 0 | 0 |
| `cudaStreamCreateWithFlags_v5000` | 268 | `0.004183` | 0 | 0 |
| `cuLaunchKernel` | 600 | `0.003782` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 3155 | `0.003781` | 0 | 0 |
| `cudaStreamCreateWithPriority_v5050` | 128 | `0.003218` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 1539 | `0.002652` | 0 | 0 |
| `cudaMemGetInfo_v3020` | 8 | `0.002493` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 219 | `0.001557` | 0 | 0 |
| `cudaStreamDestroy_v5050` | 268 | `0.001471` | 0 | 0 |
| `cuMemAddressReserve` | 194 | `0.001281` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 683 | `0.001127` | 0 | 0 |
| `cudaStreamEndCapture_v10000` | 3 | `0.000912` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000752` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `10.547570` | ranges=1 |
| kernel | 777179 | `7.666403` |  |
| runtime | 84880 | `8.054079` |  |
| memcpy | 45057 | `0.096675` | bytes=2684455872 |
| memset | 121 | `0.000200` |  |
| sync | 2255 | `5.947781` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 101545 | `1.184826` | 15.45% | 100965 | 795 |
| `graph_runtime_copy_cat_index` | 278077 | `1.247810` | 16.28% | 240665 | 1895 |
| `elementwise_graph_nodes` | 199878 | `0.742162` | 9.68% | 196977 | 1551 |
| `moe_marlin` | 44032 | `0.565877` | 7.38% | 43688 | 344 |
| `nccl_communication` | 11264 | `0.495947` | 6.47% | 11176 | 88 |
| `sampling_logits` | 44180 | `0.632412` | 8.25% | 43815 | 345 |
| `fp8_indexer` | 20992 | `0.254431` | 3.32% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.118376` | 1.54% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.128245` | 27.76% | 0 | 0 |
| `kv_compressor_cache_store` | 8212 | `0.067906` | 0.89% | 8128 | 64 |
| `unknown` | 47366 | `0.228409` | 2.98% | 46482 | 366 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 3653 | `5.954961` | 73.94% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.233787` | 15.32% | 0 | 0 |
| `kernel_launch_runtime` | 42865 | `0.331307` | 4.11% | 0 | 0 |
| `memcpy_runtime` | 36421 | `0.277963` | 3.45% | 0 | 0 |
| `allocation_runtime` | 23 | `0.014098` | 0.18% | 0 | 0 |
| `module_runtime` | 142 | `0.239892` | 2.98% | 0 | 0 |
| `other` | 1649 | `0.002070` | 0.03% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.081836` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 71828 | `0.675550` | 71247 | 561 |
| `_hc_split_pre_kernel` | 11008 | `0.428728` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.264967` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235312` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.232935` | 16510 | 130 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.224381` | 5588 | 44 |
| `_fp8_activation_quantize_kernel` | 35712 | `0.177419` | 35433 | 279 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.167974` | 16256 | 128 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.132186` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13824 | `0.129810` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.126140` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13824 | `0.117868` | 13716 | 108 |
| `_hc_post_kernel` | 11008 | `0.106158` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 24555 | `0.104117` | 24384 | 192 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097934` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.097915` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 35732 | `0.096869` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22144 | `0.096686` | 21971 | 173 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089651` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27648 | `0.087769` | 27432 | 216 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084725` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084559` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077236` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 32251 | `0.066950` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5632 | `0.059915` | 5588 | 44 |
| `_moe_route_fill_kernel` | 11008 | `0.059785` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5120 | `0.059353` | 5080 | 40 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11008 | `0.059097` | 10922 | 86 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5504 | `0.058646` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.058555` | 24384 | 192 |
| `_rms_norm_bf16_kernel` | 16640 | `0.058524` | 16510 | 130 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.057150` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16597 | `0.051436` | 16510 | 130 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050905` | 2667 | 21 |
| `_rotary_tail_kernel` | 10880 | `0.048560` | 10795 | 85 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046409` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20083 | `0.044802` | 19177 | 151 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044535` | 5461 | 43 |
| `_indexer_fp8_quantize_fold_kernel` | 2688 | `0.041667` | 2667 | 21 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 665 | `3.664583` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 257 | `2.280560` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.233787` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 41536 | `0.320015` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 36421 | `0.277963` | 0 | 0 |
| `cuModuleLoadData` | 142 | `0.239892` | 0 | 0 |
| `cudaMalloc_v3020` | 15 | `0.011847` | 0 | 0 |
| `cuLaunchKernelEx` | 1329 | `0.011292` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.003648` | 0 | 0 |
| `cudaEventQuery_v3020` | 940 | `0.003384` | 0 | 0 |
| `cudaHostAlloc_v3020` | 8 | `0.002251` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001352` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1117 | `0.001117` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000752` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000677` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000378` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 92 | `0.000213` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 234 | `0.000133` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000092` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000092` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000050` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000002` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.159318` | ranges=1 |
| kernel | 6141 | `4.256581` |  |
| runtime | 8201 | `4.058026` |  |
| memcpy | 150 | `0.002980` | bytes=2320630108 |
| memset | 21 | `0.000045` |  |
| sync | 637 | `3.770837` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 580 | `0.372760` | 8.76% | 0 | 0 |
| `graph_runtime_copy_cat_index` | 2125 | `0.344216` | 8.09% | 0 | 0 |
| `elementwise_graph_nodes` | 1860 | `0.252087` | 5.92% | 0 | 0 |
| `moe_marlin` | 344 | `0.265243` | 6.23% | 0 | 0 |
| `nccl_communication` | 88 | `0.152394` | 3.58% | 0 | 0 |
| `sampling_logits` | 365 | `0.450753` | 10.59% | 0 | 0 |
| `fp8_indexer` | 164 | `0.123032` | 2.89% | 0 | 0 |
| `prefill_sparse_attention` | 43 | `2.128245` | 50.00% | 0 | 0 |
| `kv_compressor_cache_store` | 84 | `0.039851` | 0.94% | 0 | 0 |
| `unknown` | 488 | `0.128000` | 3.01% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1255 | `3.773228` | 92.98% | 0 | 0 |
| `kernel_launch_runtime` | 6141 | `0.045528` | 1.12% | 0 | 0 |
| `memcpy_runtime` | 150 | `0.001586` | 0.04% | 0 | 0 |
| `allocation_runtime` | 7 | `0.008746` | 0.22% | 0 | 0 |
| `module_runtime` | 15 | `0.228220` | 5.62% | 0 | 0 |
| `other` | 633 | `0.000719` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.081836` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.358071` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 581 | `0.296207` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235312` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106570` | 0 | 0 |
| `_fp8_activation_quantize_kernel` | 279 | `0.101458` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.098018` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097934` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089651` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084559` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077236` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069839` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 21 | `0.065331` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.054334` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053496` | 0 | 0 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046409` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 43 | `0.039381` | 0 | 0 |
| `_indexer_fp8_quantize_fold_kernel` | 21 | `0.034171` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 44 | `0.030954` | 0 | 0 |
| `_rotary_tail_kernel` | 85 | `0.026226` | 0 | 0 |
| `_moe_route_fill_kernel` | 86 | `0.025396` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn` | 43 | `0.022259` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 43 | `0.019424` | 0 | 0 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 40 | `0.015076` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 299 | `0.014666` | 0 | 0 |
| `_rms_norm_bf16_kernel` | 130 | `0.014265` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 86 | `0.012646` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 171 | `0.011064` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 86 | `0.009982` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 216 | `0.008955` | 0 | 0 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 62 | `0.008540` | 0 | 0 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 21 | `0.007391` | 0 | 0 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 84 | `0.005999` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 103 | `0.005475` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_kernel(at::TensorIteratorBase &)::[lambda() (instance ...` | 43 | `0.005168` | 0 | 0 |
| `_moe_route_count_kernel` | 86 | `0.004331` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<fl...` | 62 | `0.002046` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 150 | `0.001879` | 0 | 0 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` | 67 | `0.001671` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c1...` | 1 | `0.001118` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `3.658892` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.228220` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.111256` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 4939 | `0.035738` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009790` | 0 | 0 |
| `cudaMalloc_v3020` | 7 | `0.008746` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 150 | `0.001586` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001352` | 0 | 0 |
| `cudaEventQuery_v3020` | 289 | `0.000715` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000372` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000350` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 205 | `0.000230` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000214` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000202` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 230 | `0.000131` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000092` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000090` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000050` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000002` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `4.010320` | ranges=1 |
| kernel | 738347 | `3.340242` |  |
| runtime | 11437 | `3.497263` |  |
| memcpy | 12594 | `0.020648` | bytes=336979592 |
| memset | 0 | `0.000000` |  |
| sync | 1548 | `2.176225` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `projection_gemm` | 100965 | `0.812065` | 24.31% | 100965 | 795 |
| `graph_runtime_copy_cat_index` | 243407 | `0.834792` | 24.99% | 240665 | 1895 |
| `elementwise_graph_nodes` | 197890 | `0.489347` | 14.65% | 196977 | 1551 |
| `moe_marlin` | 43688 | `0.300635` | 9.00% | 43688 | 344 |
| `nccl_communication` | 11176 | `0.343553` | 10.29% | 11176 | 88 |
| `sampling_logits` | 43815 | `0.181659` | 5.44% | 43815 | 345 |
| `fp8_indexer` | 20828 | `0.131399` | 3.93% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.118376` | 3.54% | 21590 | 170 |
| `kv_compressor_cache_store` | 8128 | `0.028055` | 0.84% | 8128 | 64 |
| `unknown` | 46860 | `0.100360` | 3.00% | 46482 | 366 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2305 | `2.180838` | 62.36% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.233787` | 35.28% | 0 | 0 |
| `kernel_launch_runtime` | 4033 | `0.033343` | 0.95% | 0 | 0 |
| `memcpy_runtime` | 3958 | `0.036858` | 1.05% | 0 | 0 |
| `module_runtime` | 126 | `0.011578` | 0.33% | 0 | 0 |
| `other` | 888 | `0.000860` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 71247 | `0.379344` | 71247 | 561 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.232935` | 16510 | 130 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.170046` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.167974` | 16256 | 128 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.166949` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.132186` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.126140` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.097915` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 21971 | `0.095792` | 21971 | 173 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 24384 | `0.093053` | 24384 | 192 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 35433 | `0.082203` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27432 | `0.078814` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 35433 | `0.075961` | 35433 | 279 |
| `_hc_split_pre_kernel` | 10922 | `0.070658` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13716 | `0.064372` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.058555` | 24384 | 192 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.057150` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050905` | 2667 | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16510 | `0.050527` | 16510 | 130 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 10922 | `0.046451` | 10922 | 86 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044535` | 5461 | 43 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5080 | `0.044277` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 16510 | `0.044260` | 16510 | 130 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 19933 | `0.043987` | 19177 | 151 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.040156` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 27431 | `0.036942` | 27305 | 215 |
| `_hc_post_kernel` | 10922 | `0.036319` | 10922 | 86 |
| `_moe_route_fill_kernel` | 10922 | `0.034389` | 10922 | 86 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10668 | `0.034106` | 10668 | 84 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.033685` | 5461 | 43 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at:...` | 5080 | `0.030483` | 5080 | 40 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` | 19304 | `0.030406` | 19304 | 152 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.029318` | 5461 | 43 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5588 | `0.028961` | 5588 | 44 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cub...` | 127 | `0.026493` | 127 | 1 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 5334 | `0.025465` | 5334 | 42 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2667 | `0.024234` | 2667 | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 10541 | `0.024217` | 10541 | 83 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::...` | 13716 | `0.023627` | 13716 | 108 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13716 | `0.023240` | 13716 | 108 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `2.169193` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.233787` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 3958 | `0.036858` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 3907 | `0.031852` | 0 | 0 |
| `cuModuleLoadData` | 126 | `0.011578` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 535 | `0.005119` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.003194` | 0 | 0 |
| `cudaEventQuery_v3020` | 634 | `0.002591` | 0 | 0 |
| `cuLaunchKernelEx` | 126 | `0.001491` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000858` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000742` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 4 | `0.000002` | 0 | 0 |

