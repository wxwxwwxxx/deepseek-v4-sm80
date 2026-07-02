# Post-SplitK Nsight Classification: nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`3.972283`, decode_envelope_s=`4.408601`
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
| kernel | 1013498 | `9.485660` |  |
| runtime | 610752 | `24.011412` |  |
| memcpy | 203294 | `8.441725` | bytes=171701612340 |
| memset | 219 | `0.000402` |  |
| sync | 116380 | `15.467546` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 472284 | `2.564930` | 27.04% | 267970 | 2110 |
| `elementwise_graph_nodes` | 213706 | `0.918163` | 9.68% | 202438 | 1594 |
| `fp8_activation_quant_poc` | 36549 | `0.180078` | 1.90% | 35433 | 279 |
| `fp8_indexer` | 21379 | `0.257168` | 2.71% | 20828 | 164 |
| `sparse_attention_decode` | 22100 | `0.121485` | 1.28% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.126340` | 22.42% | 0 | 0 |
| `kv_compressor_cache_store` | 8362 | `0.068657` | 0.72% | 8128 | 64 |
| `projection_gemm` | 103590 | `1.196603` | 12.61% | 100965 | 795 |
| `moe_marlin` | 67080 | `0.834696` | 8.80% | 43688 | 344 |
| `nccl_communication` | 11528 | `0.529739` | 5.58% | 11176 | 88 |
| `sampling_logits` | 45131 | `0.636800` | 6.71% | 43815 | 345 |
| `unknown` | 11746 | `0.051002` | 0.54% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 121463 | `15.558552` | 64.80% | 0 | 0 |
| `cuda_graph_launch_runtime` | 667 | `1.462884` | 6.09% | 0 | 0 |
| `kernel_launch_runtime` | 262680 | `2.323517` | 9.68% | 0 | 0 |
| `memcpy_runtime` | 195031 | `1.673797` | 6.97% | 0 | 0 |
| `allocation_runtime` | 1419 | `1.822888` | 7.59% | 0 | 0 |
| `module_runtime` | 74 | `0.232705` | 0.97% | 0 | 0 |
| `other` | 29418 | `0.937068` | 3.90% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.079931` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 85101 | `0.853184` | 82169 | 647 |
| `_hc_split_pre_kernel` | 11266 | `0.429619` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5633 | `0.280612` | 5461 | 43 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44072 | `0.279491` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30831 | `0.254308` | 29845 | 235 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.245109` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16770 | `0.244370` | 16510 | 130 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5764 | `0.243464` | 5588 | 44 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235361` | 0 | 0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21974 | `0.211117` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 50097 | `0.210117` | 27432 | 216 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 34314 | `0.197663` | 32893 | 259 |
| `_fp8_activation_quantize_kernel` | 36549 | `0.180078` | 35433 | 279 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128...` | 22446 | `0.156415` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 42269 | `0.154400` | 40894 | 322 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5547 | `0.147661` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22400 | `0.141186` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 14106 | `0.130502` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 11094 | `0.129606` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 14106 | `0.119479` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.111915` | 0 | 0 |
| `_hc_post_kernel` | 11266 | `0.107640` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5590 | `0.102913` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097745` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084741` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084413` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077081` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 11008 | `0.072754` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33396 | `0.069181` | 0 | 0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24683 | `0.061034` | 24384 | 192 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5764 | `0.060887` | 5588 | 44 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5200 | `0.060326` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 17030 | `0.060078` | 16510 | 130 |
| `_moe_route_fill_kernel` | 11266 | `0.060035` | 10922 | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11266 | `0.059870` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat...` | 22016 | `0.059737` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5633 | `0.059227` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16857 | `0.052559` | 16510 | 130 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::float8_copy_kernel_cuda(at::Ten...` | 11094 | `0.052552` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 112789 | `12.866380` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 267 | `2.670266` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 252903 | `2.256473` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 195031 | `1.673797` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.295511` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `0.942605` | 0 | 0 |
| `cudaMalloc_v3020` | 846 | `0.650045` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.278053` | 0 | 0 |
| `cuModuleLoadData` | 70 | `0.232702` | 0 | 0 |
| `cudaFree_v3020` | 346 | `0.229452` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.165432` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.123527` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.104446` | 0 | 0 |
| `cuMemCreate` | 147 | `0.095128` | 0 | 0 |
| `cuLaunchKernelEx` | 9005 | `0.062305` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.057694` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 1976 | `0.050523` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.047488` | 0 | 0 |
| `cuMemMap` | 194 | `0.044108` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.035824` | 0 | 0 |
| `cuMemRelease` | 389 | `0.034359` | 0 | 0 |
| `cudaGetDeviceProperties_v2_v12000` | 9 | `0.014898` | 0 | 0 |
| `cuKernelGetFunction` | 773 | `0.014259` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 17579 | `0.011446` | 0 | 0 |
| `cudaGraphDestroy_v10000` | 3 | `0.007519` | 0 | 0 |
| `cudaEventRecord_v3020` | 2117 | `0.006522` | 0 | 0 |
| `cudaEventQuery_v3020` | 1657 | `0.005070` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 1393 | `0.004782` | 0 | 0 |
| `cuLaunchKernel` | 772 | `0.004740` | 0 | 0 |
| `cudaStreamCreateWithPriority_v5050` | 128 | `0.004173` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 3140 | `0.003626` | 0 | 0 |
| `cudaStreamCreateWithFlags_v5000` | 268 | `0.003615` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 1539 | `0.002458` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 219 | `0.001807` | 0 | 0 |
| `cudaStreamDestroy_v5050` | 268 | `0.001417` | 0 | 0 |
| `cuMemAddressReserve` | 194 | `0.001218` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 683 | `0.001102` | 0 | 0 |
| `cudaStreamEndCapture_v10000` | 3 | `0.001035` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000865` | 0 | 0 |
| `cuMemAddressFree` | 194 | `0.000694` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `10.939888` | ranges=1 |
| kernel | 818585 | `8.138390` |  |
| runtime | 95210 | `8.550450` |  |
| memcpy | 46454 | `0.101192` | bytes=2689042604 |
| memset | 121 | `0.000216` |  |
| sync | 2649 | `6.312433` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 310296 | `1.556377` | 19.12% | 267970 | 2110 |
| `elementwise_graph_nodes` | 209192 | `0.893292` | 10.98% | 202438 | 1594 |
| `fp8_activation_quant_poc` | 35712 | `0.177559` | 2.18% | 35433 | 279 |
| `fp8_indexer` | 20992 | `0.254109` | 3.12% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.118408` | 1.45% | 21590 | 170 |
| `prefill_sparse_attention` | 43 | `2.126340` | 26.13% | 0 | 0 |
| `kv_compressor_cache_store` | 8212 | `0.067965` | 0.84% | 8128 | 64 |
| `projection_gemm` | 101545 | `1.177360` | 14.47% | 100965 | 795 |
| `moe_marlin` | 44032 | `0.582140` | 7.15% | 43688 | 344 |
| `nccl_communication` | 11264 | `0.502369` | 6.17% | 11176 | 88 |
| `sampling_logits` | 44180 | `0.632125` | 7.77% | 43815 | 345 |
| `unknown` | 11527 | `0.050347` | 0.62% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 4047 | `6.319897` | 73.91% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.295511` | 15.15% | 0 | 0 |
| `kernel_launch_runtime` | 51505 | `0.361388` | 4.23% | 0 | 0 |
| `memcpy_runtime` | 37818 | `0.293341` | 3.43% | 0 | 0 |
| `allocation_runtime` | 23 | `0.051711` | 0.60% | 0 | 0 |
| `module_runtime` | 15 | `0.226584` | 2.65% | 0 | 0 |
| `other` | 1675 | `0.002018` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.079931` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 82836 | `0.830264` | 82169 | 647 |
| `_hc_split_pre_kernel` | 11008 | `0.427616` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.265935` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30059 | `0.242948` | 29845 | 235 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.239976` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235361` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.230942` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.208334` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 33152 | `0.187380` | 32893 | 259 |
| `_fp8_activation_quantize_kernel` | 35712 | `0.177559` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 41236 | `0.148108` | 40894 | 322 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.145324` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13824 | `0.129778` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.127339` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13824 | `0.117366` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.111915` | 0 | 0 |
| `_hc_post_kernel` | 11008 | `0.106783` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.101020` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097745` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27648 | `0.087733` | 27432 | 216 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084741` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084413` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077081` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33394 | `0.069177` | 0 | 0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.059981` | 24384 | 192 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5632 | `0.059859` | 5588 | 44 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5120 | `0.059268` | 5080 | 40 |
| `_moe_route_fill_kernel` | 11008 | `0.058983` | 10922 | 86 |
| `_rms_norm_bf16_kernel` | 16640 | `0.058678` | 16510 | 130 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5504 | `0.058632` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 11008 | `0.058327` | 10922 | 86 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16597 | `0.051442` | 16510 | 130 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050961` | 2667 | 21 |
| `_rotary_tail_kernel` | 10880 | `0.048435` | 10795 | 85 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20972 | `0.047417` | 19177 | 151 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046409` | 0 | 0 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044649` | 5461 | 43 |
| `_indexer_fp8_quantize_fold_kernel` | 2688 | `0.041712` | 2667 | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10752 | `0.040187` | 10668 | 84 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | `3.640686` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 257 | `2.669363` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.295511` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 50303 | `0.351693` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 37818 | `0.293341` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.226584` | 0 | 0 |
| `cudaMalloc_v3020` | 15 | `0.049248` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009695` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.003532` | 0 | 0 |
| `cudaEventQuery_v3020` | 953 | `0.003342` | 0 | 0 |
| `cudaHostAlloc_v3020` | 8 | `0.002463` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001440` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1117 | `0.001023` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000865` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000707` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000383` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 92 | `0.000197` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 260 | `0.000152` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000091` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000087` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000046` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000002` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.174367` | ranges=1 |
| kernel | 6399 | `4.259719` |  |
| runtime | 8494 | `4.068291` |  |
| memcpy | 150 | `0.002977` | bytes=2320630108 |
| memset | 21 | `0.000043` |  |
| sync | 648 | `3.744435` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 2340 | `0.346221` | 8.13% | 0 | 0 |
| `elementwise_graph_nodes` | 1903 | `0.253084` | 5.94% | 0 | 0 |
| `fp8_activation_quant_poc` | 279 | `0.101430` | 2.38% | 0 | 0 |
| `fp8_indexer` | 164 | `0.123081` | 2.89% | 0 | 0 |
| `prefill_sparse_attention` | 43 | `2.126340` | 49.92% | 0 | 0 |
| `kv_compressor_cache_store` | 84 | `0.039855` | 0.94% | 0 | 0 |
| `projection_gemm` | 580 | `0.372280` | 8.74% | 0 | 0 |
| `moe_marlin` | 344 | `0.265286` | 6.23% | 0 | 0 |
| `nccl_communication` | 88 | `0.155698` | 3.66% | 0 | 0 |
| `sampling_logits` | 365 | `0.449929` | 10.56% | 0 | 0 |
| `unknown` | 209 | `0.026513` | 0.62% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1266 | `3.747008` | 92.10% | 0 | 0 |
| `kernel_launch_runtime` | 6399 | `0.046011` | 1.13% | 0 | 0 |
| `memcpy_runtime` | 150 | `0.001609` | 0.04% | 0 | 0 |
| `allocation_runtime` | 7 | `0.046356` | 1.14% | 0 | 0 |
| `module_runtime` | 15 | `0.226584` | 5.57% | 0 | 0 |
| `other` | 657 | `0.000724` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.079931` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.357190` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 667 | `0.297035` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235361` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.111915` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106524` | 0 | 0 |
| `_fp8_activation_quantize_kernel` | 279 | `0.101430` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097745` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.097361` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084413` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077081` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069898` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 21 | `0.065364` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.058301` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053516` | 0 | 0 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046409` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 43 | `0.039384` | 0 | 0 |
| `_indexer_fp8_quantize_fold_kernel` | 21 | `0.034174` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 44 | `0.030933` | 0 | 0 |
| `_rotary_tail_kernel` | 85 | `0.026198` | 0 | 0 |
| `_moe_route_fill_kernel` | 86 | `0.025393` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 43 | `0.019420` | 0 | 0 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 40 | `0.015080` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 342 | `0.015073` | 0 | 0 |
| `_rms_norm_bf16_kernel` | 130 | `0.014255` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 86 | `0.012640` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 214 | `0.012159` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 86 | `0.009982` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 216 | `0.008947` | 0 | 0 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 62 | `0.008547` | 0 | 0 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 21 | `0.007399` | 0 | 0 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 84 | `0.005986` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 103 | `0.005478` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_kernel(at::TensorIteratorBase &)::[lambda() (instance ...` | 43 | `0.005139` | 0 | 0 |
| `_moe_route_count_kernel` | 86 | `0.004329` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<fl...` | 62 | `0.002038` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 150 | `0.001875` | 0 | 0 |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` | 67 | `0.001679` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 259 | `0.001675` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c1...` | 1 | `0.001117` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `3.632671` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.226584` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.111055` | 0 | 0 |
| `cudaMalloc_v3020` | 7 | `0.046356` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 5197 | `0.036316` | 0 | 0 |
| `cuLaunchKernelEx` | 1202 | `0.009695` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 150 | `0.001609` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001440` | 0 | 0 |
| `cudaEventQuery_v3020` | 300 | `0.000756` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000437` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000377` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000235` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 205 | `0.000204` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000185` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 254 | `0.000149` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000088` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000087` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000046` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000002` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `4.408601` | ranges=1 |
| kernel | 779429 | `3.808678` |  |
| runtime | 21395 | `3.998206` |  |
| memcpy | 13980 | `0.023173` | bytes=341557584 |
| memset | 0 | `0.000000` |  |
| sync | 1928 | `2.567235` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `graph_runtime_copy_cat_index` | 275374 | `1.141006` | 29.96% | 267970 | 2110 |
| `elementwise_graph_nodes` | 207131 | `0.639409` | 16.79% | 202438 | 1594 |
| `fp8_activation_quant_poc` | 35433 | `0.076129` | 2.00% | 35433 | 279 |
| `fp8_indexer` | 20828 | `0.131028` | 3.44% | 20828 | 164 |
| `sparse_attention_decode` | 21590 | `0.118408` | 3.11% | 21590 | 170 |
| `kv_compressor_cache_store` | 8128 | `0.028110` | 0.74% | 8128 | 64 |
| `projection_gemm` | 100965 | `0.805080` | 21.14% | 100965 | 795 |
| `moe_marlin` | 43688 | `0.316854` | 8.32% | 43688 | 344 |
| `nccl_communication` | 11176 | `0.346671` | 9.10% | 11176 | 88 |
| `sampling_logits` | 43815 | `0.182195` | 4.78% | 43815 | 345 |
| `unknown` | 11301 | `0.023788` | 0.62% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2685 | `2.571947` | 64.33% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `1.295511` | 32.40% | 0 | 0 |
| `kernel_launch_runtime` | 12349 | `0.080818` | 2.02% | 0 | 0 |
| `memcpy_runtime` | 5344 | `0.049136` | 1.23% | 0 | 0 |
| `other` | 890 | `0.000793` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 82169 | `0.533229` | 82169 | 647 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.239976` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 29845 | `0.230789` | 29845 | 235 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.208334` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 32893 | `0.185705` | 32893 | 259 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.172641` | 5588 | 44 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.168574` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.145324` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 40894 | `0.133035` | 40894 | 322 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.127339` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.101020` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27432 | `0.078786` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 35433 | `0.076129` | 35433 | 279 |
| `_hc_split_pre_kernel` | 10922 | `0.070426` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13716 | `0.063851` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.059981` | 24384 | 192 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.050961` | 2667 | 21 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16510 | `0.050536` | 16510 | 130 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20815 | `0.046576` | 19177 | 151 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 10922 | `0.045686` | 10922 | 86 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044649` | 5461 | 43 |
| `_rms_norm_bf16_kernel` | 16510 | `0.044423` | 16510 | 130 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 5080 | `0.044188` | 5080 | 40 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039956` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28439 | `0.038837` | 27305 | 215 |
| `_hc_post_kernel` | 10922 | `0.036885` | 10922 | 86 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)6...` | 10668 | `0.034201` | 10668 | 84 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.033802` | 5461 | 43 |
| `_moe_route_fill_kernel` | 10922 | `0.033590` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` | 19304 | `0.032150` | 19304 | 152 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at:...` | 5080 | `0.030203` | 5080 | 40 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.029168` | 5461 | 43 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 5588 | `0.028925` | 5588 | 44 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cub...` | 127 | `0.026720` | 127 | 1 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, con...` | 5334 | `0.025524` | 5334 | 42 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::...` | 13716 | `0.024858` | 13716 | 108 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::Ten...` | 10541 | `0.024384` | 10541 | 83 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2667 | `0.024182` | 2667 | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13716 | `0.023254` | 13716 | 108 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 13843 | `0.022916` | 13843 | 109 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `2.558201` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `1.295511` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 12349 | `0.080818` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 5344 | `0.049136` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 913 | `0.007393` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.002990` | 0 | 0 |
| `cudaEventQuery_v3020` | 636 | `0.002510` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000853` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000790` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 6 | `0.000004` | 0 | 0 |

