# Post-SplitK Nsight Classification: nsys_target0753_fp8_indexer_node_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`6.102604`, decode_envelope_s=`6.538964`
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
| kernel | 1539094 | `11.886904` |  |
| runtime | 650678 | `27.238435` |  |
| memcpy | 203294 | `9.118901` | bytes=171699775412 |
| memset | 219 | `0.000386` |  |
| sync | 116394 | `17.662468` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.129782` | 17.92% | 0 | 0 |
| `decode_splitk_gather_split_combine` | 22100 | `0.121039` | 1.02% | 21590 | 170 |
| `indexer_logits_topk_cache` | 29741 | `0.324925` | 2.73% | 28956 | 228 |
| `runtime_copy_cat_index_kernels` | 668939 | `3.199357` | 26.91% | 458724 | 3612 |
| `fp8_projection_gemm` | 13910 | `1.206787` | 10.15% | 13589 | 107 |
| `moe_marlin_route` | 67080 | `0.835499` | 7.03% | 43688 | 344 |
| `hc_rmsnorm_logits_sampling` | 45131 | `0.637334` | 5.36% | 43815 | 345 |
| `dense_linear_other` | 89680 | `1.011540` | 8.51% | 87376 | 688 |
| `nccl_communication` | 11528 | `0.507735` | 4.27% | 11176 | 88 |
| `elementwise_math_other` | 579196 | `1.862044` | 15.66% | 556768 | 4384 |
| `other` | 11746 | `0.050861` | 0.43% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 121469 | `17.749036` | 65.16% | 0 | 0 |
| `cuda_graph_launch_runtime` | 667 | `2.410998` | 8.85% | 0 | 0 |
| `kernel_launch_runtime` | 290664 | `2.449262` | 8.99% | 0 | 0 |
| `memcpy_runtime` | 195031 | `1.629564` | 5.98% | 0 | 0 |
| `allocation_runtime` | 1351 | `1.851127` | 6.80% | 0 | 0 |
| `module_runtime` | 76 | `0.247994` | 0.91% | 0 | 0 |
| `other` | 41420 | `0.900454` | 3.31% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.083352` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 172109 | `1.252517` | 166624 | 1312 |
| `_quantized_linear_fp8_kernel` | 13910 | `1.206787` | 13589 | 107 |
| `_hc_split_pre_kernel` | 11266 | `0.430158` | 10922 | 86 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44072 | `0.280077` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5633 | `0.264226` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30831 | `0.253443` | 29845 | 235 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.245084` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16770 | `0.241822` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 78818 | `0.240304` | 76327 | 601 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5764 | `0.236911` | 5588 | 44 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235353` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 50097 | `0.212298` | 27432 | 216 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21974 | `0.211339` | 21717 | 171 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 34314 | `0.197862` | 32893 | 259 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 84364 | `0.196924` | 81788 | 644 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128...` | 22446 | `0.156482` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 50827 | `0.151207` | 49276 | 388 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5547 | `0.149231` | 5461 | 43 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 36549 | `0.144549` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22400 | `0.141012` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 42182 | `0.140740` | 40894 | 322 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 14106 | `0.130582` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 14106 | `0.119773` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.112093` | 0 | 0 |
| `_hc_post_kernel` | 11266 | `0.107728` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5590 | `0.103523` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097789` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 36549 | `0.096489` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::<unnamed>::pow_tensor_tensor_kernel(at...` | 36549 | `0.096288` | 35433 | 279 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 112789 | `13.789974` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 267 | `3.938868` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 282198 | `2.393898` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `2.145076` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 195031 | `1.629564` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `1.070476` | 0 | 0 |
| `cudaMalloc_v3020` | 781 | `0.555223` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.300801` | 0 | 0 |
| `cuModuleLoadData` | 72 | `0.247991` | 0 | 0 |
| `cudaFree_v3020` | 343 | `0.224657` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.220032` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.158750` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.106584` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.063672` | 0 | 0 |
| `cuMemCreate` | 147 | `0.062035` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.058613` | 0 | 0 |
| `cuLaunchKernelEx` | 7694 | `0.050729` | 0 | 0 |
| `cuMemRelease` | 389 | `0.049852` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.032778` | 0 | 0 |
| `cuMemMap` | 194 | `0.026197` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 29618 | `0.016959` | 0 | 0 |
| `cudaGetDeviceProperties_v2_v12000` | 9 | `0.014213` | 0 | 0 |
| `cudaGraphDestroy_v10000` | 3 | `0.012512` | 0 | 0 |
| `cuKernelGetFunction` | 773 | `0.012352` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2004 | `0.009638` | 0 | 0 |
| `cudaEventRecord_v3020` | 2117 | `0.005622` | 0 | 0 |
| `cudaStreamCreateWithFlags_v5000` | 268 | `0.004945` | 0 | 0 |
| `cudaEventQuery_v3020` | 1671 | `0.004640` | 0 | 0 |
| `cuLaunchKernel` | 772 | `0.004634` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 1393 | `0.004494` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `13.288072` | ranges=1 |
| kernel | 1332784 | `10.488728` |  |
| runtime | 99800 | `10.872903` |  |
| memcpy | 46454 | `0.101410` | bytes=2687205676 |
| memset | 121 | `0.000212` |  |
| sync | 2663 | `7.794698` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.129782` | 20.31% | 0 | 0 |
| `decode_splitk_gather_split_combine` | 21590 | `0.117957` | 1.12% | 21590 | 170 |
| `indexer_logits_topk_cache` | 29204 | `0.321193` | 3.06% | 28956 | 228 |
| `runtime_copy_cat_index_kernels` | 502980 | `2.181877` | 20.80% | 458724 | 3612 |
| `fp8_projection_gemm` | 13589 | `1.172973` | 11.18% | 13589 | 107 |
| `moe_marlin_route` | 44032 | `0.582977` | 5.56% | 43688 | 344 |
| `hc_rmsnorm_logits_sampling` | 44180 | `0.632683` | 6.03% | 43815 | 345 |
| `dense_linear_other` | 87956 | `0.996981` | 9.51% | 87376 | 688 |
| `nccl_communication` | 11264 | `0.489769` | 4.67% | 11176 | 88 |
| `elementwise_math_other` | 566419 | `1.812329` | 17.28% | 556768 | 4384 |
| `other` | 11527 | `0.050206` | 0.48% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 4061 | `7.801714` | 71.75% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `2.145076` | 19.73% | 0 | 0 |
| `kernel_launch_runtime` | 56053 | `0.379954` | 3.49% | 0 | 0 |
| `memcpy_runtime` | 37818 | `0.288922` | 2.66% | 0 | 0 |
| `allocation_runtime` | 23 | `0.013563` | 0.12% | 0 | 0 |
| `module_runtime` | 15 | `0.241702` | 2.22% | 0 | 0 |
| `other` | 1703 | `0.001971` | 0.02% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.083352` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 168063 | `1.224399` | 166624 | 1312 |
| `_quantized_linear_fp8_kernel` | 13589 | `1.172973` | 13589 | 107 |
| `_hc_split_pre_kernel` | 11008 | `0.428159` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.258132` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 30166 | `0.245675` | 29845 | 235 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.237466` | 16510 | 130 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235353` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 77055 | `0.233480` | 76327 | 601 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.226265` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.208541` | 21717 | 171 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 82432 | `0.191734` | 81788 | 644 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 33366 | `0.190256` | 32893 | 259 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 49705 | `0.147193` | 49276 | 388 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.146889` | 5461 | 43 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 35712 | `0.139541` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 41216 | `0.137062` | 40894 | 322 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 13824 | `0.129858` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13824 | `0.117600` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.112093` | 0 | 0 |
| `_hc_post_kernel` | 11008 | `0.106887` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.101653` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097789` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 35712 | `0.094395` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (ins...` | 35712 | `0.092789` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::<unnamed>::pow_tensor_tensor_kernel(at...` | 35712 | `0.092343` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27648 | `0.088746` | 27432 | 216 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084794` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084506` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077166` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | `3.935974` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 1046 | `3.856476` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `2.145076` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 55130 | `0.373250` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 37818 | `0.288922` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.241702` | 0 | 0 |
| `cudaMalloc_v3020` | 15 | `0.011052` | 0 | 0 |
| `cuLaunchKernelEx` | 923 | `0.006704` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.003425` | 0 | 0 |
| `cudaEventQuery_v3020` | 967 | `0.003191` | 0 | 0 |
| `cudaHostAlloc_v3020` | 8 | `0.002511` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001220` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1117 | `0.001095` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000819` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000618` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000357` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 92 | `0.000173` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 288 | `0.000138` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000080` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000075` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000039` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000005` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.408744` | ranges=1 |
| kernel | 10947 | `4.512234` |  |
| runtime | 13084 | `4.284783` |  |
| memcpy | 150 | `0.002962` | bytes=2320630108 |
| memset | 21 | `0.000043` |  |
| sync | 662 | `3.962965` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.129782` | 47.20% | 0 | 0 |
| `indexer_logits_topk_cache` | 248 | `0.162972` | 3.61% | 0 | 0 |
| `runtime_copy_cat_index_kernels` | 4270 | `0.494639` | 10.96% | 0 | 0 |
| `moe_marlin_route` | 344 | `0.263697` | 5.84% | 0 | 0 |
| `hc_rmsnorm_logits_sampling` | 365 | `0.450105` | 9.98% | 0 | 0 |
| `dense_linear_other` | 580 | `0.372682` | 8.26% | 0 | 0 |
| `nccl_communication` | 88 | `0.158514` | 3.51% | 0 | 0 |
| `elementwise_math_other` | 4800 | `0.453281` | 10.05% | 0 | 0 |
| `other` | 209 | `0.026561` | 0.59% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1280 | `3.965071` | 92.54% | 0 | 0 |
| `kernel_launch_runtime` | 10947 | `0.067747` | 1.58% | 0 | 0 |
| `memcpy_runtime` | 150 | `0.001492` | 0.03% | 0 | 0 |
| `allocation_runtime` | 7 | `0.008189` | 0.19% | 0 | 0 |
| `module_runtime` | 15 | `0.241702` | 5.64% | 0 | 0 |
| `other` | 685 | `0.000582` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.083352` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 1439 | `0.387524` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.357348` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.235353` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.112093` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106535` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.098056` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097789` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084506` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077166` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069924` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 21 | `0.065352` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.060409` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053530` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 644 | `0.047431` | 0 | 0 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046430` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 429 | `0.044892` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 728 | `0.044524` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 322 | `0.043711` | 0 | 0 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 43 | `0.039405` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 279 | `0.036942` | 0 | 0 |
| `_indexer_fp8_quantize_fold_kernel` | 21 | `0.034193` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 279 | `0.030943` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native...` | 44 | `0.030910` | 0 | 0 |
| `_rotary_tail_kernel` | 85 | `0.026236` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (ins...` | 279 | `0.025667` | 0 | 0 |
| `_moe_route_fill_kernel` | 86 | `0.025417` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 43 | `0.019426` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 321 | `0.015777` | 0 | 0 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` | 40 | `0.015092` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `3.848678` | 0 | 0 |
| `cuModuleLoadData` | 15 | `0.241702` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.113666` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 10024 | `0.061043` | 0 | 0 |
| `cudaMalloc_v3020` | 7 | `0.008189` | 0 | 0 |
| `cuLaunchKernelEx` | 923 | `0.006704` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 150 | `0.001492` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001220` | 0 | 0 |
| `cudaEventQuery_v3020` | 314 | `0.000634` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000352` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000281` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 205 | `0.000177` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000163` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000149` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 282 | `0.000136` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000078` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000075` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000039` | 0 | 0 |
| `cuKernelGetFunction` | 1 | `0.000005` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `6.538964` | ranges=1 |
| kernel | 1289080 | `5.905398` |  |
| runtime | 21395 | `6.111005` |  |
| memcpy | 13980 | `0.023102` | bytes=339720656 |
| memset | 0 | `0.000000` |  |
| sync | 1928 | `3.830940` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_splitk_gather_split_combine` | 21590 | `0.117957` | 2.00% | 21590 | 170 |
| `indexer_logits_topk_cache` | 28956 | `0.158221` | 2.68% | 28956 | 228 |
| `runtime_copy_cat_index_kernels` | 466128 | `1.616983` | 27.38% | 458724 | 3612 |
| `fp8_projection_gemm` | 13589 | `1.172973` | 19.86% | 13589 | 107 |
| `moe_marlin_route` | 43688 | `0.319280` | 5.41% | 43688 | 344 |
| `hc_rmsnorm_logits_sampling` | 43815 | `0.182578` | 3.09% | 43815 | 345 |
| `dense_linear_other` | 87376 | `0.624299` | 10.57% | 87376 | 688 |
| `nccl_communication` | 11176 | `0.331255` | 5.61% | 11176 | 88 |
| `elementwise_math_other` | 561461 | `1.358253` | 23.00% | 556768 | 4384 |
| `other` | 11301 | `0.023600` | 0.40% | 11049 | 87 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2685 | `3.835686` | 62.77% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `2.145076` | 35.10% | 0 | 0 |
| `kernel_launch_runtime` | 12349 | `0.079440` | 1.30% | 0 | 0 |
| `memcpy_runtime` | 5344 | `0.049909` | 0.82% | 0 | 0 |
| `other` | 890 | `0.000893` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `_quantized_linear_fp8_kernel` | 13589 | `1.172973` | 13589 | 107 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 166624 | `0.836875` | 166624 | 1312 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.237466` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 29845 | `0.229898` | 29845 | 235 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 21717 | `0.208541` | 21717 | 171 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 76327 | `0.188956` | 76327 | 601 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 32893 | `0.186341` | 32893 | 259 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.165856` | 5588 | 44 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.160076` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.146889` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 81788 | `0.144303` | 81788 | 644 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::MaxNanFunctor<...` | 35433 | `0.108598` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 49276 | `0.102301` | 49276 | 388 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 5461 | `0.101653` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 40894 | `0.093350` | 40894 | 322 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::<unnamed>::pow_tensor_tensor_kernel(at...` | 35433 | `0.089996` | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 27432 | `0.079779` | 27432 | 216 |
| `_hc_split_pre_kernel` | 10922 | `0.070811` | 10922 | 86 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::log2_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::o...` | 35433 | `0.069084` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (ins...` | 35433 | `0.067122` | 35433 | 279 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 13716 | `0.064070` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat...` | 24384 | `0.059849` | 24384 | 192 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 35433 | `0.057452` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFun...` | 35433 | `0.054813` | 35433 | 279 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::ceil_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::o...` | 35433 | `0.053915` | 35433 | 279 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bo...` | 16510 | `0.050499` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<double>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 35433 | `0.047592` | 35433 | 279 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 20815 | `0.046418` | 19177 | 151 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 10922 | `0.045817` | 10922 | 86 |
| `_rms_norm_bf16_kernel` | 16510 | `0.044449` | 16510 | 130 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `3.822208` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `2.145076` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 12349 | `0.079440` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 5344 | `0.049909` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 913 | `0.007141` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.003048` | 0 | 0 |
| `cudaEventQuery_v3020` | 636 | `0.002481` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000891` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000807` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 6 | `0.000003` | 0 | 0 |

