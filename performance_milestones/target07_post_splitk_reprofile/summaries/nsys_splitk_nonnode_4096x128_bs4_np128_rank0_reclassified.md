# Post-SplitK Nsight Classification: nsys_target07395_splitk_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`6.373556`, decode_envelope_s=`6.786952`
- Tables:
  - `CUPTI_ACTIVITY_KIND_KERNEL`: `True`
  - `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`: `True`
  - `CUPTI_ACTIVITY_KIND_RUNTIME`: `True`
  - `CUPTI_ACTIVITY_KIND_MEMCPY`: `True`
  - `CUPTI_ACTIVITY_KIND_MEMSET`: `True`
  - `NVTX_EVENTS`: `True`

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | n/a | `n/a` | ranges=None |
| kernel | 325787 | `13.102317` |  |
| runtime | 764231 | `42.090038` |  |
| memcpy | 232622 | `10.025042` | bytes=173761580864 |
| memset | 340 | `0.000572` |  |
| sync | 119066 | `33.222481` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 86 | `4.216710` | 32.18% | 0 | 0 |
| `decode_splitk_gather_split_combine` | 510 | `0.003079` | 0.02% | 0 | 0 |
| `indexer_logits_topk_cache` | 928 | `1.973256` | 15.06% | 0 | 0 |
| `runtime_copy_cat_index_kernels` | 259322 | `2.791881` | 21.31% | 0 | 0 |
| `fp8_projection_gemm` | 321 | `0.033823` | 0.26% | 0 | 0 |
| `moe_marlin_route` | 23736 | `0.777138` | 5.93% | 0 | 0 |
| `hc_rmsnorm_logits_sampling` | 1681 | `0.898600` | 6.86% | 0 | 0 |
| `dense_linear_other` | 2884 | `0.753415` | 5.75% | 0 | 0 |
| `nccl_communication` | 440 | `0.327358` | 2.50% | 0 | 0 |
| `elementwise_math_other` | 34704 | `1.272259` | 9.71% | 0 | 0 |
| `other` | 1175 | `0.054799` | 0.42% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 125534 | `33.318991` | 79.16% | 0 | 0 |
| `cuda_graph_launch_runtime` | 794 | `0.992698` | 2.36% | 0 | 0 |
| `kernel_launch_runtime` | 357721 | `2.884959` | 6.85% | 0 | 0 |
| `memcpy_runtime` | 233100 | `1.960293` | 4.66% | 0 | 0 |
| `allocation_runtime` | 1345 | `1.858687` | 4.42% | 0 | 0 |
| `module_runtime` | 76 | `0.229342` | 0.54% | 0 | 0 |
| `other` | 45661 | `0.845067` | 2.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 82 | `4.131952` | 0 | 0 |
| `_indexer_bf16_logits_kernel` | 42 | `1.844054` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 7029 | `0.828090` | 0 | 0 |
| `_hc_split_pre_kernel` | 430 | `0.710635` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 6514 | `0.594266` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 172 | `0.466816` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 2608 | `0.308618` | 0 | 0 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 44112 | `0.280153` | 0 | 0 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, in...` | 22016 | `0.245014` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 424 | `0.222103` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 498 | `0.213821` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 215 | `0.206927` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 174 | `0.194263` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 86 | `0.167574` | 0 | 0 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128...` | 22446 | `0.156503` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 300 | `0.152713` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22784 | `0.141952` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 22881 | `0.141120` | 0 | 0 |
| `_hc_post_kernel` | 430 | `0.140483` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 66790 | `0.126801` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 113834 | `21.199010` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 525 | `12.089418` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 348500 | `2.826427` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 233100 | `1.960293` | 0 | 0 |
| `cudaHostAlloc_v3020` | 29 | `1.041290` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 254 | `0.703574` | 0 | 0 |
| `cudaMalloc_v3020` | 778 | `0.565939` | 0 | 0 |
| `cuMemExportToShareableHandle` | 48 | `0.264217` | 0 | 0 |
| `cudaFree_v3020` | 340 | `0.250998` | 0 | 0 |
| `cuModuleLoadData` | 72 | `0.229340` | 0 | 0 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | `0.202664` | 0 | 0 |
| `cuMemSetAccess` | 194 | `0.150926` | 0 | 0 |
| `cuMemImportFromShareableHandle` | 48 | `0.108613` | 0 | 0 |
| `cudaGraphExecDestroy_v10000` | 3 | `0.071838` | 0 | 0 |
| `cuMemCreate` | 147 | `0.061281` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2308 | `0.059362` | 0 | 0 |
| `cuLibraryLoadData` | 15 | `0.058402` | 0 | 0 |
| `cuLaunchKernelEx` | 8449 | `0.054437` | 0 | 0 |
| `cuMemMap` | 194 | `0.033671` | 0 | 0 |
| `cuMemUnmap` | 194 | `0.023054` | 0 | 0 |

## repeat

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `13.852461` | ranges=1 |
| kernel | 57922 | `5.830580` |  |
| runtime | 101646 | `12.365491` |  |
| memcpy | 37839 | `0.073992` | bytes=2374492380 |
| memset | 121 | `0.000196` |  |
| sync | 2668 | `11.389396` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.104176` | 36.09% | 0 | 0 |
| `indexer_logits_topk_cache` | 227 | `0.984563` | 16.89% | 0 | 0 |
| `runtime_copy_cat_index_kernels` | 45495 | `0.876127` | 15.03% | 0 | 0 |
| `moe_marlin_route` | 344 | `0.262581` | 4.50% | 0 | 0 |
| `hc_rmsnorm_logits_sampling` | 365 | `0.445291` | 7.64% | 0 | 0 |
| `dense_linear_other` | 580 | `0.368200` | 6.31% | 0 | 0 |
| `nccl_communication` | 88 | `0.153150` | 2.63% | 0 | 0 |
| `elementwise_math_other` | 10302 | `0.609444` | 10.45% | 0 | 0 |
| `other` | 478 | `0.027048` | 0.46% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 4063 | `11.396512` | 92.16% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `0.326561` | 2.64% | 0 | 0 |
| `kernel_launch_runtime` | 57922 | `0.348979` | 2.82% | 0 | 0 |
| `memcpy_runtime` | 37839 | `0.291608` | 2.36% | 0 | 0 |
| `other` | 1695 | `0.001830` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.065840` | 0 | 0 |
| `_indexer_bf16_logits_kernel` | 21 | `0.921902` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 1481 | `0.398780` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.352719` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 1649 | `0.289401` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.232961` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 631 | `0.152644` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.110714` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106565` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.098370` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.096712` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.083572` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076092` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069736` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 33394 | `0.057534` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.054741` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 770 | `0.054198` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053505` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 644 | `0.047402` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 429 | `0.044605` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | `6.068325` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 1046 | `5.318471` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 57020 | `0.342220` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `0.326561` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 37839 | `0.291608` | 0 | 0 |
| `cuLaunchKernelEx` | 902 | `0.006759` | 0 | 0 |
| `cudaEventQuery_v3020` | 972 | `0.003431` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 865 | `0.003329` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001558` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 1102 | `0.000949` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 128 | `0.000778` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 121 | `0.000573` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 265 | `0.000349` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 89 | `0.000191` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 296 | `0.000175` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000089` | 0 | 0 |
| `cudaEventDestroy_v3020` | 89 | `0.000080` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000045` | 0 | 0 |

## repeat_prefill_forward

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `5.805964` | ranges=1 |
| kernel | 12816 | `5.742368` |  |
| runtime | 14959 | `5.509234` |  |
| memcpy | 171 | `0.003013` | bytes=2342650204 |
| memset | 21 | `0.000044` |  |
| sync | 668 | `5.424735` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_prefill_sparse_attention` | 43 | `2.104176` | 36.64% | 0 | 0 |
| `indexer_logits_topk_cache` | 227 | `0.984563` | 17.15% | 0 | 0 |
| `runtime_copy_cat_index_kernels` | 5509 | `0.799842` | 13.93% | 0 | 0 |
| `moe_marlin_route` | 344 | `0.262581` | 4.57% | 0 | 0 |
| `hc_rmsnorm_logits_sampling` | 365 | `0.445291` | 7.75% | 0 | 0 |
| `dense_linear_other` | 580 | `0.368200` | 6.41% | 0 | 0 |
| `nccl_communication` | 88 | `0.153150` | 2.67% | 0 | 0 |
| `elementwise_math_other` | 5451 | `0.598137` | 10.42% | 0 | 0 |
| `other` | 209 | `0.026428` | 0.46% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 1285 | `5.427402` | 98.51% | 0 | 0 |
| `kernel_launch_runtime` | 12816 | `0.079405` | 1.44% | 0 | 0 |
| `memcpy_runtime` | 171 | `0.001759` | 0.03% | 0 | 0 |
| `other` | 687 | `0.000668` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.065840` | 0 | 0 |
| `_indexer_bf16_logits_kernel` | 21 | `0.921902` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 1481 | `0.398780` | 0 | 0 |
| `_hc_split_pre_kernel` | 86 | `0.352719` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 1649 | `0.289401` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (i...` | 86 | `0.232961` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2,...` | 631 | `0.152644` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | `0.110714` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` | 108 | `0.106565` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | `0.098370` | 0 | 0 |
| `ampere_sgemm_32x128_tn` | 87 | `0.096712` | 0 | 0 |
| `ampere_sgemm_128x64_tn` | 43 | `0.083572` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076092` | 0 | 0 |
| `_hc_post_kernel` | 86 | `0.069736` | 0 | 0 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 44 | `0.054741` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 770 | `0.054198` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` | 108 | `0.053505` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 644 | `0.047402` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 429 | `0.044605` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` | 322 | `0.043385` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 83 | `5.310348` | 0 | 0 |
| `cudaDeviceSynchronize_v3020` | 1 | `0.113659` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 11914 | `0.072646` | 0 | 0 |
| `cuLaunchKernelEx` | 902 | `0.006759` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 171 | `0.001759` | 0 | 0 |
| `cudaEventRecord_v3020` | 352 | `0.001558` | 0 | 0 |
| `cudaEventQuery_v3020` | 320 | `0.000814` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 89 | `0.000412` | 0 | 0 |
| `cudaStreamWaitEvent_v3020` | 264 | `0.000346` | 0 | 0 |
| `cudaEventCreateWithFlags_v3020` | 88 | `0.000187` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 198 | `0.000182` | 0 | 0 |
| `cudaMemsetAsync_v3020` | 21 | `0.000179` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 292 | `0.000173` | 0 | 0 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 88 | `0.000089` | 0 | 0 |
| `cudaEventDestroy_v3020` | 88 | `0.000078` | 0 | 0 |
| `cudaGetFuncBySymbol_v11000` | 88 | `0.000045` | 0 | 0 |

## repeat_decode_forward_envelope

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| NVTX wall sum | 1 | `6.786952` | ranges=1 |
| kernel | 12349 | `0.030001` |  |
| runtime | 21391 | `6.422404` |  |
| memcpy | 5344 | `0.009752` | bytes=4987264 |
| memset | 0 | `0.000000` |  |
| sync | 1926 | `5.963955` |  |

Kernel categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `runtime_copy_cat_index_kernels` | 7404 | `0.018887` | 62.95% | 0 | 0 |
| `elementwise_math_other` | 4693 | `0.010536` | 35.12% | 0 | 0 |
| `other` | 252 | `0.000579` | 1.93% | 0 | 0 |

Runtime categories:

| Category | Count | Duration s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sync_wait_runtime` | 2683 | `5.968274` | 92.93% | 0 | 0 |
| `cuda_graph_launch_runtime` | 127 | `0.326561` | 5.08% | 0 | 0 |
| `kernel_launch_runtime` | 12349 | `0.080818` | 1.26% | 0 | 0 |
| `memcpy_runtime` | 5344 | `0.045996` | 0.72% | 0 | 0 |
| `other` | 888 | `0.000756` | 0.01% | 0 | 0 |

Top kernels:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` | 757 | `0.004281` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 1638 | `0.004136` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 1134 | `0.001996` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scala...` | 1134 | `0.001940` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, ...` | 882 | `0.001808` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 630 | `0.001680` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instanc...` | 756 | `0.001667` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase &, at::native::...` | 756 | `0.001663` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_ke...` | 630 | `0.001537` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 378 | `0.000945` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char *, (unsigned long)2>>(int,...` | 378 | `0.000740` | 0 | 0 |
| `void at_cuda_detail::cub::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::device_select_policy_hub<long, bool, int, (bool)0, (bool)0...` | 252 | `0.000659` | 0 | 0 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::...` | 127 | `0.000631` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::BUnaryFunctor<long, long, long, at::native::remainder_kernel_cuda(at::...` | 252 | `0.000626` | 0 | 0 |
| `void at_cuda_detail::cub::DeviceReduceSingleTileKernel<at_cuda_detail::cub::DeviceReducePolicy<int, unsigned long long, cuda::std::__4::p...` | 252 | `0.000579` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<long, long, bool, at::native::<unnamed>::CompareEqFuncto...` | 252 | `0.000546` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<i...` | 252 | `0.000509` | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char *, (unsigned long)2>>(int...` | 252 | `0.000507` | 0 | 0 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<int, at::native::func_wrapper_t<int, at::native::MaxNanFunctor<int>...` | 126 | `0.000486` | 0 | 0 |
| `void at_cuda_detail::cub::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, (bool)1>, int *>(T1, int, T2)` | 252 | `0.000432` | 0 | 0 |

Top runtime APIs:

| Name | Count | Duration s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 253 | `5.954598` | 0 | 0 |
| `cudaGraphLaunch_v10000` | 127 | `0.326561` | 0 | 0 |
| `cudaLaunchKernel_v7000` | 12349 | `0.080818` | 0 | 0 |
| `cudaMemcpyAsync_v3020` | 5344 | `0.045996` | 0 | 0 |
| `cudaStreamSynchronize_v3020` | 913 | `0.007521` | 0 | 0 |
| `cudaEventRecordWithFlags_v11010` | 757 | `0.002839` | 0 | 0 |
| `cudaEventQuery_v3020` | 634 | `0.002548` | 0 | 0 |
| `cudaEventSynchronize_v3020` | 126 | `0.000768` | 0 | 0 |
| `cudaStreamIsCapturing_v10000` | 884 | `0.000754` | 0 | 0 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 4 | `0.000002` | 0 | 0 |

