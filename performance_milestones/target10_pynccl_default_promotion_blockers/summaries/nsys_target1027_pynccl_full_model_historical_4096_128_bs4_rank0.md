# Nsight Summary: nsys_target1027_pynccl_full_model_historical_4096_128_bs4_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 20
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=15, GraphExec Creation=5

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 262071 | 5.79857 | |
| graph trace | 127 | 2.54477 | |
| runtime | 643666 | 23.3096 | |
| memcpy | 196984 | 8.23459 | bytes=171422790511 |
| NCCL kernels | 528 | 0.261941 | |
| NCCL NVTX | 973 | n/a | range=5163636726..93528657716 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.07162 |
| `_hc_split_pre_kernel` | 516 | 0.360945 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2747 | 0.315131 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)2>>` | 44110 | 0.265225 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8, (int)4, (` | 86 | 0.235274 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 22016 | 0.233716 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128, (int)1>(T1 *, at::nat` | 22446 | 0.154336 |
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16(ncclSymDevArgs)` | 261 | 0.153342 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 22400 | 0.135218 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 23140 | 0.131432 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 114404 | 12.691 |
| `cudaLaunchKernel_v7000` | 269108 | 2.55552 |
| `cudaDeviceSynchronize_v3020` | 274 | 2.51527 |
| `cudaMemcpyAsync_v3020` | 198468 | 1.76516 |
| `cudaMalloc_v3020` | 868 | 0.930524 |
| `cudaHostAlloc_v3020` | 29 | 0.894068 |
| `cuMemExportToShareableHandle` | 62 | 0.30865 |
| `cudaFree_v3020` | 337 | 0.266691 |
| `cuModuleLoadData` | 88 | 0.222678 |
| `cuMemSetAccess` | 218 | 0.179642 |

## nvtx_window

- window name: repeat:historical_4096_128_bs4:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 58357 | 4.32867 | |
| graph trace | 127 | 2.54477 | |
| runtime | 103064 | 7.26832 | |
| memcpy | 38634 | 0.0907383 | bytes=2348709343 |
| NCCL kernels | 88 | 0.10567 | |
| NCCL NVTX | 88 | n/a | range=83429161445..88509729318 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.07162 |
| `_hc_split_pre_kernel` | 86 | 0.357321 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 409 | 0.294356 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8, (int)4, (` | 86 | 0.235274 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 108 | 0.106598 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 87 | 0.105632 |
| `_fp8_activation_quantize_kernel` | 279 | 0.101459 |
| `ampere_sgemm_32x128_tn` | 87 | 0.0978148 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | 0.0896917 |
| `ampere_sgemm_128x64_tn` | 43 | 0.0845102 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 2261 | 3.65249 |
| `cudaDeviceSynchronize_v3020` | 257 | 2.51304 |
| `cudaLaunchKernel_v7000` | 56901 | 0.40462 |
| `cudaMemcpyAsync_v3020` | 38634 | 0.300725 |
| `cuModuleLoadData` | 17 | 0.215043 |
| `cudaGraphLaunch_v10000` | 127 | 0.142243 |
| `cudaMalloc_v3020` | 22 | 0.0166347 |
| `cuLaunchKernelEx` | 1456 | 0.0119709 |
| `cudaEventRecordWithFlags_v11010` | 777 | 0.00291004 |
| `cudaEventQuery_v3020` | 735 | 0.00277676 |

