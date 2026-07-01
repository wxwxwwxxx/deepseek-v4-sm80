# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3278886

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3477833 | 78.5631 | |
| graph trace | n/a | n/a | |
| runtime | 554204 | 95.5276 | |
| memcpy | 221096 | 10.9985 | bytes=169962181736 |
| NCCL kernels | 22792 | 1.34621 | |
| NCCL NVTX | 708 | n/a | range=66585682526..158703922116 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4962 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7473 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07058 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 439869 | 4.71352 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37154 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01213 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 394044 | 1.77961 |
| `_hc_split_pre_kernel` | 22274 | 0.850281 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 72048 | 0.840512 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.716006 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 49.9999 |
| `cudaDeviceSynchronize_v3020` | 525 | 30.9464 |
| `cudaGraphLaunch_v10000` | 254 | 5.88075 |
| `cudaLaunchKernel_v7000` | 224743 | 2.4158 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.67748 |
| `cuModuleLoadData` | 71 | 1.19946 |
| `cudaHostAlloc_v3020` | 29 | 1.02101 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2888 | 0.48927 |
| `cudaMalloc_v3020` | 493 | 0.37337 |
| `cuMemExportToShareableHandle` | 48 | 0.358744 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1698902 | 38.766 | |
| graph trace | n/a | n/a | |
| runtime | 103603 | 39.1185 | |
| memcpy | 54179 | 0.0945972 | bytes=481243644 |
| NCCL kernels | 11264 | 0.528177 | |
| NCCL NVTX | 88 | n/a | range=118477441125..138672898727 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1688 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35203 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.03469 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 217471 | 2.33087 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17071 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00599 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 195242 | 0.881667 |
| `_hc_split_pre_kernel` | 11008 | 0.423325 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 35670 | 0.415296 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 98943 | 0.34575 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.7996 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.7331 |
| `cudaGraphLaunch_v10000` | 127 | 2.9133 |
| `cudaLaunchKernel_v7000` | 58344 | 0.350958 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.300757 |
| `cuLaunchKernelEx` | 1115 | 0.00884428 |
| `cudaEventQuery_v3020` | 1127 | 0.00358869 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00338032 |
| `cudaEventRecord_v3020` | 352 | 0.00140377 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00106707 |

