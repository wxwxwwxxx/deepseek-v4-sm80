# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3431032

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3631798 | 78.9789 | |
| graph trace | n/a | n/a | |
| runtime | 559549 | 104.619 | |
| memcpy | 221096 | 19.423 | bytes=169962019176 |
| NCCL kernels | 22792 | 1.23124 | |
| NCCL NVTX | 708 | n/a | range=71528001411..164378788345 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4895 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7605 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07534 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 483838 | 5.66503 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 449042 | 2.30649 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01165 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 99547 | 1.72239 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 227533 | 1.05426 |
| `_hc_split_pre_kernel` | 22274 | 0.850442 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11137 | 0.615387 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 58.4654 |
| `cudaDeviceSynchronize_v3020` | 525 | 31.2407 |
| `cudaGraphLaunch_v10000` | 254 | 6.14843 |
| `cudaLaunchKernel_v7000` | 228959 | 2.54569 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.8407 |
| `cuModuleLoadData` | 65 | 1.03194 |
| `cudaHostAlloc_v3020` | 29 | 1.0267 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2888 | 0.433646 |
| `cudaMalloc_v3020` | 500 | 0.403783 |
| `cuMemExportToShareableHandle` | 48 | 0.295966 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1775039 | 39.0607 | |
| graph trace | n/a | n/a | |
| runtime | 103675 | 39.3762 | |
| memcpy | 54179 | 0.0958811 | bytes=481162364 |
| NCCL kernels | 11264 | 0.539438 | |
| NCCL NVTX | 88 | n/a | range=123724100331..143934604709 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1721 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35703 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.03672 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 239252 | 2.80048 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 222420 | 1.14117 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00577 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 49259 | 0.851018 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 112532 | 0.520894 |
| `_hc_split_pre_kernel` | 11008 | 0.423396 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | 0.303519 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.8077 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.8627 |
| `cudaGraphLaunch_v10000` | 127 | 3.05104 |
| `cudaLaunchKernel_v7000` | 58408 | 0.344811 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.287171 |
| `cuLaunchKernelEx` | 1115 | 0.0108438 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.0034719 |
| `cudaEventQuery_v3020` | 1129 | 0.00345934 |
| `cudaEventRecord_v3020` | 352 | 0.00134771 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00106936 |

