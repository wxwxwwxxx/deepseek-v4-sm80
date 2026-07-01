# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3295142

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3494303 | 78.8605 | |
| graph trace | n/a | n/a | |
| runtime | 554742 | 95.5999 | |
| memcpy | 221096 | 10.565 | bytes=169962799464 |
| NCCL kernels | 22792 | 1.58157 | |
| NCCL NVTX | 708 | n/a | range=63410053444..155954584348 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4882 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.752 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.06078 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 456339 | 4.834 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37167 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01191 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 394044 | 1.77994 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.951332 |
| `_hc_split_pre_kernel` | 22274 | 0.850165 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 72048 | 0.840513 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 49.5169 |
| `cudaDeviceSynchronize_v3020` | 525 | 30.1485 |
| `cudaGraphLaunch_v10000` | 254 | 6.73874 |
| `cudaLaunchKernel_v7000` | 225107 | 2.51295 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.78899 |
| `cuModuleLoadData` | 71 | 1.42311 |
| `cudaHostAlloc_v3020` | 29 | 1.06921 |
| `cudaMalloc_v3020` | 496 | 0.402213 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2896 | 0.396143 |
| `cuMemExportToShareableHandle` | 48 | 0.288209 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1707094 | 38.8072 | |
| graph trace | n/a | n/a | |
| runtime | 103669 | 39.1722 | |
| memcpy | 54179 | 0.0958822 | bytes=481552508 |
| NCCL kernels | 11264 | 0.531088 | |
| NCCL NVTX | 88 | n/a | range=115418299239..135626016280 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1687 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35565 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.02998 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 225663 | 2.39049 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17065 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00588 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 195242 | 0.881765 |
| `_hc_split_pre_kernel` | 11008 | 0.423262 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 35670 | 0.415255 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 98943 | 0.345709 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.7901 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.3239 |
| `cudaGraphLaunch_v10000` | 127 | 3.35057 |
| `cudaLaunchKernel_v7000` | 58408 | 0.371051 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.314139 |
| `cuLaunchKernelEx` | 1115 | 0.00959743 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00380608 |
| `cudaEventQuery_v3020` | 1127 | 0.00361049 |
| `cudaEventRecord_v3020` | 352 | 0.001547 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00110508 |

