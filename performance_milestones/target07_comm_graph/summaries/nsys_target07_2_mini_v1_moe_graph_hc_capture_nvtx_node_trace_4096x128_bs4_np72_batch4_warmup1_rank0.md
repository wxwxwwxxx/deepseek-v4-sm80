# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3695192

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3901666 | 79.824 | |
| graph trace | n/a | n/a | |
| runtime | 571228 | 106.284 | |
| memcpy | 220842 | 19.5017 | bytes=169963088008 |
| NCCL kernels | 22792 | 1.06819 | |
| NCCL NVTX | 708 | n/a | range=69934348682..164752960195 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4842 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7622 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07502 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 551178 | 5.9894 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 449042 | 2.30701 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.02496 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 99677 | 1.72314 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 261203 | 1.1536 |
| `_hc_split_pre_kernel` | 22274 | 0.850451 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 178879 | 0.637918 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 58.8374 |
| `cudaDeviceSynchronize_v3020` | 525 | 27.6081 |
| `cudaGraphLaunch_v10000` | 254 | 10.5175 |
| `cudaLaunchKernel_v7000` | 238827 | 2.85465 |
| `cudaMemcpyAsync_v3020` | 188380 | 1.88267 |
| `cudaHostAlloc_v3020` | 29 | 1.05329 |
| `cuModuleLoadData` | 57 | 0.91677 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2884 | 0.501774 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.428836 |
| `cudaMalloc_v3020` | 500 | 0.420374 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1908413 | 39.5469 | |
| graph trace | n/a | n/a | |
| runtime | 104839 | 39.9913 | |
| memcpy | 54052 | 0.0928962 | bytes=481696780 |
| NCCL kernels | 11264 | 0.525636 | |
| NCCL NVTX | 88 | n/a | range=123352601431..143721689841 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1707 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35715 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.03693 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 272532 | 2.9607 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 222420 | 1.14143 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.01237 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 49259 | 0.851126 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 129172 | 0.570123 |
| `_hc_split_pre_kernel` | 11008 | 0.423388 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 88489 | 0.3157 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.9262 |
| `cudaDeviceSynchronize_v3020` | 257 | 14.1665 |
| `cudaGraphLaunch_v10000` | 127 | 5.13432 |
| `cudaLaunchKernel_v7000` | 59832 | 0.41696 |
| `cudaMemcpyAsync_v3020` | 37669 | 0.324037 |
| `cuLaunchKernelEx` | 985 | 0.00900962 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00443656 |
| `cudaEventQuery_v3020` | 1128 | 0.0042152 |
| `cudaEventRecord_v3020` | 352 | 0.00181037 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00104408 |

