# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 6665976

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 6930930 | 86.7958 | |
| graph trace | n/a | n/a | |
| runtime | 699366 | 113.975 | |
| memcpy | 220842 | 19.355 | bytes=169961950088 |
| NCCL kernels | 22792 | 0.963896 | |
| NCCL NVTX | 708 | n/a | range=70513379162..173974540774 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.5071 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7676 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07912 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 551178 | 6.02047 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 449042 | 2.3075 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.02669 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 963221 | 1.8255 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 99677 | 1.7213 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 456314 | 1.5128 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 986230 | 1.50062 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 59.1675 |
| `cudaDeviceSynchronize_v3020` | 525 | 26.5266 |
| `cudaGraphLaunch_v10000` | 254 | 17.8199 |
| `cudaLaunchKernel_v7000` | 333771 | 3.36278 |
| `cudaMemcpyAsync_v3020` | 188380 | 1.86136 |
| `cudaHostAlloc_v3020` | 29 | 1.02833 |
| `cuModuleLoadData` | 49 | 0.92824 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.856842 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2894 | 0.449586 |
| `cudaMalloc_v3020` | 500 | 0.423118 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3405501 | 42.9535 | |
| graph trace | n/a | n/a | |
| runtime | 116529 | 43.4319 | |
| memcpy | 54052 | 0.0924396 | bytes=481127820 |
| NCCL kernels | 11264 | 0.476217 | |
| NCCL NVTX | 88 | n/a | range=129015880947..149840987439 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1745 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35917 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.03879 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 272532 | 2.97291 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 222420 | 1.14043 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.01314 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 476032 | 0.894771 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 49259 | 0.849817 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 225556 | 0.741584 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 487424 | 0.737017 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 20.1776 |
| `cudaDeviceSynchronize_v3020` | 257 | 13.0853 |
| `cudaGraphLaunch_v10000` | 127 | 9.33348 |
| `cudaLaunchKernel_v7000` | 71700 | 0.49157 |
| `cudaMemcpyAsync_v3020` | 37669 | 0.32289 |
| `cuLaunchKernelEx` | 813 | 0.00699336 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00442504 |
| `cudaEventQuery_v3020` | 1126 | 0.00398755 |
| `cudaEventRecord_v3020` | 352 | 0.00175152 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00102585 |

