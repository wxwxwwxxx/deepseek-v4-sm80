# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 6665976

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 6930930 | 86.9494 | |
| graph trace | n/a | n/a | |
| runtime | 699372 | 104.22 | |
| memcpy | 220842 | 10.3571 | bytes=169961950088 |
| NCCL kernels | 22792 | 1.11895 | |
| NCCL NVTX | 708 | n/a | range=65486494484..168205170779 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.5049 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7588 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07605 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 551178 | 6.01533 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 449042 | 2.30813 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.02505 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 963221 | 1.82122 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 99677 | 1.721 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 456314 | 1.50994 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 986230 | 1.4987 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 50.1206 |
| `cudaDeviceSynchronize_v3020` | 525 | 31.387 |
| `cudaGraphLaunch_v10000` | 254 | 12.8525 |
| `cudaLaunchKernel_v7000` | 333771 | 3.223 |
| `cudaMemcpyAsync_v3020` | 188380 | 1.67187 |
| `cudaHostAlloc_v3020` | 29 | 0.999242 |
| `cuModuleLoadData` | 49 | 0.942884 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.612747 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2898 | 0.530164 |
| `cuMemExportToShareableHandle` | 48 | 0.402148 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3405501 | 42.94 | |
| graph trace | n/a | n/a | |
| runtime | 116538 | 43.3136 | |
| memcpy | 54052 | 0.0932425 | bytes=481127820 |
| NCCL kernels | 11264 | 0.469951 | |
| NCCL NVTX | 88 | n/a | range=123420721196..144249677537 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1704 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.3572 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.0374 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 272532 | 2.97242 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 222420 | 1.1417 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.01238 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 476032 | 0.893657 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 49259 | 0.849913 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 225556 | 0.741039 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 487424 | 0.736817 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 20.1793 |
| `cudaDeviceSynchronize_v3020` | 257 | 16.1999 |
| `cudaGraphLaunch_v10000` | 127 | 6.15549 |
| `cudaLaunchKernel_v7000` | 71700 | 0.46699 |
| `cudaMemcpyAsync_v3020` | 37669 | 0.291544 |
| `cuLaunchKernelEx` | 813 | 0.00700847 |
| `cudaEventQuery_v3020` | 1129 | 0.00385745 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00379099 |
| `cudaEventRecord_v3020` | 352 | 0.00183004 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00106603 |

