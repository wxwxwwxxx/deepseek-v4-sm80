# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_capture_nvtx_node_trace_probe_4096x16_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 787320

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1029202 | 49.0452 | |
| graph trace | n/a | n/a | |
| runtime | 660482 | 66.3489 | |
| memcpy | 182706 | 10.9958 | bytes=169475336520 |
| NCCL kernels | 3080 | 0.880164 | |
| NCCL NVTX | 708 | n/a | range=66778982220..129122910967 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 1505 | 19.2147 |
| `_grouped_fp4_linear_kernel` | 1505 | 12.9938 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 1435 | 4.59951 |
| `_indexer_bf16_logits_kernel` | 672 | 1.8655 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 74282 | 1.60491 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 9065 | 1.19251 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 59954 | 0.806745 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 1540 | 0.6427 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 6055 | 0.386711 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 130165 | 0.372599 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 68178 | 50.792 |
| `cudaDeviceSynchronize_v3020` | 77 | 4.08215 |
| `cudaLaunchKernel_v7000` | 310923 | 3.29295 |
| `cudaMemcpyAsync_v3020` | 179140 | 1.56598 |
| `cudaGraphLaunch_v10000` | 30 | 1.56017 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2880 | 0.946761 |
| `cuModuleLoadData` | 49 | 0.944202 |
| `cudaHostAlloc_v3020` | 29 | 0.938228 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.619924 |
| `cudaMalloc_v3020` | 500 | 0.353629 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 454637 | 23.8301 | |
| graph trace | n/a | n/a | |
| runtime | 97097 | 23.8557 | |
| memcpy | 34984 | 0.0642372 | bytes=237821036 |
| NCCL kernels | 1408 | 0.193263 | |
| NCCL NVTX | 88 | n/a | range=104651857792..125472305217 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 688 | 9.52533 |
| `_grouped_fp4_linear_kernel` | 688 | 6.47242 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 656 | 2.29956 |
| `_indexer_bf16_logits_kernel` | 336 | 0.932697 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 34084 | 0.767783 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 4144 | 0.593673 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 27876 | 0.391054 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 2768 | 0.191368 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 59504 | 0.169556 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 10096 | 0.16762 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 234 | 20.1799 |
| `cudaDeviceSynchronize_v3020` | 33 | 2.31522 |
| `cudaGraphLaunch_v10000` | 15 | 0.738431 |
| `cudaLaunchKernel_v7000` | 60276 | 0.366608 |
| `cudaMemcpyAsync_v3020` | 33049 | 0.243745 |
| `cuLaunchKernelEx` | 701 | 0.0057394 |
| `cudaEventQuery_v3020` | 566 | 0.00154513 |
| `cudaEventRecord_v3020` | 352 | 0.00151716 |
| `cudaEventRecordWithFlags_v11010` | 193 | 0.000805577 |
| `cudaMemsetAsync_v3020` | 121 | 0.000599627 |

