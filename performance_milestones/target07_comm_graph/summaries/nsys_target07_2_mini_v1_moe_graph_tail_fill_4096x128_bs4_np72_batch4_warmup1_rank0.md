# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_tail_fill_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 264954 | 43.5118 | |
| graph trace | 254 | 42.8628 | |
| runtime | 699881 | 101.198 | |
| memcpy | 188584 | 8.82277 | bytes=169420040072 |
| NCCL kernels | 440 | 0.311171 | |
| NCCL NVTX | 708 | n/a | range=63110088538..164078195787 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9577 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2267 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13268 |
| `_indexer_bf16_logits_kernel` | 42 | 1.8436 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.16016 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10412 | 1.01633 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7844 | 0.607645 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.36154 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2866 | 0.309211 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 1148 | 0.27554 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 48.6978 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.3984 |
| `cudaLaunchKernel_v7000` | 333771 | 2.93241 |
| `cudaGraphLaunch_v10000` | 254 | 1.80442 |
| `cudaMemcpyAsync_v3020` | 188888 | 1.69107 |
| `cudaHostAlloc_v3020` | 29 | 0.957144 |
| `cuModuleLoadData` | 49 | 0.931245 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.512076 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2898 | 0.493585 |
| `cudaMalloc_v3020` | 500 | 0.426963 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 72513 | 21.2977 | |
| graph trace | 127 | 21.4321 | |
| runtime | 116804 | 42.8034 | |
| memcpy | 37923 | 0.0727779 | bytes=210172812 |
| NCCL kernels | 88 | 0.150575 | |
| NCCL NVTX | 88 | n/a | range=120051058564..140869560631 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90153 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08509 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06571 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921629 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577383 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472165 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290746 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178754 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152645 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136926 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 21.0534 |
| `cudaStreamSynchronize_v3020` | 1046 | 20.1891 |
| `cudaGraphLaunch_v10000` | 127 | 0.785774 |
| `cudaLaunchKernel_v7000` | 71700 | 0.462631 |
| `cudaMemcpyAsync_v3020` | 37923 | 0.292235 |
| `cuLaunchKernelEx` | 813 | 0.00700883 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.0039953 |
| `cudaEventQuery_v3020` | 1133 | 0.00374471 |
| `cudaEventRecord_v3020` | 352 | 0.00167701 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00111329 |

