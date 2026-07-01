# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_node_trace_probe_4096x16_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 787320

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1029202 | 48.8266 | |
| graph trace | n/a | n/a | |
| runtime | 660513 | 67.9425 | |
| memcpy | 182706 | 12.7032 | bytes=169475336520 |
| NCCL kernels | 3080 | 0.671309 | |
| NCCL NVTX | 708 | n/a | range=68718092022..131094074304 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 1505 | 19.2123 |
| `_grouped_fp4_linear_kernel` | 1505 | 12.9935 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 1435 | 4.59915 |
| `_indexer_bf16_logits_kernel` | 672 | 1.86526 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 74282 | 1.60363 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 9065 | 1.19219 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 59954 | 0.806602 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 1540 | 0.434011 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 6055 | 0.386698 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 130165 | 0.371751 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 68178 | 52.5862 |
| `cudaDeviceSynchronize_v3020` | 77 | 4.1704 |
| `cudaLaunchKernel_v7000` | 310923 | 3.10137 |
| `cudaMemcpyAsync_v3020` | 179140 | 1.60819 |
| `cudaGraphLaunch_v10000` | 30 | 1.47022 |
| `cudaHostAlloc_v3020` | 29 | 1.06307 |
| `cuModuleLoadData` | 49 | 0.948828 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2900 | 0.669803 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.634079 |
| `cudaMalloc_v3020` | 500 | 0.377276 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 454637 | 23.8254 | |
| graph trace | n/a | n/a | |
| runtime | 97121 | 23.8823 | |
| memcpy | 34984 | 0.0643094 | bytes=237821036 |
| NCCL kernels | 1408 | 0.189821 | |
| NCCL NVTX | 88 | n/a | range=106620637920..127435601472 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 688 | 9.52501 |
| `_grouped_fp4_linear_kernel` | 688 | 6.47195 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 656 | 2.29929 |
| `_indexer_bf16_logits_kernel` | 336 | 0.932547 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 34084 | 0.767697 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 4144 | 0.593602 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 27876 | 0.391281 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 2768 | 0.191394 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 59504 | 0.169456 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 10096 | 0.167677 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 234 | 20.2034 |
| `cudaDeviceSynchronize_v3020` | 33 | 2.36248 |
| `cudaGraphLaunch_v10000` | 15 | 0.689751 |
| `cudaLaunchKernel_v7000` | 60276 | 0.363443 |
| `cudaMemcpyAsync_v3020` | 33049 | 0.252432 |
| `cuLaunchKernelEx` | 701 | 0.00496564 |
| `cudaEventQuery_v3020` | 574 | 0.00147657 |
| `cudaEventRecord_v3020` | 352 | 0.00145486 |
| `cudaEventRecordWithFlags_v11010` | 193 | 0.000790151 |
| `cudaMemsetAsync_v3020` | 121 | 0.000560524 |

