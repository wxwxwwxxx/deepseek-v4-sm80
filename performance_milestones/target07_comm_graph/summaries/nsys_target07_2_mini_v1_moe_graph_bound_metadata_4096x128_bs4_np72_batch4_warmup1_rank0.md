# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_bound_metadata_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 264954 | 43.6727 | |
| graph trace | 254 | 42.8667 | |
| runtime | 699394 | 100.808 | |
| memcpy | 188076 | 8.20676 | bytes=169420031944 |
| NCCL kernels | 440 | 0.50032 | |
| NCCL NVTX | 708 | n/a | range=64764573369..165936578672 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9515 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2218 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13215 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84343 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.15991 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10412 | 1.01343 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7844 | 0.606118 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.36142 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2866 | 0.30897 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 220 | 0.302233 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 48.0668 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.5405 |
| `cudaLaunchKernel_v7000` | 333771 | 3.11716 |
| `cudaMemcpyAsync_v3020` | 188380 | 1.67447 |
| `cudaGraphLaunch_v10000` | 254 | 1.65874 |
| `cuModuleLoadData` | 49 | 0.934601 |
| `cudaHostAlloc_v3020` | 29 | 0.910116 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2912 | 0.627505 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.522209 |
| `cudaMalloc_v3020` | 500 | 0.363374 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 72513 | 21.2987 | |
| graph trace | 127 | 21.4362 | |
| runtime | 116553 | 42.791 | |
| memcpy | 37669 | 0.0724239 | bytes=210168748 |
| NCCL kernels | 88 | 0.151198 | |
| NCCL NVTX | 88 | n/a | range=121768151526..142580279451 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90121 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08578 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06576 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921649 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577382 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472201 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.29075 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178755 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152627 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136894 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 21.0948 |
| `cudaStreamSynchronize_v3020` | 1046 | 20.1991 |
| `cudaGraphLaunch_v10000` | 127 | 0.743085 |
| `cudaLaunchKernel_v7000` | 71700 | 0.446075 |
| `cudaMemcpyAsync_v3020` | 37669 | 0.288737 |
| `cuLaunchKernelEx` | 813 | 0.00652882 |
| `cudaEventQuery_v3020` | 1134 | 0.00379547 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.0036786 |
| `cudaEventRecord_v3020` | 352 | 0.00153923 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00107055 |

