# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 265699 | 43.7817 | |
| runtime | 699008 | 100.973 | |
| memcpy | 188584 | 8.63506 | bytes=169420008194 |
| NCCL kernels | 440 | 0.5938 | |
| NCCL NVTX | 708 | n/a | range=62890709715..163490414077 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9691 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2188 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13265 |
| `_indexer_bf16_logits_kernel` | 42 | 1.8436 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.15993 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10349 | 1.01306 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7340 | 0.604748 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 220 | 0.394713 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.361284 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2572 | 0.308388 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69928 | 48.4484 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.1187 |
| `cudaLaunchKernel_v7000` | 333741 | 3.20225 |
| `cudaMemcpyAsync_v3020` | 188762 | 1.65997 |
| `cudaGraphLaunch_v10000` | 254 | 1.57932 |
| `cudaHostAlloc_v3020` | 29 | 1.0093 |
| `cuModuleLoadData` | 45 | 0.950665 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2896 | 0.700207 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.502317 |
| `cudaMalloc_v3020` | 500 | 0.37638 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 73485 | 21.3007 | |
| runtime | 118152 | 42.5573 | |
| memcpy | 37986 | 0.0731252 | bytes=210173001 |
| NCCL kernels | 88 | 0.150985 | |
| NCCL NVTX | 88 | n/a | range=119586518793..140401251643 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90203 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08417 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06613 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921684 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577379 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472204 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.29079 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178699 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152631 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136926 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 20.8588 |
| `cudaStreamSynchronize_v3020` | 1109 | 20.1833 |
| `cudaGraphLaunch_v10000` | 127 | 0.727076 |
| `cudaLaunchKernel_v7000` | 72820 | 0.472213 |
| `cudaMemcpyAsync_v3020` | 37986 | 0.297428 |
| `cuLaunchKernelEx` | 665 | 0.00508057 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00407812 |
| `cudaEventQuery_v3020` | 1131 | 0.00347712 |
| `cudaEventRecord_v3020` | 352 | 0.00133345 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00102533 |

