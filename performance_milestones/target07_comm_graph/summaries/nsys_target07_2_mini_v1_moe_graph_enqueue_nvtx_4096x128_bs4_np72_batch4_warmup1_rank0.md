# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_enqueue_nvtx_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 266732 | 43.9927 | |
| graph trace | 254 | 42.8598 | |
| runtime | 701638 | 106.224 | |
| memcpy | 188584 | 12.7265 | bytes=169420040072 |
| NCCL kernels | 440 | 0.803524 | |
| NCCL NVTX | 708 | n/a | range=67533830447..169028697614 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.962 |
| `_grouped_fp4_linear_kernel` | 215 | 12.224 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13303 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84381 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.16009 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10412 | 1.01389 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7844 | 0.606256 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 220 | 0.604321 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.361551 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2866 | 0.308963 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 52.5174 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.3813 |
| `cudaLaunchKernel_v7000` | 335549 | 3.59217 |
| `cudaMemcpyAsync_v3020` | 188888 | 1.79224 |
| `cudaGraphLaunch_v10000` | 254 | 1.78739 |
| `cudaHostAlloc_v3020` | 29 | 1.03032 |
| `cuModuleLoadData` | 49 | 0.939229 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2884 | 0.915305 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.504083 |
| `cudaMalloc_v3020` | 500 | 0.381256 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 73402 | 21.302 | |
| graph trace | 127 | 21.4308 | |
| runtime | 117669 | 42.8417 | |
| memcpy | 37923 | 0.072802 | bytes=210172812 |
| NCCL kernels | 88 | 0.150213 | |
| NCCL NVTX | 88 | n/a | range=124678140391..145499327324 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90223 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08668 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06621 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921807 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577442 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472231 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290766 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178804 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152609 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.13691 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 20.9335 |
| `cudaStreamSynchronize_v3020` | 1046 | 20.1705 |
| `cudaGraphLaunch_v10000` | 127 | 0.884176 |
| `cudaLaunchKernel_v7000` | 72589 | 0.516439 |
| `cudaMemcpyAsync_v3020` | 37923 | 0.31544 |
| `cuLaunchKernelEx` | 813 | 0.0074572 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.0043341 |
| `cudaEventQuery_v3020` | 1125 | 0.00384038 |
| `cudaEventRecord_v3020` | 352 | 0.00164886 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00116226 |

