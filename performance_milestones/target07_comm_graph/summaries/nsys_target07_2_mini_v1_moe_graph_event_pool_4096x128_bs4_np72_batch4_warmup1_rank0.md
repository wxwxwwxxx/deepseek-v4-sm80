# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_event_pool_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 265699 | 43.9461 | |
| graph trace | 254 | 42.3604 | |
| runtime | 698480 | 104.278 | |
| memcpy | 188584 | 11.5856 | bytes=169420008194 |
| NCCL kernels | 440 | 0.764908 | |
| NCCL NVTX | 708 | n/a | range=69045977257..169643593223 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9693 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2185 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13179 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84334 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.15979 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10349 | 1.01249 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7340 | 0.604405 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 220 | 0.566769 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.361347 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2572 | 0.30833 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69928 | 51.4868 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.1838 |
| `cudaLaunchKernel_v7000` | 333741 | 3.37581 |
| `cudaMemcpyAsync_v3020` | 188762 | 1.70183 |
| `cudaGraphLaunch_v10000` | 254 | 1.52116 |
| `cudaHostAlloc_v3020` | 29 | 1.00799 |
| `cuModuleLoadData` | 45 | 0.954056 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2884 | 0.74794 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.501106 |
| `cudaMalloc_v3020` | 500 | 0.395404 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 73485 | 21.3 | |
| graph trace | 127 | 21.1805 | |
| runtime | 117884 | 42.5603 | |
| memcpy | 37986 | 0.0731417 | bytes=210173001 |
| NCCL kernels | 88 | 0.151882 | |
| NCCL NVTX | 88 | n/a | range=125714788303..146529238791 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.9011 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08426 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06564 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921563 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577334 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472165 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290726 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178801 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152609 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136902 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 20.9066 |
| `cudaStreamSynchronize_v3020` | 1109 | 20.1978 |
| `cudaGraphLaunch_v10000` | 127 | 0.680511 |
| `cudaLaunchKernel_v7000` | 72820 | 0.466071 |
| `cudaMemcpyAsync_v3020` | 37986 | 0.291819 |
| `cuLaunchKernelEx` | 665 | 0.00485941 |
| `cudaEventQuery_v3020` | 1127 | 0.00394622 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00362071 |
| `cudaEventRecord_v3020` | 352 | 0.00142715 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00100493 |

