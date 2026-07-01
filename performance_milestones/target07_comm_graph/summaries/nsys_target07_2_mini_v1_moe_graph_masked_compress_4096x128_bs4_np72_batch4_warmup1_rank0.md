# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_masked_compress_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 269526 | 44.0862 | |
| graph trace | 254 | 42.857 | |
| runtime | 704957 | 104.087 | |
| memcpy | 189092 | 9.32401 | bytes=169420048200 |
| NCCL kernels | 440 | 0.897996 | |
| NCCL NVTX | 708 | n/a | range=66517683220..167528397625 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9669 |
| `_grouped_fp4_linear_kernel` | 215 | 12.221 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.1322 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84352 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.15989 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10412 | 1.0122 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 220 | 0.699834 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7844 | 0.605542 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.361434 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2866 | 0.308877 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 49.2299 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.5755 |
| `cudaLaunchKernel_v7000` | 338597 | 3.46956 |
| `cudaMalloc_v3020` | 500 | 1.72398 |
| `cudaMemcpyAsync_v3020` | 189396 | 1.65009 |
| `cudaGraphLaunch_v10000` | 254 | 1.62336 |
| `cudaHostAlloc_v3020` | 29 | 1.09763 |
| `cuModuleLoadData` | 48 | 0.951469 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2896 | 0.8659 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.50387 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 74799 | 21.3027 | |
| graph trace | 127 | 21.4271 | |
| runtime | 119335 | 42.8229 | |
| memcpy | 38177 | 0.0732944 | bytes=210176876 |
| NCCL kernels | 88 | 0.149931 | |
| NCCL NVTX | 88 | n/a | range=123330248294..144139630250 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90172 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08583 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06588 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921687 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577407 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472206 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290734 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178833 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.15263 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136961 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 21.0843 |
| `cudaStreamSynchronize_v3020` | 1046 | 20.212 |
| `cudaGraphLaunch_v10000` | 127 | 0.748123 |
| `cudaLaunchKernel_v7000` | 74113 | 0.466449 |
| `cudaMemcpyAsync_v3020` | 38177 | 0.294195 |
| `cuLaunchKernelEx` | 686 | 0.00488807 |
| `cudaEventQuery_v3020` | 1130 | 0.00405585 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00361989 |
| `cudaEventRecord_v3020` | 352 | 0.00142919 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.0010777 |

