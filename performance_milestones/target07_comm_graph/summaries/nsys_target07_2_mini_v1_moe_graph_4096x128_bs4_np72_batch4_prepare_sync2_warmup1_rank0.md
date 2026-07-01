# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_prepare_sync2_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 265699 | 43.6783 | |
| graph trace | 254 | 42.3688 | |
| runtime | 699496 | 104.138 | |
| memcpy | 188584 | 12.0595 | bytes=169420008194 |
| NCCL kernels | 440 | 0.475166 | |
| NCCL NVTX | 708 | n/a | range=65910515409..166531307365 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9776 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2223 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13335 |
| `_indexer_bf16_logits_kernel` | 42 | 1.8439 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.1601 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10349 | 1.01381 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7340 | 0.605009 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.361397 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2572 | 0.308436 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 1127 | 0.275363 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69928 | 51.9488 |
| `cudaDeviceSynchronize_v3020` | 1037 | 40.5993 |
| `cudaLaunchKernel_v7000` | 333741 | 3.09297 |
| `cudaGraphLaunch_v10000` | 254 | 2.09932 |
| `cudaMemcpyAsync_v3020` | 188762 | 1.69219 |
| `cuModuleLoadData` | 45 | 0.953038 |
| `cudaHostAlloc_v3020` | 29 | 0.945735 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2880 | 0.53553 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.506107 |
| `cudaMalloc_v3020` | 500 | 0.347927 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 73485 | 21.3029 | |
| graph trace | 127 | 21.1876 | |
| runtime | 118393 | 42.5633 | |
| memcpy | 37986 | 0.0731176 | bytes=210173001 |
| NCCL kernels | 88 | 0.151181 | |
| NCCL NVTX | 88 | n/a | range=122783071108..143609153490 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90241 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08545 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.0662 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921806 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577434 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.47219 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290758 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178752 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152625 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136889 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 513 | 20.6994 |
| `cudaStreamSynchronize_v3020` | 1109 | 20.1654 |
| `cudaGraphLaunch_v10000` | 127 | 0.887218 |
| `cudaLaunchKernel_v7000` | 72820 | 0.488155 |
| `cudaMemcpyAsync_v3020` | 37986 | 0.303038 |
| `cuLaunchKernelEx` | 665 | 0.00547522 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00455611 |
| `cudaEventQuery_v3020` | 1126 | 0.00354759 |
| `cudaEventRecord_v3020` | 352 | 0.00143954 |
| `cudaEventCreateWithFlags_v3020` | 217 | 0.00115211 |

