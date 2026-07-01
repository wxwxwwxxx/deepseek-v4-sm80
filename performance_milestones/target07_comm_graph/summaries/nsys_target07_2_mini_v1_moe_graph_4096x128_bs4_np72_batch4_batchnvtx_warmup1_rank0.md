# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_batchnvtx_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 265699 | 43.4998 | |
| graph trace | 254 | 42.379 | |
| runtime | 699014 | 113.009 | |
| memcpy | 188584 | 20.049 | bytes=169420008194 |
| NCCL kernels | 440 | 0.311549 | |
| NCCL NVTX | 708 | n/a | range=69828407635..171176539448 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9804 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2253 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13332 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84388 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.16018 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10349 | 1.01528 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7340 | 0.605573 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.36142 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2572 | 0.308487 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 1127 | 0.275429 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69928 | 59.9521 |
| `cudaDeviceSynchronize_v3020` | 525 | 40.154 |
| `cudaLaunchKernel_v7000` | 333741 | 3.30962 |
| `cudaGraphLaunch_v10000` | 254 | 2.56938 |
| `cudaMemcpyAsync_v3020` | 188762 | 1.92283 |
| `cudaHostAlloc_v3020` | 29 | 1.03576 |
| `cuModuleLoadData` | 45 | 0.941391 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.741546 |
| `cudaMalloc_v3020` | 500 | 0.508821 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2900 | 0.463339 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 73485 | 21.3029 | |
| graph trace | 127 | 21.1886 | |
| runtime | 118140 | 42.6248 | |
| memcpy | 37986 | 0.0731267 | bytes=210173001 |
| NCCL kernels | 88 | 0.150227 | |
| NCCL NVTX | 88 | n/a | range=127018288603..147838366046 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.90332 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08531 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06636 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921828 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.57745 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.47221 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290731 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178682 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152623 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136926 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 20.401 |
| `cudaStreamSynchronize_v3020` | 1109 | 20.1912 |
| `cudaGraphLaunch_v10000` | 127 | 1.19166 |
| `cudaLaunchKernel_v7000` | 72820 | 0.505403 |
| `cudaMemcpyAsync_v3020` | 37986 | 0.314807 |
| `cuLaunchKernelEx` | 665 | 0.00549855 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00481971 |
| `cudaEventQuery_v3020` | 1127 | 0.0039997 |
| `cudaEventRecord_v3020` | 352 | 0.00150228 |
| `cudaEventCreateWithFlags_v3020` | 217 | 0.00110789 |

