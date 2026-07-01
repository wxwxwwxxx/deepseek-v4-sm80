# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 450913 | 47.8212 | |
| runtime | 916763 | 143.273 | |
| memcpy | 199820 | 13.0364 | bytes=169422095512 |
| NCCL kernels | 968 | 0.411417 | |
| NCCL NVTX | 1236 | n/a | range=67789634656..210531433433 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 473 | 20.0048 |
| `_grouped_fp4_linear_kernel` | 473 | 13.6773 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 451 | 4.08217 |
| `_indexer_bf16_logits_kernel` | 168 | 1.8473 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 2849 | 1.16477 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 23243 | 1.1279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 18258 | 0.662162 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 1903 | 0.362446 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 2040 | 0.352141 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 6358 | 0.320064 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1045 | 76.0252 |
| `cudaStreamSynchronize_v3020` | 72722 | 54.7789 |
| `cudaLaunchKernel_v7000` | 514965 | 3.62554 |
| `cudaGraphLaunch_v10000` | 508 | 3.14794 |
| `cudaMemcpyAsync_v3020` | 199998 | 1.75963 |
| `cudaHostAlloc_v3020` | 31 | 1.00186 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.515891 |
| `cuMemExportToShareableHandle` | 48 | 0.376562 |
| `cudaMalloc_v3020` | 493 | 0.354127 |
| `cuModuleLoadData` | 45 | 0.267281 |

## nvtx_window

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 450913 | 47.8212 | |
| runtime | 916763 | 143.273 | |
| memcpy | 199820 | 13.0364 | bytes=169422095512 |
| NCCL kernels | 968 | 0.411417 | |
| NCCL NVTX | 1236 | n/a | range=67789634656..210531433433 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 473 | 20.0048 |
| `_grouped_fp4_linear_kernel` | 473 | 13.6773 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 451 | 4.08217 |
| `_indexer_bf16_logits_kernel` | 168 | 1.8473 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 2849 | 1.16477 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 23243 | 1.1279 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 18258 | 0.662162 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 1903 | 0.362446 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 2040 | 0.352141 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 6358 | 0.320064 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1045 | 76.0252 |
| `cudaStreamSynchronize_v3020` | 72722 | 54.7789 |
| `cudaLaunchKernel_v7000` | 514965 | 3.62554 |
| `cudaGraphLaunch_v10000` | 508 | 3.14794 |
| `cudaMemcpyAsync_v3020` | 199998 | 1.75963 |
| `cudaHostAlloc_v3020` | 31 | 1.00186 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.515891 |
| `cuMemExportToShareableHandle` | 48 | 0.376562 |
| `cudaMalloc_v3020` | 493 | 0.354127 |
| `cuModuleLoadData` | 45 | 0.267281 |

