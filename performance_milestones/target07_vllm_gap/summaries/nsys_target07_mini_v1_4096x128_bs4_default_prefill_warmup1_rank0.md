# Nsight Summary: nsys_target07_mini_v1_4096x128_bs4_default_prefill_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 0

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 6663421 | 91.1536 | |
| runtime | 7389232 | 94.533 | |
| memcpy | 204475 | 9.23543 | bytes=169421078290 |
| NCCL kernels | 22528 | 1.48112 | |
| NCCL NVTX | 22532 | n/a | range=70453889807..260634244012 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11008 | 28.3609 |
| `_grouped_fp4_linear_kernel` | 11008 | 18.7128 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10496 | 8.08628 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 532968 | 6.13721 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 952064 | 2.59714 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 348072 | 2.38962 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 451112 | 2.22271 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01147 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 970816 | 1.85064 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 94486 | 1.72305 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 81709 | 49.1237 |
| `cudaLaunchKernel_v7000` | 6441047 | 37.1699 |
| `cudaMemcpyAsync_v3020` | 204475 | 1.80045 |
| `cuLaunchKernelEx` | 157096 | 1.08991 |
| `cuModuleLoadData` | 61 | 0.92697 |
| `cudaHostAlloc_v3020` | 29 | 0.864064 |
| `cudaDeviceSynchronize_v3020` | 521 | 0.525197 |
| `cuLaunchKernel` | 65278 | 0.456769 |
| `cudaMalloc_v3020` | 499 | 0.454348 |
| `cuMemExportToShareableHandle` | 48 | 0.333206 |

## nvtx_window

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 6663421 | 91.1536 | |
| runtime | 7389232 | 94.533 | |
| memcpy | 204475 | 9.23543 | bytes=169421078290 |
| NCCL kernels | 22528 | 1.48112 | |
| NCCL NVTX | 22532 | n/a | range=70453889807..260634244012 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11008 | 28.3609 |
| `_grouped_fp4_linear_kernel` | 11008 | 18.7128 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10496 | 8.08628 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 532968 | 6.13721 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 952064 | 2.59714 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 348072 | 2.38962 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 451112 | 2.22271 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01147 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 970816 | 1.85064 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 94486 | 1.72305 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 81709 | 49.1237 |
| `cudaLaunchKernel_v7000` | 6441047 | 37.1699 |
| `cudaMemcpyAsync_v3020` | 204475 | 1.80045 |
| `cuLaunchKernelEx` | 157096 | 1.08991 |
| `cuModuleLoadData` | 61 | 0.92697 |
| `cudaHostAlloc_v3020` | 29 | 0.864064 |
| `cudaDeviceSynchronize_v3020` | 521 | 0.525197 |
| `cuLaunchKernel` | 65278 | 0.456769 |
| `cudaMalloc_v3020` | 499 | 0.454348 |
| `cuMemExportToShareableHandle` | 48 | 0.333206 |

