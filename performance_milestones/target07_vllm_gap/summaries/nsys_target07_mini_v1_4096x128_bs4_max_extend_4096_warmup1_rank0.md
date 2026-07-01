# Nsight Summary: nsys_target07_mini_v1_4096x128_bs4_max_extend_4096_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 0

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 6824955 | 95.2577 | |
| runtime | 7563623 | 99.7735 | |
| memcpy | 205735 | 11.4023 | bytes=169421081704 |
| NCCL kernels | 23056 | 1.46739 | |
| NCCL NVTX | 23060 | n/a | range=69878903745..268408095698 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11266 | 30.412 |
| `_grouped_fp4_linear_kernel` | 11266 | 20.1838 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10742 | 8.02185 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 545862 | 6.25592 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 974378 | 2.66606 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 358990 | 2.44477 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 461804 | 2.17328 |
| `_indexer_bf16_logits_kernel` | 5502 | 2.01784 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 993664 | 1.8975 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 96670 | 1.78045 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 82657 | 53.2076 |
| `cudaLaunchKernel_v7000` | 6598591 | 38.6753 |
| `cudaMemcpyAsync_v3020` | 205735 | 1.89634 |
| `cuLaunchKernelEx` | 161086 | 1.14086 |
| `cudaHostAlloc_v3020` | 29 | 0.957442 |
| `cudaDeviceSynchronize_v3020` | 533 | 0.941221 |
| `cudaMalloc_v3020` | 494 | 0.520649 |
| `cuLaunchKernel` | 65278 | 0.458828 |
| `cuMemExportToShareableHandle` | 48 | 0.308625 |
| `cudaEventRecord_v3020` | 69169 | 0.290066 |

## nvtx_window

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 6824955 | 95.2577 | |
| runtime | 7563623 | 99.7735 | |
| memcpy | 205735 | 11.4023 | bytes=169421081704 |
| NCCL kernels | 23056 | 1.46739 | |
| NCCL NVTX | 23060 | n/a | range=69878903745..268408095698 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11266 | 30.412 |
| `_grouped_fp4_linear_kernel` | 11266 | 20.1838 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10742 | 8.02185 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 545862 | 6.25592 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 974378 | 2.66606 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 358990 | 2.44477 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 461804 | 2.17328 |
| `_indexer_bf16_logits_kernel` | 5502 | 2.01784 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 993664 | 1.8975 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 96670 | 1.78045 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 82657 | 53.2076 |
| `cudaLaunchKernel_v7000` | 6598591 | 38.6753 |
| `cudaMemcpyAsync_v3020` | 205735 | 1.89634 |
| `cuLaunchKernelEx` | 161086 | 1.14086 |
| `cudaHostAlloc_v3020` | 29 | 0.957442 |
| `cudaDeviceSynchronize_v3020` | 533 | 0.941221 |
| `cudaMalloc_v3020` | 494 | 0.520649 |
| `cuLaunchKernel` | 65278 | 0.458828 |
| `cuMemExportToShareableHandle` | 48 | 0.308625 |
| `cudaEventRecord_v3020` | 69169 | 0.290066 |

