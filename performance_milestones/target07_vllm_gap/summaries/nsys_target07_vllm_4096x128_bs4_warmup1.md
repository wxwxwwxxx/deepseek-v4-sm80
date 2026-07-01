# Nsight Summary: nsys_target07_vllm_4096x128_bs4_warmup1.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 7200
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=5400, GraphExec Creation=1800

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 124480 | 0.978703 | |
| runtime | 1908662 | 46.5982 | |
| memcpy | 584072 | 11.8432 | bytes=165170005376 |
| NCCL kernels | 16 | 0.0212889 | |
| NCCL NVTX | 8472 | n/a | range=15433071552..111841951240 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)2>>` | 37744 | 0.238974 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 20480 | 0.229844 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 20736 | 0.141391 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 19260 | 0.106479 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<unsigned char>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 1376 | 0.0849325 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat16) (instance 5)], std:` | 18740 | 0.0510221 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 16 | 0.0212889 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)1, (int)128, (int)1, (int)16, ` | 160 | 0.0181932 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lam` | 32 | 0.0161181 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lam` | 32 | 0.0160815 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 584064 | 18.964 |
| `cudaStreamSynchronize_v3020` | 579208 | 14.4268 |
| `cudaLaunchKernel_v7000` | 124464 | 2.30319 |
| `cudaMalloc_v3020` | 3168 | 2.12802 |
| `cudaGetDeviceProperties_v2_v12000` | 160 | 2.0319 |
| `cuMemSetAccess` | 2336 | 1.8828 |
| `cuMemImportFromShareableHandle` | 768 | 1.57619 |
| `cuMemCreate` | 1576 | 0.788901 |
| `cudaStreamIsCapturing_v10000` | 581984 | 0.686751 |
| `cuMemMap` | 2336 | 0.373707 |

## nvtx_window

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 0 | 0 | |
| runtime | 1 | 1.9947e-05 | |
| memcpy | 0 | 0 | bytes=0 |
| NCCL kernels | 0 | 0 | |
| NCCL NVTX | 3840 | n/a | range=103026480657..109328907423 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1 | 1.9947e-05 |

