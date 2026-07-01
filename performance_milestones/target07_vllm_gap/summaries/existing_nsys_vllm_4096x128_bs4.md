# Nsight Summary: nsys_vllm_4096x128_bs4.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 7200
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=5400, GraphExec Creation=1800

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 124480 | 0.973154 | |
| runtime | 1908662 | 45.2829 | |
| memcpy | 584072 | 11.5303 | bytes=165170005376 |
| NCCL kernels | 16 | 0.0155665 | |
| NCCL NVTX | 8472 | n/a | range=18350821255..180008569934 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)2>>` | 37744 | 0.239059 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 20480 | 0.229946 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 20736 | 0.141389 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 19260 | 0.106497 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<unsigned char>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 1376 | 0.0848962 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat16) (instance 5)], std:` | 18740 | 0.0510383 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)1, (int)128, (int)1, (int)16, ` | 160 | 0.0181833 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lam` | 32 | 0.0161231 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lam` | 32 | 0.0160847 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 16 | 0.0155665 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 584064 | 18.8076 |
| `cudaStreamSynchronize_v3020` | 579208 | 14.1925 |
| `cudaLaunchKernel_v7000` | 124464 | 2.34967 |
| `cuMemSetAccess` | 2336 | 1.91576 |
| `cudaMalloc_v3020` | 3168 | 1.77634 |
| `cudaGetDeviceProperties_v2_v12000` | 160 | 1.60159 |
| `cuMemImportFromShareableHandle` | 768 | 1.26808 |
| `cuMemCreate` | 1576 | 0.90295 |
| `cudaStreamIsCapturing_v10000` | 581984 | 0.688305 |
| `cuMemMap` | 2336 | 0.428531 |

## nvtx_window

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 0 | 0 | |
| runtime | 1 | 2.7952e-05 | |
| memcpy | 0 | 0 | bytes=0 |
| NCCL kernels | 0 | 0 | |
| NCCL NVTX | 3840 | n/a | range=170824163263..177132414191 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1 | 2.7952e-05 |

