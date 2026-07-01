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
| kernels | 124480 | 0.982387 | |
| graph trace | n/a | n/a | |
| runtime | 1908662 | 46.1691 | |
| memcpy | 584072 | 11.4359 | bytes=165170005376 |
| NCCL kernels | 16 | 0.0248897 | |
| NCCL NVTX | 8472 | n/a | range=14821275827..114041171875 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)2>>` | 37744 | 0.239039 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 20480 | 0.229925 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 20736 | 0.141373 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 19260 | 0.106504 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<unsigned char>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 1376 | 0.0848831 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat16) (instance 5)], std:` | 18740 | 0.0510317 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 16 | 0.0248897 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)1, (int)128, (int)1, (int)16, ` | 160 | 0.0181712 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lam` | 32 | 0.0161196 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lam` | 32 | 0.0160917 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 584064 | 18.8729 |
| `cudaStreamSynchronize_v3020` | 579208 | 14.1846 |
| `cudaLaunchKernel_v7000` | 124464 | 2.37813 |
| `cudaGetDeviceProperties_v2_v12000` | 160 | 2.05582 |
| `cuMemSetAccess` | 2336 | 1.87031 |
| `cudaMalloc_v3020` | 3168 | 1.73656 |
| `cuMemImportFromShareableHandle` | 768 | 1.62767 |
| `cuMemCreate` | 1576 | 0.738537 |
| `cudaStreamIsCapturing_v10000` | 581984 | 0.679376 |
| `cudaFree_v3020` | 24 | 0.422137 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 0 | 0 | |
| graph trace | n/a | n/a | |
| runtime | 1 | 1.6822e-05 | |
| memcpy | 0 | 0 | bytes=0 |
| NCCL kernels | 0 | 0 | |
| NCCL NVTX | 3840 | n/a | range=105125248017..111419125841 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1 | 1.6822e-05 |

