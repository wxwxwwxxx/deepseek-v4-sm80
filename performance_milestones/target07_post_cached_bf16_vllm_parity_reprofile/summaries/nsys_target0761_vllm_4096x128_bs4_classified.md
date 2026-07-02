# Post-Marlin Nsight Classification: nsys_target0761_vllm_4096x128_bs4.sqlite

- Requested NVTX window: `repeat:decode_throughput_bs8:0`

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 124480 | 0.968121 | |
| graph trace | 0 | 0 | |
| runtime | 1908662 | 46.4047 | |
| memcpy | 584072 | 11.2318 | bytes=165170005376 |

Kernel categories:

| Category | Count | Kernel duration s | Kernel share |
| --- | ---: | ---: | ---: |
| `runtime_memcpy_allocation_kernels` | 102264 | 0.671895 | 69.40% |
| `moe_route_w13_swiglu_w2_sum` | 20768 | 0.23561 | 24.34% |
| `other` | 1432 | 0.0498112 | 5.15% |
| `nccl` | 16 | 0.0108039 | 1.12% |

Runtime categories:

| Category | Count | Runtime duration s | Runtime share |
| --- | ---: | ---: | ---: |
| `memcpy_runtime` | 584072 | 18.7258 | 40.35% |
| `sync_runtime` | 579909 | 14.212 | 30.63% |
| `other` | 604112 | 8.53298 | 18.39% |
| `allocation_runtime` | 3480 | 2.47394 | 5.33% |
| `launch_runtime` | 124480 | 2.45549 | 5.29% |
| `cuda_graph_runtime` | 12592 | 0.00450189 | 0.01% |
| `module_runtime` | 17 | 2.1028e-05 | 0.00% |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueTyp` | 37744 | 0.238966 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 20480 | 0.229939 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa` | 20736 | 0.141343 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa` | 19260 | 0.106463 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<unsigned char>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 1376 | 0.0848839 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(c10::BFloat16) (instance` | 18740 | 0.0510184 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)1, (int)128, (int)1,` | 160 | 0.0181793 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() c` | 32 | 0.0161169 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() c` | 32 | 0.0160817 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 16 | 0.0108039 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 584064 | 18.7256 |
| `cudaStreamSynchronize_v3020` | 579208 | 13.9008 |
| `cudaGetDeviceProperties_v2_v12000` | 160 | 2.8206 |
| `cudaLaunchKernel_v7000` | 124464 | 2.33775 |
| `cuMemSetAccess` | 2336 | 1.89194 |
| `cudaMalloc_v3020` | 3168 | 1.8717 |
| `cuMemImportFromShareableHandle` | 768 | 1.5753 |
| `cuMemCreate` | 1576 | 0.76071 |
| `cudaStreamIsCapturing_v10000` | 581984 | 0.6936 |
| `cuMemMap` | 2336 | 0.380097 |

## nvtx_window

- Window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 0 | 0 | |
| graph trace | 0 | 0 | |
| runtime | 1 | 2.2743e-05 | |
| memcpy | 0 | 0 | bytes=0 |

Kernel categories:

| Category | Count | Kernel duration s | Kernel share |
| --- | ---: | ---: | ---: |

Runtime categories:

| Category | Count | Runtime duration s | Runtime share |
| --- | ---: | ---: | ---: |
| `sync_runtime` | 1 | 2.2743e-05 | 100.00% |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1 | 2.2743e-05 |

