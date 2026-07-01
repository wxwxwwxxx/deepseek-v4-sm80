# Post-Marlin Nsight Classification: nsys_marlin_wna16_4096x128_bs4_np128_rank0.sqlite

- Requested NVTX window: `repeat:smoke_debug:0`

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 268288 | 7.43553 | |
| graph trace | 127 | 8.20997 | |
| runtime | 662937 | 36.9547 | |
| memcpy | 194741 | 14.5916 | bytes=171387088232 |

Kernel categories:

| Category | Count | Kernel duration s | Kernel share |
| --- | ---: | ---: | ---: |
| `sparse_attention` | 172 | 2.11036 | 28.38% |
| `runtime_memcpy_allocation_kernels` | 212836 | 1.90944 | 25.68% |
| `indexer_cache` | 342 | 0.969631 | 13.04% |
| `other` | 23522 | 0.703668 | 9.46% |
| `moe_route_w13_swiglu_w2_sum` | 27548 | 0.575325 | 7.74% |
| `hc_rmsnorm_logits_sampling` | 1212 | 0.444044 | 5.97% |
| `dense_linear_other` | 2304 | 0.384948 | 5.18% |
| `nccl` | 352 | 0.338117 | 4.55% |

Runtime categories:

| Category | Count | Runtime duration s | Runtime share |
| --- | ---: | ---: | ---: |
| `sync_runtime` | 121203 | 28.4968 | 77.11% |
| `launch_runtime` | 300435 | 2.60419 | 7.05% |
| `allocation_runtime` | 1345 | 1.87447 | 5.07% |
| `memcpy_runtime` | 195177 | 1.82099 | 4.93% |
| `other` | 9258 | 1.05123 | 2.84% |
| `cuda_graph_runtime` | 35454 | 0.7225 | 1.96% |
| `module_runtime` | 65 | 0.384552 | 1.04% |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 164 | 2.06733 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921976 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() con` | 5548 | 0.428837 |
| `_hc_split_pre_kernel` | 344 | 0.357531 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa` | 4865 | 0.304836 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueTyp` | 44072 | 0.277639 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 22016 | 0.243642 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 176 | 0.234505 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8,` | 86 | 0.23423 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128, (int)1>(T1 ` | 22446 | 0.156345 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 112788 | 20.5335 |
| `cudaDeviceSynchronize_v3020` | 267 | 7.94302 |
| `cudaLaunchKernel_v7000` | 293136 | 2.55965 |
| `cudaMemcpyAsync_v3020` | 195177 | 1.82099 |
| `cudaHostAlloc_v3020` | 29 | 1.02052 |
| `cudaMalloc_v3020` | 778 | 0.619011 |
| `cuModuleLoadData` | 61 | 0.38455 |
| `cudaGraphLaunch_v10000` | 127 | 0.37317 |
| `cuMemExportToShareableHandle` | 48 | 0.312493 |
| `cudaFree_v3020` | 340 | 0.234397 |

## nvtx_window

- Window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 58132 | 5.87807 | |
| graph trace | 127 | 8.20997 | |
| runtime | 101896 | 14.3675 | |
| memcpy | 37839 | 0.0873563 | bytes=2374492380 |

Kernel categories:

| Category | Count | Kernel duration s | Kernel share |
| --- | ---: | ---: | ---: |
| `sparse_attention` | 43 | 2.10988 | 35.89% |
| `indexer_cache` | 126 | 0.968744 | 16.48% |
| `runtime_memcpy_allocation_kernels` | 44396 | 0.890159 | 15.14% |
| `other` | 10541 | 0.618552 | 10.52% |
| `hc_rmsnorm_logits_sampling` | 303 | 0.439664 | 7.48% |
| `dense_linear_other` | 580 | 0.370444 | 6.30% |
| `moe_route_w13_swiglu_w2_sum` | 2055 | 0.317591 | 5.40% |
| `nccl` | 88 | 0.163036 | 2.77% |

Runtime categories:

| Category | Count | Runtime duration s | Runtime share |
| --- | ---: | ---: | ---: |
| `sync_runtime` | 4059 | 13.0773 | 91.02% |
| `launch_runtime` | 58132 | 0.395269 | 2.75% |
| `cuda_graph_runtime` | 499 | 0.37341 | 2.60% |
| `memcpy_runtime` | 37839 | 0.290361 | 2.02% |
| `module_runtime` | 16 | 0.215858 | 1.50% |
| `allocation_runtime` | 24 | 0.0137703 | 0.10% |
| `other` | 1327 | 0.00155895 | 0.01% |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06687 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921976 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() con` | 1481 | 0.400697 |
| `_hc_split_pre_kernel` | 86 | 0.355535 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa` | 1649 | 0.289507 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8,` | 86 | 0.23423 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152617 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 212 | 0.111347 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase ` | 108 | 0.106583 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 43 | 0.0990741 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 7.94199 |
| `cudaStreamSynchronize_v3020` | 1046 | 5.12614 |
| `cudaLaunchKernel_v7000` | 57230 | 0.388717 |
| `cudaGraphLaunch_v10000` | 127 | 0.37317 |
| `cudaMemcpyAsync_v3020` | 37839 | 0.290361 |
| `cuModuleLoadData` | 16 | 0.215858 |
| `cudaMalloc_v3020` | 16 | 0.0114041 |
| `cuLaunchKernelEx` | 902 | 0.00655155 |
| `cudaEventQuery_v3020` | 965 | 0.00333539 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00323607 |

