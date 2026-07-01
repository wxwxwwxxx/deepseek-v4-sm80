# Post-Marlin Nsight Classification: nsys_target07395_splitk_4096x128_bs4_np128_rank0.sqlite

- Requested NVTX window: `repeat:smoke_debug:0`

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 325787 | 13.1023 | |
| graph trace | 254 | 12.5904 | |
| runtime | 764231 | 42.09 | |
| memcpy | 232622 | 10.025 | bytes=173761580864 |

Kernel categories:

| Category | Count | Kernel duration s | Kernel share |
| --- | ---: | ---: | ---: |
| `sparse_attention` | 86 | 4.21671 | 32.18% |
| `runtime_memcpy_allocation_kernels` | 256743 | 2.78165 | 21.23% |
| `indexer_cache` | 468 | 1.93851 | 14.80% |
| `other` | 33943 | 1.30898 | 9.99% |
| `moe_route_w13_swiglu_w2_sum` | 29708 | 0.894436 | 6.83% |
| `hc_rmsnorm_logits_sampling` | 1515 | 0.881258 | 6.73% |
| `dense_linear_other` | 2884 | 0.753415 | 5.75% |
| `nccl` | 440 | 0.327358 | 2.50% |

Runtime categories:

| Category | Count | Runtime duration s | Runtime share |
| --- | ---: | ---: | ---: |
| `sync_runtime` | 125270 | 33.3187 | 79.16% |
| `launch_runtime` | 357721 | 2.88496 | 6.85% |
| `memcpy_runtime` | 233100 | 1.96029 | 4.66% |
| `allocation_runtime` | 1345 | 1.85869 | 4.42% |
| `cuda_graph_runtime` | 36150 | 1.07408 | 2.55% |
| `other` | 10569 | 0.763932 | 1.81% |
| `module_runtime` | 76 | 0.229342 | 0.54% |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 82 | 4.13195 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84405 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() con` | 7029 | 0.82809 |
| `_hc_split_pre_kernel` | 430 | 0.710635 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa` | 6514 | 0.594266 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8,` | 172 | 0.466816 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2608 | 0.308618 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueTyp` | 44112 | 0.280153 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 22016 | 0.245014 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 424 | 0.222103 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 113834 | 21.199 |
| `cudaDeviceSynchronize_v3020` | 525 | 12.0894 |
| `cudaLaunchKernel_v7000` | 348500 | 2.82643 |
| `cudaMemcpyAsync_v3020` | 233100 | 1.96029 |
| `cudaHostAlloc_v3020` | 29 | 1.04129 |
| `cudaGraphLaunch_v10000` | 254 | 0.703574 |
| `cudaMalloc_v3020` | 778 | 0.565939 |
| `cuMemExportToShareableHandle` | 48 | 0.264217 |
| `cudaFree_v3020` | 340 | 0.250998 |
| `cuModuleLoadData` | 72 | 0.22934 |

## nvtx_window

- Window found: False

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 325787 | 13.1023 | |
| graph trace | 254 | 12.5904 | |
| runtime | 764231 | 42.09 | |
| memcpy | 232622 | 10.025 | bytes=173761580864 |

Kernel categories:

| Category | Count | Kernel duration s | Kernel share |
| --- | ---: | ---: | ---: |
| `sparse_attention` | 86 | 4.21671 | 32.18% |
| `runtime_memcpy_allocation_kernels` | 256743 | 2.78165 | 21.23% |
| `indexer_cache` | 468 | 1.93851 | 14.80% |
| `other` | 33943 | 1.30898 | 9.99% |
| `moe_route_w13_swiglu_w2_sum` | 29708 | 0.894436 | 6.83% |
| `hc_rmsnorm_logits_sampling` | 1515 | 0.881258 | 6.73% |
| `dense_linear_other` | 2884 | 0.753415 | 5.75% |
| `nccl` | 440 | 0.327358 | 2.50% |

Runtime categories:

| Category | Count | Runtime duration s | Runtime share |
| --- | ---: | ---: | ---: |
| `sync_runtime` | 125270 | 33.3187 | 79.16% |
| `launch_runtime` | 357721 | 2.88496 | 6.85% |
| `memcpy_runtime` | 233100 | 1.96029 | 4.66% |
| `allocation_runtime` | 1345 | 1.85869 | 4.42% |
| `cuda_graph_runtime` | 36150 | 1.07408 | 2.55% |
| `other` | 10569 | 0.763932 | 1.81% |
| `module_runtime` | 76 | 0.229342 | 0.54% |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 82 | 4.13195 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84405 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() con` | 7029 | 0.82809 |
| `_hc_split_pre_kernel` | 430 | 0.710635 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBa` | 6514 | 0.594266 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8,` | 172 | 0.466816 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2608 | 0.308618 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueTyp` | 44112 | 0.280153 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 22016 | 0.245014 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 424 | 0.222103 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 113834 | 21.199 |
| `cudaDeviceSynchronize_v3020` | 525 | 12.0894 |
| `cudaLaunchKernel_v7000` | 348500 | 2.82643 |
| `cudaMemcpyAsync_v3020` | 233100 | 1.96029 |
| `cudaHostAlloc_v3020` | 29 | 1.04129 |
| `cudaGraphLaunch_v10000` | 254 | 0.703574 |
| `cudaMalloc_v3020` | 778 | 0.565939 |
| `cuMemExportToShareableHandle` | 48 | 0.264217 |
| `cudaFree_v3020` | 340 | 0.250998 |
| `cuModuleLoadData` | 72 | 0.22934 |

