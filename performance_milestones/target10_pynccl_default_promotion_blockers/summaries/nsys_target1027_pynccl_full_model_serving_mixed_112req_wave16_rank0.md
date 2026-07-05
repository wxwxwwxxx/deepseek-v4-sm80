# Nsight Summary: nsys_target1027_pynccl_full_model_serving_mixed_112req_wave16_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 20
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=15, GraphExec Creation=5

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 381082 | 5.1226 | |
| graph trace | 441 | 8.97141 | |
| runtime | 810468 | 31.3527 | |
| memcpy | 216214 | 10.7149 | bytes=196460160471 |
| NCCL kernels | 1056 | 0.196963 | |
| NCCL NVTX | 1501 | n/a | range=5300624527..104054835418 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 287 | 0.605024 |
| `_hc_split_pre_kernel` | 1032 | 0.387852 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8, (int)4, (` | 602 | 0.380701 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 5201 | 0.343146 |
| `ampere_sgemm_32x128_tn` | 609 | 0.340878 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)2>>` | 45698 | 0.285136 |
| `void marlin::gptq_marlin_repack_kernel<(int)256, (int)4, (bool)0, (bool)0>(const unsigned int *, const unsigned int *, unsigned int *, int, int)` | 22016 | 0.245192 |
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16(ncclSymDevArgs)` | 870 | 0.188856 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128, (int)1>(T1 *, at::nat` | 22446 | 0.156193 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 23024 | 0.143538 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 129400 | 13.6225 |
| `cudaDeviceSynchronize_v3020` | 914 | 7.9965 |
| `cudaLaunchKernel_v7000` | 380132 | 3.13281 |
| `cudaMemcpyAsync_v3020` | 217698 | 1.90429 |
| `cudaHostAlloc_v3020` | 34 | 1.09041 |
| `cudaGraphLaunch_v10000` | 441 | 0.988392 |
| `cudaMalloc_v3020` | 857 | 0.672621 |
| `cuMemExportToShareableHandle` | 62 | 0.338953 |
| `cudaFree_v3020` | 337 | 0.268495 |
| `cuMemSetAccess` | 218 | 0.226912 |

## nvtx_window

- window name: repeat:serving_mixed_112req_wave16:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 177278 | 3.73768 | |
| graph trace | 441 | 8.97141 | |
| runtime | 269716 | 12.7611 | |
| memcpy | 57834 | 0.13307 | bytes=27386079063 |
| NCCL kernels | 616 | 0.170391 | |
| NCCL NVTX | 616 | n/a | range=84475186322..101055156748 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 287 | 0.605024 |
| `_hc_split_pre_kernel` | 602 | 0.384182 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8, (int)4, (` | 602 | 0.380701 |
| `ampere_sgemm_32x128_tn` | 609 | 0.340878 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2863 | 0.322201 |
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16(ncclSymDevArgs)` | 609 | 0.169882 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 756 | 0.114494 |
| `ampere_sgemm_64x32_sliced1x4_tn` | 301 | 0.111883 |
| `_fp8_activation_quantize_kernel` | 1953 | 0.109836 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 889 | 0.0999586 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 897 | 7.99417 |
| `cudaStreamSynchronize_v3020` | 17227 | 2.06884 |
| `cudaLaunchKernel_v7000` | 167835 | 1.11696 |
| `cudaGraphLaunch_v10000` | 441 | 0.988392 |
| `cudaMemcpyAsync_v3020` | 57834 | 0.454717 |
| `cuLaunchKernelEx` | 9296 | 0.0735264 |
| `cuModuleLoadData` | 20 | 0.0142707 |
| `cudaEventQuery_v3020` | 2995 | 0.0110429 |
| `cudaEventRecordWithFlags_v11010` | 2835 | 0.0105815 |
| `cudaMalloc_v3020` | 11 | 0.00656534 |

