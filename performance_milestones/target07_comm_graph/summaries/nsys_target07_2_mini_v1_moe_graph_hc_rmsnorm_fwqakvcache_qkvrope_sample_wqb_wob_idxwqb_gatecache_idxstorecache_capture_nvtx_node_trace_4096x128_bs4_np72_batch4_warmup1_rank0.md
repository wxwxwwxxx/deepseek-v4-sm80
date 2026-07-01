# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 2962656

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3155511 | 76.4477 | |
| graph trace | n/a | n/a | |
| runtime | 540740 | 91.7696 | |
| memcpy | 221353 | 9.81054 | bytes=169962478436 |
| NCCL kernels | 22792 | 1.22076 | |
| NCCL NVTX | 708 | n/a | range=65726758652..156163140647 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4901 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7602 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.00026 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 373219 | 4.06854 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.3747 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01077 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 349668 | 1.48223 |
| `_hc_split_pre_kernel` | 22274 | 0.850179 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.610101 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11137 | 0.597215 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 48.7243 |
| `cudaDeviceSynchronize_v3020` | 525 | 30.3872 |
| `cudaGraphLaunch_v10000` | 254 | 5.14793 |
| `cudaLaunchKernel_v7000` | 215432 | 1.89573 |
| `cudaMemcpyAsync_v3020` | 188640 | 1.66346 |
| `cuModuleLoadData` | 69 | 1.02517 |
| `cudaHostAlloc_v3020` | 29 | 0.999215 |
| `cudaMalloc_v3020` | 498 | 0.414738 |
| `cuMemExportToShareableHandle` | 48 | 0.292736 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.245578 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1539329 | 37.7901 | |
| graph trace | n/a | n/a | |
| runtime | 102150 | 38.1559 | |
| memcpy | 54306 | 0.0960814 | bytes=481391980 |
| NCCL kernels | 11264 | 0.536623 | |
| NCCL NVTX | 88 | n/a | range=116879898027..136978160652 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1712 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35485 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 3.99946 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 184447 | 2.01135 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17206 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00531 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 173226 | 0.734265 |
| `_hc_split_pre_kernel` | 11008 | 0.423264 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | 0.29663 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 80768 | 0.280079 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.7351 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.2084 |
| `cudaGraphLaunch_v10000` | 127 | 2.54985 |
| `cudaLaunchKernel_v7000` | 56929 | 0.337817 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.304856 |
| `cuLaunchKernelEx` | 1072 | 0.00820978 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00340535 |
| `cudaEventQuery_v3020` | 1128 | 0.00338585 |
| `cudaEventRecord_v3020` | 352 | 0.00136045 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00113897 |

