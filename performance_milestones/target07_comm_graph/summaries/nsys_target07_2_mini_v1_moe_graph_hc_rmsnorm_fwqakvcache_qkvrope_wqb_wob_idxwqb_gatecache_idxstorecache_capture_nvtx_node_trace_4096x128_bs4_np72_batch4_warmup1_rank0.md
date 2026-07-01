# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 2973070

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3166599 | 76.5574 | |
| graph trace | n/a | n/a | |
| runtime | 541537 | 109.438 | |
| memcpy | 221096 | 25.5828 | bytes=169962181736 |
| NCCL kernels | 22792 | 1.35713 | |
| NCCL NVTX | 708 | n/a | range=84921978378..175516896515 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4809 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7531 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 7.99758 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 373219 | 4.07149 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37465 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01022 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 349668 | 1.48243 |
| `_hc_split_pre_kernel` | 22274 | 0.850041 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.747016 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11137 | 0.597568 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 64.5053 |
| `cudaDeviceSynchronize_v3020` | 525 | 27.3396 |
| `cudaGraphLaunch_v10000` | 254 | 7.74655 |
| `cudaLaunchKernel_v7000` | 216186 | 2.8078 |
| `cudaMemcpyAsync_v3020` | 188634 | 2.15285 |
| `cuModuleLoadData` | 69 | 1.18312 |
| `cudaHostAlloc_v3020` | 29 | 1.03277 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2896 | 0.468506 |
| `cudaMalloc_v3020` | 498 | 0.445334 |
| `cuMemExportToShareableHandle` | 48 | 0.35943 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1544833 | 37.8038 | |
| graph trace | n/a | n/a | |
| runtime | 102447 | 38.202 | |
| memcpy | 54179 | 0.0955506 | bytes=481243644 |
| NCCL kernels | 11264 | 0.532128 | |
| NCCL NVTX | 88 | n/a | range=136082667760..156190412296 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.169 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35611 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 3.99849 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 184447 | 2.01268 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17212 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00509 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 173226 | 0.734494 |
| `_hc_split_pre_kernel` | 11008 | 0.423202 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | 0.29638 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 80768 | 0.279733 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.7053 |
| `cudaDeviceSynchronize_v3020` | 257 | 13.9857 |
| `cudaGraphLaunch_v10000` | 127 | 3.78741 |
| `cudaLaunchKernel_v7000` | 57226 | 0.384708 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.314277 |
| `cuLaunchKernelEx` | 1072 | 0.0102788 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00450656 |
| `cudaEventQuery_v3020` | 1128 | 0.00404094 |
| `cudaEventRecord_v3020` | 352 | 0.0018871 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.0010605 |

