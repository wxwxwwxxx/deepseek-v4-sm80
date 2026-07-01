# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_stridedkv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 2962148

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3155505 | 76.7955 | |
| graph trace | n/a | n/a | |
| runtime | 541187 | 96.6725 | |
| memcpy | 221096 | 13.651 | bytes=169962474344 |
| NCCL kernels | 22792 | 1.56549 | |
| NCCL NVTX | 708 | n/a | range=73183520366..163171126593 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.5009 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.76 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 7.99941 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 373219 | 4.06924 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37523 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01081 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 349668 | 1.48229 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.947314 |
| `_hc_split_pre_kernel` | 22274 | 0.850165 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11137 | 0.601325 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 52.5199 |
| `cudaDeviceSynchronize_v3020` | 525 | 29.4641 |
| `cudaGraphLaunch_v10000` | 254 | 5.57673 |
| `cudaLaunchKernel_v7000` | 215928 | 2.53374 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.91269 |
| `cuModuleLoadData` | 69 | 1.38869 |
| `cudaHostAlloc_v3020` | 29 | 0.949109 |
| `cudaMalloc_v3020` | 498 | 0.44023 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2892 | 0.362141 |
| `cuMemExportToShareableHandle` | 48 | 0.311917 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1539329 | 37.784 | |
| graph trace | n/a | n/a | |
| runtime | 102395 | 38.1595 | |
| memcpy | 54179 | 0.0957588 | bytes=481389948 |
| NCCL kernels | 11264 | 0.529057 | |
| NCCL NVTX | 88 | n/a | range=123943585611..144040688924 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1705 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35715 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 3.99898 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 184447 | 2.01099 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17199 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00528 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 173226 | 0.734086 |
| `_hc_split_pre_kernel` | 11008 | 0.423244 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | 0.29614 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 80768 | 0.280036 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.7216 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.1064 |
| `cudaGraphLaunch_v10000` | 127 | 2.63884 |
| `cudaLaunchKernel_v7000` | 57183 | 0.359183 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.312224 |
| `cuLaunchKernelEx` | 1072 | 0.00859612 |
| `cudaEventQuery_v3020` | 1125 | 0.00413004 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00347036 |
| `cudaEventRecord_v3020` | 352 | 0.00142031 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00106453 |

