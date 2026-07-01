# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3115056

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3310778 | 77.6203 | |
| graph trace | n/a | n/a | |
| runtime | 547102 | 101.292 | |
| memcpy | 221096 | 17.054 | bytes=169962474344 |
| NCCL kernels | 22792 | 1.07149 | |
| NCCL NVTX | 708 | n/a | range=65888673621..157884418572 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.5005 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7605 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 7.99933 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 417595 | 4.54839 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37486 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01143 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 394044 | 1.77686 |
| `_hc_split_pre_kernel` | 22274 | 0.850394 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 72048 | 0.840322 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 188897 | 0.660859 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 55.8995 |
| `cudaDeviceSynchronize_v3020` | 525 | 28.1622 |
| `cudaGraphLaunch_v10000` | 254 | 8.2465 |
| `cudaLaunchKernel_v7000` | 219583 | 2.696 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.88352 |
| `cudaHostAlloc_v3020` | 29 | 1.0263 |
| `cuModuleLoadData` | 71 | 0.915155 |
| `cudaMalloc_v3020` | 493 | 0.398905 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2884 | 0.382125 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.347312 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1616342 | 38.4434 | |
| graph trace | n/a | n/a | |
| runtime | 102966 | 38.8597 | |
| memcpy | 54179 | 0.0944071 | bytes=481389948 |
| NCCL kernels | 11264 | 0.529786 | |
| NCCL NVTX | 88 | n/a | range=117735765437..137823146429 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1727 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35795 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 3.99914 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 206463 | 2.24864 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17218 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00562 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 195242 | 0.880097 |
| `_hc_split_pre_kernel` | 11008 | 0.423367 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 35670 | 0.415151 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 93439 | 0.326627 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.6898 |
| `cudaDeviceSynchronize_v3020` | 257 | 14.4614 |
| `cudaGraphLaunch_v10000` | 127 | 3.9755 |
| `cudaLaunchKernel_v7000` | 57699 | 0.388359 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.321164 |
| `cuLaunchKernelEx` | 1115 | 0.00958055 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00444876 |
| `cudaEventQuery_v3020` | 1129 | 0.0039352 |
| `cudaEventRecord_v3020` | 352 | 0.00168404 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00104707 |

