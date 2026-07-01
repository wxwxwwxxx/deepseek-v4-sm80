# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3284220

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3483209 | 78.3065 | |
| graph trace | n/a | n/a | |
| runtime | 554277 | 94.3107 | |
| memcpy | 221096 | 10.3407 | bytes=169962019176 |
| NCCL kernels | 22792 | 1.07136 | |
| NCCL NVTX | 708 | n/a | range=67043380625..158952493259 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4871 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7632 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07065 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 445245 | 4.723 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37194 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01213 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 394044 | 1.78005 |
| `_hc_split_pre_kernel` | 22274 | 0.850238 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 72048 | 0.838839 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 200034 | 0.699149 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 49.3512 |
| `cudaDeviceSynchronize_v3020` | 525 | 31.1179 |
| `cudaGraphLaunch_v10000` | 254 | 5.7186 |
| `cudaLaunchKernel_v7000` | 224806 | 2.36505 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.64636 |
| `cuModuleLoadData` | 71 | 0.918471 |
| `cudaHostAlloc_v3020` | 29 | 0.892418 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2882 | 0.462929 |
| `cudaMalloc_v3020` | 493 | 0.391393 |
| `cuMemExportToShareableHandle` | 48 | 0.276732 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1701590 | 38.7753 | |
| graph trace | n/a | n/a | |
| runtime | 103617 | 39.0985 | |
| memcpy | 54179 | 0.0958753 | bytes=481162364 |
| NCCL kernels | 11264 | 0.525037 | |
| NCCL NVTX | 88 | n/a | range=118713454439..138911013011 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1699 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35744 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.0348 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 220159 | 2.33562 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17089 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00601 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 195242 | 0.881719 |
| `_hc_split_pre_kernel` | 11008 | 0.423307 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 35670 | 0.41442 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 98943 | 0.345637 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.8072 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.7482 |
| `cudaGraphLaunch_v10000` | 127 | 2.89982 |
| `cudaLaunchKernel_v7000` | 58365 | 0.341126 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.281658 |
| `cuLaunchKernelEx` | 1115 | 0.00849477 |
| `cudaEventQuery_v3020` | 1124 | 0.00357374 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00340872 |
| `cudaEventRecord_v3020` | 352 | 0.00137687 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00106979 |

