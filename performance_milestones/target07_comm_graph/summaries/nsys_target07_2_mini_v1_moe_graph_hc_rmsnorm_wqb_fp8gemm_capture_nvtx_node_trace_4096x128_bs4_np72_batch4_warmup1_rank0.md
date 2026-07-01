# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3376422

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3576543 | 78.8972 | |
| graph trace | n/a | n/a | |
| runtime | 557641 | 97.1377 | |
| memcpy | 221096 | 12.0672 | bytes=169962994536 |
| NCCL kernels | 22792 | 1.25015 | |
| NCCL NVTX | 708 | n/a | range=66550297547..159027359496 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4882 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7565 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.0739 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 472787 | 5.37951 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 426940 | 2.12273 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01127 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 88496 | 1.44406 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 216482 | 0.954416 |
| `_hc_split_pre_kernel` | 22274 | 0.850524 |
| `_quantized_linear_fp8_kernel` | 11051 | 0.816688 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 51.0373 |
| `cudaDeviceSynchronize_v3020` | 525 | 30.7536 |
| `cudaGraphLaunch_v10000` | 254 | 6.50632 |
| `cudaLaunchKernel_v7000` | 227411 | 2.52392 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.81095 |
| `cuModuleLoadData` | 68 | 1.08651 |
| `cudaHostAlloc_v3020` | 29 | 1.02347 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2904 | 0.433361 |
| `cudaMalloc_v3020` | 500 | 0.401612 |
| `cuMemExportToShareableHandle` | 48 | 0.290317 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1747734 | 38.9838 | |
| graph trace | n/a | n/a | |
| runtime | 103690 | 39.3003 | |
| memcpy | 54179 | 0.0958993 | bytes=481650044 |
| NCCL kernels | 11264 | 0.529114 | |
| NCCL NVTX | 88 | n/a | range=118608406444..138817058890 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1692 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35513 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.03672 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 233791 | 2.66006 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 211498 | 1.05097 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00553 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 43798 | 0.713522 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 107071 | 0.471781 |
| `_hc_split_pre_kernel` | 11008 | 0.423428 |
| `_quantized_linear_fp8_kernel` | 5461 | 0.403158 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.781 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.8101 |
| `cudaGraphLaunch_v10000` | 127 | 3.0444 |
| `cudaLaunchKernel_v7000` | 58408 | 0.346902 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.295487 |
| `cuLaunchKernelEx` | 1115 | 0.0091293 |
| `cudaEventQuery_v3020` | 1134 | 0.003741 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00367336 |
| `cudaEventRecord_v3020` | 352 | 0.00167328 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00119587 |

