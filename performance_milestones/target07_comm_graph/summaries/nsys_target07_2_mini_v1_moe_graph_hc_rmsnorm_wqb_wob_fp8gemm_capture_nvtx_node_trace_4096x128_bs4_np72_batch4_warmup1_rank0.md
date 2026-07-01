# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3321812

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3521288 | 79.129 | |
| graph trace | n/a | n/a | |
| runtime | 555707 | 94.3059 | |
| memcpy | 221096 | 9.27527 | bytes=169963449704 |
| NCCL kernels | 22792 | 1.57573 | |
| NCCL NVTX | 708 | n/a | range=63390526728..155852199800 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.5034 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.7604 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 8.07557 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 461736 | 5.09525 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.0117 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 404838 | 1.94029 |
| `_quantized_linear_fp8_kernel` | 22102 | 1.63426 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 77445 | 1.16582 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.947439 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 205431 | 0.854343 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 48.2996 |
| `cudaDeviceSynchronize_v3020` | 525 | 31.2511 |
| `cudaGraphLaunch_v10000` | 254 | 5.86969 |
| `cudaLaunchKernel_v7000` | 225863 | 2.41172 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.67778 |
| `cuModuleLoadData` | 68 | 1.43367 |
| `cudaHostAlloc_v3020` | 29 | 1.01932 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2904 | 0.43938 |
| `cudaMalloc_v3020` | 500 | 0.393191 |
| `cuMemExportToShareableHandle` | 48 | 0.298802 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1720429 | 38.9316 | |
| graph trace | n/a | n/a | |
| runtime | 103672 | 39.2616 | |
| memcpy | 54179 | 0.0958812 | bytes=481877628 |
| NCCL kernels | 11264 | 0.528407 | |
| NCCL NVTX | 88 | n/a | range=115315359669..135514260820 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1712 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35876 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 4.03743 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 228330 | 2.51923 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00582 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 200576 | 0.96086 |
| `_quantized_linear_fp8_kernel` | 10922 | 0.806602 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 38337 | 0.576018 |
| `_hc_split_pre_kernel` | 11008 | 0.423412 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 101610 | 0.42227 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.8103 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.8369 |
| `cudaGraphLaunch_v10000` | 127 | 2.95322 |
| `cudaLaunchKernel_v7000` | 58408 | 0.352591 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.287877 |
| `cuLaunchKernelEx` | 1115 | 0.0084064 |
| `cudaEventQuery_v3020` | 1128 | 0.00372589 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00357304 |
| `cudaEventRecord_v3020` | 352 | 0.00134896 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00106628 |

