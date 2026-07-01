# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_gatecache_idxstorecache_capture_nvtx_node_trace_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: False
- CUDA_GRAPH_EVENTS count: 0
- kernel graphNodeId non-null count: 3104134

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 3299641 | 78.1403 | |
| graph trace | n/a | n/a | |
| runtime | 546775 | 92.0279 | |
| memcpy | 221096 | 8.03879 | bytes=169963449704 |
| NCCL kernels | 22792 | 1.62983 | |
| NCCL NVTX | 708 | n/a | range=67729235513..159182529777 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11137 | 28.4794 |
| `_grouped_fp4_linear_kernel` | 11137 | 18.756 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 10619 | 7.99899 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 417595 | 4.54286 |
| `_quantized_linear_fp8_kernel` | 27499 | 2.37105 |
| `_indexer_bf16_logits_kernel` | 5376 | 2.01151 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 394044 | 1.77626 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 11396 | 0.982358 |
| `_hc_split_pre_kernel` | 22274 | 0.850154 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 72048 | 0.838671 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 46.8747 |
| `cudaDeviceSynchronize_v3020` | 525 | 30.7831 |
| `cudaGraphLaunch_v10000` | 254 | 5.55282 |
| `cudaLaunchKernel_v7000` | 219583 | 2.47403 |
| `cudaMemcpyAsync_v3020` | 188634 | 1.64457 |
| `cuModuleLoadData` | 69 | 1.2838 |
| `cudaHostAlloc_v3020` | 29 | 1.07757 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2896 | 0.553505 |
| `cudaMalloc_v3020` | 493 | 0.353626 |
| `cuMemExportToShareableHandle` | 48 | 0.276603 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 1610838 | 38.4376 | |
| graph trace | n/a | n/a | |
| runtime | 102923 | 38.7736 | |
| memcpy | 54179 | 0.0943786 | bytes=481877628 |
| NCCL kernels | 11264 | 0.531574 | |
| NCCL NVTX | 88 | n/a | range=119131111879..139232146241 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 5504 | 14.1706 |
| `_grouped_fp4_linear_kernel` | 5504 | 9.35672 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 5248 | 3.99909 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 206463 | 2.24794 |
| `_quantized_linear_fp8_kernel` | 13589 | 1.17152 |
| `_indexer_bf16_logits_kernel` | 2688 | 1.00566 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 195242 | 0.880625 |
| `_hc_split_pre_kernel` | 11008 | 0.423325 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::arra` | 35670 | 0.414611 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array` | 93439 | 0.326253 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 1046 | 19.7165 |
| `cudaDeviceSynchronize_v3020` | 257 | 15.626 |
| `cudaGraphLaunch_v10000` | 127 | 2.76635 |
| `cudaLaunchKernel_v7000` | 57699 | 0.343343 |
| `cudaMemcpyAsync_v3020` | 37796 | 0.300746 |
| `cuLaunchKernelEx` | 1072 | 0.00843878 |
| `cudaEventQuery_v3020` | 1129 | 0.00354676 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00354257 |
| `cudaEventRecord_v3020` | 352 | 0.00142778 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00110031 |

