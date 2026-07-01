# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_warmup1_nvtx_repeat_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 450913 | 48.0874 | |
| runtime | 916709 | 151.831 | |
| memcpy | 199820 | 20.9531 | bytes=169422095512 |
| NCCL kernels | 968 | 0.761065 | |
| NCCL NVTX | 1236 | n/a | range=71005209828..213454130659 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 473 | 19.9901 |
| `_grouped_fp4_linear_kernel` | 473 | 13.6685 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 451 | 4.08356 |
| `_indexer_bf16_logits_kernel` | 168 | 1.84775 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 2849 | 1.16409 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 23243 | 1.12496 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 18258 | 0.66114 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 484 | 0.528344 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 1903 | 0.362076 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 2040 | 0.352085 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 1045 | 76.1117 |
| `cudaStreamSynchronize_v3020` | 72722 | 62.7429 |
| `cudaLaunchKernel_v7000` | 514965 | 4.03071 |
| `cudaGraphLaunch_v10000` | 508 | 3.04952 |
| `cudaMemcpyAsync_v3020` | 199998 | 1.98439 |
| `cudaHostAlloc_v3020` | 31 | 0.990668 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.49598 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 3998 | 0.425269 |
| `cudaMalloc_v3020` | 493 | 0.395559 |
| `cuMemExportToShareableHandle` | 48 | 0.29229 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 166092 | 23.4525 | |
| runtime | 226986 | 61.8132 | |
| memcpy | 43604 | 0.0835901 | bytes=211216660 |
| NCCL kernels | 352 | 0.19673 | |
| NCCL NVTX | 352 | n/a | range=147512415709..193786821986 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 172 | 9.91879 |
| `_grouped_fp4_linear_kernel` | 172 | 6.80795 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 164 | 2.04124 |
| `_indexer_bf16_logits_kernel` | 84 | 0.923703 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1036 | 0.580025 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 8596 | 0.528739 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7280 | 0.319047 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 692 | 0.179266 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 1020 | 0.175994 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2524 | 0.158392 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 517 | 38.2052 |
| `cudaStreamSynchronize_v3020` | 2506 | 20.8201 |
| `cudaGraphLaunch_v10000` | 254 | 1.41487 |
| `cudaLaunchKernel_v7000` | 163432 | 0.970948 |
| `cudaMemcpyAsync_v3020` | 43604 | 0.344265 |
| `cuLaunchKernelEx` | 2660 | 0.0193215 |
| `cudaEventRecordWithFlags_v11010` | 1917 | 0.00857655 |
| `cudaEventQuery_v3020` | 2585 | 0.00824679 |
| `cudaMemsetAsync_v3020` | 1104 | 0.006687 |
| `cudaEventRecord_v3020` | 1408 | 0.0047964 |

