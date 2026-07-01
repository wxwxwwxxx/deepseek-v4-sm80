# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_fused_masked_locs_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 266732 | 43.7652 | |
| graph trace | 254 | 42.8607 | |
| runtime | 701665 | 103.486 | |
| memcpy | 188584 | 10.8179 | bytes=169420040072 |
| NCCL kernels | 440 | 0.579306 | |
| NCCL NVTX | 708 | n/a | range=66544120343..168068194185 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 215 | 17.9635 |
| `_grouped_fp4_linear_kernel` | 215 | 12.2226 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 205 | 4.13193 |
| `_indexer_bf16_logits_kernel` | 42 | 1.84351 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 1295 | 1.15991 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 10412 | 1.01296 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 7844 | 0.60593 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 220 | 0.381134 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 865 | 0.361379 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2866 | 0.308945 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 69802 | 50.5972 |
| `cudaDeviceSynchronize_v3020` | 525 | 41.5807 |
| `cudaLaunchKernel_v7000` | 335549 | 3.16937 |
| `cudaMemcpyAsync_v3020` | 188888 | 1.66043 |
| `cudaGraphLaunch_v10000` | 254 | 1.61526 |
| `cuModuleLoadData` | 49 | 0.946076 |
| `cudaHostAlloc_v3020` | 29 | 0.887198 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 2902 | 0.663731 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.50975 |
| `cudaMalloc_v3020` | 500 | 0.394547 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: True

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 73402 | 21.3042 | |
| graph trace | 127 | 21.4319 | |
| runtime | 117687 | 42.7952 | |
| memcpy | 37923 | 0.0727764 | bytes=210172812 |
| NCCL kernels | 88 | 0.154935 | |
| NCCL NVTX | 88 | n/a | range=123758081684..144575744438 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 43 | 8.9013 |
| `_grouped_fp4_linear_kernel` | 43 | 6.08595 |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | 2.06579 |
| `_indexer_bf16_logits_kernel` | 21 | 0.921659 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloa` | 259 | 0.577394 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 2149 | 0.472199 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 1821 | 0.290794 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFlo` | 173 | 0.178741 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 631 | 0.152631 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[l` | 238 | 0.136902 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaDeviceSynchronize_v3020` | 257 | 21.0976 |
| `cudaStreamSynchronize_v3020` | 1046 | 20.1765 |
| `cudaGraphLaunch_v10000` | 127 | 0.735435 |
| `cudaLaunchKernel_v7000` | 72589 | 0.469892 |
| `cudaMemcpyAsync_v3020` | 37923 | 0.296823 |
| `cuLaunchKernelEx` | 813 | 0.00623654 |
| `cudaEventQuery_v3020` | 1131 | 0.00376243 |
| `cudaEventRecordWithFlags_v11010` | 865 | 0.00365946 |
| `cudaEventRecord_v3020` | 352 | 0.00150807 |
| `cudaStreamIsCapturing_v10000` | 1102 | 0.00116968 |

