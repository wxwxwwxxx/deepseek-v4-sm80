# Nsight Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_prepare_sync_warmup1_rank0.sqlite

- Lite mode: False

## CUDA Graph

- CUDA_GRAPH_EVENTS present: True
- CUDA_GRAPH_EVENTS count: 12
- kernel graphNodeId non-null count: 0
- graph event names: Graph Creation=9, GraphExec Creation=3

## total

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | 118729 | 1.44341 | |
| graph trace | n/a | n/a | |
| runtime | 462750 | 14.9728 | |
| memcpy | 112616 | 7.93682 | bytes=168999727728 |
| NCCL kernels | 264 | 0.597298 | |
| NCCL NVTX | 532 | n/a | range=69807424126..81073307071 |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 132 | 0.593067 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)2, (int)128, (int)1>(T1 *, at::nat` | 22274 | 0.144772 |
| `_grouped_fp4_w13_kernel` | 129 | 0.142243 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 11008 | 0.0725691 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambd` | 6051 | 0.0679279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::float8_copy_kernel_cuda(at::TensorIteratorBase &)::[la` | 11094 | 0.052527 |
| `_grouped_fp4_linear_kernel` | 129 | 0.0500766 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_contig<at::native::<unnamed>::OpaqueType<(unsigned int)1>, unsigned int, (int)3, (int)128, (int)1>(T1 *, at::nat` | 172 | 0.0497194 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_` | 11157 | 0.0336372 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::ope` | 5202 | 0.02765 |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |
| `cudaStreamSynchronize_v3020` | 67710 | 8.43825 |
| `cudaLaunchKernel_v7000` | 188101 | 2.0449 |
| `cudaMemcpyAsync_v3020` | 112794 | 1.0616 |
| `cudaHostAlloc_v3020` | 21 | 0.884133 |
| `cudaGraphInstantiateWithFlags_v11040` | 3 | 0.503955 |
| `cudaThreadExchangeStreamCaptureMode_v10010` | 1722 | 0.50254 |
| `cudaMalloc_v3020` | 483 | 0.42725 |
| `cuMemExportToShareableHandle` | 48 | 0.296678 |
| `cuMemSetAccess` | 194 | 0.150334 |
| `cuMemImportFromShareableHandle` | 48 | 0.108435 |

## nvtx_window

- window name: repeat:decode_throughput_bs8:0
- window found: False

| Metric | Count | Duration s | Extra |
| --- | ---: | ---: | --- |
| kernels | n/a | n/a | |
| graph trace | n/a | n/a | |
| runtime | n/a | n/a | |
| memcpy | n/a | n/a | bytes=n/a |
| NCCL kernels | n/a | n/a | |
| NCCL NVTX | n/a | n/a | range=None..None |

Top kernels:

| Name | Count | Duration s |
| --- | ---: | ---: |

Top runtime APIs:

| Name | Count | Duration s |
| --- | ---: | ---: |

