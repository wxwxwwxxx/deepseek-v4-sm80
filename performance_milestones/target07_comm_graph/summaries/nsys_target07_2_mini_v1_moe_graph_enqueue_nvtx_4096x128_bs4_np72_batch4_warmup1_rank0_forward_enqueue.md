# Nsight NVTX Range Summary: nsys_target07_2_mini_v1_moe_graph_enqueue_nvtx_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Parent NVTX: repeat:decode_throughput_bs8:0
- Parent range count: 1

## batch_forward:prefill:bs4:padded4

- range count: 1
- total range duration s: 21.2914
- kernel count: 26772, duration s: 21.2065
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29331, duration s: 20.8314
- NCCL kernel count: 88, duration s: 0.150213
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26086
- memcpy count: 128, bytes: 178389340
- top kernels:
  - _grouped_fp4_w13_kernel: count=43, duration_s=8.90223
  - _grouped_fp4_linear_kernel: count=43, duration_s=6.08668
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.06621
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921807
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=259, duration_s=0.577442
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2149, duration_s=0.472231
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1821, duration_s=0.290766
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=173, duration_s=0.178804
- top runtime:
  - cudaStreamSynchronize_v3020: count=83, duration_s=20.1625
  - cudaDeviceSynchronize_v3020: count=1, duration_s=0.468655
  - cudaLaunchKernel_v7000: count=26086, duration_s=0.187656
  - cuLaunchKernelEx: count=686, duration_s=0.00594532
  - cudaEventRecord_v3020: count=352, duration_s=0.00164886
  - cudaMemcpyAsync_v3020: count=128, duration_s=0.00150782
  - cudaEventQuery_v3020: count=473, duration_s=0.00109011
  - cudaEventRecordWithFlags_v11010: count=89, duration_s=0.000500198

## batch_forward_enqueue:prefill:bs4:padded4

- range count: 1
- total range duration s: 20.8227
- kernel count: 26464, duration s: 20.73
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29302, duration s: 20.3627
- NCCL kernel count: 85, duration s: 0.146713
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26086
- memcpy count: 127, bytes: 178389324
- top kernels:
  - _grouped_fp4_w13_kernel: count=42, duration_s=8.69525
  - _grouped_fp4_linear_kernel: count=42, duration_s=5.9458
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=40, duration_s=1.98833
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921807
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=253, duration_s=0.564043
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2119, duration_s=0.463446
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1812, duration_s=0.290705
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=169, duration_s=0.17466
- top runtime:
  - cudaStreamSynchronize_v3020: count=83, duration_s=20.1625
  - cudaLaunchKernel_v7000: count=26086, duration_s=0.187656
  - cuLaunchKernelEx: count=686, duration_s=0.00594532
  - cudaEventRecord_v3020: count=352, duration_s=0.00164886
  - cudaMemcpyAsync_v3020: count=128, duration_s=0.00150782
  - cudaEventQuery_v3020: count=463, duration_s=0.00107151
  - cudaEventRecordWithFlags_v11010: count=89, duration_s=0.000500198
  - cudaEventCreateWithFlags_v3020: count=88, duration_s=0.000495207

## batch_forward:decode:bs4:padded4

- range count: 127
- total range duration s: 21.572
- kernel count: 1905, duration s: 0.0091861
- graph trace count: 127, duration s: 21.4308
- kernel graphNodeId non-null count: 0
- runtime count: 5342, duration s: 21.3982
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 1778
- memcpy count: 2794, bytes: 3801872
- top kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (int)4>>(T3): count=127, duration_s=0.00507802
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=1016, duration_s=0.00176586
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=254, duration_s=0.000638196
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000633882
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000500282
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000314591
  - _copy_masked_compressed_locs_kernel: count=127, duration_s=0.000255263
- top runtime:
  - cudaDeviceSynchronize_v3020: count=127, duration_s=20.4622
  - cudaGraphLaunch_v10000: count=127, duration_s=0.884176
  - cudaMemcpyAsync_v3020: count=2794, duration_s=0.0272354
  - cudaLaunchKernel_v7000: count=1778, duration_s=0.021419
  - cuLaunchKernelEx: count=127, duration_s=0.00151188
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.0011603
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.000504646
  - cudaEventQuery_v3020: count=4, duration_s=8.657e-06

## batch_forward_enqueue:decode:bs4:padded4

- range count: 127
- total range duration s: 1.10009
- kernel count: 1524, duration s: 0.00313455
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 5215, duration s: 0.936018
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 1778
- memcpy count: 2667, bytes: 3799840
- top kernels:
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=1016, duration_s=0.00176586
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000500282
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000314591
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=127, duration_s=0.000298554
  - _copy_masked_compressed_locs_kernel: count=127, duration_s=0.000255263
- top runtime:
  - cudaGraphLaunch_v10000: count=127, duration_s=0.884176
  - cudaMemcpyAsync_v3020: count=2794, duration_s=0.0272354
  - cudaLaunchKernel_v7000: count=1778, duration_s=0.021419
  - cuLaunchKernelEx: count=127, duration_s=0.00151188
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.0011603
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.000504646
  - cudaEventQuery_v3020: count=4, duration_s=8.657e-06
  - cudaThreadExchangeStreamCaptureMode_v10010: count=4, duration_s=2.174e-06

