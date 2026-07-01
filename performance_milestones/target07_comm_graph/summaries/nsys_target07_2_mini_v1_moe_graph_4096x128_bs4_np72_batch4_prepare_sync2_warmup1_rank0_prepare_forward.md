# Nsight NVTX Range Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_prepare_sync2_warmup1_rank0.sqlite

- Parent NVTX: repeat:decode_throughput_bs8:0
- Parent range count: 1

## batch_prepare:prefill:bs4

- range count: 1
- total range duration s: 1.28132
- kernel count: 32394, duration s: 0.0564539
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 64713, duration s: 0.431597
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 32394
- memcpy count: 32267, bytes: 26763952
- top kernels:
  - void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scalar &, at::Tensor &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(long) (instance 1)]>(T1, T2, function_traits<T2>::result_type *): count=32250, duration_s=0.0552591
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=5, duration_s=0.000189407
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(bool, int, int) (instance 1)], std::array<char *, (unsigned long)4>>(int, T2, T3): count=6, duration_s=0.000158398
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(long) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=14, duration_s=0.000128606
  - void at_cuda_detail::cub::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::DeviceRadixSortPolicy<long, at_cuda_detail::cub::NullType, unsigned long long>::Policy900, (bool)0, long, at_cuda_detail::cub::NullType, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(T7 *, T7 *, T5 *, const T5 *, T3 *, const T3 *, T4 *, const T4 *, T6, int, int, T8): count=16, duration_s=0.00010691
  - void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3): count=6, duration_s=7.0143e-05
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=12, duration_s=6.6528e-05
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(long) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=5, duration_s=6.0255e-05
- top runtime:
  - cudaMemcpyAsync_v3020: count=32267, duration_s=0.244459
  - cudaLaunchKernel_v7000: count=32394, duration_s=0.186702
  - cudaStreamSynchronize_v3020: count=12, duration_s=0.000223689
  - cudaMemsetAsync_v3020: count=20, duration_s=9.8753e-05
  - cudaDeviceSynchronize_v3020: count=1, duration_s=6.5603e-05
  - cudaEventQuery_v3020: count=7, duration_s=2.5167e-05
  - cudaEventRecordWithFlags_v11010: count=6, duration_s=1.9786e-05
  - cudaStreamIsCapturing_v10000: count=6, duration_s=3.366e-06

## batch_prepare:decode:bs4

- range count: 127
- total range duration s: 0.421094
- kernel count: 12093, duration s: 0.0291536
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 17376, duration s: 0.122767
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 12093
- memcpy count: 2576, bytes: 1150000
- top kernels:
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(long) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=1652, duration_s=0.00427515
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=635, duration_s=0.00377698
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=1143, duration_s=0.00203286
  - void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scalar &, at::Tensor &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(long) (instance 1)]>(T1, T2, function_traits<T2>::result_type *): count=1144, duration_s=0.00197121
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3): count=889, duration_s=0.00182584
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(long) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=635, duration_s=0.00171063
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(bool, int, int) (instance 1)], std::array<char *, (unsigned long)4>>(int, T2, T3): count=762, duration_s=0.00170654
  - void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3): count=762, duration_s=0.00168684
- top runtime:
  - cudaLaunchKernel_v7000: count=12093, duration_s=0.0835442
  - cudaMemcpyAsync_v3020: count=2576, duration_s=0.0257687
  - cudaStreamSynchronize_v3020: count=923, duration_s=0.00748819
  - cudaEventQuery_v3020: count=637, duration_s=0.00241902
  - cudaEventRecordWithFlags_v11010: count=510, duration_s=0.00175621
  - cudaDeviceSynchronize_v3020: count=127, duration_s=0.00147097
  - cudaStreamIsCapturing_v10000: count=510, duration_s=0.000319595

## batch_forward:prefill:bs4:padded4

- range count: 1
- total range duration s: 21.2965
- kernel count: 26982, duration s: 21.2072
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29671, duration s: 20.8355
- NCCL kernel count: 88, duration s: 0.151181
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26317
- memcpy count: 191, bytes: 178389529
- top kernels:
  - _grouped_fp4_w13_kernel: count=43, duration_s=8.90241
  - _grouped_fp4_linear_kernel: count=43, duration_s=6.08545
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.0662
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921806
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=259, duration_s=0.577434
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2149, duration_s=0.47219
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1821, duration_s=0.290758
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=173, duration_s=0.178752
- top runtime:
  - cudaStreamSynchronize_v3020: count=146, duration_s=20.1574
  - cudaDeviceSynchronize_v3020: count=1, duration_s=0.469034
  - cudaLaunchKernel_v7000: count=26317, duration_s=0.196959
  - cuLaunchKernelEx: count=665, duration_s=0.00547522
  - cudaMemcpyAsync_v3020: count=191, duration_s=0.00199722
  - cudaEventRecord_v3020: count=352, duration_s=0.00143954
  - cudaEventQuery_v3020: count=474, duration_s=0.00107672
  - cudaEventRecordWithFlags_v11010: count=89, duration_s=0.000483406

## batch_forward:decode:bs4:padded4

- range count: 127
- total range duration s: 21.3045
- kernel count: 1778, duration s: 0.0089537
- graph trace count: 127, duration s: 21.1876
- kernel graphNodeId non-null count: 0
- runtime count: 5342, duration s: 21.1614
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 1778
- memcpy count: 2794, bytes: 3801872
- top kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (int)4>>(T3): count=127, duration_s=0.00504813
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=1016, duration_s=0.0017607
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000657045
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=254, duration_s=0.000645303
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000516664
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000325849
- top runtime:
  - cudaDeviceSynchronize_v3020: count=127, duration_s=20.225
  - cudaGraphLaunch_v10000: count=127, duration_s=0.887218
  - cudaMemcpyAsync_v3020: count=2794, duration_s=0.0269872
  - cudaLaunchKernel_v7000: count=1778, duration_s=0.019372
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.00150012
  - cudaEventCreateWithFlags_v3020: count=127, duration_s=0.000963105
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.000404894
  - cudaEventQuery_v3020: count=4, duration_s=9.077e-06

