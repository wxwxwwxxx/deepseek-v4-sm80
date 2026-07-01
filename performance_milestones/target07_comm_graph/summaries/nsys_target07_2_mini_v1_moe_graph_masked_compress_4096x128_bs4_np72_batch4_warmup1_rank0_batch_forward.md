# Nsight NVTX Range Summary: nsys_target07_2_mini_v1_moe_graph_masked_compress_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Parent NVTX: repeat:decode_throughput_bs8:0
- Parent range count: 1

## batch_forward:decode:bs4:padded4

- range count: 127
- total range duration s: 21.5757
- kernel count: 3302, duration s: 0.0121472
- graph trace count: 127, duration s: 21.4271
- kernel graphNodeId non-null count: 0
- runtime count: 6993, duration s: 21.4162
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 3302
- memcpy count: 3048, bytes: 3805936
- top kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (int)4>>(T3): count=127, duration_s=0.00508107
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=1270, duration_s=0.00220448
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=254, duration_s=0.000642266
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000641882
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int, int) (instance 1)]>, std::array<char *, (unsigned long)2>>(int, T2, T3): count=254, duration_s=0.000623673
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int, int, at::native::remainder_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int, int) (instance 1)]>, std::array<char *, (unsigned long)2>>(int, T2, T3): count=254, duration_s=0.000592087
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(bool, int, int) (instance 1)], std::array<char *, (unsigned long)4>>(int, T2, T3): count=254, duration_s=0.000556824
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000499703
- top runtime:
  - cudaDeviceSynchronize_v3020: count=127, duration_s=20.6128
  - cudaGraphLaunch_v10000: count=127, duration_s=0.748123
  - cudaLaunchKernel_v7000: count=3302, duration_s=0.0276189
  - cudaMemcpyAsync_v3020: count=3048, duration_s=0.0263162
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.000933176
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.000415725
  - cudaEventQuery_v3020: count=4, duration_s=1.0561e-05
  - cudaThreadExchangeStreamCaptureMode_v10010: count=4, duration_s=2.675e-06

## batch_forward:prefill:bs4:padded4

- range count: 1
- total range duration s: 21.2802
- kernel count: 26772, duration s: 21.2042
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29346, duration s: 20.858
- NCCL kernel count: 88, duration s: 0.149931
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26086
- memcpy count: 128, bytes: 178389340
- top kernels:
  - _grouped_fp4_w13_kernel: count=43, duration_s=8.90172
  - _grouped_fp4_linear_kernel: count=43, duration_s=6.08583
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.06588
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921687
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=259, duration_s=0.577407
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2149, duration_s=0.472206
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1821, duration_s=0.290734
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=173, duration_s=0.178833
- top runtime:
  - cudaStreamSynchronize_v3020: count=83, duration_s=20.204
  - cudaDeviceSynchronize_v3020: count=1, duration_s=0.46973
  - cudaLaunchKernel_v7000: count=26086, duration_s=0.173313
  - cuLaunchKernelEx: count=686, duration_s=0.00488807
  - cudaEventQuery_v3020: count=478, duration_s=0.00146674
  - cudaEventRecord_v3020: count=352, duration_s=0.00142919
  - cudaMemcpyAsync_v3020: count=128, duration_s=0.00125348
  - cudaThreadExchangeStreamCaptureMode_v10010: count=608, duration_s=0.00050969

