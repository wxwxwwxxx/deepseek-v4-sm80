# Nsight NVTX Range Summary: nsys_target07_2_mini_v1_moe_graph_4096x128_bs4_np72_batch4_batchnvtx_warmup1_rank0.sqlite

- Parent NVTX: repeat:decode_throughput_bs8:0
- Parent range count: 1

## batch_forward:prefill:bs4:padded4

- range count: 1
- total range duration s: 21.291
- kernel count: 26982, duration s: 21.2073
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29674, duration s: 20.8499
- NCCL kernel count: 88, duration s: 0.150227
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26317
- memcpy count: 191, bytes: 178389529
- top kernels:
  - _grouped_fp4_w13_kernel: count=43, duration_s=8.90332
  - _grouped_fp4_linear_kernel: count=43, duration_s=6.08531
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.06636
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921828
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=259, duration_s=0.57745
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2149, duration_s=0.47221
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1821, duration_s=0.290731
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=173, duration_s=0.178682
- top runtime:
  - cudaStreamSynchronize_v3020: count=146, duration_s=20.1828
  - cudaDeviceSynchronize_v3020: count=1, duration_s=0.469565
  - cudaLaunchKernel_v7000: count=26317, duration_s=0.185396
  - cuLaunchKernelEx: count=665, duration_s=0.00549855
  - cudaMemcpyAsync_v3020: count=191, duration_s=0.0019074
  - cudaEventRecord_v3020: count=352, duration_s=0.00150228
  - cudaEventQuery_v3020: count=475, duration_s=0.00126776
  - cudaEventRecordWithFlags_v11010: count=89, duration_s=0.000465945

## batch_forward:decode:bs4:padded4

- range count: 127
- total range duration s: 21.3096
- kernel count: 1778, duration s: 0.00895787
- graph trace count: 127, duration s: 21.1886
- kernel graphNodeId non-null count: 0
- runtime count: 5342, duration s: 21.172
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 1778
- memcpy count: 2794, bytes: 3801872
- top kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (int)4>>(T3): count=127, duration_s=0.00504851
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3): count=1016, duration_s=0.0017604
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000659095
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=254, duration_s=0.000647988
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000516473
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000325405
- top runtime:
  - cudaDeviceSynchronize_v3020: count=127, duration_s=19.9293
  - cudaGraphLaunch_v10000: count=127, duration_s=1.19166
  - cudaMemcpyAsync_v3020: count=2794, duration_s=0.0282887
  - cudaLaunchKernel_v7000: count=1778, duration_s=0.0199603
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.00145885
  - cudaEventCreateWithFlags_v3020: count=127, duration_s=0.00090951
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.000394908
  - cudaEventQuery_v3020: count=4, duration_s=1.4258e-05

