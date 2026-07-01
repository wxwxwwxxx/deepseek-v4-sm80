# Nsight NVTX Range Summary: nsys_target07_2_mini_v1_moe_graph_bound_metadata_4096x128_bs4_np72_batch4_warmup1_rank0.sqlite

- Parent NVTX: repeat:decode_throughput_bs8:0
- Parent range count: 1

## batch_forward:prefill:bs4:padded4

- range count: 1
- total range duration s: 21.2829
- kernel count: 26772, duration s: 21.2046
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29358, duration s: 20.8453
- NCCL kernel count: 88, duration s: 0.151198
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26086
- memcpy count: 128, bytes: 178389340
- top kernels:
  - _grouped_fp4_w13_kernel: count=43, duration_s=8.90121
  - _grouped_fp4_linear_kernel: count=43, duration_s=6.08578
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.06576
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921649
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=259, duration_s=0.577382
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2149, duration_s=0.472201
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1821, duration_s=0.29075
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=173, duration_s=0.178755
- top runtime:
  - cudaStreamSynchronize_v3020: count=83, duration_s=20.1911
  - cudaDeviceSynchronize_v3020: count=1, duration_s=0.469471
  - cudaLaunchKernel_v7000: count=26086, duration_s=0.173539
  - cuLaunchKernelEx: count=686, duration_s=0.00538561
  - cudaEventRecord_v3020: count=352, duration_s=0.00153923
  - cudaMemcpyAsync_v3020: count=128, duration_s=0.00129281
  - cudaEventQuery_v3020: count=482, duration_s=0.00117546
  - cudaEventRecordWithFlags_v11010: count=89, duration_s=0.000424759

## batch_forward_enqueue:prefill:bs4:padded4

- range count: 1
- total range duration s: 20.8133
- kernel count: 26464, duration s: 20.7282
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 29320, duration s: 20.3758
- NCCL kernel count: 85, duration s: 0.147734
- cudaGraphLaunch count: 0
- cudaLaunchKernel count: 26086
- memcpy count: 127, bytes: 178389324
- top kernels:
  - _grouped_fp4_w13_kernel: count=42, duration_s=8.69423
  - _grouped_fp4_linear_kernel: count=42, duration_s=5.94487
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=40, duration_s=1.98788
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921649
  - void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=253, duration_s=0.563983
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=2119, duration_s=0.463415
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1812, duration_s=0.290688
  - void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, c10::BFloat16, (int)4, (int)4>>(T3): count=169, duration_s=0.174605
- top runtime:
  - cudaStreamSynchronize_v3020: count=83, duration_s=20.1911
  - cudaLaunchKernel_v7000: count=26086, duration_s=0.173539
  - cuLaunchKernelEx: count=686, duration_s=0.00538561
  - cudaEventRecord_v3020: count=352, duration_s=0.00153923
  - cudaMemcpyAsync_v3020: count=128, duration_s=0.00129281
  - cudaEventQuery_v3020: count=469, duration_s=0.00115458
  - cudaEventRecordWithFlags_v11010: count=89, duration_s=0.000424759
  - cudaStreamWaitEvent_v3020: count=264, duration_s=0.000337889

## batch_forward:decode:bs4:padded4

- range count: 127
- total range duration s: 21.5414
- kernel count: 1016, duration s: 0.00770612
- graph trace count: 127, duration s: 21.4362
- kernel graphNodeId non-null count: 0
- runtime count: 4199, duration s: 21.4054
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 889
- memcpy count: 2540, bytes: 3797808
- top kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (int)4>>(T3): count=127, duration_s=0.0050633
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000639767
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=254, duration_s=0.000638009
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000500249
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000315583
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<int>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000294717
  - _copy_masked_compressed_locs_kernel: count=127, duration_s=0.000254493
- top runtime:
  - cudaDeviceSynchronize_v3020: count=127, duration_s=20.6237
  - cudaGraphLaunch_v10000: count=127, duration_s=0.743085
  - cudaMemcpyAsync_v3020: count=2540, duration_s=0.0225635
  - cudaLaunchKernel_v7000: count=889, duration_s=0.0134948
  - cuLaunchKernelEx: count=127, duration_s=0.00114321
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.000956856
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.00043524
  - cudaEventQuery_v3020: count=4, duration_s=1.8683e-05

## batch_forward_enqueue:decode:bs4:padded4

- range count: 127
- total range duration s: 0.910513
- kernel count: 635, duration s: 0.00166606
- graph trace count: 0, duration s: 0
- kernel graphNodeId non-null count: 0
- runtime count: 4064, duration s: 0.781679
- NCCL kernel count: 0, duration s: 0
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 889
- memcpy count: 2413, bytes: 3795776
- top kernels:
  - void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3): count=127, duration_s=0.000500249
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000315583
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 3)]::operator ()() const::[lambda(int) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=127, duration_s=0.00030102
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<int>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=127, duration_s=0.000294717
  - _copy_masked_compressed_locs_kernel: count=127, duration_s=0.000254493
- top runtime:
  - cudaGraphLaunch_v10000: count=127, duration_s=0.743085
  - cudaMemcpyAsync_v3020: count=2540, duration_s=0.0225635
  - cudaLaunchKernel_v7000: count=889, duration_s=0.0134948
  - cuLaunchKernelEx: count=127, duration_s=0.00114321
  - cudaEventRecordWithFlags_v11010: count=127, duration_s=0.000956856
  - cudaStreamIsCapturing_v10000: count=254, duration_s=0.00043524

