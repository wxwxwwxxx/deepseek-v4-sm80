# Nsight NVTX Range Summary: nsys_marlin_wna16_4096x128_bs4_np128_rank0.sqlite

- Parent NVTX: None
- Parent range count: 0
- Event summary mode: scan

## repeat:smoke_debug:0

- range count: 1
- total range duration s: 15.887
- kernel count: 58132, duration s: 5.87807
- graph trace count: 127, duration s: 8.20997
- CUDA graph node event count: n/a, duration s: n/a
- kernel graphNodeId non-null count: 0, duration s: 0
- runtime count: 101896, duration s: 14.3675
- NCCL kernel count: 88, duration s: 0.163036
- cudaGraphLaunch count: 127
- cudaLaunchKernel count: 57230
- memcpy count: 37839, bytes: 2374492380
- top kernels:
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.06687
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921976
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=1481, duration_s=0.400697
  - _hc_split_pre_kernel: count=86, duration_s=0.355535
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1649, duration_s=0.289507
  - void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8, (int)4, (bool)0, (int)4, (int)2, (bool)0>(const int4 *, const int4 *, int4 *, int4 *, const int4 *, const float *, const int4 *, const float *, const int4 *, const int *, const int *, const int *, const int *, const float *, int, bool, int, int, int, int, int *, bool, bool, bool): count=86, duration_s=0.23423
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3): count=631, duration_s=0.152617
  - ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn: count=212, duration_s=0.111347
- top non-graph-node kernels:
  - void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams): count=41, duration_s=2.06687
  - _indexer_bf16_logits_kernel: count=21, duration_s=0.921976
  - void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7): count=1481, duration_s=0.400697
  - _hc_split_pre_kernel: count=86, duration_s=0.355535
  - void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3): count=1649, duration_s=0.289507
  - void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int)8, (int)4, (bool)0, (int)4, (int)2, (bool)0>(const int4 *, const int4 *, int4 *, int4 *, const int4 *, const float *, const int4 *, const float *, const int4 *, const int *, const int *, const int *, const int *, const float *, int, bool, int, int, int, int, int *, bool, bool, bool): count=86, duration_s=0.23423
  - void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3): count=631, duration_s=0.152617
  - ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn: count=212, duration_s=0.111347
- top runtime:
  - cudaDeviceSynchronize_v3020: count=257, duration_s=7.94199
  - cudaStreamSynchronize_v3020: count=1046, duration_s=5.12614
  - cudaLaunchKernel_v7000: count=57230, duration_s=0.388717
  - cudaGraphLaunch_v10000: count=127, duration_s=0.37317
  - cudaMemcpyAsync_v3020: count=37839, duration_s=0.290361
  - cuModuleLoadData: count=16, duration_s=0.215858
  - cudaMalloc_v3020: count=16, duration_s=0.0114041
  - cuLaunchKernelEx: count=902, duration_s=0.00655155

