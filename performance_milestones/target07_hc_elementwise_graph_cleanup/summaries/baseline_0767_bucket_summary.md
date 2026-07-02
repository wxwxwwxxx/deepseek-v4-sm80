# TARGET 07.67 Bucket Summary: nsys_target0767_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`3.155779`, decode_envelope_s=`3.591306`

## repeat_decode_forward_envelope

- wall_s=`3.591306`, kernel_s=`2.959936`, runtime_s=`3.146304`

| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sparse_attention` | 21590 | `0.118089` | 3.99% | 21590 | 170 |
| `nccl_communication` | 11176 | `0.338786` | 11.45% | 11176 | 88 |
| `direct_copy_layout` | 192641 | `0.557626` | 18.84% | 186055 | 1465 |
| `hc_elementwise` | 212719 | `0.536306` | 18.12% | 208026 | 1638 |
| `moe_routed_backend` | 43688 | `0.300138` | 10.14% | 43688 | 344 |
| `projection_gemm` | 100965 | `0.778887` | 26.31% | 100965 | 795 |
| `fp8_activation_quant` | 35433 | `0.076019` | 2.57% | 35433 | 279 |
| `index_cache_topk` | 21646 | `0.132758` | 4.49% | 20828 | 164 |
| `rmsnorm_rope_compress_store` | 24638 | `0.072145` | 2.44% | 24638 | 194 |
| `sampling_logits_other` | 5334 | `0.025479` | 0.86% | 5334 | 42 |
| `other` | 11301 | `0.023703` | 0.80% | 11049 | 87 |

## repeat_prefill_forward

- wall_s=`5.172203`, kernel_s=`4.371656`, runtime_s=`4.181690`

| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sparse_attention` | 43 | `2.108144` | 48.22% | 0 | 0 |
| `nccl_communication` | 88 | `0.306921` | 7.02% | 0 | 0 |
| `direct_copy_layout` | 1571 | `0.338660` | 7.75% | 0 | 0 |
| `hc_elementwise` | 1947 | `0.674712` | 15.43% | 0 | 0 |
| `moe_routed_backend` | 344 | `0.262601` | 6.01% | 0 | 0 |
| `projection_gemm` | 580 | `0.369152` | 8.44% | 0 | 0 |
| `fp8_activation_quant` | 279 | `0.100537` | 2.30% | 0 | 0 |
| `index_cache_topk` | 288 | `0.122377` | 2.80% | 0 | 0 |
| `rmsnorm_rope_compress_store` | 214 | `0.053728` | 1.23% | 0 | 0 |
| `sampling_logits_other` | 62 | `0.008501` | 0.19% | 0 | 0 |
| `other` | 209 | `0.026323` | 0.60% | 0 | 0 |

## repeat

- wall_s=`10.122650`, kernel_s=`7.401353`, runtime_s=`7.821819`

| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sparse_attention` | 21633 | `2.226233` | 30.08% | 21590 | 170 |
| `nccl_communication` | 11264 | `0.645707` | 8.72% | 11176 | 88 |
| `direct_copy_layout` | 226640 | `0.964392` | 13.03% | 186055 | 1465 |
| `hc_elementwise` | 214824 | `1.211817` | 16.37% | 208026 | 1638 |
| `moe_routed_backend` | 44032 | `0.562739` | 7.60% | 43688 | 344 |
| `projection_gemm` | 101545 | `1.148039` | 15.51% | 100965 | 795 |
| `fp8_activation_quant` | 35712 | `0.176556` | 2.39% | 35433 | 279 |
| `index_cache_topk` | 22088 | `0.255946` | 3.46% | 20828 | 164 |
| `rmsnorm_rope_compress_store` | 24852 | `0.125874` | 1.70% | 24638 | 194 |
| `sampling_logits_other` | 5396 | `0.033979` | 0.46% | 5334 | 42 |
| `other` | 11527 | `0.050071` | 0.68% | 11049 | 87 |

## Bucket Phase

| Bucket | Prefill kernel s | Decode envelope kernel s | Phase |
| --- | ---: | ---: | --- |
| `sparse_attention` | `2.108144` | `0.118089` | prefill-heavy |
| `nccl_communication` | `0.306921` | `0.338786` | mixed |
| `direct_copy_layout` | `0.338660` | `0.557626` | mixed |
| `hc_elementwise` | `0.674712` | `0.536306` | mixed |
| `moe_routed_backend` | `0.262601` | `0.300138` | mixed |
| `projection_gemm` | `0.369152` | `0.778887` | mixed |
| `fp8_activation_quant` | `0.100537` | `0.076019` | mixed |
| `index_cache_topk` | `0.122377` | `0.132758` | mixed |
| `rmsnorm_rope_compress_store` | `0.053728` | `0.072145` | mixed |
| `sampling_logits_other` | `0.008501` | `0.025479` | mixed |
| `other` | `0.026323` | `0.023703` | mixed |

## Top Decode Kernels

| Kernel name | Count | Kernel s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.230499` | 16510 | 130 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 49403 | `0.228213` | 49403 | 389 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.168133` | 5461 | 43 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.165361` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.159949` | 16256 | 128 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.131619` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.111860` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.097983` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIterato...` | 27432 | `0.078746` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 35433 | `0.076019` | 35433 | 279 |
| `_hc_split_pre_kernel` | 10922 | `0.070713` | 10922 | 86 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, ...` | 13716 | `0.064230` | 13716 | 108 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 24384 | `0.058359` | 24384 | 192 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.052565` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0>(cubl...` | 16510 | `0.049029` | 16510 | 130 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &...` | 10922 | `0.046462` | 10922 | 86 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 20815 | `0.046369` | 19177 | 151 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044628` | 5461 | 43 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T2, at::cud...` | 5080 | `0.044231` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 16510 | `0.043957` | 16510 | 130 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.043766` | 2667 | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)...` | 24511 | `0.040899` | 24511 | 193 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039330` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28439 | `0.038699` | 27305 | 215 |
| `_hc_post_kernel` | 10922 | `0.036269` | 10922 | 86 |
| `_moe_route_fill_kernel` | 10922 | `0.034411` | 10922 | 86 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.034370` | 5461 | 43 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)64>(T1 *, a...` | 10668 | `0.034151` | 10668 | 84 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.034131` | 5461 | 43 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at::cuda::det...` | 5080 | `0.030373` | 5080 | 40 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 19304 | `0.030240` | 19304 | 152 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_func...` | 5588 | `0.028860` | 5588 | 44 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTen...` | 127 | `0.026769` | 127 | 1 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, const T1 *, T...` | 5334 | `0.025479` | 5334 | 42 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2667 | `0.024236` | 2667 | 21 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIterato...` | 10541 | `0.024182` | 10541 | 83 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBa...` | 13716 | `0.023491` | 13716 | 108 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator (...` | 13716 | `0.023456` | 13716 | 108 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::na...` | 13843 | `0.022533` | 13843 | 109 |
| `_rotary_tail_kernel` | 10795 | `0.022170` | 10795 | 85 |

## Top Repeat Kernels

| Kernel name | Count | Kernel s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.065784` | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 49812 | `0.520708` | 49403 | 389 |
| `_hc_split_pre_kernel` | 11008 | `0.424397` | 10922 | 86 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.372526` | 5588 | 44 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.267851` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int...` | 86 | `0.232956` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.230499` | 16510 | 130 |
| `_fp8_activation_quantize_kernel` | 35712 | `0.176556` | 35433 | 279 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.159949` | 16256 | 128 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.131619` | 5461 | 43 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBa...` | 13824 | `0.130129` | 13716 | 108 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, ...` | 13824 | `0.117757` | 13716 | 108 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.111860` | 10922 | 86 |
| `_hc_post_kernel` | 11008 | `0.105966` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.097983` | 5461 | 43 |
| `ampere_sgemm_32x128_tn` | 87 | `0.096932` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.088898` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIterato...` | 27648 | `0.087644` | 27432 | 216 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.083926` | 2667 | 21 |
| `ampere_sgemm_128x64_tn` | 43 | `0.083625` | 0 | 0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.076533` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scalar &, at::T...` | 33394 | `0.068945` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_func...` | 5632 | `0.059603` | 5588 | 44 |
| `_moe_route_fill_kernel` | 11008 | `0.059556` | 10922 | 86 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T2, at::cud...` | 5120 | `0.059324` | 5080 | 40 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &...` | 11008 | `0.059013` | 10922 | 86 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 24384 | `0.058359` | 24384 | 192 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5504 | `0.058231` | 5461 | 43 |
| `_rms_norm_bf16_kernel` | 16640 | `0.058214` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)...` | 24724 | `0.055191` | 24511 | 193 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.052565` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0>(cubl...` | 16597 | `0.049927` | 16510 | 130 |
| `_rotary_tail_kernel` | 10880 | `0.048179` | 10795 | 85 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 20972 | `0.047197` | 19177 | 151 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044628` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.043766` | 2667 | 21 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.042360` | 0 | 0 |
| `_indexer_fp8_quantize_fold_kernel` | 2688 | `0.041262` | 2667 | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)64>(T1 *, a...` | 10752 | `0.040108` | 10668 | 84 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039330` | 10668 | 84 |
