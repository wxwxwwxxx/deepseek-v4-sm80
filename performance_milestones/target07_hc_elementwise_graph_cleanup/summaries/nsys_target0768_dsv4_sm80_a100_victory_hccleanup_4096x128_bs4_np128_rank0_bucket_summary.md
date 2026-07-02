# TARGET 07.67 Bucket Summary: nsys_target0768_dsv4_sm80_a100_victory_hccleanup_4096x128_bs4_np128_rank0.sqlite

- Requested repeat NVTX: `repeat:decode_throughput_bs8:0`
- Repeat range found: `True`
- Repeat child ranges: prefill_forward=1, decode_forward=127, decode_forward_sum_s=`4.412819`, decode_envelope_s=`4.849656`

## repeat_decode_forward_envelope

- wall_s=`4.849656`, kernel_s=`2.902325`, runtime_s=`4.411697`

| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sparse_attention` | 21590 | `0.117665` | 4.05% | 21590 | 170 |
| `nccl_communication` | 11176 | `0.338458` | 11.66% | 11176 | 88 |
| `direct_copy_layout` | 181719 | `0.520151` | 17.92% | 175133 | 1379 |
| `hc_elementwise` | 169031 | `0.519303` | 17.89% | 164338 | 1294 |
| `moe_routed_backend` | 43688 | `0.299245` | 10.31% | 43688 | 344 |
| `projection_gemm` | 100965 | `0.779055` | 26.84% | 100965 | 795 |
| `fp8_activation_quant` | 35433 | `0.075848` | 2.61% | 35433 | 279 |
| `index_cache_topk` | 21646 | `0.131100` | 4.52% | 20828 | 164 |
| `rmsnorm_rope_compress_store` | 24638 | `0.072346` | 2.49% | 24638 | 194 |
| `sampling_logits_other` | 5334 | `0.025476` | 0.88% | 5334 | 42 |
| `other` | 11301 | `0.023677` | 0.82% | 11049 | 87 |

## repeat_prefill_forward

- wall_s=`6.906223`, kernel_s=`3.996865`, runtime_s=`3.801470`

| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sparse_attention` | 43 | `2.126883` | 53.21% | 0 | 0 |
| `nccl_communication` | 88 | `0.152404` | 3.81% | 0 | 0 |
| `direct_copy_layout` | 1485 | `0.221424` | 5.54% | 0 | 0 |
| `hc_elementwise` | 1603 | `0.545750` | 13.65% | 0 | 0 |
| `moe_routed_backend` | 344 | `0.263838` | 6.60% | 0 | 0 |
| `projection_gemm` | 580 | `0.372519` | 9.32% | 0 | 0 |
| `fp8_activation_quant` | 279 | `0.101449` | 2.54% | 0 | 0 |
| `index_cache_topk` | 288 | `0.123403` | 3.09% | 0 | 0 |
| `rmsnorm_rope_compress_store` | 214 | `0.054148` | 1.35% | 0 | 0 |
| `sampling_logits_other` | 62 | `0.008523` | 0.21% | 0 | 0 |
| `other` | 209 | `0.026524` | 0.66% | 0 | 0 |

## repeat

- wall_s=`13.162013`, kernel_s=`6.969413`, runtime_s=`8.734684`

| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sparse_attention` | 21633 | `2.244549` | 32.21% | 21590 | 170 |
| `nccl_communication` | 11264 | `0.490861` | 7.04% | 11176 | 88 |
| `direct_copy_layout` | 215632 | `0.810142` | 11.62% | 175133 | 1379 |
| `hc_elementwise` | 170792 | `1.065850` | 15.29% | 164338 | 1294 |
| `moe_routed_backend` | 44032 | `0.563083` | 8.08% | 43688 | 344 |
| `projection_gemm` | 101545 | `1.151574` | 16.52% | 100965 | 795 |
| `fp8_activation_quant` | 35712 | `0.177298` | 2.54% | 35433 | 279 |
| `index_cache_topk` | 22088 | `0.255319` | 3.66% | 20828 | 164 |
| `rmsnorm_rope_compress_store` | 24852 | `0.126494` | 1.81% | 24638 | 194 |
| `sampling_logits_other` | 5396 | `0.033999` | 0.49% | 5334 | 42 |
| `other` | 11527 | `0.050245` | 0.72% | 11049 | 87 |

## Bucket Phase

| Bucket | Prefill kernel s | Decode envelope kernel s | Phase |
| --- | ---: | ---: | --- |
| `sparse_attention` | `2.126883` | `0.117665` | prefill-heavy |
| `nccl_communication` | `0.152404` | `0.338458` | mixed |
| `direct_copy_layout` | `0.221424` | `0.520151` | mixed |
| `hc_elementwise` | `0.545750` | `0.519303` | mixed |
| `moe_routed_backend` | `0.263838` | `0.299245` | mixed |
| `projection_gemm` | `0.372519` | `0.779055` | mixed |
| `fp8_activation_quant` | `0.101449` | `0.075848` | mixed |
| `index_cache_topk` | `0.123403` | `0.131100` | mixed |
| `rmsnorm_rope_compress_store` | `0.054148` | `0.072346` | mixed |
| `sampling_logits_other` | `0.008523` | `0.025476` | mixed |
| `other` | `0.026524` | `0.023677` | mixed |

## Top Decode Kernels

| Kernel name | Count | Kernel s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.230957` | 16510 | 130 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 38481 | `0.191869` | 38481 | 303 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5461 | `0.169537` | 5461 | 43 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5588 | `0.163506` | 5588 | 44 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.159701` | 16256 | 128 |
| `_hc_prenorm_split_pre_kernel` | 10922 | `0.144871` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.131962` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.111898` | 10922 | 86 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.096824` | 5461 | 43 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIterato...` | 27432 | `0.078110` | 27432 | 216 |
| `_fp8_activation_quantize_kernel` | 35433 | `0.075848` | 35433 | 279 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 24384 | `0.058304` | 24384 | 192 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.052531` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0>(cubl...` | 16510 | `0.049077` | 16510 | 130 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &...` | 10922 | `0.046446` | 10922 | 86 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 20815 | `0.046283` | 19177 | 151 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044666` | 5461 | 43 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T2, at::cud...` | 5080 | `0.044346` | 5080 | 40 |
| `_rms_norm_bf16_kernel` | 16510 | `0.043839` | 16510 | 130 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.043614` | 2667 | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)...` | 24511 | `0.041208` | 24511 | 193 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039145` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28439 | `0.038665` | 27305 | 215 |
| `_hc_layer_input_kernel` | 10922 | `0.038542` | 10922 | 86 |
| `_hc_post_kernel` | 10922 | `0.036981` | 10922 | 86 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | 5461 | `0.034418` | 5461 | 43 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)64>(T1 *, a...` | 10668 | `0.034040` | 10668 | 84 |
| `_moe_route_fill_kernel` | 10922 | `0.033987` | 10922 | 86 |
| `_sparse_splitk_bf16_combine_kernel` | 5461 | `0.033854` | 5461 | 43 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at::cuda::det...` | 5080 | `0.030336` | 5080 | 40 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_func...` | 5588 | `0.029102` | 5588 | 44 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTen...` | 127 | `0.026780` | 127 | 1 |
| `void at::native::<unnamed>::cunn_SpatialSoftMaxForward<float, float, float, int, at::native::<unnamed>::SoftMaxForwardEpilogue>(T3 *, const T1 *, T...` | 5334 | `0.025476` | 5334 | 42 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIterato...` | 10541 | `0.024161` | 10541 | 83 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | 2667 | `0.022644` | 2667 | 21 |
| `_rotary_tail_kernel` | 10795 | `0.022155` | 10795 | 85 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float...` | 13462 | `0.021913` | 13462 | 106 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float...` | 5334 | `0.021194` | 5334 | 42 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scala...` | 11677 | `0.021133` | 10795 | 85 |
| `_moe_route_offsets_kernel` | 10922 | `0.019778` | 10922 | 86 |

## Top Repeat Kernels

| Kernel name | Count | Kernel s | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void <unnamed>::sparse_attention_kernel<(bool)1>(<unnamed>::SparseAttentionParams)` | 41 | `2.080459` | 0 | 0 |
| `_hc_layer_input_kernel` | 11008 | `0.395147` | 10922 | 86 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 38804 | `0.366907` | 38481 | 303 |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5504 | `0.268137` | 5461 | 43 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)4, (int...` | 86 | `0.235510` | 0 | 0 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 16510 | `0.230957` | 16510 | 130 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 5632 | `0.217271` | 5588 | 44 |
| `_fp8_activation_quantize_kernel` | 35712 | `0.177298` | 35433 | 279 |
| `_hc_prenorm_split_pre_kernel` | 11008 | `0.171647` | 10922 | 86 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | 16256 | `0.159701` | 16256 | 128 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.131962` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 10922 | `0.111898` | 10922 | 86 |
| `_hc_post_kernel` | 11008 | `0.106785` | 10922 | 86 |
| `ampere_sgemm_32x128_tn` | 87 | `0.097810` | 0 | 0 |
| `void marlin_moe_wna16::Marlin<(long)1125899906909960, (long)562949953487106, (long)1125899906909960, (long)2814749767106568, (int)128, (int)1, (int...` | 5461 | `0.096824` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 169 | `0.089702` | 0 | 0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIterato...` | 27648 | `0.087058` | 27432 | 216 |
| `ampere_sgemm_128x64_tn` | 43 | `0.084500` | 0 | 0 |
| `_indexer_fp8_paged_logits_kernel` | 2688 | `0.084474` | 2667 | 21 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 150 | `0.077220` | 0 | 0 |
| `void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &, const c10::Scalar &, const c10::Scalar &, at::T...` | 33394 | `0.069392` | 0 | 0 |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_func...` | 5632 | `0.060055` | 5588 | 44 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T2, at::cud...` | 5120 | `0.059439` | 5080 | 40 |
| `_moe_route_fill_kernel` | 11008 | `0.059397` | 10922 | 86 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &...` | 11008 | `0.059096` | 10922 | 86 |
| `_q_kv_norm_rope_cache_bf16_kernel` | 5504 | `0.058824` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 24384 | `0.058304` | 24384 | 192 |
| `_rms_norm_bf16_kernel` | 16640 | `0.058134` | 16510 | 130 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)...` | 24724 | `0.055503` | 24511 | 193 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | 5461 | `0.052531` | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0>(cubl...` | 16597 | `0.049983` | 16510 | 130 |
| `_rotary_tail_kernel` | 10880 | `0.048364` | 10795 | 85 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() ...` | 20972 | `0.047119` | 19177 | 151 |
| `void <unnamed>::sparse_attention_kernel<(bool)0>(<unnamed>::SparseAttentionParams)` | 2 | `0.046424` | 0 | 0 |
| `_sparse_splitk_bf16_split_kernel` | 5461 | `0.044666` | 5461 | 43 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 2667 | `0.043614` | 2667 | 21 |
| `_indexer_fp8_quantize_fold_kernel` | 2688 | `0.041553` | 2667 | 21 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)64>(T1 *, a...` | 10752 | `0.040031` | 10668 | 84 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 28677 | `0.039159` | 27305 | 215 |
| `_sparse_bf16_gather_with_mask_kernel` | 10668 | `0.039145` | 10668 | 84 |
