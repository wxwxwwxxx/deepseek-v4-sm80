# Decode Metadata Nsight Split: nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0.sqlite

- repeat NVTX: `repeat:decode_throughput_bs8:0`
- decode forward ranges: `127`
- decode envelope wall s: `4.010320`

## Sub-Boundaries

| Sub-boundary | Kernel s | Count | Graph events | Graph nodes | Share of decode envelope |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_copy` | `0.731834` | 189732 | 188849 | 1487 | 18.25% |
| `index_elementwise_kernel` | `0.001985` | 411 | 0 | 0 | 0.05% |
| `CatArrayBatchedCopy` | `0.034106` | 10668 | 10668 | 84 | 0.85% |
| `gatherTopK` | `0.074760` | 10160 | 10160 | 80 | 1.86% |
| `arange_index_helper` | `0.025551` | 11807 | 11303 | 89 | 0.64% |
| `topk_lens_swa_compressed_index_assembly` | `0.067087` | 33962 | 32766 | 258 | 1.67% |
| `other_metadata_copy_cat_index` | `0.079118` | 18796 | 18796 | 148 | 1.97% |
| `total_selected_metadata_adjacent` | `1.014442` | 275536 | 272542 | 2146 | 25.30% |

## Top Kernels By Sub-Boundary

### `direct_copy`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[la...` | `0.379344` | 71247 | 71247 | 561 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::...` | `0.095792` | 21971 | 21971 | 173 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::ar...` | `0.082203` | 35433 | 35433 | 279 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::...` | `0.078814` | 27432 | 27432 | 216 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda...` | `0.046451` | 10922 | 10922 | 86 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[la...` | `0.043987` | 19933 | 19177 | 151 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[la...` | `0.005243` | 2794 | 2667 | 21 |

### `index_elementwise_kernel`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)...` | `0.001170` | 253 | 0 | 0 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(...` | `0.000684` | 127 | 0 | 0 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)...` | `0.000132` | 31 | 0 | 0 |

### `CatArrayBatchedCopy`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)64, (int)64>(T1 *, at::native:...` | `0.034106` | 10668 | 10668 | 84 |

### `gatherTopK`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T2, at::cuda::detail:...` | `0.044277` | 5080 | 5080 | 40 |
| `void at::native::bitonicSortKVInPlace<(int)2, (int)-1, (int)16, (int)16, float, long, at::native::GTOp<float, (bool)1>, unsigned int>(at::cuda::detail::Tenso...` | `0.030483` | 5080 | 5080 | 40 |

### `arange_index_helper`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` | `0.011992` | 5715 | 5715 | 45 |
| `void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)8, void at::native::_cuda_scatter_gather_internal_kernel<(bool)0, at::native::OpaqueType<(...` | `0.011691` | 5461 | 5461 | 43 |
| `void at_cuda_detail::cub::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::device_select_policy_hub<long, bool, int, (bool)0, (bool)0>::Policy900, at_cud...` | `0.000661` | 252 | 0 | 0 |
| `void at::native::<unnamed>::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, (int)2, (int)2, (int)-2>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda:...` | `0.000627` | 127 | 127 | 1 |
| `void at_cuda_detail::cub::DeviceReduceSingleTileKernel<at_cuda_detail::cub::DeviceReducePolicy<int, unsigned long long, cuda::std::__4::plus<void>>::Policy60...` | `0.000580` | 252 | 0 | 0 |

### `topk_lens_swa_compressed_index_assembly`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | `0.036942` | 27431 | 27305 | 215 |
| `void <unnamed>::topk_transform_global_lens_kernel<(unsigned int)512>(<unnamed>::TopKGlobalLensParams<T1>)` | `0.024234` | 2667 | 2667 | 21 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | `0.003620` | 2667 | 2667 | 21 |
| `void at_cuda_detail::cub::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, (bool)1>, int *>(T1, int, T2)` | `0.000431` | 252 | 0 | 0 |
| `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>, (int)4, TrivialOffsetCalculator<(int)0, unsi...` | `0.000413` | 252 | 0 | 0 |
| `_copy_masked_compressed_locs_kernel` | `0.000331` | 127 | 127 | 1 |
| `void at_cuda_detail::cub::DeviceScanKernel<at_cuda_detail::cub::DeviceScanPolicy<long, std::plus<long>>::Policy900, const long *, long *, at_cuda_detail::cub...` | `0.000312` | 126 | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | `0.000224` | 126 | 0 | 0 |
| `void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | `0.000222` | 126 | 0 | 0 |
| `void at_cuda_detail::cub::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, (bool)1>>(T1, int)` | `0.000212` | 126 | 0 | 0 |

### `other_metadata_copy_cat_index`

| Kernel | Kernel s | Count | Graph events | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| `_sparse_bf16_gather_with_mask_kernel` | `0.040156` | 10668 | 10668 | 84 |
| `_indexer_fp8_paged_logits_kernel` | `0.019394` | 2667 | 2667 | 21 |
| `_indexer_fp8_quantize_fold_kernel` | `0.007495` | 2667 | 2667 | 21 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | `0.006558` | 127 | 127 | 1 |
| `_indexer_fp8_paged_quant_store_kernel` | `0.005515` | 2667 | 2667 | 21 |

## NVTX Owner Breakdown

| Owner | Kernel s | Count | Dominant sub-boundaries |
| --- | ---: | ---: | --- |
| `batch_forward:decode:bs4:padded4` | `0.653641` | 171526 | direct_copy=0.4713s, other_metadata_copy_cat_index=0.0536s, gatherTopK=0.0498s |
| `batch_forward_enqueue:decode:bs4:padded4` | `0.354810` | 101397 | direct_copy=0.2587s, other_metadata_copy_cat_index=0.0255s, gatherTopK=0.0249s |
| `batch_prepare:decode:bs4` | `0.005991` | 2613 | topk_lens_swa_compressed_index_assembly=0.0022s, direct_copy=0.0018s, arange_index_helper=0.0012s |

## Notes

- Sub-boundary split intentionally includes gatherTopK/topk_transform as adjacent topk-lens metadata even when the 07.63 coarse classifier placed gatherTopK in fp8_indexer.
- The 07.63 gate bucket graph_runtime_copy_cat_index is the subset excluding gatherTopK/topk_transform kernels classified elsewhere by the older script.
