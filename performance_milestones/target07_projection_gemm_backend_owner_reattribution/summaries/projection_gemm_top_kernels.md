# Projection/GEMM Top Kernels

| Kernel | Backend family | Kernel s | Count | Graph events | Graph nodes |
| --- | --- | ---: | ---: | ---: | ---: |
| `ampere_sgemm_32x32_sliced1x4_tn` | `cuBLAS SGEMM/FP32 GEMM` | `0.230499` | 16510 | 16510 | 130 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | `CUTLASS BF16 GEMM` | `0.159949` | 16256 | 16256 | 128 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | `cuBLASLt BF16 GEMM` | `0.111860` | 10922 | 10922 | 86 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_b...` | `cuBLASLt splitK/reduce` | `0.058359` | 24384 | 24384 | 192 |
| `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn` | `cuBLASLt BF16 GEMM` | `0.052565` | 5461 | 5461 | 43 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0>(cubl...` | `cuBLASLt splitK/reduce` | `0.049029` | 16510 | 16510 | 130 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | `cuBLASLt BF16 GEMM` | `0.043766` | 2667 | 2667 | 21 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` | `CUTLASS BF16 GEMM` | `0.034370` | 5461 | 5461 | 43 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTen...` | `cuBLAS SGEMM/FP32 GEMM` | `0.026769` | 127 | 127 | 1 |
| `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn` | `cuBLASLt BF16 GEMM` | `0.011721` | 2667 | 2667 | 21 |
