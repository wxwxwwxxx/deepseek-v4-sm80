# Projection/GEMM Backend Families

- projection/GEMM denominator: `0.778887s`

## Backend Clusters

| Backend cluster | Kernel s | Count | Share | Backend families | Owner groups | Gate read |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `BF16 small-GEMM + splitK/reduce cluster` | `0.521619` | 84328 | `66.97%` | `cuBLASLt BF16 GEMM` `0.219912`<br>`CUTLASS BF16 GEMM` `0.194319`<br>`cuBLASLt splitK/reduce` `0.107388` | `attention WQA/WKV/compress` `0.119458`<br>`shared experts cached BF16` `0.085848`<br>`attention wo_a` `0.063857`<br>`attention q_wqb` `0.056392`<br>`attention wo_b local` `0.054507`<br>`indexer weight/compressor projection` `0.043647`<br>`indexer wq_b` `0.042727`<br>`HC pre linear` `0.035519`<br>`MoE router / route projection` `0.012636`<br>`residual / coarse owner` `0.006567`<br>`model HC head/expand` `0.000461` | clears same-backend cluster gate |
| `FP32/SGEMM small-GEMM cluster` | `0.257269` | 16637 | `33.03%` | `cuBLAS SGEMM/FP32 GEMM` `0.257269` | `HC pre linear` `0.142854`<br>`MoE router / route projection` `0.084473`<br>`lm_head` `0.026769`<br>`residual / coarse owner` `0.001719`<br>`model HC head/expand` `0.001454` | below same-backend cluster gate |

## Backend Families

| Backend family | Kernel s | Count | Share | Owner groups | Top kernels |
| --- | ---: | ---: | ---: | --- | --- |
| `cuBLAS SGEMM/FP32 GEMM` | `0.257269` | 16637 | `33.03%` | `HC pre linear` `0.142854`<br>`MoE router / route projection` `0.084473`<br>`lm_head` `0.026769`<br>`residual / coarse owner` `0.001719`<br>`model HC head/expand` `0.001454` | `0.230499` ampere_sgemm_32x32_sliced1x4_tn<br>`0.026769` void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool... |
| `cuBLASLt BF16 GEMM` | `0.219912` | 21717 | `28.23%` | `attention q_wqb` `0.056392`<br>`attention wo_b local` `0.054507`<br>`attention wo_a` `0.052332`<br>`indexer wq_b` `0.042727`<br>`indexer weight/compressor projection` `0.011426`<br>`residual / coarse owner` `0.002528` | `0.111860` ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn<br>`0.052565` ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn<br>`0.043766` ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn |
| `CUTLASS BF16 GEMM` | `0.194319` | 21717 | `24.95%` | `attention WQA/WKV/compress` `0.101623`<br>`shared experts cached BF16` `0.071605`<br>`indexer weight/compressor projection` `0.018296`<br>`residual / coarse owner` `0.002795` | `0.159949` void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::P...<br>`0.034370` void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_a... |
| `cuBLASLt splitK/reduce` | `0.107388` | 40894 | `13.79%` | `HC pre linear` `0.035519`<br>`attention WQA/WKV/compress` `0.017835`<br>`shared experts cached BF16` `0.014242`<br>`indexer weight/compressor projection` `0.013925`<br>`MoE router / route projection` `0.012636`<br>`attention wo_a` `0.011525`<br>`residual / coarse owner` `0.001244`<br>`model HC head/expand` `0.000461` | `0.058359` void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float...<br>`0.049029` void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float,... |
