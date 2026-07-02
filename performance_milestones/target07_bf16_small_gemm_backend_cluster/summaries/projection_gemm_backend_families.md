# Projection/GEMM Backend Families

- projection/GEMM denominator: `0.778170s`

## Backend Clusters

| Backend cluster | Kernel s | Count | Share | Backend families | Owner groups | Gate read |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `BF16 small-GEMM + splitK/reduce cluster` | `0.521012` | 84328 | `66.95%` | `cuBLASLt BF16 GEMM` `0.313612`<br>`cuBLASLt splitK/reduce` `0.111436`<br>`CUTLASS BF16 GEMM` `0.095964` | `attention WQA/WKV/compress` `0.123405`<br>`shared experts cached BF16` `0.092590`<br>`attention wo_a` `0.064244`<br>`attention q_wqb` `0.054146`<br>`attention wo_b local` `0.051837`<br>`indexer weight/compressor projection` `0.044530`<br>`indexer wq_b` `0.039692`<br>`HC pre linear` `0.037332`<br>`MoE router / route projection` `0.012772`<br>`model HC head/expand` `0.000463` | clears same-backend cluster gate |
| `FP32/SGEMM small-GEMM cluster` | `0.257158` | 16637 | `33.05%` | `cuBLAS SGEMM/FP32 GEMM` `0.257158` | `HC pre linear` `0.144151`<br>`MoE router / route projection` `0.084928`<br>`lm_head` `0.026622`<br>`model HC head/expand` `0.001457` | below same-backend cluster gate |

## Backend Families

| Backend family | Kernel s | Count | Share | Owner groups | Top kernels |
| --- | ---: | ---: | ---: | --- | --- |
| `cuBLASLt BF16 GEMM` | `0.313612` | 32639 | `40.30%` | `attention WQA/WKV/compress` `0.066027`<br>`attention q_wqb` `0.054146`<br>`attention wo_a` `0.052496`<br>`attention wo_b local` `0.051837`<br>`indexer wq_b` `0.039692`<br>`shared experts cached BF16` `0.037697`<br>`indexer weight/compressor projection` `0.011717` | `0.105983` ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_nn<br>`0.066027` ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn<br>`0.052496` ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x6_nn |
| `cuBLAS SGEMM/FP32 GEMM` | `0.257158` | 16637 | `33.05%` | `HC pre linear` `0.144151`<br>`MoE router / route projection` `0.084928`<br>`lm_head` `0.026622`<br>`model HC head/expand` `0.001457` | `0.230536` ampere_sgemm_32x32_sliced1x4_tn<br>`0.026622` void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool... |
| `cuBLASLt splitK/reduce` | `0.111436` | 40894 | `14.32%` | `HC pre linear` `0.037332`<br>`attention WQA/WKV/compress` `0.020073`<br>`shared experts cached BF16` `0.014973`<br>`indexer weight/compressor projection` `0.014074`<br>`MoE router / route projection` `0.012772`<br>`attention wo_a` `0.011748`<br>`model HC head/expand` `0.000463` | `0.050567` void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float,...<br>`0.045896` void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float...<br>`0.014973` void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat1... |
| `CUTLASS BF16 GEMM` | `0.095964` | 10795 | `12.33%` | `shared experts cached BF16` `0.039920`<br>`attention WQA/WKV/compress` `0.037305`<br>`indexer weight/compressor projection` `0.018739` | `0.056044` void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::P...<br>`0.039920` void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_nn_a... |
