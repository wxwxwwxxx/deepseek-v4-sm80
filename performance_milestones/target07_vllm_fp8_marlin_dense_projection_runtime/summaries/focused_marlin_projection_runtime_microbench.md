# Focused Marlin Projection Runtime Microbench

- created_at: `2026-07-03 00:00:09`
- python: `/workspace/venvs/vllm-dsv4/bin/python`
- torch: `2.11.0+cu128`
- device: `NVIDIA A100-SXM4-80GB`
- model_path: `/models/DeepSeek-V4-Flash`

## Latency

| Owner | M | Baseline ms | Marlin ms | Speedup | Mean abs | P99 abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention q_wqb | 1 | `0.258256` | `0.064202` | `75.14%` | `0.017649` | `0.060547` | `0.99961501` |
| attention q_wqb | 4 | `0.255962` | `0.064580` | `74.77%` | `0.015490` | `0.054688` | `0.99967110` |
| attention q_wqb | 8 | `0.256494` | `0.064062` | `75.02%` | `0.016337` | `0.056641` | `0.99964041` |
| attention q_wqb | 16 | `0.259016` | `0.064262` | `75.19%` | `0.016412` | `0.058594` | `0.99963522` |
| attention wo_b local | 1 | `0.256092` | `0.064661` | `74.75%` | `0.018297` | `0.058594` | `0.99962783` |
| attention wo_b local | 4 | `0.260682` | `0.064178` | `75.38%` | `0.018019` | `0.058594` | `0.99965119` |
| attention wo_b local | 8 | `0.255094` | `0.064388` | `74.76%` | `0.018288` | `0.058594` | `0.99965417` |
| attention wo_b local | 16 | `0.257547` | `0.064832` | `74.83%` | `0.018591` | `0.062500` | `0.99964738` |
| shared experts down | 1 | `0.252924` | `0.064396` | `74.54%` | `0.004979` | `0.015631` | `0.99963319` |
| shared experts down | 4 | `0.256383` | `0.063991` | `75.04%` | `0.005111` | `0.016785` | `0.99960858` |
| shared experts down | 8 | `0.256537` | `0.064038` | `75.04%` | `0.004992` | `0.016602` | `0.99964237` |
| shared experts down | 16 | `0.255489` | `0.063606` | `75.10%` | `0.005003` | `0.016602` | `0.99963832` |

## Prep And Persistent Bytes

| Owner | Backend | Prep ms | Persistent bytes | Workspace bytes | Original weight bytes | Original scale bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attention q_wqb | `promoted_cached_bf16` | `30.154` | `8388608` | `0` | `4194304` | `256` |
| attention q_wqb | `mini_runtime_vllm_fp8_marlin_w8a16_block` | `2666.200` | `4260272` | `432` | `4194304` | `256` |
| attention wo_b local | `promoted_cached_bf16` | `0.188` | `8388608` | `0` | `4194304` | `256` |
| attention wo_b local | `mini_runtime_vllm_fp8_marlin_w8a16_block` | `0.862` | `4260272` | `432` | `4194304` | `256` |
| shared experts down | `promoted_cached_bf16` | `0.136` | `2097152` | `0` | `1048576` | `64` |
| shared experts down | `mini_runtime_vllm_fp8_marlin_w8a16_block` | `0.629` | `1065392` | `432` | `1048576` | `64` |
