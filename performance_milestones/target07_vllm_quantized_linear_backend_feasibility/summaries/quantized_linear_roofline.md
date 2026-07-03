# Quantized-Linear Roofline

Date: 2026-07-02

Input artifact:
`../raw/focused_quantized_linear_backend_microbench.json`.

## Constants And Formula

Conservative A100 constants:

| Constant | Value |
| --- | ---: |
| BF16 tensor core peak | `312 TFLOP/s` |
| INT8 tensor core peak | `624 TOPS` |
| FP32 peak | `19.5 TFLOP/s` |
| TF32 tensor core peak | `156 TFLOP/s` |
| HBM bandwidth | `1.55 TB/s` |

For each local projection shape:

```text
FLOPs ~= 2 * M * N * K
Arithmetic intensity = FLOPs / estimated bytes
Roofline lower bound = max(FLOPs / peak_compute, bytes / HBM_bandwidth)
Efficiency = roofline_lower_bound / measured_time
Headroom = measured_time / roofline_lower_bound
```

Byte estimates are deliberately simple:

| Backend | Estimated replay bytes |
| --- | --- |
| Promoted cached BF16 dense | activation read + activation rounded write + GEMM activation read + cached BF16 weight read + BF16 output write |
| vLLM FP8 Marlin W8A16 | BF16 activation read + Marlin packed FP8 weight read + Marlin scale read + BF16 output write |
| FBGEMM-derived Marlin | Same replay estimate as Marlin, with load-time conversion excluded from replay and recorded in prep ledger |
| INT8 W8A8 | BF16 activation read + INT8 activation write + INT8 weight read + scale read + BF16 output write |
| `wo_a` grouped | Same formulas, but the Marlin/INT8 diagnostics require two linear launches per replay |

## M=4 Table

| Owner | Backend | N | K | FLOPs GF | Bytes MB | Measured ms | Roofline ms | Eff | Headroom | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention WQA/WKV/compress` | `promoted_cached_bf16_total` | 1536 | 4096 | `0.050` | `12.694` | `0.100167` | `0.008189` | `0.0818` | `12.2x` | `0.00%` |
| `attention WQA/WKV/compress` | `vllm_fp8_marlin_w8a16_block` | 1536 | 4096 | `0.050` | `6.435` | `0.077816` | `0.004151` | `0.0534` | `18.7x` | `22.31%` |
| `attention WQA/WKV/compress` | `vllm_fbgemm_fp8_marlin_derived_channel` | 1536 | 4096 | `0.050` | `6.340` | `0.077356` | `0.004090` | `0.0529` | `18.9x` | `22.77%` |
| `attention WQA/WKV/compress` | `vllm_int8_w8a8_cutlass_dynamic` | 1536 | 4096 | `0.050` | `6.359` | `0.084380` | `0.004103` | `0.0486` | `20.6x` | `15.76%` |
| `attention q_wqb` | `promoted_cached_bf16_total` | 4096 | 1024 | `0.034` | `8.446` | `0.092440` | `0.005449` | `0.0589` | `17.0x` | `0.00%` |
| `attention q_wqb` | `vllm_fp8_marlin_w8a16_block` | 4096 | 1024 | `0.034` | `4.301` | `0.066223` | `0.002775` | `0.0419` | `23.9x` | `28.36%` |
| `attention q_wqb` | `vllm_fbgemm_fp8_marlin_derived_channel` | 4096 | 1024 | `0.034` | `4.243` | `0.063482` | `0.002738` | `0.0431` | `23.2x` | `31.33%` |
| `attention q_wqb` | `vllm_int8_w8a8_cutlass_dynamic` | 4096 | 1024 | `0.034` | `4.256` | `0.083800` | `0.002746` | `0.0328` | `30.5x` | `9.35%` |
| `attention wo_b local` | `promoted_cached_bf16_total` | 4096 | 1024 | `0.034` | `8.446` | `0.091272` | `0.005449` | `0.0597` | `16.8x` | `0.00%` |
| `attention wo_b local` | `vllm_fp8_marlin_w8a16_block` | 4096 | 1024 | `0.034` | `4.301` | `0.064005` | `0.002775` | `0.0434` | `23.1x` | `29.87%` |
| `attention wo_b local` | `vllm_fbgemm_fp8_marlin_derived_channel` | 4096 | 1024 | `0.034` | `4.243` | `0.063063` | `0.002738` | `0.0434` | `23.0x` | `30.91%` |
| `attention wo_b local` | `vllm_int8_w8a8_cutlass_dynamic` | 4096 | 1024 | `0.034` | `4.256` | `0.083567` | `0.002746` | `0.0329` | `30.4x` | `8.44%` |
| `shared experts gate/up` | `promoted_cached_bf16_total` | 512 | 4096 | `0.017` | `4.297` | `0.098032` | `0.002772` | `0.0283` | `35.4x` | `0.00%` |
| `shared experts gate/up` | `vllm_fp8_marlin_w8a16_block` | 512 | 4096 | `0.017` | `2.167` | `0.079082` | `0.001398` | `0.0177` | `56.6x` | `19.33%` |
| `shared experts gate/up` | `vllm_fbgemm_fp8_marlin_derived_channel` | 512 | 4096 | `0.017` | `2.135` | `0.079748` | `0.001377` | `0.0173` | `57.9x` | `18.65%` |
| `shared experts gate/up` | `vllm_int8_w8a8_cutlass_dynamic` | 512 | 4096 | `0.017` | `2.152` | `0.085368` | `0.001389` | `0.0163` | `61.5x` | `12.92%` |
| `shared experts down` | `promoted_cached_bf16_total` | 4096 | 256 | `0.008` | `2.136` | `0.092495` | `0.001378` | `0.0149` | `67.1x` | `0.00%` |
| `shared experts down` | `vllm_fp8_marlin_w8a16_block` | 4096 | 256 | `0.008` | `1.100` | `0.064094` | `0.000710` | `0.0111` | `90.3x` | `30.70%` |
| `shared experts down` | `vllm_fbgemm_fp8_marlin_derived_channel` | 4096 | 256 | `0.008` | `1.092` | `0.062869` | `0.000704` | `0.0112` | `89.3x` | `32.03%` |
| `shared experts down` | `vllm_int8_w8a8_cutlass_dynamic` | 4096 | 256 | `0.008` | `1.101` | `0.083520` | `0.000710` | `0.0085` | `117.6x` | `9.70%` |
| `attention wo_a grouped` | `promoted_cached_bf16_grouped_bmm` | 1024 | 4096 | `0.034` | `8.495` | `0.057180` | `0.005481` | `0.0958` | `10.4x` | `0.00%` |
| `attention wo_a grouped` | `vllm_fp8_marlin_w8a16_block_grouped_two_launch` | 1024 | 4096 | `0.034` | `4.301` | `0.189748` | `0.002775` | `0.0146` | `68.4x` | `-231.84%` |

## Interpretation

The measured decode-small projections are not close to raw compute or memory
roofline.  Even the promoted cached BF16 path is only `1%` to `10%` of this
simple lower bound, and the smallest shapes have apparent headroom above
`60x`.  This is expected for M=`1..16`: launch/backend fixed cost dominates.

The useful signal is relative, not absolute:

- Marlin roughly halves replay weight/scale bytes for dense owners and removes
  mini's replay-time activation FP8 rounding from the vLLM-aligned candidate.
- The dense Marlin rows still remain launch/backend bound, but they are
  consistently faster than promoted cached BF16 on `q_wqb`, `wo_b local`, and
  shared expert down across M=`1,4,8,16`.
- `wo_a` grouped two-launch Marlin is a clear no-go despite lower bytes; the
  extra launches dominate.
- INT8 W8A8 has more theoretical compute headroom, but dynamic activation
  quantization and backend overhead leave it below the standalone gate.

Custom-kernel gate read:

- The arithmetic roofline alone shows large theoretical headroom, but the
  existing vLLM Marlin backend already clears the standalone gate for a dense
  owner subset.
- Do not open a custom kernel R&D target before a runtime opt-in profile shows
  whether the standalone Marlin gains survive graph capture, TP8, and owner
  scheduling.
- If the runtime opt-in later fails macro/profile gates, the custom surface
  should target launch grouping/fusion across dense owners or a true grouped
  `wo_a` kernel, not another per-owner software FP8 dequant wrapper.
