# TARGET 07.56 Static Scale Cache Focused Profile

Source JSON: `static_scale_cache_torch_profile.json`

Shape: FP8 projection wrapper, `M=4`, `N=512`, `K=128`, 20 profiled calls after warmup.

| Scale path | `aten::_to_copy` count | `aten::copy_` count | direct-copy CUDA kernel count | direct-copy CUDA us | `_quantized_linear_fp8_kernel` CUDA us |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw e8m0 scale | `20` | `20` | `20` | `51.0050` | `398.3320` |
| cached FP32 scale | `0` | `0` | `0` | `0.0000` | `397.2740` |

Interpretation: the opt-in cache removes the projection-wrapper scale
`float().contiguous()` cast/copy events in the focused profile, while the
underlying FP8 GEMM kernel time is unchanged.  The 4096/128 macro gain was only
`+0.35%`, so this focused reduction is not large enough to redirect the next
target away from projection/GEMM backend parity.
