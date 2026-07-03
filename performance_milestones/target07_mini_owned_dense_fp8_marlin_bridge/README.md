# TARGET 07.75: Mini-Owned Dense FP8 Marlin Extension Bridge

Status: PASS. Stop here per target rules; no TP8 runtime integration and no
`dsv4_sm80_a100_victory` promotion were attempted.

## Environment

- default Python: `/usr/bin/python`
- mini torch: `2.9.1+cu128`
- CUDA: `12.8`
- `_GLIBCXX_USE_CXX11_ABI`: `True`
- GPU: `NVIDIA A100-SXM4-80GB`, capability `sm80`
- offline vLLM helper comparison: `/workspace/venvs/vllm-dsv4/bin/python`,
  torch `2.11.0+cu128`

## Source Surface And Build

Mini-owned module:

- `python/minisgl/kernel/dense_fp8_marlin.py`
- extension name: `minisgl_dense_fp8_marlin`
- default build directory: `/root/.cache/minisgl/dense_fp8_marlin`
- compiled extension: `/root/.cache/minisgl/dense_fp8_marlin/minisgl_dense_fp8_marlin.so`

Vendored Apache-2.0 vLLM dense FP8 Marlin subset:

- `core/registration.h`
- `core/scalar_type.hpp`
- `quantization/marlin/marlin.cu`
- `quantization/marlin/gptq_marlin_repack.cu`
- `quantization/marlin/kernel.h`
- `quantization/marlin/kernel_selector.h`
- `quantization/marlin/marlin_template.h`
- `quantization/marlin/marlin.cuh`
- `quantization/marlin/marlin_dtypes.cuh`
- `quantization/marlin/marlin_mma.h`
- `quantization/marlin/dequant.h`
- `quantization/marlin/sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu`

`kernel_selector.h` is intentionally pruned to the compiled target surface:
BF16 activation, `float8_e4m3fn` weight, BF16 output/scales on sm80. This fixed
the first import-time unresolved-symbol failure from vLLM's broad selector
referencing uncompiled dtype/template variants.

Build flags:

```text
-O3
-std=c++17
--expt-relaxed-constexpr
-static-global-template-stub=false
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_80,code=compute_80
```

Measured build/load:

- clean first build in a temporary build dir: `86.324 s`
- cached default build-dir `load_ops()` from a fresh Python: `1082.762 ms`
- extension `.so` size: `5,082,552 bytes`

## Registered Ops

The bridge registers only:

- `gptq_marlin_repack`
- `marlin_gemm`

The mini Python helper exposes:

- `prepare_dense_fp8_marlin_weight(...)`
- `apply_dense_fp8_marlin_linear(...)`

It does not import vLLM and does not depend on `sgl_kernel`.

## Focused Owner Quality

Real-weight probes used `/models/DeepSeek-V4-Flash`, layer `9`, simulated
rank `0` of TP `8`, and `M = 1, 4, 8, 16`.

Correctness is measured against exact BF16 activation + BF16 dequantized
`F.linear`. Speedup is measured against the currently promoted cached-BF16
mini baseline, which includes per-call activation FP8 rounding before
`F.linear`. The Marlin bridge itself keeps activations BF16.

M=`4` quality:

| Owner | Mini median ms | Max abs vs exact | Mean abs vs exact | p99 abs vs exact | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `0.058976` | `3.8147e-06` | `2.36469e-10` | `0` | `1.00000000` |
| `attn.wo_b local` | `0.058672` | `3.05176e-05` | `1.86628e-09` | `0` | `0.99999994` |
| `shared experts down` | `0.057792` | `0` | `0` | `0` | `1.00000000` |

Gate result:

```text
all_quality_ok: true
covers owners: attn.q_wqb, attn.wo_b local, shared experts down
covers M: 1, 4, 8, 16
```

## Focused Owner Latency

Mini-owned bridge vs offline vLLM helper:

| Owner | M | Mini median ms | vLLM median ms | Mini delta | Mini speedup vs promoted | vLLM speedup vs promoted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `1` | `0.059056` | `0.063520` | `-7.03%` | `76.51%` | `74.23%` |
| `attn.q_wqb` | `4` | `0.058976` | `0.063216` | `-6.71%` | `76.73%` | `74.57%` |
| `attn.q_wqb` | `8` | `0.058912` | `0.063472` | `-7.18%` | `77.24%` | `74.53%` |
| `attn.q_wqb` | `16` | `0.058624` | `0.063232` | `-7.29%` | `76.64%` | `74.61%` |
| `attn.wo_b local` | `1` | `0.058672` | `0.063520` | `-7.63%` | `76.65%` | `73.99%` |
| `attn.wo_b local` | `4` | `0.058672` | `0.063456` | `-7.54%` | `77.16%` | `74.61%` |
| `attn.wo_b local` | `8` | `0.058560` | `0.062816` | `-6.78%` | `76.93%` | `75.06%` |
| `attn.wo_b local` | `16` | `0.058784` | `0.063072` | `-6.80%` | `76.70%` | `74.86%` |
| `shared experts down` | `1` | `0.058192` | `0.063392` | `-8.20%` | `76.75%` | `74.27%` |
| `shared experts down` | `4` | `0.057792` | `0.062944` | `-8.19%` | `77.22%` | `75.28%` |
| `shared experts down` | `8` | `0.057296` | `0.062944` | `-8.97%` | `77.45%` | `74.65%` |
| `shared experts down` | `16` | `0.057216` | `0.061776` | `-7.38%` | `76.70%` | `74.43%` |

Summary:

- mean mini-vs-vLLM median delta: `-7.47%`
- mini min speedup vs promoted cached-BF16 baseline: `76.51%`
- mini mean speedup vs promoted cached-BF16 baseline: `76.89%`

## Memory / Workspace Ledger

| Owner | Backend | Original weight | Original scale | Prepared weight | Prepared scale | Workspace | Persistent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | cached BF16 | `4,194,304` | `256` | `8,388,608` | `0` | `0` | `8,388,608` |
| `attn.q_wqb` | mini Marlin | `4,194,304` | `256` | `4,194,304` | `65,536` | `432` | `4,260,272` |
| `attn.wo_b local` | cached BF16 | `4,194,304` | `256` | `8,388,608` | `0` | `0` | `8,388,608` |
| `attn.wo_b local` | mini Marlin | `4,194,304` | `256` | `4,194,304` | `65,536` | `432` | `4,260,272` |
| `shared experts down` | cached BF16 | `1,048,576` | `64` | `2,097,152` | `0` | `0` | `2,097,152` |
| `shared experts down` | mini Marlin | `1,048,576` | `64` | `1,048,576` | `16,384` | `432` | `1,065,392` |

The original FP8 weight/scale tensors are not needed by the focused Marlin
apply path after packing. A later runtime target can decide when to release
them in the model lifecycle.

## Comparison Against vLLM Helper

The vLLM helper path was run offline with:

```text
/workspace/venvs/vllm-dsv4/bin/python
```

It used vLLM Python helpers and vLLM custom ops only for comparison. The
mini-owned extension was built and tested under the default mini torch ABI and
does not import vLLM at runtime.

Raw artifacts:

- `raw/focused_dense_fp8_marlin_bridge_microbench.mini.json`
- `raw/focused_dense_fp8_marlin_bridge_microbench.mini.md`
- `raw/focused_dense_fp8_marlin_bridge_microbench.vllm_helper.json`
- `raw/focused_dense_fp8_marlin_bridge_microbench.vllm_helper.md`
- `summaries/mini_vs_vllm_helper_comparison.json`
- `summaries/mini_vs_vllm_helper_comparison.md`

## Decision

The mini-owned dense FP8 Marlin bridge passes the 07.75 focused gate:

- default mini Python builds and imports `minisgl_dense_fp8_marlin`;
- `gptq_marlin_repack` and `marlin_gemm` run on A100/sm80;
- focused real-weight quality passes for the Phase A dense owners;
- steady-state latency is competitive with, and slightly faster than, the
  vLLM helper path;
- steady-state latency is faster than the same-run promoted cached-BF16
  baseline;
- memory/workspace accounting is recorded.

Stop here. Do not continue into TP8 model runtime integration in this target.

## Next Target

Revise TARGET 07.74's default-off runtime opt-in to use
`minisgl_dense_fp8_marlin` instead of the vLLM Python helper, then run the TP8
smoke, graph replay, profile, macro, and memory-lifecycle gates.
