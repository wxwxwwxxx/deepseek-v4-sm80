# TARGET 07.38 - DSV4 SM80 MoE Exact Backend Adaptation

Date: 2026-07-01

## Conclusion

Direct vLLM Marlin MXFP4 W4A16 expert-backend adaptation is **not implemented**
in this cut. The selected vLLM backend is exact in precision semantics, but its
runtime is not a narrow Python-level backend swap:

- weight/layout conversion requires `gptq_marlin_repack`;
- expert GEMM requires `_moe_C::moe_wna16_marlin_gemm`;
- scale/layout handling requires Marlin-specific packed WNA16 contracts;
- mini-sglang does not currently build an equivalent Marlin custom-op surface.

The TARGET 07.38 decision is therefore: **reject the direct Marlin port with a
precise blocker, keep the default grouped FP4 backend, and open a mini-owned
local exact W4A16 expert-kernel backend plan**. Do not move to TARGET 07.4 yet:
the selected vLLM backend is still W4A16 with bf16/fp16 activations and does not
prove that activation quantization is required.

## Artifacts

| Path | Contents |
| --- | --- |
| `scripts/audit_marlin_surface.py` | Reproducible source-surface audit without importing vLLM at runtime. |
| `summaries/marlin_feasibility_audit.json` | Audit output showing required Marlin surface and hard blockers. |
| `summaries/backend_adaptation_summary.json` | Compact machine-readable conclusion and key results. |
| `raw/moe_exact_backend_microbench_real_shapes.json` | MoE microbench with grouped stage split and unsupported Marlin candidate reporting. |

No vLLM source code was copied into mini. vLLM was used as an Apache-2.0 source
and design reference only.

## Implemented Guard

New selector:

```bash
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=grouped_fp4
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_mxfp4_w4a16
```

Default remains `grouped_fp4`. Setting `marlin_mxfp4_w4a16` raises an explicit
unsupported-kernel error at the routed expert boundary. It does not fall back to
grouped FP4.

New perf-matrix candidate variant:

```text
v1_moe_vllm_runner_marlin_mxfp4_w4a16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

This variant is intentionally visible and expected to fail until mini owns an
equivalent Marlin WNA16 backend. The failure is part of the silent-fallback
guard.

## Feasibility Audit

The audit confirms TARGET 07.37's semantic classification:

- backend: `MARLIN`;
- weight precision: MXFP4;
- activation precision: bf16/fp16, unquantized;
- category: exact candidate, not a precision lane.

Required surface:

| Surface | mini status | Decision |
| --- | --- | --- |
| `prepare_moe_mxfp4_layer_for_marlin` | missing | Needs local layout transform if a Marlin-like backend is pursued. |
| Marlin packed weight/scale layout | missing | Requires WNA16 contract implementation. |
| `gptq_marlin_repack` | missing | Hard blocker for direct Marlin reuse. |
| `_moe_C::moe_wna16_marlin_gemm` | missing | Hard blocker for direct Marlin reuse. |
| route metadata adapter | partially present | mini has compatible grouped route metadata, but Marlin-specific block-size behavior still needs adapter validation. |
| workspace contract | missing | Must be added with the expert kernel. |

This is larger than the intended TARGET 07.38 narrow adaptation. Hard-porting
vLLM's Marlin CUDA stack here would be a new compiled backend project, not a
safe local replacement under the existing runner shell.

## Correctness

Commands run:

```bash
python -m compileall \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_moe_route_microbench.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  performance_milestones/target07_moe_exact_backend_adapt/scripts/audit_marlin_surface.py

pytest -q -o addopts='' tests/kernel/test_deepseek_v4_wrappers.py
pytest -q -o addopts='' tests/benchmark/test_deepseek_v4_perf_matrix.py
```

Results:

| Check | Result |
| --- | --- |
| compileall | pass |
| `tests/kernel/test_deepseek_v4_wrappers.py` | 31 passed, 4 external warnings |
| `tests/benchmark/test_deepseek_v4_perf_matrix.py` | 9 passed |
| focused selector/perf variant tests | 4 passed |

TP8 text smoke was not run because there is no executable Marlin candidate; the
default grouped backend is unchanged.

## Microbench

Command:

```bash
python benchmark/offline/deepseek_v4_moe_route_microbench.py \
  --warmup 1 \
  --iters 2 \
  --include-real-shapes \
  --output performance_milestones/target07_moe_exact_backend_adapt/raw/moe_exact_backend_microbench_real_shapes.json
```

Device: `NVIDIA A100-SXM4-80GB`, capability `(8, 0)`.

| Case | Current grouped total | 07.36 runner/grouped total | W13 | activation | W2 | route_sum | W13+W2 | Marlin candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `decode_real` | 2.0419 ms | 2.0669 ms | 1.2186 ms | 0.0589 ms | 0.7286 ms | 0.0492 ms | 1.9471 ms | unsupported |
| `prefill_real` | 97.9553 ms | 98.0470 ms | 57.7976 ms | 0.0660 ms | 39.8300 ms | 0.2212 ms | 97.6276 ms | unsupported |

The grouped backend still spends almost all real-shape MoE time in W13+W2. The
Marlin candidate cannot be measured without the missing custom ops, and the
microbench records `silent_fallback: false`.

## Macro And Nsight

The 4096/128 profile-equivalent macro, 4096/1024 macro, TP8 text smoke, and
short Nsight were not rerun for Marlin because the backend is blocked before
there is an executable candidate. Running those workloads with the Marlin env
would produce an explicit unsupported error, not grouped-FP4 numbers.

The relevant 07.36 baseline remains:

| Workload | 07.36 runner output tok/s |
| --- | ---: |
| 4096/128/batch4 | 10.7579 |
| 4096/1024/batch4 | 17.8289 |

The old serving victory line remains `114.07` output tok/s.

## Final Decision

Stop direct Marlin adaptation in TARGET 07.38.

Next target: open a **local exact expert-kernel backend plan** for DSV4 SM80
W4A16 routed experts. That plan can either implement a mini-owned Marlin-like
WNA16 custom-op surface or a local exact kernel that directly reduces W13/W2
time without vLLM runtime dependency. Continue to avoid activation quantization,
INT8, MXFP8, FP8 cache, and precision-lane behavior until an exact backend cut
has been attempted under its own scope.

Do not continue here unless the project first adds a mini-owned Marlin-compatible
custom-op build surface or explicitly accepts a larger CUDA-extension port
target.
