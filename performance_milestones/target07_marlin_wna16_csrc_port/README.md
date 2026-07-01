# TARGET 07.391: DSV4 SM80 Marlin WNA16 Csrc Port

## Final conclusion

TARGET 07.391 succeeded as a mini-owned Marlin WNA16 custom-op port. The repo
now vendors the narrow vLLM Marlin WNA16 source surface, builds it as a mini
PyTorch CUDA extension, transforms/caches DSV4 MXFP4 expert weights into Marlin
layout, and exposes an explicit opt-in backend:

```bash
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
```

The default backend remains grouped FP4. The Marlin WNA16 path does not import
vLLM at runtime and does not silently fall back to grouped FP4 when explicitly
selected.

This is a real end-to-end improvement, but it is not the final TARGET 07 victory
line. The 4096/1024/batch4 macro reaches `54.47 output tok/s`, well above the
previous mini exact line but still below the old vLLM serving reference
`114.07 output tok/s`. The short Nsight profile now shows sparse attention and
indexer/cache work ahead of the Marlin expert kernel. The next target should
move to attention/indexer/cache or reduce Marlin integration overhead; it should
not open TARGET 07.4 precision lanes based on this evidence.

## Artifacts

- Prompt: `prompts/TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port.md`
- Vendored source: `python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/`
- Runtime helper: `python/minisgl/kernel/marlin_wna16.py`
- Build probe: `scripts/probe_minimal_marlin_extension_build.py`
- Synthetic op probe: `scripts/probe_minisgl_marlin_wna16_ops.py`
- Runtime helper smoke: `scripts/probe_minisgl_marlin_wna16_runtime_helper.py`
- Summary JSON: `summaries/csrc_port_summary.json`
- Nsight summaries:
  - `summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0.json`
  - `summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0.md`
  - `summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0_nvtx.json`
  - `summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0_nvtx.md`

## Ported Surface

Vendored from `/workspace/vllm-dsv4-docker/csrc` with Apache-2.0 attribution:

- `core/registration.h`
- `core/scalar_type.hpp`
- `quantization/marlin/dequant.h`
- `quantization/marlin/gptq_marlin_repack.cu`
- `quantization/marlin/marlin.cuh`
- `quantization/marlin/marlin_dtypes.cuh`
- `quantization/marlin/marlin_mma.h`
- `moe/marlin_moe_wna16/kernel.h`
- `moe/marlin_moe_wna16/kernel_selector.h`
- `moe/marlin_moe_wna16/marlin_template.h`
- `moe/marlin_moe_wna16/ops.cu`
- all `moe/marlin_moe_wna16/sm80_kernel_*.cu`

Mini adds `schema.cpp` to register `gptq_marlin_repack` and
`moe_wna16_marlin_gemm` under the mini extension namespace. The first narrow
attempt with only the DSV4-looking BF16/FE2M1F kernel linked but failed at import
because the unmodified selector references the full SM80 generated set, so this
target vendors all SM80 generated instantiations.

## Layout And Runtime Integration

Mini raw expert weights are reinterpreted without changing MXFP4 semantics:

- W13 mini raw: `[experts, 2, local_intermediate, hidden / 2]` int8 bytes.
- W13 Marlin input: `[experts, 2 * local_intermediate, hidden / 2]` uint8 bytes.
- W2 mini raw: `[experts, hidden, local_intermediate / 2]` int8 bytes.
- Scales: E8M0 bytes viewed as `torch.float8_e8m0fnu`.

The cached Marlin layout produced by the DSV4-like probe is:

| tensor | shape | dtype |
| --- | ---: | --- |
| W13 qweight | `[256, 256, 1024]` | int32 |
| W13 scale | `[256, 128, 512]` | float8_e8m0fnu |
| W2 qweight | `[256, 16, 8192]` | int32 |
| W2 scale | `[256, 8, 4096]` | float8_e8m0fnu |

`MarlinWNA16Weights` caches transformed tensors by source tensor pointer, shape,
and dtype. The model stores one cache per `DSV4FusedRoutedExperts` layer. The
explicit backend branch returns before the grouped FP4 path, so unsupported
errors do not silently fall through.

## Synthetic Results

Canonical mini-owned op command:

```bash
python performance_milestones/target07_marlin_wna16_csrc_port/scripts/probe_minisgl_marlin_wna16_ops.py \
  --tokens 4 4096 \
  --warmup 1 \
  --iters 3 \
  --output performance_milestones/target07_marlin_wna16_csrc_port/raw/minisgl_marlin_wna16_ops_probe_vendored.json
```

| case | direct W13 | local Marlin MoE | mini grouped FP4 | speedup vs grouped | 07.39 vLLM bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| T=4 | `0.053248 ms` | `0.247467 ms` | `2.118997 ms` | `8.56x` | `0.441344 ms` |
| T=4096 | `1.039360 ms` | `2.485589 ms` | `47.843328 ms` | `19.25x` | `2.343936 ms` |

T=4096 is close to the 07.39 bridge. T=4 is faster in this local harness,
likely because the local Python wrapper differs from the vLLM helper path. The
numeric comparison against grouped FP4 uses random synthetic packed bytes and is
only a layout/perf sanity check, not a real model correctness proof.

## Model Validation

Runtime helper smoke:

- Raw: `raw/minisgl_marlin_wna16_runtime_helper_smoke.json`
- Extension load/build: pass, `259.36 s` first-time JIT.
- First helper call after load: pass, `0.2819 s`, finite output.
- Second helper call with cache reuse: pass, `0.0019 s`, `cache_object_reused=true`.
- `vllm_imported_before=false`, `vllm_imported_after=false`.

TP8 text smoke:

- Raw: `raw/tp8_text_smoke_marlin_wna16_full.json`
- Status: pass for all three standard prompts.
- CUDA graph: captured `[4, 2, 1]`, replayed batch3/padded4 decode.
- Peak allocated memory: about `41.37 GB` per rank.

Macro, TP8, page size 256, batch4, graph enabled, `--num-pages 128`:

| workload | status | output tok/s | decode tok/s | prefill tok/s | TTFT mean | graph replay | unsupported skips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128 | pass | `33.81` | `61.30` | `2799.52` | `6.59 s` | `127` | `0` |
| 4096/1024 | pass | `54.47` | `61.41` | `2787.76` | `6.62 s` | `1023` | `0` |

The first 4096/128 attempt using default `memory_ratio=0.9` failed during graph
capture because the harness allocated `740608` KV-cache tokens (`51.56 GiB` per
rank), leaving too little room for graph-time Marlin weight transforms. Pinning
`--num-pages 128` is enough for batch4 4096+1024 and keeps peak memory around
`45.84 GB` per rank in the macro runs.

## Nsight

Rank-0 Nsight Systems capture:

- Raw report: `raw/nsys_marlin_wna16_4096x128_bs4_np128_rank0.nsys-rep`
- SQLite: `raw/nsys_marlin_wna16_4096x128_bs4_np128_rank0.sqlite`
- Workload summary: `raw/nsys_perf_4096_128_marlin_wna16_np128/summary.json`

The `repeat:smoke_debug:0` window shows:

- kernels: `58132`, `5.878 s`
- CUDA graph trace: `127`, `8.210 s`
- NCCL kernels: `88`, `0.163 s`
- top kernels:
  - sparse attention: `2.067 s`
  - `_indexer_bf16_logits_kernel`: `0.922 s`
  - copy/elementwise helpers: `0.401 s`, `0.290 s`
  - `_hc_split_pre_kernel`: `0.356 s`
  - Marlin WNA16 kernel: `86` launches, `0.234 s`

This is the important profile shift: the old grouped FP4 expert kernels are no
longer dominant after the port. Sparse attention, indexer/cache, metadata/copy
work, and CUDA graph/runtime overhead are now the higher-value areas.

## Decision

Proceed with the mini-owned `marlin_wna16` backend as an explicit exact backend
candidate. It is not ready to become default because first-use JIT/package
handling, per-layer transformed-weight memory, and bounded KV-cache sizing need
production hardening. The next performance target should focus on
attention/indexer/cache and metadata/runtime overhead, not activation
quantization or other precision-lane changes.
