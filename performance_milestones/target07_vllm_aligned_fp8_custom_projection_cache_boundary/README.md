# TARGET 07.72 vLLM-Aligned FP8 / Custom Projection-Cache Boundary

Date: 2026-07-02

## Scope

This target tested whether a vLLM-aligned FP8/custom projection-cache boundary
can reduce the current promoted mini DeepSeek V4 A100 projection-cache cluster.

Baseline:

- variant: `dsv4_sm80_a100_victory`
- env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- page size: `256`
- TP8, 8x A100

Inactive opt-ins intentionally not used as baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
```

Repo status at report time:

```text
 M prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md
 M prompts/target.md
?? performance_milestones/target07_precision_boundary_pivot/
?? performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/
?? prompts/TARGET_07.72_dsv4_sm80_vllm_aligned_fp8_custom_projection_cache_boundary.md
```

No runtime source files were changed.

## TARGET 07.71 Decision Summary

TARGET 07.71 rejected HC/router exact-ish precision as the next lane:

| Lane | Result | Decision |
| --- | --- | --- |
| HC/router TF32 | Quality-stable but no decode-small speedup | Reject |
| HC/router BF16-like | Faster locally but changed router top-k on larger probe | Reject |
| HC-only BF16-like | Too small and changes HC output | Reject |

The selected 07.72 lane was the coherent vLLM-aligned FP8/custom
projection-cache cluster, with a stop rule to avoid runtime implementation
unless focused microbench showed at least two representative owners improving
by `>=15%`.

## Artifacts

- `scripts/focused_fp8_projection_cache_microbench.py`
- `raw/focused_fp8_projection_cache_microbench.json`
- `raw/focused_fp8_projection_cache_microbench.md`
- `raw/focused_fp8_projection_cache_microbench_smoke.json`
- `raw/focused_fp8_projection_cache_microbench_smoke.md`
- `summaries/fp8_projection_source_parity.md`
- `summaries/fp8_projection_surface_mapping.md`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON=1 \
python performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/scripts/focused_fp8_projection_cache_microbench.py \
  --model-path /models/DeepSeek-V4-Flash \
  --layer 9 \
  --tokens 1 4 8 16 \
  --warmup 50 \
  --iters 200 \
  --output performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/raw/focused_fp8_projection_cache_microbench.json
```

## vLLM Source Parity

Full table: `summaries/fp8_projection_source_parity.md`.

| vLLM mechanism | Mini analogue | 07.72 readout |
| --- | --- | --- |
| `DeepseekV4FP8Config` / `deepseek_v4_fp8` quantized linear stack | `DSV4Linear` with checkpoint FP8 weights and promoted cached BF16 selected owners | Source parity exists, but mini direct FP8-weight kernels are slower than cached BF16 on A100. |
| `DeepseekV4Attention.fused_wqa_wkv` | Mini q-proj fused WQA/WKV BF16 cache | Clean representative mapping; direct fused FP8-weight candidate regressed badly. |
| Quantized `ColumnParallelLinear` / `RowParallelLinear` | `q_wqb`, `wo_b`, shared expert gate/up/down | Direct FP8-weight candidate is graph-safe but slower for every measured owner. |
| `fused_inv_rope_fp8_quant` + `deepseek_v4_fp8_einsum` | Mini `wo_a` grouped projection | Mini grouped FP8 Triton path regressed versus cached BF16 BMM. |
| `fused_indexer_q_rope_quant` | Mini FP8 indexer query/cache path from earlier targets | Remaining indexer projection is too small for standalone 07.72 runtime work. |
| Full `fp8_ds_mla` KV-cache | Mini attention/cache E2E | Out of scope and not touched. |
| HC/router precision | Mini HC/router promoted path | Out of scope and not touched. |

## Mini Surface Mapping

Full table: `summaries/fp8_projection_surface_mapping.md`.

| Surface | Time |
| --- | ---: |
| Named mapped owners from TARGET 07.69 | `0.466436s` |
| BF16 small-GEMM plus splitK/reduce cluster | `0.521619s` |
| Required source-mapping floor | `0.350000s` |
| Mapping gate | pass |

The candidate surface was large enough to justify a focused microbench.

## Focused Microbench

Device: one `NVIDIA A100-SXM4-80GB`, CUDA capability `(8, 0)`, PyTorch
`2.9.1+cu128`, `torch.backends.cuda.matmul.allow_tf32=False`.

Compared routes:

- promoted cached BF16: FP8-style activation rounding, cached BF16 dequantized
  weights, `F.linear` or grouped `torch.bmm`;
- direct FP8/custom candidate: original checkpoint FP8 weights/scales decoded
  inside mini Triton projection kernels;
- dequant-on-the-fly diagnostic: per-call FP8 weight dequantization then
  PyTorch linear/einsum.

M=4 gate rows:

| Owner | Baseline ms | Direct FP8 ms | Speedup | Max abs err | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention WQA/WKV/compress | `0.044003` | `0.344534` | `-682.98%` | `0.0009765625` | `1.00000000` |
| attention `q_wqb` | `0.037778` | `0.151194` | `-300.21%` | `0` | `0.99999988` |
| attention `wo_b` local | `0.038232` | `0.150913` | `-294.73%` | `0` | `0.99999994` |
| shared experts gate/up | `0.043902` | `0.342883` | `-681.01%` | `0` | `0.99999994` |
| shared experts down | `0.038083` | `0.102496` | `-169.14%` | `0` | `0.99999982` |
| attention `wo_a` | `0.062519` | `0.342240` | `-447.42%` | `0` | `1.00000012` |

Focused gate result: fail.  No representative owner improved by `>=15%`; all
direct FP8/custom projection candidates regressed versus promoted cached BF16.

## Quality / Error

The tested direct FP8/custom candidates were numerically close to the promoted
cached BF16 outputs, but latency failed the gate.

| Check | Result |
| --- | --- |
| Max abs error at M=4 | `0` to `0.0009765625` |
| P99 abs error at M=4 | `0` for all measured owners |
| Cosine at M=4 | `0.99999982` to `1.00000012` |
| Quality gate read | Numerics acceptable for focused probe |
| Runtime gate read | Fail on latency |

## Implementation Toggle / Variant

No runtime candidate was implemented.

Not added:

```text
MINISGL_DSV4_SM80_FP8_CUSTOM_PROJECTION_CACHE
dsv4_sm80_a100_victory_fp8projcache
```

Reason: the target stop rule says to stop without runtime implementation if
focused microbench does not show `>=15%` latency reduction on at least two
representative owners.

## Text Smoke

Not run.  The runtime opt-in was not implemented because the focused
microbench gate failed.

## Macro

Not run.  The target stop rule blocks macro/profile work after focused
microbench failure.

| Workload | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| 4096/128 batch4 | not rerun | N/A | stopped before runtime |
| 4096/1024 batch4 | not rerun | N/A | stopped before runtime |

## Profile Owner / Backend Comparison

Fresh Nsight profile was not captured because no runtime candidate passed the
focused gate.

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Projection/GEMM bucket | `0.778887s` | N/A | N/A |
| BF16 small-GEMM cluster | `0.521619s` | N/A | N/A |
| FP8/custom projection cluster | `0` promoted | N/A | N/A |
| Graph replay count | not rerun | N/A | N/A |
| Eager decode count | not rerun | N/A | N/A |

## Memory / Workspace Ledger

Actual runtime memory delta from this target is zero because no opt-in
projection cache was implemented.

| Item | Bytes/rank | GiB/rank | KV tokens | Pages at 256 |
| --- | ---: | ---: | ---: | ---: |
| Runtime persistent cache added by 07.72 | `0` | `0.0000` | `0` | `0` |
| Runtime decode workspace added by 07.72 | `0` | `0.0000` | `0` | `0` |

Microbench-only fused FP8 duplicates for WQA/WKV and shared gate/up summed to
`8,389,120` bytes on the single measured layer.  They were not wired into the
runtime and are not a persistent model memory cost.

## Decision

Decision:

- Outcome: stop without runtime implementation.
- Selected lane: vLLM-aligned FP8/custom projection-cache boundary.
- Toggle/variant: none.
- Touched owners: none in runtime; focused probe measured WQA/WKV, `q_wqb`,
  `wo_b`, shared gate/up, shared down, and `wo_a`.
- Quality gate result: focused numerical error acceptable.
- Projection/GEMM delta: not profiled because focused latency gate failed.
- Touched cluster delta: not profiled because focused latency gate failed.
- 4096/1024 macro delta: not run because no runtime candidate was implemented.
- Memory/workspace cost: `0` bytes/rank actual runtime delta.
- Promote status: do not promote; no opt-in to keep.
- Next target: bounded vLLM SM80 quantized-linear backend feasibility, focused
  on whether MarlinFP8ScaledMM/int8-W8A8-style projection kernels can beat the
  promoted cached BF16 owners on real shapes.
- Stop condition for next target: stop without runtime integration if an
  isolated vLLM/mini-owned SM80 quantized-linear backend cannot execute on
  A100 or does not beat promoted cached BF16 by `>=15%` on at least two
  representative owners at M=`1,4,8,16`.
