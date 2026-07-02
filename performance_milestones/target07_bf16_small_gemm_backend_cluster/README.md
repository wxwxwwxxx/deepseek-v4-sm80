# TARGET 07.70 BF16 Small-GEMM Backend Cluster

## Objective

Investigate exact-route backend changes for the TARGET 07.69 BF16 small-GEMM plus splitK/reduce cluster on the SM80 A100 victory baseline. The entry gate allowed a runtime opt-in only after focused microbench evidence showed at least two representative BF16 owners improving by at least 15% with correctness aligned.

Baseline reference:

- Variant: `dsv4_sm80_a100_victory`
- Env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- Profile source: `performance_milestones/target07_projection_gemm_backend_owner_reattribution/`
- Projection/GEMM bucket: `0.778887s`
- BF16 small-GEMM plus splitK/reduce cluster: `0.521619s`

## Candidate

The candidate is explicit opt-in only:

- Variant: `dsv4_sm80_a100_victory_bf16smallgemm`
- Toggle: `MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1`
- Runtime behavior: build pretransposed BF16 cached weights before CUDA graph capture, then route small decode GEMMs with `rows <= 16` through `torch.mm(x_2d, weight_t)`.
- Owners covered: fused WQA/WKV/compress, `attn.q_wqb`, `attn.wo_b`, `indexer.wq_b`, shared expert gate/up, shared expert down.
- Owners not changed: HC pre linear, MoE router FP32/SGEMM paths, `attn.wo_a` grouped BMM.

The toggle is registered as an experimental known toggle but is intentionally not part of `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE`.

## Focused Microbench Gate

Artifact: `raw/focused_bf16_small_gemm_microbench.{json,md}`

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
python performance_milestones/target07_bf16_small_gemm_backend_cluster/scripts/focused_bf16_small_gemm_microbench.py \
  --model-path /models/DeepSeek-V4-Flash \
  --layer 10 \
  --tokens 1 4 8 16 \
  --warmup 50 \
  --iters 200 \
  --output performance_milestones/target07_bf16_small_gemm_backend_cluster/raw/focused_bf16_small_gemm_microbench.json
```

Representative `M=4` results:

| Owner | Baseline mean ms | Best mean ms | Best route | Speedup | Max abs err |
| --- | ---: | ---: | --- | ---: | ---: |
| WQA/WKV/compress | `0.043737` | `0.036254` | pretransposed mm | `17.03%` | `0` |
| attention q_wqb | `0.037807` | `0.032045` | pretransposed mm | `15.77%` | `0` |
| attention wo_b | `0.037843` | `0.031373` | pretransposed mm | `17.16%` | `0` |
| shared expert gate/up | `0.043571` | `0.035794` | pretransposed mm | `17.70%` | `0.015625` |
| shared expert down | `0.038090` | `0.033002` | pretransposed mm | `14.03%` | `0` |
| indexer wq_b | `0.038207` | `0.032341` | pretransposed mm | `15.70%` | `0` |

Gate result: pass. Multiple representative BF16 owners improved by at least 15%, so the runtime opt-in and harness variant were implemented.

## Correctness And Graph Smoke

Artifact: `raw/text_smoke_bf16smallgemm_candidate_first.dsv4_sm80_a100_victory_bf16smallgemm.json`

Result:

- Status: `pass`
- Graph replay count: `9`
- Eager decode count: `0`
- Pretranspose cache: enabled
- Pretranspose cache bytes per rank: `1,885,339,648` bytes (`1.7559 GiB`)
- Peak allocated memory rank 0: `44,978,433,024` bytes

Memory ledger:

- Incremental pretranspose cache: `1.7559 GiB/rank`
- Approximate KV-equivalent capacity: `24,796` KV tokens, or `96.86` pages at page size `256`
- Lifetime: built during model CUDA graph preparation, before decode graph capture; no per-decode rebuild is allowed.

## Macro Results

Artifacts:

- `raw/macro_4096x128_bs4_np128/summary.json`
- `raw/macro_4096x1024_bs4_np128/summary.json`

| Scenario | Variant | Output tok/s | Decode tok/s | Replay | Eager decode | Delta vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128 bs4 np128 | candidate | `62.3750` | `169.5459` | `1016` | `0` | `+0.08%` |
| 4096/128 bs4 np128 | baseline | `62.3274` | `169.2381` | `1016` | `0` | reference |
| 4096/1024 bs4 np128 | candidate | `131.9084` | `169.6304` | `8184` | `0` | `+0.09%` |
| 4096/1024 bs4 np128 | baseline | `131.7927` | `169.3913` | `8184` | `0` | reference |

Macro gate result:

- 4096/128 no-regression gate: pass.
- 4096/1024 `+3%` gate: fail.

## Nsight Classification

Artifacts:

- `summaries/projection_gemm_owner_table.md`
- `summaries/projection_gemm_backend_families.md`
- `summaries/projection_gemm_top_kernels.md`
- SQLite and `.nsys-rep` links are in `raw/` and point to the `/tmp` exports generated for this run.

Candidate profile:

- Decode envelope: `3.567932s`
- Projection/GEMM bucket: `0.778170s` / `100965` kernels
- BF16 small-GEMM plus splitK/reduce cluster: `0.521012s`

Profile deltas vs 07.69 baseline:

| Metric | Baseline | Candidate | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Projection/GEMM bucket | `0.778887s` | `0.778170s` | `-0.000717s` (`-0.09%`) | fail `-0.10s` |
| BF16 cluster | `0.521619s` | `0.521012s` | `-0.000607s` (`-0.12%`) | fail `-20%` |
| cuBLASLt BF16 GEMM | `0.219912s` | `0.313612s` | `+42.61%` | shifted worse |
| CUTLASS BF16 GEMM | `0.194319s` | `0.095964s` | `-50.62%` | shifted better |
| cuBLASLt splitK/reduce | `0.107388s` | `0.111436s` | `+3.77%` | shifted worse |

Interpretation: pretransposing the BF16 weights changed the backend-family mix, especially moving time out of CUTLASS BF16 kernels and into cuBLASLt BF16 kernels, but the aggregate projection/GEMM bucket and BF16 cluster remained flat.

## Decision

Do not promote this candidate into the A100 victory bundle.

Keep `dsv4_sm80_a100_victory_bf16smallgemm` as an explicit audit/profiling variant because it passed microbench correctness and graph smoke, but it does not satisfy the macro or profile promotion gates. The next target should move away from this narrow BF16 layout route and focus on a precision-policy or broader backend change that can reduce the remaining graph-replay bucket directly, such as HC/router FP32 ownership, FP8 projection/cache layout, or vLLM-aligned fused indexer/rope/quant paths.
