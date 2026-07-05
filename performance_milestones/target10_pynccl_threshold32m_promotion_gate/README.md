# TARGET 10.26: PyNCCL threshold32m Promotion Gate

Status: complete.

Decision: **recommended opt-in** for the fixed DeepSeek V4 Flash TP8 A100/SM80
path with BF16 MoE reduce enabled:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

This is promoted out of experimental-only status, but it should not replace the
Torch/NCCL default yet. The gate is repeat-stable on the main long-decode and
serving cases, stays zero-eager, has no 4096x128 or prefix regression, and owner
timing points at communication-owner gains. Rollback remains one env/flag
change: remove `--use-pynccl` and `MINISGL_PYNCCL_MAX_BUFFER_SIZE`.

No production communication routing, vLLM custom all-reduce port, P2P/IPC
collective, low-precision route, attention kernel, or prefix/SWA ownership
change was implemented. The only source artifact added in this milestone is a
small profile probe under `scripts/` to confirm existing PyNCCL threshold32m
kernel behavior.

## Artifacts

- Text smoke: `raw/text_smoke_pynccl_threshold32m.json`
- Macro baseline r1-r2: `raw/macro_torch_nccl_fixedbf16_r2/`
- Macro candidate r1-r2: `raw/macro_pynccl_threshold32m_r2/`
- Macro baseline r3 extra: `raw/macro_torch_nccl_fixedbf16_r3extra/`
- Macro candidate r3 extra: `raw/macro_pynccl_threshold32m_r3extra/`
- Owner timing baseline: `raw/owner_timing_torch_nccl_hist1024_serving/`
- Owner timing candidate: `raw/owner_timing_pynccl_threshold32m_hist1024_serving/`
- PyNCCL kernel probe: `raw/pynccl_threshold32m_kernel_probe_nsys_clean.json`
- Nsight kernel CSV:
  `summaries/nsys_pynccl_kernel_probe_clean_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- Nsight memcpy CSV:
  `summaries/nsys_pynccl_kernel_probe_clean_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`

## Commands

Common fixed flags:

```bash
--model-path /models/DeepSeek-V4-Flash
--variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Baseline env/backend:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
# Torch/NCCL default path; no --use-pynccl
```

Candidate env/backend:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

Macro r1-r2 command shape:

```bash
timeout 3600 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  ${COMMON_FLAGS} \
  --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 \
    serving_mixed_112req_wave16 prefix_multi_112req_wave16 \
  --repeats 2 --warmup-repeats 0 --seed 20260705 --keep-going
```

The r3 extra pass used the same command shape with `--repeats 1 --seed
20260707`. Owner timing added `MINISGL_DSV4_OWNER_TIMING=1` and covered
`historical_4096_1024_bs4` plus `serving_mixed_112req_wave16`.

Text smoke candidate command:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 \
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M \
timeout 1800 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  ${COMMON_FLAGS} \
  --output raw/text_smoke_pynccl_threshold32m.json \
  --max-tokens 32 --fail-on-warning --use-pynccl
```

## Text Smoke

| Item | Result |
|---|---|
| Status | pass |
| Graph replay/eager | 9 / 0 |
| Errors | none, `errors=[]` |
| Warning scan | no real warnings/errors in text smoke log; only JSON `error: null` fields |

Outputs:

| Prompt | Output |
|---:|---|
| 0 | `2 + 2 等于 4。` |
| 1 | `The sky is blue on a clear day.` |
| 2 | `杭州：人间天堂，西湖美景。` |

## Repeat-Stable Macro Gate

Report-level metrics use the benchmark aggregation rule: elapsed and forward
phase times are max-across-rank, output/decode tokens are rank0 counts. r3 was
added because the short and prefix cases are the no-regression guardrails.

| Scenario | Torch elapsed | PyNCCL elapsed | elapsed delta | Torch E2E | PyNCCL E2E | E2E delta | Torch decode | PyNCCL decode | decode delta | Torch graph | PyNCCL graph | Prefix hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `historical_4096_128_bs4` | 27.543 | 27.201 | -1.24% | 55.767 | 56.469 | +1.26% | 181.941 | 190.202 | +4.54% | 381/0 | 381/0 | 0.000 |
| `historical_4096_1024_bs4` | 89.467 | 86.269 | -3.57% | 137.347 | 142.438 | +3.71% | 183.204 | 192.189 | +4.90% | 3069/0 | 3069/0 | 0.000 |
| `serving_mixed_112req_wave16` | 48.012 | 46.059 | -4.07% | 174.958 | 182.373 | +4.24% | 282.486 | 299.597 | +6.06% | 1323/0 | 1323/0 | 0.000 |
| `prefix_multi_112req_wave16` | 20.301 | 20.043 | -1.27% | 132.404 | 134.111 | +1.29% | 650.341 | 680.905 | +4.70% | 147/0 | 147/0 | 0.414 |

Per-repeat rows:

| Scenario | Repeat | Torch elapsed | PyNCCL elapsed | elapsed delta | Torch E2E | PyNCCL E2E | E2E delta | Torch decode | PyNCCL decode | decode delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `historical_4096_128_bs4` | 1 | 9.813 | 9.619 | -1.98% | 52.176 | 53.230 | +2.02% | 180.946 | 189.806 | +4.90% |
| `historical_4096_128_bs4` | 2 | 8.112 | 7.966 | -1.80% | 63.118 | 64.275 | +1.83% | 183.063 | 191.550 | +4.64% |
| `historical_4096_128_bs4` | 3 | 9.618 | 9.616 | -0.02% | 53.231 | 53.242 | +0.02% | 181.810 | 189.264 | +4.10% |
| `historical_4096_1024_bs4` | 1 | 29.878 | 28.829 | -3.51% | 137.090 | 142.079 | +3.64% | 182.762 | 191.547 | +4.81% |
| `historical_4096_1024_bs4` | 2 | 29.805 | 28.679 | -3.78% | 137.426 | 142.823 | +3.93% | 183.643 | 193.050 | +5.12% |
| `historical_4096_1024_bs4` | 3 | 29.784 | 28.761 | -3.43% | 137.526 | 142.415 | +3.56% | 183.208 | 191.976 | +4.79% |
| `serving_mixed_112req_wave16` | 1 | 16.177 | 15.520 | -4.06% | 173.090 | 180.407 | +4.23% | 275.094 | 294.040 | +6.89% |
| `serving_mixed_112req_wave16` | 2 | 15.863 | 15.296 | -3.58% | 176.506 | 183.054 | +3.71% | 286.265 | 303.416 | +5.99% |
| `serving_mixed_112req_wave16` | 3 | 15.972 | 15.243 | -4.56% | 175.306 | 183.687 | +4.78% | 285.497 | 301.500 | +5.61% |
| `prefix_multi_112req_wave16` | 1 | 6.781 | 6.720 | -0.90% | 132.131 | 133.326 | +0.90% | 652.970 | 678.665 | +3.94% |
| `prefix_multi_112req_wave16` | 2 | 6.865 | 6.742 | -1.80% | 130.513 | 132.907 | +1.83% | 647.787 | 683.647 | +5.54% |
| `prefix_multi_112req_wave16` | 3 | 6.655 | 6.582 | -1.10% | 134.634 | 136.134 | +1.11% | 650.286 | 680.421 | +4.63% |

Interpretation:

- Main long decode `historical_4096_1024_bs4`: repeat-stable +3.71% E2E,
  +4.90% decode, zero-eager.
- Serving mixed wave: repeat-stable +4.24% E2E, +6.06% decode, zero-eager.
- Short 4096x128 guardrail: E2E is near neutral to small positive, decode is
  consistently positive, no regression.
- Prefix guardrail: E2E is small positive, decode is consistently positive, hit
  rate is unchanged at 0.414, no regression.

## Communication Stats

Production communication counters are identical before/after because the
candidate changes backend implementation, not model owner labels or tensor
shapes.

| Scenario | Row | Count | GiB | Graph | Prefix hit |
|---|---|---:|---:|---:|---:|
| `historical_4096_128_bs4` | Torch r1-r2 | 1408 | 174.031 | 254/0 | 0.000 |
| `historical_4096_128_bs4` | PyNCCL r1-r2 | 1408 | 174.031 | 254/0 | 0.000 |
| `historical_4096_128_bs4` | Torch r3 | 704 | 87.015 | 127/0 | 0.000 |
| `historical_4096_128_bs4` | PyNCCL r3 | 704 | 87.015 | 127/0 | 0.000 |
| `historical_4096_1024_bs4` | Torch r1-r2 | 1408 | 174.031 | 2046/0 | 0.000 |
| `historical_4096_1024_bs4` | PyNCCL r1-r2 | 1408 | 174.031 | 2046/0 | 0.000 |
| `historical_4096_1024_bs4` | Torch r3 | 704 | 87.015 | 1023/0 | 0.000 |
| `historical_4096_1024_bs4` | PyNCCL r3 | 704 | 87.015 | 1023/0 | 0.000 |
| `serving_mixed_112req_wave16` | Torch r1-r2 | 9856 | 186.418 | 882/0 | 0.000 |
| `serving_mixed_112req_wave16` | PyNCCL r1-r2 | 9856 | 186.418 | 882/0 | 0.000 |
| `serving_mixed_112req_wave16` | Torch r3 | 4928 | 93.209 | 441/0 | 0.000 |
| `serving_mixed_112req_wave16` | PyNCCL r3 | 4928 | 93.209 | 441/0 | 0.000 |
| `prefix_multi_112req_wave16` | Torch r1-r2 | 9856 | 163.988 | 98/0 | 0.414 |
| `prefix_multi_112req_wave16` | PyNCCL r1-r2 | 9856 | 163.988 | 98/0 | 0.414 |
| `prefix_multi_112req_wave16` | Torch r3 | 4928 | 81.994 | 49/0 | 0.414 |
| `prefix_multi_112req_wave16` | PyNCCL r3 | 4928 | 81.994 | 49/0 | 0.414 |

Single-repeat by-label pattern is also identical before/after:

| Scenario | Label | Count | GiB |
|---|---|---:|---:|
| `historical_4096_*_bs4` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | 344 | 43.000 |
| `historical_4096_*_bs4` | `dsv4.embedding_all_reduce` | 8 | 1.000 |
| `historical_4096_*_bs4` | `dsv4.lm_head_all_gather` | 8 | 0.015 |
| `historical_4096_*_bs4` | `dsv4.v1_moe_reduce_once_all_reduce` | 344 | 43.000 |
| `serving_mixed_112req_wave16` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | 2408 | 45.855 |
| `serving_mixed_112req_wave16` | `dsv4.embedding_all_reduce` | 56 | 1.066 |
| `serving_mixed_112req_wave16` | `dsv4.lm_head_all_gather` | 56 | 0.432 |
| `serving_mixed_112req_wave16` | `dsv4.v1_moe_reduce_once_all_reduce` | 2408 | 45.855 |
| `prefix_multi_112req_wave16` | `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | 2408 | 40.312 |
| `prefix_multi_112req_wave16` | `dsv4.embedding_all_reduce` | 56 | 0.938 |
| `prefix_multi_112req_wave16` | `dsv4.lm_head_all_gather` | 56 | 0.432 |
| `prefix_multi_112req_wave16` | `dsv4.v1_moe_reduce_once_all_reduce` | 2408 | 40.312 |

## Owner Timing / Profile

Owner timing is instrumentation-heavy, so use it for attribution rather than
absolute macro speed. It covered the required long decode and serving scenarios.
Both runs stayed zero-eager: `historical_4096_1024_bs4` was 1023/0 and
`serving_mixed_112req_wave16` was 441/0.

| Scenario | Torch elapsed | PyNCCL elapsed | Delta | Torch decode | PyNCCL decode | Delta |
|---|---:|---:|---:|---:|---:|---:|
| `historical_4096_1024_bs4` | 44.638 | 41.448 | -7.15% | 120.444 | 132.875 | +10.32% |
| `serving_mixed_112req_wave16` | 20.634 | 19.256 | -6.68% | 193.961 | 215.663 | +11.19% |

Communication owner CUDA timing:

| Scenario | Label | Count B/C | sum-rank ms Torch | sum-rank ms PyNCCL | Delta | captured ms Torch | captured ms PyNCCL |
|---|---|---:|---:|---:|---:|---:|---:|
| `historical_4096_1024_bs4` | `dsv4.owner.attn.wo_b.row_parallel_all_reduce` | 3784/3784 | 17891.615 | 13945.349 | -22.06% | 17.174 | 10.906 |
| `historical_4096_1024_bs4` | `dsv4.owner.moe.reduce_once_all_reduce` | 3784/3784 | 4359.992 | 3964.550 | -9.07% | 16.273 | 11.293 |
| `historical_4096_1024_bs4` | `dsv4.owner.comm.dsv4.embedding_all_reduce` | 88/88 | 11499.549 | 2409.684 | -79.05% | 1.304 | 3.631 |
| `historical_4096_1024_bs4` | `dsv4.owner.comm.dsv4.lm_head_all_gather` | 88/88 | 74.696 | 4821.071 | +6354.26% | 0.476 | 0.455 |
| `serving_mixed_112req_wave16` | `dsv4.owner.attn.wo_b.row_parallel_all_reduce` | 4128/4128 | 18155.197 | 14205.798 | -21.75% | 85.123 | 49.767 |
| `serving_mixed_112req_wave16` | `dsv4.owner.moe.reduce_once_all_reduce` | 4128/4128 | 4566.853 | 4139.812 | -9.35% | 81.770 | 52.566 |
| `serving_mixed_112req_wave16` | `dsv4.owner.comm.dsv4.embedding_all_reduce` | 96/96 | 11557.478 | 2453.827 | -78.77% | 10.620 | 6.018 |
| `serving_mixed_112req_wave16` | `dsv4.owner.comm.dsv4.lm_head_all_gather` | 96/96 | 77.727 | 4823.996 | +6106.30% | 2.754 | 2.630 |

Interpretation:

- The hot BF16 all-reduce owners improve materially with PyNCCL threshold32m.
- Owner counts are unchanged, which argues against hidden communication fallback
  or owner/shape drift.
- `lm_head_all_gather` has a large non-captured total-time anomaly in owner
  timing, but captured hot-path time is essentially unchanged and macro
  throughput remains positive. This is the main reason to keep the decision at
  recommended opt-in rather than default promotion.

## NCCL Kernel Family / Count

A full serving rank0 Nsight attempt produced `raw/nsys_pynccl_threshold32m_serving_rank0.nsys-rep`,
but after interrupting a long exit hang it contained OS runtime data only and no
CUDA activity, so it is not used as evidence.

To get a clean kernel check, `scripts/pynccl_threshold32m_kernel_probe.py`
profiles the existing PyNCCL threshold32m communicator on rank0 with:

- BF16 all-reduce `[2496,4096]`, 19.5 MiB, expected symmetric path;
- BF16 all-reduce `[16384,4096]`, 128 MiB, expected direct path;
- FP32 all-gather `[16,16160] -> [128,16160]`, expected direct output.

The clean profile used one warmup plus four measured iterations. All correctness
checks passed.

| Kernel family | Instances | Meaning |
|---|---:|---|
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16` | 5 | small BF16 all-reduce through PyNCCL symmetric path |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 5 | large BF16 all-reduce through direct NCCL path |
| `ncclDevKernel_AllGather_RING_LL` | 5 | all-gather direct output |

There are no f32 all-reduce kernels in the clean threshold probe. That matches
the fixed BF16 MoE reduce candidate.

## D2D Copy Accounting

Nsight memcpy table for the clean kernel probe:

| Operation | Count | Bytes | Interpretation |
|---|---:|---:|---|
| Device-to-Device | 10 | 204,472,320 | 5 small symmetric all-reduces x copy-in/copy-out x 20,447,232 bytes |
| Host-to-Device | 338 | 34,048 | setup noise |
| Device-to-Host | 3 | 3 | correctness scalar checks |

This exactly matches the PyNCCL threshold32m implementation: all-reduce tensors
at or below 32 MiB use the symmetric buffer with two D2D copies, larger
all-reduce tensors run direct, and all-gather writes directly to output.

Expected full-model symmetric D2D copies:

| Scenario | Per-repeat D2D | Three-repeat D2D | Reason |
|---|---:|---:|---|
| `historical_4096_128_bs4` | 0 GiB | 0 GiB | hidden all-reduces are 128 MiB, above threshold, direct |
| `historical_4096_1024_bs4` | 0 GiB | 0 GiB | same hidden all-reduce size, direct |
| `serving_mixed_112req_wave16` | 185.555 GiB | 556.664 GiB | 19.5 MiB hidden all-reduces use symmetric copy-in/copy-out |
| `prefix_multi_112req_wave16` | 65.250 GiB | 195.750 GiB | only `[1024,4096]` small hidden all-reduces use symmetric copies |

No new D2D amplification beyond the expected symmetric all-reduce copy-in and
copy-out was observed in the clean kernel probe. The macro D2D accounting is
consistent with TARGET 10.25's route replay accounting.

## Promotion Decision

Classification: **recommended opt-in**.

Promotion bar check:

| Gate | Result |
|---|---|
| Text smoke pass | pass |
| Graph replay zero-eager | pass across smoke, macro, owner timing |
| Main long decode or serving repeat-stable E2E >= +2% | pass: +3.71% on `historical_4096_1024_bs4`, +4.24% on serving |
| 4096x128 and prefix no obvious regression | pass: +1.26% and +1.29% E2E, decode positive |
| Owner timing/profile supports communication benefit or no hidden fallback | pass: hot all-reduce owners improve, owner counts unchanged, clean kernel probe matches threshold path |
| Rollback to Torch/NCCL simple | pass: remove `--use-pynccl` and `MINISGL_PYNCCL_MAX_BUFFER_SIZE` |

Why not default promote yet:

- `lm_head_all_gather` owner timing has a non-captured total-time anomaly even
  though captured hot-path time is neutral and macro wins.
- The full serving Nsight trace did not save CUDA activity; the kernel evidence
  is from a clean representative probe, not the full model serving profile.
- Torch/NCCL remains the lowest-risk default and is still the simplest rollback.

## Next Steps

1. Document this as the recommended A100/SM80 TP8 opt-in command for the fixed
   BF16 MoE reduce path.
2. Keep Torch/NCCL as the default backend until a full-model serving Nsight run
   cleanly captures CUDA activity and the `lm_head_all_gather` timing anomaly is
   explained or shown harmless across more runs.
3. Do not add owner/size routing in this target. If communication work resumes,
   make it a separate target with explicit scope and cheap gates.
4. Production rollback remains:

```bash
unset MINISGL_PYNCCL_MAX_BUFFER_SIZE
# remove --use-pynccl
```
