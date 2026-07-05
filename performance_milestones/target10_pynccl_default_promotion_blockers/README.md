# TARGET 10.27: PyNCCL default-promotion blockers

Status: complete.

Decision: **default promote** for the DeepSeek V4 Flash TP8 A100/sm80 path.

Default promoted path:

```bash
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
# PyNCCL enabled by the promoted preset.
# On DeepSeek V4 sm80, the engine defaults PyNCCL max buffer size to 32M
# unless MINISGL_PYNCCL_MAX_BUFFER_SIZE is explicitly set.
```

Rollback remains simple:

```bash
MINISGL_PYNCCL_MAX_BUFFER_SIZE=1G
# or, for serving:
--disable-pynccl
# or, for benchmark comparison, use a non-PyNCCL preset / omit the promoted preset.
```

No new communication backend, low-precision path, attention kernel, prefix/SWA
ownership change, vLLM custom all-reduce port, or broad owner-routing layer was
added.

## Artifacts

- Fresh owner timing, Torch/NCCL:
  `raw/owner_timing_torch_nccl_hist1024_fresh/`
- Fresh owner timing, PyNCCL default threshold32m:
  `raw/owner_timing_pynccl_threshold32m_hist1024_fresh/`
- Owner timing summary:
  `summaries/lm_head_owner_timing_hist1024_summary.md`
- Standalone `DistributedCommunicator.all_gather` probe:
  `raw/all_gather_probe_torch.json`,
  `raw/all_gather_probe_pynccl_threshold32m.json`
- Probe summary:
  `summaries/all_gather_probe_summary.md`
- Full-model rank0 Nsight, serving:
  `raw/nsys_target1027_pynccl_full_model_serving_mixed_112req_wave16_rank0.nsys-rep`,
  `raw/nsys_target1027_pynccl_full_model_serving_mixed_112req_wave16_rank0.sqlite`
- Full-model rank0 Nsight, historical short/direct path:
  `raw/nsys_target1027_pynccl_full_model_historical_4096_128_bs4_rank0.nsys-rep`,
  `raw/nsys_target1027_pynccl_full_model_historical_4096_128_bs4_rank0.sqlite`
- Nsight summaries:
  `summaries/nsys_target1027_pynccl_full_model_serving_mixed_112req_wave16_rank0.md`,
  `summaries/nsys_target1027_pynccl_full_model_historical_4096_128_bs4_rank0.md`,
  `summaries/nsys_kernel_memcpy_summary.md`
- Default-promotion text smoke:
  `raw/text_smoke_default_pynccl32m.json`
- Default-promotion short macro sanity:
  `raw/macro_default_pynccl32m_hist128_sanity/`

## Code/docs changes

- `python/minisgl/engine/engine.py`
  - Defaults DSV4 sm80 PyNCCL max buffer size to `32M` only when
    `MINISGL_PYNCCL_MAX_BUFFER_SIZE` is not explicitly set.
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - Promotes `dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16`
    to `use_pynccl=True`.
- `benchmark/offline/deepseek_v4_text_smoke.py`
  - Same promoted preset update.
- `prompts/target.md` and
  `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
  - Updated TARGET 10.27 status and current default communication route.
- `scripts/`
  - Added target-local owner summary, all-gather probe, and rank-wrapper Nsight
    capture scripts.

## Prior macro gate

TARGET 10.26 remains the repeat-stable macro gate for the exact candidate.

| Scenario | Torch E2E tok/s | PyNCCL32M E2E tok/s | E2E delta | Torch decode tok/s | PyNCCL32M decode tok/s | Decode delta | Graph |
|---|---:|---:|---:|---:|---:|---:|---:|
| `historical_4096_128_bs4` | 55.767 | 56.469 | +1.26% | 181.941 | 190.202 | +4.54% | zero-eager |
| `historical_4096_1024_bs4` | 137.347 | 142.438 | +3.71% | 183.204 | 192.189 | +4.90% | zero-eager |
| `serving_mixed_112req_wave16` | 174.958 | 182.373 | +4.24% | 282.486 | 299.597 | +6.06% | zero-eager |
| `prefix_multi_112req_wave16` | 132.404 | 134.111 | +1.29% | 650.341 | 680.905 | +4.70% | zero-eager |

TARGET 10.27 did not invalidate those numbers; it resolved the two evidence
blockers and then re-ran smoke plus a short macro sanity after changing the
default plumbing.

## `lm_head_all_gather` owner timing

Conclusion: the TARGET 10.26 anomaly is a **non-captured first-call/phase
attribution cost**, not a PyNCCL all-gather hot-path regression.

Fresh `historical_4096_1024_bs4` owner-timing A/B:

| Run | total ms sum-rank | non-captured ms | captured ms | Timed samples | Top outlier |
|---|---:|---:|---:|---:|---|
| Torch/NCCL fresh | 62.692 | 62.258 | 0.434 | 56 | rank3 seq710 `[16,16160]` 5.742 ms |
| PyNCCL32M fresh | 5237.419 | 5236.936 | 0.482 | 56 | rank0 seq710 `[16,16160]` 655.042 ms |
| Torch/NCCL TARGET 10.26 | 74.696 | 74.220 | 0.476 | 56 | rank3 seq710 `[16,16160]` 6.713 ms |
| PyNCCL32M TARGET 10.26 | 4821.071 | 4820.616 | 0.455 | 56 | rank1 seq710 `[16,16160]` 601.362 ms |

Captured replay stays neutral:

| Run | captured shape | captured count | captured sum-rank ms | max single-rank captured ms |
|---|---|---:|---:|---:|
| Torch/NCCL fresh | `[4,16160]` | 8 | 0.434 | 0.058 |
| PyNCCL32M fresh | `[4,16160]` | 8 | 0.482 | 0.067 |
| Torch/NCCL TARGET 10.26 | `[4,16160]` | 8 | 0.476 | 0.068 |
| PyNCCL32M TARGET 10.26 | `[4,16160]` | 8 | 0.455 | 0.063 |

Shape/outlier split for the fresh PyNCCL run:

| Captured | Shape | Count | Sum ms | Mean ms | Max ms |
|---:|---|---:|---:|---:|---:|
| 0 | `[16,16160]` | 8 | 5211.272 | 651.409 | 655.042 |
| 0 | `[8,16160]` | 8 | 8.231 | 1.029 | 1.382 |
| 0 | `[4,16160]` | 16 | 4.441 | 0.278 | 0.682 |
| 0 | `[2,16160]` | 8 | 4.265 | 0.533 | 1.639 |
| 0 | `[1,16160]` | 8 | 8.728 | 1.091 | 1.671 |
| 1 | `[4,16160]` | 8 | 0.482 | 0.060 | 0.067 |

Interpretation:

- The PyNCCL spike is rank-wide and deterministic: every rank's top sample is
  seq `710`, shape `[16,16160]`, non-captured.
- Later non-captured shapes are normal, and captured replay is neutral.
- The owner timing snapshot covers graph-capture/warmup samples, while the macro
  elapsed/decode metrics cover measured repeats. The large first-call sample is
  therefore mis-attributed if the owner table is read as hot-path decode time.
- TARGET 10.26's `serving_mixed_112req_wave16` owner report also inherited the
  same earlier seq `710` outlier because owner timing samples are not reset
  between cases in that run. The fresh single-scenario rerun confirms this is
  not a serving-specific recurring all-gather problem.

No lm-head rollback or owner routing fix is needed.

## Standalone all-gather probe

The probe uses:

```text
DistributedCommunicator.all_gather(..., label="dsv4.lm_head_all_gather")
```

It exercises Torch/NCCL and PyNCCL threshold32m with real lm-head fp32 shard
shapes. Correctness passed for all ranks and shapes.

| Backend | Shape | first event max ms | warmed eager mean ms | warmed eager max ms | graph mean ms | graph max ms | Correct |
|---|---|---:|---:|---:|---:|---:|---|
| Torch/NCCL | `[16,16160]` | 1343.938 | 0.322 | 2.354 | 0.139 | 0.350 | true |
| PyNCCL32M | `[16,16160]` | 646.057 | 0.315 | 3.214 | 0.133 | 0.236 | true |
| Torch/NCCL | `[4,16160]` | 0.166 | 0.162 | 0.597 | 0.110 | 0.586 | true |
| PyNCCL32M | `[4,16160]` | 0.136 | 0.116 | 0.171 | 0.393 | 7.200 | true |
| Torch/NCCL | `[1,16160]` | 0.452 | 0.151 | 0.475 | 0.072 | 0.194 | true |
| PyNCCL32M | `[1,16160]` | 0.146 | 0.108 | 0.316 | 0.075 | 0.193 | true |

The standalone probe reproduces a one-time first-call cost, then shows warmed
eager and graph paths are neutral to slightly positive. This classifies the
model owner anomaly as `measurement/phase attribution artifact`, not a broad
PyNCCL all-gather regression.

## Full-model Nsight

TARGET 10.26's full serving Nsight attempt saved OS runtime only. TARGET 10.27
uses a target-local rank wrapper:

```bash
torchrun --standalone --nproc_per_node=8 --no-python \
  performance_milestones/target10_pynccl_default_promotion_blockers/scripts/nsys_rank_wrapper.sh \
  python benchmark/offline/deepseek_v4_perf_matrix.py ...
```

Rank0 is profiled by Nsight; ranks 1-7 run the same full TP8 model path.

Serving rank0 full-model profile:

| Section | Kernels | Graph trace | Runtime APIs | Memcpy | NCCL kernels | CUDA graph launches |
|---|---:|---:|---:|---:|---:|---:|
| total | 381082 / 5.122601 s | 441 / 8.971415 s | 810468 / 31.352673 s | 216214 / 196460160471 B | 1056 / 0.196963 s | 441 |
| repeat window | 177278 / 3.737685 s | 441 / 8.971415 s | 269716 / 12.761140 s | 57834 / 27386079063 B | 616 / 0.170391 s | 441 |

Serving repeat-window NCCL kernels:

| Kernel | Count | Duration s | Meaning |
|---|---:|---:|---|
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16` | 609 | 0.169882 | small BF16 all-reduces use PyNCCL symmetric path |
| `ncclDevKernel_AllGather_RING_LL` | 7 | 0.000509 | lm-head all-gather direct NCCL output |

Serving repeat-window memcpy:

| Kind | Count | Bytes | Duration s | Meaning |
|---|---:|---:|---:|---|
| Device-to-Device | 38227 | 27384962570 | 0.094592 | dominated by expected symmetric all-reduce copy-in/out |
| Host-to-Device pinned | 1932 | 872704 | 0.002159 | scheduler/metadata copies |
| Device-to-Host | 16107 | 218701 | 0.034641 | scalar/status reads |

Historical short rank0 full-model profile:

| Section | Kernels | Graph trace | Runtime APIs | Memcpy | NCCL kernels | CUDA graph launches |
|---|---:|---:|---:|---:|---:|---:|
| total | 262071 / 5.798571 s | 127 / 2.544768 s | 717795 / 22.730497 s | 196984 / 171422790511 B | 528 / 0.261941 s | 127 |
| repeat window | 58357 / 4.328672 s | 127 / 2.544768 s | 165738 / 6.164147 s | 38634 / 2348709343 B | 88 / 0.105670 s | 127 |

Historical repeat-window NCCL kernels:

| Kernel | Count | Duration s | Meaning |
|---|---:|---:|---|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 87 | 0.105632 | large BF16 all-reduces are above 32M and use direct NCCL |
| `ncclDevKernel_AllGather_RING_LL` | 1 | 0.000038 | lm-head all-gather direct NCCL output |

Nsight conclusion:

- Full-model/rank-scoped CUDA activity is present and non-empty.
- CUDA graph launches are visible and match benchmark graph replay counts
  (`441/0` serving, `127/0` historical short).
- Serving confirms the small BF16 all-reduce symmetric path.
- Historical short confirms the large BF16 all-reduce direct NCCL path.
- lm-head all-gather appears as `ncclDevKernel_AllGather_RING_LL`; PyNCCL source
  writes all-gather directly to the output tensor and does not use the symmetric
  internal buffer for all-gather.
- The full-model traces do not contradict the TARGET 10.26 clean representative
  probe.

## Verification after default change

Syntax:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target10_pynccl_default_promotion_blockers/scripts/all_gather_probe.py \
  performance_milestones/target10_pynccl_default_promotion_blockers/scripts/summarize_lm_head_owner_timing.py
```

Default-promotion text smoke, without `--use-pynccl` and without
`MINISGL_PYNCCL_MAX_BUFFER_SIZE`:

| Item | Result |
|---|---|
| Status | pass |
| `use_pynccl` in output config | true |
| Graph replay/eager | 9 / 0 |
| Outputs | same sane outputs as TARGET 10.26 |
| Default threshold log | `Defaulting DeepSeek V4 sm80 PyNCCL max buffer size to 32 MiB` |

Short macro sanity, without `--use-pynccl` and without
`MINISGL_PYNCCL_MAX_BUFFER_SIZE`:

| Scenario | Status | Elapsed s | E2E output tok/s | Decode tok/s | Graph replay/eager |
|---|---|---:|---:|---:|---:|
| `historical_4096_128_bs4` | pass | 9.604827 | 53.306528 | 190.405489 | 127 / 0 |

This sanity run verifies the promoted preset and engine default plumbing. The
repeat-stable macro performance decision remains the TARGET 10.26 gate.

## Final decision

Classification: **default promote**.

Gate checklist:

| Gate | Result |
|---|---|
| TARGET 10.26 repeat-stable macro wins remain valid | pass |
| Text smoke after default change | pass |
| Graph replay remains zero-eager | pass |
| `lm_head_all_gather` anomaly explained or fixed | pass, explained as one-time non-captured first all-gather/phase attribution |
| Full-model or rank-scoped Nsight captures CUDA activity | pass |
| Nsight supports expected threshold32m behavior | pass: small BF16 symmetric, large BF16 direct, lm-head direct all-gather |
| Rollback simple | pass |

PyNCCL threshold32m can enter the default A100/sm80 DeepSeek V4 path.
