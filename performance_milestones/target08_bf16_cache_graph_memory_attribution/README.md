# TARGET 08.07 BF16 Cache Graph Memory Attribution

## Recommendation

Carry the CUDA graph capture memory cost into TARGET 08.10 and TARGET 08.18.

The promoted BF16 cache paths are not a material cause of the `~19 GiB/rank`
CUDA graph private-pool delta.  They are visible as pre-capture persistent
baseline memory, but disabling them does not reduce first-graph capture memory.

No owner crossed the `2 GiB/rank` small-fix threshold.  No owner crossed even the
`1 GiB/rank` phase-2 retest threshold, so no full-bucket A/B retest was needed
beyond the full victory baseline.

## Scope

Promoted exact path:

- Variant: `dsv4_sm80_a100_victory`
- Env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- Page size: `256`
- Fixed pages: `--num-pages 128`
- Workload: `decode_ladder_bs16`
- Phase 1 graph bucket: `[16]`
- Full baseline graph buckets: `[1,2,4,8,16]`
- Lifecycle: each point was a separate `torchrun`

All GiB values use bytes / 2^30.

## Instrumentation

Added a small attribution-only false override hook:

```text
MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES
```

It is empty by default and changes no promoted serving behavior.  When set, it
is evaluated before direct env flags and before the victory whitelist, so it can
really disable a bundle-enabled cache.  Accepted aliases include `q_wqb`,
`wo_b`, `wo_a`, `indexer_wq_b`, `shared_expert`,
`projection_bf16_caches`, and `all_tested_bf16_caches`.

Code changes:

- `python/minisgl/kernel/deepseek_v4.py`
  - adds the denylist env and alias expansion
  - makes denylisted toggles return false in `dsv4_env_flag()`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - preserves this attribution env while configuring the promoted variant
  - records it in `raw_dsv4_sm80_env`
- `python/minisgl/models/deepseek_v4.py`
  - records `attribution_disable_toggles` in `model_prepare_report`
- tests cover victory whitelist false override and perf-matrix preservation

## Exact Commands

Main experiment:

```bash
cd /workspace/mini-sglang
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
performance_milestones/target08_bf16_cache_graph_memory_attribution/scripts/run_bf16_cache_graph_memory_attribution_matrix.sh
```

The script expands each case to this shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES=<denylist> \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_ladder_bs16 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 2048 \
  --repeats 1 \
  --warmup-repeats 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs <16-or-1-2-4-8-16> \
  --cuda-graph-capture-greedy-sample \
  --output-dir performance_milestones/target08_bf16_cache_graph_memory_attribution/raw/<run> \
  --keep-going
```

Summary:

```bash
python performance_milestones/target08_bf16_cache_graph_memory_attribution/scripts/summarize_bf16_cache_graph_memory.py \
  --milestone-dir performance_milestones/target08_bf16_cache_graph_memory_attribution
```

Validation:

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  performance_milestones/target08_bf16_cache_graph_memory_attribution/scripts/summarize_bf16_cache_graph_memory.py

pytest -q \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_preserves_victory_disable_toggles \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_records_wo_a_bf16_bmm_cache
```

Validation result: `3 passed`.

## Hardware And Software

| Item | Value |
| --- | --- |
| GPUs | 8x `NVIDIA A100-SXM4-80GB` |
| GPU memory | `81920 MiB` per GPU |
| Driver | `570.172.08` |
| CUDA capability | `sm80` |
| CUDA runtime | `12.8` |
| Python | `3.12.3` |
| PyTorch | `2.9.1+cu128` |
| Triton | `3.5.1` |
| NCCL | `2.27.5` |
| flashinfer | `0.6.12` |
| sgl_kernel | `0.3.21` |
| tilelang | `0.1.11+cu128.gita417b38a` |
| Branch | `dsv4-sglang-based` |
| Commit | `e3ca766ec0ad477416d2e025fe5f43cf861bf663` |
| Worktree | dirty due to instrumentation and this milestone |

## Memory Classes

| Class | Result |
| --- | --- |
| Persistent BF16 cache baseline | Full victory prepare allocates `1.588 GiB/rank` across tested BF16 cache owners before graph capture. |
| Graph private-pool delta | Full victory `[16]` costs `18.828 GiB/rank`; full buckets `[1,2,4,8,16]` cost `19.037 GiB/rank`. |
| KV/page capacity | `--num-pages 128`, page size `256`, KV cache is `2.320 GiB/rank` from the 08.06-compatible run shape. |

## Cache Owner Matrix

| Owner | Baseline layers | Baseline bytes/rank |
| --- | ---: | ---: |
| `q_wqb` | 43 | `0.336 GiB` |
| `wo_b` | 43 | `0.336 GiB` |
| `wo_a` BF16 BMM | 43 | `0.336 GiB` |
| `indexer_wq_b` | 21 | `0.328 GiB` |
| shared expert BF16 cache | 43 | `0.252 GiB` |
| all tested BF16 caches | - | `1.588 GiB` |

The A/B rows prove the denylist is real: disabled owners report
`enabled=false`, `layers_cached=0`, and `total_bytes=0` in
`model_prepare_report_rank0`.

## Single-Bucket Attribution

| Run | Enabled q/woB/woA/idx/shared | Persistent GiB | Free before/after/delta GiB | Alloc delta GiB | Reserved delta GiB | Delta vs baseline GiB | Replay/eager |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| full victory | Y/Y/Y/Y/Y | 1.588 | 55.485 / 36.657 / 18.828 | 17.820 | 18.191 | 0.000 | 63/0 |
| no projection BF16 caches | N/N/N/N/Y | 0.252 | 56.821 / 37.942 / 18.879 | 17.821 | 18.234 | +0.051 | 63/0 |
| no `q_wqb` | N/Y/Y/Y/Y | 1.252 | 55.833 / 37.013 / 18.820 | 17.820 | 18.180 | -0.008 | 63/0 |
| no `wo_b` | Y/N/Y/Y/Y | 1.252 | 55.833 / 37.009 / 18.824 | 17.820 | 18.184 | -0.004 | 63/0 |
| no `wo_a` | Y/Y/N/Y/Y | 1.252 | 55.829 / 37.005 / 18.824 | 17.820 | 18.184 | -0.004 | 63/0 |
| no `indexer_wq_b` | Y/Y/Y/N/Y | 1.260 | 55.817 / 36.989 / 18.828 | 17.821 | 18.188 | 0.000 | 63/0 |
| no shared expert | Y/Y/Y/Y/N | 1.336 | 55.599 / 36.749 / 18.850 | 17.820 | 18.207 | +0.021 | 63/0 |
| no all tested BF16 caches | N/N/N/N/N | 0.000 | 57.077 / 38.192 / 18.885 | 17.820 | 18.234 | +0.057 | 63/0 |

The largest observed single-bucket graph-delta movement is `+0.057 GiB/rank`
when all tested BF16 caches are disabled.  That is below the `1 GiB/rank`
materiality threshold and far below the `2 GiB/rank` small-fix threshold.

## Full-Bucket Baseline

| Run | Buckets | Persistent GiB | Free before/after/delta GiB | Alloc delta GiB | Reserved delta GiB | Replay/eager |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| full victory | `[1,2,4,8,16]` | 1.588 | 55.485 / 36.448 / 19.037 | 17.820 | 18.191 | 63/0 |

Per-bucket free-memory deltas:

| Captured bucket | Free delta GiB | Alloc delta GiB | Reserved delta GiB |
| ---: | ---: | ---: | ---: |
| 16 | 18.828 | 17.820 | 18.191 |
| 8 | 0.057 | 0.000 | 0.000 |
| 4 | 0.057 | 0.000 | 0.000 |
| 2 | 0.049 | 0.000 | 0.000 |
| 1 | 0.047 | 0.000 | 0.000 |

This matches TARGET 08.06: the first graph dominates and later buckets reuse the
pool.

## Attribution

BF16 caches are direct baseline memory only for this target's question.  They do
not materially inflate the CUDA graph private pool through captured GEMM/BMM
temporaries, cuBLAS/cuBLASLt workspace, or layout staging at the measured
serving shape.

Evidence:

- Full victory prepare reports `1.588 GiB/rank` of tested BF16 cache tensors
  before graph capture.
- Disabling all tested BF16 caches removes that persistent baseline from
  `model_prepare_report`.
- The same all-disabled run changes `[16]` graph delta by only `+0.057 GiB/rank`.
- Individual owner deltas are between `-0.008` and `+0.021 GiB/rank`.
- Allocated graph delta is effectively fixed at `~17.820 GiB/rank` across A/B
  rows.
- Every measured row preserved graph replay: `63/0` replay/eager.

## Decision For 08.10 And 08.18

Use the TARGET 08.05 bucket policy `[1,2,4,8,16]` for TARGET 08.10.  Treat the
`~19.04 GiB/rank` graph-capture delta as a non-BF16-cache graph private-pool
capacity cost.

Carry both costs into TARGET 08.18:

- graph private-pool capture: `~19.04 GiB/rank`;
- promoted tested BF16 cache persistent baseline: `1.588 GiB/rank`.

No pre-08.10 owner-specific memory fix is recommended from TARGET 08.07.
