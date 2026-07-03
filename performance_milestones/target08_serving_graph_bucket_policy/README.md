# TARGET 08.05 DSV4 Serving Graph Bucket Policy

## Result

Status: complete.

Recommendation for TARGET 08.10:

```text
cuda_graph_bs = [1, 2, 4, 8, 16]
```

Why: it is the smallest measured bucket set with zero eager decode across the
TARGET 08.05 prefix-off workload suite while keeping graph capture stable under
`--num-pages 128`. It also keeps the shared-prefix prefix-on workload on graph
replay. No measured workload needed `[24,32]` or larger buckets.

The benchmark harness was extended. A full online/RPS serving harness is still
not required to answer this graph-bucket question, but it remains unsupported by
the current offline scheduler.

## Exact Commands

The full run was:

```bash
cd /workspace/mini-sglang
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
REPEATS=1 WARMUP_REPEATS=0 PAGE_SIZE=256 NUM_PAGES=128 \
performance_milestones/target08_serving_graph_bucket_policy/scripts/run_bucket_policy_matrix.sh
```

The script expands to six `torchrun` passes:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios historical_4096_1024_bs4 historical_4096_128_bs4 \
    shared_prompt_reuse_bs8 decode_ladder_bs16 serving_mixed_112req_wave16 \
  --page-size 256 --num-pages 128 \
  --repeats 1 --warmup-repeats 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs <BUCKETS> \
  --output-dir performance_milestones/target08_serving_graph_bucket_policy/raw/<RUN> \
  --keep-going
```

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios shared_prompt_reuse_bs8 \
  --page-size 256 --num-pages 128 \
  --repeats 1 --warmup-repeats 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs <BUCKETS> \
  --enable-dsv4-radix-prefix-cache \
  --output-dir performance_milestones/target08_serving_graph_bucket_policy/raw/<RUN> \
  --keep-going
```

`<BUCKETS>` was one of:

```text
1 2 4
1 2 4 8
1 2 4 8 16
```

Final summary artifacts:

- `summaries/bucket_policy_summary.json`
- `summaries/bucket_policy_summary.md`
- raw reports and logs under `raw/`

## Hardware And Repo State

- Hardware: 8x NVIDIA A100-SXM4-80GB.
- Launch: TP8 with `torchrun --standalone --nproc_per_node=8`.
- Model: `/models/DeepSeek-V4-Flash`.
- Variant: `dsv4_sm80_a100_victory`.
- Env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`.
- KV policy: `--page-size 256 --num-pages 128`.
- KV cache memory: `2,491,495,680` bytes per rank, about `2.32 GiB`.
- Git worktree: dirty. It includes pre-existing TARGET08 phase-1 changes plus
  this target's benchmark/graph instrumentation and new milestone artifacts.

## Harness Changes

`benchmark/offline/deepseek_v4_perf_matrix.py` now has TARGET08-specific
scenarios and report fields:

- historical fixed scenarios:
  `historical_4096_1024_bs4`, `historical_4096_128_bs4`;
- `decode_ladder_bs16`, which steps active decode bs through
  `16,8,4,2,1`;
- `serving_mixed_112req_wave16`, an offline serving-style substitute with
  `112` total requests issued as seven same-process waves of `16`;
- per-case graph counter deltas, so multi-scenario runs do not mix graph counts;
- `bucket_coverage`: actual decode bs -> replay count, eager count, tokens,
  wall share;
- per-decode trace flags for graph replay vs eager.

`python/minisgl/engine/graph.py` now records graph capture setup time and memory
delta in `capture_status`. This is instrumentation only; replay policy is
unchanged.

## Workload Definitions

| scenario | requests | shape | purpose |
| --- | ---: | --- | --- |
| `historical_4096_1024_bs4` | 4 | prompt 4096, decode 1024, batch 4 | TARGET07 long fixed baseline |
| `historical_4096_128_bs4` | 4 | prompt 4096, decode 128, batch 4 | TARGET07 short fixed baseline |
| `shared_prompt_reuse_bs8` | 8 | 1024 shared prefix + 64 suffix, decode 16 | prefix-cache off/on A/B; exposes bs7 second stage |
| `decode_ladder_bs16` | 16 | prompt 128, mixed decode 16/24/32/48/64 | active decode bs near 1,2,4,8,16 |
| `serving_mixed_112req_wave16` | 112 | prompt cycle 64..256, mixed decode 16/24/32/48/64, wave 16 | early serving-style workload without timed arrivals |

The serving workload is a documented offline substitute. It does not model RPS,
queueing, or interleaved online arrivals.

## Bucket Comparison

| bucket set | prefix mode | captured bs | capture delta GiB | capture s | replay | eager | mean output tok/s | mean decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1,2,4]` | off | `[4,2,1]` | 18.90 | 14.39 | 1485 | 199 | 59.08 | 96.79 |
| `[1,2,4,8]` | off | `[8,4,2,1]` | 18.96 | 15.02 | 1564 | 120 | 82.00 | 146.56 |
| `[1,2,4,8,16]` | off | `[16,8,4,2,1]` | 19.04 | 17.43 | 1684 | 0 | 122.33 | 213.25 |
| `[1,2,4]` | prefix on shared | `[4,2,1]` | 18.90 | 14.00 | 15 | 15 | 16.86 | 40.74 |
| `[1,2,4,8]` | prefix on shared | `[8,4,2,1]` | 18.96 | 14.97 | 30 | 0 | 28.29 | 187.58 |
| `[1,2,4,8,16]` | prefix on shared | `[16,8,4,2,1]` | 19.04 | 15.17 | 30 | 0 | 42.66 | 187.54 |

All graph captures succeeded. No bucket set OOMed under `--num-pages 128`.

## Coverage Table

Prefix-off aggregate coverage:

| actual decode bs | `[1,2,4]` replay/eager | `[1,2,4,8]` replay/eager | `[1,2,4,8,16]` replay/eager |
| ---: | ---: | ---: | ---: |
| 1 | 143 / 0 | 143 / 0 | 143 / 0 |
| 2 | 128 / 0 | 128 / 0 | 128 / 0 |
| 4 | 1214 / 0 | 1214 / 0 | 1214 / 0 |
| 7 | 0 / 15 | 15 / 0 | 15 / 0 |
| 8 | 0 / 64 | 64 / 0 | 64 / 0 |
| 16 | 0 / 120 | 0 / 120 | 120 / 0 |

Prefix-off aggregate wall share:

| actual decode bs | `[1,2,4]` | `[1,2,4,8]` | `[1,2,4,8,16]` |
| ---: | ---: | ---: | ---: |
| 1 | 3.2% | 4.4% | 6.4% |
| 2 | 3.3% | 4.5% | 6.6% |
| 4 | 36.3% | 50.8% | 73.8% |
| 7 | 3.4% | 0.7% | 1.0% |
| 8 | 19.4% | 2.8% | 4.1% |
| 16 | 34.5% | 36.8% | 8.2% |

Interpretation:

- `[1,2,4]` fails both the historical phase-1 bs7 case and common bs8/bs16
  serving shapes.
- `[1,2,4,8]` fixes bs7 and bs8, but leaves bs16 eager. In the serving
  substitute, bs16 eager was 105 decode steps and dominated decode wall time.
- `[1,2,4,8,16]` covers every measured actual decode batch size with replay.

## Key Workload Metrics

| bucket set | scenario | output tok/s | decode tok/s | TTFT s | TPOT s | replay | eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1,2,4]` | `historical_4096_1024_bs4` | 120.88 | 168.45 | 7.60 | 0.0257 | 1023 | 0 |
| `[1,2,4,8,16]` | `historical_4096_1024_bs4` | 125.78 | 166.61 | 5.98 | 0.0260 | 1023 | 0 |
| `[1,2,4]` | `historical_4096_128_bs4` | 62.31 | 169.39 | 4.96 | 0.0256 | 127 | 0 |
| `[1,2,4,8,16]` | `historical_4096_128_bs4` | 62.00 | 167.10 | 4.96 | 0.0259 | 127 | 0 |
| `[1,2,4]` | `serving_mixed_112req_wave16` | 62.94 | 71.95 | 0.87 | 0.1835 | 280 | 161 |
| `[1,2,4,8]` | `serving_mixed_112req_wave16` | 91.70 | 106.01 | 0.60 | 0.1415 | 336 | 105 |
| `[1,2,4,8,16]` | `serving_mixed_112req_wave16` | 186.87 | 274.41 | 0.60 | 0.0289 | 441 | 0 |
| `[1,2,4]` | prefix-on `shared_prompt_reuse_bs8` | 16.86 | 40.74 | 3.06 | 0.1614 | 15 | 15 |
| `[1,2,4,8,16]` | prefix-on `shared_prompt_reuse_bs8` | 42.66 | 187.54 | 0.82 | 0.0285 | 30 | 0 |

Peak allocated memory stayed below `44.31 GiB` in the largest historical cases,
and peak reserved memory stayed below `46.46 GiB`. Prefix-on shared retained
`4` prefix pages and about `77,255,680` bytes of DSV4 cache memory, with
`7168` saved prefill tokens and no prefix evictions.

Communication counters were recorded in every raw report under
`communication_counters`. They were not used as the bucket-selection signal
because the dominant policy difference was graph replay coverage.

## Decision For TARGET 08.10

Use:

```text
[1, 2, 4, 8, 16]
```

Do not expand to `[24,32]` or `[48,64]` for TARGET 08.10 unless a new serving
trace shows frequent actual decode batch sizes above `16`. The measured suite
does not justify larger buckets, and the target asked not to blindly capture
large ranges.

The prefix-cache serving promotion gate now has a fair graph-bucket basis for
the measured shapes: bs7, bs8, and bs16 no longer accidentally fall back to
eager decode. Remaining promotion decisions can focus on prefix-cache stability,
memory/capacity, and correctness rather than bucket coverage artifacts.

## Remaining Risks

- Only one repeat per run was used. The recommendation is based on deterministic
  replay/eager coverage and large throughput effects, not fine-grained variance
  ranking.
- The serving workload is offline and wave-based; it does not model timed
  arrivals, queueing latency, RPS, cancellations, or continuous admission while
  decode is running.
- Workloads with actual decode batch size above `16` remain unsupported by the
  recommended default policy and should trigger a new evidence pass before
  adding larger buckets.
- Automatic `memory_ratio=0.9` KV sizing was intentionally not tested as a
  serving default; all runs used `--num-pages 128`.
- Prefix-on was measured for the shared-prefix scenario only, not for the
  entire 112-request serving substitute.
