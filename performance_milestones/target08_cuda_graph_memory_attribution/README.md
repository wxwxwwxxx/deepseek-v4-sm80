# TARGET 08.06 CUDA Graph Memory Attribution

## Recommendation

Keep the 08.10 serving bucket policy at `[1, 2, 4, 8, 16]`.

The measured graph-capture free-memory delta is real capacity cost for this
promoted path, about `19.04 GiB/rank` for the recommended bucket set. It should
be carried into the 08.18 memory ledger rather than blocked on a low-risk 08.10
fix. The evidence does not point to a small isolated owner such as greedy
sampling, compressed-location metadata, max sequence length, or KV page count.
The likely owner is CUDA graph private-pool capture of the full model forward
workspace/runtime allocation shape.

No stop rule fired:

- Promoted-path graph capture was stable across all 14 separate `torchrun`
  lifecycles.
- `--num-pages 128` did not OOM.
- The added instrumentation only records allocator/free-memory counters and
  provides opt-in A/B toggles; the default capture behavior is unchanged.
- Reducing the 19 GiB class cost would require broader graph/workspace planning,
  so it belongs after this measurement target.

## Scope

Promoted path:

- Variant: `dsv4_sm80_a100_victory`
- Env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- Page size: `256`
- Fixed/capped pages: `--num-pages 64` and `--num-pages 128`
- Workload: `decode_ladder_bs16`, serving-style decode ladder from TARGET 08.05
- Lifecycle: each matrix point was run in a fresh `torchrun`

All byte-to-GiB values in this report use `bytes / 2^30`.

## Instrumentation

Small measurement-only changes were added:

- `python/minisgl/engine/graph.py`
  - captures free memory before/after graph capture
  - records total and per-bucket free-memory delta
  - records capture elapsed time
  - records allocated/reserved before/after and peak allocated/reserved
  - records graph-capture buffer bytes
  - records captured batch sizes
  - records whether graph pool reuse was enabled and its anchor bucket
- `python/minisgl/attention/deepseek_v4.py`
  - adds opt-in env `MINISGL_DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH=1`
    to disable captured compressed-location metadata updates for A/B testing
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - adds `--cuda-graph-capture-greedy-sample`
  - adds `--no-cuda-graph-capture-greedy-sample`
- `tests/attention/test_deepseek_v4_backend_metadata.py`
  - adds coverage for the metadata-disable hook

The existing graph-pool behavior is unchanged: the first captured graph creates
the pool, and later captured buckets use `pool=pool`.

## Exact Commands

Full matrix command used:

```bash
cd /workspace/mini-sglang
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
performance_milestones/target08_cuda_graph_memory_attribution/scripts/run_cuda_graph_memory_attribution_matrix.sh
```

The script expands each case to this `torchrun` shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH=<0-or-1> \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_ladder_bs16 \
  --page-size 256 \
  --num-pages <64-or-128> \
  --max-seq-len <1280-or-2048-or-5120> \
  --repeats 1 \
  --warmup-repeats 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs <bucket-list> \
  --cuda-graph-capture-greedy-sample \
  --output-dir performance_milestones/target08_cuda_graph_memory_attribution/raw/<run-name> \
  --keep-going
```

Greedy-off cases replace `--cuda-graph-capture-greedy-sample` with:

```bash
--no-cuda-graph-capture-greedy-sample
```

Metadata-off cases set:

```bash
MINISGL_DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH=1
```

Summary command:

```bash
python performance_milestones/target08_cuda_graph_memory_attribution/scripts/summarize_cuda_graph_memory.py \
  --milestone-dir performance_milestones/target08_cuda_graph_memory_attribution
```

Validation commands:

```bash
python -m py_compile \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  python/minisgl/engine/graph.py \
  python/minisgl/attention/deepseek_v4.py \
  performance_milestones/target08_cuda_graph_memory_attribution/scripts/summarize_cuda_graph_memory.py

pytest -q tests/attention/test_deepseek_v4_backend_metadata.py
```

Validation result: `11 passed`.

## Matrix Cases

| Run | Buckets | max_seq_len | num_pages | greedy | compressed locs in graph |
| --- | --- | ---: | ---: | --- | --- |
| `bucketset_1_2_4_np128_sl2048_greedy_on_metadata_on` | `[1,2,4]` | 2048 | 128 | on | on |
| `bucketset_1_2_4_8_np128_sl2048_greedy_on_metadata_on` | `[1,2,4,8]` | 2048 | 128 | on | on |
| `bucketset_1_2_4_8_16_np128_sl2048_greedy_on_metadata_on` | `[1,2,4,8,16]` | 2048 | 128 | on | on |
| `single_1_np128_sl2048_greedy_on_metadata_on` | `[1]` | 2048 | 128 | on | on |
| `single_4_np128_sl2048_greedy_on_metadata_on` | `[4]` | 2048 | 128 | on | on |
| `single_8_np128_sl2048_greedy_on_metadata_on` | `[8]` | 2048 | 128 | on | on |
| `single_16_np128_sl2048_greedy_on_metadata_on` | `[16]` | 2048 | 128 | on | on |
| `greedy_off_np128_sl2048_metadata_on` | `[1,2,4,8,16]` | 2048 | 128 | off | on |
| `metadata_off_np128_sl2048_greedy_on` | `[1,2,4,8,16]` | 2048 | 128 | on | off |
| `seq1280_np64_greedy_on_metadata_on` | `[1,2,4,8,16]` | 1280 | 64 | on | on |
| `seq1280_np128_greedy_on_metadata_on` | `[1,2,4,8,16]` | 1280 | 128 | on | on |
| `seq2048_np64_greedy_on_metadata_on` | `[1,2,4,8,16]` | 2048 | 64 | on | on |
| `seq5120_np64_greedy_on_metadata_on` | `[1,2,4,8,16]` | 5120 | 64 | on | on |
| `seq5120_np128_greedy_on_metadata_on` | `[1,2,4,8,16]` | 5120 | 128 | on | on |

Raw artifacts:

- Raw run directories: `performance_milestones/target08_cuda_graph_memory_attribution/raw/*`
- Per-run reports: `raw/<run>/reports/000_decode_ladder_bs16__dsv4_sm80_a100_victory.json`
- Consolidated JSON: `summaries/cuda_graph_memory_attribution_summary.json`
- Consolidated Markdown: `summaries/cuda_graph_memory_attribution_summary.md`

## Hardware And Software

| Item | Value |
| --- | --- |
| GPUs | 8x `NVIDIA A100-SXM4-80GB` |
| GPU memory | `81920 MiB` per GPU |
| CUDA capability | `sm80` |
| Driver | `570.172.08` |
| CUDA runtime | `12.8` |
| Python | `3.12.3` |
| PyTorch | `2.9.1+cu128` |
| Triton | `3.5.1` |
| NCCL | `2.27.5` |
| flashinfer | `0.6.12` |
| sgl_kernel | `0.3.21` |
| tilelang | `0.1.11+cu128.gita417b38a` |
| Branch | `dsv4-sglang-based` |
| Commit | `737798d3dc01328991a5b18d20687e1286c4163b` |
| Worktree | dirty due to this instrumentation/report |

Kernel capability notes from the run:

- `triton`, `flashinfer`, `flash_mla`, `tilelang`, and `sgl_kernel` were
  available.
- `deep_gemm` was not usable because its extension expected `libcudart.so.13`.
- `marlin` module was not installed. This matches the existing promoted path
  state used by TARGET 08.05.

## Baseline Counters

Baseline run:
`bucketset_1_2_4_8_16_np128_sl2048_greedy_on_metadata_on`

| Counter | Value |
| --- | ---: |
| Captured buckets | `[16,8,4,2,1]` |
| Replay/eager decode | `63/0` |
| Greedy-sample replay | `63` |
| Free before capture | `55.485 GiB` |
| Free after capture | `36.448 GiB` |
| Free-memory delta | `19.037 GiB` |
| Allocated before capture | `23.241 GiB` |
| Allocated after capture | `41.062 GiB` |
| Allocated delta | `17.820 GiB` |
| Reserved before capture | `23.258 GiB` |
| Reserved after capture | `41.449 GiB` |
| Reserved delta | `18.191 GiB` |
| Peak allocated during capture | `41.310 GiB` |
| Peak reserved during capture | `41.697 GiB` |
| Capture elapsed | `17.12 s` |
| Capture buffer bytes | `8,274,176 bytes` / `7.891 MiB` |
| Graph pool reuse | `true`, anchor bucket `16` |
| KV cache per rank | `2,491,495,680 bytes` / `2.320 GiB` |

The baseline free-memory delta was identical on every rank:

| Rank statistic | Delta |
| --- | ---: |
| rank0 | `19.037 GiB` |
| min | `19.037 GiB` |
| mean | `19.037 GiB` |
| max | `19.037 GiB` |

## Bucket Sensitivity

`max_seq_len=2048`, `num_pages=128`, greedy on, compressed-location metadata in
graph.

| Buckets | Captured | Free before GiB | Free after GiB | Delta GiB | Capture s | Peak alloc/res GiB | Replay/eager |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `[1,2,4]` | `[4,2,1]` | 55.485 | 36.583 | 18.902 | 14.69 | 41.303 / 41.686 | 40/23 |
| `[1,2,4,8]` | `[8,4,2,1]` | 55.485 | 36.526 | 18.959 | 15.02 | 41.305 / 41.686 | 48/15 |
| `[1,2,4,8,16]` | `[16,8,4,2,1]` | 55.485 | 36.448 | 19.037 | 17.12 | 41.310 / 41.697 | 63/0 |

Adding buckets above `[1,2,4]` adds only about `0.135 GiB/rank`:

- `[1,2,4]` to `[1,2,4,8]`: `+0.057 GiB`
- `[1,2,4,8]` to `[1,2,4,8,16]`: `+0.078 GiB`

## Single-Bucket Sensitivity

`max_seq_len=2048`, `num_pages=128`, greedy on, compressed-location metadata in
graph.

| Buckets | Captured | Free before GiB | Free after GiB | Delta GiB | Capture s | Peak alloc/res GiB | Replay/eager |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `[1]` | `[1]` | 55.485 | 36.690 | 18.795 | 11.95 | 41.301 / 41.686 | 16/47 |
| `[4]` | `[4]` | 55.485 | 36.681 | 18.805 | 11.50 | 41.303 / 41.686 | 40/23 |
| `[8]` | `[8]` | 55.485 | 36.679 | 18.807 | 13.77 | 41.305 / 41.686 | 48/15 |
| `[16]` | `[16]` | 55.485 | 36.657 | 18.828 | 13.57 | 41.310 / 41.674 | 63/0 |

The largest bucket does not explain the 19 GiB scale. Even the `[1]` graph by
itself costs `18.795 GiB/rank`; `[16]` costs `18.828 GiB/rank`, only
`0.033 GiB/rank` more.

## Per-Bucket Delta And Pool Reuse

Bucket captures are performed from largest to smallest. With graph-pool reuse
enabled, the first captured graph claims the large private-pool footprint; later
buckets add only small graph-specific overhead.

| Run | Per-bucket free-memory delta |
| --- | --- |
| `[1,2,4,8,16]` | bs16 `18.828 GiB`; bs8 `0.057 GiB`; bs4 `0.057 GiB`; bs2 `0.049 GiB`; bs1 `0.047 GiB` |
| `[1,2,4,8]` | bs8 `18.807 GiB`; bs4 `0.057 GiB`; bs2 `0.049 GiB`; bs1 `0.047 GiB` |
| `[1,2,4]` | bs4 `18.805 GiB`; bs2 `0.051 GiB`; bs1 `0.047 GiB` |
| `[16]` | bs16 `18.828 GiB` |
| `[8]` | bs8 `18.807 GiB` |
| `[4]` | bs4 `18.805 GiB` |
| `[1]` | bs1 `18.795 GiB` |

For the recommended bucket set, per-bucket allocated/reserved deltas show the
same pattern:

| Captured bucket | Free delta GiB | Allocated delta GiB | Reserved delta GiB | Capture s |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 18.828 | 17.813 | 18.191 | 13.35 |
| 8 | 0.057 | 0.000 | 0.000 | 0.67 |
| 4 | 0.057 | 0.000 | 0.000 | 0.45 |
| 2 | 0.049 | 0.000 | 0.000 | 2.22 |
| 1 | 0.047 | 0.000 | 0.000 | 0.44 |

Conclusion: graph pool reuse is working. This is why `[1,2,4]` and
`[1,2,4,8,16]` have almost the same total capture delta.

## Greedy Sample A/B

`[1,2,4,8,16]`, `max_seq_len=2048`, `num_pages=128`, compressed-location
metadata in graph.

| Greedy capture | Free before GiB | Free after GiB | Delta GiB | Capture s | Peak alloc/res GiB | Buffer MiB | Replay/eager |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| on | 55.485 | 36.448 | 19.037 | 17.12 | 41.310 / 41.697 | 7.891 | 63/0 |
| off | 55.485 | 36.448 | 19.037 | 15.00 | 41.310 / 41.697 | 7.891 | 63/0 |

Disabling captured greedy sample changed graph-capture memory by `0.000 GiB`.
The capture buffer changed by only `64 bytes`.

## Metadata A/B

`[1,2,4,8,16]`, `max_seq_len=2048`, `num_pages=128`, greedy capture on.

| Compressed locs in graph | Free before GiB | Free after GiB | Delta GiB | Capture s | Peak alloc/res GiB | Buffer MiB | Replay/eager |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| on | 55.485 | 36.448 | 19.037 | 17.12 | 41.310 / 41.697 | 7.891 | 63/0 |
| off | 55.485 | 36.448 | 19.037 | 14.78 | 41.310 / 41.697 | 7.891 | 63/0 |

Disabling graph-captured compressed-location metadata changed graph-capture
memory by `0.000 GiB`.

## max_seq_len And num_pages Sensitivity

`[1,2,4,8,16]`, greedy capture on, compressed-location metadata in graph.

| max_seq_len | num_pages | KV GiB/rank | Free before GiB | Free after GiB | Delta GiB | Capture s | Peak alloc/res GiB | Replay/eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1280 | 64 | 1.169 | 56.640 | 37.638 | 19.002 | 16.99 | 40.141 / 40.531 | 63/0 |
| 1280 | 128 | 2.320 | 55.485 | 36.448 | 19.037 | 15.33 | 41.309 / 41.697 | 63/0 |
| 2048 | 64 | 1.169 | 56.640 | 37.638 | 19.002 | 17.36 | 40.141 / 40.531 | 63/0 |
| 2048 | 128 | 2.320 | 55.485 | 36.448 | 19.037 | 17.12 | 41.310 / 41.697 | 63/0 |
| 5120 | 64 | 1.169 | 56.640 | 37.638 | 19.002 | 17.27 | 40.141 / 40.531 | 63/0 |
| 5120 | 128 | 2.320 | 55.485 | 36.448 | 19.037 | 15.31 | 41.310 / 41.697 | 63/0 |

`max_seq_len` did not change the capture delta at these decode-ladder points.
Changing `num_pages` from 64 to 128 moved the free-memory baseline by the KV
cache size, but the graph-capture delta changed by only `0.035 GiB/rank`.

## Attribution

The 19 GiB/rank delta is real CUDA graph capture capacity cost for this
promoted path.

Evidence:

- Measurement starts after model/KV initialization and immediately before graph
  capture, so the delta is not model loading or KV allocation.
- The free-memory delta is rank-identical across all 8 ranks.
- PyTorch allocated/reserved counters rise with the same event:
  - allocated delta: `17.820 GiB`
  - reserved delta: `18.191 GiB`
  - free-memory delta: `19.037 GiB`
- Single-bucket capture already costs `18.795-18.828 GiB/rank`.
- The explicit graph input buffer is tiny by comparison:
  - `[1]`: `0.493 MiB`
  - `[4]`: `1.973 MiB`
  - `[8]`: `3.945 MiB`
  - `[16]` or `[1,2,4,8,16]`: `7.891 MiB`
- Greedy sample A/B changed memory by `0.000 GiB`.
- Compressed-location metadata A/B changed memory by `0.000 GiB`.
- `max_seq_len` A/B changed memory by `0.000 GiB`.
- `num_pages` A/B changed graph delta by only `0.035 GiB`.

The remaining plausible owner is the CUDA graph private pool preserving the
runtime allocation/workspace pattern of the captured full model forward. The
current small instrumentation can prove that pool reuse works, but it cannot
split the private-pool cost by every internal kernel/workspace owner without a
larger graph/workspace manager redesign.

## Answers For 08.10

1. Is the 19 GiB/rank delta real graph private-pool cost?

   Yes. It is visible in free memory and allocator counters, repeated across
   separate `torchrun` lifecycles, and identical across ranks.

2. Why are `[1,2,4]` and `[1,2,4,8,16]` almost the same?

   The first captured graph in each process claims about `18.8 GiB/rank`.
   Subsequent buckets reuse that graph pool and add only about `0.05-0.06 GiB`
   each. Pool reuse is effective.

3. How much does maximum bucket size matter?

   Very little at this scale. Single-bucket `[1]` is `18.795 GiB`; single-bucket
   `[16]` is `18.828 GiB`. The whole recommended set is `19.037 GiB`.

4. How much do greedy sample and compressed metadata matter?

   Both measured as `0.000 GiB` delta relative to baseline.

5. How much do `max_seq_len` and `num_pages` matter?

   `max_seq_len` from 1280 to 5120 did not change the graph delta. `num_pages`
   from 64 to 128 changed graph delta by only `0.035 GiB`, while moving the
   free-memory baseline by the expected KV-cache capacity.

6. Is there a low-risk fix needed before 08.10?

   No. The measurement does not identify a small safe toggle that removes the
   19 GiB class cost. A real reduction would likely need graph/workspace
   lifecycle redesign, which is outside this target.

7. Should 08.10 continue with `[1,2,4,8,16]`?

   Yes. This policy still removes the serving-style eager decode found in
   TARGET 08.05. Its incremental memory cost over `[1,2,4]` is only
   `0.135 GiB/rank`, while it captures all decode ladder buckets.

## Follow-Up For 08.18

Carry `19.04 GiB/rank` as a first-class CUDA graph private-pool owner in the
08.18 memory/capacity ledger. Keep fixed/capped KV pages for serving experiments
and avoid interpreting graph-capture free-memory deltas without separating:

- model/runtime baseline
- KV cache capacity
- CUDA graph private pool
- explicit graph input buffers
- later per-bucket graph overhead
