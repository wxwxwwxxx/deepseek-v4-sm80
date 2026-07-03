# TARGET 07.77 Run Commands

All commands were run from `/workspace/mini-sglang` on 2026-07-03.

## Checks

```bash
python -m py_compile \
  python/minisgl/utils/dsv4_owner_timing.py \
  python/minisgl/kernel/dense_fp8_marlin.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/distributed/impl.py \
  python/minisgl/scheduler/scheduler.py \
  benchmark/offline/deepseek_v4_perf_matrix.py
```

```bash
pytest -q -o addopts='' \
  tests/models/test_deepseek_v4_forward_fallback.py \
  -k 'marlin or bf16_weight_cache'
```

## Fair / Repeated Macro

The benchmark still constructs one Engine per torchrun, so this target kept
separate baseline/candidate invocations and used repeated generations.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 1024 --batch-size 4 \
  --repeats 2 --warmup-repeats 0 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/raw/repeat2_4096x1024_baseline_np128 \
  --keep-going
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_densefp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 1024 --batch-size 4 \
  --repeats 2 --warmup-repeats 0 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/raw/repeat2_4096x1024_candidate_np128 \
  --keep-going
```

## Owner Timing

Owner timing is env-gated and uses CUDA events during graph capture/replay plus
host timers around `_prepare_batch` substeps.  It is diagnostic only and is not
used for macro gates.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_OWNER_TIMING=1 \
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 128 --batch-size 4 \
  --repeats 1 --warmup-repeats 0 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/raw/timing_4096x128_baseline_np128 \
  --keep-going
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_OWNER_TIMING=1 \
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_densefp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 128 --batch-size 4 \
  --repeats 1 --warmup-repeats 0 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/raw/timing_4096x128_candidate_np128 \
  --keep-going
```

The same command shape was used for 4096/1024 with output directories
`raw/timing_4096x1024_baseline_np128` and
`raw/timing_4096x1024_candidate_np128`.

## Summary

```bash
python performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/scripts/summarize_target0777.py
```
