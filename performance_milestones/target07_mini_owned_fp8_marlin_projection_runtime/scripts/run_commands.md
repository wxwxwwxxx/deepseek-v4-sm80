# TARGET 07.76 Run Commands

All commands were run from `/workspace/mini-sglang` on 2026-07-03.

## Static And Unit Checks

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  tests/models/test_deepseek_v4_forward_fallback.py
```

```bash
python - <<'PY'
import os
from minisgl.kernel import deepseek_v4 as dsv4

for key in (
    dsv4.DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION_TOGGLE,
    dsv4.DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION_TOGGLE,
):
    os.environ.pop(key, None)
print("none", dsv4.dense_fp8_marlin_projection_enabled())
os.environ[dsv4.DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION_TOGGLE] = "1"
print("new", dsv4.dense_fp8_marlin_projection_enabled())
os.environ.pop(dsv4.DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION_TOGGLE, None)
os.environ[dsv4.DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION_TOGGLE] = "1"
print("legacy", dsv4.dense_fp8_marlin_projection_enabled())
PY
```

```bash
rg -n "from minisgl\\.kernel import vllm_fp8_marlin|import minisgl\\.kernel\\.vllm_fp8_marlin|vllm_fp8_marlin\\.prepare|vllm_fp8_marlin\\.apply" \
  python/minisgl benchmark/offline tests
```

```bash
pytest -q -o addopts='' \
  tests/models/test_deepseek_v4_forward_fallback.py \
  -k 'marlin or shared_experts'
```

## TP8 Text Smoke

Same-run A/B was attempted first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_densefp8marlinproj \
  --output /tmp/dsv4_target0776_text_smoke.json \
  --page-size 256
```

The baseline per-variant JSON was valid, but the candidate failed because the
current single-Engine harness constructs the LLM before per-variant env is
applied.  The candidate was then run as the first and only variant:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_densefp8marlinproj \
  --output /tmp/dsv4_target0776_text_smoke_candidate.json \
  --page-size 256
```

Artifacts copied:

- `/tmp/dsv4_target0776_text_smoke.dsv4_sm80_a100_victory.json`
  to `raw/text_smoke_baseline.json`.
- `/tmp/dsv4_target0776_text_smoke_candidate.dsv4_sm80_a100_victory_densefp8marlinproj.json`
  to `raw/text_smoke_candidate.json`.

## 4096/128 Profile

The target prompt command without `--num-pages` was run and OOMed due to the
default large KV allocation.  Its artifact is preserved as
`raw/4096x128_baseline_default_memory_oom/`.

Valid baseline, `np128`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0776_densefp8marlin_4096x128_baseline_np128 \
  --keep-going
```

Valid candidate, `np128`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_densefp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0776_densefp8marlin_4096x128_candidate_np128 \
  --keep-going
```

## 4096/1024 Macro

Baseline, `np128`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0776_densefp8marlin_4096x1024_baseline_np128 \
  --keep-going
```

Candidate, `np128`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_densefp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0776_densefp8marlin_4096x1024_candidate_np128 \
  --keep-going
```
