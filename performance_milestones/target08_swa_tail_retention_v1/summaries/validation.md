# Validation Summary

Date: 2026-07-04

## Commands Run

```bash
python performance_milestones/target08_swa_tail_retention_v1/scripts/build_swa_tail_retention_v1_summary.py

pytest -q -o addopts='' \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_deepseek_v4_kvcache.py

python -m py_compile \
  python/minisgl/scheduler/config.py \
  python/minisgl/server/args.py \
  python/minisgl/scheduler/scheduler.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  performance_milestones/target08_swa_tail_retention_v1/scripts/build_swa_tail_retention_v1_summary.py \
  tests/core/test_dsv4_cache_option_guards.py

git diff --check
```

## Results

| check | result |
| --- | --- |
| targeted pytest | `54 passed in 4.61s` |
| py_compile | pass |
| git diff --check | pass |
| TP8 guarded logits oracle | not run; V1 runtime retention is fail-closed by design |
| TP8 text smoke | not run; V1 runtime retention is fail-closed by design |
| TP8 perf A/B | not run; V1 runtime retention is fail-closed by design |
| graph replay/eager `[1,2,4,8,16]` | not run; V1 runtime retention is fail-closed by design |
