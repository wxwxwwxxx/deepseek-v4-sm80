# TARGET 07.78: Benchmark Lifecycle And Repeat-Stable Gate

Status: complete. Decision: keep
`dsv4_sm80_a100_victory_densefp8marlinproj` as an explicit opt-in.

## Lifecycle Route

Chosen route: Option B, separate `torchrun` per variant.

Option A, fresh Engine per variant inside one Python process, is risky in the
current runtime because `Engine.__init__` asserts CUDA has not already been
initialized. Reconstructing an Engine after a previous TP8 Engine shutdown would
therefore require broader CUDA/distributed lifecycle changes. Separate
invocations give the fair lifecycle this target needs: variant env is present
before process startup, `LLM`/Engine construction, weight loading, model prepare,
KV cache allocation, and CUDA graph capture.

All commands were run from `/workspace/mini-sglang` on 8x A100 with
`/models/DeepSeek-V4-Flash`, TP8, page size 256, and `--num-pages 128`.

Primary command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate/scripts/run_repeat_stable_gate.sh
```

The runner expands to one candidate smoke command and four single-variant macro
commands. For the macro runs it uses this exact template:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len <1024-or-128> \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir <raw-dir> \
  --keep-going
```

Variants:

- `dsv4_sm80_a100_victory`
- `dsv4_sm80_a100_victory_densefp8marlinproj`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_densefp8marlinproj \
  --output performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate/raw/smoke_dsv4_sm80_a100_victory_densefp8marlinproj/text_smoke.json \
  --tensor-parallel-size 8 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 4096 \
  --max-tokens 64 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 \
  --fail-on-warning
```

Summary command:

```bash
python performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate/scripts/summarize_repeat_stable_gate.py \
  --milestone-dir performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate
```

Raw logs and reports are under `raw/`; the computed summary is
`summaries/target0778_summary.json`.

Git status at report time:

```text
 M prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md
 M prompts/target.md
?? performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate/
?? prompts/TARGET_07.78_dsv4_sm80_benchmark_lifecycle_repeat_stable_gate.md
```

## Correctness / Graph / Memory Gates

Candidate TP8 text smoke passed.

| Gate | Result |
| --- | --- |
| Status | `pass` |
| Sane outputs | `3/3` |
| Graph replay | `9` |
| Eager decode | `0` |
| Captured graph sizes | `[4, 2, 1]` |
| Dense Marlin cache | `enabled=true`, `layers_cached=43` |
| Dense Marlin backend | `mini_dense_fp8_marlin_w8a16_block` |
| Duplicate BF16 cache for switched owners | `false` |
| Persistent dense Marlin bytes/rank | `412,195,248` |
| Original released bytes/rank | `405,823,680` |

Smoke outputs:

- Chinese arithmetic answer contained `4`.
- `The sky is blue on a clear day.`
- Chinese Hangzhou answer passed the expected-substring sanity check.

The candidate env in the smoke report was exactly
`MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1` and
`MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1`. Source audit confirms the
runtime path imports `minisgl.kernel.dense_fp8_marlin`; the legacy
`MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION` name remains only as an alias,
not as a dependency on `minisgl.kernel.vllm_fp8_marlin`.

Memory lifecycle stayed clean. Peak allocated bytes per rank:

| Shape | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| 4096/1024 | `47,565,686,784` | `46,758,725,632` | `-806,961,152` |
| 4096/128 | `47,565,656,064` | `46,758,694,912` | `-806,961,152` |

KV cache memory stayed identical at `2,491,495,680` bytes/rank.

## Warmup Handling

Each macro invocation ran `--warmup-repeats 1 --repeats 3`. Warmup elapsed was
reported separately and excluded from every promotion statistic below.

| Shape | Variant | Warmup elapsed s |
| --- | --- | ---: |
| 4096/1024 | `dsv4_sm80_a100_victory` | `33.8397` |
| 4096/1024 | `dsv4_sm80_a100_victory_densefp8marlinproj` | `34.0650` |
| 4096/128 | `dsv4_sm80_a100_victory` | `10.9064` |
| 4096/128 | `dsv4_sm80_a100_victory_densefp8marlinproj` | `9.1537` |

## 4096/1024 Repeat-Stable Macro

Measured repeats:

| Variant | Repeat | Output tok/s | Decode tok/s | TTFT s | Prefill fwd s | Decode fwd s | Elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | 0 | `131.6442` | `169.1297` | `4.9732` | `4.2901` | `24.1945` | `31.1142` |
| `dsv4_sm80_a100_victory` | 1 | `132.0834` | `169.7197` | `4.9571` | `4.2654` | `24.1103` | `31.0107` |
| `dsv4_sm80_a100_victory` | 2 | `132.2002` | `169.6860` | `4.9265` | `4.2410` | `24.1151` | `30.9833` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 0 | `131.4479` | `169.6079` | `4.9812` | `4.2612` | `24.1262` | `31.1606` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 1 | `132.2654` | `170.4824` | `4.9717` | `4.2854` | `24.0025` | `30.9680` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 2 | `132.0993` | `170.2599` | `4.9760` | `4.2863` | `24.0338` | `31.0070` |

Statistics:

| Metric | Variant | Mean | Median | Best | Worst | Std | CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output tok/s | `dsv4_sm80_a100_victory` | `131.9759` | `132.0834` | `132.2002` | `131.6442` | `0.2932` | `0.002221` |
| Output tok/s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `131.9375` | `132.0993` | `132.2654` | `131.4479` | `0.4321` | `0.003275` |
| Decode tok/s | `dsv4_sm80_a100_victory` | `169.5118` | `169.6860` | `169.7197` | `169.1297` | `0.3314` | `0.001955` |
| Decode tok/s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `170.1167` | `170.2599` | `170.4824` | `169.6079` | `0.4544` | `0.002671` |
| TTFT s | `dsv4_sm80_a100_victory` | `4.9523` | `4.9571` | `4.9265` | `4.9732` | `0.0237` | `0.004788` |
| TTFT s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `4.9763` | `4.9760` | `4.9717` | `4.9812` | `0.0047` | `0.000950` |
| Prefill fwd s | `dsv4_sm80_a100_victory` | `4.2655` | `4.2654` | `4.2410` | `4.2901` | `0.0246` | `0.005756` |
| Prefill fwd s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `4.2776` | `4.2854` | `4.2612` | `4.2863` | `0.0142` | `0.003329` |
| Decode fwd s | `dsv4_sm80_a100_victory` | `24.1400` | `24.1151` | `24.1103` | `24.1945` | `0.0472` | `0.001957` |
| Decode fwd s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `24.0542` | `24.0338` | `24.0025` | `24.1262` | `0.0643` | `0.002674` |
| Elapsed s | `dsv4_sm80_a100_victory` | `31.0361` | `31.0107` | `30.9833` | `31.1142` | `0.0690` | `0.002224` |
| Elapsed s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `31.0452` | `31.0070` | `30.9680` | `31.1606` | `0.1018` | `0.003280` |

Graph gate:

| Variant | Replay | Greedy sample replay | Eager decode |
| --- | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | `4092` | `4092` | `0` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | `4092` | `4092` | `0` |

4096/1024 deltas:

- median output tok/s: `+0.0121%`
- mean output tok/s: `-0.0291%`
- candidate output CV: `0.3275%`, below the looser threshold of `2.0%`
- catastrophic measured repeats worse than `-3%`: none

## 4096/128 Repeat-Stable Sanity

Measured repeats:

| Variant | Repeat | Output tok/s | Decode tok/s | TTFT s | Prefill fwd s | Decode fwd s | Elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | 0 | `62.3093` | `169.2901` | `4.9704` | `4.2596` | `3.0008` | `8.2171` |
| `dsv4_sm80_a100_victory` | 1 | `62.4297` | `169.5292` | `4.9567` | `4.2583` | `2.9965` | `8.2012` |
| `dsv4_sm80_a100_victory` | 2 | `62.3462` | `169.7027` | `4.9688` | `4.2642` | `2.9935` | `8.2122` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 0 | `62.4034` | `170.8111` | `4.9775` | `4.2579` | `2.9740` | `8.2047` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 1 | `62.4479` | `171.2888` | `4.9714` | `4.2501` | `2.9658` | `8.1988` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 2 | `59.6880` | `152.2951` | `4.9808` | `4.2828` | `3.3356` | `8.5779` |

Statistics:

| Metric | Variant | Mean | Median | Best | Worst | Std | CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output tok/s | `dsv4_sm80_a100_victory` | `62.3617` | `62.3462` | `62.4297` | `62.3093` | `0.0617` | `0.000989` |
| Output tok/s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `61.5131` | `62.4034` | `62.4479` | `59.6880` | `1.5807` | `0.025697` |
| Decode tok/s | `dsv4_sm80_a100_victory` | `169.5073` | `169.5292` | `169.7027` | `169.2901` | `0.2072` | `0.001222` |
| Decode tok/s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `164.7983` | `170.8111` | `171.2888` | `152.2951` | `10.8308` | `0.065721` |
| TTFT s | `dsv4_sm80_a100_victory` | `4.9653` | `4.9688` | `4.9567` | `4.9704` | `0.0075` | `0.001511` |
| TTFT s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `4.9766` | `4.9775` | `4.9714` | `4.9808` | `0.0048` | `0.000957` |
| Prefill fwd s | `dsv4_sm80_a100_victory` | `4.2607` | `4.2596` | `4.2583` | `4.2642` | `0.0031` | `0.000725` |
| Prefill fwd s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `4.2636` | `4.2579` | `4.2501` | `4.2828` | `0.0171` | `0.004004` |
| Decode fwd s | `dsv4_sm80_a100_victory` | `2.9969` | `2.9965` | `2.9935` | `3.0008` | `0.0037` | `0.001222` |
| Decode fwd s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `3.0918` | `2.9740` | `2.9658` | `3.3356` | `0.2112` | `0.068308` |
| Elapsed s | `dsv4_sm80_a100_victory` | `8.2102` | `8.2122` | `8.2012` | `8.2171` | `0.0081` | `0.000989` |
| Elapsed s | `dsv4_sm80_a100_victory_densefp8marlinproj` | `8.3272` | `8.2047` | `8.1988` | `8.5779` | `0.2172` | `0.026084` |

Graph gate:

| Variant | Replay | Greedy sample replay | Eager decode |
| --- | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | `508` | `508` | `0` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | `508` | `508` | `0` |

4096/128 median output tok/s delta was `+0.0917%`, so the short-shape median
sanity gate passed. Candidate repeat 2 was noisy (`59.6880` output tok/s), but
the target's 4096/128 rule is median-only and the long-shape CV/catastrophic
repeat gate remained clean.

## Variance Analysis

Long-shape variation was small and acceptable:

| Variant | 4096/1024 output CV |
| --- | ---: |
| `dsv4_sm80_a100_victory` | `0.2221%` |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | `0.3275%` |

The allowed candidate threshold is `max(1.5 * baseline_cv, 2%)`, which is `2%`.
The candidate is below it. There were no individual 4096/1024 measured-repeat
regressions worse than `-3%` against the paired baseline repeat.

## Decision

Decision: keep opt-in.

Promotion rule check:

| Rule | Result |
| --- | --- |
| Text smoke passes | pass |
| Graph replay active / eager decode zero | pass |
| Memory lifecycle clean | pass |
| 4096/1024 median output tok/s >= `+2%` | fail: `+0.0121%` |
| 4096/1024 mean output tok/s >= `+1%` | fail: `-0.0291%` |
| 4096/128 median output tok/s not worse than `-1%` | pass: `+0.0917%` |
| Candidate CV acceptable | pass |
| No repeatable catastrophic `-3%` long-repeat regression | pass |

The long-shape median is inside the target's neutral `[-1%, +2%]` band, so the
dense FP8 Marlin projection path should remain an explicit opt-in. It should not
be promoted into `dsv4_sm80_a100_victory` from this evidence. It also should not
be reverted or repaired as a kernel regression: under fair lifecycle, the
seconds-scale 07.76 regression does not reproduce, and the repeat-stable long
macro is effectively neutral.

## Next Target

No dense FP8 Marlin kernel optimization target is recommended from this gate.
The useful next benchmark work is to keep the separate-invocation runner as the
promotion harness, optionally adding more seeds or alternating variant order if a
future default-bundle decision needs tighter confidence. The implementation path
itself stays opt-in.
