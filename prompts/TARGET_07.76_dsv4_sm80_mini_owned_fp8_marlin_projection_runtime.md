# TARGET 07.76: DSV4 SM80 Mini-Owned FP8 Marlin Projection Runtime

Date: 2026-07-03

## Goal

Replace the TARGET 07.74 vLLM-helper runtime bridge with the TARGET 07.75
mini-owned dense FP8 Marlin bridge, then evaluate the full TP8 model path.

This is the shortest current victory path:

```text
mini-owned dense FP8 Marlin runtime opt-in
-> TP8 text smoke
-> CUDA graph replay check
-> 4096/128 profile
-> 4096/1024 macro
-> decide promote or keep opt-in
```

Do not broaden precision policy in this target.  Activations stay BF16.  The
Marlin path is weight-only W8A16 over native DSV4 block-FP8 weights.

## Starting Point

Promoted baseline:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Known promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

Do not use inactive opt-ins as the baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
```

## Evidence From TARGET 07.75

TARGET 07.75 passed the focused bridge gate:

- default mini Python `/usr/bin/python`;
- torch `2.9.1+cu128`;
- CUDA `12.8`;
- A100/sm80;
- extension: `minisgl_dense_fp8_marlin`;
- module: `python/minisgl/kernel/dense_fp8_marlin.py`;
- no vLLM runtime import;
- no `sgl_kernel` dependency.

Registered ops:

```text
gptq_marlin_repack
marlin_gemm
```

Focused real-weight M=`4` summary:

| Owner | Mini Marlin median ms | Max abs vs exact | Mean abs vs exact | Cosine |
| --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `0.058976` | `3.8147e-06` | `2.36469e-10` | `1.00000000` |
| `attn.wo_b local` | `0.058672` | `3.05176e-05` | `1.86628e-09` | `0.99999994` |
| shared experts down | `0.057792` | `0` | `0` | `1.00000000` |

Mini-owned bridge vs vLLM helper:

- mean mini-vs-vLLM median delta: `-7.47%`;
- mini min speedup vs the focused promoted cached-BF16 baseline: `76.51%`;
- mini mean speedup vs the focused promoted cached-BF16 baseline: `76.89%`.

Caveat: the focused promoted cached-BF16 baseline in 07.75 included per-call
activation FP8 rounding in the benchmark harness.  Do not directly extrapolate
the `76%` focused speedup to E2E.  The runtime target must measure actual
model-level graph replay, all-reduce, layout/copy, and owner attribution.

Artifact:

```text
performance_milestones/target07_mini_owned_dense_fp8_marlin_bridge/README.md
```

## Toggle And Variant

Preferred new toggle:

```text
MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1
```

Suggested candidate variant:

```text
dsv4_sm80_a100_victory_densefp8marlinproj
```

Compatibility rule:

- the old 07.74 toggle
  `MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1` may remain as a legacy
  alias for this target;
- the implementation should stop importing `minisgl.kernel.vllm_fp8_marlin`
  in the actual runtime path;
- new reports and README text should call the backend
  `mini_dense_fp8_marlin_w8a16_block`, not `vllm_fp8_marlin`.

Do not add this path to `dsv4_sm80_a100_victory` unless all promotion gates
pass.

## Owner Scope

Phase A runtime integration owners:

| Owner | Required behavior |
| --- | --- |
| `attn.q_wqb` | Use mini-owned dense FP8 Marlin W8A16 instead of promoted cached BF16 when the opt-in is enabled. |
| `attn.wo_b` local projection | Use mini-owned dense FP8 Marlin for local GEMM, preserving existing row-parallel all-reduce. |
| shared experts down | Use mini-owned dense FP8 Marlin for down projection, preserving existing shared expert all-reduce behavior. |

Explicitly out of scope:

- WQA/WKV/compress Phase B;
- shared experts gate/up Phase B;
- `wo_a` grouped two-launch Marlin;
- full FP8 KV cache or `fp8_ds_mla`;
- INT8 W8A8 MoE;
- TVM FFI migration;
- changing HC/router precision;
- changing routed MoE backend.

## Implementation Plan

Create artifacts:

```text
performance_milestones/target07_mini_owned_fp8_marlin_projection_runtime/
  README.md
  raw/
  scripts/
  summaries/
```

### 1. Replace The Runtime Bridge

Change the model runtime path that currently uses:

```text
minisgl.kernel.vllm_fp8_marlin
```

to use:

```text
minisgl.kernel.dense_fp8_marlin
```

Required helpers:

```text
prepare_dense_fp8_marlin_weight(...)
apply_dense_fp8_marlin_linear(...)
prepare_dense_fp8_marlin_report(...)
```

Preserve the existing "prepare before CUDA graph capture, never rebuild inside
forward" rule.

### 2. Keep Memory Lifecycle Tight

For switched owners:

- skip the corresponding cached BF16 weight cache;
- build Marlin-packed weight/scale/workspace before graph capture;
- release original FP8 `weight` and `weight_scale_inv` after successful pack
  unless a deliberate debug fallback is enabled;
- do not allocate or repack inside decode replay;
- report persistent bytes, workspace bytes, original released bytes, and the
  equivalent KV-page/token cost.

The previous 07.75 per-layer memory anchor:

| Owner | Cached BF16 persistent | Mini Marlin persistent |
| --- | ---: | ---: |
| `attn.q_wqb` | `8,388,608` | `4,260,272` |
| `attn.wo_b local` | `8,388,608` | `4,260,272` |
| shared experts down | `2,097,152` | `1,065,392` |

### 3. Add/Update Variant Wiring

Add the candidate variant to both offline benchmark scripts:

```text
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

The candidate should enable:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1
```

If keeping the old 07.74 variant for compatibility, make sure its behavior is
also mini-owned and does not import vLLM.

### 4. Focused Runtime Sanity Checks

Before TP8 macro, run a small focused runtime check on one GPU if possible:

- instantiate/load enough model weights to prepare the three owner groups;
- verify `prepare_for_cuda_graph_capture()` reports Marlin entries for q_wqb,
  wo_b, and shared-down;
- verify BF16 caches for switched owners are not also retained;
- verify a tiny forward path does not import vLLM.

If this focused runtime check is too expensive or awkward, document why and
proceed to TP8 smoke only after import/unit tests pass.

### 5. TP8 Text Smoke

Run page size 256 TP8 text smoke for baseline and candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_densefp8marlinproj \
  --output /tmp/dsv4_target0776_text_smoke.json
```

Required:

- no乱码 / no obvious repetition failure;
- candidate generates sane answers;
- graph replay remains active in decode;
- eager decode count remains `0`;
- no vLLM runtime import requirement.

### 6. Macro And Profile

Run same-run baseline vs candidate macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_densefp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --output-dir /tmp/dsv4_target0776_densefp8marlin_macro \
  --keep-going
```

Also run/capture the 4096/128 profile shape used in recent targets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_densefp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --output-dir /tmp/dsv4_target0776_densefp8marlin_4096x128 \
  --keep-going
```

If an nsys helper already exists in `performance_milestones/`, adapt it and
store large `.nsys-rep` files as symlinks.

## Measurement Requirements

The README must include:

- exact git status summary;
- exact env/toggles/variant;
- text smoke result;
- graph replay count and eager decode count;
- 4096/128 and 4096/1024 macro tables;
- memory ledger before/after graph capture;
- projection/GEMM owner deltas for q_wqb, wo_b, and shared-down;
- all-reduce timing around `wo_b` and shared experts if visible;
- whether any BF16 caches remained for switched owners;
- whether `vllm_fp8_marlin.py` is unused by the candidate runtime path.

## Promotion Gates

Promote into `dsv4_sm80_a100_victory` only if all are true:

- TP8 page-size-256 text smoke passes;
- graph replay remains active and eager decode remains `0`;
- candidate 4096/1024 output tok/s improves same-run baseline by at least
  `3%`;
- 4096/128 does not regress by more than `1%`;
- projection/GEMM owner attribution shows the switched owners improved or were
  removed from the top remaining projection bottleneck list;
- memory ledger shows no duplicate cached BF16 + Marlin packed weights for the
  same switched owner;
- runtime path does not depend on vLLM import or vLLM venv.

If macro improvement is positive but below `3%`, keep as opt-in and reprofile
before doing more local polishing.  If macro regresses, disable the runtime
variant and record whether the failure is caused by graph capture, launch
overhead, layout/copy, all-reduce, or owner shape mismatch.

## Stop Rules

Stop after one full smoke/profile/macro decision.  Do not add Phase B owners in
this thread.

Hard stop immediately if:

- the candidate cannot run in default mini Python without vLLM;
- TP8 text smoke fails;
- CUDA graph replay fails or eager decode appears;
- memory doubles for any switched owner;
- the dense Marlin op path silently falls back to BF16 or old vLLM helper.

The next target depends on the result:

- if 07.76 passes promotion gates, create a short promote/cleanup target;
- if 07.76 passes smoke but misses macro gates, run post-runtime reprofile and
  decide whether owner expansion, layout cleanup, or all-reduce overlap is the
  next bottleneck;
- if 07.76 fails correctness, keep 07.75 as focused-only and investigate the
  exact runtime boundary.

## Suggested README Outline

```text
# TARGET 07.76: Mini-Owned FP8 Marlin Projection Runtime

Status:

## Implementation
## Environment And Variants
## TP8 Text Smoke
## Macro Results
## Profile / Owner Attribution
## Memory Lifecycle
## vLLM Dependency Audit
## Decision
## Next Target
```

