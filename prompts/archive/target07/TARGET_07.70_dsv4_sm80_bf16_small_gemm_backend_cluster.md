# TARGET 07.70: DSV4 SM80 BF16 Small-GEMM Backend Cluster

Date: 2026-07-02

## Goal

Reduce the current promoted exact-route BF16 small-GEMM backend cluster on
A100/sm80.

TARGET 07.69 proved that the old TARGET 07.57 `_quantized_linear_fp8_kernel`
projection bottleneck is stale.  The current promoted path has already moved
the largest attention/indexer/shared-expert projections to cached BF16 weights.
What remains is not one huge projection owner; it is a cross-owner cluster of
decode-small BF16 GEMMs and their splitK/reduce backend overhead.

This is an implementation target, but it must remain evidence-driven:

- stay on the current promoted BF16/exact route;
- compare against vLLM source boundaries, but do not port vLLM low-precision
  mechanisms as if they were precision-neutral;
- try multiple backend approaches only through focused real-shape microbench
  gates;
- promote nothing unless both profile and macro gates pass.

## Current Promoted Baseline

Use:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Current confirmed promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

Promoted path includes:

- cached BF16 projection weights for `attn.q_wqb`, `attn.wo_b`,
  `indexer.wq_b`, and `attn.wo_a`;
- cached BF16 shared expert gate/up/down projection weights;
- FP8 indexer/cache pieces that are already part of the current victory stack;
- graph replay active and eager decode `0`.

Do not use these opt-in-only paths as the baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1  # TARGET 07.64, not promoted
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1          # TARGET 07.68, not promoted
```

## Starting Evidence From TARGET 07.69

TARGET 07.69 reused the 07.67 promoted Nsight profile and explained the
projection/GEMM bucket with `98.94%` named coverage:

| Metric | Value |
| --- | ---: |
| Decode envelope wall | `3.591306s` |
| Projection/GEMM bucket | `0.778887s` |
| Named/grouped owner time | `0.770601s` |
| Residual/coarse owner time | `0.008286s` |
| Named coverage | `98.94%` |

Owner readout:

| Owner group | Kernel s | Share | Backend family |
| --- | ---: | ---: | --- |
| HC pre linear | `0.178373` | `22.90%` | cuBLAS SGEMM/FP32 + splitK/reduce |
| attention WQA/WKV/compress | `0.119458` | `15.34%` | CUTLASS BF16 + splitK/reduce |
| MoE router / route projection | `0.097109` | `12.47%` | cuBLAS SGEMM/FP32 + splitK/reduce |
| shared experts cached BF16 | `0.085848` | `11.02%` | CUTLASS BF16 + splitK/reduce |
| attention `wo_a` | `0.063857` | `8.20%` | cuBLASLt BF16 + splitK/reduce |
| attention `q_wqb` | `0.056392` | `7.24%` | cuBLASLt BF16 GEMM |
| attention `wo_b` local | `0.054507` | `7.00%` | cuBLASLt BF16 GEMM |
| indexer weight/compressor projection | `0.043647` | `5.60%` | CUTLASS/cuBLASLt BF16 + splitK/reduce |
| indexer `wq_b` | `0.042727` | `5.49%` | cuBLASLt BF16 GEMM |
| `lm_head` | `0.026769` | `3.44%` | cuBLAS SGEMM/FP32 |

No single owner clears the `0.20s` single-owner gate.

Backend-cluster readout:

| Backend cluster/family | Kernel s | Share | Decision |
| --- | ---: | ---: | --- |
| BF16 small-GEMM + splitK/reduce cluster | `0.521619` | `66.97%` | selected |
| FP32/SGEMM small-GEMM cluster | `0.257269` | `33.03%` | context only |
| cuBLASLt BF16 GEMM | `0.219912` | `28.23%` | part of selected cluster |
| CUTLASS BF16 GEMM | `0.194319` | `24.95%` | part of selected cluster |
| cuBLASLt splitK/reduce | `0.107388` | `13.79%` | attribute to parent GEMMs |
| residual FP8 quantized linear | `0.000000` | `0.00%` | old 07.57 path is gone |

Focused 07.69 microbench hints:

| Case | M=1 mean ms | M=4 mean ms | M=8 mean ms | M=16 mean ms |
| --- | ---: | ---: | ---: | ---: |
| HC pre linear BF16/FP32 fallback | `0.056493` | `0.058148` | `0.057837` | `0.058296` |
| MoE router gate linear | `0.062460` | `0.067730` | `0.068906` | `0.068772` |
| WQA/WKV cached BF16 GEMM only | `0.036729` | `0.042605` | `0.042849` | `0.043011` |
| WQA/WKV cached BF16 with existing act-quant boundary | `0.102567` | `0.109819` | `0.109257` | `0.109605` |

The flat `M=1..16` curves strongly suggest decode-small backend/launch/fixed
overhead rather than saturated tensor-core math.

## vLLM Context

Use vLLM as a boundary map, not as an automatic drop-in:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/mhc.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/
/workspace/vllm-dsv4-docker/vllm/compilation/
```

TARGET 07.69's source parity conclusion:

- vLLM generally uses `deepseek_v4_fp8`, quantized/custom-op/runner contracts,
  packed FP8 cache/indexer paths, `deepseek_v4_attention`,
  `fused_inv_rope_fp8_quant`, `deepseek_v4_fp8_einsum`, and compile
  boundaries;
- mini's current promoted route uses cached BF16 projection weights and should
  stay exact in this target;
- if the only fast path requires FP8/MXFP4 carriers, FP32 `post/comb`, packed
  cache layout, or broad compile/runtime ownership changes, write a follow-up
  precision-policy target instead of forcing it into 07.70.

## Scope

In scope:

- focused microbenchmarks for the BF16 small-GEMM cluster owners selected by
  07.69;
- cuBLASLt algorithm/workspace/preference experiments for decode-small BF16
  shapes;
- tests that avoid or reduce backend-generated splitK/reduce when it is
  harmful for small `M`;
- custom or adapted CUTLASS/Triton BF16 small-M kernels for selected shapes;
- prepacked or pretransposed BF16 weight layouts when memory cost is recorded;
- owner-local grouped/batched/fused projection experiments when data
  dependencies allow them;
- existing activation-quant boundary context if it affects selected BF16
  projection owners, without introducing a new precision contract;
- opt-in implementation paths guarded by explicit env flags and benchmark
  variants;
- text smoke, focused correctness comparison, 4096/128 profile, and 4096/1024
  macro validation.

Out of scope:

- changing default precision policy;
- introducing FP8/INT8/TF32 as a promoted route;
- full `fp8_ds_mla` or KV-cache layout redesign;
- reopening 07.64 metadata deforestation or 07.68 HC graph cleanup promotion;
- NCCL/all-reduce optimization;
- sparse attention/indexer cache rewrite;
- MoE routed Marlin kernel replacement;
- broad whole-model `torch.compile` or full decoder graph rewrite;
- batching GEMMs across decoder layers when layer-to-layer data dependencies
  make that invalid.

Measurement-only precision ablations are allowed only if they help decide the
next target.  They must not be promoted in 07.70.

## Candidate Solution Lanes

Try these lanes in order.  Stop once one lane clears the focused microbench
gate and becomes the implementation candidate.

### Lane A: cuBLASLt Algorithm And SplitK Policy

Hypothesis: PyTorch/cuBLASLt is picking generic small-M BF16 algorithms and
splitK/reduce patterns that are not ideal under CUDA graph replay.

Investigate:

- whether selected owners call cuBLASLt with algorithms that create
  `cublasLt::splitKreduce_kernel`;
- whether small-M shapes are faster with no splitK, smaller splitK, different
  tile/stage choices, or a fixed workspace;
- whether `torch.mm`, `F.linear`, `torch.addmm`, `torch.bmm`, or explicit
  transposed-weight matmul select different algorithms for the same shapes;
- whether per-owner shape specialization can use a stable algorithm under CUDA
  graph capture;
- whether a small preallocated workspace avoids slow fallback paths or hidden
  allocation behavior.

Possible implementation:

- a small C++/CUDA extension around cuBLASLt matmul with explicit algorithm
  search and cached heuristic selection;
- a Python-side benchmark harness first, followed by a graph-safe opt-in
  wrapper only for the selected owners;
- per-shape algorithm cache initialized during model load or graph warmup.

Do not promote if the selected algorithm is not graph-safe or requires runtime
allocation during decode replay.

### Lane B: Prepacked / Pretransposed BF16 Weight Layout

Hypothesis: the cached BF16 weight path is compute-light enough that weight
layout and backend contract dominate.  A prepacked or alternative transposed
layout may select a better BF16 small-M backend.

Investigate:

- whether current cached BF16 weights are in the best layout for the chosen
  GEMM backend;
- whether storing both original and backend-preferred layouts improves
  selected owners enough to justify memory;
- whether WQA/WKV/compress, `q_wqb`, `wo_b`, `wo_a`, indexer `wq_b`, and
  shared expert projections need different layouts;
- whether any layout conversion currently happens during graph replay.

Memory rule:

- record extra bytes/rank;
- convert to GiB/rank, KV tokens, and pages at page size 256;
- do not hide memory cost inside a generic cache name.

### Lane C: Custom CUTLASS Or Triton BF16 Small-M Kernels

Hypothesis: generic library GEMMs are overkill for `M=1,4,8,16` decode shapes.
A specialized BF16 small-M kernel can remove fixed overhead and reduce splitK.

Candidate owners:

- WQA/WKV/compress `[M, 4096] x [1536, 4096]`;
- cached `q_wqb`;
- cached `wo_b` local projection;
- indexer `wq_b`;
- shared expert BF16 gate/up/down projections;
- `wo_a` grouped BMM residual if it remains visible after owner probes.

Implementation options:

- Triton small-M BF16 matmul specialized for fixed `K/N` shapes;
- CUTLASS/C++ extension with fixed SM80 tensor-core kernels and explicit
  epilogue;
- owner-specific kernels only for shapes that dominate the 07.69 cluster;
- optional static workspace, allocated before graph capture.

Correctness rule:

- compare output against the current promoted BF16 path for representative
  real weights and decode-small `M`;
- record max/mean/p99 absolute error and relative error;
- run TP8 text smoke before any macro claim.

### Lane D: Owner-Local Grouped Or Batched GEMM

Hypothesis: multiple independent small BF16 GEMMs inside the same layer or
owner boundary can be grouped so fixed overhead is paid fewer times.

Valid examples to check:

- projections that share the same input within a layer and are not separated
  by data dependencies;
- shared expert gate/up style paired projections if they are still separate in
  the promoted path;
- indexer/compressor projections that can be grouped without changing cache
  semantics;
- `wo_a` grouped BMM shape consolidation.

Invalid by default:

- batching the same projection across decoder layers, because layer `i+1`
  depends on layer `i`;
- batching across TP ranks without a communication redesign;
- fusing through attention/MoE boundaries unless correctness and scheduling
  are explicitly proven.

Possible implementation:

- `torch.bmm` / grouped BMM replacement when dimensions align;
- small custom grouped GEMM dispatcher;
- one fused projection kernel that emits multiple outputs for the same input.

### Lane E: Existing Activation-Quant Boundary Fusion As Context

TARGET 07.69 measured WQA/WKV GEMM-only around `0.043 ms`, while the local
boundary with the existing activation-quant helper was around `0.110 ms` at
`M=4`.  The projection/GEMM denominator in 07.69 counts BF16 GEMM kernels, but
activation quant is adjacent context.

Investigate only if the selected BF16 owner cannot be improved alone:

- whether the current activation-quant helper causes a backend choice that
  slows the following BF16 GEMM;
- whether a fused owner-local boundary can reduce total graph nodes while
  preserving the existing promoted precision contract;
- whether this belongs in a separate graph/layout or precision target.

Do not introduce a new activation quantization policy in 07.70.

### Lane F: Narrow Torch Compile / Inductor Probe

This is low priority and should stay narrow.

Try only if lanes A-D fail to produce a microbench candidate:

- compile a pure owner-local projection wrapper, not the whole decoder;
- measure graph safety and output equivalence;
- inspect whether Inductor changes the GEMM backend or only adds overhead.

Stop immediately if compile time, graph capture behavior, or generated kernels
make this unsuitable for the current benchmark harness.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_bf16_small_gemm_backend_cluster/
  README.md
  raw/
  summaries/
  scripts/
```

Record:

- git branch/status;
- current promoted variant and env bundle;
- TARGET 07.69 owner/backend tables;
- inactive 07.64 and 07.68 opt-ins;
- current memory ledger for promoted cached BF16 weights if available.

Use symlinks for large profile artifacts and benchmark directories.  Keep
small scripts and summaries in the milestone.

### 2. Build A Focused BF16 Small-GEMM Benchmark Suite

Start from 07.69's focused microbench and extend it only for selected cluster
representatives.

Required owner representatives:

- attention WQA/WKV/compress;
- one cuBLASLt cached BF16 projection among `q_wqb`, `wo_b`, or indexer
  `wq_b`;
- shared experts cached BF16 projection;
- `wo_a` grouped BMM residual if practical.

Context-only representatives:

- HC pre linear;
- MoE router / route projection.

For each representative, test decode-small `M` values:

```text
M = 1, 4, 8, 16
```

Report:

- mean/median/min ms;
- backend family and kernel names if available;
- output error vs current promoted path;
- whether graph capture/replay is expected to work;
- any extra memory/workspace.

### 3. Select One Implementation Candidate

Before editing runtime code, require:

- at least `15%` latency reduction in focused microbench for two or more BF16
  cluster representatives; or
- at least `25%` reduction for one representative that is responsible for
  `>=0.10s` in the 07.69 profile; and
- no correctness failure beyond the current promoted BF16 tolerance.

If no lane clears this gate, do not implement a runtime opt-in.  Write a
precision-policy or broader backend-design recommendation instead.

### 4. Implement One Opt-In Path

If a candidate clears the microbench gate, implement it behind a new explicit
toggle and variant.  Suggested names depend on the winning lane:

```text
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_CUBLASLT=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PREPACK=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_TRITON=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_GROUPED=1
```

Variant naming pattern:

```text
dsv4_sm80_a100_victory_bf16smallgemm
```

Rules:

- keep the toggle out of the victory bundle until promotion gates pass;
- keep stale experimental toggles out of the new variant unless they are
  already part of `dsv4_sm80_a100_victory`;
- if extra workspace/cache is needed, allocate it before graph capture and
  report owner/shape/dtype/bytes/lifetime;
- decode graph replay must remain active and eager decode must remain `0`.

### 5. Correctness And Smoke

Required:

- focused output comparison for every owner touched;
- unit/config test for any new variant/toggle;
- TP8 text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_bf16smallgemm \
  --output performance_milestones/target07_bf16_small_gemm_backend_cluster/raw/text_smoke.json
```

### 6. Macro And Profile Validation

Run same-run macro comparisons:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_bf16smallgemm \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 128 --batch-size 4 \
  --repeats 3 --warmup-repeats 1 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_bf16_small_gemm_backend_cluster/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

If the 4096/128 profile/macro is promising, run 4096/1024:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_bf16smallgemm \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 1024 --batch-size 4 \
  --repeats 3 --warmup-repeats 1 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_bf16_small_gemm_backend_cluster/raw/macro_4096x1024_bs4_np128 \
  --keep-going
```

Capture a 4096/128/batch4 profile for the candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nsys profile \
  -t cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cudabacktrace=none \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  -o performance_milestones/target07_bf16_small_gemm_backend_cluster/raw/nsys_bf16smallgemm_4096x128_bs4_np128 \
  torchrun --standalone --nproc_per_node=8 \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path /models/DeepSeek-V4-Flash \
    --variants dsv4_sm80_a100_victory_bf16smallgemm \
    --scenarios decode_throughput_bs8 \
    --prompt-len 4096 --decode-len 128 --batch-size 4 \
    --repeats 1 --warmup-repeats 0 \
    --page-size 256 --num-pages 128 \
    --output-dir performance_milestones/target07_bf16_small_gemm_backend_cluster/raw/nsys_macro_4096x128_bs4_np128 \
    --keep-going
```

Use local nsys syntax that works in this container.  Do not use unsupported
`-t nccl`.

### 7. Re-Run The 07.69 Classifier

Re-run or adapt the 07.69 projection/GEMM owner classifier on the candidate
profile and compare:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| projection/GEMM bucket | `0.778887s` | candidate | candidate |
| BF16 small-GEMM cluster | `0.521619s` | candidate | candidate |
| cuBLASLt BF16 GEMM | `0.219912s` | candidate | candidate |
| CUTLASS BF16 GEMM | `0.194319s` | candidate | candidate |
| cuBLASLt splitK/reduce | `0.107388s` | candidate | candidate |

Do not claim success from microbench alone.  The profile must show the
selected cluster actually moved.

## Promotion Gates

Correctness gate:

- focused owner output comparisons pass;
- TP8 text smoke passes;
- graph replay remains active;
- eager decode remains `0`.

Profile gate:

- projection/GEMM bucket decreases by at least `0.10s`; or
- BF16 small-GEMM cluster decreases by at least `20%`.

Strong profile goal:

- reduce BF16 small-GEMM cluster by `0.12s-0.16s`.

Macro gate:

- 4096/1024 same-run output throughput improves by at least `3%`;
- 4096/128 must not regress by more than `1%`.

Memory/workspace gate:

- report incremental bytes/rank for any prepack/cache/workspace;
- convert to GiB/rank, KV tokens, and pages;
- do not promote a memory-heavy path unless the macro gain justifies it.

Promotion decision:

- promote into `dsv4_sm80_a100_victory` only if correctness, profile, macro,
  and memory gates pass;
- otherwise keep the implementation as opt-in or remove it if it creates
  maintenance burden without evidence.

## Stop Rules

Stop without runtime implementation if:

- focused real-shape microbench cannot show at least `15%` latency reduction
  for two or more BF16 cluster representatives;
- every promising route requires changing precision, packed cache layout,
  broad compile ownership, or FP32 carrier semantics;
- the candidate is not graph-capture safe;
- the candidate requires large decode-time allocation.

Stop without promotion if:

- fresh 4096/128 profile does not reduce projection/GEMM by at least `0.10s`;
- BF16 cluster reduction is below `20%`;
- 4096/1024 same-run macro gain is below `3%`;
- text smoke or focused correctness fails after one focused fix attempt.

If exact BF16 backend work fails these gates, the final README should recommend
a dedicated precision-policy target.  It should name the vLLM mechanisms that
would have to be considered next, such as FP8 projection/cache layout,
`fused_inv_rope_fp8_quant`, `deepseek_v4_fp8_einsum`, or broader compile/runtime
ownership.

## Required Final README Contents

The milestone README must include:

- inherited TARGET 07.69 owner/backend table;
- current promoted macro baseline;
- all tried candidate lanes and why they passed or failed the microbench gate;
- selected implementation toggle and variant, if any;
- focused microbench table before/after;
- correctness and TP8 text-smoke results;
- 4096/128 and 4096/1024 same-run macro table;
- fresh 4096/128 projection/GEMM owner/backend comparison;
- memory/workspace ledger;
- vLLM source-parity note for the selected lane;
- decision: promote, keep opt-in, remove, or pivot to precision-policy target.

Suggested final decision format:

```text
Decision:
- Outcome:
- Winning lane, if any:
- Toggle/variant:
- Projection/GEMM delta:
- BF16 cluster delta:
- 4096/1024 macro delta:
- Memory/workspace cost:
- Promote status:
- Next target:
- Stop condition for next thread:
```
