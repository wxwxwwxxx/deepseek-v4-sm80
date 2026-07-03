# TARGET 07.72: DSV4 SM80 vLLM-Aligned FP8 / Custom Projection-Cache Boundary

Date: 2026-07-02

## Goal

Test whether a vLLM-aligned FP8/custom projection-cache boundary can reduce
the current promoted mini projection-cache cluster on A100/sm80.

This is an opt-in implementation target, but it must remain bounded:

- do not change the default `dsv4_sm80_a100_victory` path unless all gates pass;
- do not implement full `fp8_ds_mla` KV-cache E2E;
- do not change HC/router precision contracts in this target;
- do not treat vLLM's low-precision route as precision-neutral;
- start with source parity and focused real-shape microbench, then implement
  only if the gate is met.

The target follows TARGET 07.71's decision: exact BF16 small-GEMM layout work
has hit a practical platform, and exact-ish HC/router precision probes were
either too small or too risky.  The next promising surface is the coherent
projection-cache cluster that vLLM handles with quantized/custom boundaries.

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

Later same-run baselines may differ slightly due to normal run noise.  Always
compare candidate and baseline in the same run.

Inactive opt-ins that must not be used as baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
```

## Starting Evidence

### TARGET 07.70 Negative Result

The exact-route BF16 pretranspose path had local microbench signal but failed
profile and macro gates:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Projection/GEMM bucket | `0.778887s` | `0.778170s` | `-0.000717s` |
| BF16 cluster | `0.521619s` | `0.521012s` | `-0.000607s` |
| 4096/1024 output tok/s | `131.7927` | `131.9084` | `+0.09%` |
| Extra cache | `0` | `1.7559 GiB/rank` | too costly for no gain |

Interpretation: do not continue narrow BF16 layout/pretranspose polishing.

### TARGET 07.71 Pivot

HC/router exact-ish probes were rejected:

| Lane | Result | Decision |
| --- | --- | --- |
| HC/router TF32 | quality-stable but no decode-small speedup | reject |
| HC/router BF16-like | theoretical speed but router top-k changed on larger probe | reject |
| HC-only BF16-like | barely above `0.05s` theoretical gain and changes HC output | reject |

Selected next lane:

```text
vLLM-aligned FP8/custom projection-cache cluster
```

Target owner/cluster:

| Cluster member | Current surface |
| --- | ---: |
| WQA/WKV/compress | about `0.12s` |
| shared experts cached BF16 | about `0.09s` |
| `wo_a` | about `0.06s` |
| `q_wqb` | about `0.05s` |
| `wo_b` local | about `0.05s` |
| indexer projection pieces | about `0.08s` |
| coherent cluster | about `0.52s` |

Expected target:

- profile gain target: `0.12s-0.20s`;
- promotion floor: at least `0.10s` 4096/128 rank0 projection/GEMM reduction;
- macro floor: at least `+3%` same-run 4096/1024 output tok/s.

## Precision Policy For This Target

This target is allowed to test low-precision/custom projection boundaries only
as explicit opt-ins.  It is not a default precision-policy change.

Allowed:

- FP8/custom projection-cache wrappers for selected owners;
- vLLM-aligned FP8 weight/scale layout or custom op contracts when isolated
  from full KV-cache E2E;
- per-owner quality comparison against the promoted BF16 path;
- TP8 text smoke and focused hidden/logit smoke before macro claims.

Not allowed:

- default promotion without all gates;
- full FP8 KV-cache or `fp8_ds_mla` E2E;
- BF16-like router precision;
- HC `fn/post/comb` precision changes;
- broad whole-model compile or runtime rewrite.

Important HC note:

vLLM's HC contract stores/consumes `post/comb` in FP32, while mini's promoted
path currently keeps HC carrier tensors as BF16.  We are intentionally not
changing this in 07.72.  Large HC correctness/quality smoke should be deferred
until after the current performance route stabilizes.

## vLLM Source References

Use `/workspace/vllm-dsv4-docker` as the primary source:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/
/workspace/vllm-dsv4-docker/vllm/compilation/
```

Relevant mechanisms to inspect:

- `DeepseekV4FP8Config` and the `deepseek_v4_fp8` quantization stack;
- `DeepseekV4Attention.fused_wqa_wkv`;
- quantized `MergedColumnParallelLinear`, `ColumnParallelLinear`, and
  `RowParallelLinear` dispatch;
- `fused_inv_rope_fp8_quant`;
- `deepseek_v4_fp8_einsum`;
- `fused_indexer_q_rope_quant`;
- any SM80 fallback branches for FP8 projection/dequant/einsum;
- compile/custom-op boundaries only where directly tied to the selected
  projection-cache cluster.

Do not assume vLLM per-bucket runtime parity unless a real profile/probe
supports it.  Source parity plus focused mini probes are enough to choose the
mini implementation route.

## Mini Source Areas

Read before editing:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/layers/linear.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
performance_milestones/target07_precision_boundary_pivot/README.md
performance_milestones/target07_precision_boundary_pivot/summaries/vllm_precision_boundary_parity.md
performance_milestones/target07_projection_gemm_backend_owner_reattribution/README.md
performance_milestones/target07_bf16_small_gemm_backend_cluster/README.md
```

## Candidate Implementation Lanes

Try these lanes in order.  Stop when one lane clears the focused gate and can
be implemented as one bounded opt-in.

### Lane A: Shared FP8 Projection Wrapper For Cached-BF16 Owners

Hypothesis: the promoted cached BF16 path pays too much small-GEMM fixed cost.
A vLLM-aligned FP8/custom wrapper can keep weights/scales in a compact
projection-cache contract and reduce graph-replay GEMM cost across several
owners.

Candidate owners:

- WQA/WKV/compress;
- `q_wqb`;
- `wo_b` local;
- indexer `wq_b`;
- shared expert gate/up and down;
- indexer compressor/weights projection if source parity is clean.

Questions:

- Can mini use original FP8/scale tensors instead of cached BF16 weights for a
  selected owner?
- Can the custom wrapper avoid materializing BF16 weights or repeated scale
  tensors during graph replay?
- On SM80, does the custom path actually reduce time, or does software FP8
  dequant dominate?
- Is the output error acceptable against the promoted BF16 path?

Possible toggle:

```text
MINISGL_DSV4_SM80_FP8_CUSTOM_PROJECTION_CACHE=1
```

Possible variant:

```text
dsv4_sm80_a100_victory_fp8projcache
```

### Lane B: WQA/WKV Representative First

Hypothesis: WQA/WKV/compress is the best first representative because it maps
directly to vLLM's `fused_wqa_wkv` boundary and remains one of the largest
single pieces of the selected cluster.

Use this lane if a full shared wrapper is too broad for one thread.

Rules:

- do not treat WQA/WKV alone as sufficient for promotion unless profile impact
  is surprisingly large;
- use it to validate kernel mechanics, output error, graph capture, and
  vLLM-aligned layout;
- final report should say whether to expand the same wrapper to `q_wqb`,
  `wo_b`, shared experts, and indexer owners.

### Lane C: Shared Expert Projection Boundary

Hypothesis: shared expert cached BF16 compute remains visible after TARGET
07.66 removed staging.  A low-precision/custom projection path may reduce this
without touching routed MoE/Marlin.

Rules:

- do not change routed expert Marlin/WNA16 backend;
- do not change router precision;
- compare only shared expert gate/up/down outputs against promoted BF16 path;
- record any activation or residual error after shared expert combination.

### Lane D: `wo_a` / `wo_b` FP8 Boundary

Hypothesis: vLLM's `fused_inv_rope_fp8_quant` and `deepseek_v4_fp8_einsum`
boundary may be useful, but standalone `wo_a + wo_b` is below the current gate.

Use only as a secondary lane:

- if it composes naturally with the shared FP8 wrapper;
- or if source parity shows a surprisingly low-cost adaptation.

Do not make standalone `wo_a` the main target unless new evidence shows the
owner has re-grown.

### Lane E: Indexer Fused Q/Rope/Quant Continuation

Mini already promoted prior FP8 indexer/cache pieces.  TARGET 07.71 says the
remaining indexer projection surface is too small for a standalone target.

Use only if:

- the shared FP8 projection wrapper naturally covers indexer `wq_b`; or
- focused profiles show indexer/cache re-grew under the new boundary.

Do not select full packed indexer/cache redesign inside this target.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/
  README.md
  raw/
  summaries/
  scripts/
```

Record:

- git branch/status;
- current promoted baseline;
- TARGET 07.70 negative result;
- TARGET 07.71 pivot decision;
- inactive opt-ins;
- selected vLLM source files/functions.

### 2. Source Parity And Surface Mapping

Produce:

```text
summaries/fp8_projection_source_parity.md
summaries/fp8_projection_surface_mapping.md
```

The mapping table must include:

| Mini owner | Current time | Current contract | vLLM analogue | FP8/custom candidate | Include? |
| --- | ---: | --- | --- | --- | --- |

Minimum mapping gate:

- map the candidate custom FP8 projection boundary to at least `0.35s` of
  coherent current mini surface; or
- stop without runtime implementation and write the missing source/profiling
  evidence.

### 3. Focused Real-Shape Microbench

Build or adapt scripts under `scripts/` to test real model weights/scales for
selected owners.

Required representatives:

- WQA/WKV/compress;
- one of `q_wqb`, `wo_b`, or indexer `wq_b`;
- shared expert gate/up or down.

Optional representatives:

- `wo_a` / `wo_b` boundary;
- indexer fused q/rope/quant continuation.

For each representative, compare:

- promoted BF16 path;
- vLLM-aligned FP8/custom candidate;
- any intermediate dequant-on-load or dequant-on-the-fly variant needed to
  understand the cost.

Measure decode-small shapes:

```text
M = 1, 4, 8, 16
```

Report:

- mean/median/min latency;
- kernel family if available;
- output max/mean/p99 absolute error;
- relative error where meaningful;
- extra cache/workspace bytes;
- graph capture safety expectation.

Focused gate:

- at least two representative owners improve by `>=15%`; and
- no representative has unacceptable output error; and
- no candidate requires decode-time allocation.

If this gate fails, stop without runtime implementation.

### 4. Implement One Opt-In Candidate

If the focused gate passes, implement one explicit opt-in.

Suggested naming:

```text
MINISGL_DSV4_SM80_FP8_CUSTOM_PROJECTION_CACHE=1
dsv4_sm80_a100_victory_fp8projcache
```

If the implementation is narrower, use a narrower toggle name:

```text
MINISGL_DSV4_SM80_FP8_WQA_WKV_PROJECTION=1
MINISGL_DSV4_SM80_FP8_SHARED_EXPERT_PROJECTION=1
```

Rules:

- keep the toggle out of `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE` until all
  promotion gates pass;
- allocate any projection cache/workspace before graph capture;
- do not rebuild or allocate large buffers during decode replay;
- keep existing promoted exact BF16 path as fallback;
- do not implicitly enable 07.64, 07.68, or 07.70 opt-ins.

### 5. Correctness And Quality Gates

Required before macro:

- focused owner output comparison vs promoted BF16 path;
- if attention boundary changes, compare downstream attention/projection
  output where practical;
- if shared expert boundary changes, compare shared expert output;
- TP8 text smoke for baseline and candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_fp8projcache \
  --output performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/raw/text_smoke.json
```

Quality report must include:

- output max/mean/p99 abs error;
- relative error or cosine similarity where useful;
- any route/top-k effect if indexer is touched;
- generated text smoke pass/fail;
- a clear note that this is opt-in precision work, not default promotion.

### 6. Macro And Profile Validation

Run same-run short macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_fp8projcache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 128 --batch-size 4 \
  --repeats 3 --warmup-repeats 1 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

If short macro/profile is promising, run long macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_fp8projcache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 1024 --batch-size 4 \
  --repeats 3 --warmup-repeats 1 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/raw/macro_4096x1024_bs4_np128 \
  --keep-going
```

Capture candidate 4096/128 profile:

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
  -o performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/raw/nsys_fp8projcache_4096x128_bs4_np128 \
  torchrun --standalone --nproc_per_node=8 \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path /models/DeepSeek-V4-Flash \
    --variants dsv4_sm80_a100_victory_fp8projcache \
    --scenarios decode_throughput_bs8 \
    --prompt-len 4096 --decode-len 128 --batch-size 4 \
    --repeats 1 --warmup-repeats 0 \
    --page-size 256 --num-pages 128 \
    --output-dir performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/raw/nsys_macro_4096x128_bs4_np128 \
    --keep-going
```

Use local nsys syntax that works.  Do not use unsupported `-t nccl`.

### 7. Re-run Owner/Backend Classifier

Reuse or adapt the 07.69 classifier.

Compare:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| projection/GEMM bucket | `0.778887s` | candidate | candidate |
| BF16 small-GEMM cluster | `0.521619s` | candidate | candidate |
| FP8/custom projection cluster | baseline | candidate | candidate |
| touched owners total | baseline | candidate | candidate |
| graph replay count | baseline | candidate | candidate |
| eager decode count | baseline | candidate | candidate |

Do not claim success from focused microbench alone.

## Promotion Gates

Correctness/quality gate:

- focused owner output comparison passes under the target's stated tolerance;
- TP8 text smoke passes;
- graph replay remains active;
- eager decode remains `0`;
- no obvious generated-text corruption or repeated gibberish.

Profile gate:

- projection/GEMM bucket decreases by at least `0.10s`; or
- touched projection-cache cluster decreases by at least `20%` and the final
  reduction is at least `0.10s`.

Macro gate:

- 4096/1024 same-run output throughput improves by at least `3%`;
- 4096/128 must not regress by more than `1%`.

Memory/workspace gate:

- report extra bytes/rank for FP8/custom projection cache and workspace;
- convert to GiB/rank, KV tokens, and pages at page size 256;
- do not promote a memory-heavy path unless profile and macro gains justify it.

Promotion decision:

- promote into `dsv4_sm80_a100_victory` only if all gates pass;
- otherwise keep opt-in only or remove if maintenance cost exceeds value.

## Stop Rules

Stop without runtime implementation if:

- source parity cannot map the candidate to at least `0.35s` of coherent mini
  surface;
- focused microbench does not show `>=15%` latency reduction on at least two
  representative owners;
- output error is clearly too large before text smoke;
- the only viable route requires full FP8 KV-cache E2E or broad model compile.

Stop without macro if:

- focused correctness fails;
- graph capture/replay is not safe;
- the candidate requires large decode-time allocations.

Stop without promotion if:

- fresh 4096/128 profile reduction is below `0.10s`;
- same-run 4096/1024 macro gain is below `3%`;
- text smoke fails or shows suspicious output;
- memory cost is large and performance gain is marginal.

## Required Final README Contents

The final README must include:

- TARGET 07.71 decision summary;
- vLLM source parity table;
- mini surface mapping table;
- focused microbench before/after;
- quality/error table;
- implementation toggle and variant, if any;
- text smoke result;
- 4096/128 and 4096/1024 macro table if run;
- fresh profile owner/backend comparison;
- memory/workspace ledger;
- decision: promote, keep opt-in, remove, or pivot;
- explicit next target and stop condition.

Suggested final decision format:

```text
Decision:
- Outcome:
- Selected lane:
- Toggle/variant:
- Touched owners:
- Quality gate result:
- Projection/GEMM delta:
- Touched cluster delta:
- 4096/1024 macro delta:
- Memory/workspace cost:
- Promote status:
- Next target:
- Stop condition for next target:
```
