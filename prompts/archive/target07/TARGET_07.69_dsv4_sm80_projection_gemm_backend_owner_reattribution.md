# TARGET 07.69: DSV4 SM80 Projection/GEMM Backend and Owner Re-Attribution

Date: 2026-07-02

## Goal

Re-attribute the remaining projection/GEMM bucket after TARGET 07.66, 07.67,
and 07.68, then select the next evidence-backed implementation target.

This is a measurement and decision target.  It may add profiling-only NVTX,
classifier scripts, source probes, and focused microbenchmarks.  It should not
land a new projection kernel, precision route, cache expansion, communication
change, or CUDA graph rewrite unless the final report explicitly selects that
as the next target.

The key question is no longer the old TARGET 07.57 question.  Since 07.57, the
project has promoted cached BF16 projection weights for `attn.q_wqb`,
`attn.wo_b`, `indexer.wq_b`, `attn.wo_a`, and shared expert gate/up/down
projections.  The old `_quantized_linear_fp8_kernel` owner table is therefore
stale.  This target must explain what remains inside the current promoted
projection/GEMM bucket before any further backend work.

## Current Promoted Baseline

Use the promoted variant:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Current confirmed promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

The 07.66 shared expert BF16 cache is promoted:

```text
MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1
```

The 07.64 metadata deforestation path remains opt-in only.  The 07.68 HC graph
cleanup path also remains opt-in only:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1  # not promoted
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1          # not promoted
```

Do not use either opt-in path as the baseline for this target unless the final
report clearly labels it as an ablation.

## Starting Evidence

TARGET 07.67 fresh 4096/128/batch4 rank0 decode-envelope buckets:

| Bucket | Kernel s | Share | Current decision |
| --- | ---: | ---: | --- |
| projection/GEMM | `0.778887` | `26.31%` | largest bucket; re-attribute now |
| direct-copy/layout | `0.557626` | `18.84%` | broad and owner-diffuse after 07.66 |
| HC/elementwise | `0.536306` | `18.12%` | 07.68 tested opt-in cleanup |
| NCCL communication | `0.338786` | `11.45%` | defer until selected by evidence |
| MoE routed/backend | `0.300138` | `10.14%` | not primary after Marlin/shared-cache work |

TARGET 07.68 then tested an exact HC/elementwise opt-in:

| Metric | 07.67 promoted | 07.68 hccleanup | Delta |
| --- | ---: | ---: | ---: |
| HC/elementwise bucket | `0.536306s` | `0.519303s` | `-0.017003s` |
| direct-copy/layout bucket | `0.557626s` | `0.520151s` | `-0.037475s` |
| projection/GEMM bucket | `0.778887s` | `0.779055s` | `+0.000168s` |
| 4096/1024 output tok/s | `131.5675` same-run | `132.5223` | `+0.73%` |

07.68 correctness passed, but profile and long macro gates failed.  Keep
`MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1` as an opt-in experiment only.  The next
largest stable bucket is still projection/GEMM.

## Why This Target Exists

The project has already removed several obvious projection bottlenecks:

- cached BF16 `attn.q_wqb`;
- cached BF16 row-parallel `attn.wo_b` local projection;
- cached BF16 `indexer.wq_b`;
- cached BF16 `attn.wo_a` grouped BMM boundary;
- cached BF16 shared expert gate/up/down projections.

Those changes made the old TARGET 07.57 owner table obsolete.  A fresh
projection/GEMM bucket may now contain a very different mix:

- HC pre matmul / `linear_bf16_fp32_fallback`;
- remaining attention projections such as `wq_a`, `wkv`, `q_proj`, or `wo_b`
  communication-adjacent work;
- cached BF16 projection GEMMs whose intrinsic small-M backend is still
  expensive;
- shared expert cached BF16 compute rather than direct-copy staging;
- vocab/head or sampler-adjacent GEMMs;
- cublasLt split-K reduce kernels or small GEMM epilogues;
- miscellaneous `torch.mm`, `F.linear`, or CUTLASS/cuBLAS backend calls hidden
  under graph replay.

This target should name the owners and backend contracts precisely before any
more time is spent on local kernel polishing.

## Scope

In scope:

- fresh owner-level attribution of the current promoted projection/GEMM bucket;
- projection-specific profiling-only NVTX if existing graph/source NVTX is not
  enough;
- CUDA graph `originalGraphNodeId` mapping for GEMM/cublasLt/CUTLASS kernels;
- backend-family classification: cuBLAS SGEMM, cuBLASLt BF16, CUTLASS BF16,
  cublasLt splitK/reduce, torch `F.linear`/`bmm`, cached BF16 weight paths,
  `linear_bf16_fp32_fallback`, and residual FP8 quantized-linear paths;
- focused real-shape microbenchmarks for the top one or two owners;
- source comparison against vLLM's DeepSeek V4 SM80 projection paths;
- one clear next-target recommendation with expected profile and macro impact.

Out of scope:

- implementing a new GEMM kernel or promoting an opt-in;
- changing default precision, including FP8, INT8, TF32, or FP32 carrier
  policy;
- expanding cached BF16 weights without a fresh owner/memory case;
- changing NCCL/all-reduce contracts;
- sparse attention, indexer cache, full `fp8_ds_mla`, or KV-cache layout work;
- MoE runner/finalization cleanup unless the owner table proves it is inside
  projection/GEMM and is the largest actionable owner;
- promoting 07.64 metadata deforestation or 07.68 HC graph cleanup.

## Mini Source Areas

Read these first:

```text
prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md
prompts/TARGET_07.57_dsv4_sm80_projection_gemm_backend_parity.md
prompts/TARGET_07.66_dsv4_sm80_moe_shared_expert_staging_cleanup.md
prompts/TARGET_07.67_dsv4_sm80_post_shared_expert_reprofile.md
prompts/TARGET_07.68_dsv4_sm80_hc_elementwise_graph_cleanup.md
performance_milestones/target07_hc_elementwise_graph_cleanup/README.md
```

Primary mini files:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/layers/linear.py
python/minisgl/engine/graph.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

Use `rg` to find the current toggle names and projection-cache branches before
editing.  Do not assume the branch names from old targets are still current.

## vLLM Reference Areas

Old vLLM source root:

```text
/workspace/vllm-dsv4-docker
```

Relevant files:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/mhc.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/
/workspace/vllm-dsv4-docker/vllm/compilation/
```

For each top mini owner, report the closest vLLM source boundary.  If vLLM
uses a different precision contract, say that explicitly instead of treating
the backend as a drop-in replacement.

## Owners To Attribute

The first deliverable is a table like this:

| Owner | Kernel s | Count | Backend family | Shape hint | Source boundary | vLLM analogue | Decision |
| --- | ---: | ---: | --- | --- | --- | --- | --- |

At minimum, try to split:

| Candidate owner | Mini source boundary | Notes |
| --- | --- | --- |
| HC pre linear | `DeepseekV4DecoderLayer._hc_pre`, `linear_bf16_fp32_fallback` | 07.68 left this backend unchanged. |
| attention WQA/WKV | `DSV4Attention.forward` fused or paired projections | Check whether remaining projection bucket moved here. |
| attention `q_wqb` | cached BF16 path | Should no longer be old FP8 wrapper; verify actual backend. |
| attention `wo_a` | BF16 grouped BMM cache | It was optimized in 07.62; confirm residual is small. |
| attention `wo_b` | row-parallel local projection plus all-reduce-adjacent work | Split local GEMM from communication. |
| indexer `wq_b` | cached BF16 path | It was optimized in 07.60; verify residual. |
| indexer weight/logit projection | indexer weights/logits path | May remain small but should be named. |
| shared experts | cached BF16 shared expert gate/up/down | 07.66 removed direct-copy staging; compute may remain. |
| routed MoE projection pieces | Marlin WNA16 or runner side projections | Keep separate from routed MoE backend if possible. |
| `lm_head` / logits | vocab/head projection | Often visible in direct-copy owner tables. |
| cublasLt splitK/reduce | backend-generated reduce kernels | Attribute to parent GEMM if possible. |

If one owner cannot be split with existing NVTX, add profiling-only NVTX behind
a default-off env flag such as:

```text
MINISGL_DSV4_PROFILE_PROJECTION_NVTX=1
```

The flag must not change behavior, allocate large tensors, synchronize, or
enter the victory bundle.

## Backend Contracts To Compare

For each owner responsible for at least `0.05s` in the 4096/128 decode
envelope, identify the backend contract:

- cached BF16 dequantized weight + `F.linear` / cuBLAS / cuBLASLt;
- cached BF16 grouped BMM;
- `linear_bf16_fp32_fallback`;
- CUTLASS BF16 GEMM;
- cuBLAS SGEMM / BF16 GEMM;
- cublasLt splitK/reduce kernels;
- remaining `_quantized_linear_fp8_kernel` or activation-quant wrappers;
- Marlin WNA16 or MoE runner-owned projection-like work;
- graph/runtime copy or layout materialization accidentally counted as GEMM.

Do not collapse all GEMMs into one bucket.  The output needs to say which
backend family is slow and who owns it.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_projection_gemm_backend_owner_reattribution/
  README.md
  raw/
  summaries/
  scripts/
```

Use symlinks for large `.nsys-rep`, `.sqlite`, and `/tmp` benchmark output
directories.  Small classifier scripts, tables, and notes should be copied or
written directly into the milestone.

Record:

- git branch/status;
- current promoted variant and bundle;
- inactive 07.64 and 07.68 opt-ins;
- inherited 07.67 macro and bucket table;
- 07.68 decision and why it was not promoted.

### 2. Reuse Existing Profiles First

Before capturing a new trace, inspect available 07.67 and 07.68 SQLite/profile
artifacts.  Try to extract:

- GEMM/CUTLASS/cuBLAS kernel names and durations;
- graph node IDs and parent NVTX ranges;
- existing owner/source labels;
- whether the projection/GEMM bucket is dominated by a few kernel families or
  many small calls.

If existing profiles cannot attribute at least `60%` of the projection/GEMM
bucket to concrete owners, add projection-specific profiling NVTX and recapture.

### 3. Capture A Fresh Short Profile If Needed

Suggested workload:

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
  -o performance_milestones/target07_projection_gemm_backend_owner_reattribution/raw/nsys_projection_owner_4096x128_bs4_np128 \
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
    --output-dir performance_milestones/target07_projection_gemm_backend_owner_reattribution/raw/macro_4096x128_bs4_np128 \
    --keep-going
```

Enable only profiling flags that are needed:

```text
MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1
MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1     # optional context
MINISGL_DSV4_PROFILE_PROJECTION_NVTX=1      # only if added in this target
```

Use the local nsys syntax that works in this container.  Do not use unsupported
`-t nccl` syntax.

### 4. Build The Owner/Backend Classifier

Produce at least these summaries:

```text
summaries/projection_gemm_owner_table.md
summaries/projection_gemm_backend_families.md
summaries/projection_gemm_top_kernels.md
summaries/vllm_projection_source_parity.md
```

The owner table should include:

- owner name;
- kernel seconds;
- kernel count;
- share of projection/GEMM;
- top kernel names;
- inferred M/N/K or source shape when available;
- backend family;
- source file/function;
- vLLM analogue;
- next-action decision.

Keep residual/unattributed time visible.  Do not hide it inside "misc" if it
is large.

### 5. Run Focused Real-Shape Microbenchmarks

For the top one or two owners, build focused scripts under `scripts/` using
real loaded model weights when practical.  Measure decode-small shapes such as:

```text
M = 1, 4, 8, 16
```

Report:

- current mini path latency;
- backend kernel family;
- graph/capture safety where applicable;
- output comparison against the current promoted path;
- memory footprint if any cached data is used;
- whether a vLLM source backend looks adaptable.

Only microbench the owners that the profile selected.  Do not benchmark every
projection in the model.

### 6. Compare Against vLLM Source Boundaries

For each top owner, answer:

- Does vLLM use the same dtype contract?
- Does vLLM use a custom op, torch.compile boundary, cached weight layout,
  packed low-precision layout, or a plain cuBLAS/cuBLASLt call?
- Is the vLLM mechanism compatible with mini's current exact BF16 route?
- If not compatible, is it a precision target rather than an exact backend
  target?
- Is there a low-risk partial adaptation, or would porting require a larger
  model-boundary redesign?

Source parity is enough for this target if vLLM profiles are unavailable or
noisy.  Mark any inference as source-based rather than measured.

## Gates

Attribution gate:

- explain at least `80%` of the projection/GEMM bucket by owner/backend; or
- explain owners totaling at least `0.60s` of the 4096/128 decode-envelope
  projection/GEMM bucket.

If the attribution gate fails, stop after documenting what instrumentation is
missing.  Do not implement a projection optimization.

Next-target gate:

- select an owner/backend responsible for at least `0.20s`; or
- select a same-backend family cluster responsible for at least `0.35s`; and
- provide a plausible path to at least `3%` 4096/1024 macro gain.

If no owner/backend clears this gate, stop and recommend a broader measurement
or precision-policy target instead of speculative kernel work.

Implementation deferral rule:

- This target should end by writing the next implementation target proposal.
  It should not both discover the owner and implement a major backend in the
  same thread.

## Required Final README Contents

The milestone README must end with:

- current promoted macro baseline;
- 07.68 opt-in status and why it is not the baseline;
- projection/GEMM owner table;
- backend family table;
- vLLM source parity table for the selected owners;
- focused microbench results for the selected owner(s);
- one recommended next target;
- explicit do-not-continue condition.

Suggested final decision format:

```text
Decision:
- Next target: TARGET 07.xx <name>
- Primary owner/backend:
- Expected profile gain:
- Expected 4096/1024 macro gain:
- Why not precision yet / why precision now:
- Stop condition for the next target:
```
