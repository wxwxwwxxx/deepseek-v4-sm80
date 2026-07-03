# TARGET 07.71: DSV4 SM80 Precision / Boundary Pivot

Date: 2026-07-02

## Goal

Choose the next high-leverage direction after TARGET 07.70 showed that narrow
exact BF16 small-GEMM layout/backend work does not move macro performance.

This is a short measurement and decision target.  It may run focused
microbenchmarks, small precision ablations, source parity reviews, and one or
two smoke/macro checks.  It should not implement a full FP8 projection/cache
route, full `fp8_ds_mla`, broad compile/runtime rewrite, or promotion into the
victory bundle.

The main question:

```text
Should the next implementation target attack exact-ish FP32/SGEMM owners
such as HC pre linear and MoE router, or pivot to a vLLM-aligned
low-precision/custom-boundary path such as FP8 projection/cache, fused
indexer/rope/quant, or FP8 wo_a/einsum?
```

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

Same-run TARGET 07.70 baseline was slightly higher due to normal run noise:

| Workload | Baseline output tok/s | Candidate output tok/s | Candidate delta |
| --- | ---: | ---: | ---: |
| 4096/128/batch4 | `62.3274` | `62.3750` | `+0.08%` |
| 4096/1024/batch4 | `131.7927` | `131.9084` | `+0.09%` |

Do not use these opt-ins as baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1         # TARGET 07.64, not promoted
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1                 # TARGET 07.68, not promoted
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1     # TARGET 07.70, not promoted
```

## Starting Evidence

### TARGET 07.69

Projection/GEMM owner/backend attribution:

| Metric | Value |
| --- | ---: |
| Projection/GEMM bucket | `0.778887s` |
| Named coverage | `98.94%` |
| BF16 small-GEMM + splitK/reduce cluster | `0.521619s` |
| FP32/SGEMM small-GEMM cluster | `0.257269s` |
| Residual `_quantized_linear_fp8_kernel` | `0.000000s` |

Largest exact-route FP32 owners:

| Owner | Kernel s | Backend |
| --- | ---: | --- |
| HC pre linear | `0.178373s` | cuBLAS SGEMM/FP32 + splitK/reduce |
| MoE router / route projection | `0.097109s` | cuBLAS SGEMM/FP32 + splitK/reduce |
| `lm_head` | `0.026769s` | cuBLAS SGEMM/FP32 |

### TARGET 07.70

BF16 pretranspose candidate:

- variant: `dsv4_sm80_a100_victory_bf16smallgemm`;
- toggle: `MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1`;
- focused microbench gate passed for multiple owners;
- TP8 text smoke passed;
- graph replay stayed active and eager decode stayed `0`;
- extra cache: `1.7559 GiB/rank`, about `24,796` KV tokens or `96.86`
  pages at page size 256.

But profile and macro gates failed:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Projection/GEMM bucket | `0.778887s` | `0.778170s` | `-0.000717s` |
| BF16 cluster | `0.521619s` | `0.521012s` | `-0.000607s` |
| cuBLASLt BF16 GEMM | `0.219912s` | `0.313612s` | `+42.61%` |
| CUTLASS BF16 GEMM | `0.194319s` | `0.095964s` | `-50.62%` |
| cuBLASLt splitK/reduce | `0.107388s` | `0.111436s` | `+3.77%` |
| 4096/1024 output tok/s | `131.7927` | `131.9084` | `+0.09%` |

Interpretation: pretransposed BF16 weights changed backend-family mix but did
not reduce aggregate graph-replay projection/GEMM cost.  Do not continue
narrow BF16 layout polishing without new evidence.

## vLLM Reference Context

Use vLLM as the main comparison source:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/mhc.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/
/workspace/vllm-dsv4-docker/vllm/compilation/
```

Known vLLM boundary differences:

- `DeepseekV4FP8Config` / `deepseek_v4_fp8`;
- quantized linear layers instead of mini's exact cached BF16 projections for
  many boundaries;
- `torch.ops.vllm.deepseek_v4_attention`;
- `fused_inv_rope_fp8_quant`;
- `torch.ops.vllm.deepseek_v4_fp8_einsum`;
- fused indexer q/rope/quant and packed low-precision cache/indexer layout;
- `mhc_pre` / `mhc_post` with FP32 `post/comb` in the reference HC contract;
- `@support_torch_compile` boundaries.

This target should identify which of these differences is the most plausible
next implementation target for mini, and what precision/quality contract would
be required.

## Scope

In scope:

- short source parity review of current mini versus vLLM for HC pre, MoE
  router, WQA/WKV, `wo_a`, indexer, and FP8 projection/cache boundaries;
- focused microbench or small ablation for HC/router FP32/SGEMM precision
  choices such as TF32, BF16, or vLLM-like dtype contracts;
- focused probe of whether a vLLM-style FP8 projection/cache boundary is
  isolated enough to become the next target;
- quality-risk notes for any precision-changing path;
- one clear next-target recommendation.

Out of scope:

- promoting any opt-in path;
- implementing full FP8 projection/cache E2E;
- implementing `fp8_ds_mla` KV cache E2E;
- changing default precision policy;
- broad whole-model `torch.compile`;
- reopening exact BF16 small-GEMM pretranspose;
- continuing generic graph/layout cleanup;
- NCCL/communication optimization;
- MoE routed Marlin replacement.

## Candidate Pivot Lanes

### Lane A: Exact-ish HC / Router FP32 Ownership

Question: can the remaining FP32/SGEMM owners be reduced without accepting a
large quality risk?

Profile surface:

- HC pre linear: about `0.18s`;
- MoE router / route projection: about `0.10s`;
- combined FP32/SGEMM cluster: about `0.26s`.

Investigate:

- whether enabling TF32 for only these FP32 GEMMs is possible or whether the
  available knob is global;
- whether TF32 changes router decisions, top-k experts, or text smoke output;
- whether BF16 input/weight/output variants are acceptable for HC pre linear;
- whether router logits must remain FP32 to preserve expert routing quality;
- whether vLLM uses an equivalent precision contract or relies on a different
  fused-MoE/compile boundary.

Allowed probes:

- microbench HC pre linear and MoE router with current FP32, TF32-enabled, and
  BF16-like variants;
- compare output error, top-k route overlap, and text smoke if a probe looks
  promising;
- run a tiny same-run macro only if a probe has a plausible `>=0.05s`
  projection/GEMM reduction.

Do not promote TF32 or BF16 router changes in this target.

### Lane B: vLLM-Aligned FP8 Projection / Cache Boundary

Question: is the next real gap closer a vLLM-style low-precision/custom
boundary rather than exact BF16 backend work?

Investigate:

- WQA/WKV / fused projection boundary;
- `fused_inv_rope_fp8_quant`;
- `deepseek_v4_fp8_einsum` / `wo_a`-to-`wo_b` boundary;
- indexer fused q/rope/quant and packed cache/indexer layout;
- relationship between existing mini FP8 indexer work and vLLM's current
  source path;
- which pieces can be ported as standalone kernels versus requiring model
  boundary redesign.

Deliverables:

- a short table of candidate vLLM mechanisms;
- for each mechanism: expected mini owner/bucket, precision contract,
  required cache/layout changes, quality risk, and estimated engineering size;
- one recommended implementation target if FP8/custom-boundary is the winner.

Do not implement the full route here.

### Lane C: Boundary / Compile Ownership

Question: is the remaining gap mostly a boundary ownership issue rather than a
single kernel issue?

Investigate only briefly:

- whether vLLM's `@support_torch_compile` removes boundaries that mini still
  executes as graph nodes;
- whether narrow owner-local compile probes are likely to affect the selected
  surfaces;
- whether previous narrow compile/preflight failures already make this
  low-priority.

This lane should not become broad compile work inside 07.71.  It is mainly to
avoid missing an obvious vLLM mechanism.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_precision_boundary_pivot/
  README.md
  raw/
  summaries/
  scripts/
```

Record:

- current git status;
- promoted baseline;
- TARGET 07.69 summary;
- TARGET 07.70 negative result;
- inactive opt-ins.

### 2. Build A Pivot Table

Produce:

```text
summaries/pivot_candidate_table.md
summaries/vllm_precision_boundary_parity.md
summaries/hc_router_precision_probe.md
```

The pivot table should include:

| Lane | Target owner/bucket | Expected gain | Quality risk | Engineering size | Decision |
| --- | --- | ---: | --- | --- | --- |

### 3. Run Minimal HC/Router Precision Probes

For HC pre linear and MoE router:

- current FP32/SGEMM baseline;
- TF32-enabled variant, if locally controllable;
- BF16-like variant, only as a quality-risk probe;
- top-k route overlap for router;
- output error for HC pre;
- note whether PyTorch/global TF32 controls make this practical in mini.

Gate for choosing an HC/router implementation target:

- combined focused estimate suggests at least `0.05s` rank0 decode-envelope
  reduction; and
- router top-k overlap is acceptable; and
- TP8 text smoke or a cheap semantic smoke does not show obvious damage; and
- the implementation can be scoped to HC/router instead of globally changing
  every FP32 matmul.

If these conditions do not hold, do not choose HC/router as the next
implementation target.

### 4. Build vLLM FP8 / Boundary Candidate Ranking

For each vLLM mechanism, document:

- source file/function;
- mini analogue;
- current mini owner/bucket from 07.69/07.70;
- precision contract;
- cache/layout requirement;
- quality gate required;
- expected profile surface;
- whether it can be isolated into one target.

Candidate mechanisms to rank:

- FP8 WQA/WKV or projection cache boundary;
- `fused_inv_rope_fp8_quant`;
- `deepseek_v4_fp8_einsum` / `wo_a` boundary;
- fused indexer q/rope/quant;
- packed FP8 indexer/cache continuation from prior targets;
- vLLM compile/custom-op boundary where it directly covers one of these
  surfaces.

Gate for choosing an FP8/boundary implementation target:

- a source boundary maps to at least `0.20s` owner/backend surface, or a
  coherent cluster maps to at least `0.35s`;
- quality gate is definable before implementation;
- engineering scope can be expressed as one target, not a full model rewrite.

### 5. Optional Tiny Macro/Ablation

Run a tiny macro only if a precision probe is concrete enough to be meaningful.
Otherwise avoid long benchmark runs and finish with a decision report.

Any macro must use:

```text
--variants dsv4_sm80_a100_victory <explicit opt-in/probe variant>
--prompt-len 4096
--decode-len 128
--batch-size 4
--page-size 256
--num-pages 128
```

Do not spend time on 4096/1024 unless a 4096/128 probe is clearly promising.

## Decision Rules

Choose exact-ish HC/router next only if:

- the probe can plausibly reduce at least `0.05s` rank0 decode-envelope time;
- quality risk is low and measurable;
- implementation can be isolated to HC/router or a tightly scoped owner.

Choose vLLM-aligned FP8/boundary next if:

- exact-ish HC/router probes are too small or too risky;
- one FP8/custom boundary maps to a large enough owner/cluster;
- quality and fallback gates are clear.

Choose a broader architecture target only if:

- neither HC/router nor FP8/custom-boundary has an isolated target;
- the evidence points to compile/runtime ownership or cache-layout architecture
  as the next bottleneck.

## Stop Rules

Stop after the pivot decision.  Do not implement the selected large target in
the same thread.

Hard stops:

- HC/router probes show less than `0.05s` plausible rank0 profile gain;
- router precision probes noticeably change top-k routing without a clear
  quality path;
- FP8/custom-boundary candidates cannot be mapped to a concrete mini
  owner/cluster;
- the next candidate requires full FP8 KV-cache E2E or whole-model compile
  before any isolated value can be tested.

## Required Final README Contents

The final README must include:

- TARGET 07.70 negative-result summary;
- HC/router precision probe table;
- vLLM FP8/custom-boundary ranking table;
- quality-risk notes;
- one selected next target;
- explicit "do not continue" note for lanes that failed.

Suggested final decision format:

```text
Decision:
- Next target:
- Selected lane:
- Target owner/cluster:
- Expected profile gain:
- Expected macro gain:
- Quality gate:
- Why not the other lanes:
- Stop condition for the next target:
```
