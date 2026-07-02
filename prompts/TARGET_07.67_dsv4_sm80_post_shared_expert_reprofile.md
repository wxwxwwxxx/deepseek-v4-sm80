# TARGET 07.67: DSV4 SM80 Post-Shared-Expert Reprofile and Next Bottleneck Reset

Date: 2026-07-02

## Goal

Reprofile the current promoted A100/sm80 victory path after TARGET 07.66 and
select the next evidence-backed implementation target.

This is a measurement and decision target.  It should not implement a new
performance optimization.  TARGET 07.66 changed the bottleneck shape enough
that the previous owner order is stale; do not continue into runner
finalization, communication, HC/elementwise, shared-expert overlap, or INT8 MoE
until the fresh profile selects one.

## Current Promoted Baseline

Current promoted variant:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

TARGET 07.66 promoted:

```text
MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1
```

The explicit audit variant remains available:

```text
dsv4_sm80_a100_victory_sharedbf16
```

After TARGET 07.66, both variants should activate the shared expert BF16 weight
cache.  Use the promoted `dsv4_sm80_a100_victory` name for new official
reports, and use the audit variant only when comparing old artifacts.

## Starting Evidence

TARGET 07.66 result:

| Workload | 07.63/07.65 baseline output tok/s | 07.66 output tok/s | Delta | 07.66 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `59.5264` | `62.2034` | `+4.50%` | `168.6592` |
| 4096/1024/batch4 | `119.4153` | `131.7707` | `+10.35%` | `169.1898` |

Direct-copy owner movement from 07.66:

| Metric | 07.65 baseline | 07.66 shared cache | Delta |
| --- | ---: | ---: | ---: |
| total direct_copy | `0.737039s` | `0.449052s` | `-39.07%` |
| MoE/shared staging group | `0.379204s` | `0.097361s` | `-74.32%` |
| `shared_experts.gate_up_proj` | `0.165751s` | `0.000000s` | removed |
| `shared_experts.down_proj` | `0.119724s` | `0.000000s` | removed |

Memory tradeoff:

| Cache | Bytes/rank | GiB/rank | KV pages/rank |
| --- | ---: | ---: | ---: |
| new shared expert BF16 cache | `270,532,608` | `0.251953` | `14.01` |
| total BF16 projection cache after 07.66 | `1,704,984,576` | `1.587891` | see 07.66 ledger |

The old serving victory line remains:

```text
114.07 output tok/s
```

The fresh vLLM offline reference remains much higher, about:

```text
201.99 output tok/s on 4096/1024/batch4
```

vLLM's fast path is not precision-neutral.  It uses `deepseek_v4_fp8`, packed
`fp8_ds_mla` KV/cache pieces, FP8 indexer/cache paths, graph/runtime machinery,
and fused MoE/shared-expert runner behavior.  Keep that context in mind, but
select the next mini target from fresh mini evidence first.

## Why This Target Exists

Before TARGET 07.66, the largest remaining direct-copy group was MoE/shared
expert staging.  That group has now been reduced by `0.281843s`, and the
largest projection owners disappeared from the direct-copy table.

The remaining 07.66 direct-copy top owners are much more diffuse:

| Owner | 07.66 s | Comment |
| --- | ---: | --- |
| `dsv4.layer*.mlp.runner.experts` | `0.054546` | routed runner cost, not shared projection cache |
| `dsv4.lm_head` | `0.044360` | sampler/logits/head area |
| `dsv4.layer*.hc_ffn_pre` | `0.041932` | hidden-carrier staging |
| `dsv4.layer*.hc_attn_pre` | `0.038834` | hidden-carrier staging |
| `dsv4.layer*.mlp.runner.shared` | `0.031721` | remaining shared runner boundary |
| `dsv4.layer*.attn.kv_quant` | `0.029445` | attention/cache boundary |
| `runner_finalize_to_fp32` | `0.023179` | MoE finalization dtype boundary |
| `runner_shared_to_fp32` | `0.022688` | MoE finalization dtype boundary |

No remaining direct-copy owner alone is as strong as the eliminated
`shared_experts.gate_up_proj` owner.  Therefore the next step should be a
fresh whole-profile bottleneck reset, not automatic local polishing of the
runner finalization path.

## Scope

In scope:

- rerun promoted macro benchmarks with `dsv4_sm80_a100_victory`;
- capture a fresh 4096/128/batch4 rank0 nsys profile;
- classify CUDA kernels into high-level buckets;
- reuse the 07.65/07.66 direct-copy owner classifier;
- summarize communication by label from perf matrix reports;
- compare current mini buckets to vLLM source mechanisms when possible;
- select exactly one next implementation target with evidence and stop gates.

Out of scope:

- implementing performance optimizations;
- changing the victory bundle;
- promoting 07.64 metadata deforestation;
- runner finalization cleanup implementation;
- shared-expert overlap implementation;
- communication/NCCL implementation;
- INT8 MoE or any new precision route;
- full FP8 KV cache or `fp8_ds_mla`;
- broad cache/workspace manager implementation.

## Required Artifacts

Create:

```text
performance_milestones/target07_post_shared_expert_reprofile/
  README.md
  raw/
  summaries/
  scripts/
```

Use symlinks for large `.nsys-rep`, `.sqlite`, and `/tmp` benchmark output
directories.  Small summaries and scripts can be copied into the milestone.

## Work Plan

### 1. Freeze Baseline Context

Record:

- git status and branch;
- active milestone variant;
- current promoted env bundle;
- shared expert cache toggle state;
- 07.66 macro and direct-copy results;
- memory ledger from 07.66;
- whether 07.64 metadata deforestation is disabled.

### 2. Confirm Promoted Macro

Run the promoted variant, not only the audit variant:

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
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_shared_expert_reprofile/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

If the short macro confirms the promoted path, run the long macro:

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
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_shared_expert_reprofile/raw/macro_4096x1024_bs4_np128 \
  --keep-going
```

If the promoted variant does not match the 07.66 audit variant within normal
run-to-run noise, stop and debug bundle expansion before profiling anything
else.

### 3. Capture Fresh Nsight Profile

Capture a short TP8 4096/128/batch4 profile with graph/source NVTX and
direct-copy owner NVTX enabled.

Suggested script:

```text
performance_milestones/target07_post_shared_expert_reprofile/scripts/nsys_post_shared_expert_4096x128_bs4.sh
```

It should run:

```text
--variants dsv4_sm80_a100_victory
--prompt-len 4096
--decode-len 128
--batch-size 4
--page-size 256
--num-pages 128
--repeats 1
--warmup-repeats 0
```

Enable:

```text
MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1
MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1
```

Use the same nsys trace categories as recent target07 profiles that work in
this container.  Do not use unsupported `-t nccl` syntax if this local nsys
does not support it.

### 4. Produce Bucket-Level Kernel Summary

Write or adapt a script to summarize rank0 CUDA kernels into buckets such as:

| Bucket | Examples |
| --- | --- |
| sparse attention | `sparse_attention_kernel`, split-K reduce kernels |
| NCCL communication | `ncclDevKernel_*` |
| direct-copy/layout | `direct_copy`, `bfloat16_copy`, `float8_copy`, CatArray |
| HC/elementwise | `_hc_split_pre_kernel`, `_hc_post_kernel`, pow/mean/mul |
| MoE routed/backend | Marlin WNA16 kernels, route kernels, repack if present |
| projection/GEMM | cuBLAS/cuBLASLt/CUTLASS GEMM names |
| FP8 activation quant | `_fp8_activation_quantize_kernel` |
| index/cache/topk | index kernels, gatherTopK, indexer logits |
| RMSNorm/rope/compress/store | owner-specific DSV4 kernels |

Report:

- total kernel seconds by bucket;
- count by bucket;
- top 30 kernel names;
- decode-envelope owner mapping when NVTX allows it;
- whether a bucket is prefill-heavy, decode-heavy, or mixed.

Do not overfit to raw kernel names if existing classifier scripts already have
better owner mapping.  Prefer structured owner summaries where available.

### 5. Reuse Direct-Copy Owner Classifier

Run the direct-copy owner classifier on the new promoted profile and compare:

- 07.65 baseline;
- 07.66 audit variant;
- 07.67 promoted variant.

Confirm:

- shared expert projection owners remain absent;
- remaining MoE/shared finalization direct-copy is still around the 07.66
  order of magnitude;
- no new direct-copy owner grew unexpectedly after promotion.

### 6. Summarize Communication

Use perf matrix report fields and Nsight kernel names to summarize:

- `dsv4.v1_moe_reduce_once_all_reduce`;
- `dsv4.attn.wo_b.row_parallel_projection_all_reduce`;
- embedding all-reduce;
- lm_head all-gather;
- total communication bytes and counts.

If NCCL is now a top-two bucket, the next target should be communication
parity/overlap/reduce-contract analysis, not local direct-copy cleanup.

### 7. Compare Against vLLM Mechanisms

Do source-level comparison only for the selected top candidates.

Reference roots:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/
```

Examples:

- if communication dominates, inspect vLLM TP/EP/reduce behavior and whether
  shared/routed finalization changes all-reduce surfaces;
- if HC/elementwise dominates, inspect vLLM compile/custom-op boundaries around
  HC head, RMSNorm, and model staging;
- if sparse attention dominates, separate prefill from decode and compare only
  the relevant path;
- if remaining MoE finalization dominates, inspect vLLM fused-MoE runner
  finalization and shared expert ordering.

### 8. Select Exactly One Next Target

The README must end with one recommended next target.

Valid next target categories include:

1. **Communication parity / reduce-contract target**  
   If NCCL all-reduce/all-gather is top-two and has a plausible vLLM/mini
   contract difference.

2. **HC / elementwise graph cleanup target**  
   If HC pre/post, pow/mean/mul, RMSNorm, or related graph elementwise kernels
   form a clean top bucket with a plausible fused/compiled boundary.

3. **MoE runner finalization cleanup target**  
   If `runner_finalize_to_fp32`, `runner_shared_to_fp32`, and
   `runner_output_to_flat_dtype` remain large enough as a clean group and
   correctness-sensitive precision gates are clear.

4. **Shared expert overlap audit target**  
   If shared expert compute remains large but direct-copy staging is no longer
   the issue, and vLLM's shared-expert stream/order mechanism looks adaptable.

5. **Sparse attention / prefill-decode split target**  
   If sparse attention is selected, first prove whether the selected cost is
   decode-relevant for the 4096/1024 workload or mainly a short-profile prefill
   artifact.

6. **INT8 MoE feasibility target**  
   Only if the report explicitly chooses a precision route and defines
   accuracy/performance gates.  Do not mix it with an exact-path cleanup target.

## Gates

Measurement gate:

- promoted `dsv4_sm80_a100_victory` must reproduce the 07.66 audit-variant
  macro within normal noise, or the target must stop and explain the mismatch;
- graph replay remains active;
- eager decode remains `0`;
- the new profile must provide bucket-level kernel seconds and direct-copy
  owner attribution.

Next-target gate:

- select an implementation target only if a bucket or clean owner group has at
  least `0.15s` in the 4096/128 rank0 profile or a credible `>=3%` expected
  4096/1024 macro upside;
- if no bucket clears the bar, recommend a narrower measurement target instead
  of implementation.

Scope gate:

- no runtime optimization code lands in TARGET 07.67;
- no precision route changes;
- no bundle promotion/demotion except documenting the inherited 07.66 state.

## Stop Conditions

Stop and write the report when:

- the promoted macro is confirmed and the fresh profile selects exactly one
  next target;
- the promoted macro does not match 07.66 and bundle expansion must be debugged;
- no clean next implementation target clears the evidence gate;
- the next idea would require changing precision, communication, or runtime
  ownership before evidence is available.

Do not continue into implementation inside TARGET 07.67.

