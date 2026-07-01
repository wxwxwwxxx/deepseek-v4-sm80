# TARGET 07.40: DSV4 SM80 Post-SplitK Reprofile

## Goal

After TARGET 07.395, establish the new bottleneck ranking for the strongest
exact mini-sglang DeepSeek V4 sm80 path.

This is a profiling and decision target.  Do not implement major runtime or
kernel changes here unless they are strictly needed to make attribution
trustworthy.

## Background

TARGET 07.395 implemented exact bf16 gather/mask plus split-K sparse decode:

- sparse-only decode: `0.5768 ms -> 0.2284 ms`;
- globaltopk + indexer + sparse: `0.7890 ms -> 0.4350 ms`;
- 4096/1024/batch4: `55.05 -> 68.81 output tok/s`.

The comparable vLLM gather+split-K decode probe was about `0.2258 ms`, so mini
has effectively matched vLLM at this one decode sparse boundary while keeping
bf16 flat cache.

The remaining macro gap is still large:

- old serving victory line: `114.07 output tok/s`;
- fresh vLLM offline line: `201.99 output tok/s`;
- current exact mini line: `68.81 output tok/s`.

Therefore the next question is no longer "is sparse split-K worth it?".  It is:
what is the next true bottleneck after sparse decode is reduced?

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/summaries/target07_395_results_summary.json`
- `performance_milestones/target07_bf16_sparse_decode_splitk/summaries/nsys_splitk_4096x128_bs4_np128_rank0_summary.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_post_marlin_reprofile/README.md`

Important source paths:

- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`

## Scope

In scope:

- create `performance_milestones/target07_post_splitk_reprofile/`;
- rerun or reuse the best exact split-K macro for 4096/128 and 4096/1024;
- capture a short 4096/128/batch4 Nsight profile for the best exact variant;
- improve profile classification enough to split:
  - decode split-K gather/split/combine kernels;
  - legacy prefill/extend sparse attention;
  - indexer logits/topk/cache store;
  - runtime copy/allocation/cat/index kernels;
  - graph replay overhead;
  - MoE/Marlin and communication buckets;
- identify whether the current Nsight NVTX window is missing or too broad;
- produce a new bottleneck ranking and an explicit next-target decision.

Out of scope:

- optimizing split-K sparse decode again;
- changing default cache precision to FP8;
- implementing FP8/FP4 indexer cache;
- broad graph/runtime rewrites before attribution is clear;
- MoE/Marlin work.

## Suggested Commands

Use the current best exact variant:

```bash
BEST_VARIANT=v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Short macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants "${BEST_VARIANT}" \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0740_splitk_4096x128_bs4 \
  --keep-going
```

Long macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants "${BEST_VARIANT}" \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0740_splitk_4096x1024_bs4 \
  --keep-going
```

## Decision Rules

Select TARGET 07.41 if:

- indexer/cache/runtime/copy remains a top-two exact-path bottleneck;
- the likely fixes can preserve bf16 flat cache and exact default semantics.

Select TARGET 07.50 if:

- evidence shows the remaining gap is dominated by packed cache/indexer
  precision/layout, not exact bf16 kernel boundaries;
- vLLM's FP8 cache/indexer path is the clearest remaining source of macro
  advantage.

Select a smaller profile/tooling follow-up only if:

- the Nsight window is still too incomplete to support a decision;
- kernel categories cannot distinguish decode split-K from prefill legacy
  sparse attention.

## Validation

Required:

- no runtime behavior change unless it is observability-only;
- macro summaries for 4096/128 and 4096/1024, or a written reason for reuse;
- Nsight or equivalent profile;
- updated classified kernel/runtime report;
- `README.md` in `performance_milestones/target07_post_splitk_reprofile/`.

The README must end with:

- current best exact output tok/s;
- new top-five bottleneck ranking;
- next target selection: 07.41, 07.50, or profiling follow-up;
- do-not-continue condition.
