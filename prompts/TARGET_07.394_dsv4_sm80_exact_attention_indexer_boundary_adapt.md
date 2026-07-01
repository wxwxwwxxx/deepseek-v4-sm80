# TARGET 07.394: DSV4 SM80 Exact Attention/Indexer Boundary Adapt

## Goal

Implement the smallest exact bf16 attention/indexer/cache boundary adaptation
justified by TARGET 07.393.

Do not change mini's default precision policy.  The default remains bf16
activation/cache plus MXFP4 Marlin WNA16 expert weights.  vLLM's packed
`fp8_ds_mla` KV cache and FP8 indexer cache belong to a separate opt-in
precision/cache target.

## Required Inputs

Read first:

- `performance_milestones/target07_attention_indexer_cache_runtime/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.json`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/subgraph_microbench.json`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/candidate_decision_table.md`
- `performance_milestones/target07_post_marlin_reprofile/summaries/post_marlin_reprofile_summary.json`
- `performance_milestones/target07_subgraph_parity/README.md`

Relevant source paths:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu`
- `python/minisgl/engine/graph.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

## Evidence From 07.393

T=4, history=4096, page size=256, A100/sm80:

- mini bf16 sparse attention direct cache read: `0.578 ms`;
- vLLM packed-cache gather/dequant plus split-K decode: `0.226 ms`;
- mini bf16 indexer select: `0.339 ms`;
- vLLM indexer Q RoPE+FP8 quant only: `0.087 ms` but full indexer logits/topk is engine-bound;
- mini compressed/indexer cache store probe: `0.976 ms`;
- vLLM `compute_global_topk_indices_and_lens`: `0.0566 ms`;
- mini profile shares: sparse attention `13.28%`, metadata/runtime/copy `12.27%`, indexer/cache `6.10%`.

The selected route is `adapt_vllm_design`, not `direct_port_vllm`.

## Scope

Implement one exact-path cut first.  Acceptable cuts:

1. Global topk/lens consolidation:
   - adapt vLLM's `compute_global_topk_indices_and_lens` idea for mini C4 topk;
   - keep int32 outputs and bf16 cache policy;
   - reduce page-table/topk metadata work around graph replay.

2. Exact bf16 two-scope gather/mask plus split-K sparse decode:
   - preserve mini bf16 SWA/C4/C128 cache layout;
   - either gather bf16 cache rows into a merged buffer then run a split-K
     decode core, or build a split-K kernel that reads mini bf16 caches
     directly;
   - compare against current `dsv4_sparse_attention_two_source_bf16`.

3. Graph-stable attention/indexer metadata buffers:
   - only after the topk/sparse boundary is chosen;
   - reduce replay-time copy/fill/index kernels without changing outputs.

Out of scope:

- packed `fp8_ds_mla` cache as default;
- FP8 or FP4 indexer cache as default;
- porting vLLM sparse prefill reference path;
- MoE/Marlin work;
- broad graph/runtime rewrites not tied to attention/indexer/cache evidence.

## Implementation Rules

- Start with one cut only.  Prefer the cut with the cleanest exact bf16
  correctness story and a paired microbench target.
- Keep a feature flag or clearly isolated wrapper if the implementation changes
  kernel dispatch.
- Maintain exact bf16 cache dtype/layout assertions.
- Preserve graph capture safety.  If a fallback is not graph-safe, fail loudly
  under graph capture rather than silently switching to eager torch ops.
- Do not remove the old path until the new path passes macro and text smoke.

## Validation

Required after any runtime/kernel change:

- focused unit/smoke test for the changed wrapper;
- before/after microbench using 07.393 scripts;
- TP8 text smoke, page size 256;
- 4096/128/batch4 macro on the Marlin WNA16 exact variant;
- 4096/1024/batch4 macro if 4096/128 improves or decode replay is touched
  broadly;
- Nsight or equivalent profile if macro improves.

Success threshold:

- at least `20%` improvement in the selected top-two subgraph, or
- at least `5%` macro throughput improvement.

Stop and write a precision/cache target instead if exact bf16 adaptation cannot
close the measured sparse/indexer/cache gap without adopting packed FP8 cache.

## Expected Outputs

Create a new milestone directory:

- `performance_milestones/target07_exact_attention_indexer_boundary_adapt/README.md`
- `scripts/` for before/after probes;
- `raw/` for raw microbench/profile artifacts;
- `summaries/` with before/after microbench, macro, and profile summaries;
- update the TARGET 07 gap closure notes with the result and next bottleneck.
