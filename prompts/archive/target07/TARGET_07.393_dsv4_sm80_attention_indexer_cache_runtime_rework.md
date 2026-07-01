# TARGET 07.393: DSV4 SM80 Attention/Indexer/Cache Runtime Rework

## Goal

Close the next measured DeepSeek V4 Flash performance gap after the mini-owned
Marlin WNA16 MoE backend by focusing on sparse attention, indexer/cache, and
replay-time metadata/runtime overhead on A100/sm80.

This target must start like the successful MoE path: first identify the exact
vLLM DeepSeek V4 sm80 dispatch/backend choices for attention, indexer, cache,
and graph/runtime boundaries; then decide whether mini should port vLLM code,
adapt vLLM's design, optimize the existing mini path, or defer a precision-lane
variant.

Do not begin with speculative local kernel polish.

## Current Evidence

TARGET 07.392 completed the post-Marlin reprofile:

- mini Marlin WNA16 exact, TP8, 4096/1024/batch4:
  `54.64 output tok/s`, `61.50 decode tok/s`;
- fresh vLLM offline, same workload shape:
  `201.99 output tok/s`;
- old serving victory line remains:
  `114.07 output tok/s`;
- mini 4096/128 rank0 Nsight window:
  - sparse attention: `2.110 s`, `13.28%` wall share;
  - metadata/runtime/copy visible overhead: `1.949 s`, `12.27%` wall share;
  - indexer/cache: `0.969 s`, `6.10%` wall share;
  - whole visible MoE bucket: `0.318 s`, `2.00%` wall share;
  - Marlin WNA16 expert kernel only: `0.234 s`, `1.47%` wall share.

Therefore the primary target is the attention/indexer/cache/runtime cluster.
MoE hardening is a side quest unless fresh evidence contradicts this ranking.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.392_dsv4_sm80_post_marlin_reprofile.md`
- `performance_milestones/target07_post_marlin_reprofile/README.md`
- `performance_milestones/target07_post_marlin_reprofile/summaries/post_marlin_reprofile_summary.json`
- `performance_milestones/target07_post_marlin_reprofile/summaries/bottleneck_ranking.md`
- `performance_milestones/target07_post_marlin_reprofile/summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0_classified.md`
- `performance_milestones/target07_subgraph_parity/README.md`
- `performance_milestones/target07_marlin_wna16_csrc_port/README.md`

Mini paths:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

vLLM paths:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/rope_kvcache_fusion.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/utils/multi_stream_utils.py`
- `/workspace/vllm-dsv4-docker`
- `/workspace/venvs/vllm-dsv4`

Existing run helpers:

- `performance_milestones/target07_post_marlin_reprofile/scripts/`
- `performance_milestones/vllm/scripts/`

## Precision Policy

The default mini path remains exact bf16 activation/cache policy plus MXFP4
Marlin WNA16 expert weights. Do not silently convert the default path to vLLM's
`deepseek_v4_fp8` policy.

However, vLLM's current winning path uses FP8-related cache/indexer behavior:

- engine quantization reports `deepseek_v4_fp8`;
- attention uses `fp8_ds_mla` KV cache;
- indexer uses FP8 indexer cache;
- MoE uses MXFP4/Marlin on sm80.

This target may probe those vLLM precision/cache choices and may add an
explicit experimental opt-in only if the evidence shows the remaining gap is
mainly cache layout/precision. If that becomes the dominant conclusion, write a
new precision/cache target instead of quietly changing the exact baseline.

## Scope

In scope:

- create `performance_milestones/target07_attention_indexer_cache_runtime/`;
- identify the actual vLLM dispatch/backends for DeepSeek V4 sm80 attention,
  sparse attention decode, indexer, compressor/cache insert, cache layout,
  graph capture, and stream overlap;
- identify mini's corresponding dispatch/backends and boundaries;
- build paired microbench or probe scripts for the dominant subgraphs;
- implement at most one or two tightly scoped changes after backend/parity
  evidence selects the best route;
- validate correctness and macro performance after any implementation;
- decide whether the next target is a direct vLLM port, mini rewrite, runtime
  consolidation, precision/cache experiment, or reprofile.

Out of scope:

- more Marlin WNA16 expert GEMM optimization;
- broad communication/all-reduce rewrites;
- making Marlin WNA16 default;
- broad precision-lane work such as FP8 activation, FP4 activation, or INT8
  Tensor Core MoE;
- porting vLLM's known OOM-prone sparse prefill reference path as mini default.

## Phase 1: vLLM Dispatch And Backend Identification

Produce a report similar in spirit to TARGET 07.37's MoE backend
identification. Do not rely on vague labels such as "vLLM attention is faster."
Name the exact code path and backend.

For vLLM, determine and record:

- which DeepSeek V4 attention module is instantiated:
  `DeepseekV4Attention`, `DeepseekV4MLAModules`,
  `DeepseekV4MultiHeadLatentAttentionWrapper`, or another wrapper;
- whether decode uses `torch.ops.vllm.deepseek_v4_attention`, FlashMLA,
  `flash_mla_with_kvcache`, `flash_mla_sparse_fwd`,
  `_dsv4_sm80_sparse_attn_decode_triton`, a reference gather path, or another
  custom op;
- which indexer backend is used:
  `DeepseekV4Indexer`, `DeepseekV4IndexerCache`, `SparseAttnIndexer`,
  `fused_indexer_q_rope_quant`, or another operator;
- exact cache layouts and dtypes for:
  - main MLA/SWA cache;
  - compressed C4/C128 cache;
  - indexer cache;
  - scale/metadata side buffers;
- whether vLLM is using FP8, FP4, or bf16 for each cache;
- block size, page size, head dimensions, indexer top-k width, compression
  ratio, and graph capture sizes;
- torch.compile / custom-op / CUDA graph boundaries;
- auxiliary stream use, especially attention stream, KV insert/compressor
  overlap, and indexer overlap;
- relevant source files, functions, env/config flags, and observed runtime logs.

If possible, add a small vLLM probe script under the milestone `scripts/`
directory that prints or serializes these dispatch facts using
`/workspace/venvs/vllm-dsv4`. If the probe cannot instantiate the full engine,
fall back to source inspection plus fresh run logs and record the limitation.

## Phase 2: Mini Dispatch And Boundary Map

Produce the same map for mini:

- `DSV4Attention` / attention backend call chain;
- `DSV4Indexer` and compressor path;
- `DeepSeekV4KVCache` layout policy, including `bf16_flat` and any dormant
  `flashmla_fp8_packed` surface;
- cache store paths:
  `q_kv_norm_rope_cache_fallback`, `k_norm_rope_cache_fallback`,
  `compress_norm_rope_store_fallback`, `store_indexer_fallback`,
  `store_compressed_fallback`, `copy_masked_compressed_locs`;
- sparse attention paths:
  `dsv4_sparse_attention_two_source_bf16`,
  `paged_mqa_attention_fallback`, torch fallback;
- indexer paths:
  `indexer_bf16_logits_fallback`, `indexer_select_bf16_fallback`,
  `topk_transform_512_*`, `fused_q_indexer_rope_*`;
- CUDA graph capture/replay boundary and which metadata is copied into graph
  buffers;
- small PyTorch copy/index/cat/fill kernels around attention/cache.

The output should explicitly pair each mini boundary with the closest vLLM
boundary.

## Phase 3: Paired Microbench And Profiling

Build paired probes where practical. The goal is to isolate latency and kernel
counts, not to make a full benchmark suite.

Required mini probes:

- sparse attention decode for DSV4-like shapes:
  - batch tokens `T=4`;
  - prompt/history length around `4096`;
  - page size `256`;
  - local attention heads and head dims from DeepSeek V4;
  - C4 and C128 compressed-cache cases if applicable;
- indexer logits/select/top-k path for the same rows and indexer width;
- cache store/update path for SWA, compressed C4/C128, and indexer cache;
- replay-time metadata copy/update path;
- combined attention/indexer/cache subgraph probe if the individual probes
  under-explain the Nsight bucket.

vLLM probes should be attempted when feasible:

- use vLLM custom ops directly if callable without running the full engine;
- otherwise use a short engine run with targeted NVTX/log instrumentation;
- record when child-process CUDA attribution is incomplete, as in TARGET 07.392.

All probes must record:

- shapes, dtypes, cache layout, block/page size, top-k widths;
- kernel names and counts;
- wall latency, CUDA event latency, and any graph replay status;
- numerical comparison strategy against current mini or a reference path;
- whether the candidate is exact or precision-changing.

## Phase 4: Candidate Selection

Create a candidate table with one row per possible implementation path:

- `direct_port_vllm`: narrow vLLM source/custom-op port is plausible;
- `adapt_vllm_design`: vLLM layout/boundary idea is useful but direct port is
  too invasive;
- `optimize_mini_existing`: mini kernel/boundary is close and local changes are
  lower risk;
- `precision_cache_experiment`: likely requires FP8/FP4 cache/indexer policy
  and should move to a dedicated opt-in target;
- `defer`: below threshold or too risky.

For each row, record:

- target subgraph;
- measured mini contribution and mini-vs-vLLM gap if known;
- expected E2E upside;
- correctness risk;
- implementation risk;
- source files to modify or port;
- validation plan.

Only proceed to implementation if the selected candidate has:

- at least `5%` expected E2E upside, or
- at least `20%` improvement expected in a top-two measured subgraph, or
- it removes a blocker preventing accurate comparison.

## Phase 5: Implementation Rules

If a candidate is selected, implement the smallest useful opt-in path first.

Acceptable examples:

- consolidate replay-time attention/indexer/cache metadata buffers;
- reduce or remove PyTorch copy/index/cat/fill kernels around the attention
  subgraph;
- adapt vLLM's cache/indexer boundary while preserving mini's exact default;
- introduce a mini-owned sparse attention/indexer custom-op if a paired
  microbench proves the existing kernel boundary is the bottleneck;
- add a clearly named experimental cache-layout backend if and only if the
  evidence points to cache layout/precision as the real gap.

Unacceptable examples:

- rewriting unrelated HC/RMSNorm/logits/sampling just because it is nearby;
- continuing MoE optimization without new evidence;
- landing a precision-changing path as the default exact backend;
- porting a large vLLM subsystem without a narrow dispatch/backend report.

## Canonical Macro Baseline

Use the TARGET 07.391/07.392 exact mini variant:

```bash
MARLIN_VARIANT=v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Representative macro commands:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants "${MARLIN_VARIANT}" \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target07393_marlin_4096x128_bs4_np128 \
  --keep-going
```

and:

```bash
--prompt-len 4096 --decode-len 1024 --batch-size 4
```

Use TP8 text smoke with page size 256 after any implementation that touches
attention, indexer, cache layout, or graph replay.

## Validation

At minimum:

- focused unit or smoke tests for touched wrappers;
- microbench before/after for the selected subgraph;
- TP8 text smoke, page size 256;
- 4096/128/batch4 macro;
- 4096/1024/batch4 macro if the 4096/128 result improves or touches decode
  replay broadly;
- Nsight or equivalent profile after implementation.

Correctness should include:

- finite outputs;
- no obvious text corruption;
- shape/dtype/layout assertions;
- comparison against the previous exact mini path where possible;
- explicit labeling when a probe or candidate changes precision.

## Decision Rules

- If the top candidate is attention/indexer/cache exact-boundary work, implement
  only that candidate first.
- If the top candidate is cache precision/layout, write a follow-up target
  rather than changing the default here.
- If vLLM dispatch proves a narrow custom op can be ported like Marlin WNA16,
  open a dedicated csrc-port target unless the port is very small.
- If two implementation cuts produce less than `5%` macro throughput gain and
  less than `10%` selected-subgraph improvement, stop and write a reprofile
  report.
- If sparse attention plus indexer/cache drops below the top two contributors,
  stop this target and re-rank instead of continuing local polish.

## Expected Outputs

Create:

- `performance_milestones/target07_attention_indexer_cache_runtime/README.md`
- `scripts/` with probes and runners used in this target;
- `raw/` with symlinks to large run/profile artifacts;
- `summaries/dispatch_backend_report.json`
- `summaries/dispatch_backend_report.md`
- `summaries/subgraph_microbench.json`
- `summaries/candidate_decision_table.md`
- post-change macro summaries if implementation occurs.

The README must include:

- exact vLLM dispatch/backend facts;
- exact mini dispatch/backend facts;
- mini-vs-vLLM boundary comparison;
- microbench and profile evidence;
- chosen implementation path and why;
- changes made, if any;
- correctness results;
- macro before/after;
- next target;
- `do not continue here unless...`.

## Done Criteria

This target is done when one of these is true:

- a measured exact-path attention/indexer/cache/runtime optimization improves
  macro throughput and the next bottleneck is identified;
- evidence shows the winning route is a larger vLLM custom-op/cache-layout port,
  and a narrower follow-up target is written or recommended;
- evidence shows the remaining gap is primarily precision/cache policy, and a
  dedicated opt-in precision/cache target is recommended;
- the attention/indexer/cache cluster is no longer a top-two contributor after
  fresh profiling.

Do not close with only a vague statement that "attention is slow." The final
artifact must name the concrete backend, boundary, layout, and next action.
