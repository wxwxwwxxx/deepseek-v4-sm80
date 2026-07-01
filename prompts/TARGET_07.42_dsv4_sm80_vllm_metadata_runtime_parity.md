# TARGET 07.42: DSV4 SM80 vLLM Metadata/Runtime Parity Evidence

## Goal

Build a hard evidence chain for the remaining mini-sglang vs vLLM performance
gap after Marlin WNA16 MoE and bf16 split-K sparse decode.

This is an evidence-first target.  Do not optimize random mini-side details.
The main output is a mini-vs-vLLM parity map that explains which engine-level
mechanisms account for the remaining gap and which mechanism should be ported,
adapted, rejected, or deferred.

The target may implement one small proof-of-concept only if the evidence points
to a vLLM core mechanism with a plausible end-to-end gain of at least `5%`.

## Required Input

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`
- `prompts/TARGET_07.41_dsv4_sm80_indexer_cache_runtime_exact.md`
- `performance_milestones/target07_post_splitk_reprofile/README.md`
- `performance_milestones/target07_indexer_cache_runtime_exact/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`

Useful prior facts:

- current best exact 4096/1024/bs4: `68.8097 output tok/s`;
- old serving victory line: `114.07 output tok/s`;
- fresh vLLM offline 4096/1024/bs4: about `201.99 output tok/s`;
- mini sparse decode boundary is already close to the vLLM probe
  (`0.2284 ms` vs `0.2258 ms`);
- TARGET 07.41 metacopy improved a real micro-subgraph but did not improve
  macro throughput, so further local metadata-copy polish is not justified
  without new evidence.

## Reference Code

mini-sglang:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `python/minisgl/engine/model_runner.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`
- `/workspace/vllm-dsv4-docker/vllm/v1/cudagraph_dispatcher.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/utils/multi_stream_utils.py`

## Evidence Questions

Answer these before implementing any optimization:

1. Which mini 07.40 buckets still dominate after the 07.41 negative result?
2. For each bucket, what exact mini source path and kernel names create it?
3. What is the vLLM analogous mechanism or custom-op boundary?
4. Is vLLM faster because of:
   - engine/runtime buffer ownership;
   - CUDA graph capture/replay discipline;
   - fused custom-op boundaries;
   - aux-stream overlap;
   - packed FP8 KV/indexer layout;
   - different algorithmic work;
   - or missing/incomparable evidence?
5. Can the mechanism be adapted while keeping mini's current exact bf16
   default, or does it belong to a precision/cache target such as TARGET 07.50?

## Work Plan

### 1. Rebuild the mini-side hotspot map

Use TARGET 07.40 and TARGET 07.41 artifacts as the baseline.

Map every top bucket to exact kernel names and source owners:

- runtime/copy/cat/index graph nodes;
- elementwise graph nodes;
- legacy prefill/extend sparse attention;
- indexer logits/topk/cache;
- FP8 projection GEMM;
- dense GEMM, NCCL, MoE only as sanity checks.

Do not treat broad buckets as actionable until their source-level owners are
identified.

### 2. Build the vLLM mechanism map

For each mini bucket, identify the closest vLLM mechanism:

- attention custom op boundary;
- `SparseAttnIndexer`;
- packed `fp8_ds_mla` cache insert/gather/dequant;
- FP8 indexer cache and paged logits;
- `CudagraphDispatcher` persistent buffers;
- graph capture stream and replay buffer registration;
- async sampled-token/output copies;
- `AuxStreamType.Attention` overlap;
- compiled/fused regions around DeepSeek V4 forward.

Record whether the vLLM mechanism is exact-portable, bf16-adaptable,
precision-changing, or not relevant to the measured mini bucket.

### 3. Improve vLLM timing evidence if feasible

If feasible in the current environment, capture a short vLLM profile that can
attribute child-process CUDA graph nodes.  Prefer a short workload with the
same structure as mini's 4096/128/bs4 profiling target.

Try to collect:

- CUDA graph node attribution, for example with `NSYS_CUDA_GRAPH_TRACE=node`;
- child process activity from vLLM workers;
- NVTX ranges or another reliable repeat window;
- enough kernel names to classify attention/indexer/cache/runtime work.

If a reliable profile cannot be captured quickly, stop profiling attempts and
write down exactly why.  The target may still succeed through code topology plus
mini-side measured evidence, but the comparability limit must be explicit.

### 4. Produce a parity table

The README must contain this table:

| Mini bucket | Mini source/kernels | Mini measured cost | vLLM mechanism | vLLM measured cost | Precision/layout dependency | Decision |
| --- | --- | ---: | --- | ---: | --- | --- |

Decision must be one of:

- `port`: direct vLLM mechanism can be introduced with acceptable scope;
- `adapt-bf16`: use vLLM's mechanism but keep mini's exact bf16 default;
- `precision-target`: likely important, but requires packed FP8/FP4 cache or
  activation/indexer quantization;
- `defer`: not top-two, not proven, or not actionable;
- `reject`: mismatched, unsafe, OOM-prone, or lower-value than the alternatives.

### 5. Optional proof-of-concept

Implement at most one proof-of-concept only if all are true:

- the mechanism comes from a vLLM core design, not a mini-local microcut;
- the parity table shows it targets a top-two remaining mini bucket;
- expected E2E gain is at least `5%`;
- correctness boundaries are clear;
- it does not silently change the default exact precision policy.

If implemented, gate it behind an opt-in environment variable or benchmark
variant and run focused correctness plus the normal 4096/128 and 4096/1024
macro checks.

## Out Of Scope

- more replay metadata copy polish from TARGET 07.41;
- more split-K sparse decode polish;
- MoE/Marlin changes;
- broad precision changes as the default path;
- packed FP8 KV/indexer cache as default;
- speculative stream overlap without a mini-vLLM mechanism map;
- spending the whole thread trying to force vLLM profiling if the environment
  blocks it.

## Success Criteria

This target succeeds if it produces a credible evidence report, even with no
code change.

Required:

- source-level ownership for the top mini buckets;
- vLLM mechanism map for the same buckets;
- explicit statement of which timings are directly comparable and which are
  only code-topology evidence;
- one recommended next target with expected gain and risk;
- do-not-continue guidance for non-bottleneck work.

Optional:

- a reliable vLLM child-process CUDA graph node profile;
- one opt-in proof-of-concept for a proven vLLM core mechanism.

## Expected Output

Create:

- `performance_milestones/target07_vllm_metadata_runtime_parity/README.md`
- `scripts/`, `raw/`, and `summaries/` under that directory as needed.

The README must end with:

- current best exact result;
- strongest proven gap source;
- strongest unproven suspicion;
- next target recommendation;
- whether TARGET 07.50 should start or remain deferred.

## Stop Rules

Stop the thread when one of these happens:

- the parity table identifies a next implementation target with at least `5%`
  expected E2E gain;
- evidence shows the next required step is a precision/cache experiment, so
  TARGET 07.50 should begin;
- vLLM profiling is blocked and code-topology plus mini-side evidence is enough
  to choose the next target;
- two attempted proof-of-concept cuts fail to reach `5%` macro gain;
- the next proposed work is a mini-local micro-optimization that does not map
  to a vLLM core mechanism.
