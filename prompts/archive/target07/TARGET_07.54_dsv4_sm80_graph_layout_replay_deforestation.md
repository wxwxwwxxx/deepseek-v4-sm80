# TARGET 07.54: DSV4 SM80 Graph/Layout Replay Deforestation

## Goal

Reduce repeated decode CUDA graph layout overhead in the current opt-in
FP8-indexer DeepSeek V4 path.

TARGET 07.53 showed that after the vLLM-aligned FP8 indexer port, the largest
remaining mini-sglang gap is no longer indexer logits, sparse decode, MoE, or
KV-cache store.  The repeated decode graph body is dominated by:

| Bucket | Decode-envelope kernel s | Decode-envelope wall share |
| --- | ---: | ---: |
| projection/GEMM | `1.7973` | `27.49%` |
| graph/runtime/copy/cat/index | `1.6170` | `24.73%` |
| elementwise graph nodes | `1.3583` | `20.77%` |

The graph/layout cluster is:

```text
graph_runtime_copy_cat_index + elementwise_graph_nodes
= 2.9752 s
= 45.50% of the 4096/128/batch4 decode-envelope wall
```

The goal of this target is to identify and remove or fuse one high-impact
repeated graph-layout subgraph.  This is not a broad "optimize everything"
target.  It should produce one focused PoC and a clear keep/pivot decision.

## Win Condition

Primary win condition:

- remove at least `10%` of
  `graph_runtime_copy_cat_index + elementwise_graph_nodes` decode-envelope
  kernel time in the 4096/128/batch4 FP8-indexer profile; or
- improve 4096/128/batch4 output throughput by at least `5%` in a graph-correct
  single-variant run.

Secondary win condition:

- if 4096/128 passes, run 4096/1024/batch4 and show at least `3%` output
  throughput gain, or explain why the short-run gain does not carry to the
  long decode workload.

Pivot condition:

- if one focused PoC cannot hit the primary win condition, stop graph/layout
  work and recommend a projection/GEMM backend parity target.

## Current Baseline

Current best exact result:

- 4096/128/batch4: `38.94 output tok/s`;
- 4096/1024/batch4: `68.81 output tok/s`.

Current best opt-in FP8-indexer result:

- 4096/128/batch4: `41.66 output tok/s` from TARGET 07.53;
- 4096/1024/batch4: `73.67 output tok/s` from TARGET 07.52;
- graph replay is active and eager decode count is `0`;
- exact BF16 default remains unchanged.

vLLM reference:

- 4096/128/batch4: about `82.28 output tok/s`;
- 4096/1024/batch4: about `202.03 output tok/s`;
- vLLM uses compile/CUDA graph sizes `[1,2,4]`, custom-op boundaries around
  DeepSeek V4 attention, and the `deepseek_v4_fp8` precision/layout lane.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.52_dsv4_sm80_vllm_fp8_indexer_backend_port.md`
- `prompts/TARGET_07.53_dsv4_sm80_post_fp8_indexer_reprofile.md`
- `performance_milestones/target07_vllm_fp8_indexer_backend_port/README.md`
- `performance_milestones/target07_post_fp8_indexer_reprofile/README.md`
- `performance_milestones/target07_post_fp8_indexer_reprofile/summaries/nsys_fp8_indexer_node_4096x128_bs4_np128_rank0_classified.md`
- `performance_milestones/target07_post_fp8_indexer_reprofile/summaries/target07_53_post_fp8_indexer_reprofile_decision_summary.json`

Mini source areas to inspect:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

vLLM comparison source areas:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_compressor.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`
- `/workspace/vllm-dsv4-docker/vllm/compilation/`

## Scope

In scope:

- attribution/instrumentation around repeated decode replay layout work;
- removing or fusing one high-impact repeated graph-layout subgraph;
- replacing a chain of PyTorch copy/cat/index/elementwise ops with an existing
  local fused helper or a narrow mini-owned Triton helper;
- changing internal opt-in FP8-indexer variant implementation if exact default
  behavior remains unchanged;
- comparing the chosen boundary against vLLM's source-level custom-op or
  compile boundary.

Out of scope:

- broad projection/GEMM replacement before the graph-layout PoC gate;
- full `fp8_ds_mla` KV-cache E2E;
- standalone `quantize_and_insert_k_cache`;
- old mini-owned FP8 indexer polishing;
- split-K sparse decode polishing;
- MoE/Marlin revisit;
- communication/NCCL work;
- changing exact BF16 default behavior.

## Work Plan

### 1. Preserve The 07.53 Baseline

Create:

```text
performance_milestones/target07_graph_layout_replay_deforestation/
```

Record in the new README:

- 07.53 macro lines;
- 07.53 decode-envelope bucket table;
- the graph/layout cluster total `2.9752 s`;
- the target's primary and pivot gates.

### 2. Attribute The Graph/Layout Cluster To Source Boundaries

Before changing kernels, map the largest repeated graph nodes back to Python or
wrapper boundaries.

Start from the 07.53 top graph-layout kernels:

- direct-copy kernels, especially the large direct-copy bucket;
- bf16 copy kernels;
- float8 copy kernels;
- index/gather kernels;
- `CatArrayBatchedCopy`;
- clamp/log2/reduce/pow elementwise helpers.

Add minimal NVTX or counter instrumentation if needed.  Prefer instrumentation
that can be left behind as a debug-only profiling aid.

The attribution table must include:

| Candidate | Kernel evidence | Suspected source boundary | vLLM analogous boundary | PoC idea | Expected gain |
| --- | ---: | --- | --- | --- | ---: |

Do not implement until at least one candidate has a plausible `>=10%` cut of
the graph-layout cluster or `>=5%` short-macro upside.

### 3. Pick One PoC

Select exactly one primary PoC for this target.  Good candidates are:

- fuse a repeated copy/cast/scale chain around FP8 projection or indexer
  staging;
- replace repeated cat/index assembly with preallocated contiguous buffers or
  a fused gather/scatter helper;
- remove an unnecessary dtype round-trip that manifests as bf16/float8 copy
  kernels;
- fuse clamp/log2/pow/reduce staging if it is repeatedly recomputing a stable
  scale or mask inside decode replay;
- move a static layout transform out of replay capture if it is safe and does
  not break graph semantics.

Bad candidates:

- a one-off prefill-only sparse attention change;
- small copy kernels whose total contribution is below `5%` of the
  graph-layout cluster;
- anything that only improves eager mode;
- a change that requires changing default exact BF16 behavior;
- a projection/GEMM backend rewrite.  That is the pivot target if this one
  fails.

### 4. Implement The Narrow Cut

Keep the implementation small and reversible.

Requirements:

- opt-in FP8-indexer path must still pass text smoke;
- CUDA graph replay must remain active;
- eager decode count must remain `0`;
- the old exact BF16 default path must not regress;
- any new Triton helper must have a focused unit/micro test or comparison
  against the previous implementation.

Suggested smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --page-size 256 \
  --output /tmp/dsv4_target0754_text_smoke.json
```

Use the actual variant name if it changes.

### 5. Reprofile And Gate

Rerun the same 4096/128/batch4 FP8-indexer single-variant macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0754_graph_layout_4096x128_bs4_np128 \
  --keep-going
```

If the macro moves, capture a focused rank0 Nsight profile and classify with
the 07.53 bucket taxonomy.  If macro does not move but the targeted subgraph is
claimed to improve, the Nsight classification is still required.

Run 4096/1024/batch4 only if:

- 4096/128 output tok/s improves by at least `5%`; or
- the targeted graph-layout cluster shrinks by at least `10%`.

### 6. vLLM Comparison

For the chosen PoC boundary, compare against vLLM source behavior.

The README must answer:

- Does vLLM avoid this copy/cat/index/elementwise boundary with a custom op,
  compile boundary, persistent buffer, or different cache/layout contract?
- Is mini now closer to vLLM after the PoC?
- If not, is the remaining difference intrinsic projection/GEMM rather than
  layout staging?

Do not claim vLLM per-bucket parity unless a fresh or existing vLLM profile
actually supports it.  Source-level boundary comparison is acceptable when the
existing vLLM profile lacks precise child-process CUDA kernels.

## Decision Rules

End with exactly one decision:

- `Decision: continue graph/layout deforestation with a second cut`
  if the PoC hits the primary win condition and graph/layout remains top-two.
- `Decision: promote graph/layout cut and run 4096/1024 validation`
  if the short macro passes but long macro was not yet run.
- `Decision: pivot to projection/GEMM backend parity`
  if the PoC misses the primary win condition or projection/GEMM remains the
  dominant top bucket after the cut.
- `Decision: blocked by attribution`
  only if source mapping cannot identify any candidate large enough to justify
  implementation; include the missing profiler data needed.

## Stop Rules

Stop after one focused PoC.  Do not keep stacking tiny graph-layout cuts in the
same thread.

Hard stops:

- text smoke fails and one focused fix does not restore it;
- CUDA graph replay is lost or eager decode count becomes nonzero;
- targeted cluster improvement is below `10%` and 4096/128 throughput gain is
  below `5%`;
- a fresh profile shows graph/layout is no longer top-two;
- the next proposed change is really projection/GEMM, MoE, communication, or
  packed KV-cache work.

## Expected Output

Create:

- `performance_milestones/target07_graph_layout_replay_deforestation/README.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/scripts/`
- `performance_milestones/target07_graph_layout_replay_deforestation/raw/`
- `performance_milestones/target07_graph_layout_replay_deforestation/summaries/`

The README must include:

- 07.53 baseline summary;
- attribution table;
- selected PoC and rejected candidates;
- implementation summary;
- correctness/text smoke result;
- macro before/after table;
- profile bucket before/after table;
- vLLM source-boundary comparison;
- final decision and do-not-continue condition.

