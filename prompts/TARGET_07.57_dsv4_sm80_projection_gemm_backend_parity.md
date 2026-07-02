# TARGET 07.57: DSV4 SM80 Projection/GEMM Backend Parity

## Goal

Prove and reduce the remaining DeepSeek V4 SM80 projection/GEMM gap against
the vLLM-based framework.

TARGET 07.55 showed that broad graph/layout cleanup is no longer the best next
lever.  TARGET 07.56 then confirmed that low-cost projection-adjacent staging
cleanup is real but too small: static scale caching removed focused
`scale.float().contiguous()` copy/cast events, yet 4096/128/batch4 improved
only from `43.0685` to `43.2194 output tok/s` (`+0.35%`).

The next major bottleneck is therefore projection/GEMM itself:

| Bucket | 07.54/07.55 decode-envelope kernel s | Note |
| --- | ---: | --- |
| projection/GEMM | `1.7968` | co-dominant with remaining graph/layout |
| `_quantized_linear_fp8_kernel` | `1.1726` | largest named projection kernel |
| graph/layout cluster | `1.8271` | no remaining concentrated graph/layout PoC |

This target must first attribute the projection/GEMM bucket to concrete
projection owners, then compare the mini backend contract against vLLM's
projection/quant/custom-op boundaries.  Quantization+GEMM fusion is an
important candidate, but only after attribution proves the selected projection
path is large enough.

## Win Condition

Primary evidence gate:

- identify one projection subpath or backend contract responsible for at least
  `0.50 s` of the 4096/128/batch4 decode-envelope projection/GEMM bucket; and
- separate intrinsic GEMM kernel time from wrapper staging, activation
  quantization, scale handling, layout/copy, and graph node count.

Primary implementation gate, if a PoC is attempted:

- reduce focused projection/GEMM kernel time by at least `15%`; or
- improve 4096/128/batch4 output throughput by at least `5%` over the active
  baseline `43.0685 output tok/s`, with graph replay preserved and eager
  decode count `0`.

Secondary validation:

- if the 4096/128 gate passes, run 4096/1024/batch4 and require at least `3%`
  output throughput gain over `87.0831 output tok/s`, or explain with profile
  evidence why the short-run gain does not carry to long decode.

Pivot condition:

- if no projection subpath reaches `0.50 s`, or if attribution shows the gap is
  dominated by many small unrelated projection pieces, stop and propose the
  next evidence target instead of writing speculative kernels.

## Current Baseline

Active promoted opt-in baseline remains the 07.54/07.55 path:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/batch4 | `87.0831` | `104.3427` | `1023` | `0` |

07.56 scale-cache successor context:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_scalecache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

- 4096/128/batch4: `43.2194 output tok/s`, only `+0.35%`;
- correct and graph-safe, but not promoted as the official baseline.

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- vLLM 4096/128/batch4: about `82.28 output tok/s`;
- vLLM 4096/1024/batch4: about `202.03 output tok/s`.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md`
- `prompts/TARGET_07.55_dsv4_sm80_remaining_graph_layout_or_projection_pivot.md`
- `prompts/TARGET_07.56_dsv4_sm80_low_cost_graph_layout_compile_preflight.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/README.md`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/README.md`
- `performance_milestones/target07_low_cost_graph_layout_compile_preflight/README.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.md`

Mini source areas:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM source areas:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## Scope

In scope:

- attribution of the projection/GEMM bucket by projection owner and backend;
- microbenchmarks using real DSV4 module weights/scales for decode-small
  `M <= 16`;
- source and probe comparison against vLLM projection/quant/custom-op
  boundaries;
- one focused opt-in implementation PoC if evidence identifies a large enough
  projection path;
- quantization+GEMM fusion when it directly targets the selected projection
  path;
- small-M FP8/FP4/BF16 projection kernel retuning if attribution proves
  intrinsic GEMM is slow;
- adapting vLLM's SM80 `wo_a` BMM/reference branch or
  `deepseek_v4_fp8_einsum` boundary if `wo_a` is dominant.

Out of scope:

- another broad graph/layout cleanup;
- compiling the full model or whole decoder layer;
- full `fp8_ds_mla` KV-cache E2E;
- standalone `quantize_and_insert_k_cache`;
- MoE/Marlin revisit unless projection attribution proves shared-expert
  projection is the dominant path and not already counted in MoE;
- sparse attention split-K polishing;
- NCCL/communication work;
- changing default exact BF16 behavior.

## Projection Owners To Attribute

The first deliverable is an owner-level table.  At minimum, attribute:

| Mini owner | Expected source boundary | vLLM comparison boundary |
| --- | --- | --- |
| `attn.q_proj` / `wq_a` and `wkv` | `DSV4Attention.forward`, fused WQA/WKV shared activation path | vLLM `fused_wqa_wkv(hidden_states)` plus compile/custom boundary |
| `attn.q_wqb` | `self.wq_b.forward(q_lora)` | vLLM lifted `wq_b(qr)` before `deepseek_v4_attention` |
| `attn.wo_a` | inverse-RoPE output projection, `wo_a_grouped_projection_fp8` or fallback | vLLM SM80 `wo_a` BMM/reference branch or `deepseek_v4_fp8_einsum` non-reference path |
| `attn.wo_b` | row-parallel output projection | vLLM `wo_b` linear path |
| indexer `wq_b` | `DSV4Indexer.prepare_fp8_query` | vLLM indexer/query projection path |
| indexer weights projection | `weights_proj.forward(x)` | vLLM indexer weights/logits path |
| FFN/shared expert projections | shared expert and any non-Marlin projection pieces | vLLM fused MoE/shared expert projection boundaries |
| `lm_head` | vocab-parallel embedding linear | vLLM output/lm-head path if relevant to decode-envelope projection bucket |

Use existing `MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1` ranges where possible.  Add
only minimal extra NVTX if current ranges cannot split the projection bucket.

## Backend Contracts To Compare

For each large owner, identify which contract mini uses:

- `_quantized_linear_fp8_kernel`;
- `quantize_fp8_activation_ref` plus `_fp8_activation_quantize_kernel`;
- `quantized_linear_ref` fallback;
- BF16 `F.linear`, cuBLAS, or CUTLASS path;
- `wo_a_grouped_projection_fp8`;
- Marlin WNA16 or other MoE/shared-expert backend;
- static scale cache on/off;
- wrapper-level reshape/contiguous/copy staging.

Compare against vLLM:

- `QuantFP8` and `_C.scaled_fp8_quant`;
- quantized `ColumnParallelLinear` / `RowParallelLinear`;
- `deepseek_v4_attention` custom op boundary;
- `fused_inv_rope_fp8_quant`;
- `deepseek_v4_fp8_einsum`;
- SM80 `wo_a` BMM/reference branch;
- compile/noop/RMS+quant fusion passes.

Do not claim vLLM per-bucket CUDA parity unless a real profile or probe
supports it.  Source-level boundary comparison is acceptable when profiler
coverage is incomplete.

## Quantization+GEMM Fusion Policy

Quantization+GEMM fusion is a high-priority candidate, but not the first line
of code.

On A100/SM80 there is no native FP8 Tensor Core path.  Therefore the realistic
fusion options are:

- fuse activation fake-quant, scale handling, and packed-weight dequant/load
  into a projection kernel that still uses BF16/FP32 math internally;
- avoid materializing intermediate BF16/FP8 activations and scale tensors
  between quant and GEMM;
- use a vLLM-style custom-op boundary so quant/layout staging is part of the
  projection backend contract;
- specialize decode-small `M <= 16` shapes instead of relying on a generic
  large-GEMM strategy;
- for `wo_a`, evaluate whether vLLM's SM80 BMM/cache strategy is better than
  mini's current grouped projection path.

Do not introduce a broad precision change in this target.  Use the same opt-in
FP8-indexer/FP8-activation path unless the selected projection owner requires a
local precision contract comparison.  Any accuracy-affecting change must keep a
quality smoke and a fallback.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_projection_gemm_backend_parity/
```

Record:

- inherited 07.54/07.55 baseline;
- 07.56 scale-cache result and why it was not promoted;
- projection/GEMM bucket and `_quantized_linear_fp8_kernel` baseline;
- vLLM reference throughput lines.

### 2. Attribute Projection/GEMM By Owner

Run or reuse a 4096/128/batch4 rank0 Nsight profile with projection-level NVTX.
If necessary, add debug-only NVTX to split:

- `attn.q_proj`;
- `attn.q_wqb`;
- `attn.wo_a`;
- `attn.wo_b`;
- indexer `wq_b`;
- indexer weights projection;
- shared expert / FFN projection;
- `lm_head`.

Produce:

| Owner | Kernel s | Runtime/copy s | Graph nodes | Top kernels | Backend contract | Keep/Pivot |
| --- | ---: | ---: | ---: | --- | --- | --- |

The target should not implement a kernel before this table exists.

### 3. Build Real-Weight Microbenchmarks

Add or reuse scripts under the milestone directory to measure representative
projection modules with real DSV4 weights/scales:

- decode-small `M = 1, 4, 8, 16`;
- active input/hidden shapes from the selected owner;
- wrapper time, intrinsic kernel time, activation quant time, scale/cache time,
  and output parity against current implementation;
- static scale cache on/off only as context, not as a promoted baseline unless
  it materially helps the selected owner.

For vLLM, prefer comparable module-level probes if the environment can import
the relevant custom ops.  If vLLM cannot run the probe, record source-level
backend contract comparison and the missing package/runtime blocker.

### 4. Select One Focused PoC

Only after attribution, choose at most one primary PoC:

- retune/replace mini `_quantized_linear_fp8_kernel` for the dominant
  decode-small owner;
- fuse quantization+GEMM for the dominant FP8 projection owner;
- adapt vLLM's `wo_a` SM80 BMM/cache strategy if `wo_a` dominates;
- adapt a vLLM custom-op boundary such as `deepseek_v4_fp8_einsum` if it is
  portable and directly maps to the dominant owner;
- route a BF16 projection owner to a better cuBLAS/CUTLASS/torch path if
  microbench proves mini's current dispatch is shape-mismatched.

Bad PoCs:

- a second static-scale/cache-only cleanup;
- a broad no-op reshape cleanup;
- a full model/layer compile;
- a MoE/Marlin rewrite;
- a full KV-cache FP8 precision migration.

### 5. Verify Correctness And Graph Semantics

Required if code changes:

- focused unit or micro test against current implementation;
- text smoke with the active or successor variant;
- graph replay remains active;
- eager decode count remains `0`;
- default exact BF16 behavior remains unchanged.

Suggested text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --page-size 256 \
  --output performance_milestones/target07_projection_gemm_backend_parity/raw/text_smoke.json
```

Use a successor variant name if a new opt-in toggle is added.

### 6. Macro And Profile Gate

Run 4096/128/batch4 single-variant macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_projection_gemm_backend_parity/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

Capture or update a focused rank0 Nsight profile and classify before/after:

- projection/GEMM total;
- selected owner;
- `_quantized_linear_fp8_kernel`;
- activation quant/fused quant;
- wrapper copy/cast/layout staging;
- graph/layout cluster, to ensure regressions do not hide elsewhere.

Run 4096/1024/batch4 only if the 4096/128 macro passes the `5%` gate or the
projection/GEMM focused profile shows a large enough reduction to plausibly
matter for long decode.

## Decision Rules

End with exactly one decision:

- `Decision: promote projection backend cut`
  if correctness passes and the implementation clears the macro/profile gate.
- `Decision: continue projection/GEMM with named second owner`
  if the first owner is proven and reduced, but another named owner remains
  larger than `0.50 s`.
- `Decision: pivot to quantization+GEMM fusion target`
  if attribution proves the bottleneck is the quant/layout/GEMM contract but
  the implementation is too large for this target.
- `Decision: pivot away from projection/GEMM`
  only if fresh attribution shows projection/GEMM is no longer top-two or no
  owner/contract is large enough to justify work.
- `Decision: blocked by vLLM probe/runtime`
  only if source comparison is insufficient and the missing vLLM package/profile
  data is explicitly required.

Do not end with "keep optimizing projection generally."  Name the exact owner
or backend contract.

## Stop Rules

Hard stops:

- no projection owner or contract reaches `0.50 s`;
- the selected PoC improves focused projection time by less than `15%` and
  4096/128 output throughput by less than `5%`;
- graph replay is lost or eager decode becomes nonzero;
- text smoke fails and one focused fix does not restore it;
- the next implementation would require a broad precision migration or full
  model/layer compile;
- two consecutive projection/GEMM PoCs miss the macro/profile gate.

## Expected Output

Create:

- `performance_milestones/target07_projection_gemm_backend_parity/README.md`
- `performance_milestones/target07_projection_gemm_backend_parity/scripts/`
- `performance_milestones/target07_projection_gemm_backend_parity/raw/`
- `performance_milestones/target07_projection_gemm_backend_parity/summaries/`

The README must include:

- inherited baseline and 07.56 context;
- owner-level projection/GEMM attribution table;
- mini-vs-vLLM backend contract comparison;
- real-weight microbench results;
- selected PoC or no-PoC rationale;
- correctness/text smoke if code changed;
- macro and profile before/after;
- final decision and exact next target recommendation.
