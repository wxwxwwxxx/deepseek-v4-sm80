# TARGET 07.55: DSV4 SM80 Remaining Graph/Layout Or Projection Pivot

## Goal

Run exactly one more evidence-first cut on the remaining repeated decode
graph/layout overhead in the current opt-in FP8-indexer + FP8 activation
quantization path, then decide whether to stop graph/layout work and pivot to
projection/GEMM backend parity against vLLM.

TARGET 07.54 proved that vLLM-style hard custom-op boundaries can remove real
mini-sglang replay overhead: the FP8 activation fake-quant chain was fused into
one Triton helper and the 4096/128/batch4 graph/layout cluster dropped by
`38.59%`.  However, 4096/128 output throughput improved only `3.38%`, while
4096/1024 improved `18.21%`.  The remaining profile now has two co-dominant
clusters:

| Bucket | 07.54 decode-envelope kernel s | Note |
| --- | ---: | --- |
| graph/runtime/copy/cat/index + elementwise graph nodes | `1.8271` | Still large after 07.54. |
| projection/GEMM | `1.7968` | Effectively tied with graph/layout. |

This target is a final graph/layout triage target.  It should not become a
broad cleanup thread.  Either find one remaining concentrated layout subgraph
with strong vLLM-boundary evidence and cut it, or declare graph/layout no
longer the best next lever and create a projection/GEMM backend parity plan.

## Win Condition

Primary win condition for continuing graph/layout:

- remove at least `10%` of the fresh 4096/128/batch4 graph/layout cluster
  (`graph_runtime_copy_cat_index + elementwise_graph_nodes`) from the 07.54
  comparable profile; or
- improve 4096/128/batch4 output throughput by at least `5%` over the 07.54
  baseline `43.0685 output tok/s` in a graph-correct single-variant run.

Secondary validation:

- if the cluster gate passes but 4096/128 output gain is below `5%`, run
  4096/1024/batch4 and require at least `3%` output throughput gain over the
  07.54 baseline `87.0831 output tok/s`, or give a profiler-backed explanation
  for why the cut does not translate to long decode.

Pivot condition:

- if no one concentrated graph/layout PoC clears the gate, stop graph/layout
  work and pivot to projection/GEMM backend parity against vLLM.

## Current Baseline

Use the 07.54 opt-in stack as the baseline:

- Marlin WNA16 MoE backend;
- global topk/lens;
- bf16 gather/mask plus split-K sparse decode;
- vLLM-aligned FP8 paged indexer cache;
- opt-in Triton FP8 activation quant helper;
- DSV4 decode CUDA graph replay;
- page size 256, `--num-pages 128`;
- TP8 on 8x A100 sm80.

Representative variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

07.54 macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/batch4 | `87.0831` | `104.3427` | `1023` | `0` |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- fresh vLLM 4096/128/batch4: about `82.28 output tok/s`;
- fresh vLLM 4096/1024/batch4: about `202.03 output tok/s`.

07.54 rank0 4096/128/batch4 decode-envelope profile:

| Bucket | Kernel s |
| --- | ---: |
| graph/runtime/copy/cat/index | `1.1875` |
| elementwise graph nodes | `0.6396` |
| graph/layout cluster | `1.8271` |
| FP8 activation quant PoC kernel | `0.0759` |
| projection/GEMM | `1.7968` |
| FP8 indexer | `0.1311` |
| sparse attention decode | `0.1179` |
| NCCL/communication | `0.3428` |
| MoE/Marlin | `0.3170` |
| sampling/logits | `0.1838` |

The remaining top kernels include direct-copy, bf16 copy, CatArrayBatchedCopy,
index/gather, `pow`, `mean`, `mul`, `_quantized_linear_fp8_kernel`,
`ampere_sgemm_*`, CUTLASS bf16 GEMMs, NCCL, and Marlin.  Do not assume the
remaining copy/elementwise kernels are all layout waste; prove the source
boundary before implementing.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.53_dsv4_sm80_post_fp8_indexer_reprofile.md`
- `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md`
- `performance_milestones/target07_post_fp8_indexer_reprofile/README.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/README.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/summaries/target07_54_graph_layout_replay_deforestation_decision_summary.json`

Mini source areas to inspect:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM comparison source areas:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_compressor.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## Scope

In scope:

- fresh source attribution for the remaining graph-visible direct-copy,
  bf16/float8 copy, index/gather, CatArray, and elementwise kernels;
- comparing each candidate boundary to vLLM's custom op, compile pass,
  persistent-buffer, or layout contract;
- implementing one narrow PoC if and only if the attribution supports a
  plausible `>=10%` graph/layout cluster cut or `>=5%` short-macro gain;
- keeping the implementation opt-in and preserving the exact BF16 default path;
- adding a focused test or microbench for any new helper;
- producing a clear projection/GEMM pivot plan if graph/layout cannot justify
  another implementation cut.

Out of scope:

- broad projection/GEMM replacement before the graph/layout gate fails;
- full `fp8_ds_mla` KV-cache E2E;
- standalone `quantize_and_insert_k_cache`;
- mini-owned FP8 indexer redesign;
- split-K sparse decode polishing;
- MoE/Marlin revisit;
- communication/NCCL work;
- changing exact BF16 default behavior;
- stacking multiple small layout cuts in this thread.

## Work Plan

### 1. Create The Milestone Record

Create:

```text
performance_milestones/target07_remaining_graph_layout_or_projection_pivot/
```

Record:

- 07.54 macro baseline;
- 07.54 decode-envelope bucket table;
- the exact graph/layout gate and pivot rule;
- the source files and vLLM files inspected.

### 2. Re-Attribute The Remaining Graph/Layout Cluster

Start from the 07.54 classified profile.  Build a table with at least these
candidate groups:

| Candidate | 07.54 kernel evidence | Mini source boundary | vLLM analogous boundary | PoC idea | Expected gain | Decision |
| --- | ---: | --- | --- | --- | ---: | --- |
| Remaining direct-copy kernels | Include top direct-copy duration and graph node count. | TBD | TBD | TBD | TBD | keep/reject |
| BF16/float8 copy kernels | Include bf16/float8 copy duration and node count. | TBD | TBD | TBD | TBD | keep/reject |
| CatArray/cat/index/gather assembly | Include CatArray/index/gather durations. | TBD | TBD | TBD | TBD | keep/reject |
| Remaining pow/mean/mul elementwise graph nodes | Include durations and whether they are still quant/projection related. | TBD | TBD | TBD | TBD | keep/reject |
| Projection-adjacent layout staging | Separate staging kernels from intrinsic GEMM kernels. | TBD | TBD | TBD | TBD | keep/reject |

Do not implement until the table identifies one candidate with concentrated
cost and a clear vLLM-inspired boundary.

Important: if the remaining graph/layout kernels are mostly thin wrappers
around projection/GEMM input/output contracts, the right answer may be to skip a
layout PoC and pivot directly to projection/GEMM backend parity.

### 3. Compare Against vLLM Before Coding

For the selected candidate, answer:

- Does vLLM avoid this boundary with a custom op?
- Does vLLM avoid it with `torch.compile` or a compile pass such as noop
  elimination?
- Does vLLM keep a different persistent buffer or cache layout?
- Does vLLM move the work inside `torch.ops.vllm.deepseek_v4_attention`,
  `QuantFP8`, `scaled_fp8_quant`, `deepseek_v4_fp8_einsum`, or another custom
  op boundary?
- Is the vLLM source behavior portable to mini without importing a large
  unrelated subsystem?

Source-level evidence is acceptable if the available vLLM profile lacks
complete child-process per-bucket CUDA timing.  Do not claim measured vLLM
per-bucket parity unless there is profiler evidence.

### 4. Pick Exactly One Path

Choose one of these outcomes:

1. Implement one focused graph/layout PoC.
2. Skip implementation and pivot to projection/GEMM because no graph/layout
   candidate is concentrated enough.

Good graph/layout PoCs:

- fuse a repeated projection-adjacent copy/cast/scale/layout staging chain;
- replace a repeated cat/index/gather assembly with a persistent buffer or
  narrow fused helper;
- remove a dtype round-trip that vLLM avoids with a custom-op boundary;
- move a static layout transform out of replay capture if graph semantics stay
  correct.

Bad PoCs:

- a prefill-only sparse attention change;
- a change that only improves eager mode;
- a candidate below `5%` of the graph/layout cluster;
- broad refactors of attention, projection, MoE, or communication;
- precision changes not directly tied to the selected boundary.

### 5. Verify Correctness And Graph Semantics

Required checks for any implementation:

- relevant unit/micro test for any new helper;
- text smoke with the 07.54 variant or its direct successor;
- graph replay remains active;
- eager decode count remains `0`;
- no change to default exact BF16 behavior.

Suggested text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --page-size 256 \
  --output performance_milestones/target07_remaining_graph_layout_or_projection_pivot/raw/text_smoke.json
```

Use the actual successor variant name if the implementation adds a new toggle.

### 6. Reprofile And Gate

Run a single-variant 4096/128/batch4 macro:

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
  --output-dir performance_milestones/target07_remaining_graph_layout_or_projection_pivot/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

Capture a focused rank0 Nsight profile for 4096/128 and classify it with the
07.54 taxonomy or a direct extension of it.

Run 4096/1024/batch4 only if:

- 4096/128 output throughput improves by at least `5%`; or
- the graph/layout cluster shrinks by at least `10%`; or
- the result is a pivot decision and a long-decode sanity line is needed to
  avoid choosing based on a short-run artifact.

### 7. If Pivoting, Produce The Projection/GEMM Backend Parity Plan

If graph/layout does not pass the gate, write a concrete next target plan in
the README.  The plan should compare mini and vLLM for:

- mini `_quantized_linear_fp8_kernel`;
- mini CUTLASS BF16 projection GEMMs;
- mini projection wrappers in `DSV4Linear.forward` and attention WQA/WKV/QWQB,
  KV, WO paths;
- vLLM `QuantFP8`;
- vLLM `_C.scaled_fp8_quant` / dynamic scaled FP8 quant;
- vLLM `deepseek_v4_fp8_einsum`;
- vLLM projection and attention custom-op boundaries;
- whether vLLM has a packed layout or compile-boundary advantage that mini
  should adapt instead of tuning its current projection wrappers.

The pivot plan must include a microbench/profiler gate, not just source
reading.  The goal is to prove whether the remaining gap is intrinsic GEMM
backend speed, projection-adjacent staging, graph node count, or precision
layout mismatch.

## Decision Rules

End with exactly one decision:

- `Decision: promote second graph/layout cut`
  if the PoC clears the gate and produces a meaningful macro or long-decode
  gain.
- `Decision: pivot to projection/GEMM backend parity`
  if the PoC misses the gate, graph/layout is no longer clearly above
  projection/GEMM, or the next candidate is projection-adjacent GEMM work.
- `Decision: blocked by missing profiler evidence`
  only if the existing mini/vLLM reports cannot attribute the remaining
  boundary and the thread needs a specific new profile to proceed.

Do not end with "continue graph/layout generally."  If graph/layout remains
interesting after this target, the README must name the exact next subgraph and
show why it is still a better lever than projection/GEMM.

## Stop Rules

Hard stops:

- text smoke fails and one focused fix does not restore it;
- CUDA graph replay is lost or eager decode count becomes nonzero;
- the selected graph/layout candidate improves the target subgraph by less
  than `10%` and improves 4096/128 output throughput by less than `5%`;
- a fresh profile shows projection/GEMM is equal to or larger than the
  remaining graph/layout cluster and no single graph/layout candidate is
  clearly larger;
- the next proposed implementation is really a projection/GEMM backend change.

## Expected Output

Create:

- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/README.md`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/scripts/`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/raw/`
- `performance_milestones/target07_remaining_graph_layout_or_projection_pivot/summaries/`

The README must include:

- 07.54 baseline summary;
- source-attribution table for remaining graph/layout candidates;
- vLLM boundary comparison;
- selected PoC or explicit no-PoC pivot;
- implementation summary if code changed;
- correctness/text smoke result if code changed;
- macro before/after table;
- profile bucket before/after table;
- final decision;
- concrete next target recommendation.
