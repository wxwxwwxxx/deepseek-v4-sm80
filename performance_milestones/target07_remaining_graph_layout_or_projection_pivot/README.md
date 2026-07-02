# TARGET 07.55: Remaining Graph/Layout Or Projection Pivot

Date: 2026-07-02

## Result

This target completed the final graph/layout triage pass without changing
runtime code.  The 07.54 profile was re-attributed at source level, compared
against vLLM's custom-op and compile boundaries, and no single remaining
graph/layout PoC met both requirements:

- concentrated enough to plausibly cut at least `0.1827 s` from the fresh
  `1.8271 s` graph/layout cluster; and
- attributable to a narrow mini boundary that maps cleanly to a portable vLLM
  boundary.

The largest remaining kernel bucket is now projection/GEMM at `1.7968 s`,
effectively tied with the full graph/layout cluster.  The next best work is
therefore projection/GEMM backend parity, not another general layout cleanup.

No correctness or text-smoke run was required because no model, kernel,
benchmark variant, or default behavior changed.  The 07.54 graph-correct lines
remain the active baseline.

## Required Inputs Inspected

Prompt and milestone inputs:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.53_dsv4_sm80_post_fp8_indexer_reprofile.md`
- `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md`
- `performance_milestones/target07_post_fp8_indexer_reprofile/README.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/README.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.md`
- `performance_milestones/target07_graph_layout_replay_deforestation/summaries/target07_54_graph_layout_replay_deforestation_decision_summary.json`

Mini source areas inspected:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM source areas inspected:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_compressor.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## 07.54 Baseline

Representative active opt-in variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/bs4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/bs4 | `87.0831` | `104.3427` | `1023` | `0` |

Decode-envelope profile baseline:

| Bucket | Kernel s | Decode wall share | Count | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| graph/runtime/copy/cat/index | `1.1875` | `24.51%` | `288963` | `2217` |
| elementwise graph nodes | `0.6396` | `13.20%` | `207131` | `1594` |
| graph/layout cluster | `1.8271` | `37.71%` | n/a | `3811` |
| FP8 activation quant PoC | `0.0759` | `1.57%` | `35433` | `279` |
| projection/GEMM | `1.7968` | `37.09%` | `100965` | `795` |
| FP8 indexer | `0.1311` | `2.71%` | `20828` | `164` |
| sparse attention decode | `0.1179` | `2.43%` | `21590` | `170` |
| NCCL/communication | `0.3428` | `7.08%` | `11176` | `88` |
| MoE/Marlin | `0.3170` | `6.54%` | `43688` | `344` |
| sampling/logits | `0.1838` | `3.79%` | `43815` | `345` |

Gate values:

| Gate | Required value |
| --- | ---: |
| 10% graph/layout cut | `0.1827 s` from the 07.54 cluster |
| 5% 4096/128 macro gain | `45.2219 output tok/s` |
| 3% 4096/1024 secondary gain | `89.6956 output tok/s` |

## 07.55 Attribution Artifacts

Generated summaries:

- `summaries/remaining_graph_layout_candidate_summary.json`
- `summaries/remaining_graph_layout_candidate_summary.md`

Reproduction command:

```bash
python performance_milestones/target07_remaining_graph_layout_or_projection_pivot/scripts/summarize_remaining_graph_layout.py \
  --json-out performance_milestones/target07_remaining_graph_layout_or_projection_pivot/summaries/remaining_graph_layout_candidate_summary.json \
  --md-out performance_milestones/target07_remaining_graph_layout_or_projection_pivot/summaries/remaining_graph_layout_candidate_summary.md
```

The candidate-group slices below are not additive; they are kernel-name based
evidence cuts used to decide whether a single PoC is justified.

| Candidate | 07.54 kernel evidence | Mini source boundary | vLLM analogous boundary | PoC idea | Expected gain | Decision |
| --- | ---: | --- | --- | --- | ---: | --- |
| Remaining direct-copy kernels | `0.9456 s`, `191622` launches, `1487` graph nodes; largest direct-copy kernel `0.5804 s`. | Mixed across graph replay input copy in `GraphCaptureBuffer.copy_from`, attention replay metadata in `_copy_metadata_for_replay`, projection reshape/contiguous in `_flatten_linear_input`, and tensor views/copies around `DSV4Linear.forward`. | vLLM reduces some copies with `torch.compile`, noop elimination, `torch.ops.vllm.deepseek_v4_attention`, and quant/projection custom op boundaries. | A broad direct-copy eliminator would need multiple owners. | Unknown; kernel-name aggregate is large, but no single source boundary owns it. | Reject as too diffuse. |
| BF16/float8 copy kernels | BF16 copy `0.1318 s`, `40894` launches, `322` graph nodes; float8 copy no longer visible in this decode envelope after 07.54. | Remaining casts near FP8 projection inputs/outputs, BF16 projection wrappers, and decode staging. | vLLM hides quant casts behind `QuantFP8` and `_C.scaled_fp8_quant`; projection output layout is tied to its linear/attention custom op path. | Fuse a copy/cast around one projection. | Below `10%` graph/layout gate by itself. | Reject. |
| CatArray/cat/index/gather assembly | Combined kernel-name slice `0.1830 s`, `65789` launches, `488` graph nodes, only barely above the `0.1827 s` gate and includes top-k/indexer kernels outside pure layout. | `_merge_indexer_rows`, `_merge_indexer_lengths`, `_make_sparse_compressed_indices`, `_gather_full_locs`, `_copy_metadata_for_replay`, and top-k row merge paths in `DSV4AttentionBackend`. | vLLM uses `SparseAttnIndexer`, persistent top-k, `topk_indices_buffer`, and attention/cache utility Triton helpers. 07.43 already showed persistent-topk ablation was not a macro driver. | Fused index/cat row assembly or persistent sparse-index buffers. | Barely gate-sized only when multiple subpaths are stacked. | Reject for this target; keep only as profiler context. |
| Remaining pow/mean/mul elementwise graph nodes | `0.5148 s`, `143373` launches, `1119` graph nodes after excluding GEMM split-K reductions; top pieces are generic mul, mean, rsqrt, pow, clamp, and reductions. | Distributed across RMSNorm-like math, HC/sampling/logits math, MoE route helpers, attention projection staging, and remaining quant/projection-adjacent math. | vLLM has fusion passes for RMS/quant/all-reduce patterns and custom quant ops, but no single remaining mini source boundary maps cleanly after the 07.54 FP8 activation quant cut. | Fuse another math chain. | Potentially large, but not attributable to one stable subgraph from current evidence. | Reject as under-attributed. |
| Projection-adjacent layout staging | Projection/GEMM intrinsic bucket is `1.7968 s`; `_quantized_linear_fp8_kernel` alone is `1.1726 s`. Remaining copy/elementwise kernels are often adjacent to these wrappers. | `DSV4Linear.forward`, `quantized_linear_ref`, `_quantized_linear_fp8_kernel`, attention WQA/WKV/QWQB/KV/WO paths, and BF16 CUTLASS/sgemm projections. | vLLM uses `QuantFP8`, `_C.scaled_fp8_quant`, `ColumnParallelLinear`/`RowParallelLinear` quant paths, `deepseek_v4_fp8_einsum`, `fused_inv_rope_fp8_quant`, and `torch.ops.vllm.deepseek_v4_attention`. | Compare and port/tune projection backend boundaries. | High, but it is projection/GEMM work, not graph/layout cleanup. | Keep as next target. |

## vLLM Boundary Comparison

vLLM does avoid several mini-visible boundaries, but the remaining portable
unit is no longer a small graph/layout helper:

- Quantization is a custom-op boundary through `QuantFP8` and
  `ops.scaled_fp8_quant`.  TARGET 07.54 already copied this strategy for the
  large FP8 activation quant chain.
- Attention uses `torch.ops.vllm.deepseek_v4_attention`, with q/kv RMSNorm and
  `wq_b` lifted to let Inductor fuse wrapper-level math while the attention
  body owns cache/indexer/compressor work.
- O projection uses `fused_inv_rope_fp8_quant` plus
  `torch.ops.vllm.deepseek_v4_fp8_einsum` on the non-reference path; SM80 also
  has a `wo_a` BMM/cache strategy in the reference branch.
- Sparse indexer work is behind `SparseAttnIndexer`, persistent top-k, and
  persistent `topk_indices_buffer`, but mini's FP8 indexer is already a small
  `0.1311 s` bucket and persistent top-k was not a vLLM macro factor.
- vLLM compile passes such as noop elimination remove reshape/slice clutter
  around quant ops.  Mini does not have an equivalent compiler pass, but
  copying a broad pass is out of scope and would not specifically address the
  now co-dominant projection/GEMM bucket.

Source-level evidence is sufficient for topology comparison, but the existing
vLLM Nsight capture still should not be used for per-bucket CUDA timing parity.

## Path Chosen

No graph/layout PoC was implemented.  The best remaining pure graph/layout
candidate either misses the gate by itself, barely reaches it only by stacking
multiple unrelated subpaths, or is really projection-adjacent GEMM work.

Implementation summary: no code changed outside this milestone record and its
analysis script.

Correctness and graph semantics: no new smoke was run because there was no code
change.  The active 07.54 baseline remains graph-correct with 4096/128 replay
`127`, 4096/1024 replay `1023`, and eager decode `0`.

## Macro Before/After

Because no PoC was selected, there is no new 07.55 macro line.

| Workload | 07.54 baseline output tok/s | 07.55 output tok/s | Note |
| --- | ---: | ---: | --- |
| 4096/128/bs4 | `43.0685` | n/a | no implementation |
| 4096/1024/bs4 | `87.0831` | n/a | no implementation |

## Profile Bucket Before/After

Because no PoC was selected, the 07.54 profile remains the comparable current
profile.

| Bucket | 07.54 kernel s | 07.55 kernel s | Note |
| --- | ---: | ---: | --- |
| graph/runtime/copy/cat/index | `1.1875` | n/a | no implementation |
| elementwise graph nodes | `0.6396` | n/a | no implementation |
| graph/layout cluster | `1.8271` | n/a | pivot gate baseline |
| projection/GEMM | `1.7968` | n/a | co-dominant next lever |

## Projection/GEMM Backend Parity Plan

Next target objective: prove whether the remaining gap is intrinsic GEMM
backend speed, projection-adjacent staging, graph node count around projection,
or precision/layout mismatch.

Minimum profiler and microbench gate:

1. Keep the same 07.54/07.55 variant and enable `MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1`
   to split projection ranges for `attn.q_proj`, `attn.q_wqb`, `attn.wo_a`,
   `attn.wo_b`, indexer `wq_b`, shared expert projection, and FFN projections.
2. Add or reuse a projection microbench that loads real DSV4 weights and
   measures mini `_quantized_linear_fp8_kernel`, mini BF16/CUTLASS/sgemm
   projections, and wrapper overhead in `DSV4Linear.forward`.
3. Run matching vLLM module-level probes for `QuantFP8`, `_C.scaled_fp8_quant`,
   quantized `ColumnParallelLinear`/`RowParallelLinear`, `deepseek_v4_fp8_einsum`,
   and the `wo_a` BMM/reference branch where SM80 uses it.
4. Reprofile 4096/128/bs4 and classify projection kernels separately from
   projection-adjacent copies and elementwise staging.

Backend comparison checklist:

| Mini item | vLLM comparison | Question |
| --- | --- | --- |
| `_quantized_linear_fp8_kernel` | vLLM quantized linear path plus `QuantFP8`/scaled FP8 quant | Is mini's per-call Triton GEMM intrinsically slower at decode `M <= 16`, or is staging dominant? |
| BF16/CUTLASS projection GEMMs | vLLM BF16/Marlin/linear dispatch for unfused projection pieces | Are the CUTLASS and cuBLAS kernels shape-mismatched for decode batch 4? |
| `DSV4Linear.forward` wrappers | vLLM compiled `ColumnParallelLinear`/`RowParallelLinear` wrappers | Are reshape/contiguous/cast nodes avoidable only through a compile boundary? |
| Attention WQA/WKV/QWQB/KV/WO paths | vLLM `fused_wqa_wkv`, lifted `wq_b`, `wo_a` BMM, `wo_b`, and `deepseek_v4_fp8_einsum` | Which projection owns the `1.1726 s` `_quantized_linear_fp8_kernel` time? |
| `wo_a_grouped_projection_fallback` | vLLM `fused_inv_rope_fp8_quant` plus `deepseek_v4_fp8_einsum` or SM80 BMM cache | Is mini missing a packed/grouped projection layout rather than a better generic GEMM? |
| Projection-adjacent FP8 activation quant | vLLM `QuantFP8` and compile quant matchers | Is the 07.54 helper now fast enough, or should quant feed directly into a better GEMM contract? |

Promotion gate for the projection target:

- identify one projection subpath responsible for at least `0.50 s` of the
  4096/128 decode-envelope projection/GEMM bucket; and
- show either at least `15%` reduction of projection/GEMM kernel time in a
  focused profile or at least `5%` 4096/128 output-throughput improvement in a
  graph-correct single-variant run; and
- preserve graph replay with eager decode `0`.

Recommended first implementation candidates only after the gate:

- replace or retune mini `_quantized_linear_fp8_kernel` for decode-small `M`
  using the vLLM/Marlin/CUTLASS dispatch behavior as reference;
- adapt vLLM's `wo_a` BMM or `deepseek_v4_fp8_einsum` boundary if `wo_a` owns
  the dominant projection time;
- move projection quant/layout staging into a projection custom-op boundary if
  profiling proves staging, not GEMM math, owns the delta.

Decision: pivot to projection/GEMM backend parity
