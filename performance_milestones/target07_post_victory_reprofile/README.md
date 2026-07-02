# TARGET 07.63 Post-Victory Reprofile

Date: 2026-07-02

This is the post-victory confirmation pass for `dsv4_sm80_a100_victory`.  It
does not continue into a new optimization.  The fresh bottleneck reset says the
next implementation target should be the decode graph/index/copy/cat metadata
bucket, not another projection-cache path by inertia.

## Repository And Environment

| Item | Value |
| --- | --- |
| workspace | `/workspace/mini-sglang` |
| branch | `dsv4-sglang-based` (`origin/main` ahead 36) |
| commit | `700374bc9c8d366781d404a33d6501852810d742` |
| tag/describe | `dsv4-sm80-woa-bmm-victory-dirty` |
| dirty state | pre-existing: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`, `prompts/target.md`, `prompts/TARGET_07.63_dsv4_sm80_post_victory_reprofile_and_next_bottleneck.md`; this target: `python/minisgl/kvcache/deepseek_v4_pool.py`, `performance_milestones/target07_post_victory_reprofile/` |
| GPUs | 8x `NVIDIA A100-SXM4-80GB`, compute capability `8.0`, 85,095,874,560 bytes each |
| CUDA_VISIBLE_DEVICES | `0,1,2,3,4,5,6,7` |
| PyTorch/CUDA/NCCL | `torch 2.9.1+cu128`, CUDA runtime `12.8`, NCCL `2.27.5` |
| profiler | NVIDIA Nsight Systems `2025.1.1.0` |
| model | `/models/DeepSeek-V4-Flash` |

Small config-path fix applied before promotion: the KV cache FP8 side-cache
allocation now checks `dsv4_env_flag(MINISGL_DSV4_SM80_INDEXER_FP8_CACHE)` so the
milestone bundle expansion and raw-env fallback agree.  This fixed the initial
text-smoke allocation failure for the bundle path; no new kernel or performance
optimization was added.

Artifacts are under:

- `raw/`: copied or symlinked raw smoke, macro, `.nsys-rep`, and SQLite files.
- `summaries/`: parsed env, macro, profile, owner, and memory-ledger summaries.
- `scripts/`: exact repro scripts used for this target.

## Variant Expansion

| Field | Value |
| --- | --- |
| milestone variant | `dsv4_sm80_a100_victory` |
| raw env | `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1` |
| MoE expert backend | `marlin_wna16` |
| CUDA graph | enabled |
| greedy sample graph capture | enabled |
| legacy alias equality | `dsv4_sm80_a100_victory == target0762_woabf16bmmcache` at env-expansion level: `true` |
| stale FP8 opt-ins | `MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM=false`, `MINISGL_DSV4_SM80_WO_B_FP8_GEMM=false`, `MINISGL_DSV4_SM80_INDEXER_WQB_FP8_GEMM=false` |

Expanded active toggles:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE
MINISGL_DSV4_SM80_BF16_PROJECTION_CACHE
MINISGL_DSV4_SM80_COMPRESS
MINISGL_DSV4_SM80_COMPRESS_STORE
MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON
MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE
MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT
MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE
MINISGL_DSV4_SM80_GATE_FP32_WEIGHT_CACHE
MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS
MINISGL_DSV4_SM80_HC
MINISGL_DSV4_SM80_INDEXER_BF16
MINISGL_DSV4_SM80_INDEXER_FP8_CACHE
MINISGL_DSV4_SM80_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE
MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_KV_BF16
MINISGL_DSV4_SM80_MOE_ROUTE
MINISGL_DSV4_SM80_MOE_V2
MINISGL_DSV4_SM80_MOE_VLLM_RUNNER
MINISGL_DSV4_SM80_PAGED_MQA_BF16
MINISGL_DSV4_SM80_Q_NORM_ROPE
MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_REPLAY_METADATA_COPY
MINISGL_DSV4_SM80_RMSNORM
MINISGL_DSV4_SM80_ROPE
MINISGL_DSV4_SM80_SPARSE_ATTN_BF16
MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16
MINISGL_DSV4_SM80_SWIGLU
MINISGL_DSV4_SM80_TOPK
MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE
MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE
```

Source artifact: `summaries/variant_env_expansion.json`.

## Text Smoke

Command family: TP8, page-size 256,
`--variants dsv4_sm80_a100_victory`.

| Prompt | Output | Sanity |
| --- | --- | --- |
| Chinese math | `2 + 2 等于 4。` | pass |
| English sky | `The sky is blue on a clear day.` | pass |
| Hangzhou | `杭州是风景如画的历史文化名城。` | pass |

Result: `pass`.  No garbage text, mojibake, repetition, or obvious factual
error was observed.  Graph replay was active during smoke (`replay_count=9`,
`greedy_sample_replay_count=9`, `eager_decode_count=0`, captured batch sizes
`[4, 2, 1]`).

Source artifacts:

- `raw/text_smoke_dsv4_sm80_a100_victory.json`
- `raw/text_smoke_dsv4_sm80_a100_victory.variant.json`
- `raw/text_smoke_dsv4_sm80_a100_victory.log`

## Macro Confirmation

Commands used `--page-size 256 --num-pages 128 --repeats 3 --warmup-repeats 1`
and `--variants dsv4_sm80_a100_victory`.  The required prompt did not spell out
`--num-pages`; the recent victory runs used 128 pages, so this confirmation kept
that memory shape fixed and records it explicitly.

| Workload | Status | Output tok/s | Decode tok/s | TTFT mean s | Prefill tok/s | Peak alloc/rank | Fallback calls | Unsupported skips | Repeat spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | pass | `59.5264` | `150.2022` | `4.9759` | `3829.9628` | `47,294,730,240` | `25,408` | `0` | `0.41%` |
| 4096/1024/batch4 | pass | `119.4153` | `149.1220` | `4.9652` | `3846.5327` | `47,294,760,960` | `25,408` | `0` | `0.53%` |

The 4096/1024/batch4 result stays above the old serving victory line
(`114.07 output tok/s`) by `+4.68%`.

Per-repeat output tok/s:

| Workload | Repeat 0 | Repeat 1 | Repeat 2 |
| --- | ---: | ---: | ---: |
| 4096/128/batch4 | `59.3936` | `59.5586` | `59.6381` |
| 4096/1024/batch4 | `119.0343` | `119.5501` | `119.6645` |

Source artifacts:

- `summaries/macro_4096x128_bs4_np128_summary.json`
- `summaries/macro_4096x1024_bs4_np128_summary.json`
- `raw/macro_4096x128_bs4_np128`
- `raw/macro_4096x1024_bs4_np128`

## Graph Replay And Eager Decode

| Workload | Graph replay | Greedy sample replay | Eager decode | Captured BS | Replay input copy bytes | Schedule decode batches |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| text smoke | `9` | `9` | `0` | `[4, 2, 1]` | `432` | n/a |
| 4096/128/batch4 | `508` | `508` | `0` | `[4, 2, 1]` | `24,384` | `381` |
| 4096/1024/batch4 | `4092` | `4092` | `0` | `[4, 2, 1]` | `196,416` | `3069` |
| nsys 4096/128/batch4 short run | `127` | `127` | `0` | `[4, 2, 1]` | from profile run | `127` |

Graph gate passes: decode graph replay is active and measured decode loops have
zero eager decode.

## Nsight Profile

Short rank-0 Nsight Systems profile:

- workload: 4096/128/batch4, page-size 256, num-pages 128
- variant: `dsv4_sm80_a100_victory`
- repeats: 1, warmup repeats: 0
- graph capture NVTX: `MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1`
- exported SQLite: `raw/nsys_target0763_post_victory_4096x128_bs4_np128_rank0.sqlite`
- report: `raw/nsys_target0763_post_victory_4096x128_bs4_np128_rank0.nsys-rep`

The profiled run is slower because of profiler overhead
(`48.9427 output tok/s`, `136.9254 decode tok/s`), so macro throughput is taken
from the non-profiled confirmation above.  The profile is still usable for
relative decode-envelope attribution.

Decode envelope:

- wall time: `3.942934s`
- CUDA kernel time classified: `3.357065s`
- coverage of decode envelope by classified CUDA kernel buckets: `85.14%`
- CUDA kernel count: `746,663`
- memcpy inside envelope: `0.023252s`, `338,080,832` bytes

## Fresh Top-Bucket Bottlenecks

The 07.62 `attn.wo_a` cache changed the ranking.  Current top buckets for the
4096/128/batch4 decode envelope are:

| Rank | Bucket | Kernel s | Share of decode envelope | Kernel count | Graph node count | Interpretation |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `graph_runtime_copy_cat_index` | `0.846795` | `21.48%` | `248,069` | `1,895` | top remaining concentration; copy/index/cat/gather metadata boundary |
| 2 | `projection_gemm` | `0.812100` | `20.60%` | `100,965` | `795` | diffuse across projections and shared experts after BF16 caches |
| 3 | `elementwise_graph_nodes` | `0.497965` | `12.63%` | `201,670` | `1,551` | many graph-visible small ops; overlaps with metadata deforestation opportunity |
| 4 | `nccl_communication` | `0.340015` | `8.62%` | `11,176` | `88` | row-parallel and MoE reductions |
| 5 | `moe_marlin` | `0.300516` | `7.62%` | `43,688` | `344` | Marlin WNA16 expert path |
| 6 | `sampling_logits` | `0.181868` | `4.61%` | `43,815` | `345` | sampler/logits path |
| 7 | `fp8_indexer` | `0.131447` | `3.33%` | `20,828` | `164` | indexer compute/cache path |
| 8 | `sparse_attention_decode` | `0.118446` | `3.00%` | `21,590` | `170` | sparse decode kernel path |
| 9 | `unknown` | `0.099853` | `2.53%` | `46,734` | `366` | residual unclassified CUDA kernels |
| 10 | `kv_compressor_cache_store` | `0.028062` | `0.71%` | `8,128` | `64` | no longer a top bucket |

Projection-owner context:

| Owner | Decode kernel s | Intrinsic GEMM s | Graph nodes |
| --- | ---: | ---: | ---: |
| `mlp.routed_experts` | `0.408459` | `0.000000` | `688` |
| `shared_experts.gate_up_proj` | `0.272958` | `0.045904` | `387` |
| `attn.wo_b` | `0.229213` | `0.057773` | `129` |
| `shared_experts.down_proj` | `0.195412` | `0.029315` | `387` |
| `attn.q_proj_wqa_wkv` | `0.131247` | `0.089719` | `215` |
| `indexer.compressor` | `0.100438` | `0.026739` | `252` |
| `attn.q_wqb` | `0.079818` | `0.068381` | `86` |
| `lm_head` | `0.077000` | `0.026498` | `5` |
| `attn.wo_a` | `0.068947` | `0.068947` | `86` |
| `indexer.wq_b` | `0.056181` | `0.050902` | `42` |

`attn.wo_a` remains low after TARGET 07.62 (`0.068947s`) and should not be the
next target.

Source artifacts:

- `summaries/nsys_target0763_post_victory_4096x128_bs4_np128_rank0_classified.json`
- `summaries/nsys_target0763_post_victory_4096x128_bs4_np128_rank0_projection_owner.json`

## mini-vs-vLLM Parity

vLLM timing caveat: existing vLLM Nsight summaries in this workspace have a
found NVTX repeat window but `kernel.count=0` inside that window, so they are not
reliable for bucket timing.  This table therefore uses mini owner timing plus
vLLM source-dispatch parity, as allowed by the target prompt.

| Bucket | mini current behavior | vLLM source/profile evidence | Difference | Plausible E2E upside | Next action |
| --- | --- | --- | --- | ---: | --- |
| `graph_runtime_copy_cat_index` | `0.846795s`, `21.48%`, `248,069` kernels, `1,895` graph nodes in decode envelope | vLLM has fused helpers in `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`: `flat_index_dequant_gather_blocked` says it replaces a 20-op PyTorch gather/dequant path; `compute_global_topk_indices_and_lens` fuses block-table lookup, valid-entry counting, and padding masking; `combine_topk_swa_indices` fuses topk/SWA combination. Dispatch sites are in `deepseek_v4_attention.py` around decode topk and gather paths. | mini still exposes a large graph/runtime copy/index/cat surface after projection caches; vLLM has named fused source boundaries for the same semantic work. | `5-8%` if 30-40% of this bucket is removed and the gain repeats through 4096/1024 decode | selected |
| `projection_gemm` | `0.812100s`, `20.60%`, but spread across routed/shared experts, `wo_b`, `q_wqb`, `wo_a`, indexer, and logits | vLLM includes SM80 reference `wo_a` BMM and FP8 inverse-RoPE/einsum paths in `deepseek_v4_attention.py`; mini already paid for four BF16 projection caches and `attn.wo_a` is `0.068947s`. | remaining projection cost is diffuse; no single stale FP8-GEMM opt-in is the current best path. | below selection bar for a single named owner | hold |
| `elementwise_graph_nodes` | `0.497965s`, `12.63%`, `201,670` kernels, `1,551` graph nodes | vLLM source fuses adjacent attention/indexer metadata boundaries; this likely overlaps with the first bucket rather than being an independent target. | treating this as generic graph cleanup would violate the target prompt; use it only as secondary validation for the selected metadata boundary. | tied to selected bucket | hold |

Additional lower buckets (`nccl_communication=0.340015s`, `moe_marlin=0.300516s`)
are meaningful but are not top-two after the post-victory reset.

## Cache And Workspace Memory Ledger

Current run memory shape: page-size `256`, num-pages `128`,
KV cache bytes/rank max `2,491,495,680`, equivalent
`76,034.41 bytes/token/rank`.

| Owner | Enabled | Layers | Shape/rank | Dtype | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank | Lifecycle | Decode allocation |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `attn.q_wqb` | yes | 43 | `[4096, 1024]` | `torch.bfloat16` | `360,710,144` | `0.3359` | `4,744.04` | `18.53` | prebuilt before graph capture | no |
| `attn.wo_b` | yes | 43 | `[4096, 1024]` | `torch.bfloat16` | `360,710,144` | `0.3359` | `4,744.04` | `18.53` | prebuilt before graph capture | no |
| `indexer.wq_b` | yes | 21 | `[8192, 1024]` | `torch.bfloat16` | `352,321,536` | `0.3281` | `4,633.71` | `18.10` | prebuilt before graph capture | no |
| `attn.wo_a` | yes | 43 | `[1, 4096, 1024]` | `torch.bfloat16` | `360,710,144` | `0.3359` | `4,744.04` | `18.53` | prebuilt before graph capture | no |
| `indexer.fp8_paged_cache` | yes | 21 | `[21, 128, 8448]` | `torch.uint8` | `22,708,224` | `0.0211` | `298.66` | `1.17` | allocated with KV cache pool; populated during store | no |
| `moe_v2_workspace` | no | 0 | lazy reusable buffers | mixed | `0` | `0.0000` | `0.00` | `0.00` | inactive with current Marlin WNA16 backend | no |

| Total | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank |
| --- | ---: | ---: | ---: | ---: |
| cached BF16 projection stack | `1,434,451,968` | `1.3359` | `18,865.83` | `73.69` |
| listed extra cache/workspace | `1,457,160,192` | `1.3571` | `19,164.48` | `74.86` |

Delta versus the pre-`wo_a` BF16-cache baseline artifact:

- peak allocated delta: `+360,710,144` bytes/rank
- peak reserved delta: `+381,681,664` bytes/rank

Ownership note: `indexer.fp8_paged_cache` is still owned inside
`DeepSeekV4KVCache`; the grouped MoE workspace abstraction exists but is not
materialized in the current Marlin WNA16 path.  A future unified cache/workspace
manager would be cleaner, but that is not an implementation item for this
target.

Source artifacts:

- `summaries/cache_workspace_memory_ledger.json`
- `summaries/cache_workspace_memory_ledger.md`

## Selected Next Implementation Target

Exactly one next target is recommended:

`DSV4 SM80 decode metadata deforestation for graph_runtime_copy_cat_index`

Measured current cost:

- bucket: `graph_runtime_copy_cat_index`
- workload/profile: 4096/128/batch4 short nsys decode envelope
- cost: `0.846795s`, `21.48%` of decode envelope
- surface: `248,069` kernels and `1,895` graph nodes
- expected 4096/1024 relevance: this work repeats across decode steps, and the
  macro long-decode path is decode dominated.

Target shape:

- fuse or eliminate the concentrated gather/index/copy/cat/topk-lens boundary in
  the decode metadata path;
- use vLLM source parity as the design compass:
  `flat_index_dequant_gather_blocked`,
  `compute_global_topk_indices_and_lens`, and
  `combine_topk_swa_indices`;
- preserve the milestone bundle name and smoke/macro gates.

Expected payoff gate:

- remove at least `0.25s` from the 4096/128 decode envelope, or
- show at least `5%` E2E gain on 4096/1024/batch4.

Stop conditions for that future implementation target:

- TP8/page-size-256 text smoke fails;
- graph replay is inactive or eager decode becomes nonzero;
- fallback wrapper calls or unsupported kernel skips increase unexpectedly;
- 4096/1024/batch4 drops below `114.07 output tok/s` after confirming variant
  expansion, page-size 256, num-pages 128, and graph replay state;
- the fused metadata path cannot remove at least `0.25s` from the profiled
  4096/128 decode envelope;
- the change requires broad cache/workspace-manager refactoring.

## Do Not Continue In This Thread

Stop here after this report.  Do not implement the selected metadata
deforestation target in TARGET 07.63.

Do not continue by:

- using the historical `target0762_woabf16bmmcache` name as the new primary
  report variant;
- enabling stale `Q_WQB/WO_B/INDEXER_WQB_FP8_GEMM` opt-ins in the best path;
- adding another projection cache simply because projection was historically
  hot;
- starting generic graph/layout cleanup without the named
  `graph_runtime_copy_cat_index` bucket and measurable stop gates;
- treating vLLM profile timing as reliable until the repeat-window kernel-count
  issue is fixed.

