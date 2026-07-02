# TARGET 07: DeepSeek V4 SM80 vLLM Gap Closure

## Goal

Close the remaining DeepSeek V4 Flash performance gap between mini-sglang and
the old vLLM-based framework on A100/sm80.

Primary win condition:

- TP8, single-node 8x A100 sm80;
- page/block size 256;
- `/models/DeepSeek-V4-Flash`;
- 4096 input tokens/request;
- 1024 output tokens/request;
- batch size 4;
- output throughput strictly above the old vLLM serving baseline:
  `114.07 output tok/s`.

The promoted default path must remain exact unless a later precision target
explicitly proves and accepts a quality tradeoff.

## Current Best Milestone Result

Current best milestone stack:

- variant: `dsv4_sm80_a100_victory`;
- top-level env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`;
- Marlin WNA16 MoE backend;
- global topk/lens;
- bf16 gather/mask plus split-K sparse decode;
- FP8 indexer cache backend and fused FP8 activation quant helper;
- cached BF16 projection stack for `q_wqb`, `wo_b`, `indexer.wq_b`, and
  `wo_a` grouped BMM;
- cached BF16 shared expert gate/up/down projection weights;
- DSV4 decode CUDA graph replay;
- page size 256;
- TP8 on 8x A100.

Current confirmed macro from TARGET 07.67:

| Workload | Output tok/s | Note |
| --- | ---: | --- |
| 4096/128/batch4 | `62.1364` | Promoted path after shared expert BF16 cache promotion |
| 4096/1024/batch4 | `131.6263` | Stable above old `114.07` victory line |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- fresh vLLM offline 4096/1024/batch4: `201.99 output tok/s`;
- vLLM's fast path uses `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, FP8
  indexer/cache pieces, and additional graph/runtime machinery, so it is not
  precision-neutral.

This milestone is an opt-in best path, not a declaration that every component
should become default.  TARGET 07.63 confirmed the bundle after opt-in cleanup
and rebuilt the bottleneck order.  TARGET 07.64 added a correct metadata
deforestation opt-in, but it did not meet the promote gate.  TARGET 07.65 then
attributed the remaining direct-copy owners and selected MoE/shared-expert
staging.  TARGET 07.66 promoted the shared expert BF16 weight cache into the
victory bundle and changed the bottleneck order again.  TARGET 07.67 confirmed
the promoted path and selected HC/elementwise graph cleanup as the next exact
implementation target.  TARGET 07.68 proved that HC cleanup is correct but too
small to promote, so the active next step is projection/GEMM backend and owner
re-attribution on the current promoted path.  TARGET 07.69 proved that the old
FP8 quantized-linear projection bottleneck is gone and selected the current
BF16 small-GEMM backend cluster as the next implementation surface.

## New TARGET 07 Organization

Completed history is now merged into larger readable files.  Future work stays
split into small executable target files for separate Codex threads.

| Stage | Prompt | Status | Purpose |
| --- | --- | --- | --- |
| TARGET 07.10 | `prompts/TARGET_07.10_dsv4_sm80_foundation_history.md` | completed history | Fair rebench, communication/CUDA graph, and subgraph parity history. |
| TARGET 07.20 | `prompts/TARGET_07.20_dsv4_sm80_moe_history.md` | completed history | MoE route from MoE V2 through mini-owned Marlin WNA16, including why MoE is no longer primary. |
| TARGET 07.30 | `prompts/TARGET_07.30_dsv4_sm80_attention_history.md` | completed history | Attention/indexer/cache route through global topk/lens and bf16 split-K sparse decode. |
| TARGET 07.40 | `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md` | completed | Reprofiled the post-splitK exact stack; decode split-K is no longer the main bottleneck. |
| TARGET 07.41 | `prompts/TARGET_07.41_dsv4_sm80_indexer_cache_runtime_exact.md` | completed | Validated an exact replay metadata-copy cut, but macro gain was negligible; do not continue local metacopy polish. |
| TARGET 07.42 | `prompts/TARGET_07.42_dsv4_sm80_vllm_metadata_runtime_parity.md` | completed | Evidence report found no justified exact runtime PoC; recommended precision/cache, but vLLM per-bucket timing remains incomplete. |
| TARGET 07.43 | `prompts/TARGET_07.43_dsv4_sm80_vllm_ablation_before_precision.md` | completed | vLLM ablations found aux stream and persistent topk are not standalone macro factors; eager-vs-graph confirms graph is mandatory but not the next mini action item. |
| TARGET 07.50 | `prompts/TARGET_07.50_dsv4_sm80_fp8_cache_indexer_precision.md` | completed | Narrow mini-owned FP8 indexer cache/logits slice passed quality smoke but was slower than bf16; stop that slice. |
| TARGET 07.51 | `prompts/TARGET_07.51_dsv4_sm80_vllm_fp8_backend_parity.md` | completed | Isolated vLLM's actual FP8 indexer and `fp8_ds_mla` gather/dequant backends; decision is to port/adapt vLLM's FP8 indexer backend next. |
| TARGET 07.52 | `prompts/TARGET_07.52_dsv4_sm80_vllm_fp8_indexer_backend_port.md` | completed | Ported a vLLM-aligned FP8 paged indexer backend as an opt-in path; microbench reached vLLM-adjacent speed and 4096/1024 improved to `73.67 output tok/s`, but the main gap remains. |
| TARGET 07.53 | `prompts/TARGET_07.53_dsv4_sm80_post_fp8_indexer_reprofile.md` | completed | Reprofiled the FP8-indexer stack; graph/layout copy/cat/index plus elementwise graph nodes are the largest actionable cluster, with projection/GEMM as pivot. |
| TARGET 07.54 | `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md` | completed | Fused the repeated FP8 activation fake-quant chain into an opt-in Triton helper; graph/layout cluster dropped `38.59%`, 4096/1024 improved to `87.08 output tok/s`, but projection/GEMM is now tied with remaining graph/layout. |
| TARGET 07.55 | `prompts/TARGET_07.55_dsv4_sm80_remaining_graph_layout_or_projection_pivot.md` | completed | Re-attributed the remaining graph/layout cluster; no single concentrated layout PoC met the gate, and the decision is to pivot to projection/GEMM backend parity. |
| TARGET 07.56 | `prompts/TARGET_07.56_dsv4_sm80_low_cost_graph_layout_compile_preflight.md` | completed | Static scale cache removed focused wrapper copy/cast events but only improved 4096/128 by `+0.35%`; no low-cost graph/layout cut was promoted. |
| TARGET 07.57 | `prompts/TARGET_07.57_dsv4_sm80_projection_gemm_backend_parity.md` | completed | Attributed projection/GEMM by owner; selected `_quantized_linear_fp8_kernel` across `attn.q_wqb`, `attn.wo_b`, and `indexer.wq_b` as the next backend contract. |
| TARGET 07.58 | `prompts/TARGET_07.58_dsv4_sm80_cached_bf16_projection_backend.md` | completed | Promoted opt-in cached BF16 dequantized weights for `attn.q_wqb`; 4096/128 improved to `47.9464 output tok/s`, 4096/1024 to `92.5170`, with only `0.3359 GiB/rank` extra cache. |
| TARGET 07.59 | `prompts/TARGET_07.59_dsv4_sm80_cached_bf16_wo_b_projection_backend.md` | completed | Extended cached BF16 to row-parallel `attn.wo_b`; 4096/128 reached `49.6585 output tok/s`, 4096/1024 reached `98.6953`, and `wo_b` local compute fell to `0.070595s` while all-reduce remained `0.161865s`. |
| TARGET 07.60 | `prompts/TARGET_07.60_dsv4_sm80_cached_bf16_indexer_wq_b_projection_backend.md` | completed | Extended cached BF16 to `indexer.wq_b`; 4096/128 reached `51.2962 output tok/s`, 4096/1024 reached `105.7645`, and the three cached owners cost exactly `1.0000 GiB/rank`. |
| TARGET 07.61 | `prompts/TARGET_07.61_dsv4_sm80_post_cached_bf16_vllm_parity_reprofile.md` | completed | Completed post-cached-BF16 parity reprofile. vLLM runtime bucket timing is still unavailable, but mini owner timing plus vLLM source parity select `attn.wo_a` as the next narrow boundary. |
| TARGET 07.62 | `prompts/TARGET_07.62_dsv4_sm80_wo_a_attention_boundary_parity.md` | completed | Adapted mini's `attn.wo_a` boundary to an opt-in BF16 grouped BMM cache; 4096/1024 reached `116.2553 output tok/s`, crossing the old `114.07` victory line. |
| TARGET 07.63 | `prompts/TARGET_07.63_dsv4_sm80_post_victory_reprofile_and_next_bottleneck.md` | completed | Confirmed `dsv4_sm80_a100_victory`: text smoke pass, 4096/1024 reached `119.4153 output tok/s`, graph replay stayed active, eager decode stayed `0`, and fresh profile selected `graph_runtime_copy_cat_index` as the next implementation target. |
| TARGET 07.64 | `prompts/TARGET_07.64_dsv4_sm80_decode_metadata_deforestation.md` | completed | Added an opt-in decode metadata helper. Microbench was strong and macro improved to `122.9414 output tok/s`, but `graph_runtime_copy_cat_index` only fell by `0.012003s`; keep opt-in, do not promote. |
| TARGET 07.65 | `prompts/TARGET_07.65_dsv4_sm80_direct_copy_owner_attribution.md` | completed | Measurement-only attribution reached `99.97%` direct-copy owner coverage and selected MoE/shared-expert staging as the next implementation target. |
| TARGET 07.66 | `prompts/TARGET_07.66_dsv4_sm80_moe_shared_expert_staging_cleanup.md` | completed | Promoted shared expert BF16 weight cache into `dsv4_sm80_a100_victory`; MoE/shared direct-copy staging fell `0.379204s -> 0.097361s`, and 4096/1024 reached `131.7707 output tok/s`. |
| TARGET 07.67 | `prompts/TARGET_07.67_dsv4_sm80_post_shared_expert_reprofile.md` | completed | Confirmed the promoted post-07.66 path at `131.6263 output tok/s` on 4096/1024 and selected HC/elementwise graph cleanup from fresh bucket evidence. |
| TARGET 07.68 | `prompts/TARGET_07.68_dsv4_sm80_hc_elementwise_graph_cleanup.md` | completed | Added an exact BF16 opt-in HC graph cleanup path. Focused HC pre improved, correctness passed, but 4096/1024 macro and profile gates failed; keep opt-in, do not promote. |
| TARGET 07.69 | `prompts/TARGET_07.69_dsv4_sm80_projection_gemm_backend_owner_reattribution.md` | completed | Re-attributed the current promoted projection/GEMM bucket with `98.94%` named coverage. No single owner clears `0.20s`; the selected next surface is the BF16 small-GEMM + splitK/reduce cluster at `0.521619s`. |
| TARGET 07.70 | `prompts/TARGET_07.70_dsv4_sm80_bf16_small_gemm_backend_cluster.md` | next todo | Try exact-route BF16 small-GEMM backend solutions: cuBLASLt algorithm/splitK policy, prepacked layout, CUTLASS/Triton small-M kernels, owner-local grouped/fused GEMM, and narrow compile probes only if earlier lanes fail. |

The old fine-grained TARGET 07 prompt files remain as archival references.  Do
not use them as the main project map unless a thread needs exact historical
commands or context.

## Archived Fine-Grained Prompts

Foundation:

- `prompts/archive/target07/TARGET_07.1_dsv4_sm80_fair_rebench_vllm_diff.md`
- `prompts/archive/target07/TARGET_07.2_dsv4_sm80_comm_cuda_graph.md`
- `prompts/archive/target07/TARGET_07.25_dsv4_sm80_vllm_subgraph_parity.md`

MoE:

- `prompts/archive/target07/TARGET_07.3_dsv4_sm80_moe_v2_exact.md`
- `prompts/archive/target07/TARGET_07.35_dsv4_sm80_post_moe_reparity.md`
- `prompts/archive/target07/TARGET_07.36_dsv4_sm80_vllm_fused_moe_runner_adapt.md`
- `prompts/archive/target07/TARGET_07.37_dsv4_sm80_moe_backend_identification.md`
- `prompts/archive/target07/TARGET_07.38_dsv4_sm80_moe_exact_backend_adapt.md`
- `prompts/archive/target07/TARGET_07.39_dsv4_sm80_marlin_custom_op_bridge.md`
- `prompts/archive/target07/TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port.md`

Attention/indexer/cache:

- `prompts/archive/target07/TARGET_07.392_dsv4_sm80_post_marlin_reprofile.md`
- `prompts/archive/target07/TARGET_07.393_dsv4_sm80_attention_indexer_cache_runtime_rework.md`
- `prompts/archive/target07/TARGET_07.394_dsv4_sm80_exact_attention_indexer_boundary_adapt.md`
- `prompts/archive/target07/TARGET_07.395_dsv4_sm80_bf16_sparse_decode_splitk.md`

Broad precision archive:

- `prompts/archive/target07/TARGET_07.4_dsv4_sm80_precision_lanes.md`

## Current Sequencing

Run TARGET 07.70 next.

Current milestone: `dsv4_sm80_a100_victory`.

TARGET 07.395 proved that mini's exact bf16 sparse decode boundary can match
the comparable vLLM gather+split-K decode boundary:

- mini exact sparse-only decode: about `0.2284 ms`;
- vLLM prior gather+split-K decode probe: about `0.2258 ms`.

TARGET 07.40 then showed that the post-splitK top exact-path costs moved to
runtime/copy/cat/index graph nodes, elementwise graph nodes, legacy
prefill/extend sparse attention, indexer/cache/topk work, and projection GEMM.
TARGET 07.41 optimized one real replay metadata-copy subgraph, but macro moved
only from `38.9379` to `39.0028` output tok/s on 4096/128 and from `68.8097`
to `68.6314` output tok/s on 4096/1024.

TARGET 07.42 built the mini-vs-vLLM parity table and did not find a justified
exact runtime PoC.  Its strongest evidence-backed next hypothesis is vLLM's
packed FP8 KV/indexer cache lane, but it also identified an unproven suspicion:
vLLM's attention custom-op plus aux-stream overlap and V1 graph/runtime
discipline may hide part of mini's runtime/copy/elementwise bucket.

TARGET 07.43 then ran the vLLM ablation pass:

- aux-stream overlap off: `-0.54%` on 4096/128;
- persistent topk/indexer fast path off: `+0.21%` on 4096/128;
- enforce eager: `-69.82%` on 4096/128 and `-84.89%` on 4096/1024.

This confirms that vLLM's graph/compile path is mandatory, but mini already has
decode graph replay.  The remaining actionable hypothesis is not "add CUDA
graph"; it is that vLLM's graph executes a lighter FP8 cache/indexer layout.
TARGET 07.50 implemented the first narrow FP8 indexer cache/logits slice.  It
passed quality smoke but failed performance:

- same-run exact control 4096/128: `37.9237 output tok/s`;
- FP8 indexer cache/logits 4096/128: `29.6691 output tok/s`;
- FP8 logits microbench was slower than bf16 logits on all measured shapes.

This did not disprove vLLM's full FP8 lane, because the 07.50 implementation
was mini-owned and did not prove backend parity with vLLM's actual
`fp8_paged_mqa_logits_triton` or `fp8_ds_mla` gather/dequant kernels.

TARGET 07.51 then isolated vLLM's real FP8 backend pieces.  The main result:

- vLLM FP8 Q path at batch16/history4096: `0.0839 ms`, vs mini FP8 Q
  `0.2308 ms`;
- vLLM FP8 indexer K store: `0.0964 ms`, vs mini FP8 store `0.2941 ms`;
- vLLM FP8 paged decode logits: `0.1529 ms`, vs mini bf16 logits
  `0.3076 ms` and mini FP8 logits `1.3072 ms`;
- vLLM logits plus topk: `0.1804 ms`, vs mini bf16 select `0.3586 ms`;
- quality was acceptable for an opt-in path, with top-k overlap about `0.973`
  at the largest measured shape.

The important nuance is that vLLM's model path does not use the standalone
`quantize_and_insert_k_cache` wrapper for SM80 KV-cache store.  That standalone
probe compiles `tl.float8e4nv` and fails on A100.  vLLM's real model path uses
fused compressor/insert kernels with SM80 software-FP8 branches.  Therefore
TARGET 07.52 should port or closely adapt the FP8 indexer backend first, and
must not start by porting standalone `quantize_and_insert_k_cache` or full
`fp8_ds_mla` KV cache E2E.

TARGET 07.52 succeeded as an opt-in FP8 indexer port:

- mini FP8 paged indexer logits at batch16/history4096: `0.1845 ms`, within
  `1.21x` of vLLM's isolated `0.1529 ms`;
- large-shape FP8 select: `0.2472 ms`, vs mini bf16 select `0.3709 ms`;
- text smoke passed with graph replay;
- 4096/128/batch4: `41.63 output tok/s`;
- 4096/1024/batch4: `73.67 output tok/s`.

This is a real improvement, but it is not the decisive vLLM gap closer.  The
4096/1024 run remains decode dominated: total `55.60 s`, decode forward
`47.46 s`, prefill forward `5.47 s`.  Relative to the historical exact
4096/1024 result `68.81`, the opt-in FP8 indexer macro gain is about `7%`.
TARGET 07.53 must therefore refresh the Nsight/profile attribution with FP8
indexer enabled and compare the new top buckets against vLLM before selecting
the next implementation target.

TARGET 07.53 completed that reprofile.  The opt-in FP8-indexer path was:

- 4096/128/batch4: `41.66 output tok/s`;
- 4096/1024/batch4: `73.67 output tok/s`;
- vLLM reference remains about `82.28` on 4096/128 and `202.03` on
  4096/1024.

Fresh 4096/128/batch4 rank0 node trace with FP8 indexer enabled showed decode
envelope top buckets:

- projection/GEMM: `1.7973 s`, `27.49%`;
- graph/runtime/copy/cat/index: `1.6170 s`, `24.73%`;
- elementwise graph nodes: `1.3583 s`, `20.77%`;
- FP8 indexer: `0.1301 s`, `1.99%`;
- sparse attention decode: `0.1180 s`, `1.80%`;
- KV/cache store: `0.0281 s`, `0.43%`.

The combined graph/layout cluster was `2.9752 s`, or `45.50%` of the measured
decode-envelope wall.  TARGET 07.54 then proved that this mismatch was real by
fusing the repeated FP8 activation fake-quant chain into an opt-in Triton
helper:

- graph/runtime/copy/cat/index fell from `1.6170 s` to `1.1875 s`;
- elementwise graph nodes fell from `1.3583 s` to `0.6396 s`;
- graph/layout cluster fell from `2.9752 s` to `1.8271 s`, a `38.59%`
  reduction;
- 4096/128/batch4 improved from `41.66` to `43.07 output tok/s`;
- 4096/1024/batch4 improved from `73.67` to `87.08 output tok/s`.

The 07.54 result is a real step forward, but it also changed the bottleneck
shape.  Remaining graph/layout is now effectively tied with projection/GEMM:

- graph/layout cluster: `1.8271 s`;
- projection/GEMM: `1.7968 s`.

TARGET 07.55 is therefore the last graph/layout triage target before a likely
projection/GEMM pivot.  It should attribute the remaining direct-copy,
bf16/float8 copy, CatArray/index/gather, and pow/mean/mul nodes against vLLM
source boundaries; implement at most one concentrated graph/layout PoC; and
stop if the next best change is really projection/GEMM backend work.

TARGET 07.55 completed that triage without changing runtime code.  Its
conclusion was:

- remaining direct-copy kernels are large (`0.9456 s`) but too diffuse across
  graph input copy, attention metadata, projection reshape/contiguous, and
  linear wrappers;
- BF16/float8 copy kernels (`0.1318 s`) are below the standalone graph/layout
  gate;
- CatArray/index/gather/topk assembly barely reaches the old gate only when
  multiple subpaths are stacked (`0.1830 s`);
- pow/mean/mul elementwise nodes remain sizable (`0.5148 s`) but are not
  attributable to one stable source boundary;
- projection/GEMM (`1.7968 s`) is now effectively tied with the whole remaining
  graph/layout cluster (`1.8271 s`).

The next major direction is projection/GEMM backend parity against vLLM.
However, code review found a few low-cost preflight candidates that are worth
settling first because they sit on projection-adjacent staging boundaries:

- cache static projection scales that are repeatedly converted with
  `scale.float().contiguous()`;
- try a narrow `torch.compile` probe for pure HC-head math, similar to vLLM's
  compiled `hc_head`;
- audit no-op reshape/view/contiguous clutter around projection/quant
  boundaries, inspired by vLLM's compile cleanup passes.

TARGET 07.56 should run only this short preflight.  If it does not produce a
small justified win, proceed directly to projection/GEMM backend parity.

TARGET 07.56 completed the preflight with one tiny opt-in PoC:

- `MINISGL_DSV4_SM80_STATIC_SCALE_CACHE=1`;
- focused wrapper microbench removed `scale.float().contiguous()` copy/cast
  events and saved about `16-17 us` per decode-small FP8 wrapper call;
- 4096/128/batch4 moved only from `43.0685` to `43.2194 output tok/s`
  (`+0.35%`), below the `+2%` preflight gate;
- HC-head compile and no-op cleanup were not implemented because static scale
  cache already proved low-cost staging cuts were too small to be the main
  path.

TARGET 07.57 completed the projection/GEMM backend attribution without landing
a large kernel PoC.  Its key result is that the strongest bottleneck is the
mini `_quantized_linear_fp8_kernel` backend contract, not a broad graph/layout
issue and not `wo_a`:

- `attn.q_wqb`: `0.404178 s` intrinsic projection time;
- `attn.wo_b`: `0.403710 s` intrinsic projection time, plus row-parallel
  all-reduce;
- `indexer.wq_b`: `0.364756 s` intrinsic projection time;
- combined same-contract total: `1.172645 s`.

The real-weight microbench in 07.57 showed that current wrapper time is much
larger than cached-dequant BF16 `F.linear` time:

- `attn.q_wqb`: about `0.412 ms` wrapper vs about `0.053 ms` cached-dequant
  BF16 matmul;
- `attn.wo_b`: about `0.660 ms` wrapper vs about `0.052 ms` cached-dequant
  BF16 matmul;
- `indexer.wq_b`: about `0.168 ms` wrapper vs about `0.019 ms`
  cached-dequant BF16 matmul.

The next target is therefore TARGET 07.58: an opt-in cached BF16 dequantized
weight backend for the dominant FP8 projection contract.  It should start with
`attn.q_wqb` only, because that owner is large and does not include `wo_b`'s
communication.  The target must record the VRAM cost and convert it to lost KV
cache tokens/pages before expanding to `wo_b` or `indexer.wq_b`.

TARGET 07.58 completed successfully and promoted the q_wqb cached BF16 path as
an opt-in backend:

- 4096/128/batch4 improved from `43.0685` to `47.9464 output tok/s`;
- 4096/1024/batch4 improved from `87.0831` to `92.5170 output tok/s`;
- decode tok/s improved by about `7.7%`;
- text smoke passed, graph replay stayed active, and eager decode remained `0`;
- actual TP8 local q_wqb shape was `[4096, 1024]`, so 43 layers cost only
  `360,710,144 bytes/rank` (`0.3359 GiB/rank`), about `4744` KV tokens or
  `18.53` pages at page size 256.

The next target is TARGET 07.59: extend the same cached BF16 dequantized-weight
backend to `attn.wo_b`.  The main additional care is that `wo_b` is
row-parallel, so the target must report local projection compute separately
from row-parallel all-reduce.  If local projection improves but macro gain is
masked by communication, the next decision should name that explicitly rather
than treating cached BF16 as failed.

TARGET 07.59 completed successfully:

- 4096/128/batch4 improved from `47.9464` to `49.6585 output tok/s`;
- 4096/1024/batch4 improved from `92.5170` to `98.6953 output tok/s`;
- `wo_b` local projection dropped from the old FP8 path to
  `0.059160s` BF16 GEMM plus `0.011435s` activation quant;
- `wo_b` row-parallel all-reduce remains `0.161865s`;
- q_wqb+wo_b cached BF16 memory is `721,420,288 bytes/rank`
  (`0.6719 GiB/rank`), about `9488` KV tokens or `37.06` pages.

The largest remaining same-contract projection owner is now `indexer.wq_b`:

- intrinsic `_quantized_linear_fp8_kernel`: `0.364997s`;
- activation quant: `0.005293s`;
- copy/layout: `0.012928s`;
- C4/indexer layers: `21`.

The next target is TARGET 07.60: extend cached BF16 to `indexer.wq_b`.  After
that target, the original q_wqb/wo_b/indexer_wq_b projection sequence is
complete, so the thread must run a fresh profile and choose the next bottleneck
from evidence rather than continuing cached-weight work by inertia.

TARGET 07.60 completed successfully:

- 4096/128/batch4 improved from `49.6585` to `51.2962 output tok/s`;
- 4096/1024/batch4 improved from `98.6953` to `105.7645 output tok/s`;
- `indexer.wq_b` intrinsic dropped from `0.364997s` to `0.050961s`;
- graph replay stayed active, eager decode remained `0`, and text smoke passed;
- three-owner cached BF16 memory is exactly `1.0000 GiB/rank`, about
  `14121.79` KV tokens or `55.16` pages per rank.

The fresh 07.60 post-sequence profile says not to continue cached-weight work
without new evidence.  The largest current mini clusters are:

- graph/runtime/copy/cat/index: `1.141006s`, `25.88%`;
- projection/GEMM: `0.805080s`, `18.26%`;
- elementwise graph nodes: `0.639409s`, `14.50%`;
- NCCL communication: `0.346671s`, `7.86%`;
- MoE/Marlin: `0.316854s`, `7.19%`.

TARGET 07.61 is therefore an evidence target: freeze this mini baseline,
compare it against vLLM at macro, bucket, owner, and source-dispatch levels,
then select exactly one next implementation target.  Do not start another
cached-weight expansion or generic graph/layout pass before that parity report.

TARGET 07.61 completed without landing a runtime optimization.  It confirmed:

- mini 07.60 remains at `105.7645 output tok/s` on 4096/1024/batch4, about
  `7.28%` below the old `114.07 output tok/s` serving victory line;
- vLLM macro remains much faster, about `202.03 output tok/s` on
  4096/1024/batch4, but the fresh vLLM Nsight repeat window did not contain
  usable CUDA kernels, so vLLM per-bucket timing remains unavailable;
- the strongest mini single-owner boundary is `attn.wo_a`, with `0.481377s`
  owner time, including `0.290148s` copy/layout and `0.137695s` elementwise;
- vLLM has a concrete SM80 source boundary for this owner: per-group BF16 BMM
  weight cache via `_ensure_wo_a_bmm_weight()` and `_apply_wo_a_bmm()`.

TARGET 07.62 should therefore test an opt-in `wo_a` BF16 grouped BMM/cache path.
This target must preserve graph replay, keep eager decode at `0`, record the
new memory cost, and stop if the owner or macro gates do not move.  Do not jump
straight to vLLM's fused inverse-RoPE plus FP8 einsum path unless the final
report recommends it as a separate precision-boundary target.

TARGET 07.62 completed successfully:

- new milestone variant: `dsv4_sm80_a100_victory`;
- compatibility alias: `target0762_woabf16bmmcache`;
- top-level bundle toggle: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`;
- projection-cache sub-bundle: `MINISGL_DSV4_SM80_BF16_PROJECTION_CACHE=1`;
- 4096/128/batch4 improved from `51.2962` to `53.5877 output tok/s`;
- 4096/1024/batch4 improved from `105.7645` to `116.2553 output tok/s`;
- `attn.wo_a` replay owner dropped from `0.481377s` to `0.068948s`;
- graph replay stayed active and eager decode stayed `0`;
- total cached BF16 projection memory is `1.3359 GiB/rank`, about
  `18865.83` KV tokens or `73.69` pages per rank.

The milestone bundle intentionally no longer enables the stale
`Q_WQB/WO_B/INDEXER_WQB_FP8_GEMM` opt-ins because the BF16 projection-cache
paths supersede them in the current best stack.  Keep the individual cache
toggles for ablation because they have explicit memory tradeoffs.

TARGET 07.63 completed the post-victory confirmation and bottleneck reset:

- text smoke passed for `dsv4_sm80_a100_victory`;
- 4096/128/batch4 reached `59.5264 output tok/s`;
- 4096/1024/batch4 reached `119.4153 output tok/s`, `+4.68%` above the old
  `114.07` serving line;
- graph replay stayed active and eager decode stayed `0`;
- the compatibility alias `target0762_woabf16bmmcache` still expands to the
  same env, but new reports should use `dsv4_sm80_a100_victory`;
- the stale `Q_WQB/WO_B/INDEXER_WQB_FP8_GEMM` opt-ins are inactive in the
  current best path;
- a config-path fix made `DeepSeekV4KVCache` use
  `dsv4_env_flag(MINISGL_DSV4_SM80_INDEXER_FP8_CACHE)` so bundle expansion and
  FP8 indexer side-cache allocation agree.

The fresh 4096/128/batch4 profile selected the next implementation target:

| Bucket | Kernel s | Share | Decision |
| --- | ---: | ---: | --- |
| `graph_runtime_copy_cat_index` | `0.846795` | `21.48%` | selected |
| `projection_gemm` | `0.812100` | `20.60%` | hold; diffuse after cached BF16 projection wins |
| `elementwise_graph_nodes` | `0.497965` | `12.63%` | secondary validation only |
| `nccl_communication` | `0.340015` | `8.62%` | hold |
| `moe_marlin` | `0.300516` | `7.62%` | hold |

TARGET 07.64 should implement a narrow decode metadata deforestation pass for
`graph_runtime_copy_cat_index`.  The vLLM source-parity references are
`compute_global_topk_indices_and_lens`, `combine_topk_swa_indices`, and
`flat_index_dequant_gather_blocked` under
`/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`.
Do not continue into projection, communication, MoE, broad graph cleanup, or
full FP8 KV cache inside 07.64.

TARGET 07.64 completed with a correct opt-in helper but did not clear the
promotion gate:

- new opt-in variant: `dsv4_sm80_a100_victory_metadatadeforest`;
- new toggle: `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`;
- helper microbench showed `6.8x-8.5x` local metadata construction speedup;
- text smoke passed, graph replay stayed active, eager decode stayed `0`;
- 4096/1024/batch4 improved from `119.4153` to `122.9414 output tok/s`
  (`+2.95%`), below the `+5%` gate;
- `graph_runtime_copy_cat_index` moved only from `0.846795s` to `0.834792s`
  (`-0.012003s`), far below the `-0.25s` gate;
- the optimized source-owned `batch_prepare:decode:bs4` metadata slice dropped
  from `0.019838s` to `0.005991s`, but this slice was too small to move the
  overall bucket;
- the large remaining direct-copy surface is under
  `batch_forward:decode:bs4:padded4` and
  `batch_forward_enqueue:decode:bs4:padded4`.

Therefore keep the 07.64 helper as an opt-in ablation and do not add it to
`dsv4_sm80_a100_victory`.  TARGET 07.65 should be measurement-only direct-copy
owner attribution.  It may add profiling-only NVTX and classifier scripts, but
must not implement a performance optimization.

TARGET 07.65 completed that attribution pass.  It added default-off
profiling-only direct-copy NVTX and classifier support, then mapped replay
direct-copy kernels through CUDA graph `originalGraphNodeId`.  The result
assigned `99.97%` of the 4096/128/batch4 rank0 `direct_copy` bucket to named
owners:

- total direct_copy: `0.737039s`;
- named owner direct_copy: `0.736794s`;
- residual: `0.000245s`;
- MoE/shared expert staging: `0.379204s`, `51.45%`;
- attention/indexer boundary: `0.138539s`, `18.80%`;
- graph/replay metadata: only `0.000290s`, `0.04%`.

The dominant individual owners are `dsv4.shared_experts.gate_up_proj`
(`0.165751s`) and `dsv4.shared_experts.down_proj` (`0.119724s`).  The 07.64
metadata opt-in did not change this shape, so TARGET 07.66 should focus on
MoE/shared-expert projection and finalization staging.  It should stay on the
current bf16/exact route; INT8 MoE belongs in a separate future target with
independent quality and backend gates.

TARGET 07.66 completed and promoted the shared expert BF16 weight cache:

- new promoted toggle:
  `MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1`;
- `dsv4_sm80_a100_victory` now includes the shared expert cache;
- `dsv4_sm80_a100_victory_sharedbf16` remains as an explicit audit variant;
- shared expert gate/up/down projection owners disappeared from the
  direct-copy table;
- total direct_copy fell from `0.737039s` to `0.449052s`;
- MoE/shared staging fell from `0.379204s` to `0.097361s`;
- 4096/128/batch4 improved from `59.5264` to `62.2034 output tok/s`;
- 4096/1024/batch4 improved from `119.4153` to `131.7707 output tok/s`;
- incremental cache cost is `270,532,608 bytes/rank` (`0.251953 GiB/rank`),
  about `14.01` KV pages/rank.

The remaining direct-copy owners are diffuse.  Do not automatically continue
into runner finalization just because it remains visible; TARGET 07.67 should
first reprofile the promoted post-07.66 path and reset the whole bottleneck
order.

TARGET 07.67 completed that reset without runtime changes:

- promoted `dsv4_sm80_a100_victory` reproduced the 07.66 audit variant within
  normal noise;
- 4096/128/batch4 reached `62.1364 output tok/s`;
- 4096/1024/batch4 reached `131.6263 output tok/s`;
- graph replay stayed active and eager decode stayed `0`;
- total direct_copy stayed stable at `0.449200s`;
- shared expert projection owners remained absent.

Fresh 4096/128/batch4 rank0 decode-envelope buckets:

| Bucket | Kernel s | Share | Decision |
| --- | ---: | ---: | --- |
| projection/GEMM | `0.778887` | `26.31%` | largest bucket, but likely backend/precision work |
| direct-copy/layout | `0.557626` | `18.84%` | broad and owner-diffuse after 07.66 |
| HC/elementwise | `0.536306` | `18.12%` | selected exact cleanup target |
| NCCL communication | `0.338786` | `11.45%` | important, but not top-two |
| MoE routed/backend | `0.300138` | `10.14%` | backend compute, not finalization |

TARGET 07.68 completed the HC / elementwise graph boundary cleanup as an
opt-in experiment:

- new opt-in toggle: `MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1`;
- new opt-in variant: `dsv4_sm80_a100_victory_hccleanup`;
- focused HC microbench improved `hc_pre` from `0.247450 ms` to
  `0.196063 ms` and reduced HC-pre kernel events from `11` to `6`;
- output comparison, unit tests, TP8 text smoke, graph replay, and eager-decode
  gates passed;
- same-run 4096/128 improved `+3.30%`;
- same-run 4096/1024 improved only `+0.73%`;
- HC/elementwise bucket fell only `0.536306s -> 0.519303s`;
- combined HC/elementwise plus HC-owned direct-copy reduction was about
  `0.0533s`, below the `0.15s` profile gate.

Decision: keep the HC cleanup path as an opt-in experiment and do not add it to
`dsv4_sm80_a100_victory`.  It establishes a useful mini-owned MHC boundary,
but it does not move the promoted long-decode path enough.  A stronger HC
target would need to move prenorm/squared-sum closer to the matmul backend or
make an explicit FP32-carrier decision against vLLM's `post/comb` behavior.

TARGET 07.69 completed projection/GEMM backend and owner re-attribution without
runtime changes.  Existing 07.67/07.68 profiles were sufficient; no new profile
or projection-specific NVTX was required.  The key results were:

- projection/GEMM bucket: `0.778887s`;
- named/grouped owner coverage: `98.94%`;
- residual/coarse owner time: `0.008286s`;
- residual `_quantized_linear_fp8_kernel` time: `0.000000s`;
- largest single owner: HC pre linear, `0.178373s`, below the `0.20s`
  single-owner gate;
- selected backend cluster: BF16 small-GEMM + splitK/reduce,
  `0.521619s`, `66.97%` of projection/GEMM;
- FP32/SGEMM small-GEMM cluster: `0.257269s`, useful context but not the
  selected surface.

This proves that TARGET 07.57's old projection conclusion is obsolete.  The
promoted cached BF16 projection stack removed the old FP8 quantized-linear
bottleneck.  The remaining exact-route projection problem is now many
decode-small BF16 GEMMs with fixed backend/launch/splitK overhead spread across
attention WQA/WKV/compress, shared experts, `wo_a`, `q_wqb`, `wo_b`, indexer
projections, and related owners.

The active next target is TARGET 07.70: BF16 small-GEMM backend cluster.  It
should try exact-route backend solutions first: cuBLASLt algorithm and splitK
policy, prepacked/pretransposed BF16 layouts, custom CUTLASS/Triton small-M
kernels, owner-local grouped/fused GEMM where dependencies allow it, and narrow
compile probes only if the backend lanes fail.  Promotion requires a fresh
profile reduction of at least `0.10s` in projection/GEMM or at least `20%` in
the BF16 cluster, plus at least `3%` same-run 4096/1024 macro improvement.

## Precision Policy

Default exact path:

- bf16 activation/cache for attention, indexer, and DSV4 cache state;
- Marlin WNA16 for MXFP4 expert weights;
- cached BF16 dequantized projection weights for the promoted attention,
  indexer, `wo_a`, and shared expert boundaries;
- no activation quantization by default;
- no packed `fp8_ds_mla` KV cache by default;
- no FP8/FP4 indexer cache by default.

Precision/cache experiments are allowed only as explicit opt-in targets.  They
must be compared against the best exact stack and must include quality gates.

## vLLM Reference

Old vLLM framework:

- source root: `/workspace/vllm-dsv4-docker`;
- virtualenv: `/workspace/venvs/vllm-dsv4`;
- DSV4 model:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`;
- DSV4 attention:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`;
- DSV4 attention ops:
  `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`;
- Fused MoE:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`.

Known caveat: vLLM's sm80 sparse prefill reference path has triggered OOM in
this environment.  Do not port it as mini's default.  Study the design, then
adapt only the pieces that make sense for mini.

## Thread Stop Rules

Each subtarget thread must stop when it has achieved its evidence objective,
selected the next target, or shown that its scoped bottleneck is no longer the
best use of time.

Hard stop conditions:

- an implementation subtarget newly clears its stated macro gate and TP8
  page-size-256 text smoke passes; stop and reprofile instead of continuing
  local polishing inside the same thread;
- the target's named bottleneck is no longer in the top two contributors after
  a fresh profile;
- two consecutive implementation cuts produce less than `5%` macro throughput
  gain and less than `10%` improvement in the targeted subgraph;
- the next proposed change is outside the target scope and lacks evidence for
  at least `5%` expected E2E gain;
- correctness is unstable after one focused fix attempt.

Every target README should end with:

- current best milestone/exact result as applicable;
- next target decision;
- do-not-continue condition.
