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

## Current Best Exact Result

Current best exact stack:

- Marlin WNA16 MoE backend;
- global topk/lens;
- bf16 gather/mask plus split-K sparse decode;
- DSV4 decode CUDA graph replay;
- page size 256, `--num-pages 128`;
- TP8 on 8x A100.

Representative variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Best recorded macro from TARGET 07.395:

| Workload | Output tok/s | Note |
| --- | ---: | --- |
| 4096/128/batch4 | `38.94` | `+14.05%` over 07.394 |
| 4096/1024/batch4 | `68.81` | `+24.99%` over 07.394 |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- fresh vLLM offline 4096/1024/batch4: `201.99 output tok/s`;
- vLLM's fast path uses `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, and
  FP8 indexer cache, so it is not precision-neutral.

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
| TARGET 07.52 | `prompts/TARGET_07.52_dsv4_sm80_vllm_fp8_indexer_backend_port.md` | next todo | Port or closely adapt vLLM's FP8 paged indexer backend as an opt-in path; keep exact bf16 default and defer full `fp8_ds_mla` KV cache E2E. |

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

Run TARGET 07.52 next.

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

## Precision Policy

Default exact path:

- bf16 activation/cache for attention, indexer, and DSV4 cache state;
- Marlin WNA16 for MXFP4 expert weights;
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

- official 4096/1024/batch4 output throughput exceeds `114.07 tok/s` and TP8
  page-size-256 text smoke passes;
- the target's named bottleneck is no longer in the top two contributors after
  a fresh profile;
- two consecutive implementation cuts produce less than `5%` macro throughput
  gain and less than `10%` improvement in the targeted subgraph;
- the next proposed change is outside the target scope and lacks evidence for
  at least `5%` expected E2E gain;
- correctness is unstable after one focused fix attempt.

Every target README should end with:

- current best exact result;
- next target decision;
- do-not-continue condition.
