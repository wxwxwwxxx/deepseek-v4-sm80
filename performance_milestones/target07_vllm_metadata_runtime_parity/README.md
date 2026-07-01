# TARGET 07.42: vLLM Metadata/Runtime Parity Evidence

## Status

Evidence report complete.  No new mini-sglang optimization was implemented.

TARGET 07.41 already proved that replay metadata copy was a real microcut but
not a macro win:

| Workload | 07.40 baseline | 07.41 metacopy | Delta |
| --- | ---: | ---: | ---: |
| 4096/128/bs4 | `38.9379 output tok/s` | `39.0028 output tok/s` | `+0.17%` |
| 4096/1024/bs4 | `68.8097 output tok/s` | `68.6314 output tok/s` | `-0.26%` |

That result blocks more local metadata-copy polish.  The remaining question is
whether a vLLM core mechanism can explain enough of the gap to justify an
opt-in proof-of-concept.  The answer from this pass is no for exact bf16
runtime work, and yes for opening the opt-in precision/cache lane.

## Inputs Used

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`
- `prompts/TARGET_07.41_dsv4_sm80_indexer_cache_runtime_exact.md`
- `performance_milestones/target07_post_splitk_reprofile/README.md`
- `performance_milestones/target07_indexer_cache_runtime_exact/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`

Additional artifacts created here:

- `summaries/vllm_node_trace_attempt.json`
- `raw/dsv4_target0742_nsys_vllm_node_4096x128_bs4`

## Comparability

Directly comparable:

- mini exact bf16 sparse-only decode after split-K: `0.2284 ms`;
- vLLM prior gather + split-K decode probe: `0.2258 ms`;
- macro lines for the same 4096/128/bs4 and 4096/1024/bs4 shapes.

Not directly comparable:

- vLLM per-bucket runtime/indexer/cache timing.  The older vLLM Nsight artifact
  has no repeat-window child-process kernel attribution.
- A fresh 07.42 vLLM node-trace attempt passed the workload
  (`80.0348 output tok/s` for 4096/128/bs4), but `nsys profile` waited
  indefinitely for re-parented worker processes and did not write `.nsys-rep`
  or `.sqlite`.  Use this attempt only as macro/code-path evidence.

vLLM code-path observations from the fresh run and source:

- engine quantization was `deepseek_v4_fp8`;
- `fp8_ds_mla` KV cache was selected;
- FP8 indexer cache was selected;
- MXFP4 MoE used Marlin;
- CUDA graph capture used piecewise and full decode sizes `[1, 2, 4]`.

## Parity Table

| Mini bucket | Mini source/kernels | Mini measured cost | vLLM mechanism | vLLM measured/code evidence | Precision/layout dependency | Decision |
| --- | --- | ---: | --- | --- | --- | --- |
| Runtime/copy/cat/index graph nodes | `GraphRunner._replay_to_buffer` calls `buffer.copy_from`, `prepare_for_replay`, then graph replay in `python/minisgl/engine/graph.py`; DSV4 metadata copy lives in `DSV4AttentionBackend._copy_metadata_for_replay`; top kernels are PyTorch `direct_copy`, `index_elementwise`, `cat`, `fill` style graph nodes.  07.41 added `_copy_decode_metadata_for_replay_kernel` / `copy_decode_metadata_for_replay`. | `2.7523 s` repeat, `1.8949 s` decode-envelope.  Metacopy microbench `0.2536 -> 0.1272 ms`, `18 -> 1` launches, but macro `+0.17% / -0.26%`. | Persistent runner buffers, `CudagraphDispatcher`, graph-owned batch descriptors, custom-op-owned attention buffers, async output-copy stream. | vLLM has persistent `input_ids`, `positions`, `seq_lens`, `req_indices`, etc. in `gpu_model_runner.py`; `CudagraphDispatcher` owns valid full/piecewise graph keys; `deepseek_v4_attention` mutates preallocated `out`.  Fresh node-trace workload passed at `80.0348 tok/s`, but profile export failed, so no per-bucket timing. | Exact-portable subset was already tried by 07.41 and missed macro.  Broader vLLM graph/runtime ownership is real but not isolated to a >=5% exact-bf16 gain. | `defer` |
| Elementwise math graph nodes | Broad exact attention/model staging in `python/minisgl/models/deepseek_v4.py`: q/KV RMSNorm, q_wqb, q_norm_rope, kv_norm_rope, indexer/compress/store, o_rope, wo_a/wo_b.  Profile top kernels include vectorized add/mul/clamp, bf16/float8 copies, reductions. | `2.0827 s` repeat, `1.4838 s` decode-envelope. | vLLM lifts q/KV RMSNorm and wq_b around a custom-op boundary, uses compiled regions and custom ops to keep some work graph-visible while fusing/owning attention internals. | `DeepseekV4MLAModules.forward` calls fused q/KV RMSNorm, wq_b, then `torch.ops.vllm.deepseek_v4_attention`; inside `attention_impl`, vLLM overlaps indexer with kv_insert/compressor and calls MLA attention.  No fresh per-bucket timing. | Mostly exact-portable in principle, but it is a broad fusion/compile-boundary project.  Current evidence does not identify one top-two exact cut with >=5% E2E. | `defer` |
| Legacy prefill/extend sparse attention | `DSV4AttentionBackend._sparse_attention_two_source` dispatches decode to split-K only when `max_seqlen_q <= 1`; prefill/extend keeps `dsv4_sparse_attention_two_source_bf16` / `sparse_attention_kernel`. | `2.1044 s` repeat and prefill; `0.0000 s` decode-envelope. | vLLM sm80 reference decode uses gathered selected indices plus split-K; sparse prefill/indexer path is tied to vLLM metadata and packed cache and has prior OOM risk in this environment. | vLLM `_forward_decode` uses `compute_global_topk_indices_and_lens`, `gather_dequant_two_scopes_with_mask`, then `_dsv4_sm80_sparse_attn_decode_triton`; previous notes warn not to port vLLM sm80 sparse prefill as default.  No reliable vLLM prefill timing. | Decode mechanism already adapted in mini.  Prefill path is not proven portable and may be memory-risky. | `defer` |
| Indexer logits/topk/cache | `DSV4AttentionBackend.select_indexer` calls `indexer_select_bf16_fallback`; mini logits are `_indexer_bf16_logits_kernel` over bf16 flat indexer cache, then `topk_transform_512_full_fallback`.  Store path uses `store_indexer` / `compress_norm_rope_store_fallback`. | `1.1973 s` repeat, `0.9845 s` prefill, `0.2128 s` decode.  Prior microbench: `mini_indexer_select_bf16=0.3394 ms`, `mini_compressed_indexer_cache_store=0.9762 ms`. | `SparseAttnIndexer` custom op over FP8 indexer cache; fused Q/RoPE/quant; `fp8_paged_mqa_logits_triton`; persistent `topk_indices_buffer`; `persistent_topk`. | vLLM code uses FP8 path with q scale folded into weights, `fp8_paged_mqa_logits_triton` for decode, `persistent_topk` for topk512, and a model-level reusable `topk_indices_buffer`.  Prior microbench measured `vllm_indexer_q_rope_quant=0.0871 ms`, but full vLLM SparseAttnIndexer timing was engine-bound and not directly comparable. | Strongly precision/layout dependent: FP8 indexer cache and paged logits are the mechanism.  Direct adoption would change mini's exact default. | `precision-target` |
| FP8 projection GEMM | Current mini exact stack uses projection FP8 GEMM through `quantized_linear_ref` / `_quantized_linear_fp8_kernel`; attention projection sites include q_wqb and wo_b in `python/minisgl/models/deepseek_v4.py`. | `1.1720 s` repeat and decode-envelope. | vLLM uses quantized projection kernels and compile/custom-op boundaries, but this is not the packed KV/indexer-cache mechanism. | vLLM fresh run selected `deepseek_v4_fp8` and Marlin-family kernels; old vLLM profile total had FP8/copy kernels, but repeat window is not attributable. | Not metadata/runtime/indexer/cache parity.  Already part of mini exact stack as selective projection work. | `defer` |
| Decode split-K gather/split/combine sanity | `dsv4_sparse_attention_two_source_splitk_bf16` dispatches Triton `sparse_attention_splitk_bf16`; legacy bf16 path retained for A/B. | `0.1180 s` repeat, `0.1180 s` decode-envelope.  Microbench sparse-only `0.2284 ms`; globaltopk+indexer+sparse `0.4350 ms`. | vLLM sm80 reference decode: `compute_global_topk_indices_and_lens -> gather_dequant_two_scopes_with_mask -> _dsv4_sm80_sparse_attn_decode_triton`. | vLLM prior combined gather+split-K decode probe `0.2258 ms`; mini sparse-only is effectively parity. | No remaining top bottleneck.  vLLM gather reads packed FP8 cache, but the exact bf16 split-K boundary is already close. | `defer` |
| MoE/Marlin and NCCL sanity | Mini Marlin WNA16 MoE and NCCL buckets from 07.40 profile. | MoE/Marlin `0.5835 s` repeat; NCCL `0.4779 s` repeat. | vLLM uses MXFP4/Marlin-family MoE and graph-registered communication buffers. | Fresh vLLM log selected Marlin MXFP4; no repeat-window timing.  Both buckets are below mini top five/top two. | Not current metadata/runtime/indexer/cache target. | `defer` |

## Interpretation

The top mini bucket after split-K is still runtime/copy/cat/index plus adjacent
small elementwise nodes, but 07.41 already attacked the clean exact-portable
metadata-copy subgraph and got no macro gain.  That means the broad bucket is
not actionable as another local metadata microcut.

The strongest code-level vLLM mechanisms that still differ are:

- packed `fp8_ds_mla` KV cache: vLLM stores most of each 512-dim token as FP8
  bytes plus scales and keeps the RoPE tail bf16;
- FP8 indexer cache and paged FP8 logits;
- `SparseAttnIndexer` as a custom-op boundary with persistent topk workspace;
- attention custom-op ownership with aux-stream overlap of indexer versus
  kv_insert/compressor;
- V1 graph dispatcher and persistent runner buffers.

Only the first two are both strongly evidenced and large enough to plausibly
explain a meaningful part of the remaining macro gap.  They are precision/layout
changes and therefore belong in an opt-in precision/cache target, not in the
exact bf16 default.

The strongest unproven suspicion is vLLM's aux-stream/custom-op overlap around
indexer, compressor, and cache insert.  The code path is clear, but this pass
did not obtain a reliable vLLM node profile, and mini has no isolated timing
showing that porting overlap alone will exceed `5%` E2E while preserving exact
bf16 layout.

## Optional PoC Decision

Do not implement a 07.42 PoC.

Reason:

- the only exact runtime/buffer ownership cut already tested was 07.41
  metacopy, and it did not reach the `5%` macro bar;
- vLLM's best-supported remaining mechanisms are packed FP8 KV/indexer cache
  and FP8 paged indexer logits, which would change mini's default exact bf16
  precision policy;
- no fresh vLLM node-trace profile was available to prove a standalone
  exact-bf16 aux-stream/custom-op or dispatcher port would clear `5%` E2E.

## Next Target

Start **TARGET 07.50** as an explicit opt-in precision/cache experiment.

Recommended scope:

- add an opt-in packed `fp8_ds_mla`-style KV/cache lane or the narrowest
  equivalent needed to exercise vLLM's cache/indexer layout advantage;
- include FP8 indexer cache and paged FP8 logits in the experiment, because the
  indexer bucket is the clearest remaining cache/layout mismatch;
- keep the exact bf16 stack as default and baseline;
- require 4096/128 and 4096/1024 macro checks plus quality/correctness gates;
- continue to treat vLLM aux-stream/custom-op overlap as a secondary suspicion
  until a reliable node-trace profile attributes it.

Do not continue:

- replay metadata-copy polish;
- split-K sparse decode polish;
- MoE/Marlin work;
- exact-bf16 prefill sparse rewrites based only on vLLM sm80 prefill topology;
- broad graph/runtime rewrites without a fresh profile showing a top-two bucket
  and at least `5%` expected E2E.

## Required Ending

Current best exact result:

- 4096/128/bs4: `38.9379 output tok/s`.
- 4096/1024/bs4: `68.8097 output tok/s`.

Strongest proven gap source:

- vLLM's fast DSV4 path is a different precision/layout lane:
  `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, and FP8 indexer cache.
  Mini's exact path remains bf16 flat cache.  Exact sparse decode parity plus
  the failed 07.41 macro promotion make this the strongest evidence-backed
  next gap source.

Strongest unproven suspicion:

- vLLM's attention custom-op + aux-stream overlap + V1 graph dispatcher may
  hide part of mini's runtime/copy/elementwise bucket, but a fresh node-trace
  profile did not export, so this is code-topology evidence only.

Next target recommendation:

- run TARGET 07.50, opt-in packed FP8 KV/indexer-cache parity, with `>=5%`
  macro gain required before any promotion discussion.

TARGET 07.50:

- should start now, as an opt-in precision/cache target; it should not change
  the default exact bf16 policy.
