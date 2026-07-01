# TARGET 07.53: Post-FP8-Indexer Reprofile

Date: 2026-07-02

## Result

This target is complete as an evidence and re-ranking target.  No model,
kernel, runtime, or default precision behavior was changed.

Decision: start graph/layout replay deforestation target.

The narrow 07.52 vLLM-aligned FP8 indexer port is still a successful opt-in
path and should be kept.  It is no longer the top remaining bottleneck.  The
fresh mini profile shows the repeated decode graph body is now dominated by:

1. projection/GEMM work: `1.7973 s`, `27.49%` of decode-envelope wall;
2. graph/runtime/copy/cat/index nodes: `1.6170 s`, `24.73%`;
3. elementwise graph nodes: `1.3583 s`, `20.77%`.

The graph/layout cluster (`copy/cat/index + elementwise`) is `2.9752 s`, or
`45.50%` of the decode-envelope wall.  That is the best next target because it
is large, repeated inside graph replay, and more likely to explain the
mini-vs-vLLM topology difference than another isolated indexer slice.

Projection/GEMM is large enough to be the explicit pivot gate for the next
target, but not the first standalone target: the existing vLLM profile is not
complete enough to prove a pure GEMM backend gap, and the top GEMM bucket is
interleaved with many layout/copy/elementwise graph nodes.

## 07.52 Baseline Preserved

TARGET 07.52 ported a vLLM-aligned FP8 indexer backend under the explicit
`MINISGL_DSV4_SM80_INDEXER_FP8_CACHE=1` opt-in guard.  Exact BF16 default
behavior remains unchanged.

07.52 backend microbench, large shape batch16/history4096:

| Piece | Mini FP8 indexer | vLLM isolated | Mini BF16 reference |
| --- | ---: | ---: | ---: |
| Q fold | `0.0870 ms` | `0.0839 ms` | n/a |
| K store | `0.1215 ms` | `0.0964 ms` | n/a |
| Paged logits | `0.1845 ms` | `0.1529 ms` | `0.3516 ms` |
| Logits plus select | `0.2472 ms` | `0.1804 ms` | `0.3709 ms` |

07.52 quality and graph status:

| Check | Result |
| --- | --- |
| Large-shape logits mean/max abs | `0.02147` / `0.14102` |
| Large-shape top-k overlap mean/min | `0.9744` / `0.9648` |
| K dequant mean/max abs | `0.01796` / `0.25` |
| Text smoke | pass |
| CUDA graph smoke | captured `[4, 2, 1]`, replay `9`, eager decode `0` |

07.52 macro:

| Shape | FP8-indexer output tok/s | Decode tok/s | Graph replay |
| --- | ---: | ---: | ---: |
| 4096/128/bs4 | `41.63` | `85.76` | `127` |
| 4096/1024/bs4 | `73.67` | `86.21` | `1023` |

Important retained rule: exact and FP8 variants must be run as separate
processes when CUDA graph capture happens at engine init.  Two-variant
same-process runs are not graph-semantics-fair.

Do not resume local polishing of the old mini-owned FP8 indexer slice, and do
not port standalone `quantize_and_insert_k_cache`.

## Commands Used

Fresh mini FP8-indexer short macro:

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
  --output-dir performance_milestones/target07_post_fp8_indexer_reprofile/raw/macro_4096x128_bs4_np128_fp8_indexer \
  --keep-going
```

Fresh mini rank0 Nsight node-trace profile:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OUTPUT_DIR=/tmp/dsv4_target0753_fp8_indexer_4096x128_bs4_np128_nsys \
NSYS_BASE=/tmp/nsys_target0753_fp8_indexer_node_4096x128_bs4_np128 \
performance_milestones/target07_post_fp8_indexer_reprofile/scripts/nsys_fp8_indexer_4096x128_bs4.sh
```

Classification:

```bash
python performance_milestones/target07_post_fp8_indexer_reprofile/scripts/summarize_fp8_indexer_nsys.py \
  performance_milestones/target07_post_fp8_indexer_reprofile/raw/nsys_target0753_fp8_indexer_node_4096x128_bs4_np128_rank0.sqlite \
  --repeat-nvtx repeat:decode_throughput_bs8:0 \
  --json-out performance_milestones/target07_post_fp8_indexer_reprofile/summaries/nsys_fp8_indexer_node_4096x128_bs4_np128_rank0_classified.json \
  --md-out performance_milestones/target07_post_fp8_indexer_reprofile/summaries/nsys_fp8_indexer_node_4096x128_bs4_np128_rank0_classified.md \
  --top 40
```

vLLM control artifacts were reused from TARGET 07.43 because they are stable
and directly comparable by workload shape:

- `performance_milestones/vllm/raw/dsv4_vllm_ablation_control_4096x128_bs4`
- `performance_milestones/vllm/raw/dsv4_vllm_ablation_control_4096x1024_bs4`

The existing vLLM Nsight profile was reused with caveat:

- `performance_milestones/vllm/raw/nsys_vllm_4096x128_bs4.sqlite`

It contains the requested `repeat:decode_throughput_bs8:0` NVTX range, but the
existing summary finds no child-process CUDA kernels inside that window.  The
full capture reports only `0.982 s` summed visible kernel time for a roughly
`6.32 s` repeat.  Use it for macro/source-topology sanity and visible total
kernel names, not for precise vLLM per-bucket shares.

## Artifacts

| Path | Contents |
| --- | --- |
| `scripts/nsys_fp8_indexer_4096x128_bs4.sh` | Reproducible mini rank0 Nsight launcher. |
| `scripts/nsys_rank_wrapper.sh` | Rank-selective `nsys profile` wrapper. |
| `scripts/summarize_fp8_indexer_nsys.py` | 07.53 bucket classifier. |
| `raw/macro_4096x128_bs4_np128_fp8_indexer` | Fresh non-Nsight mini FP8-indexer macro. |
| `raw/dsv4_target0753_fp8_indexer_4096x128_bs4_np128_nsys` | Symlink to fresh mini macro under Nsight. |
| `raw/nsys_target0753_fp8_indexer_node_4096x128_bs4_np128_rank0.nsys-rep` | Fresh rank0 Nsight report symlink. |
| `raw/nsys_target0753_fp8_indexer_node_4096x128_bs4_np128_rank0.sqlite` | Fresh exported rank0 SQLite symlink. |
| `summaries/nsys_fp8_indexer_node_4096x128_bs4_np128_rank0_classified.*` | New 07.53 classification. |
| `summaries/target07_53_post_fp8_indexer_reprofile_decision_summary.json` | Machine-readable decision summary. |

## Macro Comparison

| Engine/path | Shape | Output tok/s | Decode tok/s | TTFT/elapsed note | Graph |
| --- | --- | ---: | ---: | --- | --- |
| mini exact historical best | 4096/128/bs4 | `38.94` | `79.53` | TARGET 07.395/07.40 | replay, eager `0` |
| mini FP8 indexer, fresh | 4096/128/bs4 | `41.66` | `85.76` | TTFT `6.106 s` | replay `127`, eager `0` |
| mini FP8 indexer, under Nsight | 4096/128/bs4 | `38.53` | `80.66` | node-trace overhead | replay `127`, eager `0` |
| vLLM control | 4096/128/bs4 | `82.28` | n/a | 3-repeat mean from 07.43 | graph sizes `[1,2,4]` |
| mini exact historical best | 4096/1024/bs4 | `68.81` | `80.06` | TARGET 07.395/07.40 | replay, eager `0` |
| mini FP8 indexer, 07.52 | 4096/1024/bs4 | `73.67` | `86.21` | total `55.60 s`, decode dominated | replay `1023`, eager `0` |
| vLLM control | 4096/1024/bs4 | `202.03` | n/a | 07.43 control | graph sizes `[1,2,4]` |

Interpretation:

- FP8 indexer gives a real mini macro gain: 4096/128 is `41.66` vs historical
  exact `38.94`, and 4096/1024 is `73.67` vs `68.81`.
- The remaining vLLM gap is still large: vLLM is `1.98x` faster at 4096/128
  and `2.74x` faster at 4096/1024 versus mini FP8 indexer.
- The 4096/1024 mini FP8 run remains decode dominated.  The next target must
  improve repeated decode graph work, not isolated indexer logits.

## Mini Profile Classification

Fresh mini FP8-indexer rank0 node trace:

| Metric | Value |
| --- | ---: |
| Repeat wall | `13.2881 s` |
| Prefill forward wall | `5.4087 s` |
| Decode forward ranges | `127` |
| Decode forward summed wall | `6.1026 s` |
| Decode forward envelope wall | `6.5390 s` |
| Repeat kernel time | `10.4887 s` |
| Decode-envelope kernel time | `5.9054 s` |

Decode-envelope bucket ranking:

| Rank | Bucket | Kernel s | Decode-envelope wall share | Count | Graph nodes | Representative kernels |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | projection/GEMM | `1.7973` | `27.49%` | `100965` | `795` | `_quantized_linear_fp8_kernel`, `ampere_sgemm_32x32_sliced1x4_tn`, CUTLASS BF16 GEMM |
| 2 | graph/runtime/copy/cat/index | `1.6170` | `24.73%` | `466128` | `3612` | PyTorch direct-copy, bf16/float8 copy, cat/index kernels |
| 3 | elementwise graph nodes | `1.3583` | `20.77%` | `561461` | `4384` | mul/div/clamp/reduce/log2/pow elementwise helpers |
| 4 | NCCL/communication | `0.3313` | `5.07%` | `11176` | `88` | all-reduce f32/bf16 ring kernels |
| 5 | MoE/Marlin | `0.3193` | `4.88%` | `43688` | `344` | Marlin WNA16 and repack/route helpers |
| 6 | sampling/logits | `0.1826` | `2.79%` | `43815` | `345` | HC split/post and sampling/logits helpers |
| 7 | FP8 indexer | `0.1301` | `1.99%` | `20828` | `164` | `_indexer_fp8_paged_logits_kernel`, top-k/indexer helpers |
| 8 | sparse attention decode | `0.1180` | `1.80%` | `21590` | `170` | split-K gather/split/combine |
| 9 | KV/compressor/cache store | `0.0281` | `0.43%` | `8128` | `64` | cache-store/norm/RoPE helpers |

Repeat-level fixed-cost bucket ranking:

| Rank | Bucket | Kernel s | Repeat wall share | Notes |
| ---: | --- | ---: | ---: | --- |
| 1 | graph/runtime/copy/cat/index | `2.1819` | `16.42%` | repeated decode plus prefill layout work |
| 2 | projection/GEMM | `2.1700` | `16.33%` | decode dominated |
| 3 | prefill sparse attention | `2.1298` | `16.03%` | 4096/128 fixed prefill cost |
| 4 | elementwise graph nodes | `1.8123` | `13.64%` | repeated decode graph body |
| 5 | sampling/logits | `0.6327` | `4.76%` | below primary threshold |

Important detail: node trace inflates runtime API timings, especially
`cudaGraphLaunch`, just as in TARGET 07.40.  Use this trace for graph-node
kernel attribution, not for strict graph API overhead.

## vLLM Comparison

| Axis | Mini FP8-indexer evidence | vLLM evidence | Decision impact |
| --- | --- | --- | --- |
| Graph/runtime boundary | Decode graph replay works (`127`, eager `0`), but replay contains `1.6170 s` copy/cat/index plus `1.3583 s` elementwise graph-node kernels in the decode envelope. | vLLM uses compile/CUDA graph sizes `[1,2,4]`, wraps attention in `torch.ops.vllm.deepseek_v4_attention`, and lifts selected projections around the custom-op boundary. Existing vLLM profile is incomplete for per-bucket shares but visibly does not expose a comparable mini-style graph body. | Start graph/layout replay deforestation. The issue is graph contents and layout boundaries, not lack of graph replay. |
| Indexer | FP8 indexer decode bucket is only `0.1301 s`, `1.99%` of decode-envelope wall. 07.52 paged logits is `0.1845 ms`, within `1.21x` of vLLM isolated. | vLLM indexer decode uses `fp8_paged_mqa_logits_triton` and persistent top-k path; 07.43 persistent-topk ablation was not a macro factor. | Do not continue indexer backend polishing. |
| KV/cache path | Mini still stores main DSV4 attention/cache state mostly as BF16, but fresh rank0 decode KV/cache-store bucket is only `0.0281 s`, `0.43%`. | vLLM canonicalizes DeepSeek V4 KV cache to `fp8_ds_mla` and uses fused compressor kernels such as `_fused_kv_compress_norm_rope_insert_sparse_attn`; standalone `quantize_and_insert_k_cache` remains the wrong SM80 target. | Defer fused `fp8_ds_mla` until KV/cache gather/store becomes top-two or a graph/layout PoC proves layout traffic is the real packed-KV symptom. |
| Sparse attention decode | Mini split-K gather/split/combine is `0.1180 s`, `1.80%`. | Earlier vLLM sparse boundary probe was around the same scale as mini exact split-K; vLLM full path uses packed `fp8_ds_mla`. | Do not revisit split-K sparse decode now. |
| Projection/GEMM | Combined projection/GEMM is the largest individual decode bucket at `1.7973 s`, `27.49%`. | vLLM also has projection work (`wq_b`, `wo_a`/`wo_b`, `deepseek_v4_fp8_einsum`), but the existing vLLM profile cannot prove a pure GEMM delta. | Keep as pivot gate, not first standalone target. If graph/layout cleanup leaves intrinsic GEMM top-two, start projection/GEMM next. |
| MoE/Marlin | `0.3193 s`, `4.88%` decode-envelope wall. | vLLM fused/MXFP4/Marlin path is still relevant, but 07.392 and this profile both keep MoE below top-two. | Do not start MoE/Marlin revisit. |
| Communication | NCCL kernels are `0.3313 s`, `5.07%` decode-envelope wall; macro counters are `704` collectives and `139.6 GB` per repeat. | vLLM custom all-reduce and graph registration may matter after compute shrinks, but 07.43 did not identify communication as the macro driver. | Do not start communication target now. |

## Bottleneck Re-ranking

Top-two actionable ranking after FP8 indexer:

| Rank | Candidate | Evidence | Expected first-target gain |
| ---: | --- | --- | --- |
| 1 | graph/layout replay deforestation | `2.9752 s` combined copy/cat/index plus elementwise graph-node time in a `6.5390 s` decode envelope; thousands of graph nodes; vLLM source has custom-op/compile boundaries that avoid exposing the same Python/Torch graph body. | A focused `25-30%` reduction of this cluster is about `0.74-0.89 s` on the 4096/128 measured repeat, roughly `6-7%` output-throughput upside. For 4096/1024, decode dominance makes the same per-token reduction plausibly `5-8%` E2E. |
| 2 | projection/GEMM pivot | `1.7973 s`, `27.49%` of decode envelope; top kernel is `_quantized_linear_fp8_kernel` at `1.1730 s`. | Do this only if graph/layout cleanup fails or leaves intrinsic GEMM top-two. A pure GEMM target needs a clearer vLLM backend comparison first. |

Lower-ranked candidates:

| Candidate | Evidence | Decision |
| --- | --- | --- |
| fused `fp8_ds_mla` KV-cache store/gather | Main mini KV/cache-store bucket is `0.0281 s` in decode envelope; packed KV is real vLLM topology but not top-two in this profile. | Defer. |
| FP8 indexer | `0.1301 s`, `1.99%`; 07.52 backend already passed microbench and macro gates. | Stop. |
| MoE/Marlin | `0.3193 s`, `4.88%`. | Defer. |
| Communication | `0.3313 s`, `5.07%`, visible but not top-two. | Defer until compute/layout shrinks. |

## Next Target Decision

Decision: start graph/layout replay deforestation target.

Minimum PoC:

1. Add finer NVTX or graph-node attribution around the decode replay body to
   split projection/GEMM intrinsic time from layout staging.
2. Remove or fuse one repeated graph-layout subgraph that contributes at least
   `10%` of `graph_runtime_copy_cat_index + elementwise_graph_nodes` decode
   kernel time.  Candidate names include direct-copy, bf16/float8 copy,
   CatArrayBatchedCopy, index kernels, clamp/log2/reduce/pow staging around
   projection/indexer/cache boundaries.
3. Reprofile 4096/128/bs4 FP8-indexer single-variant after the cut, then run
   4096/1024 only if 4096/128 improves at least `5%` or removes at least `10%`
   of the targeted decode graph-node cluster.

Why this should beat alternatives:

- FP8 indexer is no longer high-rank.
- Split-K sparse decode is no longer high-rank.
- KV/cache store/gather is not top-two in the mini profile even though vLLM's
  packed KV path remains a later strategic target.
- MoE and communication are visible but below top-two.
- Projection/GEMM is large, but graph/layout nodes around it are larger as a
  cluster and are the clearest mini-vs-vLLM topology mismatch with current
  evidence.

Do-not-continue condition for the next target:

Stop the graph/layout target if one focused PoC cannot remove at least `10%` of
`graph_runtime_copy_cat_index + elementwise_graph_nodes` decode-envelope kernel
time or produce at least `5%` 4096/128 output-throughput gain.  If that stop
condition is hit and intrinsic projection/GEMM remains top-two, pivot to a
projection/GEMM target with a fresh vLLM profile or targeted vLLM microbench as
the first gate.

Do not continue old mini-owned FP8 indexer work, standalone
`quantize_and_insert_k_cache`, split-K sparse decode polishing, MoE/Marlin
revisit, or communication work unless a fresh profile makes that bucket top-two.

## Current Best Lines

Current best exact result:

- 4096/128/bs4: `38.94 output tok/s`.
- 4096/1024/bs4: `68.81 output tok/s`.

Current best opt-in FP8-indexer result:

- 4096/128/bs4: `41.66 output tok/s` fresh in this target.
- 4096/1024/bs4: `73.67 output tok/s` from 07.52, reused here because the
  fresh 4096/128 single-variant run reproduced the 07.52 short line.
