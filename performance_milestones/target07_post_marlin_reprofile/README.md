# TARGET 07.392: DSV4 SM80 Post-Marlin Reprofile

## Final conclusion

TARGET 07.392 is complete as an evidence-only profiling target.  No broad
optimization was implemented here.

Fresh TP8 macro results validate the TARGET 07.391 mini-owned Marlin WNA16
exact path:

| Engine | Shape | Output tok/s | Decode tok/s | Prefill tok/s | TTFT mean | Graph replay | Unsupported skips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mini Marlin WNA16 exact | 4096/128/bs4 | `33.97` | `61.36` | `2802.10` | `6.55 s` | `254` | `0` |
| mini Marlin WNA16 exact | 4096/1024/bs4 | `54.64` | `61.50` | `2809.04` | `6.53 s` | `2046` | `0` |
| vLLM fresh offline | 4096/128/bs4 | `82.08` | n/a | n/a | n/a | graph sizes `[1,2,4]` | n/a |
| vLLM fresh offline | 4096/1024/bs4 | `201.99` | n/a | n/a | n/a | graph sizes `[1,2,4]` | n/a |

The mini graph replay counts above are the benchmark-reported totals across
warmup plus measured repeat.  The measured-repeat schedule contains 127 decode
steps for 4096/128 and 1023 decode steps for 4096/1024.

The old serving victory line is still `114.07 output tok/s`; fresh mini
4096/1024 is `2.09x` below that line.  Fresh vLLM offline 4096/1024 is `3.70x`
above fresh mini.  The fresh vLLM numbers agree with the older fair artifacts
(`80.90` and `201.87 output tok/s`), so the macro comparison is stable.

The new mini bottleneck is not Marlin expert GEMM.  In the post-Marlin
4096/128 Nsight window, sparse attention is the largest contributor, followed
by metadata/runtime/copy overhead and indexer/cache.  The Marlin WNA16 expert
kernel is only `0.234 s` in a `15.887 s` workload window, about `1.47%` wall
share.  The whole visible MoE bucket is about `2.00%` wall share.

## Artifacts

| Path | Contents |
| --- | --- |
| `scripts/run_mini_marlin_matrix.sh` | Fresh mini Marlin WNA16 macro runner for 4096/128 and 4096/1024. |
| `scripts/nsys_mini_marlin_4096x128_bs4.sh` | Rank-selective mini Nsight runner using `cuda,nvtx,osrt,cublas`. |
| `scripts/run_vllm_matrix_attempt.sh` | Fresh vLLM macro attempt wrapper. |
| `scripts/nsys_vllm_4096x128_bs4_attempt.sh` | vLLM Nsight attempt wrapper using `cuda,nvtx,osrt,cublas`. |
| `scripts/summarize_nsys_post_marlin.py` | SQLite classifier for the TARGET 07.392 categories. |
| `raw/dsv4_target07392_marlin_4096x128_bs4_np128` | Symlink to fresh mini 4096/128 macro output. |
| `raw/dsv4_target07392_marlin_4096x1024_bs4_np128` | Symlink to fresh mini 4096/1024 macro output. |
| `raw/dsv4_target07392_vllm_4096x128_bs4` | Symlink to fresh vLLM 4096/128 macro output. |
| `raw/dsv4_target07392_vllm_4096x1024_bs4` | Symlink to fresh vLLM 4096/1024 macro output. |
| `raw/nsys_marlin_wna16_4096x128_bs4_np128_rank0.sqlite` | Symlink to validated post-Marlin mini Nsight SQLite from TARGET 07.391. |
| `raw/nsys_vllm_4096x128_bs4.sqlite` | Symlink to existing vLLM Nsight SQLite. |
| `summaries/post_marlin_reprofile_summary.json` | Machine-readable macro, Nsight, Amdahl, and decision summary. |
| `summaries/bottleneck_ranking.md` | Ranked Amdahl-style candidate table. |
| `summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0_classified.md` | Mini Nsight category summary. |
| `summaries/nsys_vllm_4096x128_bs4_classified.md` | vLLM Nsight category summary with profile-risk caveat. |

## Run configuration

Common workload:

- TP8, single node, 8x A100-SXM4-80GB, sm80.
- Model: `/models/DeepSeek-V4-Flash`.
- Prompt length `4096`, batch size `4`, decode lengths `128` and `1024`.
- Mini page size `256`, `--num-pages 128`.
- vLLM block size `256`, `max_num_batched_tokens=4096`,
  `enable_chunked_prefill=True`, `max_num_seqs=4`.

Mini variant:

```bash
v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Mini first-use Marlin extension JIT happened during the first fresh 4096/128
run before measured repeat.  The subsequent 4096/1024 graph capture reused the
extension cache and was much shorter.  The measured throughput above excludes
the initialization/JIT phase.

vLLM fresh macro completed.  It emitted a torch compile cache reload warning
for `vllm::mhc_pre`, then recompiled and proceeded successfully.  This is a
fresh-run instability note, not a blocker for the macro comparison.

## Macro comparison

| Shape | mini output tok/s | vLLM fresh output tok/s | vLLM old artifact output tok/s | Fresh vLLM / mini | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 4096/128/bs4 | `33.97` | `82.08` | `80.90` | `2.42x` | Fresh vLLM matches old fair artifact. |
| 4096/1024/bs4 | `54.64` | `201.99` | `201.87` | `3.70x` | Fresh mini validates 07.391 `54.47` line. |

Fresh vLLM effective path differs from mini exact policy:

- vLLM reports `deepseek_v4_fp8` quantization in the engine config.
- vLLM attention uses `fp8_ds_mla` KV cache.
- vLLM indexer uses FP8 indexer cache.
- vLLM MoE uses MXFP4/Marlin on sm80.

Therefore the vLLM macro comparison is fair for user-visible workload shape,
but not precision-neutral.  It is useful for target selection, not a reason to
open TARGET 07.4 inside this target.

## Nsight attribution

Mini Nsight source:
`raw/nsys_marlin_wna16_4096x128_bs4_np128_rank0.sqlite`, window
`repeat:smoke_debug:0`.

vLLM Nsight source:
`raw/nsys_vllm_4096x128_bs4.sqlite`, requested window
`repeat:decode_throughput_bs8:0`.

Important caveat: the vLLM SQLite repeat NVTX range contains no child-process
CUDA kernels in this artifact, while the full capture reports only `0.982 s`
summed kernel time for a `6.32 s` repeat.  This matches the TARGET 07.25
profile blocker.  Use vLLM macro and code-path evidence for comparison; do not
use this vLLM SQLite for per-subgraph kernel shares.

Mini 4096/128 rank0 workload window:

| Category | Kernel duration | Kernel share | Wall share |
| --- | ---: | ---: | ---: |
| sparse attention | `2.110 s` | `35.89%` | `13.28%` |
| indexer/cache | `0.969 s` | `16.48%` | `6.10%` |
| runtime/copy/metadata kernels | `0.890 s` | `15.14%` | `5.60%` |
| HC/RMSNorm/logits/sampling | `0.440 s` | `7.48%` | `2.77%` |
| dense linear other | `0.370 s` | `6.30%` | `2.33%` |
| whole visible MoE bucket | `0.318 s` | `5.40%` | `2.00%` |
| NCCL kernels | `0.163 s` | `2.77%` | `1.03%` |
| Marlin WNA16 expert kernel only | `0.234 s` | top-kernel item | `1.47%` |

Top mini kernels in the window:

| Kernel | Count | Duration |
| --- | ---: | ---: |
| sparse attention | `41` | `2.067 s` |
| `_indexer_bf16_logits_kernel` | `21` | `0.922 s` |
| PyTorch direct-copy unrolled kernel | `1481` | `0.401 s` |
| `_hc_split_pre_kernel` | `86` | `0.356 s` |
| PyTorch direct-copy elementwise kernel | `1649` | `0.290 s` |
| Marlin WNA16 expert kernel | `86` | `0.234 s` |

Runtime/API signals in the same mini window:

- `58132` CUDA kernels and `101896` CUDA runtime calls.
- `127` CUDA graph launches, `0.373 s` runtime API time.
- `37839` memcpy events, `0.087 s` device activity, `2.37 GB` copied.
- `cudaDeviceSynchronize` plus `cudaStreamSynchronize` dominate CPU runtime
  wait time (`13.08 s`).  This is mostly wait-on-GPU and should not be added to
  kernel time as an independent optimization bucket.
- NCCL kernels total `0.163 s`; fresh macro communication counters report
  `1408` collectives and `279.2 GB` across warmup plus repeat, about `704`
  collectives and `139.6 GB` per measured-repeat equivalent.

## Mini vs vLLM boundary comparison

| Axis | mini Marlin WNA16 exact | vLLM fresh path | Decision impact |
| --- | --- | --- | --- |
| Operator boundary | mini has fused wq_a/wkv and q/KV norm/RoPE/cache helpers, but attention metadata, bf16 indexer select, sparse attention, MoE route, shared experts, and reduce are still separate mini boundaries. | vLLM wraps DeepSeek V4 attention in `torch.ops.vllm.deepseek_v4_attention`, with indexer/compressor/KV insert inside the attention implementation and FusedMoE runner integration. | Next target should consolidate attention/indexer/cache/runtime boundaries, not Marlin expert GEMM. |
| Tensor shape | Decode is T=4, hidden=4096, local attention heads=8, head_dim=512, indexer heads=64, index_dim=128, MoE experts=256, topk=6, local intermediate=256. | Same model shape, but vLLM packs KV/indexer cache into byte-oriented fp8 layouts and graph-managed buffers. | Shape parity is close enough; layout and boundary are the gap. |
| Precision lane | mini keeps exact bf16 activation/KV/indexer cache policy with mini-owned WNA16 Marlin for MXFP4 expert weights; no FP8 activation/KV lane was enabled here. | vLLM uses `deepseek_v4_fp8`, `fp8_ds_mla` KV cache, FP8 indexer cache, and MXFP4/Marlin MoE. | Precision is a known macro advantage for vLLM, but current mini profile does not justify TARGET 07.4 yet. |
| Graph capture | mini graph captures batch sizes `[4,2,1]`, captures greedy sampling, and replays decode on the current graph path. | vLLM uses full and piecewise CUDA graphs, capture sizes `[1,2,4]`, separate graph capture stream, and graph-aware custom all-reduce registration. | Study graph/runtime/metadata staging as part of the next attention/cache target. |
| Kernel/runtime count | mini window: `58132` kernels, `101896` runtime calls. mini total capture: `268288` kernels, `662937` runtime calls. | vLLM total capture: `124480` kernels, `1908662` runtime calls, but repeat-window kernel attribution is missing in this SQLite. | vLLM call counts are not subgraph-comparable from current SQLite; use fresh macro and code topology. |
| Streams/overlap | mini path is mostly single current-stream graph replay around attention/MoE/cache metadata. | vLLM overlaps indexer with KV insert/compressor on an aux stream and can overlap shared experts with router/routed work on an aux stream. | Add overlap only after exact dependency and profiling proof; do not speculative-port streams in this target. |
| Communication placement | mini has labeled embedding, row-parallel projection, MoE reduce-once, and lm-head all-gather. Per-repeat equivalent for 4096/1024 is about `704` collectives and `139.6 GB`. | vLLM custom all-reduce supports CUDA graph buffer registration; fresh logs registered `348` then `522` graph addresses. | Communication is structurally important but rank0 NCCL kernel time is not the current top bottleneck. |
| Memory/KV policy | mini `--num-pages 128` allocates `32768` KV tokens, `2.28 GiB` K+V per rank; peak allocated memory is about `45.84 GB`. | vLLM block size 256 reports GPU KV cache size `35956` tokens and available KV memory `44.03 GiB`; KV/indexer cache uses uint8 fp8 layouts. | Memory policy and cache layout are part of the attention/indexer/cache target. |

## Amdahl estimates

Using the mini Nsight window duration `15.887 s`:

| Rank | Candidate | Measured seconds | Wall share | Max E2E gain | Example E2E gain | Interpretation |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | sparse attention | `2.110` | `13.28%` | `15.31%` | `10.25%` if reduced 70% | Top kernel bucket and vLLM has a different packed-cache attention boundary. |
| 2 | metadata/runtime/copy visible overhead | `1.949` | `12.27%` | `13.99%` | `6.54%` if reduced 50% | Includes copy kernels plus launch/memcpy/graph runtime; sync wait excluded as non-additive. |
| 3 | indexer/cache | `0.969` | `6.10%` | `6.49%` | `3.80%` if reduced 60% | Above 5% max threshold and directly tied to vLLM's FP8/cache/indexer layout. |
| 4 | HC/RMSNorm/logits/sampling | `0.440` | `2.77%` | `2.85%` | `1.40%` if reduced 50% | Below target threshold. |
| 5 | dense linear other | `0.370` | `2.33%` | `2.39%` | `1.18%` if reduced 50% | Not a primary target. |
| 6 | whole visible MoE bucket | `0.318` | `2.00%` | `2.04%` | `1.01%` if reduced 50% | Below threshold. |
| 7 | Marlin WNA16 expert GEMM only | `0.234` | `1.47%` | `1.50%` | `0.74%` if reduced 50% | Far below the 10% "keep optimizing expert GEMM" threshold. |
| 8 | NCCL kernels | `0.163` | `1.03%` | `1.04%` | `0.52%` if reduced 50% | Revisit later if compute shrinks. |

The sparse-attention + metadata/runtime/copy + indexer/cache trio is the only
measured cluster that clears the 5% decision rule.  Even perfect Marlin expert
GEMM elimination would not move E2E enough to justify another primary MoE
kernel target.

## Ranked decision

Open a new attention/indexer/cache/runtime target.  The recommended scope is:

- isolate mini sparse attention decode cost versus metadata/copy/runtime cost;
- compare exact bf16 cache layout against vLLM's fp8_ds_mla/FP8 indexer layout
  without changing mini precision policy by default;
- prototype boundary consolidation for replay-time metadata, cache reads,
  indexer output staging, and sparse attention launch structure;
- add profile-complete NVTX ranges around mini attention/indexer/cache stages
  and, if feasible, a vLLM child-process profile that attributes kernels inside
  the repeat window;
- keep communication and MoE hardening as secondary observations unless new
  evidence changes their share.

Do not implement that optimization in TARGET 07.392.

## Required ending

- next target: **TARGET 07.393 DSV4 SM80 attention/indexer/cache runtime rework**.
- do not continue here unless... a fresh Nsight capture contradicts the current
  ranking, or vLLM Nsight becomes profile-complete enough to replace the risky
  existing vLLM kernel-attribution artifact.
- MoE hardening: **side quest**, not the primary bottleneck.  Marlin WNA16
  expert GEMM is below `10%` of the workload window, and the whole visible MoE
  bucket is below the `5%` Amdahl threshold for a primary target.
- precision lanes: **continue deferred**.  vLLM-only precision/cache behavior is
  real and recorded, but this profile still points first to
  attention/indexer/cache/runtime structure rather than a precision-only gap.
