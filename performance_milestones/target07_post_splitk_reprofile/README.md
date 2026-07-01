# TARGET 07.40: Post-SplitK Reprofile

## Result

TARGET 07.40 is complete as a profiling and decision target.  No model,
kernel, runtime, or default precision behavior was changed.

The current best exact stack remains:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro numbers were reused from TARGET 07.395 because they are the same
variant, TP8 setup, page size `256`, and `--num-pages 128`, and this target did
not change runtime behavior:

| Workload | Output tok/s | Decode tok/s | Prefill tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128/bs4 | `38.9379` | `79.5257` | `2811.1748` | `254` | `0` |
| 4096/1024/bs4 | `68.8097` | `80.0571` | `2822.5816` | `2046` | `0` |

The 4096/1024/bs4 exact line is still below the old serving victory line
`114.07 output tok/s` and the fresh vLLM offline line `201.99 output tok/s`.

## New Profile

The old 07.395 Nsight profile did not enable CUDA graph node tracing, so the
decode replay kernels were mostly hidden behind graph trace activity while
prefill kernels appeared as normal CUDA kernels.  TARGET 07.40 captured a new
rank0 4096/128/bs4 Nsight profile with:

```text
NSYS_CUDA_GRAPH_TRACE=node
```

The node-trace macro under Nsight reported:

| Metric | Value |
| --- | ---: |
| output tok/s | `36.2066` |
| decode tok/s | `74.5391` |
| repeat wall | `14.1411 s` |
| prefill forward wall | `5.8083 s` |
| decode forward envelope wall | `7.0521 s` |
| decode forward NVTX ranges | `127` |
| graph replay count | `254` |
| eager decode count | `0` |

The node-trace SQLite contains `3,006,852` kernel events with a non-null
`graphNodeId`, so it can attribute decode replay work.  It no longer emits the
older `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` table; that is expected for node trace.

One caveat matters: node trace inflates the `cudaGraphLaunch` API duration.
The new node trace reports `2.7254 s` for `127` `cudaGraphLaunch` calls inside
the measured repeat, while the non-node 07.395 trace reclassified with the same
NVTX window reports `0.3266 s`.  Therefore:

- use the node trace for graph-kernel attribution;
- use the non-node trace for graph API overhead sanity;
- do not treat the node-trace `cudaGraphLaunch` duration as real E2E overhead.

## Artifacts

| Path | Contents |
| --- | --- |
| `scripts/summarize_post_splitk_nsys.py` | SQLite classifier with total, repeat, prefill, and decode-envelope sections. |
| `raw/reused_dsv4_target07395_splitk_4096x128_bs4_np128` | Symlink to reused 07.395 4096/128 macro output. |
| `raw/reused_dsv4_target07395_splitk_4096x1024_bs4_np128` | Symlink to reused 07.395 4096/1024 macro output. |
| `raw/dsv4_target0740_nsys_splitk_node_4096x128_bs4_np128` | New node-trace macro-under-Nsight output. |
| `raw/nsys_target0740_splitk_node_4096x128_bs4_np128_rank0.nsys-rep` | New rank0 node-trace Nsight report. |
| `raw/nsys_target0740_splitk_node_4096x128_bs4_np128_rank0.sqlite` | New rank0 node-trace SQLite. |
| `raw/nsys_target07395_splitk_4096x128_bs4_np128_rank0_nonnode.sqlite` | Symlink to prior non-node SQLite used for graph API cross-check. |
| `summaries/nsys_splitk_node_4096x128_bs4_np128_rank0_classified.json` | New node-trace classified summary. |
| `summaries/nsys_splitk_node_4096x128_bs4_np128_rank0_classified.md` | New node-trace classified report. |
| `summaries/nsys_splitk_nonnode_4096x128_bs4_np128_rank0_reclassified.json` | Non-node reclassification for runtime/API cross-check. |
| `summaries/reused_target07_395_macro_summary.json` | Reused 07.395 macro and microbench summary. |
| `summaries/target07_40_post_splitk_decision_summary.json` | Machine-readable decision summary. |

## Attribution

Measured repeat, node trace, 4096/128/bs4:

| Bucket | Kernel s | Repeat wall share | Notes |
| --- | ---: | ---: | --- |
| runtime/copy/cat/index kernels | `2.7523` | `19.46%` | PyTorch direct-copy, cat, index, gather, fill, and similar graph nodes. |
| legacy prefill sparse attention | `2.1044` | `14.88%` | `sparse_attention_kernel` is prefill/extend, not decode split-K. |
| elementwise math graph nodes | `2.0827` | `14.73%` | Many small exact projection, normalization, reduction, and staging kernels. |
| indexer logits/topk/cache | `1.1973` | `8.47%` | Mostly prefill indexer logits plus smaller decode cache/topk work. |
| FP8 projection GEMM | `1.1720` | `8.29%` | Current selective projection GEMM path; not packed FP8 KV/indexer cache. |
| dense linear other | `0.9907` | `7.01%` | Non-MoE dense GEMMs/CUTLASS. |
| HC/RMSNorm/logits/sampling | `0.6279` | `4.44%` | Below top-five. |
| MoE/Marlin route | `0.5835` | `4.13%` | Below top-five. |
| NCCL communication kernels | `0.4779` | `3.38%` | Below top-five on rank0. |
| decode split-K gather/split/combine | `0.1180` | `0.83%` | Not a remaining top bottleneck. |

Prefill forward, node trace:

| Bucket | Kernel s | Prefill wall share | Notes |
| --- | ---: | ---: | --- |
| legacy prefill sparse attention | `2.1044` | `36.23%` | `41` compressed sparse kernels plus `2` SWA-only sparse kernels. |
| indexer logits/topk/cache | `0.9845` | `16.95%` | `_indexer_bf16_logits_kernel` is the top named prefill indexer kernel. |
| runtime/copy/cat/index kernels | `0.7999` | `13.77%` | Mostly non-graph eager prefill copy/index work. |
| elementwise math graph nodes | `0.5982` | `10.30%` | Eager prefill math/reduction helpers. |

Decode forward envelope, node trace:

| Bucket | Kernel s | Decode-envelope wall share | Notes |
| --- | ---: | ---: | --- |
| runtime/copy/cat/index kernels | `1.8949` | `26.87%` | Largest decode-phase kernel bucket. |
| elementwise math graph nodes | `1.4838` | `21.04%` | Many small graph nodes around exact staging/projection work. |
| FP8 projection GEMM | `1.1720` | `16.62%` | Existing selective projection GEMM work. |
| dense linear other | `0.6224` | `8.83%` | Dense GEMM/CUTLASS bucket. |
| NCCL communication kernels | `0.3257` | `4.62%` | Visible but not top-two. |
| MoE/Marlin route | `0.3207` | `4.55%` | Visible but not primary. |
| indexer logits/topk/cache | `0.2128` | `3.02%` | Decode indexer/cache is much smaller than prefill indexer. |
| decode split-K gather/split/combine | `0.1180` | `1.67%` | Split-K decode is no longer the attention bottleneck. |

## Interpretation

The old `sparse_attention` bucket was misleading.  With node trace and correct
NVTX windows, it splits into two very different things:

- legacy `sparse_attention_kernel` is a prefill/extend cost;
- decode split-K gather/split/combine is only `0.1180 s` in the measured repeat.

This confirms the 07.395 microbench conclusion: mini's exact bf16 split-K sparse
decode boundary is no longer the main vLLM gap.  Continuing split-K sparse
decode polish would be poorly targeted.

The strongest remaining exact-path cluster is runtime/copy/cat/index plus
adjacent small elementwise graph nodes.  It is largest in decode, repeats across
graph replay, and is not explained by the packed FP8 KV/indexer cache precision
lane.  The next exact cluster is prefill legacy sparse plus prefill indexer
logits/topk/cache.  Those costs matter especially for 4096/128 and still matter
for 4096/1024 through TTFT/fixed prompt work.

The profile does show vLLM-relevant precision/layout facts from earlier
targets: vLLM uses packed `fp8_ds_mla` KV cache and FP8 indexer cache.  However,
this profile does not prove that packed FP8 KV/indexer layout is now the sole
or top remaining gap.  There is still a large exact runtime/copy/indexer/cache
cluster before opening a default-precision-changing target.

MoE/Marlin and NCCL are visible but not top-two contributors in the rank0
profile.  Graph replay is working (`eager_decode_count=0`); the independent
non-node `cudaGraphLaunch` API cost is about `0.3266 s` for the measured repeat,
so graph API overhead alone is not the top target.  The bigger issue is what is
inside replay: many copy/cat/index and elementwise nodes.

## Decision

Run **TARGET 07.41** next.

Recommended 07.41 focus:

- reduce exact runtime/copy/cat/index graph-node work and metadata staging;
- attack bf16 indexer logits/topk/cache store and prefill cache/indexer costs;
- keep legacy prefill sparse attention in scope as a measured fixed-cost
  contributor;
- preserve bf16 flat KV/indexer cache semantics by default;
- keep TARGET 07.50 reserved for an opt-in packed FP8 KV/indexer cache lane.

Do not run TARGET 07.50 yet.  The packed FP8 KV/indexer cache lane may still be
needed later to chase vLLM's full `deepseek_v4_fp8` macro line, but the current
post-splitK profile still has a large exact runtime/copy/indexer/prefill cluster
that should be addressed first.

## Validation

- No runtime behavior changed.
- New 4096/128/bs4 rank0 node-trace Nsight profile captured successfully.
- 4096/128 and 4096/1024 macro summaries were reused from 07.395 with written
  justification and copied/symlinked into this milestone.
- The classifier now distinguishes:
  - decode split-K gather/split/combine kernels;
  - legacy prefill/extend sparse attention;
  - indexer logits/topk/cache;
  - runtime copy/cat/index kernels;
  - graph replay API overhead;
  - MoE/Marlin and communication buckets.

## Required Ending

Current best exact output tok/s:

- 4096/128/bs4: `38.9379 output tok/s`.
- 4096/1024/bs4: `68.8097 output tok/s`.

New top-five bottleneck ranking:

| Rank | Bucket | Evidence |
| ---: | --- | --- |
| 1 | runtime/copy/cat/index kernels | `2.7523 s` repeat kernel time, `1.8949 s` decode-envelope kernel time. |
| 2 | legacy prefill/extend sparse attention | `2.1044 s` prefill kernel time; absent from decode graph replay. |
| 3 | elementwise math graph nodes | `2.0827 s` repeat kernel time, mostly repeated decode graph-node work. |
| 4 | indexer logits/topk/cache | `1.1973 s` repeat kernel time, mostly prefill plus smaller decode cache/topk work. |
| 5 | FP8 projection GEMM | `1.1720 s` repeat kernel time; current projection GEMM path, not packed FP8 KV/indexer cache. |

Next target selection: **TARGET 07.41**.

Do-not-continue condition: do not continue split-K sparse decode polish unless a
fresh profile makes decode split-K gather/split/combine a top-two contributor
again.
