# TARGET 08.32 DSV4 SM80 CUDA Graph Private-Pool Micro Attribution

## Result

No `>= 2 GiB/rank` concrete micro/partial owner was found.

The largest repeated-owner measurement was the C4 indexer/topk repeated skeleton:
`N=21` captured at `0.111 GiB/rank`.  That collapses the tempting single-owner
projection (`0.105 GiB * 21 = 2.215 GiB`) and shows this path does not add
linearly inside one captured graph pool.

The `18.8-19.0 GiB/rank` first-graph private-pool cost remains unattributed by
these lightweight probes.  The evidence rules out:

- generic empty/PyTorch graph overhead;
- simple captured out-of-place temporaries;
- simple cuBLAS BF16 matmul/cuBLASLt workspace;
- DSV4 synthetic SWA/C4/C128 attention workspace;
- C4 indexer/topk workspace after repeated validation;
- q/kv norm + RoPE + cache-store;
- direct graph metadata/deforest helper;
- 32 MiB/rank TP8 NCCL all-reduce graph communication workspace.

No fix PoC was attempted because every measured repeated graph stayed at
`<= 0.111 GiB/rank`, far below the target's `>= 2 GiB/rank` threshold and far
below the old full-model `18.795-19.037 GiB/rank` line.

## Old Evidence Recap

TARGET 08.06 and 08.07 established the capacity problem:

| Evidence | Result |
| --- | ---: |
| Single `[1]` graph | `18.795 GiB/rank` |
| Single `[16]` graph | `18.828 GiB/rank` |
| `[1,2,4,8,16]` bucket set | `19.037 GiB/rank` |
| Later buckets after first graph | only `~0.05-0.08 GiB/rank` each |
| Graph input buffers | `7.891 MiB`, not material |
| Greedy sampling capture | `0.000 GiB` movement |
| Compressed-location metadata capture | `0.000 GiB` movement |
| `max_seq_len` tested range | `0.000 GiB` movement |
| `num_pages 64 -> 128` | only `~0.035 GiB` movement |
| Disabling tested BF16 caches | removed `1.588 GiB` persistent memory but changed graph delta only `+0.057 GiB` |

Interpretation carried into this target: the big owner is likely the private
pool preserving the captured runtime allocation/workspace shape of the full
decode forward, not KV pages, graph inputs, greedy sampling, direct metadata, or
BF16 projection/shared-expert caches themselves.

## Harness Design

Files:

```text
performance_milestones/target08_cuda_graph_private_pool_micro_attribution/
  README.md
  raw/
  scripts/
    graph_private_pool_micro.py
    distributed_graph_comm_micro.py
    summarize_graph_private_pool_micro.py
  summaries/
    graph_private_pool_micro_summary.json
    graph_private_pool_micro_summary.md
```

The main harness:

- runs every single-GPU case in a fresh Python process in suite mode;
- never loads `/models/DeepSeek-V4-Flash` checkpoint weights;
- uses per-rank DSV4 decode shapes: `hidden=4096`, local Q heads `8`,
  `head_dim=512`, page size `256`, pages `128`;
- warms up once, calls `torch.cuda.empty_cache()`, then captures one
  `torch.cuda.CUDAGraph`;
- records free/allocated/reserved/peak memory, capture/replay elapsed time,
  replay sanity, and explicit input/output/workspace/cache/weight/metadata
  bytes;
- emits raw JSON per case plus Markdown/JSON summaries.

The communication harness uses `torchrun --nproc_per_node=8`, captures one NCCL
all-reduce graph per rank, gathers per-rank memory deltas to rank 0, and also
does not load model weights.

All GiB values below use bytes / `2^30`.

## Commands

Main micro suite:

```bash
cd /workspace/mini-sglang
CUDA_VISIBLE_DEVICES=0 \
python performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/graph_private_pool_micro.py \
  --suite all \
  --output-dir performance_milestones/target08_cuda_graph_private_pool_micro_attribution \
  --cuda-visible-devices 0
```

Communication controls:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/distributed_graph_comm_micro.py \
  --dtype bf16 --elements 16777216 \
  --json-out performance_milestones/target08_cuda_graph_private_pool_micro_attribution/raw/comm_all_reduce_bf16_32mib_tp8.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/distributed_graph_comm_micro.py \
  --dtype fp32 --elements 8388608 \
  --json-out performance_milestones/target08_cuda_graph_private_pool_micro_attribution/raw/comm_all_reduce_fp32_32mib_tp8.json
```

Summary regeneration:

```bash
python performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/summarize_graph_private_pool_micro.py \
  --milestone-dir performance_milestones/target08_cuda_graph_private_pool_micro_attribution
```

Validation:

```bash
python -m py_compile \
  performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/graph_private_pool_micro.py \
  performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/distributed_graph_comm_micro.py \
  performance_milestones/target08_cuda_graph_private_pool_micro_attribution/scripts/summarize_graph_private_pool_micro.py
```

Result: all `63` recorded cases succeeded with graph capture and replay.

## Control Cases

| Case | Free delta GiB | Alloc delta GiB | Reserved delta GiB | Capture s | Explicit MiB | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| empty graph | `0.021` | `0.000` | `0.002` | `0.0103` | `0.0` | Graph floor. |
| copy/staging bs16 | `0.021` | `0.000` | `0.002` | `0.0125` | `0.0` | Static input staging is tiny. |
| elementwise chain bs16 | `0.023` | `0.000` | `0.004` | `0.0116` | `0.1` | Captured elementwise temps are tiny. |
| elementwise prealloc bs16 | `0.023` | `0.000` | `0.004` | `0.0112` | `0.3` | Prealloc does not move the floor. |
| BF16 matmul bs16 | `0.043` | `0.008` | `0.023` | `0.0103` | `32.1` | Single cuBLAS BF16 matmul is small. |
| BF16 matmul prealloc bs16 | `0.041` | `0.008` | `0.021` | `0.0094` | `32.4` | `out=` saves only `~0.002 GiB`. |
| repeated matmul N=43 | `0.043` | `0.008` | `0.023` | `0.0101` | `32.1` | Does not scale with repeated calls. |
| repeated matmul N=43 prealloc | `0.041` | `0.008` | `0.021` | `0.0099` | `32.4` | Same conclusion. |
| repeated matmul N=43 keep-all | `0.047` | `0.008` | `0.027` | `0.0106` | `32.1` | Even forced retained outputs stay tiny. |
| TP8 NCCL all-reduce BF16 32 MiB/rank | `0.002` | `0.000` | `0.002` | `0.0037` | `32.0` | Communication graph workspace is tiny. |
| TP8 NCCL all-reduce FP32 32 MiB/rank | `0.002` | `0.000` | `0.002` | `0.0036` | `32.0` | Same for FP32. |

Control interpretation: the full-model graph cost is not generic
`CUDAGraph` overhead, simple PyTorch allocator behavior, simple out-of-place
temporary retention, simple BF16 matmul/cuBLAS workspace, or NCCL graph
workspace.

## DSV4 Subgraph Probes

| Case | bs | Free delta GiB | Alloc delta GiB | Reserved delta GiB | Explicit MiB | Simple projection | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SWA attention | 1 | `0.043` | `0.008` | `0.023` | `32.0` | `1.848` (`x43`) | Small. |
| SWA attention | 16 | `0.043` | `0.008` | `0.023` | `32.1` | `1.848` (`x43`) | No bs growth. |
| C4 sparse attention/C4A | 1 | `0.043` | `0.008` | `0.023` | `40.0` | `0.902` (`x21`) | Small. |
| C4 sparse attention/C4A | 16 | `0.082` | `0.008` | `0.062` | `40.2` | `1.723` (`x21`) | Below threshold. |
| C128 attention | 1 | `0.043` | `0.008` | `0.023` | `32.3` | `0.859` (`x20`) | Small. |
| C128 attention | 16 | `0.043` | `0.009` | `0.023` | `32.4` | `0.859` (`x20`) | Small. |
| C4 indexer/topk | 1 | `0.043` | `0.008` | `0.023` | `2.0` | `0.902` (`x21`) | Small. |
| C4 indexer/topk | 16 | `0.105` | `0.008` | `0.086` | `2.3` | `2.215` (`x21`) | Single-owner projection crosses threshold, but repeated validation below disproves additivity. |
| q/kv norm + RoPE + cache-store | 1 | `0.023` | `0.000` | `0.004` | `32.0` | `1.008` (`x43`) | Small. |
| q/kv norm + RoPE + cache-store | 16 | `0.023` | `0.000` | `0.004` | `32.1` | `1.008` (`x43`) | Small. |
| metadata/deforest helper | 1 | `0.023` | `0.000` | `0.004` | `0.0` | `0.023` | Small. |
| metadata/deforest helper | 16 | `0.023` | `0.000` | `0.004` | `0.0` | `0.023` | Small. |

The only suspicious single-owner projection was C4 indexer/topk at bs16.  The
targeted repeated test below is the deciding evidence.

## One-Layer / Repeated-Layer Scaling

| Skeleton | N=1 free GiB | Intermediate behavior | Full repeated N free GiB | Readout |
| --- | ---: | --- | ---: | --- |
| BF16 matmul out-of-place | `0.043` | N=2/4/8/16 stayed `0.043` | N=43 `0.043` | No cumulative cuBLAS/private-pool growth. |
| BF16 matmul prealloc | `0.041` | N=2/4/8/16 stayed `0.041` | N=43 `0.041` | Prealloc-output is not a material fix. |
| BF16 matmul keep-all | n/a | n/a | N=43 `0.047` | Forced retained outputs still tiny. |
| attention-only DSV4 ratio pattern | `0.043` | N=4/8/16 plateaued at `0.084` | N=43 `0.088` | Attention workspace below threshold. |
| C4 indexer/topk only | `0.105` | N=2/4/8/16 plateaued at `0.107-0.111` | N=21 `0.111` | Disproves `0.105 * 21` additive projection. |
| projection-only skeleton | `0.043` | N=2/4/8/16 stayed `0.043` | N=43 `0.047` | Projection temporaries/workspace tiny. |
| MoE-only route/topk/expand/reduce skeleton | `0.045` | N=2/4/8 `0.045`, N=16 `0.049` | N=43 `0.055` | Synthetic MoE temps tiny. |
| attention+MLP skeleton | `0.043` | N=2/4/8/16 stayed `0.043` | N=43 `0.049` | Simple full-layer composition tiny. |

Important nuance: the summary includes simple single-layer projections for
triage, but repeated validation is the decision point.  Every repeated full
owner/skeleton stayed at or below `0.111 GiB/rank`.

## Attribution Decision

| Candidate owner | Judgment | Evidence |
| --- | --- | --- |
| PyTorch/CUDA graph allocator as a generic cost | No | Empty graph `0.021 GiB`; elementwise `0.023 GiB`. |
| Captured out-of-place temporaries | No | Repeated matmul N=43 `0.043 GiB`; keep-all `0.047 GiB`; attention/projection/MoE repeated skeletons also tiny. |
| cuBLAS/cuBLASLt workspace | No | BF16 matmul N=1 `0.043 GiB`; N=43 `0.043 GiB`; prealloc-output only saves `~0.002 GiB`. |
| Attention C4/C128/indexer workspace | No as isolated owner | Largest single C4 indexer/topk `0.105 GiB`; repeated C4 indexer/topk N=21 `0.111 GiB`; C4 attention bs16 `0.082 GiB`; attention-only N=43 `0.088 GiB`. |
| Communication workspace | No | TP8 NCCL BF16/FP32 32 MiB/rank graph all-reduce each `0.002 GiB/rank`. |
| Full-model composition | Still plausible but not reproduced | Synthetic projection/MoE/attention+MLP repeated skeletons all `<=0.088 GiB` except C4 indexer repeated `0.111 GiB`. The real full graph may include module/backend-specific allocation paths absent from these synthetic probes. |
| Still unattributed | Yes | No lightweight owner explains even `1 GiB/rank` in repeated validation, let alone `18.8-19.0 GiB/rank`. |

## Stop Reason

This target's materiality rule was not met.  The only apparent `>=2 GiB`
single-owner projection, C4 indexer/topk, was falsified by repeated capture:

```text
single C4 indexer/topk bs16: 0.105 GiB
simple x21 projection:      2.215 GiB
repeated N=21 measurement:  0.111 GiB
```

That means a local PoC such as preallocating topk outputs, moving static staging
outside capture, or reducing one synthetic temporary would chase at most
`~0.1 GiB/rank`, not the `19 GiB` class problem.

## Recommendation

Do not run a broad full-model A/B matrix and do not start a graph/workspace
manager rewrite from this evidence.

Smallest useful next evidence step, if this is reopened:

1. Build a real mini `DeepSeekV4Layer` partial-model capture using synthetic
   weights only, with the actual module call sites/backends but no checkpoint
   load.
2. Add owner-scoped capture-time allocation counters around real full decode
   call sites, especially optional backend wrappers that the synthetic harness
   did not instantiate.
3. Only then run one small full-model confirmation if the real-module partial
   capture finds a `>=2 GiB/rank` owner.

Current recommendation: **defer graph-memory optimization until a real-module
partial or full-model owner is identified**.  The micro evidence rules out the
cheap fixes tested here.
