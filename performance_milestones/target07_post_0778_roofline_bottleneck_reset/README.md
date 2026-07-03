# TARGET 07.79: Post-07.78 Roofline And Bottleneck Reset

Date: 2026-07-03

Status: complete.  No runtime kernel, precision, graph, NCCL, or bundle change
was made.

Decision: start TARGET 08 radix prefix cache.  The current default
`dsv4_sm80_a100_victory` path is stable under the fixed benchmark capacity, but
the remaining exact-speed surfaces are fragmented or need broader risky work.
Prefix reuse can skip the measured multi-second prefill surface on shared-prefix
workloads.

## Scope

Measured path:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

This report intentionally does not revisit
`dsv4_sm80_a100_victory_densefp8marlinproj` as a speed candidate.  TARGET 07.78
kept it as an explicit memory-oriented opt-in.

Primary artifacts:

- `raw/`: torchrun logs, perf matrix JSON, smoke JSON, hardware captures, and
  automatic capacity probe output.
- `summaries/post0778_roofline_summary.json`: computed numeric ledger.
- `summaries/post0778_roofline_summary.md`: generated repeat/MFU snippets.
- `scripts/`: exact runner, capacity probe, and summarizer.

## Commands

Main command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
performance_milestones/target07_post_0778_roofline_bottleneck_reset/scripts/run_default_retest.sh
```

The runner expands the primary macro shape to:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_0778_roofline_bottleneck_reset/raw/4096x1024_victory \
  --keep-going
```

The short macro is identical except `--decode-len 128` and
`raw/4096x128_victory`.

The owner-timing pass is separate and is not used for throughput:

```bash
MINISGL_DSV4_OWNER_TIMING=1 \
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=60000 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_0778_roofline_bottleneck_reset/raw/timing_4096x128_owner_victory \
  --keep-going
```

Automatic KV sizing probe:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target07_post_0778_roofline_bottleneck_reset/scripts/capacity_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --output performance_milestones/target07_post_0778_roofline_bottleneck_reset/raw/capacity_auto_victory/capacity_probe.json \
  --tensor-parallel-size 8 \
  --page-size 256 \
  --memory-ratio 0.9 \
  --max-running-req 4 \
  --max-seq-len 5120 \
  --max-extend-tokens 4096 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 \
  --cuda-graph-capture-greedy-sample
```

The capacity probe sets `cuda_graph_capture_fail_open=True` only to preserve the
capacity ledger after graph capture OOM.  It is not a valid throughput or graph
replay run.

## Repository And Hardware

| Item | Value |
| --- | --- |
| Workspace | `/workspace/mini-sglang` |
| Branch | `dsv4-sglang-based` |
| Commit | `fad8bce325bf4c4bd48d47ad00303b67f2e8f93b` |
| Git status at hardware capture | only this milestone directory untracked |
| Model | `/models/DeepSeek-V4-Flash` |
| GPUs | 8x `NVIDIA A100-SXM4-80GB` |
| Memory | `81920 MiB` per GPU |
| CUDA capability | `sm80` |
| Driver / CUDA from `nvidia-smi` | `570.172.08` / `12.8` |
| PyTorch | `2.9.1+cu128` |
| NCCL | `2.27.5` |
| Python | `3.12.3` |
| A100 form factor assumption | SXM, not PCIe |

Roofline assumptions:

| Assumption | Per GPU | TP8 aggregate |
| --- | ---: | ---: |
| BF16 Tensor Core peak | `312 TFLOP/s` | `2496 TFLOP/s` |
| TF32 Tensor Core peak | `156 TFLOP/s` | `1248 TFLOP/s` |
| FP32 CUDA core peak | `19.5 TFLOP/s` | `156 TFLOP/s` |
| HBM bandwidth | `2.039 TB/s` | `16.312 TB/s` |
| BF16 TC / HBM crossover | `153 FLOPs/byte` | same threshold |

## Text Smoke

TP8 text smoke passed for `dsv4_sm80_a100_victory`.

| Gate | Result |
| --- | --- |
| Status | `pass` |
| Sane outputs | `3/3` |
| Graph replay | `9` |
| Greedy sample replay | `9` |
| Eager decode | `0` |
| Captured graph sizes | `[4, 2, 1]` |
| Unexpected fallback / disabled promoted feature | none observed |
| Dense FP8 Marlin projection default | `enabled=false` |

Smoke outputs were sane for the arithmetic, sky-color, and Hangzhou prompts.
Raw output is in `raw/smoke_dsv4_sm80_a100_victory/`.

## Stable Macro Results

All macro runs used TP8, page size `256`, `--num-pages 128`,
`--warmup-repeats 1`, and `--repeats 3`.  Warmup is excluded from every measured
statistic.

| Shape | Warmup elapsed s |
| --- | ---: |
| `4096x1024` | `34.1310` |
| `4096x128` | `11.2609` |

### 4096/1024, Batch 4

| Repeat | Output tok/s | Decode tok/s | TTFT s | Prefill fwd s | Decode fwd s | Elapsed s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `131.7518` | `169.6495` | `4.9473` | `4.2243` | `24.1203` | `31.0888` |
| 1 | `131.7225` | `169.6440` | `4.9561` | `4.2435` | `24.1211` | `31.0957` |
| 2 | `131.7940` | `169.9815` | `4.9616` | `4.2380` | `24.0732` | `31.0788` |

| Metric | Mean | Median | Best | Worst | Std | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output tok/s | `131.7561` | `131.7518` | `131.7940` | `131.7225` | `0.0360` | `0.0273%` |
| Decode tok/s | `169.7583` | `169.6495` | `169.9815` | `169.6440` | `0.1933` | `0.1139%` |
| TTFT s | `4.9550` | `4.9561` | `4.9473` | `4.9616` | `0.0072` | `0.1449%` |
| Prefill fwd s | `4.2353` | `4.2380` | `4.2243` | `4.2435` | `0.0099` | `0.2331%` |
| Decode fwd s | `24.1049` | `24.1203` | `24.0732` | `24.1211` | `0.0274` | `0.1138%` |
| Elapsed s | `31.0877` | `31.0888` | `31.0788` | `31.0957` | `0.0085` | `0.0273%` |

Aggregate across the three measured repeats:

| Field | Value |
| --- | ---: |
| Prompt tokens | `49152` |
| Actual output tokens | `12288` |
| Decode tokens in decode phase | `12276` |
| Prefill forward s | `12.7796` |
| Decode forward s | `72.5736` |
| Prefill prepare s | `2.2093` |
| Decode prepare s | `5.5096` |
| Elapsed s | `93.2640` |
| Peak allocated/rank | `47,565,686,784` bytes |
| Peak reserved/rank | `49,855,594,496` bytes |
| KV cache/rank | `2,491,495,680` bytes |

### 4096/128, Batch 4

| Repeat | Output tok/s | Decode tok/s | TTFT s | Prefill fwd s | Decode fwd s | Elapsed s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `62.3537` | `168.3407` | `4.9446` | `4.2393` | `3.0177` | `8.2112` |
| 1 | `62.3411` | `168.7250` | `4.9535` | `4.2549` | `3.0108` | `8.2129` |
| 2 | `62.4825` | `169.4417` | `4.9486` | `4.2481` | `2.9981` | `8.1943` |

| Metric | Mean | Median | Best | Worst | Std | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output tok/s | `62.3925` | `62.3537` | `62.4825` | `62.3411` | `0.0783` | `0.1254%` |
| Decode tok/s | `168.8358` | `168.7250` | `169.4417` | `168.3407` | `0.5588` | `0.3310%` |
| TTFT s | `4.9489` | `4.9486` | `4.9446` | `4.9535` | `0.0045` | `0.0899%` |
| Prefill fwd s | `4.2474` | `4.2481` | `4.2393` | `4.2549` | `0.0079` | `0.1849%` |
| Decode fwd s | `3.0089` | `3.0108` | `2.9981` | `3.0177` | `0.0099` | `0.3307%` |
| Elapsed s | `8.2061` | `8.2112` | `8.1943` | `8.2129` | `0.0103` | `0.1254%` |

Graph gate:

| Shape | Replay | Greedy sample replay | Eager decode | Captured BS |
| --- | ---: | ---: | ---: | --- |
| `4096x1024` | `4092` | `4092` | `0` | `[4, 2, 1]` |
| `4096x128` | `508` | `508` | `0` | `[4, 2, 1]` |

## Context And Memory Capacity Ledger

### Fixed Benchmark Mode

Fixed benchmark setting:

```text
--num-pages 128
page_size 256
logical KV token capacity = 128 * 256 = 32768 tokens per TP rank
```

The allocated pool includes one dummy page, so the measured pool memory is
`2,491,495,680` bytes/rank for `129` physical pages.  Dividing that measured
pool by the logical benchmark capacity gives `76,034.41` bytes/logical
token/rank.

| Workload | Prompt | Decode | Batch | Live tokens | Capacity use |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4096x1024` | `4096` | `1024` | `4` | `20480` | `62.5%` |
| `4096x128` | `4096` | `128` | `4` | `16896` | `51.6%` |

Under `np128`, the benchmark is not close to page exhaustion.  The fixed pool
can hold `8` independent 4096-token prompts or `6` independent
4096+1024-token requests by raw page capacity.  The perf matrix still constrains
the long run to `max_seq_len=5120` and `max_running_req=4`.

### Automatic KV Sizing Mode

The automatic probe left `--num-pages` unset with `memory_ratio=0.9`, page size
`256`, `max_seq_len=5120`, `max_extend_tokens=4096`, and graph sizes `[1,2,4]`.

| Field | Value |
| --- | ---: |
| Free before loading model | `78.82 GiB/rank` |
| Free after model + persistent prepare, before KV | `57.85 GiB/rank` |
| Free after auto KV allocation, before graph capture | `7.81 GiB/rank` |
| Chosen `num_pages` | `2778` |
| Logical KV token capacity | `711168` |
| KV bytes/page/rank | `19,313,920` |
| Logical KV bytes/rank | `53,654,069,760` bytes (`49.97 GiB`) |
| Pool pages including dummy | `2779` |
| Engine `max_seq_len` in probe | `5120` |
| Graph capture result | OOM during bs=4 capture, recorded via fail-open probe |

Conclusion: automatic sizing currently chooses a very large KV pool and leaves
too little headroom for promoted graph capture.  The fixed `np128` benchmark
path is stable; automatic graph-mode serving should cap pages or lower memory
ratio before being treated as a supported default.

TARGET 07.78 dense FP8 projection memory mode saved about `806,961,152`
bytes/rank.  Converted using the current KV page size, that is about `41.78`
extra pages, `10,696` extra KV tokens, `2.61` extra 4096-token prompts, or
`2.09` extra 4096+1024 requests per rank.  This remains a capacity result, not
a speed-promotion result.

## Hardware Efficiency Ledger

The FLOP model is an active sparse estimate:

- dense projection FLOPs use `2*M*N*K`;
- routed MoE counts only the `6` activated routed experts per token;
- shared expert FLOPs count the executed shared expert;
- attention FLOPs are reported as a lower/upper range, with the upper bound
  adding a capped sparse QK/PV estimate using `window_size + index_topk = 640`;
- communication is excluded from FLOPs and reported separately.

### 1. Whole-Run Throughput And MFU-Like Metrics

| Shape | Split | Seconds | Active FLOPs lower | Active FLOPs upper | TFLOP/s lower | TFLOP/s upper | MFU lower | MFU upper |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4096x1024` | prefill | `12.7796` | `1.221 PF` | `1.398 PF` | `95.5190` | `109.3924` | `3.827%` | `4.383%` |
| `4096x1024` | decode | `72.5736` | `0.318 PF` | `0.362 PF` | `4.3800` | `4.9902` | `0.175%` | `0.200%` |
| `4096x1024` | whole | `93.2640` | `1.539 PF` | `1.760 PF` | `16.4970` | `18.8728` | `0.661%` | `0.756%` |
| `4096x128` | prefill | `12.7423` | `1.221 PF` | `1.398 PF` | `95.7989` | `109.7129` | `3.838%` | `4.396%` |
| `4096x128` | decode | `9.0405` | `0.039 PF` | `0.045 PF` | `4.3651` | `4.9731` | `0.175%` | `0.199%` |
| `4096x128` | whole | `24.6196` | `1.260 PF` | `1.443 PF` | `51.1852` | `58.6099` | `2.051%` | `2.348%` |

Interpretation: prefill is far more efficient than decode because it has larger
effective matrix shapes.  Decode is only about `0.18-0.20%` of TP8 BF16 TC peak
by active sparse FLOPs, which is a strong latency/communication/occupancy signal
rather than a pure Tensor Core compute ceiling.

### 2. Per-Bucket Roofline Table

| Bucket | Wall time basis | Envelope share | Estimated FLOPs | Estimated min bytes | AI | Achieved | Bound | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Prefill forward | `12.7796s` aggregate long run | `13.70%` elapsed | `1.221-1.398 PF` | tens of GB lower bound | above A100 TC/HBM crossover | `95.5-109.4 TFLOP/s` | compute/occupancy-bound | theoretical roofline headroom is large, but not a single clean exact target |
| Decode graph replay | `72.5736s` aggregate long run | `77.81%` elapsed | `0.318-0.362 PF` | `55.8 TB` projection-cache decode read estimate plus KV/index traffic | roughly `5.7-6.5 FLOPs/byte` on projection-read lower bound | `4.38-4.99 TFLOP/s` | latency/launch/occupancy plus memory and comm effects | exact buckets are fragmented |
| Prepare / metadata | `7.7190s` aggregate long run | `8.28%` elapsed | n/a | small structured metadata and staging traffic | n/a | n/a | latency/CPU-GPU staging | likely below `2%` E2E without broad graph/metadata redesign |
| TP communication | counters: `2816` collectives, `558.4 GB` | wall not isolated in stable macro | n/a | `558.4 GB` counter bytes | n/a | `7.69 GB/s` vs decode-forward wall | communication/latency-bound | needs NCCL timeline before a separate target |
| Cached BF16 projection owners | owner-timing pass top local owners: q_wqb `67.2ms`, wo_b `63.7ms`, shared_down `42.6ms` max-rank | instrumented one-repeat only | GEMM FLOPs from cached BF16 projections | repeated packed weight reads | low-to-moderate AI at batch 4 decode | low absolute wall | small-GEMM occupancy/latency-bound | too small and diffuse for next exact TARGET 07 |
| KV/index/sparse attention | fallback/counter surface: attention, KV writes, metadata construction present | not separately timed in stable macro | incomplete | KV min stream lower bound `37.1 GB` aggregate | low | low MBU | memory/latency mixed | no single selected bucket |

Owner-timing caveat: the owner pass intentionally adds CUDA event ranges and is
slower (`41.998 output tok/s` on the short one-repeat pass).  It is used only to
rank labeled owners, not to report promoted-path throughput.

### 3. Memory Bandwidth Utilization

| Surface | Estimated bytes | Time basis | Estimated bandwidth | MBU vs TP8 HBM |
| --- | ---: | ---: | ---: | ---: |
| Decode projection BF16 cache reads | `55.814 TB` | `72.5736s` decode | `0.769 TB/s` | `4.71%` |
| KV min stream traffic | `37.075 GB` | `93.2640s` elapsed | `0.398 GB/s` | `0.0024%` |
| Replay metadata input copies | `196,416 bytes` | long run graph replay | negligible | negligible |

The low MFU and low MBU together point at small-shape latency, launch, graph
node, synchronization, occupancy, and communication effects.  HBM bandwidth is
not saturated.

### 4. Communication Efficiency

Long-shape communication counters:

| Label | Op | Count | Bytes |
| --- | --- | ---: | ---: |
| `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | all-reduce | `1376` | `184,683,593,728` |
| `dsv4.v1_moe_reduce_once_all_reduce` | all-reduce | `1376` | `369,367,187,456` |
| `dsv4.embedding_all_reduce` | all-reduce | `32` | `4,294,967,296` |
| `dsv4.lm_head_all_gather` | all-gather | `32` | `66,191,360` |
| Total | mixed | `2816` | `558,411,939,840` |

Derived:

| Metric | Value |
| --- | ---: |
| Counter bytes/output token | `45.44 MB` |
| Counter bytes / elapsed wall | `5.99 GB/s` |
| Counter bytes / decode-forward wall | `7.69 GB/s` |

The communication surface is visible, but this target did not collect a clean
NCCL wall-time profile.  It is not selected as the next target without a
dedicated NCCL timeline.

## Remaining Optimization Opportunities

| Rank | Target surface | Current time or percent | Max plausible benefit | Bound type | Confidence | Next step |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | TARGET 08 radix prefix cache | Prefill forward `12.7796s` aggregate, `13.70%` elapsed; TTFT mean `4.9550s` | workload dependent; shared 4096-token prefixes can skip most prefill on hits | feature/cache reuse | high for shared-prefix workloads | start TARGET 08 |
| 2 | Decode/prefill prepare metadata | Prepare `7.7190s`, `8.28%` elapsed | likely below `2%` E2E without broad graph/metadata redesign | latency/staging | medium | no exact TARGET 07 action |
| 3 | TP communication | `2816` collectives, `558.4 GB` counter bytes | unknown without NCCL wall timeline | communication/latency | medium | no action in this target |
| 4 | Dense FP8 Marlin projection memory mode | `807 MB/rank` saved in 07.78, speed neutral | no default speed gain | memory/capacity | high | memory/capacity target only if context becomes limiter |
| 5 | Precision/low-bit research | exact-route buckets are fragmented | possible but quality/correctness risk | precision/research | low-medium | not the immediate next step |

## Decision

Do not continue with another exact TARGET 07 speed target right now.

The decision-rule check fails for exact TARGET 07:

- no single fresh default-path bucket has both clean timing attribution and a
  credible `>=2%` E2E exact-route gain;
- the visible decode losses are fragmented across tiny graph replay shapes,
  communication, metadata/staging, small GEMMs, and cache/index traffic;
- precision or low-bit work would be a different research target with broader
  correctness risk;
- memory mode is useful, but TARGET 07.78 already showed it is speed neutral for
  default promotion.

Recommended next action:

```text
start TARGET 08 radix prefix cache
```

Operational caveat for TARGET 08: start with fixed or capped page counts such
as `--num-pages 128`.  The automatic `memory_ratio=0.9` KV sizing path currently
chooses `2778` pages and OOMs during promoted graph capture, so automatic
graph-mode capacity should be repaired or capped before treating it as a serving
default.
