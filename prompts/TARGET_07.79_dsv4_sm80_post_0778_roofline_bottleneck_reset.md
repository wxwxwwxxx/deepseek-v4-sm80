# TARGET 07.79: DSV4 SM80 Post-07.78 Roofline And Bottleneck Reset

Date: 2026-07-03

## Goal

Re-establish the performance map for the current promoted DeepSeek V4 Flash
SM80 path after TARGET 07.78:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

This target is a measurement and analysis target.  It should not optimize
kernels.  Its job is to answer:

1. What are the current stable E2E performance numbers?
2. What are the largest remaining bottleneck buckets?
3. Which buckets are compute-bound, memory-bound, communication-bound, or
   latency/launch/occupancy-bound on A100?
4. What is the approximate whole-model and per-module hardware efficiency
   (MFU/HFU-like, bandwidth utilization, roofline headroom)?
5. What is the current maximum supported context/token capacity, both under the
   fixed benchmark `--num-pages 128` setting and under automatic KV-cache
   sizing?
6. Should the project continue with another TARGET 07 speed optimization, or
   should it pivot to TARGET 08 radix prefix cache?

Primary outcome:

```text
a stable post-07.78 bottleneck + hardware-efficiency report
```

## Starting Evidence

TARGET 07.78 completed the dense FP8 Marlin projection promotion gate:

```text
performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate/README.md
```

Key conclusion:

- `dsv4_sm80_a100_victory_densefp8marlinproj` is neutral on 4096/1024
  (`+0.0121%` median, `-0.0291%` mean output tok/s);
- it saves about `807 MB/rank`;
- it should remain an explicit memory-oriented opt-in;
- it should not be promoted into the default speed bundle;
- fair comparisons for lifecycle-sensitive variants should use separate
  `torchrun` invocations or another mechanism that applies env before
  `LLM`/Engine construction, weight loading, model prepare, KV allocation, and
  CUDA graph capture.

Therefore this target should reset the default-path bottleneck map instead of
continuing to tune dense FP8 Marlin projection.

## Non-Goals

Do not do these in this target:

- change runtime kernels;
- add new opt-ins;
- promote or demote existing opt-ins;
- modify precision boundaries;
- tune NCCL;
- change CUDA graph capture;
- implement radix prefix cache;
- make conclusions from a single noisy run;
- continue optimizing any bucket after it is identified.

Small benchmark/reporting scripts are allowed.

## Artifacts

Create:

```text
performance_milestones/target07_post_0778_roofline_bottleneck_reset/
  README.md
  raw/
  scripts/
  summaries/
```

Large profiler outputs should stay under `raw/` or be symlinked from `/tmp`.

## Required Measurement Route

Use separate `torchrun` invocations for lifecycle-sensitive comparisons.

For the primary default-path measurement, run only:

```text
dsv4_sm80_a100_victory
```

At minimum:

```text
TP8
page size 256
--num-pages 128
prompt_len 4096
decode_len 1024
batch_size 4
warmup_repeats >= 1
measured repeats >= 3
```

Also run:

```text
prompt_len 4096
decode_len 128
batch_size 4
warmup_repeats >= 1
measured repeats >= 3
```

Run TP8 text smoke for `dsv4_sm80_a100_victory` before macro conclusions:

- page size 256;
- sane text outputs;
- graph replay active;
- eager decode `0`;
- no unexpected fallback or disabled promoted feature.

If time permits, run a separate optional memory-mode comparison for:

```text
dsv4_sm80_a100_victory_densefp8marlinproj
```

This optional comparison is for memory/capacity accounting only.  Do not use it
as a speed-promotion decision unless it satisfies the TARGET 07.78 stable gate.

## Required Hardware Baseline

Record the actual hardware and software assumptions:

- `nvidia-smi -L`;
- GPU model, memory size, and clocks if available;
- CUDA driver/runtime;
- PyTorch version;
- NCCL version if available;
- torch CUDA capability;
- whether the system appears to be A100 SXM or PCIe.

For roofline calculations, use detected values if available.  If not available,
state the assumption explicitly.  A reasonable default for DGX A100 80GB SXM is:

```text
BF16 dense Tensor Core peak: 312 TFLOP/s per GPU
TF32 Tensor Core peak:      156 TFLOP/s per GPU
FP32 CUDA core peak:        19.5 TFLOP/s per GPU
HBM bandwidth:              2.039 TB/s per GPU
TP8 BF16 TC peak:           2496 TFLOP/s aggregate
TP8 HBM bandwidth:          16.312 TB/s aggregate
```

If the detected device is A100 PCIe, use the PCIe HBM bandwidth assumption
instead and report it clearly.

## Required Efficiency Metrics

The README must include a "Hardware Efficiency Ledger" with at least these
sections.

### 1. Whole-Run Throughput And MFU-Like Metrics

Report for prefill, decode, and whole request:

- output tok/s;
- total tok/s;
- prefill forward seconds;
- decode forward seconds;
- elapsed seconds;
- estimated active FLOPs;
- active FLOP/s;
- MFU-like percentage against TP8 BF16 Tensor Core peak;
- if a bucket is mostly FP32/TF32, also compare it against the corresponding
  FP32/TF32 peak.

Use sparse active FLOPs, not dense theoretical full-model FLOPs:

- dense/GEMM FLOPs: `2 * M * N * K`;
- MoE FLOPs: count only activated routed experts plus shared experts that are
  actually executed;
- attention FLOPs: estimate QK/PV work from observed prefill/decode lengths,
  selected sparse windows/pages, heads, and head dimensions;
- elementwise FLOPs: use a simple documented estimate per element;
- communication is not FLOPs and should be reported separately.

If exact formulas are incomplete, report the estimate as lower/upper bounds and
explain what is missing.

### 2. Per-Bucket Roofline Table

For each major bucket from owner timing / profiler attribution, report:

- bucket name;
- wall time in the measured envelope;
- percent of measured envelope;
- estimated FLOPs;
- estimated minimum HBM bytes;
- arithmetic intensity `FLOPs / byte`;
- achieved FLOP/s;
- achieved HBM GB/s or TB/s where meaningful;
- roofline bound:
  - `compute-bound`;
  - `memory-bound`;
  - `communication-bound`;
  - `latency/launch/occupancy-bound`;
  - `unknown`;
- estimated remaining speedup headroom.

Use A100 roofline thresholds.  For BF16 Tensor Core on A100 SXM, the compute vs
HBM crossover is roughly:

```text
312 TFLOP/s / 2.039 TB/s ~= 153 FLOPs/byte
```

This threshold is only a guide.  Small GEMMs, sparse gathers, graph replay nodes,
and reductions may be occupancy/latency-bound even when the simple arithmetic
intensity suggests otherwise.

### 3. Memory Bandwidth Utilization

For memory-heavy buckets, estimate:

```text
MBU = estimated_bytes_moved / (time * HBM_bandwidth)
```

Separate:

- HBM tensor reads/writes;
- CPU/GPU staging or direct-copy traffic;
- packed weight reads;
- BF16 cached weight reads;
- KV cache reads/writes;
- metadata/index/gather traffic.

If a bucket has low MFU and low MBU, classify it as likely latency, launch,
shape, synchronization, or occupancy limited rather than pure compute/memory
limited.

### 4. Communication Efficiency

From mini communication counters and/or profile ranges, report:

- all-reduce/all-gather/reduce-scatter count;
- tensor shapes and dtypes;
- bytes per request and per output token;
- communication wall time if available;
- approximate achieved bandwidth;
- whether the communication surface is large enough for a separate NCCL target.

Do not implement communication changes in this target.

### 5. Context And Memory Capacity Ledger

Report context capacity in two modes.

Fixed benchmark mode:

```text
--num-pages 128
page_size 256
logical KV token capacity = 128 * 256 = 32768 tokens per TP rank
```

Also report the actual maximum safe workload shape under this fixed setting,
including:

- prompt tokens;
- generated tokens;
- batch size;
- total live tokens;
- whether the benchmark is close to page capacity.

Automatic KV sizing mode:

- run a lightweight capacity probe with `--num-pages` unset, or reuse an
  existing Engine initialization report if available;
- record free memory before loading model;
- record free memory after model/cache preparation;
- record model + persistent cache bytes;
- record KV cache bytes per page;
- record chosen `num_pages`;
- record `max_seq_len`;
- compute total KV token capacity:

```text
num_pages * page_size
```

- estimate maximum single-request context length and maximum batch-concurrent
  token capacity under the current scheduler constraints;
- state any limits from `max_seq_len`, `max_extend_tokens`, page table sizing,
  model config, or smoke/benchmark args.

If optional dense FP8 Marlin projection memory mode is measured, convert its
memory savings into:

- extra pages;
- extra KV tokens;
- equivalent 4096-token prompts;
- equivalent 4096+1024 requests.

## Suggested Commands

Use the existing perf matrix when possible.  Example primary macro:

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

Short profile shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_0778_roofline_bottleneck_reset/raw/4096x128_victory \
  --keep-going
```

Add scripts under `scripts/` to make the runs reproducible.  If new summary
helpers are needed, prefer small standalone scripts that read existing JSON
reports and write `summaries/*.json`.

## Required README Content

The README must include:

- exact commands;
- git status summary;
- hardware/software baseline;
- text smoke result;
- repeat tables for 4096/1024 and 4096/128;
- warmup handling;
- graph replay / eager decode status;
- memory peak and KV-cache memory;
- fixed `--num-pages 128` capacity ledger;
- automatic KV sizing capacity ledger;
- whole-run MFU-like table;
- per-bucket roofline table;
- memory bandwidth utilization table;
- communication efficiency table;
- a ranked "remaining optimization opportunities" table.

For the ranked opportunities table, include:

- target surface;
- current time or percent;
- estimated max possible speedup;
- likely bound type;
- confidence;
- whether the next step should be:
  - another exact TARGET 07 speed target;
  - a precision/low-bit research target;
  - a memory/capacity target;
  - TARGET 08 radix prefix cache;
  - no action.

## Decision Rules

Recommend another exact TARGET 07 implementation target only if all are true:

- a single bucket or coherent cluster is at least `5%` of the measured decode
  envelope or `3%` of E2E elapsed;
- the roofline/owner evidence suggests at least `2%` E2E improvement is
  plausible;
- the fix does not require broad correctness-risk precision changes;
- the target is more than a local microbench polish.

Recommend a precision/low-bit target if:

- exact-route buckets are fragmented or near platform limits;
- a low-bit backend has a clear vLLM or hardware precedent;
- the expected E2E gain is at least `3%`;
- quality/correctness gates can be isolated.

Recommend TARGET 08 radix prefix cache if:

- remaining non-prefix exact speed opportunities are fragmented, noisy, or below
  the thresholds above;
- shared-prefix workloads can skip substantial prefill work;
- context capacity and cache accounting look stable enough to support the next
  feature step.

Recommend a memory/capacity target if:

- E2E speed is near platform plateau;
- memory pressure or maximum context length is the practical limiter;
- a mode such as dense FP8 Marlin projection materially increases available KV
  tokens without correctness regressions.

## Stop Rules

Stop after producing the bottleneck and hardware-efficiency report.  Do not
continue into implementation inside this target.

Hard stop and report blocked if:

- text smoke fails;
- graph replay is unexpectedly disabled;
- eager decode is nonzero for the promoted path;
- benchmark CV is high enough that ranking buckets is meaningless;
- capacity probing risks OOM without producing useful data.

The final answer from the thread should explicitly state whether the next
recommended action is:

```text
continue TARGET 07 exact speed optimization
continue TARGET 07 precision/low-bit research
start a memory/capacity target
start TARGET 08 radix prefix cache
```

