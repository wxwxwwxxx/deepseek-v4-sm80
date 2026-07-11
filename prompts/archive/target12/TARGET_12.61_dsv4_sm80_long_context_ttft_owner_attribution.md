# TARGET 12.61: DSV4 SM80 Long-Context TTFT Owner Attribution And Backend Parity

## Status

Current after TARGET 12.606 promoted and tagged the `v0.0.0` recipe baseline.
This is an attribution and design-selection target. It must rank the real
512K-prefill owners before any context-aware dispatch or kernel rewrite begins.

## Purpose

Explain why DSV4 A100/sm80 time to first token degrades strongly with committed
context, determine whether the bounded FP8 indexer remains the dominant owner,
and compare mini's exact long-context backend/dispatch with SGLang and vLLM.

The target must separate:

```text
fixed 8192-token chunk work
work that scales with committed context
one-time/cache-growth and metadata work
CPU/scheduler/prepare overhead
GPU kernel and communication time
```

It produces a ranked owner ledger and an evidence-backed next implementation
target. It does not tune every owner, publish final 512K/1M throughput, or
assume attention itself needs to be rewritten.

## Release Baseline

```text
tag:       v0.0.0
commit:    005f879e73fe9fe7a1e74f3adedf1c8eeceed41b
hardware:  8 x A100-SXM4-80GB, TP8, sm80
recipe:    dsv4_sm80_long_context_512k (req4 / graph4)
page:      256
chunk:     8192
precision: release BF16 path with FP8 paged indexer cache; MTP disabled
```

Read first:

```text
prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md
performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/README.md
performance_milestones/target12_post_indexer_long_context_envelope/README.md
performance_milestones/target12_c128_one_surface_1m_promotion/README.md
performance_milestones/target12_release_fallback_census_native_backend_gate/README.md
prompts/archive/target12/TARGET_12.57_dsv4_sm80_release_fallback_census_native_backend_gate.md
prompts/archive/target12/TARGET_12.59_dsv4_sm80_c128_prefill_metadata_contract_native_micro.md
```

## Current Evidence And Primary Hypothesis

TARGET 12.58 measured this older but still relevant ladder:

| Context | TTFT / wall | Prefill tok/s | Rank0 bounded-indexer select time | Share |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | about 28.1 s | about 2,621 | about 3.82 s | 13.6% |
| 262,144 | about 192.4 s | about 1,454 | about 59.39 s | 30.9% |
| 524,288 | about 714.4 s | about 760 | about 343.37 s | 48.1% |

At 512K, the indexer used:

```text
logits backend: triton_fp8_paged_vllm
top-k backend:  mini local_cuda_global_topk_lens
temporary:      at most 512 MiB FP32 logits per query slice
slices/call:    up to 8
persistent out: [rows, 512] int32
```

The implementation is memory-bounded by slicing but its accumulated work and
launch count grow with context. TARGET 12.606 later proved exact 1M capability
and healthy large-M decode scaling, but did not refresh this owner timing.

TARGET 12.605's newer req4/graph4 512K run reported about 590.7 seconds TTFT
and 907 prefill tokens/s, faster than the older TARGET 12.58 row but without
the same detailed owner instrumentation. Do not combine the older 343-second
indexer timing with the newer total TTFT as though they came from one process.
The checkpointed run in this target must refresh numerator and denominator
under one recipe and instrumentation contract.

Primary hypothesis: the bounded FP8 indexer remains the largest avoidable TTFT
owner and may need a streaming/fused logits-plus-top-k path or a mature
SGLang/vLLM backend. This is not yet a conclusion. C4/C128 attention,
metadata/cache lookup, projections, MoE, and communication must remain visible
until the fresh owner ledger closes.

## Source-Parity Contract

Inspect actual dispatch and implementation before benchmarking alternatives.

Mini:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
benchmark/offline/deepseek_v4_perf_matrix.py
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/
/workspace/sglang-main/python/sglang/srt/layers/attention/
```

vLLM:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/
```

The vLLM source is runnable with `/workspace/venvs/vllm-dsv4`; mini uses the
system interpreter. Runtime probes are allowed when cheaper and reliable, but
source-derived conclusions must be clearly separated from measured runtime
evidence. Do not classify an sm80 backend as unavailable solely because an
installed wheel has an ABI/build problem; inspect its source/build guards.

For each framework record:

```text
cache dtype and layout
query and context blocking
logits materialization shape/dtype
top-k algorithm and fusion boundary
temporary/workspace policy
C4/C128 attention backend and dispatch conditions
metadata representation and materialization point
context-length-dependent dispatch, if any
sm80 support and build/runtime requirements
```

Prefer adapting a mature backend or decomposition over inventing a local one.

## Required Work

### 1. Add Low-Perturbation Owner Timing

Instrument only the long-prefill path and keep instrumentation disabled by
default. Use CUDA events or Nsight/NVTX in a way that does not synchronize each
layer/chunk on the host. Validate overhead on a 16K or 64K control; if enabling
instrumentation changes TTFT by more than 2%, report the perturbation and use a
coarser method.

At minimum split:

```text
scheduler/chunk construction and metadata prepare
embedding and HC/projection work
FP8 indexer cache write/quantization
indexer logits generation
indexer local/global top-k and remap
C4 sparse attention
C128 attention
SWA attention and cache writes
MoE route / Marlin experts / shared expert / reductions
PyNCCL/NCCL and dense TP collectives
remaining kernels
```

Record selected backend strings, launch/slice counts, temporary bytes, output
shapes, and representative layer/rank variation. Aggregate all layers/ranks for
the owner table; do not extrapolate one convenient layer without checking that
its role is representative.

### 2. Capture One Checkpointed Long-Prefill Ladder

Avoid separate fresh 512K runs for every checkpoint. Prefer one legal req4/
graph4 single-request run that records per-chunk aggregates at committed-context
checkpoints:

```text
8K, 16K, 32K, 64K, 128K, 256K, 384K, 512K
```

Use a short output sanity decode. Do not run decode-1K or full 1M profiling in
this target. A 1M short-decode smoke may be reused from TARGET 12.606 and is not
required for owner ranking.

For each checkpoint report:

```text
chunk forward and prepare latency
cumulative TTFT and effective prefill tok/s
owner time and percentage
owner time per chunk/layer/query row
indexer slices, logits elements/bytes, and top-k work
C4/C128 attended key counts and metadata sizes
collective count/bytes
allocated/reserved/free memory and temporary high-water mark
```

Also run a cheap 16K/64K control in a fresh process to establish timing noise
and instrumentation overhead. Do not repeat the complete TARGET 12.606 M grid.

### 3. Separate Algorithmic Growth From Backend Inefficiency

For every material owner derive the expected work as context grows. Distinguish
unavoidable model algorithm cost from avoidable implementation cost:

- repeated full/logits scans versus streaming or hierarchical selection;
- bounded memory that still rereads the same data or launches excessive slices;
- final-output bytes versus temporary HBM traffic;
- metadata width growth versus fixed 8192-token query-row work;
- C4/C128 sparse key count growth versus dense masked attention;
- launch-bound, bandwidth-bound, and tensor-core compute-bound regions.

Report effective bandwidth, achieved FLOP/s or tensor-core use when meaningful,
arithmetic intensity, launch count, and an A100 roofline-style upper bound. Do
not report a whole-model MFU as the primary signal for this heterogeneous
prefill path.

### 4. Build Production-Shape Micro/Subgraph Benches

After the macro owner ranking, build only the benches needed for the top one or
two owners. Prefer captured production tensors or faithful generated metadata
over synthetic dense shapes that omit paging, component ownership, top-k, or
cache layout.

Sweep committed context while keeping query chunk rows fixed:

```text
context: 16K, 64K, 128K, 256K, 512K
query rows: representative release values up to 8192
```

Measure current mini and any directly callable SGLang/vLLM backend. When cross-
framework execution is not ABI-compatible, use source/algorithm parity plus a
mini-owned port/probe only after proving that the source backend supports sm80.

Microbench outputs must include correctness against the current release path,
latency distribution after warmup, workspace/temporary bytes, and launch
count. A faster approximation that changes index selection or precision is not
an exact-path candidate in this target.

### 5. Rank Owners And Select One Next Route

Produce a table with:

```text
owner
512K time/share
growth from 64K -> 256K -> 512K
mini backend
SGLang/vLLM backend difference
avoidable work and theoretical ceiling
credible 512K TTFT reduction
implementation/risk estimate
recommended action
```

Select exactly one primary implementation route. Examples include:

```text
INDEXER_STREAMING_OR_FUSED_TOPK_PORT
C4_OR_C128_CONTEXT_AWARE_ATTENTION_DISPATCH
METADATA_OR_CACHE_LOOKUP_REWORK
NO_KERNEL_REWRITE_OWNER_IS_ALGORITHMIC
BLOCKED_BY_ATTRIBUTION_QUALITY
```

Secondary owners may be recorded for later, but do not start several kernel
rewrites in this census target.

## Correctness Gates

- Preserve v0.0.0 BF16/FP8 cache precision and exact index/top-k semantics.
- Preserve page size 256, chunk size 8192, component ownership, SWA independent
  lifecycle, prefix behavior, and MTP-disabled release state.
- Long-prefill short-decode text/token sanity and finite outputs must pass.
- Instrumented and uninstrumented control outputs must match the existing
  numerical contract.
- No temporary may become unbounded with committed context.
- No capacity-impossible concurrent workload is required.

## Stop Conditions

- Stop once the top owner, growth mechanism, backend difference, and credible
  next implementation route are supported by macro plus micro/source evidence.
- Do not repeatedly run full 512K/1M jobs to narrow a sub-boundary that can be
  reproduced with a production-shape microbench.
- Do not optimize large-M decode; TARGET 12.606 found no anomalous release
  scaling there.
- Do not tune a secondary owner whose maximum E2E upside is below measurement
  noise or clearly dominated by the first owner.
- Do not implement a local backend before checking whether SGLang/vLLM already
  has an adaptable sm80 path.
- If instrumentation cannot attribute owners without more than 2% perturbation,
  stop and report `BLOCKED_BY_ATTRIBUTION_QUALITY` with the smallest missing
  boundary rather than guessing.

## Required Decision

```text
LONG_CONTEXT_OWNER_RANKED
INDEXER_STREAMING_OR_FUSED_TOPK_PORT
C4_OR_C128_CONTEXT_AWARE_ATTENTION_DISPATCH
METADATA_OR_CACHE_LOOKUP_REWORK
NO_KERNEL_REWRITE_OWNER_IS_ALGORITHMIC
BLOCKED_BY_ATTRIBUTION_QUALITY
```

The report may combine `LONG_CONTEXT_OWNER_RANKED` with exactly one selected
implementation route.

## Output

```text
performance_milestones/target12_long_context_ttft_owner_attribution/README.md
```
