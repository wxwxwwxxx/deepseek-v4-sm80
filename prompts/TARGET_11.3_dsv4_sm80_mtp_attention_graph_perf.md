# TARGET 11.3: DSV4 SM80 MTP Attention, Graph, And Performance Closure

## Goal

After TARGET 11.14 proves the unified target-verify runtime contract and exact
accepted-KV commit / batch-shape correctness for `bs=1/2/4/5/6` and a
rerun of the eager path proves useful target-pass reduction, align the DSV4
attention/compressed metadata and CUDA graph path with SGLang enough to decide
whether MTP should be kept as an opt-in, optimized further, or promoted.

Do not run this target before accepted-KV commit exactness and target-pass
reduction gates pass.  TARGET 11.29 found that target-verify metadata and
acceptance bookkeeping can be owned in mini, but accepted commit remains blocked
until TARGET 11.295 resolves online C128 MTP pending/write/commit ownership.
TARGET 11.295 made the C128 lifecycle ready but still found greedy drift.
TARGET 11.296 fixed the visible bs=1 token drift, but row0 full logits still
differed enough to require TARGET 11.297 hidden-parity bisection before this
graph/perf target starts.  TARGET 11.297 found the first owner at layer0
`wo_a` projection, so TARGET 11.298 must close that projection batch-shape
parity issue before this graph/perf target starts.  TARGET 11.298 closed `wo_a`
for bs=1/2 row0 parity, but multi-request bs=2/bs=4 contract failures remain,
so TARGET 11.299 must close those before graph/perf starts.  TARGET 11.299
closed the row/depth and mixed-length class enough to reveal a narrower `bs=4`
post-commit state drift; TARGET 11.5 must identify and fix the first
non-equivalent accepted-commit state owner before this graph/perf target starts.
TARGET 11.5 fixed `bs=1/2/4`, but light exposure found a new `bs=5` failure
that may be normal-target batch-shape sensitivity; TARGET 11.6 must classify
and fix or narrow that blocker before this graph/perf target starts.  TARGET
11.6 fixed `bs=1/2/4/5`, but exposure still fails at `bs=6+`; TARGET 11.7
must enumerate remaining MTP path branches and close or classify the `bs=6`
minimal failure before this graph/perf target starts.  TARGET 11.7 found
competing target-verify numerical contracts rather than a safe narrow fix, so
TARGET 11.8 must write a unified target-verify runtime contract before this
graph/perf target starts. TARGET 11.8 stopped after writing that contract and
showing that another local patch is unsafe; TARGET 11.9 must implement a single
SGLang-aligned target-verify runtime mode and prove eager exactness before this
graph/perf target starts. TARGET 11.9 implemented the explicit mode but left it
as a diagnostic no-go because the first owner is layer0 target-verify
attention/KV producer parity under `sglang_prefill_extend`; TARGET 11.10 must
close that owner before this graph/perf target starts. TARGET 11.10 fixed that
owner and exposed `layer0.final_attention_output`, so TARGET 11.11 must close
`attn.wo_b` projection/all-reduce parity before this graph/perf target starts.
TARGET 11.11 closed bs=1 `wo_b` parity but the matrix still fails with
rank-local downstream owners, so TARGET 11.12 must census and close or rank
those owners before this graph/perf target starts. TARGET 11.12 ranked q/RoPE
as the top common rank-local owner, so TARGET 11.13 must build the
operator-parity framework and close or precisely no-go the q_norm_rope pilot
before this graph/perf target starts. TARGET 11.13 showed q_norm_rope is a
same-kernel amplifier for a non-bit-exact `q_wqb_output`, so TARGET 11.14 must
close or precisely no-go the upstream q_lora/q_norm/wq_b precision boundary
before this graph/perf target starts.

## Primary Question

Can MTP improve real serving throughput on A100/sm80 after paying for:

- MTP draft forward;
- target verification;
- DSV4 C4/C128/indexer/compression metadata;
- graph capture/replay;
- prefix/SWA/component lifecycle bookkeeping;
- extra persistent MTP weights/state?

## SGLang References

Inspect and map these first:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
```

Important reference points:

- DeepSeek V4 MTP top-k is limited to top-k 1 in the current SGLang path.
- The attention backend has MTP-specific metadata preparation.
- Online C128 compression state has dedicated MTP support.
- CUDA graph runner has speculative/MTP-specific inputs and replay handling.

## Work Plan

1. Build a source-parity table:
   - SGLang component;
   - mini equivalent;
   - same / different / missing;
   - correctness risk;
   - performance risk.
2. Identify whether mini's verify path is paying extra metadata copies or
   kernels compared with SGLang.
3. Add or adapt direct metadata buffers for MTP verify if the current path
   rebuilds graph inputs unnecessarily.
4. Add graph buckets only after eager MTP exactness is stable.
5. Profile small workloads first:
   - no-weight or partial-layer metadata probes where possible;
   - batch sizes 1, 2, 4, 8, 16;
   - draft lengths 2 and 4;
   - prefix-hit and no-hit cases.
6. Then run full TP8 macro gates.

## Benchmarks

Minimum macro:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16` if prefix cache is enabled;
- at least one low-acceptance prompt mix to measure worst-case overhead.

Compare against the latest promoted non-MTP exact baseline, not an old TARGET 07
baseline.

## Metrics

Record:

- output tok/s and request latency;
- average accepted draft tokens;
- committed target-verify rows, separated into accepted draft rows and
  correction rows where available;
- target verify batch shape;
- graph replay/eager counts;
- CUDA graph memory delta;
- extra persistent memory for MTP;
- metadata/kernel census for verify and draft;
- per-module time attribution where possible.

## Stop Lines

Stop optimizing MTP if any of these remain true after obvious graph/metadata
fixes:

- acceptance is too low on realistic workloads;
- MTP improves only synthetic short tests but regresses serving mixes;
- metadata/graph overhead dominates draft savings;
- prefix/SWA correctness becomes unstable;
- memory overhead meaningfully reduces max context or serving capacity without
  enough throughput gain.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_attention_graph_perf/README.md
```

Include:

- SGLang parity table;
- graph bucket results;
- small-workload and macro profile summaries;
- promotion decision;
- next optimization target if MTP remains promising.
