# TARGET 12.47: DSV4 SM80 In-Graph Metadata Final Promotion Rerun

## Status

Active child target under:

```text
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

This target follows:

```text
performance_milestones/target12_c4_sparse_oracle_contract/README.md
performance_milestones/target12_ingraph_metadata_promotion_soak/README.md
```

TARGET 12.46 fixed the promotion blocker from TARGET 12.45.  The
`c4_sparse_raw_indices` mismatch was an oracle boundary bug around
indexer-mutated C4 sparse fields, not an in-graph Triton formula bug.

This target reruns the promotion subset after that fix and decides whether to
write a default-promotion cleanup target.

## Current Evidence

TARGET 12.45:

```text
short repeat median:
  output tok/s  +5.68%
  decode tok/s +11.37%

four-scenario soak:
  historical_4096_128_bs4   output +6.78%
  historical_4096_1024_bs4  output +16.77%
  serving_mixed_112req      output +22.22%
  prefix_multi_112req       output +25.22%
```

TARGET 12.46:

```text
text oracle: pass
historical_4096_128_bs4 short oracle: pass
non-oracle sanity: baseline 50.4490 output tok/s, in-graph 53.1865 output tok/s
```

Known pitfall from TARGET 12.46:

```text
Do not rely on a combined multi-variant run when checking whether the opt-in is
active.  One combined command produced a second row with
prep_metadata_in_graph=false because graph init followed the first variant.
Use one variant per fresh torchrun for promotion evidence.
```

## Goal

Rerun the promotion subset after the oracle-boundary fix and recommend one of:

```text
default-promotion cleanup target
keep opt-in and fix a newly exposed boundary
move to TARGET 12.5 direct/fused graph metadata writers
no-go
```

The promotion decision must be based on correctness, repeat stability, fallback
behavior, and supported-scenario performance.

## Non-Goals

- Do not implement default promotion in this target.
- Do not add new in-graph metadata features beyond tiny correctness/reporting
  fixes required by the rerun.
- Do not implement TARGET 12.5 direct/fused writers.
- Do not revisit multi-stream, MTP, MoE, communication, low precision, or
  scheduler policy.
- Do not run both compared variants in one Python process for evidence that
  depends on per-process env and graph capture state.

## Required Inputs

Read:

```text
prompts/target.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
prompts/archive/target12/TARGET_12.45_dsv4_sm80_ingraph_metadata_promotion_soak.md
prompts/archive/target12/TARGET_12.46_dsv4_sm80_ingraph_metadata_c4_sparse_oracle_contract.md
performance_milestones/target12_c4_sparse_oracle_contract/README.md
performance_milestones/target12_ingraph_metadata_promotion_soak/README.md
```

Inspect changed code only as needed:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/engine/graph.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
tests/attention/test_deepseek_v4_backend_metadata.py
```

## Hard Rule: Fresh Process Per Variant

Run baseline and in-graph variants as separate `torchrun` invocations whenever
the result is used for promotion evidence.

Use separate output dirs such as:

```text
/tmp/dsv4_target12_47_short_pair1_baseline
/tmp/dsv4_target12_47_short_pair1_ingraph
/tmp/dsv4_target12_47_soak_baseline
/tmp/dsv4_target12_47_soak_ingraph
```

For every in-graph row, verify:

```text
prep_metadata_in_graph_requested=true
prep_metadata_in_graph=true
prep_metadata_in_graph_unsupported_reason=null
```

If an in-graph row reports `prep_metadata_in_graph=false`, discard it for
promotion evidence and identify why.

## Variants

Baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
```

Opt-in:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata
```

Oracle:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1
```

## Measurement Plan

### 1. Correctness Gates

Run:

```bash
python -m py_compile \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/engine/graph.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
python -m pytest -q tests/kernel/test_deepseek_v4_wrappers.py
git diff --check
```

### 2. Oracle Gates

Run text oracle:

```bash
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_47_text_smoke_ingraph_oracle.json
```

Run short historical oracle:

```bash
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --output-dir /tmp/dsv4_target12_47_short_oracle_ingraph \
  --keep-going
```

### 3. Repeat-Paired Short Probe

Run at least two fresh-process pairs for:

```text
historical_4096_128_bs4
```

Baseline command template:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --output-dir /tmp/dsv4_target12_47_short_pair1_baseline \
  --keep-going
```

Opt-in command template:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --output-dir /tmp/dsv4_target12_47_short_pair1_ingraph \
  --keep-going
```

Repeat with `pair2` output dirs.

### 4. Four-Scenario Soak

Run one fresh-process baseline and one fresh-process opt-in across:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

Use separate `torchrun` invocations and separate output dirs.

### 5. Fallback Boundary

Confirm the known unsupported boundary still falls back explicitly:

```text
--enable-dsv4-swa-independent-lifecycle
```

Expected:

```text
prep_metadata_in_graph_requested=true
prep_metadata_in_graph=false
prep_metadata_in_graph_unsupported_reason=swa_independent_lifecycle_not_supported
text smoke passes
```

### 6. Residual Owner Snapshot

Run owner timing only if needed to compare against TARGET 12.45 residuals:

```bash
MINISGL_DSV4_OWNER_TIMING=1 \
MINISGL_DSV4_OWNER_TIMING_CUDA=0 \
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=200000 \
...
```

Do not optimize residual `raw_graph_copy` here.  Only record whether it remains
small enough to defer.

## Metrics To Report

For every benchmark row:

```text
status
output tok/s
decode tok/s
prefill tok/s
decode_forward_enqueue_s
decode_forward_s
decode_prepare_s
schedule_trace medians
graph replay/eager count
captured buckets
comm calls/bytes
wrapper calls
capture memory delta
capture buffer bytes
prefix saved tokens where relevant
prep_metadata_in_graph_requested/active/unsupported_reason
```

## Promotion Criteria

Recommend a default-promotion cleanup target only if:

- py_compile, unit, wrapper, and diff-check gates pass;
- text oracle and short historical oracle pass;
- supported opt-in rows show `prep_metadata_in_graph=true`;
- zero eager decode is preserved;
- communication calls/bytes and wrapper counters remain unchanged or explained;
- capture/private-pool memory remains unchanged or explained;
- short repeat-paired median improves clearly;
- four-scenario soak is positive or neutral on output and decode throughput;
- fallback boundaries are explicit and safe.

If these pass, do **not** default-promote directly in this target.  Write the
next target as a small cleanup/promotion plan covering default env behavior,
fallback/opt-out, docs, and final smoke/soak.

If a new oracle failure appears, keep opt-in and write a focused contract target.

If performance is inconsistent despite oracle passing, keep opt-in and report
the unstable scenario.

If all correctness passes but residual raw staging is the remaining top owner,
recommend TARGET 12.5 direct/fused graph metadata writers after promotion
decision.

## Deliverables

Create:

```text
performance_milestones/target12_ingraph_metadata_final_promotion_rerun/README.md
```

The README must include:

- git commit and dirty-state summary;
- commands and output dirs;
- correctness/oracle/fallback gate results;
- short repeat-paired table;
- four-scenario soak table;
- invariant table;
- residual owner snapshot if run;
- final recommendation: default-promotion cleanup, keep opt-in and fix boundary,
  move to TARGET 12.5, or no-go.
