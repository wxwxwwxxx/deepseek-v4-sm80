# TARGET 12.45: DSV4 SM80 In-Graph Metadata Promotion Soak

## Status

Active child target under:

```text
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

This target follows:

```text
performance_milestones/target12_sglang_in_graph_metadata_prep/README.md
performance_milestones/target12_safe_timing_kernel_census/README.md
```

TARGET 12.4 implemented an opt-in SGLang-style in-graph decode metadata prep
PoC behind:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
```

The PoC passed unit/wrapper/text/oracle gates and improved the short
`historical_4096_128_bs4` probe, but it is not ready for default promotion
until repeat stability, serving/prefix scenarios, fallback boundaries, and
residual owners are checked.

## Current Evidence

TARGET 12.4 short-probe result:

```text
historical_4096_128_bs4, TP8, page size 256, num pages 128

baseline:
  output tok/s              43.674646
  decode tok/s              140.355356
  decode_forward_s          3.619385
  decode_prepare_s          0.300457
  graph replay/eager        127 / 0

in-graph metadata opt-in:
  output tok/s              52.971615
  decode tok/s              178.713354
  decode_forward_s          2.842541
  decode_prepare_s          0.099537
  graph replay/eager        127 / 0
```

Owner timing showed the expected mechanism:

```text
prepare_for_replay.total:       464.011 ms -> 32.611 ms
compressed_read_clamp:          256.352 ms -> absent
copy_metadata:                  202.217 ms -> absent
raw_graph_copy:                 absent     -> 31.460 ms
```

Capture/private-pool memory did not increase in that run.

Important caution: the TARGET 12.4 baseline row was lower than earlier recent
baselines, so this target must use repeat-paired runs and steady-state medians
before any default-promotion decision.

## Goal

Decide whether the in-graph metadata opt-in is ready for a default-promotion
target, should stay opt-in with a narrower contract, or should redirect to
TARGET 12.5 direct/fused final graph metadata writers.

Answer:

1. Is the short-probe win repeat-stable across fresh processes?
2. Does the win hold on the four important non-MTP scenarios?
3. Does the opt-in preserve text sanity, zero-eager graph replay, prefix/radix
   behavior, component-loc ownership, and PyNCCL/communication invariants?
4. Does oracle mode still pass on representative short/text runs?
5. What residual replay metadata owner remains after the opt-in?
6. Are unsupported cases explicit and safe fallback, rather than silent partial
   activation?
7. Should the next target be promotion cleanup/defaulting, TARGET 12.5
   direct/fused writers, or a support-contract fix?

## Non-Goals

- Do not implement multi-stream overlap.
- Do not restart MTP.
- Do not add a new low-precision, MoE, communication, or scheduler feature.
- Do not rewrite in-graph metadata beyond small fixes needed for correctness,
  fallback clarity, instrumentation cleanup, or obvious low-risk residual owner
  removal.
- Do not promote the opt-in by default in this target.

## Required Inputs

Read:

```text
prompts/target.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
prompts/TARGET_12.4_dsv4_sm80_sglang_in_graph_metadata_prep.md
performance_milestones/target12_sglang_in_graph_metadata_prep/README.md
performance_milestones/target12_safe_timing_kernel_census/README.md
```

Inspect changed code:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/engine/graph.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

SGLang references only as needed:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
```

## Variants

Compare:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata
```

Oracle diagnostic:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1
```

Keep env vars fixed before process start.  If running multiple variants in one
Python process risks Engine/CUDA lifecycle contamination, run each variant as a
separate `torchrun` and merge the report manually.

## Measurement Plan

### 1. Repeat-Paired Short Probe

Run at least two fresh-process pairs for:

```text
historical_4096_128_bs4
```

Command template:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
             dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --output-dir /tmp/dsv4_target12_45_short_pair \
  --keep-going
```

Record per run:

```text
output tok/s
decode tok/s
prefill tok/s
decode_forward_enqueue_s
decode_forward_s
decode_prepare_s
schedule_trace medians
graph replay/eager decode count
captured buckets
communication calls/bytes
wrapper counters
capture memory delta
capture buffer bytes
```

### 2. Four-Scenario Soak

Run the same variant pair on:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

If runtime is high, one fresh-process pair is acceptable for the four-scenario
soak after the short probe has already been repeated.

### 3. Oracle And Text Sanity

Run text smoke for the opt-in:

```bash
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
  --output /tmp/dsv4_target12_45_text_smoke_ingraph.json
```

Run oracle on at least text smoke plus one short historical probe:

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
  --output /tmp/dsv4_target12_45_text_smoke_ingraph_oracle.json
```

Do not run oracle for every long scenario unless a mismatch is suspected.

### 4. Residual Owner Attribution

Use host-only owner timing sparingly because it is intrusive:

```bash
MINISGL_DSV4_OWNER_TIMING=1 \
MINISGL_DSV4_OWNER_TIMING_CUDA=0 \
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=200000 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 ...
```

Compare baseline vs opt-in on the short scenario and report:

```text
prepare_for_replay.total
prepare_for_replay.raw_graph_copy
prepare_for_replay.compressed_read_clamp
prepare_for_replay.copy_metadata
decode attention metadata source build owners
in-graph metadata kernel/helper counters
graph replay/captured forward time
```

If the remaining owner is below roughly `0.2 ms/step`, do not polish it in this
target.  Save it as a note for later cleanup.

### 5. Fallback Boundary Audit

Explicitly record:

```text
prep_metadata_in_graph_requested
prep_metadata_in_graph
prep_metadata_in_graph_unsupported_reason
replay_count/eager_decode_count
```

Check that unsupported states fall back cleanly.  At minimum, confirm the report
captures the current known unsupported boundary:

```text
independent SWA lifecycle -> unsupported/fallback
```

Do not try to implement independent SWA support here unless the soak fails only
because of a trivial reporting bug.

## Correctness Gates

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
```

Also run `git diff --check`.

## Promotion Criteria

Recommend a default-promotion cleanup target only if all are true:

- text smoke and oracle smoke pass;
- unit/wrapper gates pass;
- zero eager decode is preserved in supported scenarios;
- communication calls/bytes and wrapper counters do not regress unexpectedly;
- capture/private-pool memory is unchanged or the increase is explained and
  acceptable;
- short probe repeat-paired median improves clearly;
- four-scenario soak is neutral-or-positive in important serving/prefix cases;
- fallback boundaries are explicit and safe;
- diagnostics/instrumentation are default-off and do not materially affect
  owner-timing-off performance.

If the short probe wins but serving/prefix scenarios regress, keep opt-in and
write the next target around the failing boundary.

If supported-scenario performance is good but residual raw staging remains the
next material owner, write TARGET 12.5 direct/fused final graph metadata writer.

## Stop Conditions

Stop when one is true:

1. The opt-in passes promotion criteria and the report recommends a default
   promotion cleanup target.
2. The opt-in is correct but not repeat-stable, and the report identifies the
   unstable metric or workload.
3. A correctness/oracle/fallback failure identifies a concrete field or contract
   to fix.
4. The residual owner clearly points to TARGET 12.5 direct/fused writers.
5. The opt-in is a no-go for mini's current graph boundary.

Do not spend this target optimizing tiny residual owners.

## Deliverables

Create:

```text
performance_milestones/target12_ingraph_metadata_promotion_soak/README.md
```

The README must include:

- git commit and dirty-state summary;
- exact commands and env vars;
- short repeat-paired results;
- four-scenario soak table;
- text smoke and oracle results;
- fallback/support boundary table;
- owner residual attribution table;
- capture/private-pool memory table;
- instrumentation/diagnostic overhead decision;
- recommendation: promote cleanup, keep opt-in and fix boundary, move to TARGET
  12.5, or no-go.

