# TARGET 12.2: DSV4 SM80 Decode Replay Safe Timing And Kernel Census

## Status

Active child target under:

```text
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

This target follows:

```text
performance_milestones/target12_source_parity_and_contract/README.md
performance_milestones/misc_post_mtp_cleanup_replay_attribution/README.md
```

Decision for this phase: do **not** pursue multi-stream overlap. The current
route is to split the replay metadata hot path safely, then choose between
low-risk deforestation, SGLang-style in-graph metadata prep, or direct/fused
graph metadata writers.

## Background

TARGET 12.1 found:

- mini's replay path is closest to SGLang's out-of-graph full-metadata copy
  path, not SGLang's raw-decode in-graph metadata prep path.
- mini already has stable captured metadata buffers, fused replay metadata copy,
  direct C4 graph metadata, and component-table helpers.
- the remaining regression is not explained by communication count/bytes,
  wrapper count, graph replay coverage, or a simple fused-helper fallback.
- `prepare_for_replay` remains the reliable hot owner, but previous CUDA-event
  timing inside graph replay destabilized `cudaGraphLaunch`.

The immediate problem is that `prepare_for_replay` is still too coarse:

```text
GraphRunner._replay_to_buffer
  -> GraphCaptureBuffer.copy_from
  -> attn_backend.prepare_for_replay
      -> optional source metadata rebuild
      -> compressed-read clamp / validation guards
      -> _copy_metadata_for_replay
          -> fused copy_decode_metadata_for_replay
          -> component page-table staging
          -> direct C4/C128/SWA metadata generation
          -> component write-loc helper
          -> fallback per-field copies
  -> g.replay()
```

TARGET 12.2 must identify which sub-boundaries still matter before any larger
implementation work starts.

## Goal

Produce a graph-safe owner and kernel census for decode replay metadata.

Answer these questions:

1. Inside `prepare_for_replay`, which sub-owner accounts for the remaining
   per-step cost?
2. Is the cost mostly Python/object assembly, CPU-side branching, D2D copies,
   small Triton kernels, or graph replay enqueue/dependency boundary?
3. Are the expensive owners stable across short repeat runs, or dominated by
   one-time first-replay outliers?
4. Which next target is justified by evidence:
   - TARGET 12.3 low-risk replay deforestation;
   - TARGET 12.4 SGLang-style in-graph metadata prep;
   - TARGET 12.5 direct/fused final graph metadata writers;
   - or a no-go / different diagnosis?

## Non-Goals

- Do not implement multi-stream overlap.
- Do not port SGLang in-graph metadata prep in this target.
- Do not rewrite attention, MoE, communication, or low-precision paths.
- Do not tune MTP. TARGET 11 is paused.
- Do not use CUDA-event owner timing in the replay hot path if it can affect
  graph launch correctness.
- Do not add per-step `.item()`, `.cpu()`, `.tolist()`, unconditional
  `torch.cuda.synchronize()`, or verbose per-token logging to the hot path.

Small instrumentation-only code changes are allowed. A tiny low-risk cleanup is
allowed only if it is required to make the census reliable and is separately
reported.

## Required Inputs

Read first:

```text
prompts/target.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
performance_milestones/target12_source_parity_and_contract/README.md
performance_milestones/misc_post_mtp_cleanup_replay_attribution/README.md
```

Inspect mini code:

```text
python/minisgl/engine/graph.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/utils/dsv4_owner_timing.py
benchmark/offline/deepseek_v4_perf_matrix.py
```

Use SGLang/vLLM only as boundary references if needed:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner_utils/buffers.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py
```

## Measurement Plan

### 1. Establish The Same Short Probe

Use the current non-MTP baseline:

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
  --output-dir /tmp/dsv4_target12_2_short \
  --keep-going
```

Record:

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
comm call/byte counts
wrapper counters
```

If a first-replay outlier dominates totals, repeat once and report steady-state
median plus total-time separately.

### 2. Add Graph-Safe Owner Ranges

Use host-only timing and counters. Keep owner labels stable and rank-scoped.

At minimum, split:

```text
graph.copy_from.total
graph.copy_from.input_ids
graph.copy_from.out_loc
graph.copy_from.positions

dsv4.graph_replay.prepare_for_replay.total
dsv4.graph_replay.prepare_for_replay.maybe_rebuild_source_metadata
dsv4.graph_replay.prepare_for_replay.compressed_read_clamp
dsv4.graph_replay.prepare_for_replay.debug_guard
dsv4.graph_replay.prepare_for_replay.copy_metadata

dsv4.replay_copy.fused_helper
dsv4.replay_copy.component_page_tables
dsv4.replay_copy.direct_index_metadata
dsv4.replay_copy.swa_out_loc
dsv4.replay_copy.component_write_locs
dsv4.replay_copy.fallback_fields

dsv4.metadata.decode.object_assembly
dsv4.metadata.decode.scalar_source
dsv4.metadata.decode.page_table_source
dsv4.metadata.decode.swa_indices_source
dsv4.metadata.decode.c128_indices_source
dsv4.metadata.decode.component_page_tables_source
dsv4.metadata.decode.component_write_locs_source
```

If some labels already exist, reuse them rather than inventing duplicate names.
If a sub-owner is only measurable by counters and not timing, record calls,
bytes, shapes, dtypes, status, and whether it launches a kernel.

### 3. Build A Kernel Census Without Unsafe CUDA Events

The report must include a kernel/call census for replay metadata work. Use one
or more safe methods:

- owner counters around wrapper calls and Triton helper calls;
- existing benchmark counters if present;
- one-owner-at-a-time differential env toggles;
- short Nsight Systems capture only if it captures CUDA activity cleanly and
  does not alter graph replay;
- synthetic/no-weight replay probes for metadata helpers, if easier and safe.

Do not rely on CUDA events placed inside graph replay unless first proven safe.

For each helper, report:

```text
helper/kernel name
calls per decode step
total calls
approx bytes touched or metadata element count
input/output shapes
fixed-address destination or temporary output
inside graph, before graph replay, or normal eager helper
whether it is mandatory for current scenario
```

### 4. Differential Probes

Run only small, evidence-driven probes. Examples:

```text
disable direct graph metadata
disable component page-table cache only if correctness-preserving
force old fallback only as a diagnostic, not as an optimization
skip debug validation paths with env off
disable owner timing to measure instrumentation overhead
```

Every differential probe must preserve text sanity or be clearly marked
diagnostic-only.

Do not run a full soak in 12.2 unless a very small change unexpectedly recovers
the regression and passes correctness.

## Correctness Gates

Run focused tests if instrumentation touches metadata code:

```bash
python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
python -m pytest -q tests/kernel/test_deepseek_v4_wrappers.py
```

If any runtime behavior changes, also run text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_2_text_smoke.json
```

## Expected Decision Logic

Choose the next target based on evidence:

- If host/object assembly, repeated env checks, disabled-feature branches, or
  redundant Python container construction dominate: write TARGET 12.3.
- If multiple small mandatory metadata kernels/copies dominate and match
  SGLang's raw-to-full metadata boundary: write TARGET 12.4.
- If direct writers exist but still write intermediate buffers or duplicate
  work: write TARGET 12.5.
- If the measured overhead is mostly graph replay enqueue/captured forward
  rather than metadata prep, stop and redirect TARGET 12 toward graph boundary
  or captured-kernel census.
- If no sub-owner is material after repeat measurement, document no-go and
  recommend returning to macro-level profile.

## Stop Conditions

Stop this child target when one is true:

1. The `prepare_for_replay` hot owner is split into specific sub-owners with
   enough evidence to choose 12.3, 12.4, or 12.5.
2. Instrumentation overhead or graph instability prevents reliable splitting;
   report the failed methods and propose a safer synthetic/no-weight probe.
3. A small required fix restores short-probe performance close to TARGET10.27
   and passes correctness.
4. The remaining gap is proven outside replay metadata.

Do not spend time polishing non-material owners below roughly `0.1 ms/step`
unless they combine into a clear repeated launch/copy cluster.

## Deliverables

Create:

```text
performance_milestones/target12_safe_timing_kernel_census/README.md
```

The README must contain:

- git commit and dirty-state summary;
- exact commands and env vars used;
- baseline short-probe table;
- instrumentation overhead check;
- owner timing table by rank0, and any cross-rank anomalies if observed;
- kernel/helper census table;
- differential-probe table;
- correctness gate results;
- clear next-target recommendation with rationale.
