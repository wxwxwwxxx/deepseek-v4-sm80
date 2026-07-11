# TARGET 12.4: DSV4 SM80 SGLang-Style In-Graph Metadata Prep

## Status

Active child target under:

```text
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

This target follows:

```text
performance_milestones/target12_source_parity_and_contract/README.md
performance_milestones/target12_safe_timing_kernel_census/README.md
```

TARGET 12.2 established that the remaining non-MTP regression is still in
decode replay metadata preparation. The next step is an opt-in PoC that adapts
SGLang's raw-decode metadata -> in-graph full-metadata materialization boundary
to mini-sglang.

## Current Evidence

TARGET 12.2 measured the short TP8 `historical_4096_128_bs4` probe:

```text
baseline, no instrumentation code:
  output tok/s              50.6158
  decode tok/s              163.8353
  decode_forward_enqueue_s  0.3741
  decode_forward_s          3.1007
  decode_prepare_s          0.2998
  enqueue median            2.4230 ms
  forward median            23.9739 ms
  prepare median            1.9186 ms
```

Host-only owner timing is intrusive, but useful for ratios.  The important
rank0 owner split was:

```text
dsv4.graph_replay.prepare_for_replay.total                 3.189 ms median
dsv4.graph_replay.prepare_for_replay.compressed_read_clamp 1.737 ms median
  c4_indices                                               0.750 ms median
  c128_indices                                             0.821 ms median
dsv4.graph_replay.prepare_for_replay.copy_metadata         1.402 ms median
dsv4.replay_copy.component_page_tables                     0.321 ms median
dsv4.replay_copy.fused_helper                              0.167 ms median
dsv4.replay_copy.component_write_locs                      0.167 ms median
dsv4.replay_copy.direct_index_metadata                     0.102 ms median
graph.copy_from.total                                      0.096 ms median
```

Source decode metadata build before replay was also material:

```text
dsv4.prepare.decode.attention_metadata                     2.317 ms median
dsv4.metadata.decode.c128_indices_source                   0.662 ms median
dsv4.metadata.decode.component_write_locs_source           0.250 ms median
dsv4.metadata.decode.object_assembly                       0.414 ms median
```

The helper census showed tiny byte volume but many small operations:

```text
compressed_read_clamp_c4_indices       about 7 launches/step, 12 KB total
compressed_read_clamp_c128_indices     about 5 launches/step, 392 KB total
copy_decode_metadata_for_replay        1 launch/step, 705 KB total
component page-table staging           3 launches/step, 103 KB total
direct_decode_index_metadata_for_replay 1 launch/step, 3.1 MB total
copy_component_write_locs_for_replay   1 launch/step, 6 KB total
```

Interpretation: the bytes are small; the cost is fixed per-step metadata
materialization, small eager torch/Triton operations, launch count, and the
dependency boundary before `g.replay()`.

## Goal

Build and evaluate an opt-in in-graph decode metadata prep path.

The target should answer:

1. Can mini prepare only raw decode metadata before `g.replay()` and derive the
   expensive C4/C128/component graph-consumed metadata inside the captured CUDA
   graph?
2. Does this reduce `prepare_for_replay` and improve output/decode throughput,
   rather than merely moving the same cost into `decode_forward_s`?
3. Which SGLang mechanism should mini permanently adapt: raw metadata, captured
   compression metadata kernel, fixed-address metadata taxonomy, or a smaller
   direct writer fallback?
4. If full in-graph prep is too risky, what exact subset should move to TARGET
   12.5 direct/fused final graph metadata writers?

## Non-Goals

- Do not implement or benchmark multi-stream overlap.
- Do not touch MTP; TARGET 11 remains paused.
- Do not change MoE, communication, low precision, sampling, or scheduler
  policy unless a tiny local hook is needed to expose raw decode metadata.
- Do not weaken prefix, SWA, component-loc ownership, or CUDA graph fixed-address
  contracts.
- Do not promote the path by default in this target. Keep it opt-in until a
  promotion gate proves correctness and repeat-stable performance.
- Do not keep TARGET 12.2 instrumentation-only code in the final path unless it
  remains necessary for a documented diagnostic mode.

## Reference Design

Read the SGLang reference first:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
```

Pay special attention to:

```text
SGLANG_PREP_IN_CUDA_GRAPH
DSV4RawDecodeMetadata
init_forward_metadata_decode()
make_forward_metadata_from_raw_decode()
init_forward_metadata_in_graph()
DSV4AttnMetadata.init_compression_metadata()
dsv4/metadata_kernel.py::init_compression_metadata
refresh_for_breakable_cuda_graph_replay_()
```

Relevant SGLang pattern:

```text
outside graph:
  keep raw decode metadata:
    req_pool_indices
    seq_lens
    out_cache_loc

inside graph:
  build full DSV4 metadata:
    core attention metadata
    indexer metadata
    C4/C128 compressed metadata
    SWA out-cache/write locations
```

Mini code to inspect:

```text
python/minisgl/engine/graph.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/utils/dsv4_owner_timing.py
```

Key mini functions and concepts:

```text
GraphCaptureBuffer.copy_from()
GraphRunner.capture()
GraphRunner._replay_to_buffer()
DSV4AttentionBackend.init_capture_graph()
DSV4AttentionBackend.prepare_for_capture()
DSV4AttentionBackend.bind_capture_graph_inputs()
DSV4AttentionBackend.stage_capture_metadata_for_graph()
DSV4AttentionBackend.prepare_for_replay()
DSV4AttentionBackend._build_metadata()
DSV4AttentionBackend._empty_decode_metadata()
DSV4AttentionBackend._copy_metadata_for_replay()
DSV4AttentionBackend._clamp_graph_replay_compressed_read_metadata()
```

## Proposed Opt-In

Use an explicit experimental toggle, for example:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
```

Add it to the known/experimental toggle registry if implemented.  Keep the
current replay metadata path as the default fallback.

If a group flag is useful, keep it narrow and documented, for example:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_GROUPS=c4,c128,component
```

But prefer one minimal decode-only opt-in first; avoid a large option matrix.

## Implementation Plan

### 1. Preserve The Current Path As Oracle

Before changing graph capture behavior, create a small oracle/debug path that
can compare current out-of-graph metadata to the candidate in-graph output.

Requirements:

- exact equality for int32 metadata buffers;
- compare only active rows and active widths, not padded fill values unless the
  fill contract is part of correctness;
- cover C4 sparse indices, C128 indices, component page tables, component write
  locations, `seq_lens`, `positions`, `raw_out_loc`, and any SWA fields used by
  this scenario;
- run under decode-only, page size `256`, prefix/radix enabled, component-loc
  ownership enabled.

### 2. Define A Raw Decode Metadata Surface

Introduce the minimal raw decode metadata surface mini needs for graph replay.
Likely inputs:

```text
req_table_indices / req_pool_indices
seq_lens / req_seq_lens
raw_out_loc / out_loc
positions
page_table or request-to-token-derived table handle
component page-table handles if component ownership needs them
```

Do not add CPU `.tolist()` / `.item()` to replay.  Raw fields must either be
stable graph input buffers or persistent capture metadata tensors.

### 3. Capture Metadata Materialization Inside The Graph

During CUDA graph capture, record the operations that derive graph-consumed
metadata from raw decode metadata.

Start with the largest owners from TARGET 12.2:

```text
C4 compressed read metadata
C128 compressed read metadata
component write locations
component page-table staging if it can be represented as stable graph inputs
```

A useful first subset is:

```text
raw_out_loc + seq_lens + positions + page_table
  -> c4_out_loc
  -> c128_out_loc
  -> c4/c128 topk lengths
  -> c4/c128 raw/page/full index buffers
```

If component ownership blocks the existing `stage_capture_metadata_for_graph`
route, either:

- adapt the route to component-owned page tables with a clear fixed-address
  contract; or
- stop early and write TARGET 12.5 for a direct/fused component-aware final
  graph-buffer writer.

### 4. Shrink `prepare_for_replay`

On the opt-in path, `prepare_for_replay` should avoid rebuilding/copying fields
that the captured graph now derives.

Measure the delta in:

```text
dsv4.graph_replay.prepare_for_replay.total
dsv4.graph_replay.prepare_for_replay.compressed_read_clamp
dsv4.graph_replay.prepare_for_replay.copy_metadata
dsv4.prepare.decode.attention_metadata
decode_forward_enqueue_s
decode_forward_s
decode_prepare_s
```

Important: if `prepare_for_replay` drops but `decode_forward_s` increases by
the same or more, treat that as boundary movement, not a performance win.  The
candidate must improve output/decode tok/s or clearly reduce launch overhead in
steady state.

### 5. Keep Fallback And Recovery Simple

The opt-in path must fall back to the current path when:

```text
not decode
not CUDA graph replay
unsupported page size
unsupported padded bs
unsupported component/SWA ownership state
metadata oracle mismatch under debug
unsupported device / Triton kernel unavailable
```

Fallback should be explicit in counters/reporting, not silent.

### 6. Clean Up Instrumentation

TARGET 12.2 added instrumentation-only code in:

```text
python/minisgl/engine/graph.py
python/minisgl/attention/deepseek_v4.py
```

During 12.4:

- it is fine to keep that instrumentation while measuring the PoC;
- before finishing, decide whether each instrumentation hook is still useful;
- remove instrumentation-only changes that are not needed by the PoC or a
  documented diagnostic mode;
- preserve any real correctness/performance fix that is not instrumentation.

Report this explicitly in the README.

## Measurement Plan

Use the short TP8 probe:

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
  --output-dir /tmp/dsv4_target12_4_short \
  --keep-going
```

Run at least:

```text
baseline current default
opt-in in-graph metadata prep
opt-in with oracle/debug check if available
owner timing off and on, clearly separated
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
prepare_for_replay owner split
source metadata build owner split
helper/kernel census
graph private-pool/capture memory delta
fallback counters
oracle mismatch counters
```

If the first replay outlier dominates totals, repeat the short probe once and
use steady-state medians for diagnosis.

## Correctness Gates

Run focused unit tests after code changes:

```bash
python -m py_compile python/minisgl/attention/deepseek_v4.py python/minisgl/engine/graph.py
python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
python -m pytest -q tests/kernel/test_deepseek_v4_wrappers.py
```

Run text smoke if the opt-in path reaches macro:

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
  --output /tmp/dsv4_target12_4_text_smoke.json
```

Run a four-scenario soak only if the short probe shows a real win and correctness
is clean:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

## Decision Logic

Promote to the next promotion target only if:

- zero eager decode is preserved;
- text smoke passes;
- metadata oracle/debug checks pass or are convincingly covered by unit tests;
- `prepare_for_replay` drops materially;
- output/decode tok/s improves repeatably on the short probe;
- graph private-pool/capture memory growth is acceptable and recorded.

If the opt-in is correct but only moves time from `prepare_for_replay` into
`decode_forward_s`, do not promote.  Decide whether to:

- optimize captured metadata kernels;
- split to TARGET 12.5 direct/fused final graph metadata writers;
- or abandon in-graph prep as no-go for mini's current graph boundary.

If component ownership makes full in-graph prep unsafe or too invasive, stop
and write TARGET 12.5 around the exact remaining component/C4/C128 writer
boundary.

## Stop Conditions

Stop this child target when one is true:

1. An opt-in in-graph prep PoC runs the short TP8 probe with clean correctness
   and a clear win/loss result.
2. The oracle/debug path shows a metadata mismatch and identifies the field or
   contract that blocks the route.
3. Component/SWA/prefix ownership makes SGLang-style full in-graph prep too
   invasive; write a focused TARGET 12.5 direct/fused writer plan.
4. The patch risks weakening existing prefix/SWA/component ownership contracts.
5. The remaining gap is proven outside replay metadata after the opt-in.

Do not spend this target polishing unrelated metadata owners below roughly
`0.1 ms/step`.

## Deliverables

Create:

```text
performance_milestones/target12_sglang_in_graph_metadata_prep/README.md
```

The README must include:

- git commit and dirty-state summary;
- whether TARGET 12.2 instrumentation was kept, removed, or converted to a
  diagnostic mode;
- SGLang source mapping and mini implementation mapping;
- implemented opt-in env vars and fallback conditions;
- metadata oracle/debug coverage;
- correctness gate results;
- short-probe baseline vs opt-in metrics;
- owner timing and helper census before/after;
- graph capture/private-pool memory impact;
- clear recommendation: promote, continue with a smaller in-graph subset, move
  to TARGET 12.5 direct/fused writers, or no-go.
