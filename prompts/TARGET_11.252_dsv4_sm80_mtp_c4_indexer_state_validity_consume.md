# TARGET 11.252: DSV4 SM80 MTP C4 Indexer State Validity And Consume

## Status

Next after TARGET 11.251.

TARGET 11.251 fixed the analyzer's online C128 parity planner and reclassified
the old C128 raw-loc owner:

```text
c128_attention_state.layer11 / commit_mapping_owner
=> false raw-loc owner
```

After the C128-aware planner, the first comparable checksum surface is:

```text
component: c4_indexer_state.layer10
owner: component_state_owner
full_loc: 3075
c4 indexer state loc: 99
baseline checksum: huge / sometimes NaN-looking
MTP checksum: zero
preliminary classification: analyzer_unstable / likely uninitialized baseline state
```

11.251 did not prove a runtime C4 bug.  It proved the next analyzer-visible
surface is suspicious and needs validity/write/consume instrumentation before
runtime patching.

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes.

## Debug Harness Policy

Reusable MTP debug harnesses now live under:

```text
debug/mtp/
```

Use the tracked harnesses first:

```text
debug/mtp/run_matrix.py
debug/mtp/analyze_state_parity.py
```

Do not create new long-lived debug scripts only under `performance_milestones/`.
Milestone directories are ignored and should contain reports, raw outputs, and
one-off artifacts.  If this target needs a reusable helper, put it under
`debug/mtp/` and write outputs to:

```text
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/
```

## Goal

Determine whether `c4_indexer_state.layer10` is a real MTP correctness owner or
an analyzer artifact caused by comparing uninitialized or unconsumed state.

The target passes with one of these classifications:

1. `c4_indexer_state_uninitialized_skip`: the baseline huge/NaN rows are not
   valid initialized/consumed state.  The analyzer is updated to skip or
   explicitly classify them, and the next real owner is identified.
2. `c4_indexer_state_missing_publication`: the C4 indexer state row is valid and
   consumed, but MTP never writes/publishes it.
3. `c4_indexer_state_wrong_loc`: MTP writes a valid row, but to the wrong mapped
   loc.
4. `c4_indexer_state_wrong_value`: MTP writes the expected loc, but the checksum
   differs from a valid baseline state.
5. `c4_indexer_state_restore_clear`: MTP writes the expected value and later
   clears or restores over it before consumption.
6. `c4_indexer_state_instrumentation_no_go`: current traces cannot establish
   validity or consumption and the missing fields require a narrower
   instrumentation target.

If `c4_indexer_state` is not a real owner, continue the bisection only far
enough to identify the next credible checksum owner.  Do not start a deep repair
inside this target unless the C4 owner is clearly valid and the fix is minimal.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/README.md
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/raw/
performance_milestones/target11_mtp_online_c128_read_surface_port/README.md
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
prompts/TARGET_11.251_dsv4_sm80_mtp_online_c128_parity_planner_next_owner.md
prompts/TARGET_11.250_dsv4_sm80_mtp_online_c128_read_surface_port.md
prompts/TARGET_11.249_dsv4_sm80_mtp_online_c128_main_state_contract_port.md
prompts/TARGET_11.247_dsv4_sm80_mtp_accepted_commit_state_parity_after_moe_microbatch.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Key artifacts:

```text
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/raw/analysis_prefill_bank0_bs6_c128_planner.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/mtp_matrix_1_2_4_5_6_prefill_bank0.json
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 unless explicitly
running a before/after comparison.
Preserve TARGET 11.249/11.250 C128 main-state storage/read-surface behavior
unless source parity proves it is wrong.
Do not restore fail-closed accepted commit as a way to pass exactness.
Do not reopen MoE row-shape work unless focused guards prove it regressed.
Do not patch logits/sampler.
Do not start CUDA graph/perf, PyNCCL, communication-policy work, or low
precision research.
Do not branch on uid, position, layer, loc, bs, request id, token, rank, or
prompt text.
```

## References

Mini:

```text
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_state_parity.py
debug/mtp/run_matrix.py
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor_v2.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Likely terms and surfaces:

```text
c4_indexer_state
c4_indexer FP8 output
c4 attention state
indexer compress state pool
store_indexer / indexer store path
component state loc ownership
state parity snapshot
target verify accepted commit snapshot/restore
```

## Non-Goals

- Do not repair C4 before proving the row is valid and consumed.
- Do not patch logits/sampler to hide token drift.
- Do not disable MTP accepted commit or target verify.
- Do not change C128 main-state contract except for trace fields needed by this
  target.
- Do not put reusable harnesses only under `performance_milestones/`.

## Work Plan

### 1. Instrument C4 Indexer State Validity

Add focused trace metadata for C4 indexer state rows around the first owner:

```text
component: c4_indexer_state.layer10
full_loc: 3075 and neighbor full_locs
state loc: 99 and neighbor state locs
event pair: baseline_after_normal_decode vs mtp_after_normal_before_verify
```

Record for each row:

```text
valid / initialized flag if available;
write event count and writer path;
clear event count;
restore/snapshot event count;
read/consume event count;
last writer event;
last read event;
checksum before and after each lifecycle edge;
whether the row belongs to dummy, padding, or an inactive component slot.
```

If no validity bit exists, add one in trace/debug metadata or derive a
conservative validity state from actual write/consume events.

### 2. Source-Parity Audit

Compare Mini and SGLang C4 indexer state behavior:

```text
when C4 indexer state is allocated;
whether it is initialized eagerly or lazily;
which rows are valid for non-boundary positions;
whether uninitialized rows are expected to be ignored;
which downstream kernels consume c4_indexer_state;
how target verify / accepted commit should copy or restore C4 indexer state.
```

The report must answer:

```text
Should c4_indexer_state.layer10 loc99 be valid for full_loc 3075?
Is loc99 consumed before the bs6 token drift?
Does baseline's huge/NaN-looking state affect logits, or is it an analyzer-only
artifact?
```

### 3. Update Analyzer Validity Rules

Use `debug/mtp/analyze_state_parity.py` as the maintained analyzer.

If C4 rows are uninitialized or unconsumed, update the analyzer to:

```text
classify them as uninitialized_or_unconsumed_state;
skip them for first-owner ranking by default;
include them in a side list for diagnostics;
continue bisection to the next valid consumed checksum owner.
```

If C4 rows are valid and consumed, keep them ranked and classify the lifecycle
owner.

If milestone-local analyzers are still used by old commands, either update them
or make this target's commands use the tracked `debug/mtp/analyze_state_parity.py`
and record the migration.

### 4. Reanalyze And Optionally Rerun

First reanalyze frozen 11.250 artifacts:

```bash
python debug/mtp/analyze_state_parity.py \
  --baseline performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json \
  --mtp performance_milestones/target11_mtp_online_c128_read_surface_port/raw/mtp_matrix_1_2_4_5_6_prefill_bank0.json \
  --output performance_milestones/target11_mtp_c4_indexer_state_validity_consume/raw/analysis_bs6_c4_validity.json \
  --batch-size 6
```

If runtime instrumentation changes are needed, rerun:

```text
TP8
/models/DeepSeek-V4-Flash
page_size=256
num_pages=16
draft_len=2
decode_len=8
max_running_req=4
CUDA graph disabled
PyNCCL disabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
batch sizes: 1 2 4 5 6
```

Use `debug/mtp/run_matrix.py` for new matrix runs unless a target-specific
runner is absolutely required.

### 5. Classify The C4 Owner

Classify `c4_indexer_state.layer10 loc99` as one of:

```text
uninitialized skip;
missing publication;
wrong loc;
wrong value;
restore/clear;
instrumentation no-go.
```

If it is a real owner and the fix is small, a minimal fix may be attempted.
Otherwise stop with a precise next repair target.

If it is skipped as uninitialized/unconsumed, identify the next valid consumed
owner after skipping invalid state rows.

### 6. Validation

Static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/run_matrix.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

If runtime code changes, rerun:

```text
bs1/2/4/5/6 greedy exactness matrix;
MTP text sanity;
non-MTP baseline text sanity;
focused 11.246 MoE microbatch guard;
11.247 accepted-commit state/KV guard.
```

If analyzer-only changes are made, reusing 11.250 frozen artifacts is acceptable,
but the report must say runtime behavior did not change.

## Stop Conditions

Stop and write the milestone report when one of these happens:

1. C4 indexer state is proven uninitialized/unconsumed and skipped, with the next
   valid owner identified.
2. C4 indexer state is proven valid and consumed, and classified into missing
   publication, wrong loc, wrong value, or restore/clear.
3. C4 validity cannot be determined from current traces, and the missing
   instrumentation is explicitly listed.
4. A minimal C4 fix makes bs1/2/4/5/6 exact.  If this happens, run all required
   guards and recommend the next MTP promotion target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/README.md
```

The report must include:

- final classification;
- debug harnesses used or added under `debug/mtp/`;
- C4 indexer state validity/write/consume table;
- source-parity notes against SGLang;
- before/after first-owner ranking;
- exactness matrix status;
- tests/static checks;
- next target recommendation.
