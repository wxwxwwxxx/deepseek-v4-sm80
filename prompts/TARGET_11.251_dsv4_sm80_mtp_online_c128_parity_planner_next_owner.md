# TARGET 11.251: DSV4 SM80 MTP Online C128 Parity Planner And Next Owner

## Status

Next after TARGET 11.250.

TARGET 11.250 reached:

```text
classification: online_c128_read_surface_partial_owner
old blocker removed: c128_online_main_state_compressor_read_surface_not_ported
target_verify_calls > 0
draft_tokens_verified > 0
accepted commit active
fail_closed_exact_batches = 0
bs1/2/4/5 exact
bs6 not exact
```

Important progress from 11.250:

```text
C128 online main-state storage from 11.249 is preserved.
C128 online read/planner surface is active enough to run target verify.
accepted pending banks publish to bank0.
rejected pending banks remain isolated.
MTP and non-MTP text sanity pass on the victory variant.
```

Remaining issue:

```text
The state bisection still reports c128_attention_state.layer11 /
commit_mapping_owner because the analyzer compares baseline legacy C128 loc
1539 with Mini online C128 chunk-state loc 24 for the same full_loc 3075.
After the online C128 main-state contract, this raw loc difference is expected.
```

The next visible checksum owner in 11.250 is reported as:

```text
c4_indexer_state.layer10
baseline: nonzero / sometimes NaN or huge abs_sum in analyzer output
MTP: zero
owner: component_state_owner
```

Do not assume that C4 indexer is the real culprit until the parity planner has
been updated to compare online C128 state by logical full-loc/chunk semantics
and the bs6 bisection is rerun.

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes.

## Goal

Fix the MTP state parity/analyzer planner so that it understands the online
C128 main-state contract introduced by TARGET 11.249 and activated by TARGET
11.250.

Then rerun the accepted-commit state/KV bisection and identify the first real
checksum owner after expected online C128 loc remapping is filtered out.

The target passes with one of these classifications:

1. `parity_planner_fixed_next_owner_found`: online C128 raw-loc false owners are
   removed, and the first real checksum owner is identified with enough
   evidence to write the next repair target.
2. `parity_planner_fixed_exact`: the planner fix reveals that the existing
   runtime is already exact or bs6 exactness passes after a small analyzer-only
   correction plus no runtime changes.  This is unlikely; prove with the full
   matrix.
3. `c4_indexer_state_owner_confirmed`: after C128-aware comparison, the first
   true owner is C4 indexer state, and the report localizes whether it is
   missing publication, wrong loc, wrong value, restore/clear, or analyzer
   checksum instability.
4. `parity_planner_no_go`: current trace data does not contain enough fields to
   compare baseline legacy C128 state and MTP online chunk state.  Add the
   smallest instrumentation needed, or write the next instrumentation target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_online_c128_read_surface_port/README.md
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
performance_milestones/target11_mtp_c128_component_state_publication_parity/README.md
prompts/TARGET_11.250_dsv4_sm80_mtp_online_c128_read_surface_port.md
prompts/TARGET_11.249_dsv4_sm80_mtp_online_c128_main_state_contract_port.md
prompts/TARGET_11.248_dsv4_sm80_mtp_c128_component_state_publication_parity.md
prompts/TARGET_11.247_dsv4_sm80_mtp_accepted_commit_state_parity_after_moe_microbatch.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Key 11.250 artifacts:

```text
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/analysis_prefill_bank0_bs6.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/analysis_read_surface_bs6.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/mtp_matrix_1_2_4_5_6_prefill_bank0.json
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 unless explicitly
running a before/after comparison.
Preserve TARGET 11.249 main-state C128 storage and TARGET 11.250 read-surface
behavior unless new source parity proves they are wrong.
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
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/engine/engine.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
performance_milestones/target11_mtp_accepted_commit_state_parity_after_moe_microbatch/scripts/analyze_state_parity.py
```

SGLang and contract references:

```text
/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor_v2.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

## Non-Goals

- Do not tune performance.
- Do not enable CUDA graph for MTP.
- Do not change sampling behavior.
- Do not implement INT8/FP8 research.
- Do not change PyNCCL or communication routing.
- Do not undo TARGET 11.249/11.250 C128 contract work unless source parity
  proves it is wrong.
- Do not start fixing C4 indexer state until the C128-aware parity planner has
  proved C4 is the first real checksum owner.

## Work Plan

### 1. Define C128-Aware Parity Semantics

Write down the comparison rule for online C128:

```text
baseline legacy C128 state:
    raw loc is legacy component/ring state loc, e.g. 1539.

MTP online C128 main-state:
    raw loc is chunk state loc, e.g. full_loc 3075 -> 3075 // 128 = 24.

comparison key:
    logical full_loc plus C128 chunk id, not raw state loc.
```

The analyzer must treat this as expected mapping, not as
`commit_mapping_owner`, when all of these hold:

```text
component starts with c128_attention_state;
baseline and MTP full_loc match;
MTP storage is online C128 main-state / bank0;
MTP loc equals the expected chunk state loc for the same full_loc;
state trace identifies the component as committed bank0, not a pending bank.
```

If the trace lacks any required field, add focused trace metadata rather than
guessing.

### 2. Patch Or Add A Planner-Aware Analyzer

Implement a reusable helper in the appropriate analysis/debug location.

Preferred options:

```text
Add C128-aware comparison in analyze_state_parity.py if the analyzer is still
milestone-local.
Add reusable helpers in python/minisgl/utils/dsv4_mtp_debug.py if runtime traces
need to emit clearer logical state ids.
```

The planner should report both:

```text
raw_loc_mapping_expected:
    baseline loc and MTP loc differ, but logical full_loc/chunk mapping agrees.

checksum_mismatch:
    logical mapping agrees, but the checksums differ on the comparable state.
```

Do not hide real C128 checksum mismatches.  Only suppress or reclassify raw loc
differences that the online C128 main-state contract makes expected.

### 3. Reanalyze 11.250 Artifacts

Rerun the analyzer on:

```text
baseline_matrix_1_2_4_5_6.json
mtp_matrix_1_2_4_5_6_prefill_bank0.json
```

Required output:

```text
the old c128_attention_state.layer11 / commit_mapping_owner is no longer the
  first owner if it is only raw-loc remapping;
the first real checksum owner is reported with component, layer, full_loc,
  logical loc/chunk, baseline checksum, MTP checksum, event pair, and request;
if the next owner is c4_indexer_state.layer10, classify it with evidence;
if C128 checksum still differs after logical mapping, classify the C128 value
  owner and stop before C4 work.
```

### 4. Validate Or Reject C4 Indexer As The Next Owner

If the first true owner is still `c4_indexer_state.layer10`, do a minimal
classification before writing the next target:

```text
missing publication:
    MTP state is zero and no write event exists for the logical row.

wrong loc:
    MTP wrote nonzero C4 state, but at a different mapped loc.

wrong value:
    MTP wrote to the expected loc, but checksum differs.

restore/clear:
    MTP wrote the expected value and later cleared or restored over it.

analyzer_unstable:
    baseline contains NaN/huge uninitialized-looking values and trace semantics
    are insufficient to decide whether this state is valid.
```

For NaN or huge baseline checksums, record:

```text
whether the baseline state is actually consumed by downstream attention/indexer;
whether the value is deterministic across two baseline runs;
whether the compared row is initialized by the current workload;
whether SGLang/Mini contract expects that state to exist at this position.
```

### 5. Runtime Exactness Check

This target is primarily an analyzer/planner reset, but rerun the focused matrix
if any runtime instrumentation or state publication logic changes.

Required matrix if runtime code changes:

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

If only analyzer code changes, reusing 11.250 frozen artifacts is acceptable,
but the report must say no runtime behavior changed.

### 6. Regression Checks

Run static checks for touched files:

```bash
python -m py_compile \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

If runtime code changes, also rerun:

```text
MTP text sanity;
non-MTP baseline text sanity;
C128 online lifecycle unit tests;
focused 11.246 MoE microbatch guard;
11.247 accepted-commit state/KV guard.
```

## Stop Conditions

Stop and write the milestone report when one of these happens:

1. Online C128 raw-loc false owners are filtered/reclassified, and the first
   real checksum owner is identified with enough evidence for the next repair
   target.
2. C4 indexer state is confirmed as the next owner and classified into
   missing-publication, wrong-loc, wrong-value, restore-clear, or
   analyzer-unstable.
3. C128 checksum remains a real owner after logical mapping.  Stop and write a
   C128 value/lifecycle repair target instead of moving to C4.
4. The traces lack necessary logical mapping fields.  Add instrumentation if
   small; otherwise stop with a precise instrumentation target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/README.md
```

The report must include:

- final classification;
- C128-aware parity planner rule;
- before/after bisection output for the 11.250 bs6 artifacts;
- whether `c128_attention_state.layer11 / commit_mapping_owner` was a false
  raw-loc owner or a real value owner;
- first true checksum owner after planner correction;
- if C4 is next, a minimal C4 owner classification and evidence;
- tests/static checks run;
- exact next target recommendation.
