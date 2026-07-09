# TARGET 11.262: DSV4 SM80 MTP target-verify pivot feasibility

## Status

Next after TARGET 11.261, before deciding whether to pause TARGET 11 or continue
MTP development.

TARGET 11.261 classified the current q_wqb-oracle anchor as:

```text
classification: shared_expert_backend_row_shape_owner
```

Important 11.261 conclusion:

```text
The rank2/layer1 shared-expert anchor is not random nondeterminism.
The same failing row is stable as a 1-row execution, but target/full-row or
parent-like row shapes recover the true no-spec shared output.  A local
full-shape shared-expert candidate fixes the anchor, but regresses the full
1/2/4/5/6 token matrix.
```

This suggests the current MTP path has a broader target-verify runtime contract
mismatch, not a single local owner that can be safely patched in isolation.
This target is a short pivot experiment.  Its job is not to fix MTP by chasing
another first owner.  Its job is to decide whether exact MTP has a bounded next
path, or whether TARGET 11 should be paused and archived for a later
SGLang-aligned target-verify rewrite.

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes.

## Decision Principle

Default release should not include MTP unless target verification is equivalent
to no-spec target decode under the exact greedy contract:

```text
baseline greedy output == MTP speculative greedy output
target-verify committed state == no-spec canonical target state
```

Approximate/non-strict MTP may exist later behind an explicit experimental flag,
but this target should not promote an approximate mode.  If exactness cannot be
bounded here, recommend pausing MTP rather than adding more local patches.

## Debug Harness Policy

Reusable MTP debug harnesses live under:

```text
debug/mtp/
```

Use existing harnesses first:

```text
debug/mtp/run_matrix.py
debug/mtp/analyze_state_parity.py
debug/mtp/analyze_layer_input_upstream_parity.py
debug/mtp/analyze_moe_aggregate_parity.py
debug/mtp/analyze_shared_expert_parity.py
```

If this target needs reusable transcript, teacher-forced, or canonical-commit
tools, put them under `debug/mtp/`.  Milestone raw outputs should go under:

```text
performance_milestones/target11_mtp_target_verify_pivot_feasibility/
```

## Goal

Run a small number of contract-level experiments that answer:

1. Does target-verify runtime still diverge when draft/acceptance is removed and
   no-spec baseline tokens are used as teacher-forced verify candidates?
2. If target-verify logits/decisions are usable, does exactness recover when
   committed state is produced by a canonical normal target decode replay rather
   than copied from target-verify-produced state?
3. Can a semantic transcript and shape-signature dashboard explain the current
   failures as row construction, row-shape, commit-state production, or
   acceptance/correction/bonus logic?
4. Is there a bounded next target, or should MTP be paused for release?

This target passes with one of these classifications:

1. `pause_mtp_teacher_forced_verify_fails`: teacher-forced target verify fails
   against no-spec baseline, so the target-verify runtime itself is not
   equivalent.
2. `pause_mtp_unbounded_contract_mismatch`: experiments expose multiple
   non-local target-verify contract mismatches with no bounded next fix.
3. `commit_state_production_owner`: teacher-forced verify is acceptable, but
   exactness only recovers when committed state is produced by canonical normal
   target replay.
4. `acceptance_correction_bonus_owner`: teacher-forced verify and canonical
   replay commit pass, but real MTP fails, pointing to acceptance, correction,
   bonus, or logits-processor logic.
5. `shape_signature_contract_owner`: failures correlate cleanly with target
   verify rows/kernel shape signatures, and a canonical shape strategy is the
   bounded next target.
6. `semantic_row_construction_owner`: row identity, position, full_loc,
   page-offset, depth, read range, or write destination is not equivalent.
7. `resume_mtp_contract_port_feasible`: the pivot produces a narrow,
   SGLang-aligned target-verify contract port with a clear next step.
8. `pivot_instrumentation_no_go`: required transcript/replay hooks cannot be
   built safely in this target; specify the missing hook and pause local fixes.

Do not branch on layer1, rank2, uid0, position5, full_loc3077, bs6, token,
trace index, or prompt text.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_rank2_layer1_shared_expert_parity_after_q_wqb_oracle/README.md
performance_milestones/target11_mtp_rank2_layer1_shared_expert_parity_after_q_wqb_oracle/raw/q_wqb_rank2_layer1_shared_expert_analysis.json
performance_milestones/target11_mtp_layer1_moe_aggregate_before_reduce_parity/README.md
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/README.md
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/README.md
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/README.md
prompts/TARGET_11.261_dsv4_sm80_mtp_rank2_layer1_shared_expert_parity_after_q_wqb_oracle.md
prompts/TARGET_11.260_dsv4_sm80_mtp_layer1_moe_aggregate_before_reduce_parity.md
prompts/TARGET_11.259_dsv4_sm80_mtp_layer10_input_upstream_parity_after_q_wqb_oracle.md
prompts/TARGET_11.258_dsv4_sm80_mtp_accepted_commit_swa_contract_after_q_wqb_oracle.md
prompts/TARGET_11.257_dsv4_sm80_mtp_q_wqb_cached_bf16_row_shape_contract.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule as the main exactness surface.
Keep q_wqb target-normal-shape as debug evidence only; do not promote it.
Do not promote local shared-expert full-shape candidate from 11.261.
Do not start graph/perf.
Do not patch SWA/C128/aggregate/all-reduce unless the pivot proves they are the
current causal contract owner.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/run_matrix.py
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang to understand target-verify row construction, candidate flatten
order, target/correction/bonus semantics, temporary KV/cache lifecycle, and
commit rules.  Do not copy broad code in this pivot unless it is needed for a
small observational harness.

## Non-Goals

- Do not chase a new first visible owner after rank2/layer1.
- Do not patch q_wqb, shared expert, SWA, C128, aggregate, or all-reduce as a
  local fix.
- Do not promote approximate MTP.
- Do not benchmark throughput.
- Do not add graph capture or performance optimizations.
- Do not special-case the known anchor or batch size.

## Work Plan

### 1. Build A Semantic Transcript / Shape-Signature Dashboard

For no-spec baseline and MTP target verify, record a compact transcript for the
first mismatch and for at least one passing case:

```text
uid / request_id / decode_step;
draft_depth;
row_type = accepted_candidate / correction / bonus / dummy / normal_decode;
token_in;
token_scored;
token_out if visible;
absolute position;
full_loc;
page_id = full_loc // page_size;
page_offset = full_loc % page_size;
rope position;
target flat row id;
parent batch row;
batch rows / verify rows;
kernel shape signatures for q_wqb, shared expert, MoE, attention if cheap;
KV read slots;
SWA read window and write destination;
C4/C128 read/write surface if available;
commit eligibility and commit source;
accept/correction/bonus decision.
```

Also add a small control scenario where the first generated position is not
equal to `full_loc % 256` if this is easy.  The current anchor has:

```text
full_loc = 3077 = 12 * 256 + 5
position = 5
page_offset = 5
```

That coincidence can hide position/page-offset mistakes.  If changing prompt
length is costly, document that this control remains future work.

### 2. Teacher-Forced Target-Verify Replay

Bypass draft proposal and acceptance uncertainty:

```text
Use no-spec baseline tokens as target-verify candidates.
Run the target-verify runtime on those candidates.
Compare target-verify logits/top1/top2/margin and produced state against
no-spec target decode for the same semantic prefix and row.
```

Questions to answer:

```text
Does teacher-forced target verify match no-spec greedy tokens?
Does it match processed logits argmax?
Are mismatches near-tie numerical flips or large-margin contract bugs?
Does it produce equivalent committed-state candidates?
```

Record:

```text
no_spec_top1;
verify_top1;
no_spec_top2;
margin = no_spec_top1_logit - no_spec_top2_logit;
max_abs_logit_delta;
topk_overlap@10;
first hidden/state mismatch if available.
```

If teacher-forced target verify fails before acceptance/commit is involved,
classify `pause_mtp_teacher_forced_verify_fails` unless a single bounded
semantic-row bug is proven.

### 3. Canonical Replay Commit

Separate verify decisions from committed-state production.

Add a debug mode or harness that:

```text
uses target verify to decide accepted/correction rows, or uses teacher-forced
known-good rows;
does not copy target-verify-produced KV/SWA/C4/C128/component state into main
state;
instead replays accepted/correction tokens through the canonical normal target
decode path to produce committed state;
then continues decode and compares against no-spec baseline.
```

This may be slow; performance is irrelevant here.

Classify:

```text
canonical replay commit passes:
    verify-produced state is unsafe to commit; next route is canonical state
    production or SGLang-aligned state lifecycle.

canonical replay commit still fails:
    acceptance/correction/logits/row semantics are already wrong before commit,
    or the replay harness is not equivalent.
```

If the full replay is too large, do the smallest focused replay that can
distinguish "verify logits usable" from "verify-produced state unsafe".

### 4. Observational Rows / Shape Sweep

Do not use interventional oracles as fixes.  Use observational probes to explain
shape sensitivity:

```text
rows = 1, 2, 3, 4, 5, 6, 8 if cheap;
padded-to-no-spec-shape if available;
current target verify rows;
parent-like rows;
normal target decode rows.
```

For a small set of known anchors, record:

```text
q_wqb_output delta;
shared_expert_output delta;
MoE aggregate/reduce input delta;
attention output delta if available;
final logits delta;
top1/top2 margin;
argmax flip;
committed state checksum delta.
```

If one shape family consistently matches no-spec and another does not, classify
`shape_signature_contract_owner` and recommend a canonical shape strategy rather
than patching one operator.

### 5. SGLang Contract Comparison

Write a source-derived comparison table:

```text
Mini current target verify row construction vs SGLang;
Mini current candidate flatten order vs SGLang;
Mini current target/correction/bonus row types vs SGLang;
Mini current temporary state/commit rule vs SGLang;
Mini current hidden-state surface and shape signature vs SGLang;
Mini current logits processor application point vs SGLang if implemented.
```

Mark each row:

```text
aligned;
probably aligned but unproven;
known mismatch;
not implemented;
not applicable to Mini.
```

### 6. Decision Report

End with one of these recommendations:

```text
pause_mtp_for_release:
    exact MTP remains unbounded; keep debug harnesses and docs, no release path.

continue_with_contract_port:
    next target is a bounded SGLang-aligned target-verify runtime contract port.

continue_with_canonical_commit:
    next target is canonical replay/state-production commit, accepting lower
    speed for correctness bring-up.

continue_with_acceptance_fix:
    target verify and canonical commit are sound; next target is
    acceptance/correction/bonus/logits processor.
```

## Validation

Run static checks for any touched reusable harnesses:

```bash
python -m py_compile \
  debug/mtp/run_matrix.py \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/analyze_layer_input_upstream_parity.py \
  debug/mtp/analyze_moe_aggregate_parity.py \
  debug/mtp/analyze_shared_expert_parity.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

If runtime code changes are kept, run the full `1/2/4/5/6` matrix.  If all
changes are observational harnesses, explain why token exactness is unchanged.

## Stop Conditions

Stop this target when one of these is true:

1. Teacher-forced target verify proves runtime non-equivalence.
2. Canonical replay commit proves verify-produced state is unsafe to commit.
3. Transcript/shape signature proves a bounded semantic-row or shape contract
   owner.
4. The pivot shows multiple non-local owners and recommends pausing MTP.
5. Required instrumentation cannot be built safely in this small target; name
   the missing hook and recommend pausing local fixes.

Do not continue into another local operator-owner repair after this target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_pivot_feasibility/README.md
```

Include:

- final classification;
- explicit release recommendation: pause or continue;
- teacher-forced target-verify replay result;
- canonical replay commit result;
- semantic transcript / shape-signature dashboard;
- rows/shape sweep summary;
- SGLang contract comparison table;
- exactness matrix if any runtime mode is tested;
- code changes and whether they are observational or interventional;
- commands/tests run;
- next target recommendation or pause/archive recommendation.
