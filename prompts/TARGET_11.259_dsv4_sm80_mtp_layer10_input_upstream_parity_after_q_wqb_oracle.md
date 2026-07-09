# TARGET 11.259: DSV4 SM80 MTP layer10.input upstream parity after q_wqb oracle

## Status

Next after TARGET 11.258.

TARGET 11.258 classified the accepted-commit SWA line as:

```text
classification: swa_commit_contract_upstream_non_swa_owner
```

Important conclusion from 11.258:

```text
accepted-commit SWA producer/source/copy/destination/restore/cleanup is not
the culprit for the q_wqb-oracle view;
swa.layer10 faithfully preserves the value it is given;
the wrong value is already present at layer10.input under the q_wqb
target-normal-shape oracle;
default MTP path still exposes a different earlier visible owner:
swa.layer1 after accepted commit.
```

Known q_wqb-oracle anchor:

```text
uid: 0
position: 5
full_loc: 3077
swa_loc: 3077
depth: 0
component: swa.layer10
q_wqb oracle: MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE=1
layer10.input baseline sha: 353f7e1800a29a36
layer10.input MTP sha:      a2bded982f9aecc7
```

This target must trace upstream of `layer10.input`.  Do not patch SWA
accepted-commit lifecycle, q_wqb gates, logits, sampler, graph/perf, or
low-precision paths.  The visible culprit has changed across oracles; treat
`layer10.input` as a boundary to explain, not as a local patch site.

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes.

## Debug Harness Policy

Reusable MTP debug harnesses live under:

```text
debug/mtp/
```

Use the tracked harnesses first:

```text
debug/mtp/run_matrix.py
debug/mtp/analyze_state_parity.py
debug/mtp/analyze_q_wqb_projection_parity.py
```

Do not create new long-lived debug scripts only under `performance_milestones/`.
Milestone directories should contain reports, raw outputs, and one-off
artifacts.  If this target needs reusable layer-input/upstream analyzers, put
them under `debug/mtp/` and write outputs under:

```text
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/
```

## Contract To Check

For greedy exact MTP, the target-verify producer path must satisfy this
layer-input contract:

```text
For each logical target row T = (uid, position, full_loc, depth), the hidden
state entering decoder layer L in MTP target verify must equal the hidden state
that no-spec target decode would feed into decoder layer L for the same logical
token, under the chosen oracle/debug gates.

If layer L input differs, then every cache/component row produced by layer L is
downstream and should not be patched directly.
```

Because recent visible owners have moved with oracle gates, every conclusion
must name the view being analyzed:

```text
default MTP path
q_wqb target-normal-shape oracle path
```

The target should prefer contract-level explanations over local fixes.

## Goal

Find the first upstream boundary that makes q_wqb-oracle MTP `layer10.input`
different from no-spec baseline for the carried logical row.  Also explain how
this owner relates to the default path's `swa.layer1` first owner.

The target passes with one of these classifications:

1. `layer10_input_oracle_artifact`: `layer10.input` mismatch is introduced by
   the q_wqb target-normal-shape oracle or its metadata/schedule side effects,
   and is not present under default-path equivalent tracing.
2. `layer10_input_layer0_owner`: the first mismatch under the q_wqb oracle is
   already at layer0 output or an internal layer0 sub-boundary.
3. `layer10_input_layerN_owner`: layers before N are exact under the q_wqb
   oracle, and the first mismatch is inside decoder layer N, where N is between
   1 and 9.
4. `layer10_input_accumulated_numerical_owner`: no single coarse boundary
   appears as a contract/lifecycle mismatch, but repeated small non-bit-exact
   row-shape/backend differences accumulate into layer10.input drift; name the
   repeated operator class and evidence.
5. `layer10_input_row_identity_owner`: the compared row is not the same logical
   target row because row/depth/parent-batch/full-loc metadata diverges.
6. `layer10_input_hidden_publication_owner`: the right hidden value is computed
   but the wrong row is published or forwarded to the next layer.
7. `layer10_input_analyzer_owner`: the analyzer compares non-equivalent rows or
   lacks validity/comparability rules for the chosen view.
8. `layer10_input_fix`: a generic upstream contract fix lands and improves or
   passes the full exactness matrix.
9. `layer10_input_instrumentation_no_go`: current hooks cannot split the
   producer path enough; add the smallest missing instrumentation or write a
   narrower target.

Do not branch on layer10, uid0, position5, full_loc3077, bs6, trace index, rank,
token, or prompt text.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/README.md
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/raw/q_wqb_layer10_lifecycle_anchor_summary.json
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/raw/q_wqb_layer10_producer_boundary_summary.json
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/raw/default_state_parity_analysis.json
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/raw/q_wqb_target_normal_shape_state_parity_analysis.json
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/README.md
prompts/TARGET_11.258_dsv4_sm80_mtp_accepted_commit_swa_contract_after_q_wqb_oracle.md
prompts/TARGET_11.257_dsv4_sm80_mtp_q_wqb_cached_bf16_row_shape_contract.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule, not isolated bs6.
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1.
Use q_wqb target-normal-shape only as an oracle/debug gate.
Do not promote q_wqb gates.
Preserve TARGET 11.249/11.250 C128 main-state/read-surface behavior.
Preserve TARGET 11.251/11.252 analyzer validity rules.
Do not restore fail-closed accepted commit.
```

Known 11.258 producer boundary:

```text
layer10.input baseline sha:              353f7e1800a29a36
layer10.input q_wqb-oracle MTP sha:      a2bded982f9aecc7
layer10.final_attention_output baseline: a12eac2e0880c0b9
layer10.final_attention_output MTP:      efe2ceb6897b0980
layer10.post_moe_residual baseline:      e8b80da5debf88fe
layer10.post_moe_residual MTP:           ce64ae42dcbb5576
swa.layer10.store_input baseline:        ee6a68c388efa80d
swa.layer10.store_input MTP:             ba9a1cc90877eb90
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_state_parity.py
debug/mtp/analyze_q_wqb_projection_parity.py
debug/mtp/run_matrix.py
```

Likely Mini code regions:

```text
target-verify flattened row construction;
row_depths / row_to_batch_index / row_to_parent_batch_index metadata;
hidden state forwarding between decoder layers;
layer input/output debug capture;
q_wqb target-normal-shape oracle gate;
MoE target-verify microbatch path;
attention q_wqb / q_norm_rope / attention output / wo_b paths;
accepted target hidden and correction-row state publication.
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang to understand target-verify row construction and whether it preserves
parent-batch shape, active rows, or hidden state publication differently.  Do
not copy broad code before identifying the violated Mini contract.

## Non-Goals

- Do not patch SWA accepted-commit copy/restore.
- Do not patch `swa.layer10` store or analyzer rows without upstream proof.
- Do not continue q_wqb performance or promotion work.
- Do not change C4/C128 state lifecycle unless this target proves shared
  target-row metadata is wrong.
- Do not change MoE microbatching unless a guard proves it regressed.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf, communication-policy, PyNCCL, or low-precision work.

## Work Plan

### 1. Reproduce The Two Views

Use the full `1/2/4/5/6` schedule and collect or reuse:

```text
A. default MTP path
B. q_wqb target-normal-shape oracle path
```

For each view, report:

```text
first comparable state owner;
exactness matrix;
accepted commit stats;
the carried logical row metadata if present.
```

Explain whether the q_wqb-oracle `layer10.input` mismatch is absent, masked, or
already present in default-path traces.

### 2. Build Coarse Upstream Layer Bisection

For the q_wqb-oracle view and the carried logical row, compare no-spec baseline
versus MTP at coarse hidden boundaries:

```text
embedding/model input;
layer0.input;
layer0.post_moe_residual;
layer1.input;
layer1.post_moe_residual;
...
layer10.input.
```

If all-layer capture is too large, use binary search over layer checkpoints:

```text
layer0, layer2, layer5, layer8, layer10
```

Then refine the first mismatching interval.  Use filtered layer captures to keep
artifacts bounded.

### 3. Split The First Mismatching Layer

Once the first mismatching layer N is known, split only that layer:

```text
layerN.input;
attention input / q_lora / q_wqb / q_norm_rope / kv branch;
consumed attention metadata and component state;
attention output;
wo_a / wo_b local and post-reduce;
post-attention residual;
MoE input;
router/topk;
expert/shared output;
MoE reduce;
post-MoE residual.
```

Classify whether this is a row-identity issue, operator row-shape issue,
metadata/consume issue, reduce issue, hidden publication issue, or accumulated
numerical drift.

### 4. Check Row Identity And Hidden Publication First

Before treating a tensor mismatch as numerical drift, verify:

```text
uid, input token, target token, position, depth, full_loc, swa_loc;
row_to_batch_index and row_to_parent_batch_index;
parent_batch_size and active rows;
request table slot;
sequence length / cached length;
whether the row was accepted/correction/bonus;
which hidden row is forwarded to the next layer.
```

If metadata differs, classify `layer10_input_row_identity_owner`.

If metadata matches and a correct hidden row exists but the next layer receives
a different row, classify `layer10_input_hidden_publication_owner`.

### 5. Decide Whether q_wqb Oracle Is An Artifact

Run a focused comparison between default and q_wqb-oracle views:

```text
Does q_wqb target-normal-shape change accepted prefix/mismatch_depth?
Does it change emitted tokens before the layer10 anchor is produced?
Does it change target-verify event order or row depths?
Does it introduce req0 failures in bs4/bs5/bs6 before the default failure?
```

If the oracle changes schedule/acceptance enough that `layer10.input` is not
the same logical debugging surface as default, classify
`layer10_input_oracle_artifact` and recommend returning to the default-path
owner with the improved contract tooling.

### 6. Minimal Generic Fix Policy

If a violated contract is clear and the fix is small, it may be implemented.
Prefer fixes that:

```text
align target-verify row construction with SGLang/no-spec semantics;
make hidden row publication explicit and metadata-driven;
preserve accepted/correction/bonus row identity;
centralize parent-batch/depth metadata for all target-verify operators;
keep q_wqb gates debug-only unless a later exactness decision promotes them.
```

Avoid fixes that:

```text
special-case layer10, uid0, bs6, or full_loc3077;
skip analyzer rows to hide mismatches;
disable accepted commits;
promote q_wqb oracle gates as a workaround for upstream drift;
patch SWA copy/restore after 11.258 proved they preserve the value given.
```

### 7. Validation

If runtime code changes, validate:

```text
full 1/2/4/5/6 exactness matrix;
text sanity smoke;
TARGET 11.246 MoE microbatch guard;
TARGET 11.249/11.250 C128 guards;
TARGET 11.251/11.252 analyzer validity guards;
TARGET 11.253 SWA commit/copy guard;
TARGET 11.256 q_wqb owner guard;
TARGET 11.257 q_wqb oracle evidence remains explainable;
TARGET 11.258 SWA accepted-commit lifecycle remains faithful.
```

Also run static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/analyze_q_wqb_projection_parity.py \
  debug/mtp/run_matrix.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

## Stop Conditions

Stop this target when one of these is true:

1. The first upstream boundary causing q_wqb-oracle `layer10.input` drift is
   classified with evidence.
2. The q_wqb-oracle `layer10.input` owner is proven to be an oracle artifact,
   and the default-path first owner is restated as the next target.
3. A generic upstream producer contract fix lands and the exactness matrix
   improves or passes.
4. The analyzer is proven to compare invalid/non-equivalent rows, and the
   comparability rule is fixed or a precise follow-up is written.
5. Instrumentation is insufficient and the missing layer/boundary/checkpoint is
   specified exactly.

Do not continue into graph/perf or broad non-MTP optimization after the
upstream contract segment is classified.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/README.md
```

Include:

- final classification;
- default-path versus q_wqb-oracle first-owner comparison;
- layer-input contract summary;
- coarse upstream layer bisection table;
- first mismatching layer sub-boundary table;
- row identity and hidden publication metadata table;
- q_wqb-oracle artifact analysis;
- SGLang source-parity notes;
- any code changes and gates;
- before/after exactness matrix if a fix lands;
- commands and tests run;
- next target recommendation if exactness still fails.
