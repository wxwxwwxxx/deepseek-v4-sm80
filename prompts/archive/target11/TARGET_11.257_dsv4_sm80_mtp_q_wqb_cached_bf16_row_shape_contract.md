# TARGET 11.257: DSV4 SM80 MTP q_wqb cached-BF16 row-shape contract

## Status

Next after TARGET 11.256.

TARGET 11.256 classified the first internal owner inside rank6 layer0 `q_wqb`
as:

```text
classification: q_wqb_row_shape_owner
anchor: uid0 / position5 / full_loc3077 / depth0 correction row
rank: 6
layer: 0
operator: q_wqb / wq_b
```

Important conclusion from 11.256:

```text
q_wqb input is exact;
q_wqb source FP8 weight, scale, and cached BF16 weight are aligned;
both sides use the cached-BF16 q_wqb path;
baseline normal decode runs q_wqb as a rows=4 batched projection;
MTP target verify runs q_wqb as a rows=3 row-invariant projection;
the same q_wqb path is not bitwise row-invariant under batched rows=4.
```

The next target must define and test the q_wqb row-shape contract for MTP
target verification.  The goal is exact greedy MTP, not performance tuning.
Do not reopen SWA accepted commit, C4/C128 state, layer1 store, MoE, logits,
sampler, graph, communication, or low-precision work unless this target
disproves the q_wqb row-shape owner.

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
artifacts.  If this target needs a reusable q_wqb row-shape oracle helper, put
it under `debug/mtp/` and write outputs under:

```text
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/
```

## Goal

Choose and validate a source-aligned q_wqb projection contract that makes MTP
target-verify rows match no-spec normal decode for the carried full-schedule
anchor and then the full greedy exactness matrix.

Compare these candidate contracts:

1. `target_row_invariant_contract`: keep target verify q_wqb row-invariant and
   make normal decode use row-invariant cached-BF16 q_wqb as well.  This is a
   strong batch-invariance oracle, but may be too slow for default runtime.
2. `target_normal_shape_contract`: make target verify execute q_wqb in a
   normal-decode-compatible row shape.  For the carried anchor this means
   reproducing the baseline rows=4 q_wqb matmul shape instead of rows=3
   single-row chunks.
3. `correctness_gate_contract`: disable or replace the shape-sensitive
   cached-BF16 q_wqb fast path for target verify only, if a slower reference
   path matches no-spec baseline better and can be used as a correctness gate.

The target passes with one of these classifications:

1. `q_wqb_target_normal_shape_fix`: target verify can reproduce normal decode
   q_wqb results by padding/microbatching to a normal-decode-compatible row
   shape, and the full matrix improves or passes.
2. `q_wqb_global_row_invariant_fix`: making both normal decode and target verify
   row-invariant fixes exactness, with measured performance cost.
3. `q_wqb_reference_gate_fix`: a reference or alternative q_wqb path fixes
   exactness when used under a correctness gate.
4. `q_wqb_contract_no_go`: all candidate q_wqb contracts fail, and the target
   names the next earlier or downstream owner with evidence.
5. `q_wqb_runtime_metadata_blocker`: target verify lacks the metadata needed to
   reconstruct normal-decode row shape generically; specify the missing field
   and where it should be produced.
6. `q_wqb_instrumentation_no_go`: current hooks cannot test the contract enough;
   add the smallest missing instrumentation or write a narrower target.

If a minimal generic fix is clear, it should be attempted.  Do not branch on
rank6, layer0, bs6, uid, token, position, full loc, prompt text, or a specific
trace index.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_rank6_layer0_q_wqb_projection_parity/README.md
performance_milestones/target11_mtp_rank6_layer0_q_wqb_projection_parity/raw/q_wqb_rank6_anchor_analysis.json
performance_milestones/target11_mtp_rank6_layer0_q_wqb_projection_parity/raw/q_wqb_rank6_anchor_summary.json
performance_milestones/target11_mtp_layer0_output_subboundary_parity/README.md
prompts/TARGET_11.256_dsv4_sm80_mtp_rank6_layer0_q_wqb_projection_parity.md
prompts/TARGET_11.255_dsv4_sm80_mtp_layer0_output_subboundary_parity.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule, not isolated bs6.
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1.
Preserve TARGET 11.249/11.250 C128 main-state/read-surface behavior.
Preserve TARGET 11.251/11.252 analyzer validity rules.
Do not restore fail-closed accepted commit.
Do not branch on rank, uid, position, layer, loc, bs, request id, token, or
prompt text.
```

Relevant 11.256 evidence:

```text
baseline q_wqb path: mini.wq_b.forward_fp8_cached_bf16_weight
target q_wqb path:   mini.wq_b.forward_fp8_cached_bf16_weight.row_invariant_local
baseline rows: 4
target rows:   3
baseline row-vs-per-row max delta: 6.103515625e-05
target row-vs-per-row max delta:   0.0
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_q_wqb_projection_parity.py
debug/mtp/analyze_state_parity.py
debug/mtp/run_matrix.py
```

Likely Mini code regions:

```text
_fp8_cached_bf16_weight_local_projection
_fp8_cached_bf16_weight_local_projection_row_invariant
QuantizedLinear.forward_fp8_cached_bf16_weight
DeepseekV4Attention._q_wqb_per_row_probe
DeepseekV4Attention.forward q_wqb block
target-verify batch construction and row_to_batch_index metadata
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang to understand whether target verify preserves normal batch row
shape, pads verification rows, or relies on row-invariant kernels.  If SGLang's
implementation is not directly comparable, document that and proceed with the
Mini-specific contract.

## Non-Goals

- Do not change SWA accepted commit/copy/restore.
- Do not change C4/C128 state lifecycle.
- Do not change MoE microbatching unless a guard proves it regressed.
- Do not change q_norm_rope, attention backend, wo_a, wo_b, all-reduce,
  logits, or sampler unless q_wqb is disproven.
- Do not disable target verify or accepted commit.
- Do not start graph/perf, communication-policy, PyNCCL, or low-precision
  research.

## Work Plan

### 1. Reproduce The q_wqb Row-Shape Owner

Use the full `1/2/4/5/6` schedule and confirm the current baseline:

```text
first q_wqb owner: q_wqb_row_shape_owner
carried anchor: uid0 / position5 / full_loc3077 / rank6 / layer0
```

Keep a before-fix artifact so any improvement can be attributed to the q_wqb
contract change.

### 2. Build q_wqb Contract Oracles

Implement temporary, gated probes for q_wqb only.  Suggested gates:

```text
MINISGL_DSV4_MTP_Q_WQB_CONTRACT_ORACLE
MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE
MINISGL_DSV4_MTP_Q_WQB_GLOBAL_ROW_INVARIANT
```

The exact names may differ, but all experiments must be opt-in until proven
safe.

For the carried anchor, compare:

```text
baseline rows=4 batched q_wqb output
target rows=3 current row-invariant q_wqb output
target q_wqb padded to rows=4 and sliced back to active rows
target q_wqb padded to rows=B_normal_decode with anchor at the same row index
normal decode q_wqb forced row-invariant
reference q_wqb path, if affordable
```

Important checks:

- Does padding target q_wqb to rows=4 reproduce the baseline raw sha?
- Does the output depend only on `M` shape, or also on dummy row values?
- Does the anchor row need to occupy the same row index as normal decode?
- Is rows=4 sufficient only for this artifact, or can the required normal
  decode row count be derived generically?

### 3. Derive Generic Runtime Metadata

If the best contract requires normal-decode-compatible row shape, identify how
target verify can know it generically.

Candidate metadata:

```text
normal_decode_active_rows
row_to_batch_index
request table index
max_running_req / active decode batch size
original normal decode row slot for each target-verify correction row
```

If the metadata is missing, classify `q_wqb_runtime_metadata_blocker` and state
exactly where to add it.  Do not hard-code rows=4.

### 4. Implement The Smallest Generic Fix

Preferred fix order:

1. `target_normal_shape_contract`: for target verify q_wqb only, execute the
   cached-BF16 projection in a normal-decode-compatible shape, then gather the
   real target rows back.  Use dummy/padded rows only if tests prove their
   values do not affect the real row output.
2. `q_wqb_reference_gate_fix`: if normal shape metadata is not available yet,
   use a correctness-gated q_wqb path for target verify and leave a clear
   follow-up for performance.
3. `global_row_invariant_contract`: use row-invariant q_wqb for both normal
   decode and target verify only if it is the only exact option.  Measure the
   performance cost and do not silently promote it if it is expensive.

Any fix must be generic across ranks, layers, batch sizes, positions, and
requests.

### 5. Validate Exactness And Guard Against New Owners

After any runtime fix, run:

```text
full 1/2/4/5/6 exactness matrix
text sanity smoke
TARGET 11.246 MoE microbatch guard
TARGET 11.247 accepted-commit guard
TARGET 11.249/11.250 C128 guards
TARGET 11.251/11.252 analyzer validity guards
TARGET 11.253/11.254/11.255/11.256 carried anchors
```

If q_wqb is fixed but the matrix still fails, immediately run the state parity
analyzer and report the next first owner.  Do not start a broad new bisection
inside this target after q_wqb is classified.

### 6. Measure Performance Cost Lightly

This is a correctness target, but if a runtime fix passes exactness, collect a
small cost signal:

```text
owner timing for q_wqb before/after, if available;
macro smoke throughput if cheap;
number of extra q_wqb calls/kernels introduced by padding or row-invariant mode.
```

Do not optimize performance in this target.  The purpose is to decide whether
the fix is a default candidate, a correctness gate, or a temporary oracle.

## Stop Conditions

Stop this target when one of these is true:

1. A generic q_wqb row-shape contract fix lands and the full exactness matrix
   passes.
2. A q_wqb fix lands and the old owner is gone, but a new first owner appears;
   name the new owner and recommend the next target.
3. All q_wqb contract oracles fail and q_wqb is disproven.
4. The target proves missing runtime metadata blocks a generic normal-shape
   fix; specify the missing metadata and producer.
5. Instrumentation is insufficient and the missing probe is specified exactly.

Do not keep tuning q_wqb performance after correctness is resolved.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/README.md
```

Include:

- final classification;
- q_wqb contract/oracle comparison table;
- before/after exactness matrix;
- anchor raw sha comparison before/after;
- normal-shape metadata analysis;
- SGLang source-parity notes;
- runtime code changes and gates;
- performance cost signal, if a fix lands;
- commands and tests run;
- next target recommendation if the matrix still fails.
