# TARGET 11.245: DSV4 SM80 MTP Target-Verify Layer0 MoE Row-Shape / Precision Contract

## Status

Next after TARGET 11.244.

TARGET 11.244 narrowed the current MTP correctness owner to layer0 MoE
row-shape / precision semantics under target verify:

```text
classification: target_verify_row_shape_owner / backend_precision_owner

bad locs:
  loc263: event0 depth0 accepted, token/pos 1275 / 7
  loc264: event0 depth1 correction, token/pos 2353 / 8
  loc266: event1 depth0 accepted, token/pos 2693 / 10

control:
  loc267: event1 depth1 correction, token/pos 751 / 11
```

Observed layer0 MoE behavior:

```text
moe_input:       exact for all anchors
router_input:    exact for all anchors
router_logits:   exact for all anchors
topk_ids:        exact for all anchors
topk_weights:    non-bit-exact for all anchors, including loc267 control
routed/shared:   tiny row/rank-dependent drift, including control
post-reduce:     non-bit-exact for all anchors
moe_output:      drift for loc263/264/266, exact for loc267
```

The key shape difference is:

```text
baseline normal writer shape: [2, 4096]
MTP target-verify shape:      [6, 4096]
target rows active:           all active, no padding
MoE backend:                  fused runner / marlin_wna16
```

This target should define and validate the correct target-verify MoE execution
contract before attempting a source fix.  Do not patch locs or tokens directly.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Determine which layer0 MoE execution shape/precision contract makes target-
verify rows equivalent to baseline normal decode for the current anchors, and
identify a source-aligned fix path.

The target should answer:

```text
Should target-verify MoE execute as full verify batch [6,*], active-only batch,
normal-shape-compatible microbatches, or row-by-row row-stable reference in
order to preserve greedy exactness?
```

The target passes when it produces one of these:

1. `row_stable_contract`: row-by-row or normal-shape-compatible MoE makes bad
   rows exact while full target-verify batch does not.
2. `active_only_contract`: active-only target rows are exact but full target
   tensor is not, implying inactive/padding/mask contamination.
3. `topk_precision_contract`: topk normalization/weight dtype or accumulation
   is the first precision owner and has a source-aligned fix.
4. `expert_backend_contract`: routing/topk are acceptable, but Marlin WNA16 or
   the active expert backend is not row-shape stable.
5. `reduce_cast_contract`: pre-reduce values are acceptable, but reduce/cast
   order determines BF16 final exactness.
6. `sglang_contract_gap`: Mini's target-verify MoE contract differs from
   SGLang in a way that should be ported before local fixes.
7. `instrumentation_no_go`: current debug hooks cannot compare the required
   shape/precision variants.

If a minimal source-aligned fix is clear and scoped, it may be attempted.
Otherwise close with the contract and a smaller implementation target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_target_verify_layer0_moe_output_subboundary_parity/README.md
performance_milestones/target11_mtp_target_verify_layer0_moe_output_subboundary_parity/raw/
performance_milestones/target11_mtp_target_verify_layer2_input_producer_parity/README.md
prompts/TARGET_11.244_dsv4_sm80_mtp_target_verify_layer0_moe_output_subboundary_parity.md
prompts/TARGET_11.243_dsv4_sm80_mtp_target_verify_layer2_input_producer_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Historical related evidence:

```text
performance_milestones/target11_mtp_moe_output_subboundary_parity/README.md
performance_milestones/target11_mtp_moe_post_reduce_parity/README.md
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
performance_milestones/target07_marlin_wna16_csrc_port/README.md
performance_milestones/target07_mini_owned_fp8_marlin_projection_runtime/README.md
```

Carry forward:

```text
Do not reopen SWA store/commit/restore, layer2 attention read-side state,
logits, sampler, graph/perf, or low-precision research.
Do not branch on batch size, uid, event id, depth, rank, token, expert, layer,
loc, or prompt text.
Treat row-by-row and reference paths as correctness oracles unless performance
is explicitly measured and accepted.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/distributed/impl.py
```

Relevant Mini hooks/paths to inspect:

```text
DeepseekV2MoE.forward / target_verify_row_invariant_local
apply_experts_row_invariant
router/topk normalization
moe_route_dispatch_bf16_marlin_wna16_prepacked
moe_route_dispatch_bf16_marlin_wna16
shared expert path
post-expert all-reduce / reduce dtype
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v2.py
/workspace/sglang-main/python/sglang/srt/layers/moe/
/workspace/sglang-main/python/sglang/srt/layers/moe/topk.py
/workspace/sglang-main/python/sglang/srt/layers/moe/utils.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
```

Source-parity focus:

```text
SGLang target-verify hidden_states shape entering DeepseekV2MoE
whether SGLang computes target-verify rows as full verify batch or per-row chain
topk normalization dtype/order
expert dispatch row ordering and padding
expert backend used for SM80 DSV4
aggregation/reduce/cast order
```

## Non-Goals

- Do not start CUDA graph or throughput optimization.
- Do not change SWA/cache lifecycle.
- Do not patch final norm, lm_head, sampler, C4/C128, PyNCCL, or communication
  policy.
- Do not special-case anchors `263/264/266/267`.
- Do not special-case `bs=2` or `bs=6`.
- Do not promote a slow row-by-row path as default unless the report includes a
  correctness win, perf measurement, and explicit trade-off decision.

## Work Plan

### 1. Reproduce Anchors And Baseline Shape Gap

Use the same environment as TARGET 11.244:

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
accepted commit enabled
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
```

Confirm:

```text
normal writer MoE shape: [2, 4096]
target-verify MoE shape: [6, 4096]
loc263/264/266: moe_input exact, moe_output drift
loc267: moe_input exact, moe_output exact
router_logits and topk_ids exact for all anchors
```

If the anchor moves, follow the new first MoE contract owner and explain why the
old anchors are stale.

### 2. SGLang Contract Census

Before implementing oracles, inspect SGLang source and write a table:

```text
Question
SGLang answer
Mini current answer
Verdict
```

Answer at least:

```text
What shape does SGLang pass to MoE during target verify?
Are verify rows computed as a single full batch, active-only rows, or row-by-row?
Does SGLang keep rejected/padded rows in the MoE tensor?
What dtype/order does SGLang use for topk normalization?
Which expert backend does SGLang use on SM80 for DSV4?
What dtype/order does SGLang use for aggregation and reduce?
When does SGLang cast back to hidden dtype?
```

If SGLang has a clear contract that Mini does not follow, prefer a port/adapt
plan over inventing a Mini-only rule.

### 3. Shape Oracle Matrix

For layer0 MoE and anchors `263/264/266/267`, compare these execution variants:

```text
A. normal actual shape [2,*] baseline
B. target actual full shape [6,*]
C. target active-only rows
D. target row-by-row reference
E. target normal-shape-compatible microbatch, if feasible
F. SGLang-shaped oracle, if different from B/C/D/E
```

For each variant, record:

```text
topk_weights exactness
routed_expert_output exactness
shared_expert_output exactness
expert_aggregate_before_reduce exactness
expert_reduce_output exactness
moe_output exactness
post_moe_residual exactness
runtime warning if path is slow/reference-only
```

The decisive question:

```text
Does any source-plausible variant make loc263/264/266 exact while keeping loc267
exact?
```

### 4. TopK Precision Oracle

If topk weight drift remains the strict first bit drift:

```text
Compare topk normalization with FP32 row sum, target-shape row sum, and row-wise
normalization.
Compare ordering of division/cast.
Compare exact topk weights before and after normalization.
Compare whether tiny topk drift alone can reproduce the bad final BF16 rows
when expert outputs are held fixed.
```

Close with `topk_precision_contract` if this is the minimal owner.

### 5. Expert Backend Row-Shape Oracle

For Marlin WNA16 / active expert backend:

```text
Run routed expert for the same routed inputs under full [6,*], active-only, and
row-by-row shapes.
Compare expert dispatch order, token counts, expert ids, padding, and output.
Check whether backend output changes with row grouping despite identical row
input and expert id.
```

Close with `expert_backend_contract` if the expert backend is the first
shape-sensitive owner.

### 6. Aggregation / Reduce / Cast Oracle

If topk and expert outputs are close but not decisive:

```text
Hold routed/shared outputs constant where possible.
Compare aggregation in FP32 vs hidden dtype.
Compare reduce input/output and all-reduce dtype/order.
Compare final cast to BF16 for bad rows and loc267 control.
Find whether loc267 exactness is due to BF16 rounding cancellation.
```

Close with `reduce_cast_contract` if the deciding owner is reduce/cast order.

### 7. Minimal Fix Policy

A fix is allowed only when the shape/precision contract is proven and
source-aligned.

Allowed examples:

```text
Make target-verify MoE use a row-stable local path for active rows.
Canonicalize target-verify MoE shape to the SGLang-equivalent shape.
Move topk normalization/cast order to match SGLang/normal decode.
Use a row-shape-stable expert backend only for target-verify correctness path.
Align reduce/cast order with baseline/SGLang.
```

Forbidden examples:

```text
branch on loc 263/264/266/267
branch on bs2 or bs6
branch on expert id, token id, uid, event, depth, rank, or prompt text
overwrite moe_output from debug oracle
disable accepted commit
patch SWA store/commit/read again
```

### 8. Validation

After attribution, and after any minimal fix if attempted:

```text
focused bs2 layer0 MoE shape oracle for locs 263/264/266/267
focused bs2 producer-boundary trace from TARGET 11.243 or equivalent
focused bs2 layer2 SWA store/read trace from TARGET 11.242 or equivalent
full-matrix bs6 MoE guard
bs=1/2/4/5/6 exactness matrix
accepted commit stats
```

If a correctness fix is attempted, include a lightweight perf warning or timing
for the new target-verify MoE path, but do not start graph/perf tuning.

Minimum static checks:

```bash
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/distributed/impl.py

git diff --check
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_layer0_moe_row_shape_precision_contract/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
SGLang contract census
shape oracle matrix
topk precision oracle
expert backend row-shape oracle
aggregation/reduce/cast oracle
chosen target-verify MoE contract
implementation summary or precise no-go
bs6 full-matrix guard
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- None of the oracles can compare the same row values across shape variants.
- SGLang's target-verify MoE contract is materially different and requires a
  larger backend port target.
- Row-by-row or reference MoE fixes correctness but is too slow to promote;
  close with a correctness-oracle result and a separate performance plan.
- A proposed fix only works by branching on batch size, uid, event, depth, rank,
  token, expert, layer, loc, or prompt text.
- A safe fix improves bs2 but regresses bs4/bs5, SWA store/commit sanity, or
  layer2 attention read-side controls.
