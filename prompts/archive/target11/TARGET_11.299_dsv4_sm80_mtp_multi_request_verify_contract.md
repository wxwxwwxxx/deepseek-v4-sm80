# TARGET 11.299: DSV4 SM80 MTP Multi-Request Target-Verify Contract

## Status

Next after TARGET 11.298.

TARGET 11.298 closed the `wo_a` projection batch-shape owner for target verify:

```text
layer0.attention_wo_a_output max_abs_delta = 0.0
lm_head_logits max_abs_delta = 0.0
```

bs=1/draft_len=2 accepted-commit exactness passes with accepted commit enabled.
The remaining blockers are multi-request / multi-row target-verify contract
issues:

```text
bs=2 visible token drift
bs=4 mixed active verify length crash
```

Do not work on CUDA graph, macro throughput, serving datasets, or broad MTP
promotion in this target.

## Goal

Make multi-request MTP target verify exact for the small deterministic gate:

```text
TP8 /models/DeepSeek-V4-Flash
page_size=256
draft_len=2
decode_len=8
CUDA graph disabled
PyNCCL disabled
bs=1/2/4
accepted commit enabled
```

or identify the exact SGLang target-verify contract mechanism that mini still
needs to port.

## Starting Evidence

From TARGET 11.298:

bs=1 passes:

```text
baseline = [11111, 64465, 361, 582, 9628, 3362, 582, 18]
MTP      = [11111, 64465, 361, 582, 9628, 3362, 582, 18]
```

bs=2 fails:

```text
baseline req0 = [11111, 64465, 361, 582, 9628, 3362, 223, 18]
MTP req0      = [11111, 64465, 361, 582, 671, 6102, 294, 8760]
baseline req1 = [1275, 2353, 1121, 2693, 751, 621, 7557, 90738]
MTP req1      = [1275, 2353, 1121, 2693, 751, 621, 7557, 90738]
```

Suspicious bs=2 trace:

```text
verify input row group = [361, 582, 671, 2693, 751, 303]
req0 draft_tokens      = [582, 671]
req0 target_tokens     = [582, 671, 6102]
req0 accepted_prefix   = 2
```

This accepts `671` for req0, but baseline for req0 should continue with
`9628`, not `671`.  Since row0 hidden/logit parity is exact after TARGET
11.298, the next suspect is row-depth semantics: row1/row2 target-verify input,
causal metadata, row packing, or acceptance indexing.

bs=4 crashes before exactness:

```text
DeepSeek V4 MTP target-verify metadata requires a fixed active verify length
per request in one flattened batch; got verify_lens=[1, 1, 2, 1].
```

## SGLang References

Use SGLang as the source contract, especially for top-k 1 target verify and
mixed request handling:

```text
/workspace/sglang-main/python/sglang/srt/speculative/eagle_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/spec_info.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Map these points before broad changes:

- top-k 1 draft token order;
- target verify token order;
- target verify row/depth mapping;
- whether `speculative_num_draft_tokens` is fixed and padded even when a request
  has fewer active tokens left;
- how target verify masks/pads inactive rows;
- how accepted prefix length is computed;
- how correction/bonus rows are selected and committed;
- how rejected tails remain invisible.

## Speculative Stats Reminder

Keep these separate:

```text
draft_tokens_accepted:
    accepted draft tokens.

accepted_kv_copied_tokens:
    committed target-verify rows; may include correction/bookkeeping rows.

target_verify_rows:
    all target verify rows, including accepted draft rows, correction rows, and
    padded/bonus rows.
```

Do not treat committed row count as accepted draft token count.

## Work Plan

### 1. Preserve controls

Keep:

- bs=1 exact accepted-commit control;
- target verify `wo_a` row-invariant fix;
- online C128 lifecycle enabled;
- accepted commit enabled;
- CUDA graph disabled;
- PyNCCL disabled.

Do not hide the issue by disabling accepted commit or recomputing accepted rows
through normal sequential decode.

### 2. Build a row-depth parity harness for bs=2

For the failing bs=2 request 0, compare normal sequential target decode against
flattened target verify at each depth:

```text
depth0 row: input 361 -> target 582
depth1 row: input 582 -> should match sequential target after prefix+582
depth2/bonus/correction row: input 671 or correction candidate
```

Record for each request and depth:

- input token;
- position;
- row index in flattened batch;
- target token;
- draft token being compared;
- accepted/rejected decision;
- top-k logits and margins;
- metadata row summary;
- hidden/logit parity against sequential oracle when practical.

The target should answer: is req0 row1 predicting `671` because the model state
really says so under mini's target-verify metadata, or because row/depth mapping
is wrong?

### 3. Audit row packing and acceptance indexing

Check that flattened rows are grouped and interpreted consistently:

```text
row_start = request_index * verify_len
row_depth = 0..verify_len-1
target_tokens[row_start + depth]
draft_tokens[depth]
commit row depth
emitted token order
```

Verify this for bs=2 with two requests of different committed lengths:

```text
req0 committed_seq_len = 7
req1 committed_seq_len = 10
verify_len = 3
```

Pay special attention to:

- whether row1 input should be the draft token or the previous target row;
- whether target verify is expected to verify a draft chain as `[last target,
  draft0, draft1]`;
- whether mini is accidentally using a future draft token as input for the wrong
  target row;
- whether correction/bonus row selection is off by one.

### 4. Handle mixed active verify lengths

Fix or design the bs=4 mixed `verify_lens` case:

```text
verify_lens=[1, 1, 2, 1]
```

Preferred approaches, in order:

1. Match SGLang's contract if it pads to fixed `speculative_num_draft_tokens`
   rows and masks inactive rows.
2. If SGLang groups by active verify length, implement the same grouping.
3. If neither can be confirmed quickly, implement a conservative mini grouping
   by `verify_len` as a correctness path and mark the performance risk.

Do not crash on mixed verify lengths.

### 5. Fix the smallest proven contract bug

Possible fixes:

- correct row/depth input construction;
- correct accepted-prefix computation;
- correct correction/bonus row emission;
- pad/mask target-verify rows to fixed draft length;
- group target verify batches by active verify length;
- adjust commit row selection to match accepted prefix and correction semantics.

Do not change low-level kernels unless row-depth parity points there.

### 6. Rerun narrow exactness

After a fix:

```text
bs=1 draft_len=2
bs=2 draft_len=2
bs=4 draft_len=2
```

Only if all pass should a later target consider graph/perf.

## Verification Commands

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_multi_request_verify_contract/raw/baseline_bs124.json \
  --page-size 256 --num-pages 16 --decode-len 8 \
  --disable-pynccl --draft-len 2
```

MTP debug:

```bash
MINISGL_DSV4_MTP_ROW0_DEBUG=1 \
MINISGL_DSV4_MTP_SPEC_TRACE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_multi_request_verify_contract/raw/mtp_commit_bs124_draft2_debug.json \
  --page-size 256 --num-pages 16 --decode-len 8 \
  --disable-pynccl --enable-spec --draft-len 2
```

Add any row-depth oracle commands/scripts and record them.

## Success Criteria

Minimum:

```text
bs=1 remains exact
bs=2 exactness passes
bs=4 no longer crashes on mixed verify_lens
accepted commit remains enabled
```

Full:

```text
bs=1/2/4 draft_len=2 exactness passes
row/depth target-verify semantics are documented against SGLang or a justified
mini equivalent
```

## Stop Lines

Stop and report if:

- SGLang uses a target-verify mechanism that is too broad to port here;
- row-depth parity cannot be captured clearly;
- exactness can only pass by disabling accepted commit;
- exactness can only pass by recomputing accepted rows sequentially;
- fixing bs=2 breaks bs=1.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_multi_request_verify_contract/README.md
```

Include:

- SGLang contract notes;
- bs=2 row-depth parity table;
- row packing / acceptance indexing table;
- mixed verify-length design and result;
- exactness matrix;
- fix summary or next narrow target;
- whether TARGET 11.3 graph/perf is unblocked.
