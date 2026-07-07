# TARGET 11.298: DSV4 SM80 MTP wo_a Projection Batch-Shape Parity

## Status

Next after TARGET 11.297.

TARGET 11.297 identified the first stable row0 mismatch owner:

```text
first owner = layer0.attention_wo_a_output
previous boundary = layer0.merged_attention_output_after_inverse_rope
previous boundary max_abs_delta = 0.0
```

All target-verify sparse attention calls already use the intended
`splitk_target_verify` path, and accepted commit remains enabled.  The remaining
problem is that `wo_a` projection output is not row-invariant between normal
one-row decode and flattened target-verify batch shape.

Do not work on CUDA graph, macro throughput, serving datasets, or broad MTP
promotion in this target.

## Goal

Make target-verify row0 `wo_a` projection output match normal one-row decode for
the same row0 input and same `wo_a` weight/cache, or prove the exact projection
backend/shape owner and write the next narrow target.

Primary passing condition:

```text
normal one-row decode row0 attention_wo_a_output
  ==
flattened target-verify row0 attention_wo_a_output
```

within a justified tolerance, while keeping:

```text
accepted commit enabled
draft_tokens_accepted > 0
baseline greedy token ids == MTP token ids
```

## Starting Evidence

From TARGET 11.297:

```text
embedding                                      delta = 0.0
layer0.input                                   delta = 0.0
layer0.attention_input                         delta = 0.0
layer0.merged_attention_output_after_inverse_rope = 0.0
layer0.attention_wo_a_output                   delta = 0.0078125 / 0.015625
layer0.final_attention_output                  delta ~= 0.06-0.09
lm_head_logits                                 delta ~= 0.63-0.87
```

The active path in the run was:

```text
wo_a_bf16_bmm_cache.enabled = True
wo_a_bf16_bmm_cache.layers_cached = 43
disabled_toggles = []
```

So the highest-probability owner is the A100 victory `wo_a` BF16 BMM cache
projection under different M shapes.

## Work Plan

### 1. Preserve controls

Keep:

- TP8 `/models/DeepSeek-V4-Flash`;
- `page_size=256`;
- CUDA graph disabled;
- PyNCCL disabled for exactness;
- accepted commit enabled;
- the TARGET 11.297 row0 hidden-parity instrumentation.

Do not hide the issue by disabling accepted commit or recomputing accepted rows
through normal sequential decode.

### 2. Locate the current `wo_a` projection paths

Map source locations for:

- `wo_a_bf16_bmm_cache`;
- normal decode `wo_a` projection;
- target-verify `wo_a` projection;
- any fallback dequant + einsum/matmul path;
- any Triton/grouped/FP8 projection path that may be selected by env toggles.

Record which path is active for:

```text
normal one-row decode
flattened target verify M=2/M=3
```

### 3. Build a no-model or partial-model micro-oracle

Create a focused microbench/oracle that compares identical row0 input and
identical weight/cache under different batch shapes:

```text
M=1 normal row0 projection
M=2 flattened target-verify row0 projection
M=3 flattened target-verify row0 projection
```

Compare at least:

- current `wo_a_bf16_bmm_cache` path;
- fallback dequant + torch/einsum/matmul path;
- any available existing `wo_a` backend alternative.

Record:

```text
max_abs_delta
mean_abs_delta
top differing dims
dtype
shape
backend
whether row0 output changes when extra rows are appended
```

If the micro-oracle cannot be built without the full model, keep it as a
smallest possible partial-layer or single-layer harness.  Do not start with a
full macro benchmark.

### 4. Decide whether this is numerical or semantic

Classify the delta:

```text
numerical:
    BF16 accumulation/order difference, stable small deltas, no row coupling in
    exact fallback.

semantic/backend bug:
    row0 changes only in the cached/BMM path, cache indexing or reshape depends
    on M, or fallback row0 is equal while cached path is not.
```

If it is only unavoidable BF16 batch-shape math, prove whether the drift is
harmless with a narrow exactness expansion.  If it is a backend bug, fix the
backend or route target verify through a row-invariant projection path.

### 5. Fix options

Prefer, in order:

1. Fix `wo_a_bf16_bmm_cache` so row0 output is invariant to appended rows.
2. Add a target-verify-specific row-invariant `wo_a` path if the cached path is
   inherently batch-shape dependent.
3. Temporarily route target verify through fallback projection only if the
   correctness win is clear and the report records the performance risk.

Do not promote a slow fallback by default without later profiling.  The goal is
correctness and owner closure first.

### 6. Re-run exactness and parity

After any fix:

- rerun the `wo_a` micro-oracle;
- rerun bs=1/draft_len=2 accepted-commit exactness;
- rerun row0 hidden parity enough to show the first owner moved or disappeared;
- if fixed, run a narrow exactness expansion:
  - bs=1/2/4;
  - draft_len=2;
  - graph disabled;
  - pynccl disabled.

Do not run macro throughput in this target.

## Verification Commands

Minimum TP8 command:

```bash
MINISGL_DSV4_MTP_ROW0_DEBUG=1 \
MINISGL_DSV4_MTP_ROW0_LAYER_PARITY=1 \
MINISGL_DSV4_MTP_SPEC_TRACE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_wo_a_projection_batch_shape_parity/raw/mtp_commit_bs1_draft2_woa.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl --enable-spec --draft-len 2
```

Add micro-oracle commands/scripts as needed and record them.

## Success Criteria

Minimum success:

```text
first mismatch owner no longer layer0.attention_wo_a_output
bs=1/draft_len=2 accepted-commit exactness passes
accepted commit remains enabled
```

Full success:

```text
wo_a row0 projection parity passes for M=1 vs M=2/M=3
row0 full-logit parity is equal within agreed tolerance, or remaining delta is
  proven harmless by narrow exactness expansion
bs=1/2/4 draft_len=2 exactness passes
```

## Stop Lines

Stop and report if:

- the `wo_a` micro-oracle cannot reproduce the row0 delta;
- the delta is in an upstream tensor despite TARGET 11.297 evidence;
- the only exact fix disables accepted commit;
- the only exact fix recomputes all accepted rows sequentially;
- the owner turns out to be a broader target-verify execution-shape mismatch
  requiring a separate SGLang parity port.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_wo_a_projection_batch_shape_parity/README.md
```

Include:

- source map of `wo_a` projection paths;
- active backend table for normal decode vs target verify;
- micro-oracle results;
- before/after row0 hidden/logit parity;
- exactness matrix;
- fix summary or next narrow target;
- whether TARGET 11.3 graph/perf is unblocked.
