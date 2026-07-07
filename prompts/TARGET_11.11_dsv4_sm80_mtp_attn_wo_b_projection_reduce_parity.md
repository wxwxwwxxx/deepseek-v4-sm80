# TARGET 11.11: DSV4 SM80 MTP attn.wo_b Projection/Reduce Parity

## Status

Next after TARGET 11.10.

TARGET 11.10 fixed the TARGET 11.9 first owner:

```text
old owner: layer0.merged_attention_output_before_wo
runtime: MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
fix: use decode-row target-verify attention for the SGLang-shaped runtime
```

After that fix, layer0 metadata, KV producer/store, KV gather/readback, causal
lengths, attention kernel output, and `attention_wo_a_output` are exact for the
bs=1 and bs=2 targeted traces.  The new first owner is:

```text
layer0.final_attention_output
```

The input to that boundary is exact:

```text
layer0.attention_wo_a_output max_delta = 0.0
```

So the remaining first owner is the layer0 `attn.wo_b` row-parallel projection
and all-reduce boundary.

## Goal

Make `sglang_prefill_extend` target-verify `attn.wo_b` produce the same output
as normal target decode for the same visible row.

The target passes when:

```text
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
bs=1 exact
layer0.attention_wo_a_output exact
layer0.final_attention_output exact
accepted commit remains enabled
```

Then expand to:

```text
bs=1/2/4/5/6 exact, or fail with a new first owner after wo_b parity is proven
```

Do not start CUDA graph, throughput tuning, C128 boundary gates, or speculative
acceptance tuning in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer0_attention_kv_parity/README.md
performance_milestones/target11_mtp_sglang_aligned_target_verify_runtime_mode/README.md
prompts/TARGET_11.10_dsv4_sm80_mtp_target_verify_layer0_attention_kv_producer_parity.md
```

Important TARGET 11.10 result:

```text
layer0.attention_wo_a_output: exact
layer0.final_attention_output: max_delta = 0.0625
owner: attn.wo_b row-parallel projection/all-reduce
```

Exactness after 11.10:

```text
sglang_prefill_extend bs=1: exact
sglang_prefill_extend bs=2/4/5/6: not exact
accepted commit: enabled and healthy
```

Interpretation:

```text
MTP target-verify metadata, KV producer/store, attention consumer, and wo_a are
no longer the first owner.  Do not reopen them unless wo_b attribution proves
the prior trace was invalid.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/communication.py
```

Relevant mini locations:

```text
python/minisgl/models/deepseek_v4.py:2098-2124
DSV4Linear.forward_fp8_cached_bf16_weight
DSV4Linear.forward_fp8_marlin_weight
DSV4Linear.forward
reduce_label="dsv4.attn.wo_b.row_parallel_projection_all_reduce"
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/linear.py
/workspace/sglang-main/python/sglang/srt/layers/communication.py
```

Relevant SGLang behavior:

```text
o, _ = self.wo_b(o.flatten(1))
if self.tp_size > 1:
    o = attn_tp_all_reduce(o)
```

Prefer SGLang's `RowParallelLinear + attn_tp_all_reduce` semantics when mini
and SGLang differ, unless mini has a measured and documented reason to
intentionally diverge.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, or observed
  token numerical branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6` as the fix.
- Do not sequentially recompute accepted rows as the final runtime.
- Do not retune attention metadata/KV producer unless the wo_b trace proves the
  11.10 boundary attribution was wrong.
- Do not start graph/perf work.

## Work Plan

### 1. Reproduce The New First Owner

Run the focused repro:

```text
TP8
/models/DeepSeek-V4-Flash
page_size=256
num_pages=16
draft_len=2
decode_len=8
CUDA graph disabled
PyNCCL disabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
accepted commit enabled
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
bs=1 and bs=2 targeted traces
```

Confirm:

```text
layer0.attention_wo_a_output exact
layer0.final_attention_output non-exact
```

If the first owner changes, document the change and follow the new first owner.

### 2. Make SGLang wo_b Semantics The Contract

Before testing many kernel combinations, inspect SGLang's `wo_b` path and write
a source-parity table:

```text
concept
SGLang behavior
mini normal decode behavior
mini target-verify behavior
same/different/unknown
action
```

Cover at least:

- input shape to `wo_b` after `o.flatten(1)`;
- row ordering for normal decode and target verify;
- dtype at `wo_b` input and output;
- whether weight path is fp8, cached bf16, Marlin, or fallback;
- bias handling, if any;
- local row-parallel matmul output before reduce;
- all-reduce dtype, op, stream, and communication backend;
- whether normal decode and target verify use the same reduce path;
- whether mini's PyNCCL/torch.distributed default changes the result in this
  no-PyNCCL correctness run;
- whether a cached BF16 or Marlin `wo_b` path changes row-shape numerical
  behavior.

If mini only matches the high-level shape but not the `RowParallelLinear +
attn_tp_all_reduce` semantics, port or adapt the missing behavior rather than
trying many local kernel flags.

### 3. Split wo_b Into Local Projection And Reduce Boundaries

Add or reuse debug traces around:

```text
wo_b input = layer0.attention_wo_a_output
wo_b local matmul output before reduce
wo_b reduce input dtype/shape
wo_b post-all-reduce output
layer0.final_attention_output
```

Compare normal target decode row0 and target-verify row0 for the same prefix.

Classify the owner as one of:

```text
wo_b input shape/flattening mismatch
wo_b local matmul kernel/path mismatch
wo_b weight cache or Marlin layout mismatch
wo_b dtype/cast mismatch
wo_b all-reduce semantic mismatch
wo_b reduce stream/order mismatch
debug oracle construction mismatch
```

Do not proceed to full bs matrix until this boundary is classified.

### 4. Test Minimal Correctness Oracles

Acceptable temporary probes:

- force `wo_b` fallback for normal decode and target verify under the same
  runtime mode;
- force cached BF16 `wo_b` for both paths;
- force the same communication backend for both paths;
- trace rank-local pre-reduce tensors and post-reduce tensors;
- compare only layer0 row0 tensors under a debug env.

Not acceptable as final behavior:

- a bs-specific `wo_b` branch;
- disabling reduce for target verify;
- silently changing global projection precision without a correctness note;
- using sequential recompute accepted rows as the final MTP path.

### 5. Fix Or Precisely No-Go

Preferred fixes:

1. If SGLang semantics differ:
   - port/adapt the relevant `RowParallelLinear + attn_tp_all_reduce` behavior
     for target-verify rows.
2. If mini local matmul path differs by row shape:
   - make the `wo_b` projection use a row-invariant correctness path for
     target verify.
3. If cached BF16 or Marlin layout is the owner:
   - either make it row-invariant or disable that specific `wo_b` fast path for
     `sglang_prefill_extend` until a later performance target.
4. If all-reduce is the owner:
   - align target verify with normal decode's reduce dtype/op/backend and
     document whether PyNCCL/torch.distributed changes correctness.
5. If the oracle is wrong:
   - fix the oracle and rerun the boundary trace before touching runtime.

If the necessary fix is a larger SGLang linear/communication port, stop with a
precise plan instead of adding another approximate mini-local path.

### 6. Validate Incrementally

Minimum after the fix:

```text
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
bs=1 exact
layer0.attention_wo_a_output max_delta = 0.0
layer0.final_attention_output max_delta = 0.0
accepted_kv_commit_fail_closed = false
target_commit_kv_copies > 0
accepted_kv_copied_tokens > 0
```

Then run:

```text
bs=1/2/4/5/6 exactness matrix
bs=7/8/16 light exposure only if bs=1/2/4/5/6 passes
```

If another owner appears, report the first failing layer/submodule and confirm
it is not a new batch-size special case.

## Success Criteria

Minimum:

```text
wo_b owner is split into local projection vs all-reduce vs oracle
classification is supported by tensor evidence
no new batch-size or request-slot branch is introduced
accepted commit remains enabled
```

Full:

```text
sglang_prefill_extend bs=1 exact
layer0 final_attention_output exact
bs=1/2/4/5/6 exact, or next first owner is precisely identified
TARGET 11.3 remains blocked or unblocked with explicit evidence
```

## Stop Lines

Stop and report if:

- exactness requires a parent batch size, active verify length, or request slot
  branch;
- exactness requires disabling accepted commit;
- normal decode and target-verify cannot share a `wo_b` correctness path without
  a broader linear/communication port;
- the owner is communication backend dependent and needs a dedicated comm target;
- layer0 `wo_b` parity passes but full exactness fails at a new owner that needs
  a separate correctness target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_wo_b_projection_reduce_parity/README.md
```

Include:

- reproduction of the TARGET 11.10 new first owner;
- SGLang `wo_b` source-parity table;
- local projection vs all-reduce boundary trace;
- implementation summary or precise no-go;
- exactness matrix for `sglang_prefill_extend`;
- accepted commit stats;
- decision on whether TARGET 11.3 can start or the next correctness target is
  needed.
