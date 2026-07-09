# TARGET 11.297: DSV4 SM80 MTP Target-Verify Batch-Shape Hidden Parity

## Status

Next after TARGET 11.296.

TARGET 11.296 fixed the visible bs=1/draft_len=2 `223` token drift by routing
target-verify decode rows through the splitk decode attention path.  The narrow
smoke now emits the same greedy token ids as baseline, but full row0 logits are
still not equal:

```text
normal one-row decode row0 top1 = 582
target-verify row0 top1     = 582
selected logit deltas       ~= 0.10-0.12
```

Do not work on CUDA graph, macro throughput, serving datasets, or broad MTP
promotion in this target.

## Goal

Find the first layer/submodule where row0 hidden states diverge between:

```text
normal one-row target decode
```

and

```text
flattened multi-row MTP target verify, row0
```

under the same committed prefix, input token, position, and KV/component state.

If practical, fix the smallest proven owner.  If the first owner is a broader
SGLang target-verify execution mechanism, stop and write the next narrow porting
target.

## Starting Evidence

From TARGET 11.296:

```text
normal oracle at final row:
  input_ids = [3362]
  positions = [10]
  seq_lens = [11]
  req_seq_lens = [11]
  extend_lens = [1]
  max_seqlen_q = 1
  max_seqlen_k = 11
  top1 = 582
  logits[582] = 32.230064
  logits[223] = 32.139542

target verify final row0:
  input_ids = [3362, 582]
  positions = [10, 11]
  seq_lens = [11, 12]
  req_seq_lens = [12]
  extend_lens = [2]
  max_seqlen_q = 2
  max_seqlen_k = 12
  top1 = 582
  logits[582] = 32.123913
  logits[223] = 32.016010
```

The token now matches, but logits still differ.  Because the top1/top2 margin is
small, the current exact smoke is not enough to unblock graph/perf.

The same-run oracle catches row0 logit deltas before accepted commit is the
obvious first owner, so do not start by changing accepted-commit copy direction.

## SGLang References

Use these only where they clarify target-verify execution semantics:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py
```

The target should report whether mini's flattened target-verify batch shape is
still materially different from SGLang's target-verify execution.

## Work Plan

### 1. Preserve controls

Keep these controls:

- baseline normal greedy decode;
- accepted-commit MTP bs=1/draft_len=2 exact smoke from TARGET 11.296;
- same-run one-row normal decode oracle;
- CUDA graph disabled;
- PyNCCL disabled for exactness runs.

Do not hide the remaining gap by disabling accepted commit or by recomputing all
accepted rows through normal sequential decode.

### 2. Verify target-verify attention backend selection

Before deeper bisection, confirm in debug output that target-verify rows really
take the intended splitk decode-row sparse attention path for every relevant
layer.

Record per layer:

```text
layer_id
target_verify_decode_rows flag
sparse attention backend selected
rows
max_seqlen_q
max_seqlen_k
```

If any target-verify row still uses the wrong attention path, fix that first and
rerun the row0 oracle.

### 3. Add row0 per-layer hidden parity capture

Add debug-only instrumentation behind an env flag such as:

```text
MINISGL_DSV4_MTP_ROW0_LAYER_PARITY=1
```

For the same-run oracle and flattened target verify, capture row0 values after
each major layer boundary:

```text
layer input
post attention / attention output
post residual after attention
post MLP/MoE output
post residual after MLP/MoE
hidden before final norm if available
final logits
```

To keep logs small, store compact summaries:

```text
max_abs_delta
mean_abs_delta
relative delta
top few differing dims
small hash/checksum
dtype
shape
```

Do not dump full tensors unless explicitly gated and small.

### 4. Find the first divergent owner

Run the failing/sensitive bs=1 case and compare:

```text
normal one-row decode oracle
vs
flattened target-verify row0
```

Find the first layer and first submodule boundary where row0 delta becomes
meaningful.

Use a rough tolerance appropriate for BF16/A100 numerics.  A tiny BF16-level
rounding delta is not enough to block progress; a stable delta that can flip
top1 on small margins is.

### 5. Bisect within the first divergent layer

Depending on the first owner, inspect:

- attention path:
  - query/key/value inputs;
  - sparse attention indices;
  - SWA/C4/C128/indexer metadata;
  - splitk decode-row output vs one-row output;
  - whether row0 attends to row1 or sees row1 state accidentally.
- indexer/compressor path:
  - row0 compressor/indexer input;
  - C4/C128 write locs;
  - state update side effects;
  - row coupling from flattened batch.
- MoE/MLP path:
  - router top-k for row0;
  - grouped/expert dispatch row selection;
  - batch-shape-dependent GEMM/MoE path;
  - all-reduce/reduce owner if touched.
- norms/residuals:
  - shape-dependent numerical path;
  - residual ordering or dtype casts.

### 6. Fix or split

If the owner is local and small, implement the fix and rerun:

```text
bs=1/draft_len=2 accepted-commit exact smoke
row0 logits/top-k parity
same-run oracle parity
```

If the owner is broad, write the next target with a precise owner, such as:

```text
MTP target-verify attention metadata parity
MTP target-verify MoE row-coupling parity
MTP target-verify SGLang execution-shape port
```

## Verification Gates

Minimum commands:

```bash
MINISGL_DSV4_MTP_ROW0_DEBUG=1 \
MINISGL_DSV4_MTP_SPEC_TRACE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_target_verify_batch_shape_hidden_parity/raw/mtp_commit_bs1_draft2_oracle.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl --enable-spec --draft-len 2
```

Add the new layer-parity env/debug flag once implemented.

Required before TARGET 11.3 can be unblocked:

```text
baseline greedy token ids == MTP token ids
row0 normal-decode logits and target-verify row0 logits are equal within an
  agreed tolerance, or the remaining difference is proven harmless across a
  broader exactness gate
accepted commit remains enabled
draft_tokens_accepted > 0
accepted_kv_commit_blocker is empty/None
```

If full logit equality is unrealistic because of unavoidable BF16 batch-shape
math, prove stability with a narrow exactness expansion before graph/perf:

```text
bs=1/2/4
draft_len=2
several short prompts with low target top1/top2 margins if available
```

Do not run macro throughput in this target.

## Stop Lines

Stop and report if:

- instrumentation cannot identify the first divergent owner;
- the first owner is broad enough to need a dedicated SGLang parity port;
- exactness can only pass by disabling accepted commit;
- exactness can only pass by recomputing accepted rows sequentially.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_batch_shape_hidden_parity/README.md
```

Include:

- exact commands;
- baseline vs MTP token ids;
- target-verify attention backend selection table;
- row0 per-layer/submodule parity table;
- first divergent layer/submodule owner;
- fix summary or next narrow target;
- whether TARGET 11.3 graph/perf is unblocked.
