# TARGET 11.6: DSV4 SM80 MTP BS5 Batch-Shape Oracle And State Parity

## Status

Next after TARGET 11.5.

TARGET 11.5 fixed the deterministic `bs=4` accepted-commit exactness gate by
narrowing the proven owner to the mini DSV4 SM80 target-verify attention
consumer for parent batch sizes greater than 2. Accepted commit remains enabled
and accepted rows are not recomputed sequentially.

The lightweight exposure gate still found a smaller new blocker:

```text
bs=5 not exact
draft_len=2
decode_len=8
page_size=256
TP8
CUDA graph disabled
PyNCCL disabled
accepted commit enabled
current parent-bs>2 torch target-verify fallback enabled
```

Do not reopen broad row packing or verify group size exploration. Treat TARGET
11.299 and TARGET 11.5 as sufficient evidence that the row/depth contract and
the `bs=4` accepted-commit class are not the next broad search area.

Important new suspicion: mini has known historical batch-invariance gaps, and
the reduced evidence shows baseline outputs can differ across batch sizes for
the same prompt.  This target must first determine whether the `bs=5` MTP drift
is caused by target-model batch-shape sensitivity rather than by a new
accepted-commit state bug.

## Starting Evidence

Raw outputs:

```text
performance_milestones/target11_mtp_bs4_accepted_commit_state_parity/raw/reduce_baseline_bs5_6_7_after_fix.json
performance_milestones/target11_mtp_bs4_accepted_commit_state_parity/raw/reduce_mtp_bs5_6_7_after_fix.json
```

Smallest reduced failure:

```text
bs=5 mismatch index 4
baseline [9641, 14, 535, 16251, 14, 19106, 14, 4936]
MTP      [9641, 14, 535, 16251, 14, 19106, 14, 12196]
```

Cross-batch warning:

```text
The same prompt/request can legitimately produce different baseline tokens at
different batch sizes in the current mini implementation, because batch
invariance is not yet guaranteed.

Example from TARGET 11.5 raw outputs:
bs=5 baseline req4 ends with 4936
bs=6/7 baseline req4 ends with 12196
```

This does not by itself prove an MTP bug.  The MTP correctness contract is
same-batch exactness:

```text
baseline(bs=N, same scheduler knobs) == MTP(bs=N, same scheduler knobs)
```

However, if MTP changes the later normal target decode microbatch shape for the
same visible prefix, and the target path is batch-shape sensitive, MTP can fail
same-batch exactness even when target-verify row packing and accepted commit are
otherwise correct.

Wider exposure still fails:

```text
bs=8 mismatches request indices 0, 3, 4, 7
bs=16 mismatches request indices 1, 3, 4, 7, 13
```

## Goal

Find and fix or precisely classify the first owner that makes the reduced
`bs=5` exposure non-exact while preserving the TARGET 11.5 `bs=1/2/4`
exactness gate.

Primary question:

```text
For the failing bs=5 req4 prefix, does normal target decode produce different
next-token logits/tokens when the batch shape, active request set, slot, or row
order changes?
```

If yes, the next owner is batch-shape-sensitive target decode/scheduler
parity.  If no, continue to accepted-commit state parity.

The target passes only when:

```text
bs=1/2/4/5 exact
accepted commit remains enabled
the bs=5 failure is classified as batch-shape target-decode sensitivity or
accepted-commit state drift
the parent-bs>2 torch target-verify fallback is either still exact or replaced
by a proven exact fast path
light bs=8/16 exposure is rerun and documented
TARGET 11.3 go/no-go is updated
```

## Work Plan

### 1. Freeze the reduced bs=5 reproducer

Use the same knobs as TARGET 11.5:

```text
TP8 /models/DeepSeek-V4-Flash
page_size=256
draft_len=2
decode_len=8
max_running_req=5
CUDA graph disabled
PyNCCL disabled
accepted commit enabled
current parent-bs>2 torch target-verify fallback enabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
```

Record:

- baseline output;
- MTP output;
- first mismatching request and token index;
- target-verify contract trace;
- normal target forward trace after each accepted commit.

### 2. Separate cross-batch variability from same-batch MTP drift

Make a small table for the failing prompt/request across:

```text
baseline bs=1/2/4/5/6/7/8/16 when practical
MTP bs=1/2/4/5/6/7/8/16 when practical
```

This table is for interpretation only.  Do not require global batch invariance
as a pass condition for MTP.  Use it to identify whether MTP is producing the
token associated with another baseline batch shape.

### 3. Build a normal-target batch-shape oracle for bs=5 req4

At the first failing visible prefix for request index 4, run normal target
decode or a narrow forward oracle with identical request state under multiple
batch contexts:

```text
solo req4 only
req4 in its original bs=5 slot
req4 in a bs=5 batch with same active req ids/order as baseline
req4 in a bs=5 batch with same active req ids/order as MTP after accepted commits
req4 padded with inert requests if needed
req4 moved to another slot if easy
```

For each case record:

- active request ids and order;
- slot / table index;
- input token;
- position;
- full loc / SWA loc / component locs;
- target forward phase and metadata;
- top-k logits and margin;
- predicted next token.

Expected classification:

```text
If the same prefix flips between 4936 and 12196 under normal target decode
only because batch shape/order changes, root cause is batch-shape-sensitive
target decode/scheduler parity.

If normal target decode is shape-invariant for this prefix, root cause remains
MTP accepted-commit state drift.
```

### 4. If batch-shape sensitive, find the first batch-sensitive owner

Use hidden/logit parity bisection between the shape contexts above.  Reuse the
style from TARGET 11.297/11.298:

- compare row0/req4 hidden states layer by layer;
- record first owner and max_abs_delta;
- check whether the owner is attention, `wo_a`, metadata/indexer, MoE, lm head,
  or sampler;
- check whether the owner is a known mini batch-invariance gap.

Do not immediately "fix" by forcing every normal target decode to solo mode.
That would make MTP exact but destroy serving throughput.  Find the smallest
batch-sensitive owner first.

### 5. If shape-invariant, continue accepted-commit state parity

Only after the batch-shape oracle is negative, compare against sequential
baseline at the same visible token prefix:

- full/SWA rows;
- C4 compressed/indexer rows and state;
- C128 compressed rows;
- online C128 MTP state;
- page/component mapping;
- request `cached_len` / `device_len`.

Check whether the failure starts before accepted commit, during accepted commit,
or after later normal target decode reads committed state.

### 6. Fix the smallest proven owner

Do not:

- disable accepted commit;
- sequentially recompute accepted rows;
- broaden verify-group-size search unless direct evidence points there;
- paper over batch-shape sensitivity by making all MTP decode single-request.

Prefer a fix that preserves normal serving batch execution while making the MTP
path exact for the same-batch baseline.

### 7. Rerun gates

After a fix:

```text
bs=1/2/4/5 exact
bs=8/16 light exposure
```

If `bs=5` passes but `bs=8+` still fails, reduce to the next smallest failing
batch and write a narrow follow-up.  Do not debug `bs=16` directly unless it is
the smallest remaining reproducer.

## Stop Lines

Stop and report if:

- the bs=5 failure is proven to be target batch-shape sensitivity but the first
  owner cannot be isolated;
- normal target decode changes token under different batch shapes before any
  MTP-specific commit is involved;
- exactness only passes by disabling accepted commit;
- exactness only passes by sequentially recomputing accepted rows;
- a fix for bs=5 breaks the TARGET 11.5 `bs=1/2/4` gate;
- the next failure requires a new smallest reproducer.

## Expected Writeup

Create:

```text
performance_milestones/target11_mtp_bs5_exposure_state_parity/README.md
```

Include:

- exact commands and env flags;
- cross-batch baseline/MTP variability table;
- bs=5 normal-target batch-shape oracle result;
- first bad event or first batch-sensitive owner;
- component parity table if the oracle points to accepted-commit state;
- fix summary;
- validation matrix for `bs=1/2/4/5` plus `bs=8/16`;
- explicit TARGET 11.3 go/no-go.
