# TARGET 11.6: DSV4 SM80 MTP BS5 Exposure State Parity

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
11.299 and TARGET 11.5 as sufficient evidence that the next owner is
request-local target-verify row content/state parity under the larger batch
schedule.

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

Wider exposure still fails:

```text
bs=8 mismatches request indices 0, 3, 4, 7
bs=16 mismatches request indices 1, 3, 4, 7, 13
```

## Goal

Find and fix the first state owner that makes the reduced `bs=5` exposure
non-exact while preserving the TARGET 11.5 `bs=1/2/4` exactness gate.

The target passes only when:

```text
bs=1/2/4/5 exact
accepted commit remains enabled
the parent-bs>2 torch target-verify fallback is either still exact or replaced
by a proven exact fast path
light bs=8/16 exposure is rerun and documented
TARGET 11.3 go/no-go is updated
```

## Work Plan

1. Freeze the reduced `bs=5` reproducer with the same knobs as TARGET 11.5.
2. Locate the first bad commit or first bad normal target decode for request
   index 4.
3. Compare against sequential baseline at the same visible token prefix:
   full/SWA rows, C4 compressed/indexer rows and state, C128 compressed rows,
   online C128 MTP state, page/component mapping, and request
   cached_len/device_len.
4. Check whether the failure starts before accepted commit, during accepted
   commit, or after later normal target decode reads committed state.
5. Fix the smallest proven owner. Do not disable accepted commit, do not
   sequentially recompute accepted rows, and do not broaden verify-group-size
   search unless direct evidence points there.
6. Rerun exact `bs=1/2/4/5` and light exposure `bs=8/16`.

## Expected Writeup

Create:

```text
performance_milestones/target11_mtp_bs5_exposure_state_parity/README.md
```

Include:

- exact commands and env flags;
- first bad event;
- component parity table;
- fix summary;
- validation matrix for `bs=1/2/4/5` plus `bs=8/16`;
- explicit TARGET 11.3 go/no-go.
