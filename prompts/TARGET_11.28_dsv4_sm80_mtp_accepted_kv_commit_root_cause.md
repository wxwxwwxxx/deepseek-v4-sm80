# TARGET 11.28: DSV4 SM80 MTP Accepted-KV Commit Root Cause

## Status

Next after TARGET 11.27 no-go.

TARGET 11.27 proved flattened verify shape and rollback-only correctness, but
accepted-KV commit is disabled because committing flattened verify KV changed
later greedy output.  This target should find the exact owner of that mismatch.

Do not optimize throughput.  Do not enable CUDA graph.  Do not promote MTP.

## Goal

Make accepted-KV commit exact for the smallest possible deterministic case, or
produce a strong no-go explaining why mini's current DSV4 metadata/runtime
cannot safely commit flattened verify KV yet.

The first passing gate is:

```text
bs=1, draft_len=2, first accepted token commit
baseline greedy token ids == MTP flattened verify + accepted-KV commit token ids
```

## Starting Evidence From TARGET 11.27

What worked:

- flattened target verify shape was implemented;
- rollback-only path preserved exact greedy token ids for bs=1/2/4 and
  draft_len=2/4;
- rejected-tail isolation worked;
- DSV4 state snapshot/restore was broad enough for rollback correctness.

What failed:

- accepted-KV commit was disabled in the final exact path;
- direct commit variants, including a conservative first-row-only attempt,
  changed later greedy output;
- final exact path had `accepted_kv_copied_tokens=0` and `target_commit_kv_copies=0`;
- performance got worse because flattened verify added target work without
  making accepted verify rows visible.

Therefore the blocker is not MTP weight loading, draft generation, or frozen-KV
read-only semantics.  The blocker is exact DSV4 accepted-state commit from
flattened verify.

## Key Hypothesis

Committing an accepted flattened verify row is not equivalent to committing the
same token through normal sequential target decode.

Possible owners:

- raw target KV row differs;
- SWA compact row differs;
- C4/C128 compressed rows differ;
- C4 indexer row differs;
- compression/indexer state pool differs;
- online C128 pending/commit semantics are missing;
- positions, sequence lengths, or `out_loc` differ between flattened verify and
  normal decode;
- the first verify row uses the wrong token boundary or hidden/HC state;
- accepted row commit copies only part of DSV4 state.

## SGLang References

Use SGLang as the source oracle:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_worker_v2.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Map specifically:

- verify input token order;
- verify positions;
- temporary verify `out_cache_loc` / commit destination;
- which KV/component/cache rows are moved or marked pending;
- how C128 online compression state is committed;
- how rejected tails are discarded.

## Source-First Investigation

Do not start with broad layer/component bisection.  First build a source-derived
checklist from SGLang and compare mini against it.  Only run expensive bisection
after these higher-probability mismatches are ruled out.

Prioritized SGLang-derived checks:

1. Verify token count and token order.
   - SGLang target-verify metadata uses `speculative_num_draft_tokens` rows per
     request, not an arbitrary `accepted + correction` layout.
   - In `eagle_worker_v2.py`, top-k 1 accepted path is already the front chain;
     no tree compaction is needed.
   - Confirm mini's flattened verify input order and accepted-index semantics
     match that front-chain rule before copying any KV.

2. Target-verify `seq_lens` and `extend_lens`.
   - `deepseek_v4_backend.py:init_forward_metadata_target_verify_old` sets
     `seq_lens = seq_lens + speculative_num_draft_tokens` and
     `extend_seq_lens = [speculative_num_draft_tokens] * batch_size`.
   - `make_forward_metadata_from_raw_verify` repeats the same assumption for
     graph/raw metadata.
   - Confirm mini does not verify with per-request shortened lengths after a
     mismatch; SGLang builds metadata for the fixed draft-token count.

3. `out_cache_loc` geometry.
   - SGLang's target verify metadata is parameterized by `out_cache_loc` for the
     flattened verify rows.
   - For top-k 1, the accepted path is expected to be placed in the front chain.
     If mini writes verify rows to scratch locations and then copies them, the
     destination mapping must be exactly equivalent to SGLang's front-chain
     layout.

4. Online C128 MTP pending/commit semantics.
   - `OnlineC128MTPController.prepare_forward()` commits pending state from the
     previous verify, then `begin_verify()` marks new pending state.
   - `write_prefix_states()` writes ratio-128 verify prefix states.
   - `commit_pending()` later commits those pending states into the main
     compression-state pool.
   - This is a high-priority suspect.  If mini copies raw/C4/C128 rows but does
     not reproduce pending/commit state semantics, accepted flattened KV may not
     equal sequential target decode.

5. Draft rope position is frozen, verify position is not.
   - `frozen_kv_mtp_utils.set_frozen_kv_positions()` freezes draft rope phase to
     `seq_lens - 1`.
   - Target verify metadata instead advances over the draft-token extension.
   - Confirm mini does not accidentally reuse frozen draft positions for target
     verify rows.

6. Rejected-tail visibility.
   - SGLang makes rejected tails not visible through accepted path selection and
     later freeing/overshoot handling.
   - Mini's rollback-only path is correct but too conservative.  Any commit fix
     must keep rejected tails invisible without restoring accepted front-chain
     rows.

Required output before bisection:

- a table with each item above marked `same`, `different`, or `unknown`;
- exact mini code locations responsible for each item;
- if an item is `different`, test that hypothesis before doing full
  layer/component scans.

## Work Plan

1. Preserve the TARGET 11.27 rollback-only path as the correctness control.
2. Complete the source-first checklist above.
3. Add low-cost runtime assertions/logging for the top suspects:
   - verify token order;
   - `seq_lens` before and after verify metadata;
   - `extend_lens`;
   - `out_cache_loc` source and commit destination;
   - target verify positions;
   - C128 pending/commit state ownership if present.
4. Build a minimal deterministic repro:
   - TP8 real model;
   - bs=1;
   - `draft_len=2`;
   - decode_len small;
   - one known prompt with an accepted first draft token if possible.
5. Test the highest-priority source-derived hypotheses first:
   - token order / accepted-index front-chain mismatch;
   - target verify length/position mismatch;
   - `out_cache_loc` front-chain mapping mismatch;
   - missing online C128 pending/commit semantics.
6. Only if those do not explain the bug, capture baseline sequential target
   state after the same accepted token.
7. Capture flattened verify temp/commit candidate state for that token.
8. Compare per layer and per component:
   - raw/full KV;
   - SWA KV;
   - C4 compressed KV;
   - C128 compressed KV;
   - C4 indexer KV;
   - compression/indexer state pool;
   - hidden/logits before the next token.
9. Find the first component/layer where flattened commit diverges from baseline.
10. Implement the smallest fix and rerun exactness.
11. Only after bs=1 first-row commit is exact, extend to:
   - bs=1 full accepted prefix;
   - bs=2/4;
   - `draft_len=4`.

## Guardrails

- Keep graph disabled.
- Keep MTP opt-in only.
- Do not optimize target verify latency until commit exactness passes.
- Do not weaken rejected-tail isolation or duplicate-free guards.
- Do not hide mismatch by recomputing accepted tokens unless it is explicitly
  recorded as an oracle path, not the performance path.

## Success Criteria

Minimum success:

- first accepted row commit is exact for bs=1/draft_len=2;
- the report identifies which DSV4 component caused the 11.27 commit mismatch;
- rejected tails remain isolated.

Full success:

- accepted-KV commit is exact for bs=1/2/4 and draft_len=2;
- draft_len=4 either passes or has a smaller documented blocker;
- target verify can make accepted rows visible without rollback-only fallback.

## No-Go Criteria

Stop and report if:

- the first-row commit mismatch cannot be localized after layer/component
  bisection;
- exact commit requires recomputing target tokens, making flattened verify
  structurally unable to reduce target passes;
- SGLang's required C128/pending-state mechanism is too large to port in this
  target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_accepted_kv_commit_root_cause/README.md
```

Include:

- exact repro command;
- SGLang source-parity notes, including the source-first checklist;
- first mismatching layer/component;
- before/after exactness matrix;
- whether TARGET 11.27 should be rerun, split further, or closed as no-go.
