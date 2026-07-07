# TARGET 11.25: DSV4 SM80 MTP Frozen-KV Verify Runtime

## Status

Next after TARGET 11.2.

TARGET 11.2 proved the conservative greedy sidecar correctness gate, but it did
not implement the runtime that can improve throughput.  This target should build
the first real MTP speculative runtime while keeping CUDA graph disabled.

## Goal

Implement an opt-in greedy top-k 1 MTP runtime that:

1. uses a frozen target KV read-only draft path;
2. proposes more than one token when `draft_len > 1`;
3. verifies proposed tokens with the target model;
4. accepts the longest matching prefix;
5. commits only accepted target tokens to `Req`, token pool, KV/cache, prefix,
   SWA, and component ownership state;
6. preserves exact greedy token ids versus the non-MTP baseline.

Do not optimize graph replay in this target.  Do not promote by default.

## Starting Evidence From TARGET 11.2

The V1 sidecar is healthy for its narrow scope:

- real TP8 `/models/DeepSeek-V4-Flash` MTP load smoke passes;
- real one-step GPU MTP smoke passes with finite logits;
- TP8 greedy exactness passes for batch sizes 1, 2, and 4;
- extra memory is acceptable: about `0.475 GiB/rank` observed, close to the
  `0.455 GiB/rank` MTP weight ledger;
- sidecar agreement rate was `36 / 49 = 73.47%`, but this is not a throughput
  projection because V1 still advances one target token per request.

V1 limitations that this target must resolve:

- `draft_len=1` only;
- no frozen-KV read-only draft attention;
- no multi-token target verification;
- no accepted-token speedup;
- CUDA graph disabled.

## SGLang References

Use SGLang as the runtime oracle first:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Key behaviors to preserve or explicitly justify changing:

- draft reads target KV read-only and owns no KV pool;
- draft positions follow the frozen target sequence state;
- target verification decides acceptance;
- rejected draft tokens never become visible in target KV/cache/prefix state;
- top-k 1 first; no tree/top-k > 1 in this target;
- DSV4 C4/C128/indexer/compression-state behavior must stay correct.

## Implementation Plan

1. Add a clear runtime state model.
   - store the last target hidden needed by MTP per request;
   - keep draft hidden/token state separate from committed request state;
   - add metrics for proposed, verified, accepted, rejected, fallback, and
     finite failures.

2. Add frozen-KV draft metadata.
   - no target KV writes during draft;
   - no prefix/SWA/component ownership mutation during draft;
   - match SGLang position behavior as closely as possible;
   - start eager and debug-friendly.

3. Add multi-token target verify.
   - flatten proposed tokens into a verify batch if practical;
   - write target KV only for locations that can be committed safely, or use a
     temporary/rollback-safe policy for rejected tail tokens;
   - compare target greedy predictions against draft tokens;
   - compute accepted prefix length per request.

4. Update scheduler/request state.
   - current mini has one-token decode semantics; add MTP-specific multi-token
     acceptance without breaking the default path;
   - append accepted tokens only;
   - handle EOS inside an accepted prefix;
   - update token pool/page table/KV/cache only for accepted tokens;
   - preserve prefix-cache disabled and enabled paths when the code touches
     ownership state.

5. Keep CUDA graph disabled.
   - after exact eager runtime is healthy, TARGET 11.3 can add graph buckets and
     performance closure.

## Correctness Gates

Minimum gates:

- greedy token ids exactly match baseline for batch sizes 1, 2, 4;
- `draft_len=2` exactness passes;
- `draft_len=4` exactness passes or has a clear blocker;
- no乱码 text smoke;
- finite logits/hidden checks;
- no rejected-token leakage into KV/cache/token pool;
- prefix cache disabled path passes;
- prefix cache enabled path is either proven or explicitly deferred with a
  fail-closed guard.

Exactness must compare token ids, not only decoded text.

## Metrics

Record:

- proposed tokens;
- verified tokens;
- accepted tokens;
- acceptance histogram;
- average accepted tokens per target verify;
- target verify batch shape;
- draft latency;
- target verify latency;
- scheduler/metadata overhead;
- memory overhead after real runtime init;
- fallback reasons.

## Stop Lines

Stop and report instead of optimizing if:

- greedy exactness fails;
- rejected tokens can leak into committed state;
- prefix/SWA/component ownership invariants fail;
- average accepted tokens per target verify is not above `1.2` on small realistic
  prompts after obvious correctness fixes;
- the implementation needs graph capture to be correct.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_frozen_kv_verify_runtime/README.md
```

Include:

- SGLang parity table for frozen-KV MTP behavior;
- implementation summary;
- exactness matrix;
- acceptance and latency metrics;
- memory ledger;
- recommendation: proceed to TARGET 11.3 graph/perf closure, keep opt-in, or
  stop.

