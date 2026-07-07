# TARGET 11.27: DSV4 SM80 MTP Flattened Verify And Temp-KV Commit

## Status

Next after TARGET 11.25.

TARGET 11.25 proved greedy top-k 1 correctness for frozen-KV MTP with
`draft_len=2` and `draft_len=4`, but it uses conservative sequential target
verification.  That is correct but cannot realize the SGLang performance design.

Do not start CUDA graph work before this target.

## Goal

Replace the sequential target verify path with a SGLang-style flattened
verification path that can reduce target passes while preserving the exact
ownership invariants from TARGET 11.25.

The desired runtime shape is:

1. MTP draft proposes up to `draft_len` tokens per request using frozen target
   KV read-only.
2. The target model verifies the proposed draft sequence in a flattened
   verify/extend-style batch, not one depth at a time.
3. The verifier accepts the longest matching prefix.
4. Accepted target KV/cache/component/SWA state is committed safely.
5. Rejected draft/verify tail state is discarded or never made visible.
6. Greedy token ids still exactly match the non-MTP baseline.

Keep CUDA graph disabled.  This is still an eager correctness/performance
bridge target.

## Starting Evidence From TARGET 11.25

Correctness is good:

- `draft_len=2`: exact token ids for bs=1/2/4.
- `draft_len=4`: exact token ids for bs=1/2/4.
- rejected draft tokens are isolated from `Req`, token pool, KV/cache, prefix,
  SWA, and component state.
- memory overhead remains acceptable, about `+1.0 GiB/rank` in the smoke run.

Performance is not good yet:

- target calls remained `24` in the correctness run;
- target verify was sequential and added about `2.815s` for `draft_len=2` and
  `3.357s` for `draft_len=4`;
- verify shapes stayed depth-serial, effectively `[bs, 1]`, instead of a
  flattened multi-token verify batch;
- average accepted prefix length was low but not a no-go yet because a
  flattened verifier can emit `accepted_prefix + correction` tokens per target
  pass.

Approximate potential from the TARGET 11.25 histogram:

```text
draft_len=2: accepted prefix avg ~= 11 / 21 = 0.52
             emitted per flattened verify ~= 1.52 tokens/pass

draft_len=4: accepted prefix avg ~= 9 / 14 = 0.64
             emitted per flattened verify ~= 1.64 tokens/pass
```

This is not enough to promote MTP, but it is enough to justify a flattened
verify implementation before any graph/perf closure decision.

## SGLang References

Study and compare first:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Important source questions:

- How does SGLang flatten verify tokens for frozen-KV MTP?
- Does it write target verify KV into temporary slots and then move accepted KV?
- Which DSV4 C4/C128/indexer/compression states are written during verify?
- How are rejected tails prevented from becoming visible?
- What metadata is rebuilt per verify and what is cached?

## Implementation Options

Prefer the first option if practical.

### Option A: Temporary Verify KV Plus Accepted-KV Move

Run target verify into temporary or rollback-safe KV/component/SWA locations.
After computing the accepted prefix, move or remap only accepted target states to
committed request locations.

Pros:

- closest to SGLang's performance design;
- avoids target recompute for accepted tokens;
- rejected tail is naturally isolated.

Risks:

- requires careful DSV4 component/SWA/indexer state movement;
- needs exact ownership tests.

### Option B: No-Store Verify Plus Accepted Recompute Oracle

Use no-store flattened target verify to decide acceptance, then recompute only
accepted target tokens into committed locations.

Pros:

- simpler correctness oracle;
- useful for validating flattened verify logits and acceptance logic.

Risks:

- likely too slow for final performance;
- should not be promoted as the final runtime unless it unexpectedly wins.

### Option C: Direct Commit With Rollback

Write verify tokens directly into committed locations and roll back rejected
tails.

This is not recommended as the first implementation because TARGET 08 showed
that DSV4 prefix/SWA/component ownership bugs are expensive to debug.

## Required Work

1. Build a source-parity table against SGLang verify/commit behavior.
2. Implement a flattened verify path for `draft_len=2`.
3. Extend to `draft_len=4` only after `draft_len=2` exactness passes.
4. Add temp/accepted KV movement or a no-store/recompute oracle.
5. Preserve `Batch.frozen_kv_read_only` for draft.
6. Add explicit rejected-tail isolation checks.
7. Add counters:
   - target verify passes;
   - flattened verify tokens;
   - accepted prefix length;
   - emitted tokens per target verify pass;
   - temp KV bytes;
   - accepted KV moved/copied/remapped bytes;
   - recompute tokens if Option B is used.

## Correctness Gates

Minimum:

- greedy token ids exactly match baseline for bs=1/2/4;
- `draft_len=2` exactness passes;
- `draft_len=4` exactness passes or has a clear blocker;
- no乱码 text smoke;
- no finite failures;
- rejected tails do not leak into token pool, KV/cache, prefix, SWA, or
  component state;
- prefix cache disabled path passes;
- prefix cache enabled path is either proven or fail-closed with a clear reason.

## Performance Gates

This target does not need to beat the full baseline yet, but it must prove that
the runtime is moving in the right direction:

- target verify should be flattened, not depth-serial `[bs, 1]` repeated;
- target passes per emitted token should improve versus TARGET 11.25;
- target verify latency should drop versus TARGET 11.25 for the same exactness
  workload;
- report emitted tokens per target verify pass, not only accepted tokens.

If flattened verify still cannot reach about `1.2` emitted tokens per target
verify pass on the smoke prompts, stop and report MTP as low-ROI for the current
prompt mix before doing graph work.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_flattened_verify_temp_kv/README.md
```

Include:

- SGLang source-parity table;
- chosen implementation option and rationale;
- exactness matrix;
- ownership/leak checks;
- verify-shape logs;
- target-pass and latency comparison against TARGET 11.25;
- memory/temp-KV ledger;
- recommendation: proceed to TARGET 11.3 graph/perf closure, iterate verify, or
  stop MTP performance work.

