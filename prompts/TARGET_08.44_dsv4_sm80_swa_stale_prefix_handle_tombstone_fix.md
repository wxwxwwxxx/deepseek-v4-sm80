# TARGET 08.44: DSV4 SM80 SWA Stale Prefix-Handle Tombstone Fix

## Status

Active TARGET 08 correctness follow-up after TARGET 08.43.

TARGET 08.43 verified that the TARGET 08.42 dummy full-token fix remains
healthy, but blocked SWA independent lifecycle promotion on a new long-decode
failure:

```text
historical_4096_1024_bs4
+ dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent
-> RuntimeError: DSV4 KV cache double free detected in SWA page slots
```

The same failure appears with Marlin WNA16 release + SWA independent, but the
pure SWA independent variant fails first.  Therefore this target is not about
Marlin release, not about the old 08.42 CUDA illegal memory access, and not
about performance overhead.  It is a SWA lifecycle ownership fix.

## Goal

Make active decode out-of-window SWA release and prefix-cache SWA tombstone
mutually consistent:

```text
active decode releases old SWA pages
-> prefix/cache handles that can later be tombstoned no longer contain those
   released page ids
-> finish-time cache_req(..., finished=True) cannot decrement the same SWA
   page refcount twice
-> the real double-free guard remains active for true duplicate ownership bugs
```

After this target, fixed128 long decode must pass for pure SWA independent and
Marlin release + SWA independent.  Then TARGET 08.43 promotion soak can be
rerun.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_independent_post_fix_promotion_soak/README.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/correctness_graph_soak.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/swa_tail_runtime_counters.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/promotion_decision.md
performance_milestones/target08_swa_independent_post_fix_promotion_soak/raw/no_weight_stale_prefix_swa_double_free.log
prompts/TARGET_08.43_dsv4_sm80_swa_independent_post_fix_promotion_soak.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Important TARGET 08.43 evidence:

- Text smoke passed for fixed128, cap4096, and auto Marlin release + SWA
  independent; the 08.42 dummy-token bug did not recur.
- Fixed128 short historical `4096/128/bs4` passed for SWA independent and
  Marlin release + SWA independent.
- Fixed128 long historical `4096/1024/bs4` failed on all ranks with
  `DSV4 KV cache double free detected in SWA page slots`.
- Graph replay before failure was healthy: replay `1277`, eager `0`.
- The no-weight repro shows the lifecycle shape:
  1. prefix tombstone keeps a live SWA tail page in a prefix handle;
  2. active decode out-of-window release frees that same SWA page;
  3. finish-time prefix tombstone revisits the stale handle and releases the
     same SWA page again.

Relevant source areas:

```text
python/minisgl/scheduler/scheduler.py
python/minisgl/scheduler/cache.py
python/minisgl/kvcache/radix_cache.py
python/minisgl/kvcache/deepseek_v4_pool.py
tests/core/test_deepseek_v4_kvcache.py
```

## Non-Goals

- Do not make SWA page release globally idempotent in a way that hides real
  duplicate ownership bugs.
- Do not remove or weaken the `DSV4 KV cache double free detected in SWA page
  slots` guard except for a narrowly proven already-tombstoned handle path.
- Do not change Marlin WNA16 release semantics.
- Do not implement FP8 KV/cache or INT8 MoE.
- Do not optimize decode prepare / attention metadata overhead in this target.
- Do not rerun the full 08.43 serving/prefix/auto-capacity soak until the
  fixed128 long-decode correctness gate passes.

## Required Work

### 1. Reproduce And Preserve The Small Failure

Turn the no-weight stale-prefix SWA handle repro from TARGET 08.43 into a
focused regression test.

The test should simulate, without loading model weights:

1. create a small DSV4 KV/prefix setup with SWA independent lifecycle enabled;
2. create or emulate a prefix SWA handle containing several SWA pages;
3. prefix-tombstone the out-of-window head while retaining a live tail;
4. active-release the same tail page after it falls outside the decode window;
5. run finish-time tombstone on the stale handle.

Before the fix, this should reproduce:

```text
RuntimeError: DSV4 KV cache double free detected in SWA page slots
```

After the fix, it should pass and assert:

- released SWA pages are not present in handles that can be tombstoned later;
- SWA refcounts stay non-negative;
- freed page counts advance exactly once per live page;
- dummy SWA page is never released;
- a deliberately invalid duplicate owner still trips the double-free guard.

Prefer placing the regression under `tests/core/test_deepseek_v4_kvcache.py` or
another focused core/cache test file.

### 2. Fix Ownership Synchronization

Investigate and implement the smallest durable fix.

Preferred direction:

- When active decode releases out-of-window SWA pages, also tombstone or
  invalidate matching SWA pages in the active request's cache/prefix handle
  state so later finish-time prefix tombstone cannot revisit them.

Acceptable alternatives:

- Make finish-time prefix SWA tombstone skip pages that are provably already
  released by the active out-of-window path, while still raising on true
  duplicate ownership bugs.
- Introduce an explicit released-page/tombstone mask attached to
  `DSV4SWAPageHandles` or the active request handle if that is cleaner than
  mutating existing handle page tensors.

Be careful with these invariants:

- `DSV4SWAPageHandles` is currently frozen and returns updated copies from
  `tombstone_tokens`; do not mutate shared objects accidentally.
- Prefix-cache node handles are snapshots of SWA page ids.  If active release
  frees a page represented by a handle snapshot, that snapshot must be updated
  before it can be reused for tombstone.
- Prefix cache eviction and finish-time tombstone must continue to release live
  pages exactly once.
- Dummy page and negative tombstones must remain ignored by release.
- Component loc ownership for C4/C128/indexer/state must not be affected.

### 3. Source And Unit Gates

Run focused tests before any full-model macro:

```bash
pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/engine/test_marlin_wna16_release_credit.py
```

Also add any smaller targeted test needed for the exact fix owner.

### 4. Full-Model Correctness Gate

After unit/no-weight tests pass, run the minimal TP8 full-model gates that
failed in 08.43.

Required settings:

```text
--model-path /models/DeepSeek-V4-Flash
--page-size 256
--num-pages 128
--allow-dsv4-cuda-graph
--cuda-graph-bs 1 2 4 8 16
```

Required variants:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent
```

Required scenarios:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
```

Pass criteria:

- no SWA page double-free;
- no CUDA illegal memory access;
- no garbled text in any accompanying smoke output;
- graph replay remains zero-eager for captured buckets;
- SWA counters remain internally sane:
  - non-negative refcounts;
  - dummy SWA page retained;
  - freed/tombstoned counts do not double count stale handles;
  - available/current page counts stay consistent.

### 5. Mini Soak Before Returning To 08.43

If the failing long historical gates pass, run one narrow serving/prefix smoke
to ensure the fix did not break the common prefix path:

- fixed128 `serving_mixed_112req_wave16` for SWA independent if runtime allows;
- fixed128 `prefix_multi_112req_wave16` for SWA independent if runtime allows;
- fixed128 `prefix_eviction_pressure_96req_wave16` or closest available
  eviction pressure workload if runtime allows.

This is not the full promotion soak.  The full fixed/cap4096/auto capacity
matrix remains TARGET 08.43 rerun after this target.

### 6. Rerun Recommendation

End with a precise recommendation:

- If all gates pass, rerun
  `prompts/TARGET_08.43_dsv4_sm80_swa_independent_post_fix_promotion_soak.md`.
- If a new lifecycle owner appears, write the next shortest correctness target.
- If correctness passes but overhead remains visible, do not optimize it here;
  let the rerun 08.43 attribution decide whether a metadata-overhead target is
  needed.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/
```

Required files:

- `README.md` with final verdict;
- `root_cause.md`;
- `no_weight_repro_and_test.md`;
- `fix_summary.md`;
- `focused_tests.md`;
- `full_model_long_decode_gate.md`;
- `mini_soak.md` if run;
- raw logs/JSON under `raw/`.

The README must answer:

1. Was the stale prefix-handle double-free reproduced without model weights?
2. What exact object or handle was stale?
3. What code now synchronizes active release with prefix-handle tombstone?
4. Does the real double-free guard still catch invalid duplicate ownership?
5. Do fixed128 `4096/1024/bs4` SWA independent and Marlin release + SWA
   independent pass?
6. Should TARGET 08.43 be rerun next?

## Stop Conditions

Stop and report rather than broadening scope if:

- the no-weight repro cannot be made deterministic;
- the fix requires weakening the double-free guard without a narrow proof;
- fixed128 `4096/1024/bs4` still double-frees after the first fix attempt and
  evidence points to a different owner;
- CUDA illegal memory access or text corruption appears;
- the fix touches attention kernels, Marlin release, or low-precision paths
  without direct evidence.
