# TARGET 08.46: DSV4 SM80 SWA Contract-Based Code Audit

## Status

Queued TARGET 08 audit target after TARGET 08.45.

Run this only after `prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md` has
been written.  This target should audit the current implementation against the
contract and produce a unified fix/test plan.  It should not perform broad
repairs unless a tiny assertion or debug hook is needed to classify a risk.

## Goal

Audit every mini-sglang path that produces, stores, mutates, frees, or consumes
DeepSeek V4 SWA independent lifecycle state.

The output should be a concrete risk table and repair plan:

```text
path | current behavior | contract rule | compliant? | risk | proposed fix | tests
```

The goal is to avoid more isolated SWA patches.  The next implementation target
should be able to follow this audit and fix the lifecycle coherently.

## Required Inputs

Read first:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.45_dsv4_sm80_swa_independent_lifecycle_contract.md
performance_milestones/target08_swa_lifecycle_contract/README.md
performance_milestones/target08_swa_lifecycle_contract/mini_contract_risks.md
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/README.md
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/full_model_long_decode_gate.md
```

Audit these mini files:

```text
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/kvcache/radix_cache.py
python/minisgl/scheduler/cache.py
python/minisgl/scheduler/scheduler.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
tests/core/test_deepseek_v4_kvcache.py
tests/attention/test_deepseek_v4_backend_metadata.py
```

Use SGLang only to clarify contract questions:

```text
/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py
/workspace/sglang-main/python/sglang/srt/mem_cache/common.py
/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
```

## Audit Checklist

### 1. Physical SWA Ownership

Check:

- whether `_full_to_swa_page` is owner, translation table, or both;
- whether `_swa_page_refcount` represents active owners, prefix owners, or a
  mixed model;
- whether `DSV4SWAPageHandles` increments/decrements refcount consistently;
- whether `release_swa_page_handles()` remains non-idempotent;
- whether `_clear_full_to_swa_mappings_for_swa_pages()` can invalidate active
  metadata that will still be consumed.

### 2. Radix Prefix SWA Component

Check:

- whether radix node `_dsv4_swa_pages` behaves as owning component value or
  borrowed snapshot;
- whether node split, insert overlap, eviction, lock/unlock, and tombstone
  preserve the contract;
- whether active decode may mutate radix node SWA handles;
- whether TARGET 08.44's `tombstone_dsv4_swa_pages()` should remain,
  be changed to request-local state, or be restricted to finish/evict
  boundaries.

### 3. Active Decode Release Frontier

Check:

- whether `CacheManager.release_active_dsv4_swa_out_of_window()` respects the
  contract's protected frontier;
- whether mini needs a SGLang-like monotonic `swa_evicted_seqlen`;
- whether the current `release_end = cached_len - tail_tokens` can free pages
  still protected by radix/prefix cache;
- whether a one-page margin before insert boundary is needed;
- whether release is ordered after all current-step consumers are done.

### 4. Finish-Time And Unfinished Cache Insert

Check:

- `cache_req(req, finished=True)`;
- `cache_req(req, finished=False)`;
- old handle unlock / new handle lock ordering;
- already-cached indices free;
- tail free;
- prefix SWA tombstone after insertion/merge;
- component-owned full-head release.

Verify there is no path where a SWA page is both physically freed and still
available as an active metadata source.

### 5. Attention Metadata And Kernel Inputs

Check:

- `_make_swa_page_tables()`;
- `_build_swa_page_table_row()`;
- `_make_swa_indices_from_page_table()`;
- `_debug_check_swa_index_bounds()`;
- graph metadata copy/replay paths;
- direct SWA metadata disablement in independent mode;
- `store_swa_fallback()` full-to-SWA translation.

For every active length covered by `swa_topk_lengths`, verify that kernel input
indices are:

- non-negative;
- in range;
- not dummy unless deliberately ignored;
- not present in the free SWA page list;
- not stale after graph replay.

### 6. CUDA Graph Contract

Check:

- whether SWA metadata sources are regenerated or safely copied before replay;
- whether graph input buffers can retain old SWA indices after ownership
  version changes;
- whether debug/replay counters can detect stale SWA metadata;
- whether graph capture itself produces invalid SWA metadata for dummy rows or
  padded rows.

### 7. Tests And Debug Instrumentation

Propose tests before implementation:

- no-weight/refcount tests;
- active release protected-frontier tests;
- radix insert/evict/tombstone tests;
- metadata active-range `-1` / free-page membership tests;
- graph-replay stale metadata tests;
- synchronous full-model attribution command for the 08.44 illegal memory
  access:

```bash
CUDA_LAUNCH_BLOCKING=1 \
MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --repeats 1 \
  --warmup-repeats 0 \
  --output-dir /tmp/dsv4_swa_contract_audit_sync \
  --keep-going
```

Do not spend hours running full TP8 matrices in this audit target.  Use them
only when needed to classify the 08.44 illegal memory access owner.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_contract_based_code_audit/
```

Required files:

- `README.md` with final audit verdict;
- `contract_compliance_table.md`;
- `ownership_risk_table.md`;
- `metadata_graph_risk_table.md`;
- `sglang_parity_delta.md`;
- `recommended_fix_plan.md`;
- `required_tests.md`;
- raw/debug logs under `raw/` if any commands are run.

The README must answer:

1. Which current paths violate or ambiguously satisfy the contract?
2. Is 08.44's active-time radix handle tombstone contract-compliant?
3. What is the most likely owner of the full-model CUDA illegal memory access?
4. What exact implementation changes should TARGET 08.47 perform?
5. Which tests must pass before rerunning TARGET 08.43 promotion soak?

## Stop Conditions

Stop and write the audit rather than drifting into implementation if:

- more than one ownership model is plausible and needs user/project decision;
- a proposed fix would require broad kernel rewrites;
- the audit finds contract violations in multiple layers that should be fixed
  together;
- the only remaining issue is a full-model CUDA illegal memory access that
  needs synchronous attribution before patching.
