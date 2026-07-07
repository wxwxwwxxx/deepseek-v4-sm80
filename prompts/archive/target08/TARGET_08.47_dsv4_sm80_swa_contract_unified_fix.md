# TARGET 08.47: DSV4 SM80 SWA Contract Unified Fix

## Status

Queued TARGET 08 implementation target after TARGET 08.46.

Run this only after TARGET 08.45 has written the SWA independent lifecycle
contract and TARGET 08.46 has audited the current code against that contract.
This target should implement the unified fix plan from the audit, then re-run
the minimum correctness gates needed to return to TARGET 08.43 promotion soak.

## Goal

Make mini-sglang's DeepSeek V4 SWA independent lifecycle implementation comply
with:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
performance_milestones/target08_swa_contract_based_code_audit/recommended_fix_plan.md
```

The expected fix should concentrate in CPU/radix/metadata lifecycle code unless
the audit proves a kernel contract bug.  Do not rewrite the attention kernels
or change precision paths just to work around invalid metadata.

## Required Inputs

Read first:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.45_dsv4_sm80_swa_independent_lifecycle_contract.md
prompts/TARGET_08.46_dsv4_sm80_swa_contract_based_code_audit.md
performance_milestones/target08_swa_contract_based_code_audit/README.md
performance_milestones/target08_swa_contract_based_code_audit/recommended_fix_plan.md
performance_milestones/target08_swa_contract_based_code_audit/required_tests.md
performance_milestones/target08_swa_stale_prefix_handle_tombstone_fix/README.md
```

## Preferred Fix Scope

Prefer fixes in these areas:

- protected-frontier calculation for active SWA release;
- per-request SWA eviction frontier;
- radix SWA component ownership and tombstone rules;
- metadata active-range validation and safe regeneration;
- graph metadata invalidation/versioning for SWA independent mode;
- focused tests and debug gates.

Prefer the finish/cache owner-boundary model from
`recommended_fix_plan.md`: active release records a monotonic
`swa_evicted_seqlen`, while prefix SWA tombstones are committed at
cache insert/finish/eviction ownership boundaries.  A request-local
released-page overlay may be added only as a narrow debug/oracle aid or if the
owner-boundary implementation needs a non-owning mask.  It must not become a
second physical owner and must not mutate radix SWA component values.

Avoid unless the audit proves necessity:

- broad attention kernel rewrites;
- changes to MoE/Marlin release;
- FP8/INT8 precision changes;
- making SWA release globally idempotent;
- hiding invalid metadata by silently clipping all kernel inputs.

## Implementation Rules

### 1. Preserve Contract Guards

- `release_swa_page_handles()` must still catch true duplicate owners.
- Dummy SWA page must remain pinned and unfreeable.
- `-1` remains tombstone/padding, not a valid kernel input inside active
  lengths.
- Route B C4/C128/indexer/state ownership must not regress.
- Marlin WNA16 release + component-slot clear must remain compatible.

### 2. Fix Ownership Before Kernel Workarounds

If the 08.44 illegal memory access is caused by invalid SWA metadata, fix the
producer:

- protected frontier too aggressive;
- active-time radix tombstone unsafe;
- graph replay stale metadata;
- prefix handle ownership ambiguity;
- full-to-SWA mapping cleared before current-step consumers finish.

Only add kernel-side checks as debug or defensive safety gates.

### 3. Keep Tests Close To The Contract

Update or add tests for each contract rule that was violated:

- no-weight ownership/refcount tests;
- active release protected-frontier tests;
- prefix insert/finish/evict tests;
- SWA metadata active-range bounds tests;
- graph replay stale-metadata tests if relevant;
- full-model gates.

Run and pass the no-weight/unit gates before launching full-model TP8 gates.
If a no-weight ownership test fails, stop and fix that contract violation
instead of trying to diagnose it through full-model CUDA illegal memory access.

## Required Test Gates

Run focused tests:

```bash
pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/engine/test_marlin_wna16_release_credit.py
```

Run synchronous attribution/fix gate for the 08.44 illegal memory access:

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
  --output-dir /tmp/dsv4_swa_contract_fix_sync \
  --keep-going
```

Then run the fixed128 full-model gates from 08.44:

```text
variants:
  dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent
  dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent

scenarios:
  historical_4096_128_bs4
  historical_4096_1024_bs4

settings:
  --page-size 256
  --num-pages 128
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
```

If these pass, run one narrow serving/prefix/eviction smoke if runtime allows:

```text
serving_mixed_112req_wave16
prefix_multi_112req_wave16
prefix_eviction_pressure_96req_wave16 or closest available
```

Do not run the full 08.43 fixed/cap4096/auto promotion soak in this target.
That remains the next target after unified correctness is fixed.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_contract_unified_fix/
```

Required files:

- `README.md` with final verdict;
- `contract_fixes.md`;
- `ownership_fix_summary.md`;
- `metadata_graph_fix_summary.md`;
- `focused_tests.md`;
- `full_model_fixed128_gate.md`;
- `mini_soak.md` if run;
- `rerun_0843_recommendation.md`;
- raw logs/JSON under `raw/`.

The README must answer:

1. Which contract violations were fixed?
2. What happened to the 08.44 full-model CUDA illegal memory access?
3. Do fixed128 `4096/128/bs4` and `4096/1024/bs4` pass for SWA independent and
   Marlin release + SWA independent?
4. Did graph replay remain zero-eager for captured buckets?
5. Should TARGET 08.43 promotion soak be rerun next?

## Stop Conditions

Stop and report rather than expanding scope if:

- the audit's chosen ownership model cannot be implemented without broad
  redesign;
- a fix requires weakening double-free guards;
- CUDA illegal memory access remains after metadata/protected-frontier fixes and
  points to a specific kernel contract bug;
- graph replay cannot be made safe without disabling SWA independent lifecycle;
- text corruption appears.
