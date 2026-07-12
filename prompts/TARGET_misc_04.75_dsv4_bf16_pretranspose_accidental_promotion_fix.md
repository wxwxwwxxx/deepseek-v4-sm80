# TARGET misc 04.75: DSV4 BF16 Pretranspose Accidental Promotion Fix

## Status

Planned short release correction after misc04/misc04.5 and before the final
misc05 release soak.

This target is not a new small-GEMM optimization project. It removes an
experimental path that was previously rejected for promotion but was
accidentally folded into the typed `optimized` runtime during opt-in cleanup.

## Goal

Restore the intended `v0.0.0` release behavior:

```text
optimized:
  keep the promoted cached BF16 dequantized projection weights
  do not build duplicate pretransposed BF16 small-GEMM weights
  use the normal F.linear/cached-BF16 projection path

fallback:
  remain unchanged
```

Recover the corresponding KV-cache capacity without a meaningful correctness
or performance regression, then regenerate the affected graph64 rows in the
public performance document.

## Root Cause And Evidence

Read first:

```text
performance_milestones/target07_bf16_small_gemm_backend_cluster/README.md
performance_milestones/target07_precision_boundary_pivot/README.md
prompts/archive/target07/TARGET_07.71_dsv4_sm80_precision_boundary_pivot.md
performance_milestones/misc_release_two_path_runtime_cleanup/README.md
prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md
PERFORMANCE.md
```

Historical TARGET 07.70 contract:

```text
toggle: MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
status: experimental opt-in, explicitly not in the A100 victory bundle
cache:  1,885,339,648 bytes/rank (1.7559 GiB)
macro:  +0.08% for 4096/128/bs4
macro:  +0.09% for 4096/1024/bs4
profile: aggregate projection/GEMM effectively flat
decision: DO NOT PROMOTE
```

Cleanup commit `3f8c6b6` replaced the old toggle checks with
`dsv4_kernel.dsv4_optimized_enabled()` at the preparation and execution
boundaries. This accidentally made pretranspose mandatory in `optimized`.

Current release-candidate evidence collected on 2026-07-12:

```text
bf16_small_gemm_pretranspose_cache_total: 1,885,339,648 bytes/rank
weights_and_transformed_cache_bytes:
  v0.0.0 graph64:       40,804,286,464
  current candidate:    43,228,594,176
  observed delta:        2,424,307,712 bytes/rank

graph64 KV tokens:
  v0.0.0:               848,384
  current candidate:    771,328
  observed loss:         77,056 tokens
```

The persistent pretranspose tensor bytes are distributed across fused
WQA/WKV, q_wqb, wo_b, indexer wq_b, and shared-expert projections. Marlin raw
expert release still applies its 17.13 GiB credit; CUDA graph physical memory
did not grow. Do not misattribute this regression to Marlin release, SWA, or
CUDA graph reserve.

Pre-fix experiment artifacts are local and intentionally untracked:

```text
/tmp/dsv4_sm80_release_graph64_grid_20260712
/tmp/dsv4_sm80_release_capacity_probes_20260712
```

## Required Implementation

### 1. Remove The Accidental Optimized Route

In `python/minisgl/models/deepseek_v4.py`:

- stop creating BF16 pretransposed copies during
  `prepare_for_cuda_graph_capture()`;
- restore `_linear_cached_bf16_weight()` to the promoted cached-BF16
  `F.linear` behavior rather than `torch.mm(x, weight_t)` for rows `<=16`;
- remove pretranspose-only preparation methods, metadata, report fields, and
  stale cache attributes when they no longer have a runtime owner;
- keep the original cached BF16 dequantized projection weights used by the
  release path;
- keep model-defined FP32 state and FP8/FP4 checkpoint storage unchanged.

Do not solve this by introducing another public argument, environment toggle,
or third runtime path. The final product remains exactly:

```text
optimized | fallback
```

### 2. Keep Fallback And Other Release Features Intact

Do not change:

- Marlin WNA16 expert prebuild/release/capacity credit;
- radix prefix cache and component ownership;
- independent SWA lifecycle;
- graph bucket policy or graph reserve planner;
- PyNCCL threshold policy;
- in-graph metadata preparation;
- BF16 projection dequant caches themselves.

The worktree contains ongoing release-review changes. Work with them; do not
reset, checkout, or revert unrelated user modifications.

### 3. Update Tests And Reports

Update focused tests so that optimized model preparation proves:

```text
no persistent *_pretransposed BF16 projection tensor is created
no pretranspose-only forward route remains reachable
the ordinary cached BF16 projection route remains active
fallback remains independent of optimized caches
```

Prefer deleting obsolete test expectations over retaining a fake zero-valued
feature report. If a compact memory report remains useful, it may explicitly
state that no duplicate pretranspose cache exists, but it must not imply the
old candidate is still selectable.

## Required Validation

### Phase A: Static And Unit Gates

Run:

```bash
python -m compileall -q python/minisgl tests debug/dsv4
python -m pytest -q
ruff check <changed Python files>
git diff --check
```

Audit production code for residual pretranspose-only ownership. Historical
archive text may remain.

### Phase B: TP8 Correctness And Capacity

On `/models/DeepSeek-V4-Flash`, TP8, page size 256:

1. run optimized multilingual/text sanity;
2. confirm finite output and no乱码;
3. confirm CUDA graph replay remains active with zero unexpected eager decode;
4. confirm Marlin raw-weight release and capacity credit still apply;
5. run a bounded fallback text smoke;
6. collect fresh graph64, graph128, and graph256 capacity probes.

The capacity report must show:

- pretranspose persistent bytes removed;
- `weights_and_transformed_cache_bytes` reduced materially;
- graph64 KV capacity recovered materially from 771,328 tokens;
- graph64/128/256 capacity remains monotonic as graph coverage grows;
- no hidden increase in graph physical memory or fixed SWA reservation.

Do not require exact recovery to the old 848,384-token value if CUDA/NCCL or
allocator versions differ. Explain the residual bytes if recovery is less than
the removed persistent cache predicts.

### Phase C: Performance Gate And Public Table

Use independent fresh TP8 processes, release chunk size 8192, output length
1024, simultaneous admission, and the graph64 recipe. Re-run every resident
combination:

```text
active M = 4, 16, 64
prompt   = 1K, 4K, 16K
```

Skip `M=64, prompt=16K` if the planner still proves it cannot fit in one wave.
Do not report a pending/multi-wave result as active M=64.

For each measured row require:

```text
status=pass
completed requests=M
actual output tokens=M*1024
graph replay=1023
unexpected eager decode=0
no scheduler rejection or non-finite output
```

Compare against the pre-fix files in
`/tmp/dsv4_sm80_release_graph64_grid_20260712` when available. Historical
TARGET 07.70 predicts only about 0.1% macro impact. Treat sub-1% movement as
noise; investigate a repeat-stable regression above 3%, but do not reopen
unrelated kernel tuning inside this target.

After acceptance, update `PERFORMANCE.md` with the post-fix graph64 grid and
the post-fix graph64/128/256 memory-capacity table. Keep its language concise
and user-facing. Do not publish pre-fix capacity numbers as the release result.

## Deliverables

```text
performance_milestones/misc_bf16_pretranspose_accidental_promotion_fix/README.md
performance_milestones/misc_bf16_pretranspose_accidental_promotion_fix/capacity.json
performance_milestones/misc_bf16_pretranspose_accidental_promotion_fix/performance.json
performance_milestones/misc_bf16_pretranspose_accidental_promotion_fix/changed_owner_census.md
```

Keep raw large benchmark output under `/tmp` and link or summarize it rather
than copying it into the repository. `performance_milestones/` remains ignored.

## Stop Conditions

Stop and report a blocker rather than broadening scope if:

- removing pretranspose breaks the ordinary cached BF16 weight route;
- optimized text sanity, fallback sanity, graph replay, Marlin release, or SWA
  lifecycle regresses;
- recovered memory is consumed by another newly persistent cache and cannot be
  attributed;
- a repeat-stable performance regression exceeds 3%;
- fixing the issue would require reintroducing research env toggles or a third
  runtime mode.

Do not spend time optimizing sub-1% noise or unrelated kernels. Hand any new
independent bottleneck to misc05 or a future post-release target.

## Completion Criteria

- The accidental pretranspose promotion is removed from optimized.
- No user-facing opt-in replaces it.
- Cached BF16 projection weights remain functional.
- TP8 optimized and fallback correctness pass.
- Graph64 KV capacity is materially recovered.
- Graph64 performance remains within the 3% release gate.
- `PERFORMANCE.md` contains only accepted post-fix release-candidate numbers.
- The final report authorizes proceeding to TARGET misc05.
