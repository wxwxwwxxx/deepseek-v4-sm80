# TARGET misc: DSV4 SM80 Post-MTP-Cleanup Replay Attribution

## Status

New narrow attribution target after the non-MTP post-cleanup soak.

The post-MTP cleanup baseline is functionally healthy, but it regressed versus
the TARGET 10.27 PyNCCL32M/prefix baseline.  Initial investigation found one
real non-MTP regression in the decode graph replay metadata path: a per-replay
GPU `.item()` synchronization inside compressed-read metadata clamping.  That
fix recovered part of the gap, but not all of it.

This target exists to explain the remaining gap before making more broad
changes.

## Goal

Attribute the remaining post-cleanup decode performance regression against the
TARGET 10.27 baseline, with special focus on CUDA graph replay metadata
overhead:

```text
GraphRunner._replay_to_buffer
  -> attn_backend.prepare_for_replay
  -> _copy_metadata_for_replay
  -> compressed/C4/SWA/component metadata staging
```

The target should answer:

```text
Did post-TARGET10 non-MTP code add hot-path CPU, D2D, memset, sync, or small
kernel work during captured decode graph replay?
If yes, which owner/function accounts for the remaining regression?
Can the fix be made without weakening prefix/SWA/component ownership
correctness?
If no, what evidence shows the regression comes from another subsystem?
```

## Non-Goals

- Do not revive or modify MTP code.
- Do not change communication backend routing unless evidence points there.
- Do not change low-precision, Marlin, MoE, or attention kernels unless the
  attribution proves they are the regression owner.
- Do not perform a broad benchmark sweep before the short attribution probes.
- Do not weaken the SWA independent lifecycle contract or prefix correctness
  invariants.
- Do not chase sub-1% improvements after the main regression owner is
  explained.

## Starting Evidence

Read first:

```text
performance_milestones/post_mtp_cleanup_baseline/README.md
performance_milestones/target10_pynccl_default_promotion_blockers/README.md
prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Important baseline facts from the post-cleanup report:

```text
MTP env/flags: absent
text sanity: pass
prefix savings: 49152 tokens
CUDA graph decode fallback: 0 eager
communication count/bytes: matches promoted TARGET 10 path
kernel wrapper counters: matches the comparable TARGET 10 path
```

The original post-cleanup `historical_4096_128_bs4` result:

```text
output tok/s:              49.418
decode tok/s:              153.071
decode_forward_enqueue_s:  0.549982
decode_forward_s:          3.313768
decode_prepare_s:          0.296594
graph replay/eager:        127 / 0
```

TARGET 10.27 comparable short sanity:

```text
output tok/s:              53.306
decode tok/s:              190.405
decode_forward_enqueue_s:  0.143486
decode_forward_s:          2.661946
decode_prepare_s:          0.277315
graph replay/eager:        127 / 0
```

Initial local fix:

- File: `python/minisgl/attention/deepseek_v4.py`
- Function: `_clamp_graph_replay_compressed_read_metadata`
- Problem: every graph replay could run a GPU `torch.all(...).item()` no-op
  check, forcing a hidden sync.
- Fix direction: preserve clamp correctness, but decide no-op cases from CPU
  request metadata before touching GPU tensors.

After that fix, the same short TP8 probe improved to:

```text
output tok/s:              50.766
decode tok/s:              164.261
decode_forward_enqueue_s:  0.363022
decode_forward_s:          3.091720
decode_prepare_s:          0.284935
graph replay/eager:        127 / 0
```

Interpretation: the hidden sync was real, but there is still unexplained
hot-path overhead versus TARGET 10.27.

## Required Work

### 1. Confirm The Starting Point

Verify the current tree is the intended non-MTP cleanup tree:

```text
git status --short
git log --oneline -8
rg -n "mtp|nextn|speculative" python benchmark tests || true
```

Do not remove the allowed checkpoint-skip handling for weight names matching
`mtp.*`; that is the accepted non-MTP behavior.
It is also expected that archived TARGET 11 prompt/report documents still
mention MTP; this target only cares about active runtime, benchmark, and test
code.

Verify whether the compressed-read clamp `.item()` fix is already present.  If
it is missing, reintroduce the CPU no-op fast path first and rerun the focused
unit tests.

Required correctness gate for replay metadata edits:

```bash
python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
```

### 2. Establish A Short, Repeatable Probe

Use the smallest TP8 full-model probe that reproduces the regression:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  debug/dsv4/benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --output-dir /tmp/dsv4_misc_replay_attribution_short \
  --keep-going
```

Record:

- output tok/s and decode tok/s;
- `decode_forward_enqueue_s`;
- `decode_forward_s`;
- `decode_prepare_s`;
- graph replay/eager counts;
- communication owner count/bytes;
- wrapper/kernel counters.

Repeat once if variance is high.  Do not run the four-scenario soak until a
candidate fix exists.

### 3. Compare Against TARGET 10.27 At The Right Level

Use reports under:

```text
performance_milestones/target10_pynccl_default_promotion_blockers/raw/
performance_milestones/post_mtp_cleanup_baseline/raw/
```

Compare the current short probe with TARGET 10.27 on:

```text
phase timings
communication owner count/bytes
wrapper counters
graph replay/eager count
captured bucket set
model_prepare_report feature flags
```

If count/bytes/wrapper counters still match, treat the regression as:

```text
CPU enqueue overhead, hidden sync, D2D copy, memset, or small replay helper
overhead
```

instead of a major algorithmic/kernel/communication-path rollback.

### 4. Attribute Replay Metadata Owners

Instrument only the replay metadata hot path.  Prefer temporary, local
diagnostic owner timing over broad permanent logging.

Primary surfaces:

```text
python/minisgl/engine/graph.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/scheduler/cache.py
python/minisgl/utils/dsv4_owner_timing.py
```

Suggested owner labels:

```text
graph.replay_to_buffer.total
graph.replay_to_buffer.copy_inputs
attn.prepare_for_replay.total
attn.copy_metadata_for_replay.total
attn.copy_metadata_for_replay.c4
attn.copy_metadata_for_replay.compressed
attn.copy_metadata_for_replay.swa
attn.compressed_read_clamp
attn.component_page_table_cache
attn.direct_graph_metadata_buffers
```

Keep instrumentation lightweight:

- no unconditional `torch.cuda.synchronize()` in hot path;
- no per-step `.item()` on CUDA tensors;
- no per-token logging;
- gate temporary diagnostics behind an env var;
- remove or default-disable diagnostics before final validation.

If owner timing itself adds too much overhead, use differential one-owner-at-a
time probes or CUDA event timing outside captured graph replay.

### 5. Check For Known Post-TARGET10 Hot-Path Additions

Audit diffs after the TARGET 10.27 baseline, especially Target08 lifecycle and
Marlin-release work.  Look for code that now executes every replay even when
the feature is disabled.

High-priority checks:

- compressed-read clamp no-op path;
- SWA direct replay metadata branches;
- component page-table/cache version checks;
- direct graph metadata buffer copies;
- disabled owner-timing wrappers that still perform repeated env lookups;
- debug/sentinel/guard code around released Marlin weights;
- extra Python object/list construction in `prepare_for_replay`;
- any CUDA tensor `.item()`, `.cpu()`, `.tolist()`, or shape-dependent
  allocation in decode replay.

Allowed outcome: prove that a candidate is not active in the tested variant.
For example, if `swa_independent_lifecycle=false`, an SWA direct D2D copy may be
dead for this baseline and should not be blamed.

### 6. Fix Only Proven Hot-Path Regressions

Acceptable fix patterns:

- compute no-op decisions on CPU metadata before GPU tensor work;
- cache env flags outside per-step hot paths;
- skip disabled feature branches before building tensors/lists;
- reuse preallocated buffers instead of allocating per replay;
- split correctness clamps into cheap no-op fast paths and rare slow paths;
- move debug checks behind explicit env gates.

Every fix must preserve:

```text
prefix cache correctness
component loc ownership correctness
SWA independent lifecycle contract
zero eager decode fallback for captured scenarios
```

Run the focused metadata unit gate after each replay metadata change.

### 7. Validation

Minimum final validation:

```bash
python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
```

Then rerun:

```text
historical_4096_128_bs4
```

with the canonical TP8 command.  If the regression owner is fixed and the short
probe is close to TARGET 10.27, run the four-scenario non-MTP soak:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

Text sanity is required only if the fix touches metadata semantics, cache
ownership, or replay buffer contents.

## Stop Conditions

Stop this target once one of these is true:

1. The remaining post-cleanup regression is attributed to one or more concrete
   owners, and a correctness-preserving fix recovers most of the short-probe
   gap.
2. The current tree matches TARGET 10.27 within normal variance on the short
   probe and the original regression is explained as benchmark variance or
   stale comparison.
3. The regression is proven to be outside graph replay metadata; write the next
   target with the new owner and stop.
4. After two focused fix attempts, the remaining unexplained gap is below about
   `2%` output tok/s on the short probe.  Record it as residual and stop.

Do not continue optimizing random small owners after the main gap is explained.

## Deliverables

Write results to:

```text
performance_milestones/misc_post_mtp_cleanup_replay_attribution/
```

Required files:

```text
README.md
raw/
```

The README must include:

- exact git commit and dirty-state summary;
- whether the compressed-read clamp sync fix was present;
- short-probe before/after metrics;
- TARGET 10.27 comparison table;
- owner attribution table for replay metadata hot path;
- any code changes and their correctness gates;
- final go/no-go recommendation for another full non-MTP soak.
