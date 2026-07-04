# TARGET 08.195: DSV4 Exact-Path Slot/Page Invariance

## Status

Active next TARGET 08 subtarget.

Run this after TARGET 08.19 and before TARGET 08.20.

TARGET 08.19 found that phase-1 prefix-cache metadata is clean, but full
vocabulary logits diverge before sampling.  The most important observation is
that the prefix-disabled control is not a clean oracle: identical prompts in
different batch slots produced different logits.  Therefore the blocker is a
DSV4 exact-path slot/page-location correctness issue, not a SWA/component
retention issue.

Do not start TARGET 08.20 until this target either fixes the issue or produces a
stable slot-pinned/page-normalized oracle that TARGET 08.20 can use.

## Goal

Make the promoted DSV4 exact path invariant to batch slot, request-table row,
physical page location, and CUDA graph padding for the tested boundaries.

The target should answer:

1. Why do identical prefix-disabled prompts in different batch slots produce
   different logits?
2. Why does the no-hit SWA boundary batch of size 3 differ between graph bucket
   4 replay and eager decode?
3. Which layer or submodule first diverges:
   - embedding;
   - pre/post RMSNorm or HC cleanup;
   - Q/KV projection, fused norm/RoPE/store, or compressor;
   - SWA/C4/C128 attention;
   - C4 indexer and indexer FP8/BF16 cache path;
   - MoE/shared experts;
   - lm_head or sampling.
4. Is the issue caused by logical metadata, physical cache addresses, graph
   padded rows, cached BF16/FP8 projection/indexer state, or a custom kernel
   implementation bug?
5. What minimal fix or guard is required before TARGET 08.20?

## Starting Point

Read:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.19_dsv4_sm80_prefix_cache_logit_metadata_correctness.md`
- `performance_milestones/target08_prefix_cache_logit_metadata_correctness/README.md`
- `performance_milestones/target08_prefix_cache_logit_metadata_correctness/summaries/logits_comparison.md`
- `performance_milestones/target08_prefix_cache_logit_metadata_correctness/summaries/metadata_comparison.md`
- `performance_milestones/target08_prefix_cache_logit_metadata_correctness/scripts/run_dsv4_prefix_logit_probe.py`
- `python/minisgl/utils/dsv4_prefix_debug.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/engine/graph.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`

Use the promoted exact path unless a specific A/B disables one suspected toggle:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
cuda_graph_bs=[1,2,4,8,16]
```

## Required Reproductions

Build a narrow reproduction suite that does not require prefix cache.

At minimum cover:

- identical prompts in a batch, prefix cache disabled, eager prefill/decode;
- identical prompts in a batch, prefix cache disabled, graph decode;
- one prompt run alone versus the same prompt in slot 0/1/2/3;
- same prompt with different request-table rows;
- same prompt with different physical page locations, for example by allocating
  and freeing dummy requests before the measured request;
- no-hit SWA boundary prompts with lengths `127`, `128`, and `129`;
- batch size 3 eager versus graph bucket 4 replay;
- page boundary around `255/256/257/258`;
- C4 and C128 boundaries around multiples of `4` and `128`.

The first goal is to reproduce the TARGET 08.19 failures without involving
prefix hits.  After a fix or strong hypothesis, rerun the TARGET 08.19 probe.

## Required Instrumentation

Add only opt-in debug hooks.

Capture enough data to identify the first divergent boundary:

- batch/request metadata:
  - batch row;
  - request `uid`;
  - request table index;
  - `out_loc`;
  - page table rows;
  - physical full/SWA/C4/C128/indexer locations;
  - positions and sequence lengths.
- per-layer or checkpoint activations:
  - embedding output;
  - input layernorm output;
  - attention input and output;
  - compressor / indexer output when relevant;
  - MoE input and output;
  - final norm;
  - lm_head logits.
- graph replay metadata:
  - real batch size;
  - padded batch size;
  - graph bucket used;
  - padded row contents copied into graph static inputs.

Do not save huge tensors by default.  Use row slices, hashes, norms, max-abs
diffs, top-k, and an opt-in full-tensor dump for the first failing scenario.

## Toggle Bisection

Run a focused bisection over promoted TARGET 07 toggles.  Prefer disabling one
suspect group at a time via `MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES`.

Prioritize:

- `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST`;
- `MINISGL_DSV4_SM80_REPLAY_METADATA_COPY`;
- `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE`;
- `MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16`;
- `MINISGL_DSV4_SM80_FUSED_TOPK_SWA_INDICES`;
- `MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE`;
- `MINISGL_DSV4_SM80_COMPRESS_STORE`;
- `MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT`;
- `MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE`;
- BF16 projection caches:
  - `q_wqb`;
  - `wo_a`;
  - `wo_b`;
  - `indexer_wqb`;
  - `shared_expert`;
- MoE backend:
  - `MINISGL_DSV4_SM80_MOE_VLLM_RUNNER`;
  - `MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND`.

The bisection is not complete until it maps the issue to a minimal toggle group
or proves the issue exists even on a conservative exact fallback.

## Analysis Rules

Use logits and activation diffs, not sampled tokens, as primary evidence.

Report the earliest divergent checkpoint in this order:

```text
metadata -> embedding -> layer N input -> layer N attention -> layer N MoE ->
final norm -> lm_head logits -> sampled token
```

When comparing graph and eager, separate:

- true graph replay differences for real rows;
- padded-row contamination;
- metadata copy differences;
- sampler-only differences.

When comparing prefix-on and prefix-off, do not call prefix-on wrong until the
prefix-disabled oracle is slot/page invariant for the same logical request.

## Required Fix Or Guard

If a small fix is found, implement it and rerun:

- the new slot/page invariance suite;
- TARGET 08.19 probe;
- a short prefix-cache serving smoke from TARGET 08.10.

If a full fix is too large, add a conservative guard or define a stable oracle:

- disable the offending graph bucket/path under prefix-cache correctness probes;
- force a safe fallback for affected sizes;
- or build a slot-pinned/page-normalized comparison mode for TARGET 08.20.

Do not hide the issue by widening tolerances.

## Deliverables

Create:

```text
performance_milestones/target08_exact_path_slot_page_invariance/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- reproduction table;
- toggle bisection table;
- first-divergent-layer/module table;
- graph bucket 3-to-4 analysis;
- fix/guard description if implemented;
- rerun result for TARGET 08.19, or a clear reason it could not be rerun;
- decision for TARGET 08.20.

## Decision Rules

Proceed to TARGET 08.20 only if one of these is true:

- identical prompts are logits-invariant across tested slots/page locations;
- graph bucket 4 replay is logits-equivalent to eager for real rows in the
  tested bs3 no-hit scenario;
- TARGET 08.19 passes after the fix;
- or a documented slot-pinned/page-normalized oracle exists and is sufficient
  for TARGET 08.20 component-retention work.

Keep prefix cache opt-in only if any exact-path invariance issue remains.

## Stop Rules

Stop and report blocked if:

- the first divergence requires a broad rewrite of attention/cache/model runtime;
- the issue cannot be reproduced outside TARGET 08.19 artifacts;
- instrumentation perturbs the batch layout enough to hide the bug;
- toggle bisection points to multiple unrelated correctness bugs.

## Non-Goals

- Implementing TARGET 08.20 or TARGET 08.21 component retention.
- Promoting prefix cache to default.
- Low-precision research.
- Performance tuning beyond what is needed to fix correctness.
- General CUDA graph memory attribution.
