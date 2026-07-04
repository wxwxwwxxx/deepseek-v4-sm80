# TARGET 08.196: DSV4 Batched Attention/Indexer Row-Coupling Isolation

## Status

Complete, narrowed by TARGET 08.197.

Run this after TARGET 08.195 and before TARGET 08.20.

TARGET 08.195 found and fixed a real DSV4 compressor bug: flattened batches
were pooled every `ratio` rows, so C4/C128 compressed windows could cross
request boundaries when request lengths were not multiples of the compression
ratio.  That fix materially reduced identical-prompt drift, but did not make
the promoted exact path fully invariant.  Remaining drift appears around
attention/indexer outputs when multiple requests share a batch, while
single-request table/page churn probes pass exactly.

This target added the needed attention/indexer debug direction and exact-bs
graph guard, but did not clear broad component-retention correctness.  TARGET
08.197 later classified the layer0 q-path issue as GEMM shape numeric drift, so
the active follow-up is TARGET 08.198 post-layer0 same-shape decode drift
isolation.

## Goal

Isolate and fix the remaining DSV4 batched attention/indexer row-coupling.

The target should answer:

1. Why does a target prompt run alone differ from the same target prompt embedded
   in a multi-request batch, even with prefix cache disabled?
2. Is the first remaining divergence truly in attention output, indexer output,
   or a lower-level subcomponent that the current checkpoints do not expose?
3. Which path owns the row coupling:
   - Q/WKV projection and shared activation fusion;
   - Q/KV norm, RoPE, and fused norm/RoPE/store;
   - SWA attention;
   - C4 indexer logits/top-k/select;
   - C4 sparse attention;
   - C128 attention;
   - compressed cache store/load;
   - graph replay metadata copy or padded rows.
4. What minimal fix, fallback, or guard makes the correctness oracle usable for
   TARGET 08.20?

## Starting Point

Read:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.195_dsv4_sm80_exact_path_slot_page_invariance.md`
- `performance_milestones/target08_exact_path_slot_page_invariance/README.md`
- `performance_milestones/target08_exact_path_slot_page_invariance/summaries/logits_comparison.md`
- `performance_milestones/target08_exact_path_slot_page_invariance/summaries/first_divergent_layer.md`
- `performance_milestones/target08_exact_path_slot_page_invariance/summaries/graph_bucket_analysis.md`
- `performance_milestones/target08_exact_path_slot_page_invariance/summaries/eager_identical_after_fix/first_divergent_layer.md`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/utils/dsv4_prefix_debug.py`
- `python/minisgl/engine/graph.py`

Use the promoted exact path unless a specific A/B disables one suspected path:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
cuda_graph_bs=[1,2,4,8,16]
prefix cache disabled for the main invariance oracle
```

## Required First Step: Preserve The Compressor Fix

Before chasing the remaining drift, confirm that the TARGET 08.195 compressor
boundary fix is present and covered.

Add or keep focused coverage for:

- C4 compression with request lengths not divisible by `4`;
- C128 compression with request lengths not divisible by `128`;
- multiple requests in a flattened batch;
- vectorized and fallback compressor paths using the same per-request
  position-contiguous rule;
- CUDA graph capture behavior around compressed updates.

The report must state whether the compressor fix is retained, revised, or
rejected.  Do not regress to flattened `x[:usable_tokens]` pooling.

## Required Reproductions

Build a narrow reproduction suite that separates batch effects from allocator
effects.

At minimum cover:

- target prompt alone;
- target prompt in slot `0/1/2/3` with unrelated filler prompts;
- four identical prompts in one batch;
- target prompt in batch with filler prompts of:
  - same length;
  - lengths crossing C4 boundaries;
  - lengths crossing C128 boundaries;
  - lengths around SWA `127/128/129`;
- single target with request-table churn;
- single target with physical-page churn;
- bs=3 no-hit SWA boundary eager versus graph bucket 4;
- page boundary `255/256/257/258`;
- C4/C128 boundary lengths.

The target should keep the successful 08.195 single-request page/table oracle,
but it must not treat general multi-request batches as a clean oracle until this
row-coupling issue is fixed.

## Required Instrumentation

Use opt-in debug hooks only.

The current 08.195 checkpoints are useful but not fine-grained enough.  Add
more precise row-level checkpoints inside DSV4 attention/indexer for the first
few divergent layers:

- attention input;
- WQA/WKV outputs, including fused shared-activation output if enabled;
- Q after Q norm;
- KV after KV norm;
- Q/K after RoPE or fused norm/RoPE/store;
- compressor input/output;
- indexer query/key/logits/top-k indices/top-k scores;
- C4 sparse selected page/full indices;
- SWA attention output;
- C4 sparse attention output;
- C128 attention output;
- merged attention output before WO_A/WO_B;
- final attention output.

For each checkpoint, compare:

- target row run alone;
- same target row embedded in a batch;
- identical rows within one batch.

Use row hashes, norms, max-abs diffs, top-k summaries, and optional full tensor
dumps for the first failing checkpoint.  Keep full dumps opt-in and prune large
raw files after summaries are written.

## Toggle And Backend Bisection

Continue the TARGET 08.195 bisection, but make it more surgical around
attention/indexer.

Prioritize:

- `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE`;
- `MINISGL_DSV4_SM80_INDEXER_BF16`;
- `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16`;
- `MINISGL_DSV4_SM80_PAGED_MQA_BF16`;
- `MINISGL_DSV4_SM80_FUSED_TOPK_SWA_INDICES`;
- `MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE`;
- `MINISGL_DSV4_SM80_FUSED_Q_KV_RMSNORM`;
- `MINISGL_DSV4_SM80_COMPRESS`;
- `MINISGL_DSV4_SM80_COMPRESS_STORE`;
- `MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT`;
- `MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE`;
- BF16 projection caches:
  - `q_wqb`;
  - `wo_a`;
  - `wo_b`;
  - `indexer_wqb`.

If disabling a broad toggle reduces but does not remove drift, split that path
with lower-level instrumentation instead of declaring the whole toggle guilty.

## Graph Bucket 3-To-4 Guard

TARGET 08.195 showed:

```text
bs=3 no-hit SWA boundary eager vs graph bucket 4 decode: FAIL max ~= 2.19
```

This target must decide one of:

- fix graph replay metadata/static-input copy for real bs=3 padded to bucket 4;
- capture and use a real bucket 3 if that is acceptable for serving policy;
- force eager fallback for this unsafe shape class;
- or define a clear correctness-probe-only guard that excludes this comparison.

Do not let this graph issue pollute the attention/indexer row-coupling analysis.

## Analysis Rules

Use logits and activation diffs as primary evidence.  Sampled tokens are
secondary.

Report earliest divergence in this order:

```text
metadata -> attention input -> WQA/WKV -> Q/K norm -> RoPE/store ->
compressor/indexer -> SWA/C4/C128 attention -> attention merge ->
attention output -> MoE -> final norm -> lm_head logits -> sampled token
```

For row-coupling, distinguish:

- identical rows inside one batch;
- target-alone versus target-in-batch;
- target-in-batch with different filler lengths;
- allocator-induced page/table churn;
- graph padded-row contamination.

If a remaining difference is only due to legal floating-point non-associativity,
the report must justify that with magnitude, top-k stability, and a comparison
to a reference/fallback path.  Do not widen tolerances as a substitute for
root-cause analysis.

## Required Fix Or Guard

If a small fix is found, implement it and rerun:

- TARGET 08.196 row-coupling suite;
- TARGET 08.195 invariance suite;
- TARGET 08.19 prefix logit/metadata probe;
- a short TARGET 08.10 prefix-cache serving smoke.

If the fix is too large, add a conservative guard or define a stable oracle:

- disable the unsafe attention/indexer backend for correctness-sensitive prefix
  probes;
- force safe fallback for affected shape classes;
- forbid graph bucket 4 replay for real bs=3 SWA-boundary correctness probes;
- or define a slot-pinned/page-normalized single-target oracle that TARGET 08.20
  may use, with explicit limitations.

## Deliverables

Create:

```text
performance_milestones/target08_batched_attention_indexer_row_coupling/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- compressor-fix preservation result;
- reproduction table;
- attention/indexer checkpoint diff table;
- toggle/backend bisection table;
- graph bucket 3-to-4 decision;
- fix or guard description;
- rerun results for TARGET 08.195 and TARGET 08.19, or clear reasons they were
  not rerun;
- decision for TARGET 08.20.

## Decision Rules

Proceed to TARGET 08.20 only if one of these is true:

- target-alone versus target-in-batch logits are invariant for the tested
  attention/indexer boundaries;
- remaining differences are proven harmless and bounded by a documented
  reference/fallback comparison;
- an affected backend is guarded off for prefix-cache correctness work;
- or a stable single-target slot-pinned/page-normalized oracle exists and is
  sufficient for TARGET 08.20 component-retention testing.

Keep prefix cache opt-in only if any exact-path row-coupling issue remains.

## Stop Rules

Stop and report blocked if:

- the first divergence requires a broad rewrite of DSV4 attention/indexer;
- the current debug hooks cannot isolate the lower-level owner without changing
  execution semantics;
- bisection points to multiple independent attention/indexer correctness bugs;
- fixing graph bucket 3-to-4 would require a broad CUDA graph runtime redesign.

## Non-Goals

- Implementing TARGET 08.20 or TARGET 08.21 component retention.
- Prefix-cache default promotion.
- Low-precision research.
- Performance tuning beyond what is needed to preserve correctness.
- General graph memory attribution.
