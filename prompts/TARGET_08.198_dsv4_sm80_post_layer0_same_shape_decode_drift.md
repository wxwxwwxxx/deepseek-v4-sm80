# TARGET 08.198: DSV4 Post-Layer0 Same-Shape Decode Drift Isolation

## Status

Active next TARGET 08 correctness subtarget.

Run this after TARGET 08.197 and before TARGET 08.20.

TARGET 08.197 classified the layer0 q-path issue from TARGET 08.196 as
**GEMM shape numeric drift**, not a q_norm/RoPE row-coupling bug.  That removes
one large suspect, but it does not unblock component retention: same-shape and
identical-row decode logits still drift later in the network, and some tested
pairs changed sampled tokens.

Do not start TARGET 08.20 until this target either fixes the remaining
same-shape decode drift, proves it is harmless under a stable oracle, or defines
a narrow correctness guard that TARGET 08.20 may safely use.

## Goal

Find the first owner of the remaining same-shape batched decode drift after the
layer0 q-path has been proven layout-invariant.

The target should answer:

1. In a fixed same-shape batch, where is the first checkpoint after layer0 q-path
   where target slot, filler content, or identical rows diverge?
2. Is the remaining drift caused by a real row-coupling bug, legal numerical
   drift that becomes visible only when margins are small, or a sampler/oracle
   design issue?
3. Does the first owner live in:
   - layer0 attention after q-path, for example KV/cache, SWA/C4/C128 merge, or
     WO projections;
   - later-layer attention/indexer/compressor paths;
   - MoE/shared expert paths;
   - residual/RMSNorm/HC elementwise cleanup;
   - lm_head/logit slicing;
   - sampler state or graph/eager boundary;
   - graph padded-row metadata, if graph is enabled.
4. What minimal fix, fallback, or oracle guard is required before
   SGLang-style SWA component retention can be tested?

## Current Evidence

Read and preserve the conclusions from:

- `performance_milestones/target08_q_path_same_shape_same_input_invariance/README.md`
- `performance_milestones/target08_batched_attention_indexer_row_coupling/README.md`
- `performance_milestones/target08_exact_path_slot_page_invariance/README.md`
- `prompts/TARGET_08.196_dsv4_sm80_batched_attention_indexer_row_coupling.md`
- `prompts/TARGET_08.195_dsv4_sm80_exact_path_slot_page_invariance.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`

Important facts from TARGET 08.197:

- Same-shape `bs=4, len=257` target-row comparisons are exact through
  `layer0.q_after_q_norm_rope` and `layer0.final_attention_output`, even when
  target slot or filler content changes.
- Same-input microbench proves active `q_norm_rope` and active fused
  `q_kv_norm_rope_store` are exact for the target row in single-row and batched
  layouts.
- The layer0 q-path drift appears only when changing shape from bs1 to bs4:
  small `wqa/q_wqb` differences are amplified by q_norm/RoPE and later logits.
- The q-path classification is:

```text
GEMM shape numeric drift
```

- It is not classified as:

```text
kernel row-coupling bug
```

- However, same-shape and identical-row decode logits still drift later and can
  change sampled tokens.  This target owns that remaining blocker.

Concrete TARGET 08.197 magnitudes to keep in the report:

```text
layer0.wqa_output                 0.000976562
layer0.q_wqb_output               0.00195312
layer0.q_after_q_norm_rope        0.0351562  # bs1->bs4 only
layer0.final_attention_output     0.0625     # bs1->bs4 only
prefill logits                    0.848237
```

Same-shape decode sampled-token changes observed in TARGET 08.197:

```text
target_slot0_fixed_fillers vs target_slot1_fixed_fillers: 223 -> 603
identical_prompts_batch row0 vs row2:                    322 -> 603
```

These decode changes are the problem.  Do not spend this target trying to make
bs1 and bs4 GEMMs bit-exact.

## Required Baseline

Use the promoted exact path unless a single suspected path is being bisected:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1
page_size=256
--num-pages 128
prefix cache disabled for the main oracle
```

For the main first-owner search, hold shape fixed:

```text
bs=4
prompt_len=257
decode_len >= 2
same padded graph/eager policy across compared runs
```

The main oracle must not compare `bs=1` against `bs=4`.  That comparison is
useful for numerical sensitivity notes only, not for correctness ownership.

## Required Reproductions

Build or reuse a narrow same-shape probe suite.  At minimum cover:

- `target_slot0_fixed_fillers`;
- `target_slot1_fixed_fillers`;
- `target_slot2_fixed_fillers`;
- `target_slot3_fixed_fillers`;
- `target_slot0_altA_fillers`;
- `target_slot0_altB_fillers`;
- `identical_prompts_batch`, comparing row 0 against rows 1/2/3;
- optional shape-change `single_target_alone` only as a numerical-sensitivity
  reference, not as the pass/fail oracle.

Run both:

- prefill final-token logits;
- decode step 0 and decode step 1 logits.

Decode is the priority because TARGET 08.197 saw same-shape sampled-token
changes there.

## Required Instrumentation

Use opt-in debug hooks only.  Keep large full tensors opt-in and summarize with
row hashes, max-abs, mean-abs, norms, top-k, and sampled ids.

Start from the already-clean layer0 q-path checkpoints and scan forward:

- `layer0.final_attention_output`;
- `layer0.attention_output_after_wo_a`;
- `layer0.attention_output_after_wo_b`;
- `layer0.post_attention_residual`;
- `layer0.moe_input`;
- `layer0.moe_router_logits` and selected experts, if available;
- `layer0.moe_output`;
- `layer0.layer_output`;
- for layers 1..N:
  - layer input;
  - attention input;
  - q/wkv projection outputs only if they become the first divergent owner;
  - q/k norm and RoPE only if same-shape exactness no longer holds;
  - indexer query/key/logits/top-k indices/top-k scores;
  - SWA attention output;
  - C4 sparse attention output;
  - C128 attention output;
  - merged attention output before WO_A/WO_B;
  - final attention output;
  - MoE input/output;
  - layer output;
- final norm;
- lm_head logits;
- sampler input and sampled token.

If the first observed divergence is already `layer0.final_attention_output` in a
new run, confirm whether it is caused by:

- graph/eager policy mismatch;
- stale debug tensor from TARGET 08.196/08.197 instrumentation;
- WO_A/WO_B or final attention merge rather than q-path;
- page/table location churn accidentally reintroduced into the same-shape
  oracle.

## Analysis Rules

The main comparison classes are:

```text
same-shape target slot
same-shape filler content
same-shape identical rows
```

Report them separately.  Do not merge them into one "batch drift" number.

Use this first-owner order:

```text
metadata -> layer0 q-clean boundary -> layer0 attention merge/WO ->
layer0 MoE -> later-layer attention/indexer -> later-layer MoE ->
HC/final norm -> lm_head logits -> sampler
```

For every failing pair, include:

- checkpoint name;
- phase: prefill, decode0, or decode1;
- pair name;
- max abs and mean abs;
- whether top1 id changed;
- whether sampled id changed;
- top1 margin;
- whether `2 * max_abs >= margin`;
- whether the same checkpoint is exact for identical rows.

If logits differ but sampled ids change only because top1 margins are tiny, do
not call it solved.  Instead propose a stable oracle for component-retention
tests, for example logit tolerance plus top-k/margin reporting, greedy-only
fixed seed, or a slot-pinned/page-normalized reference.

## Toggle And Fix Guidance

Only bisect after the first divergent checkpoint is known.

If the first owner is attention/indexer, prioritize:

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
- `MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE`.

If the first owner is MoE/shared expert, prioritize:

- `MINISGL_DSV4_SM80_MOE_VLLM_RUNNER`;
- `MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND`;
- shared expert staging/direct-copy paths;
- router logits/top-k stability.

If the first owner is graph/runtime, prioritize:

- `MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY`;
- graph static-input copy for real rows only;
- padded-row initialization;
- sampler graph/eager input ownership.

Do not turn this into a broad performance target.  Correctness ownership comes
first.

## Deliverables

Create:

```text
performance_milestones/target08_post_layer0_same_shape_decode_drift/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- confirmation that TARGET 08.197 q-path classification still holds;
- same-shape reproduction table;
- prefill/decode first-owner table;
- checkpoint diff tables for target-slot, filler-content, and identical-row
  comparisons;
- top-k/margin/sampled-token analysis;
- toggle bisection table only after the first owner is identified;
- implemented fix, fallback, or oracle guard if one exists;
- decision for TARGET 08.20.

## Decision Rules

Proceed to TARGET 08.20 only if one of these is true:

- same-shape decode logits are invariant for the tested target-slot,
  filler-content, and identical-row cases;
- the first owner is fixed and TARGET 08.19/08.196-style probes no longer show
  generated-token instability for the tested boundaries;
- remaining differences are proven bounded and harmless with margin/top-k
  evidence, and a stable oracle is documented for component-retention work;
- or an affected backend/shape is conservatively guarded off for prefix-cache
  correctness tests.

Keep prefix cache opt-in if any exact-path same-shape decode drift remains
unexplained.

## Stop Rules

Stop and report blocked if:

- first-owner search requires broad instrumentation that changes execution
  semantics;
- the first divergence moves between unrelated owners across repeated runs;
- bisection points to multiple independent correctness bugs;
- the only "fix" is widening tolerances without explaining top-k/margin risk;
- resolving the first owner would require a broad model-runtime rewrite.

## Non-Goals

- Implementing TARGET 08.20 or TARGET 08.21 component retention.
- Prefix-cache default promotion.
- Making bs1 and bs4 GEMMs bit-exact.
- Low-precision research.
- Performance tuning beyond what is necessary to preserve correctness.
- CUDA graph memory attribution.
