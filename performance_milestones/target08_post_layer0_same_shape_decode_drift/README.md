# TARGET 08.198: DSV4 Post-Layer0 Same-Shape Decode Drift

Date: 2026-07-04

## Verdict

08.197's layer0 q-path classification still holds for same-input phases:
`prefill` and `decode0` are exact through `layer0.q_after_q_norm_rope` and
`layer0.final_attention_output` for target-slot, filler-content, and
identical-row comparisons.

The remaining same-shape drift first appears after that clean boundary in
later-layer attention/indexer paths:

- target-slot: `layer2.indexer_select.logits`
- filler-content: `layer1.attention_backend.merged_attention_output_before_wo`
- identical-row: `layer2.indexer_select.logits`

The first nonzero activation diffs are tiny, but later logits have small top1
margins. In `decode0`, target-slot and identical-row comparisons can change
top1 and sampled ids. `decode1` target-slot/identical-row divergence is then
sampler feedback: the second decode step consumes different tokens produced by
`decode0`, so it must not be re-attributed to layer0 q-path.

No small correctness fix was found. Focused toggles around the first owner did
not eliminate the drift:

- disabling `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE`: no fix;
- disabling `MINISGL_DSV4_SM80_PAGED_MQA_BF16`: no change;
- disabling `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16`: changes the numeric trajectory
  and failing pair, but same-shape decode sample drift remains.

## Decision For TARGET 08.20

TARGET 08.20 may continue only under a conservative correctness oracle:

- Keep the main component-retention oracle slot-pinned and same-layout.
- Do not use cross-slot, filler-content, or identical-row sampled-token equality
  as a pass/fail oracle.
- Compare logits with max/mean diffs plus top-k and margin reporting.
- Treat sampled-token changes as unstable when `2 * max_abs >= top1_margin`;
  fail only if sampled/top1 changes while the margin is safely larger than the
  observed logit envelope.
- For decode step 1+, either use a teacher-forced/fixed-token probe or label the
  result as sampler feedback once decode0 sampled ids differ.
- Keep `MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1` for graph correctness probes.

This is a guard/oracle conclusion, not a default-promotion green light.

## Prior Conclusions Preserved

TARGET 08.195:

- Fixed a real compressor bug: flattened C4/C128 compression windows could pool
  across requests when lengths were not multiples of the ratio.
- Single-request page/table churn can be exact.
- General multi-slot no-hit batches remained unsafe as a broad oracle.

TARGET 08.196:

- Preserved the compressor fix.
- Added attention/indexer debug hooks and `MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1`.
- Found layer0 q-path drift in bs1-vs-bs4 shape-change probes, but did not prove
  row-coupling.

TARGET 08.197:

- Classified the layer0 q-path issue as `GEMM shape numeric drift`, not a
  q_norm/RoPE row-coupling bug.
- Same-shape bs=4 target-row comparisons were exact through layer0 q-path and
  layer0 final attention.
- Same-shape decode logits/sampled tokens still drifted later; this target owns
  that blocker.

## Exact Commands

Baseline same-shape run:

```bash
performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/run_target08_198_post_layer0_probe.sh
```

Expanded baseline command:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/run_target08_197_q_path_probe.py \
  --mode eager \
  --output-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/raw/same_shape_eager_decode2 \
  --model-path /models/DeepSeek-V4-Flash \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --probe-max-tokens 3 \
  --prelude-max-tokens 1 \
  --capture-activations \
  --debug-attention-components \
  --max-activation-rows 4
```

Baseline summary:

```bash
python performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/summarize_target08_198_post_layer0.py \
  --run-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/raw/same_shape_eager_decode2 \
  --output-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/summaries/same_shape_eager_decode2
```

Focused toggle runs:

```bash
performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/run_target08_198_toggle_subset.sh \
  toggle_disable_indexer_fp8_cache MINISGL_DSV4_SM80_INDEXER_FP8_CACHE

performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/run_target08_198_toggle_subset.sh \
  toggle_disable_paged_mqa_bf16 MINISGL_DSV4_SM80_PAGED_MQA_BF16

performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/run_target08_198_toggle_subset.sh \
  toggle_disable_sparse_attn_bf16 MINISGL_DSV4_SM80_SPARSE_ATTN_BF16
```

Toggle summaries:

```bash
python performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/summarize_target08_198_post_layer0.py \
  --run-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/raw/toggle_disable_indexer_fp8_cache \
  --output-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/summaries/toggle_disable_indexer_fp8_cache

python performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/summarize_target08_198_post_layer0.py \
  --run-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/raw/toggle_disable_paged_mqa_bf16 \
  --output-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/summaries/toggle_disable_paged_mqa_bf16

python performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/summarize_target08_198_post_layer0.py \
  --run-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/raw/toggle_disable_sparse_attn_bf16 \
  --output-dir performance_milestones/target08_post_layer0_same_shape_decode_drift/summaries/toggle_disable_sparse_attn_bf16
```

Verification:

```bash
bash -n \
  performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/run_target08_198_post_layer0_probe.sh \
  performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/run_target08_198_toggle_subset.sh

python -m py_compile \
  performance_milestones/target08_post_layer0_same_shape_decode_drift/scripts/summarize_target08_198_post_layer0.py

git diff --check
```

All verification commands passed.

## Git Status Summary

The worktree was already dirty before this target. This target added only:

```text
performance_milestones/target08_post_layer0_same_shape_decode_drift/
```

Run-time `git status --short`:

```text
 M performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py
 M prompts/TARGET_08.196_dsv4_sm80_batched_attention_indexer_row_coupling.md
 M prompts/TARGET_08.20_dsv4_sm80_sglang_style_swa_component_retention.md
 M prompts/TARGET_08_radix_prefix_dsv4.md
 M prompts/target.md
 M python/minisgl/attention/deepseek_v4.py
 M python/minisgl/engine/graph.py
 M python/minisgl/kernel/deepseek_v4.py
 M python/minisgl/models/deepseek_v4.py
 M tests/kernel/test_deepseek_v4_wrappers.py
?? performance_milestones/target08_batched_attention_indexer_row_coupling/
?? performance_milestones/target08_post_layer0_same_shape_decode_drift/
?? performance_milestones/target08_q_path_same_shape_same_input_invariance/
?? prompts/TARGET_08.198_dsv4_sm80_post_layer0_same_shape_decode_drift.md
?? tests/engine/test_graph_runner.py
```

## Same-Shape Reproduction Table

Full table:
`summaries/same_shape_eager_decode2/same_shape_reproduction.md`.

| Scenario | Prompt lens | Target row | Prefill bs | Decode0 source | Decode1 source | Decode padded | Target table_idx |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_target_alone | 257 | 0 | 1 | eager | eager | 1 | 15 |
| identical_prompts_batch | 257,257,257,257 | 0 | 4 | eager | eager | 4 | 15 |
| target_slot0_fixed_fillers | 257,257,257,257 | 0 | 4 | eager | eager | 4 | 12 |
| target_slot1_fixed_fillers | 257,257,257,257 | 1 | 4 | eager | eager | 4 | 14 |
| target_slot2_fixed_fillers | 257,257,257,257 | 2 | 4 | eager | eager | 4 | 14 |
| target_slot3_fixed_fillers | 257,257,257,257 | 3 | 4 | eager | eager | 4 | 12 |
| target_slot0_altA_fillers | 257,257,257,257 | 0 | 4 | eager | eager | 4 | 12 |
| target_slot0_altB_fillers | 257,257,257,257 | 0 | 4 | eager | eager | 4 | 15 |

`single_target_alone` is retained only as a sensitivity reference; it is not the
pass/fail oracle for this target.

## 08.197 Q-Path Classification Check

Full table:
`summaries/same_shape_eager_decode2/q_path_classification_check.md`.

| Phase | Checkpoint | Target-slot | Filler-content | Identical-row |
| --- | --- | --- | --- | --- |
| prefill | layer0.q_after_q_norm_rope | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| prefill | layer0.final_attention_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.q_after_q_norm_rope | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.final_attention_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode1 | layer0.q_after_q_norm_rope | diff after sampler feedback for target-slot/identical | exact for filler-content | diff after sampler feedback for identical-row |

Interpretation: 08.197 still holds for same-input phases. Decode1 is natural
autoregressive continuation; once decode0 samples differ, decode1 is no longer a
same-input q-path test.

## First-Owner Table

Full table:
`summaries/same_shape_eager_decode2/first_owner_table.md`.

| Phase | Group | Owner bucket | First checkpoint | Worst pair | Max abs | Mean abs |
| --- | --- | --- | --- | --- | ---: | ---: |
| prefill | target-slot | later-layer attention/indexer | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.000812054 | 1.26883e-05 |
| prefill | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000976562 | 4.592e-07 |
| prefill | identical-row | later-layer attention/indexer | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 6.81877e-05 | 1.06543e-06 |
| decode0 | target-slot | later-layer attention/indexer | layer2.indexer_select.logits | target_slot0_fixed_fillers[0] vs target_slot3_fixed_fillers[3] | 0.000867605 | 1.35563e-05 |
| decode0 | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.000244141 | 1.52737e-07 |
| decode0 | identical-row | later-layer attention/indexer | layer2.indexer_select.logits | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.000104427 | 1.63168e-06 |
| decode1 | target-slot | sampler feedback | decode0.sampled_token_ids | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 0.129613 |
| decode1 | filler-content | later-layer attention/indexer | layer1.attention_backend.merged_attention_output_before_wo | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 6.10352e-05 | 1.67638e-08 |
| decode1 | identical-row | sampler feedback | decode0.sampled_token_ids | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 0.10798 |

Physical page/full-index tensors were excluded from semantic first-owner
selection. Indexer logits/raw/top-k score checkpoints were retained.

## Checkpoint Diff By Group

Full selected checkpoint table:
`summaries/same_shape_eager_decode2/checkpoint_diff_selected.md`.

Key decode0 checkpoints:

| Group | layer0.final_attention_output | First later checkpoint | final_norm | lm_head_logits |
| --- | --- | --- | --- | --- |
| target-slot | exact max=0 | layer2.indexer_select.logits max=0.000867605 | max=0.15625 | max=1.51967 |
| filler-content | exact max=0 | layer1.attention_backend.merged_attention_output_before_wo max=0.000244141 | max=0.125 | max=0.9766 |
| identical-row | exact max=0 | layer2.indexer_select.logits max=0.000104427 | max=0.1875 | max=0.940769 |

This is the main post-layer0 evidence: layer0 q-path and layer0 final attention
are clean, then later-layer attention/indexer emits tiny nonzero differences
that later become large enough at logits.

## Top-K, Margin, Sampled-Token Analysis

Full table:
`summaries/same_shape_eager_decode2/topk_margin_sampled_analysis.md`.

Representative rows:

| Phase | Group | Pair | Logit max abs | Top1 ids | Sampled ids | Left top1 margin | 2*max_abs >= margin |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| prefill | target-slot | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | 1.22151 | 32->32 | 32->32 | 0.929865 | yes |
| decode0 | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 223->603 | 223->603 | 0.156403 | yes |
| decode0 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.9766 | 223->223 | 223->223 | 0.156403 | yes |
| decode0 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 322->603 | 322->603 | 0.116428 | yes |
| decode1 | target-slot | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 8.30493 | 223->327 | 223->327 | 2.06961 | yes |
| decode1 | filler-content | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.633387 | 223->223 | 223->223 | 2.06961 | no |
| decode1 | identical-row | identical_prompts_batch[0] vs identical_prompts_batch[2] | 7.86887 | 1018->223 | 1018->223 | 0.704569 | yes |

All sampled-token changes occur in rows where `2 * max_abs >= margin`, so the
observed sample changes are consistent with small-margin instability. They are
still real oracle instability and should not be hidden by tolerance-only
acceptance.

## Toggle Bisection

Toggle bisection started only after the first owner was identified as
later-layer attention/indexer.

| Run | Disabled toggle | Decode0 first owner | Decode0 sampled-change status | Conclusion |
| --- | --- | --- | --- | --- |
| same_shape_eager_decode2 | none | target-slot/identical: layer2.indexer_select.logits; filler: layer1 attention merge | target-slot 223->603; identical 322->603 | baseline fails |
| toggle_disable_indexer_fp8_cache | MINISGL_DSV4_SM80_INDEXER_FP8_CACHE | same owner class; layer2 indexer logits remains first for target/identical | same sampled changes as baseline | not owner |
| toggle_disable_paged_mqa_bf16 | MINISGL_DSV4_SM80_PAGED_MQA_BF16 | unchanged from baseline | same sampled changes as baseline | not owner |
| toggle_disable_sparse_attn_bf16 | MINISGL_DSV4_SM80_SPARSE_ATTN_BF16 | target-slot moves to layer4.indexer_select.logits; filler remains layer1 attention; identical remains layer2 indexer | drift remains, failing pair changes | changes trajectory, not fix |

`MINISGL_DSV4_FORCE_TORCH_TOPK=1` was not run here because the baseline first
owner is before top-k index selection for the failing target/identical decode0
cases, and TARGET 08.196 had already shown raw top-k indices were not the first
semantic owner.

## Fix, Guard, Oracle

No small code fix was made. The evidence points to legal numerical drift in
later-layer attention/indexer paths that becomes sample-visible under small
top1 margins, plus sampler feedback on subsequent decode steps.

Guard for follow-on work:

- use slot-pinned, same-layout logits as the component-retention oracle;
- keep cross-slot/filler/identical-row suites as diagnostics, not pass/fail;
- report max_abs, mean_abs, top-k stability, sampled ids, top1 margin, and
  `2 * max_abs >= margin`;
- treat decode1+ natural autoregressive mismatches as sampler feedback unless a
  teacher-forced fixed-token decode probe is added;
- do not promote prefix/component-retention based on generated-token equality
  across multi-row batches.

Continue TARGET 08.20:

```text
Yes, guarded only.
```

Do not interpret this as broad DSV4 multi-request exactness or default prefix
promotion readiness.

## Artifacts

Raw:

```text
raw/same_shape_eager_decode2/                 2.8G
raw/toggle_disable_indexer_fp8_cache/         2.0G
raw/toggle_disable_paged_mqa_bf16/            2.0G
raw/toggle_disable_sparse_attn_bf16/          2.0G
```

Summaries:

```text
summaries/same_shape_eager_decode2/
summaries/toggle_disable_indexer_fp8_cache/
summaries/toggle_disable_paged_mqa_bf16/
summaries/toggle_disable_sparse_attn_bf16/
```
