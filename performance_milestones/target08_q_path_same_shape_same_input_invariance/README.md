# TARGET 08.197: DSV4 Q-Path Same-Shape/Same-Input Invariance

Date: 2026-07-04

## Verdict

For the layer0 q-path drift identified in 08.196, the root cause is classified
as **GEMM shape numeric drift**, not a q_norm/RoPE row-coupling bug.

Evidence:

- Same-shape bs=4/len=257 target-row comparisons are exact through
  `layer0.q_after_q_norm_rope` and `layer0.final_attention_output`, even when
  the target slot or filler content changes.
- Same-input microbench uses identical `layer0.q_wqb_output`/`wkv_output`
  tensors in single-row and batched-row layouts. Active q_norm_rope and active
  fused q_kv_norm_rope_store both return exact target-row matches for every
  target slot.
- The only layer0 q-path drift appears when changing shape from bs1 to bs4:
  small `wqa/q_wqb` differences are amplified by q_norm/RoPE and later logits.

This does **not** green-light 08.20. Same-shape and identical-row decode logits
still drift later in the network, with observed sampled-token changes. The next
target should fix or guard that post-layer0 same-shape batched logit drift
before component retention.

## Exact Commands

Same-shape eager probe:

```bash
performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/run_target08_197_same_shape_probe.sh
```

Expanded form:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/run_target08_197_q_path_probe.py \
  --mode eager \
  --output-dir performance_milestones/target08_q_path_same_shape_same_input_invariance/raw/same_shape_eager \
  --model-path /models/DeepSeek-V4-Flash \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --probe-max-tokens 2 \
  --prelude-max-tokens 1 \
  --capture-activations \
  --debug-attention-components \
  --max-activation-rows 4
```

Same-input q_norm_rope microbench:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
python performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/q_norm_rope_same_input_microbench.py \
  --run-dir performance_milestones/target08_q_path_same_shape_same_input_invariance/raw/same_shape_eager \
  --output-dir performance_milestones/target08_q_path_same_shape_same_input_invariance/summaries/same_input_microbench \
  --model-path /models/DeepSeek-V4-Flash \
  --device cuda:0
```

Summary:

```bash
python performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/summarize_target08_197_q_path.py \
  --run-dir performance_milestones/target08_q_path_same_shape_same_input_invariance/raw/same_shape_eager \
  --microbench-json performance_milestones/target08_q_path_same_shape_same_input_invariance/summaries/same_input_microbench/same_input_q_norm_rope_microbench.json \
  --output-dir performance_milestones/target08_q_path_same_shape_same_input_invariance/summaries/same_shape_eager \
  --activation-atol 2e-2 \
  --activation-rtol 2e-2
```

Verification:

```bash
python -m py_compile \
  performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/run_target08_197_q_path_probe.py \
  performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/q_norm_rope_same_input_microbench.py \
  performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/summarize_target08_197_q_path.py
```

## Git Status Summary

At run time the worktree already included the 08.196 modifications/artifacts.
This target adds only files under:

```text
performance_milestones/target08_q_path_same_shape_same_input_invariance/
```

Run-time `git status --short` summary:

```text
 M performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py
 M python/minisgl/attention/deepseek_v4.py
 M python/minisgl/engine/graph.py
 M python/minisgl/kernel/deepseek_v4.py
 M python/minisgl/models/deepseek_v4.py
 M tests/kernel/test_deepseek_v4_wrappers.py
?? performance_milestones/target08_batched_attention_indexer_row_coupling/
?? performance_milestones/target08_q_path_same_shape_same_input_invariance/
?? tests/engine/test_graph_runner.py
```

## Same-Shape Oracle

Controlled scenarios:

| Scenario | Prompt lens | Target row | Prefill bs | Decode source | Decode padded | Target table_idx |
| --- | --- | --- | --- | --- | --- | --- |
| single_target_alone | 257 | 0 | 1 | eager | 1 | 15 |
| identical_prompts_batch | 257,257,257,257 | 0 | 4 | eager | 4 | 15 |
| target_slot0_fixed_fillers | 257,257,257,257 | 0 | 4 | eager | 4 | 12 |
| target_slot1_fixed_fillers | 257,257,257,257 | 1 | 4 | eager | 4 | 14 |
| target_slot2_fixed_fillers | 257,257,257,257 | 2 | 4 | eager | 4 | 14 |
| target_slot3_fixed_fillers | 257,257,257,257 | 3 | 4 | eager | 4 | 12 |
| target_slot0_altA_fillers | 257,257,257,257 | 0 | 4 | eager | 4 | 12 |
| target_slot0_altB_fillers | 257,257,257,257 | 0 | 4 | eager | 4 | 15 |

Layer0 target-row comparison:

| Phase | Checkpoint | Shape change bs1->bs4 | Same-shape target slot | Same-shape filler content | Identical rows |
| --- | --- | --- | --- | --- | --- |
| prefill | layer0.attention_input | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.wqa_output | pass max=0.000976562 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.q_lora_after_norm | pass max=0.000488281 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.q_wqb_output | pass max=0.00195312 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.q_after_q_norm_rope | FAIL max=0.0351562 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.final_attention_output | FAIL max=0.0625 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.attention_input | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.wqa_output | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.q_lora_after_norm | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.q_wqb_output | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.q_after_q_norm_rope | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.final_attention_output | FAIL max=0.078125 | pass max=0, exact | pass max=0, exact | pass max=0, exact |

Interpretation: layer0 q-path is invariant to target row position and filler
content when the flattened GEMM shape is held fixed. The 08.196 prefill
`layer0.q_after_q_norm_rope` drift only appears for bs1 shape versus bs4 shape.

## Same-Input Microbench

Inputs: captured `single_target_alone` `layer0.q_wqb_output` and `wkv_output`.
The exact same target row is inserted into a synthetic bs=4 layout at slots
0/1/2/3. All rows use position `256`.

Active toggles:

```text
q_norm_rope_triton_enabled=true
fused_q_kv_norm_rope_store_enabled=true
fused_q_kv_backend=q_kv_norm_rope_cache_bf16
```

| Target slot | Reference single-vs-batch | Active q_norm single-vs-batch | Fused q_kv single-vs-batch | Active q_norm vs ref | Fused q_kv vs ref |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0.0625 | 0.0625 |
| 1 | 0 | 0 | 0 | 0.0625 | 0.0625 |
| 2 | 0 | 0 | 0 | 0.0625 | 0.0625 |
| 3 | 0 | 0 | 0 | 0.0625 | 0.0625 |

The active fused path differs from the pure torch/FP32-arithmetic reference at
max `0.0625`, but that difference is identical for single-row and batched-row
layouts. This is not row coupling.

## Reference Path

The microbench implements a pure torch reference for q RMSNorm plus RoPE:

- q normalization is computed in FP32;
- RoPE frequencies/cos/sin are computed in torch FP32;
- output is cast back to BF16 at the same boundary as the runtime q tensor.

08.196 also ran a broad full-model bisection with
`MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE` disabled. That did not fix the
batched logit drift and worsened logits in that quick oracle. The same-input
result here is the narrower evidence needed for q-path classification: the
active q_norm/RoPE kernels are layout-invariant for identical input.

## Magnitude And Top-K

Shape-change propagation, prefill target row:

| Shape-change path | Max abs | Mean abs | Gain vs previous |
| --- | --- | --- | --- |
| layer0.attention_input | 0 | 0 | n/a |
| layer0.wqa_output | 0.000976562 | 4.26938e-05 | n/a |
| layer0.q_lora_after_norm | 0.000488281 | 2.49695e-05 | 0.5x |
| layer0.q_wqb_output | 0.00195312 | 0.000203445 | 4x |
| layer0.q_after_q_norm_rope | 0.0351562 | 0.00533461 | 18x |
| layer0.final_attention_output | 0.0625 | 0.0138616 | 1.78x |
| logits(prefill) | 0.848237 | 0.145079 | 13.6x |

Top-k/logit summary:

| Phase | Group | Worst pair | Logit max abs | Top1 ids | Top10 same | Sampled ids | Sampled same | Left top1 margin | 2*max_abs >= margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | shape_change | single_target_alone[0] vs target_slot0_fixed_fillers[0] | 0.848237 | 32->32 | no | 32->32 | yes | 1.11026 | yes |
| prefill | same_shape_position | target_slot0_fixed_fillers[0] vs target_slot2_fixed_fillers[2] | 1.22151 | 32->32 | no | 32->32 | yes | 0.929865 | yes |
| prefill | same_shape_filler | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 1.00624 | 32->32 | no | 32->32 | yes | 0.929865 | yes |
| prefill | identical_rows | identical_prompts_batch[0] vs identical_prompts_batch[3] | 0.887962 | 32->32 | no | 32->32 | yes | 1.07734 | yes |
| decode | shape_change | single_target_alone[0] vs target_slot0_fixed_fillers[0] | 0.756046 | 223->223 | no | 223->223 | yes | 0.149835 | yes |
| decode | same_shape_position | target_slot0_fixed_fillers[0] vs target_slot1_fixed_fillers[1] | 1.51967 | 223->603 | no | 223->603 | no | 0.156403 | yes |
| decode | same_shape_filler | target_slot0_fixed_fillers[0] vs target_slot0_altA_fillers[0] | 0.9766 | 223->223 | no | 223->223 | yes | 0.156403 | yes |
| decode | identical_rows | identical_prompts_batch[0] vs identical_prompts_batch[2] | 0.940769 | 322->603 | no | 322->603 | no | 0.116428 | yes |

The shape-change q-path drift is large enough to close the observed top1 margin
in decode (`2 * 0.756046 >= 0.149835`), even though this exact shape-change pair
kept the same sampled id. Separately, same-shape/identical decode pairs already
show sampled-token changes, so 08.19/08.196 generated-token mismatch is not
blocked on a q_norm/RoPE kernel bug.

## Artifacts

- `raw/same_shape_eager/` (`run.json` plus rank0 debug trace, about 2.3 GiB)
- `summaries/same_input_microbench/same_input_q_norm_rope_microbench.{json,md}`
- `summaries/same_shape_eager/same_shape_comparison.md`
- `summaries/same_shape_eager/magnitude_propagation.md`
- `summaries/same_shape_eager/logits_topk_analysis.md`
- `summaries/same_shape_eager/reference_path_comparison.md`
- `summaries/same_shape_eager/comparison_summary.json`

## Decision

Root-cause classification for the 08.196 layer0 q-path drift:

```text
GEMM shape numeric drift
```

Not classified as:

```text
kernel row-coupling bug
```

Continue to 08.20:

```text
No.
```

Reason: q-path same-shape/same-input invariance is proven, but same-shape and
identical-row decode logits still drift later and can change sampled tokens.
The next target should isolate that post-layer0 same-shape batched drift before
prefix component retention.
