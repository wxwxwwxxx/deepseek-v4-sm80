# TARGET 08.19 DSV4 Prefix Cache Logit And Metadata Correctness

## Result

Decision: **do not continue to TARGET 08.20 yet**.

The phase-1 prefix cache metadata boundary is clean in this probe: `cached_len`,
suffix ranges, position/seq-len rows, SWA/C4/C128/indexer semantic metadata,
physical-index valid counts, and prefix page reuse all matched the expected
boundary.

The first deterministic mismatch is not metadata.  It is the first
suffix-prefill logits for hit workloads.  Generated-token mismatch is only
secondary evidence.

Important nuance: this run also confirms the TARGET 08.10 concern that the
prefix-disabled control is not a clean oracle.  In the identical-prompt batch,
prefix-disabled logits differed across slots for the same prompt
(`row0-row1 max_abs=4.6967`, argmax `344` vs `84`), while prefix-on reused the
same cached prefix and produced identical logits across the four slots
(`max_abs=0`).  So the blocker is a DSV4 exact-path slot/page-location
correctness boundary, not a component-retention rewrite.

Default promotion remains blocked.  TARGET 08.20 should wait until either the
slot/page-dependent exact-path issue is fixed, or a slot-pinned/page-normalized
logits oracle exists for component-retention work.

## Artifacts

```text
performance_milestones/target08_prefix_cache_logit_metadata_correctness/
  README.md
  raw/
    prefix_off_graph/
    prefix_on_graph/
    prefix_on_eager/
  scripts/
    run_all_dsv4_prefix_logit_probe.sh
    run_dsv4_prefix_logit_probe.py
    summarize_dsv4_prefix_logit_probe.py
  summaries/
    comparison_summary.json
    scenario_table.md
    metadata_comparison.md
    logits_comparison.md
    generated_tokens.md
    earliest_mismatch.md
```

Each raw run contains rank0 `debug_trace/` JSON batch records, metadata `.pt`
files, and full-logits `.pt` files.

## Exact Commands

One-shot reproduction:

```bash
cd /workspace/mini-sglang
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
performance_milestones/target08_prefix_cache_logit_metadata_correctness/scripts/run_all_dsv4_prefix_logit_probe.sh
```

Expanded commands used:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_prefix_cache_logit_metadata_correctness/scripts/run_dsv4_prefix_logit_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --mode prefix_off \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir performance_milestones/target08_prefix_cache_logit_metadata_correctness/raw/prefix_off_graph

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_prefix_cache_logit_metadata_correctness/scripts/run_dsv4_prefix_logit_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --mode prefix_on \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir performance_milestones/target08_prefix_cache_logit_metadata_correctness/raw/prefix_on_graph

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_prefix_cache_logit_metadata_correctness/scripts/run_dsv4_prefix_logit_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --mode prefix_on_eager \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --output-dir performance_milestones/target08_prefix_cache_logit_metadata_correctness/raw/prefix_on_eager

python performance_milestones/target08_prefix_cache_logit_metadata_correctness/scripts/summarize_dsv4_prefix_logit_probe.py \
  --prefix-off performance_milestones/target08_prefix_cache_logit_metadata_correctness/raw/prefix_off_graph \
  --prefix-on performance_milestones/target08_prefix_cache_logit_metadata_correctness/raw/prefix_on_graph \
  --prefix-on-eager performance_milestones/target08_prefix_cache_logit_metadata_correctness/raw/prefix_on_eager \
  --output-dir performance_milestones/target08_prefix_cache_logit_metadata_correctness/summaries \
  --atol 2e-2 \
  --rtol 2e-2
```

## Git Status

```text
 M python/minisgl/engine/engine.py
?? performance_milestones/target08_prefix_cache_logit_metadata_correctness/
?? python/minisgl/utils/dsv4_prefix_debug.py
```

## Scenario Table

| Scenario | Coverage | Warm lens | Probe lens | Expected cached_len | Graph |
| --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | single-request full hit, page boundary around 256 | [257] | [257] | [256] | True |
| single_partial_hit_769_c128 | single-request partial hit, C128 boundary around 128 | [257] | [769] | [256] | True |
| identical_prompts_batch_slots | identical prompts in batch slots, single-request full hit | [257] | [257, 257, 257, 257] | [256, 256, 256, 256] | True |
| mixed_hit_miss_batch | mixed hit/miss batch | [257] | [257, 257, 257, 257, 257, 257, 257, 257] | [256, 0, 256, 0, 256, 0, 256, 0] | True |
| swa_boundary_127_128_129_no_hit | SWA boundary around 128, prefix-disabled equivalent miss path | [] | [127, 128, 129] | [0, 0, 0] | True |
| c4_boundary_partial261 | C4 boundary around 4 | [257] | [261] | [256] | True |
| page_boundary_255_256_257_258 | page boundary around 256, mixed hit/miss batch | [258] | [255, 256, 257, 258] | [0, 0, 256, 256] | True |

## Metadata Comparison

| Scenario | Expected cached | Actual cached | cached_len | suffix range | semantic metadata | physical counts | prefix page reuse | earliest metadata mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | [256] | [256] | yes | yes | yes | yes | yes | none |
| single_partial_hit_769_c128 | [256] | [256] | yes | yes | yes | yes | yes | none |
| identical_prompts_batch_slots | [256, 256, 256, 256] | [256, 256, 256, 256] | yes | yes | yes | yes | yes | none |
| mixed_hit_miss_batch | [256, 0, 256, 0, 256, 0, 256, 0] | [256, 0, 256, 0, 256, 0, 256, 0] | yes | yes | yes | yes | yes | none |
| swa_boundary_127_128_129_no_hit | [0, 0, 0] | [0, 0, 0] | yes | yes | yes | yes | yes | none |
| c4_boundary_partial261 | [256] | [256] | yes | yes | yes | yes | yes | none |
| page_boundary_255_256_257_258 | [0, 0, 256, 256] | [0, 0, 256, 256] | yes | yes | yes | yes | yes | none |

Semantic metadata means suffix-row `positions`, `seq_lens`, SWA lengths, C4/C128
lengths, C4/C128 raw indices, and indexer C4 sequence lengths.  Physical page
ids are expected to differ across independent runs, so the comparison checks
valid-count shape and prefix-on reuse of the warm page entries.

## Logits Comparison

Tolerance: `atol=2e-2`, `rtol=2e-2`, full vocabulary logits.

| Scenario | suffix prefill off/on | decode off/on | decode graph/eager | prefill top10 | decode top10 |
| --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | FAIL max=0.472356 | FAIL max=0.590127 | pass max=0 | no | no |
| single_partial_hit_769_c128 | FAIL max=1.72707 | FAIL max=10.0014 | pass max=0 | no | no |
| identical_prompts_batch_slots | FAIL max=4.86287 | FAIL max=9.56691 | pass max=0 | no | no |
| mixed_hit_miss_batch | FAIL max=4.37167 | FAIL max=10.4436 | pass max=0 | no | no |
| swa_boundary_127_128_129_no_hit | pass max=0 | FAIL max=0.245584 | FAIL max=1.9489 | yes | no |
| c4_boundary_partial261 | FAIL max=1.47294 | FAIL max=1.82026 | pass max=0 | no | no |
| page_boundary_255_256_257_258 | FAIL max=4.24537 | FAIL max=7.13711 | pass max=0 | no | no |

Prefix-on graph vs prefix-on eager prefill logits matched exactly for every
scenario.  Prefix-on graph vs eager decode also matched exactly for all hit
scenarios.  The only graph/eager decode mismatch was the no-hit SWA batch of
size 3, which is padded to graph bucket 4.

## Generated Tokens

Generated tokens are auxiliary evidence only.

| Scenario | off/on match | on graph/eager match | off tokens | on tokens | on eager tokens |
| --- | --- | --- | --- | --- | --- |
| single_full_hit_page257 | no | yes | [[89, 223]] | [[89, 269]] | [[89, 269]] |
| single_partial_hit_769_c128 | no | yes | [[294, 710]] | [[11, 223]] | [[11, 223]] |
| identical_prompts_batch_slots | no | yes | [[344, 928], [84, 223]] +2 rows | [[344, 928], [344, 928]] +2 rows | [[344, 928], [344, 928]] +2 rows |
| mixed_hit_miss_batch | no | yes | [[740, 446], [80, 201]] +6 rows | [[740, 446], [80, 201]] +6 rows | [[740, 446], [80, 201]] +6 rows |
| swa_boundary_127_128_129_no_hit | yes | yes | [[271, 988], [223, 223]] +1 rows | [[271, 988], [223, 223]] +1 rows | [[271, 988], [223, 223]] +1 rows |
| c4_boundary_partial261 | yes | yes | [[223, 223]] | [[223, 223]] | [[223, 223]] |
| page_boundary_255_256_257_258 | no | yes | [[223, 993], [223, 223]] +2 rows | [[223, 223], [223, 223]] +2 rows | [[223, 223], [223, 223]] +2 rows |

## Earliest Mismatch

| Scenario | Earliest mismatch |
| --- | --- |
| single_full_hit_page257 | suffix prefill logits |
| single_partial_hit_769_c128 | suffix prefill logits |
| identical_prompts_batch_slots | suffix prefill logits |
| mixed_hit_miss_batch | suffix prefill logits |
| swa_boundary_127_128_129_no_hit | decode logits |
| c4_boundary_partial261 | suffix prefill logits |
| page_boundary_255_256_257_258 | suffix prefill logits |

Boundary order:

```text
metadata: pass
suffix prefill logits: first failure for hit/page/C4/C128 scenarios
decode logits: downstream failure after suffix mismatch; independent bs3 graph/eager issue on SWA miss
sampled token: secondary, sometimes stable despite logit drift
```

## Conclusion For 08.20

Do **not** start TARGET 08.20 from this state.

The phase-1 prefix cache should remain explicit opt-in only.  The next useful
step is a narrow DSV4 exact-path correctness target that isolates slot/page
location dependence:

- why prefix-disabled identical prompts in different batch slots produce
  different logits;
- why no-hit decode with batch size 3 differs between graph bucket 4 replay and
  eager;
- whether a slot-pinned/page-normalized oracle can compare prefix-on suffix
  prefill against a stable prefix-disabled control.

This blocker does not ask for SGLang-style component retention and does not
justify implementing TARGET 08.20/08.21 inside 08.19.
