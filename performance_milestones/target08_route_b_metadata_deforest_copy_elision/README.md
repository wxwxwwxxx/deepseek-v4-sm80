# TARGET 08.24 DSV4 Route B Metadata Deforest And Copy Elision

Date: 2026-07-04

## Result

Decision: **keep_experimental**.

The implementation keeps Route B metadata deforest behind `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`.  The new path consumes `c4_page_table`, `c128_page_table`, and `c4_indexer_page_table` instead of deriving component locations with `full_loc // 4` or `full_loc // 128`.

## Exact Command

```bash
bash performance_milestones/target08_route_b_metadata_deforest_copy_elision/scripts/run_route_b_metadata_deforest_copy_elision.sh
```

## Git Status Summary

```text
 M benchmark/offline/deepseek_v4_perf_matrix.py
 M benchmark/offline/deepseek_v4_text_smoke.py
 M python/minisgl/attention/deepseek_v4.py
 M python/minisgl/engine/engine.py
 M python/minisgl/kernel/deepseek_v4.py
 M python/minisgl/kernel/triton/deepseek_v4.py
 M python/minisgl/server/args.py
 M tests/kernel/test_deepseek_v4_wrappers.py
?? performance_milestones/target08_route_b_metadata_deforest_copy_elision/
```

## Overhead Attribution

Route B baseline still builds eager decode metadata and stages it into graph buffers.  The largest repeated tensors are SWA indices, C4 sparse raw/page/full indices, C128 raw/page/full indices, and component page tables.  The deforest opt-in moves C4/C128 index assembly to one component-aware Triton kernel and generates replay write locs from component page tables.

The perf decision uses the uninstrumented `perf_route_b_metadata_deforest` run.  Owner-timing counters come from the separate `perf_route_b_metadata_deforest_profile` run so profiling overhead does not pollute throughput comparisons.

See `summaries/metadata_build_bytes.md` and `summaries/replay_copy_bytes.md` for field-level byte counters from the owner-timing profile run.  Per-scenario attribution is in `summaries/*_by_scenario.md`.

## Field Stability

| class | fields |
| --- | --- |
| per-token | `raw_out_loc`, `positions`, `seq_lens`, SWA indices/lengths, C4 sparse metadata, write locs |
| per-request/per-hit | `page_table`, `c4_page_table`, `c128_page_table`, `c4_indexer_page_table`, C128 prefix metadata |
| per-bucket | `cu_seqlens_q`, graph capture buffers |

## Component Formula

`component_loc = component_page_table[row, raw_index // component_page_size] * component_page_size + raw_index % component_page_size`.  Missing or tombstoned component pages yield `-1`; full-token tombstones only affect `*_full_indices`, not component page indices.

## Serving A/B

Full table: `summaries/serving_ab.md`.

| mode | mean TTFT s | mean output tok/s | decode prepare s | saved prefill | graph replay/eager |
| --- | --- | --- | --- | --- | --- |
| prefix_off | 1.0818 | 50.6558 | 1.4710 | 0 | 679/0 |
| phase1_prefix_on | 0.6981 | 64.8296 | 1.4436 | 183040 | 679/0 |
| route_b_graph_baseline | 0.7909 | 52.5981 | 7.3987 | 165376 | 679/0 |
| route_b_metadata_deforest | 0.7733 | 47.4662 | 13.2645 | 165376 | 679/0 |

## Deforest Effect

Full table: `summaries/deforest_effect.md`.

## Gate Notes

This gate does **not** promote the Route B metadata deforest path.  Correctness, text smoke, saved-prefill, and graph replay all pass, but the performance threshold is not met: aggregate decode prepare increases versus Route B baseline and output throughput remains below 0.90x phase1.

The component-aware helper is active and safe, but it mostly changes how decode metadata is generated.  It does not eliminate the dominant graph staging copies of `c4_sparse_*`, `swa_page_indices`, and `c128_*` buffers.  The next useful lever is direct graph-buffer generation or row reuse for stable request/prefix-hit fields, not SWA-tail ownership work or attention kernel algorithm tuning.

## Text Smoke

Full table: `summaries/text_smoke.md`.

## Correctness And Safety

| check | result |
| --- | --- |
| focused unit/kernel tests | pass if `raw/pytest_route_b_metadata_deforest_correctness.log` has exit 0 |
| component tombstone fail-safe | kernel test covers live component pages with tombstoned/missing full pages |
| graph buckets [1,2,4,8,16] | see serving A/B graph replay/eager counts |
| text smoke | see `summaries/text_smoke.md` |

## Decision Inputs

| input | value |
| --- | --- |
| decision | keep_experimental |
| correctness_ok | yes |
| text_ok | yes |
| graph_ok | yes |
| saved_prefill_ratio_vs_phase1 | 0.9035 |
| decode_prepare_reduction_vs_route_b | -0.7928 |
| output_throughput_vs_phase1 | 0.7322 |
| baseline_output_tok_s | 52.5981 |
| deforest_output_tok_s | 47.4662 |
