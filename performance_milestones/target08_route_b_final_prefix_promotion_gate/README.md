# TARGET 08.22 DSV4 Route B Final Prefix Promotion Gate

Date: 2026-07-04

## Result

Decision: **blocked**.

Route B serving correctness failed before promotion could be evaluated

Route B failed serving correctness before promotion could be evaluated. The first failing scenario is `prefix_full_hit_512_bs4`; the rank report traceback is `RuntimeError: DSV4 component mapping is missing for active C4 full pages` from `DeepSeekV4KVCache.make_component_page_handles()` during `CacheManager.cache_req(...)`.

Failed Route B scenarios: `prefix_full_hit_512_bs4`, `prefix_full_hit_513_bs4`, `prefix_full_hit_768_bs4`, `prefix_full_hit_769_bs4`, `prefix_full_hit_513_longout_bs4`, `prefix_partial_hit_769_bs8`, `prefix_mixed_hit_miss_bs16`, `prefix_multi_112req_wave16`, `prefix_eviction_pressure_96req_wave16`

## Exact Commands

Primary command:

```bash
bash performance_milestones/target08_route_b_final_prefix_promotion_gate/scripts/run_final_prefix_promotion_gate.sh
```

The script runs focused pytest coverage, then separate `torchrun` processes for `prefix_off`, `phase1_prefix_on`, and `route_b_graph`, followed by separate TP8 text-smoke processes for the same three modes.

Key Route B command shape:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 --num-pages 128 \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir performance_milestones/target08_route_b_final_prefix_promotion_gate/raw/perf_route_b_graph
```

## Git Status Summary

```text
M benchmark/offline/deepseek_v4_perf_matrix.py
 M tests/benchmark/test_deepseek_v4_perf_matrix.py
?? performance_milestones/target08_route_b_final_prefix_promotion_gate/
```

## Correctness

| check | result | evidence |
| --- | --- | --- |
| focused unit tests | pass | KV ownership, metadata graph copy, option guards, graph exact-bs guard |
| serving reports | fail | Route B perf_matrix reports hit a serving correctness blocker |
| Route B text smoke | not_run | not run because Route B serving gate stopped on the ownership blocker |
| slot-pinned guarded oracle | pass | B1/B2 CPU ownership and B3 direct-table graph-copy oracles pass; cross-slot generated equality remains diagnostic per TARGET 08.198 |
| stale read/double-free/leak | pass | component/state no-stale-reuse, repeated eviction, pool assert_no_leak |


## Text Smoke

| mode | status | replay/eager | outputs |
| --- | --- | --- | --- |
| prefix_off | missing/None | None/None |  |
| phase1_prefix_on | missing/None | None/None |  |
| route_b_graph | missing/None | None/None |  |


## Serving A/B

Full CSV/Markdown tables are in `summaries/serving_ab.*`.

| mode | mean TTFT s | mean output tok/s | hit rate | saved prefill | graph replay/eager |
| --- | --- | --- | --- | --- | --- |
| prefix_off | 1.6016 | 47.3723 | 0.0000 | 0 | 679/0 |
| phase1_prefix_on | 0.6940 | 65.4750 | 0.3359 | 65536 | 679/0 |
| route_b_graph | 0.7923 | 92.9083 | 0.0227 | 768 | 510/0 |


## Graph Replay

| mode | captured buckets | replay | eager | exact-bs | deforest guarded |
| --- | --- | --- | --- | --- | --- |
| prefix_off | - | 679 | 0 | see `summaries/graph_replay.md` | n/a |
| phase1_prefix_on | - | 679 | 0 | see `summaries/graph_replay.md` | n/a |
| route_b_graph | [1, 2, 4, 8, 16] | 510 | 0 | see `summaries/graph_replay.md` | yes |


Route B decode metadata deforest stayed guarded off. The visible proxy for this cost is the per-scenario `decode_prepare_s` delta in `summaries/deforest_guard_cost.md`.

## Capacity Ledger

See `summaries/capacity_ledger.md` for retained full/SWA pages, C4/C128/indexer slots, state slots, and recovered full/SWA pages/tokens/GiB.

## SWA-Tail Guard

| prompt len | phase-1 hit | Route B hit | shortened |
| --- | --- | --- | --- |
| 256 | 0 | 0 | 0 |
| 257 | 256 | 256 | 0 |
| 512 | 256 | 0 | 256 |
| 513 | 512 | 512 | 0 |
| 768 | 512 | 0 | 512 |
| 769 | 768 | 768 | 0 |
| 1024 | 768 | 0 | 768 |
| 1025 | 1024 | 1024 | 0 |


Exact page-multiple frequency and actual saved-token impact are in `summaries/swa_tail_guard_workload_frequency.md` and `summaries/swa_tail_guard_actual_impact.md`.

## Final Decision Inputs

| input | value |
| --- | --- |
| decision | blocked |
| reason | Route B serving correctness failed before promotion could be evaluated |
| correctness_ok | no |
| text_ok | no |
| graph_ok | yes |
| route_b_graph_replay | 624 |
| route_b_graph_eager | 0 |
| route_b_captured_buckets | [1, 2, 4, 8, 16] |
| phase1_saved_prefill_tokens | 65536 |
| route_b_saved_prefill_tokens | 768 |
| route_b_saved_prefill_ratio_vs_phase1 | 0.0117 |
| exact_multiple_reuse_fraction | 0.0577 |
| theoretical_shortened_probe_tokens | 2304 |
| actual_saved_prefill_token_delta | -64768 |
| route_b_recovered_pages_rough_sum | 27 |

