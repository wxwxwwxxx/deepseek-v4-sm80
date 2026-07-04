# TARGET 08.22 DSV4 Route B Final Prefix Promotion Gate Rerun

Date: 2026-07-04

## Result

Decision: **Route_B_preferred_opt_in**.

correctness/text/graph passed; saved-prefill and TTFT stayed close to phase-1, capacity recovery is meaningful, and the remaining output-throughput gap is attributable to guarded Route B decode metadata deforest rather than SWA-tail loss

The rerun cleared the TARGET 08.22.1 lifecycle blocker. Route B serving reports, text smoke, and graph replay completed; the remaining promotion decision is driven by performance, capacity, and SWA-tail guard impact.

## Exact Commands

Primary command:

```bash
bash performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/scripts/run_final_prefix_promotion_gate_rerun.sh
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
  --output-dir performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/raw/perf_route_b_graph
```

## Git Status Summary

```text
M benchmark/offline/deepseek_v4_perf_matrix.py
 M prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md
 M prompts/TARGET_08_radix_prefix_dsv4.md
 M prompts/target.md
 M python/minisgl/kvcache/radix_cache.py
 M python/minisgl/scheduler/cache.py
 M tests/benchmark/test_deepseek_v4_perf_matrix.py
 M tests/core/test_deepseek_v4_kvcache.py
?? performance_milestones/target08_route_b_component_mapping_lifecycle_fix/
?? performance_milestones/target08_route_b_final_prefix_promotion_gate/
?? performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/
?? prompts/TARGET_08.22.1_dsv4_sm80_route_b_component_mapping_lifecycle_fix.md
```

## Correctness

| check | result | evidence |
| --- | --- | --- |
| focused unit tests | pass | KV ownership, metadata graph copy, option guards, graph exact-bs guard |
| serving reports | pass | all perf_matrix reports completed without crash |
| Route B text smoke | pass | no invalid-byte/garbled/degenerate warning from text_sanity |
| slot-pinned guarded oracle | pass | B1/B2 CPU ownership and B3 direct-table graph-copy oracles pass; cross-slot generated equality remains diagnostic per TARGET 08.198 |
| stale read/double-free/leak | pass | component/state no-stale-reuse, repeated eviction, pool assert_no_leak |


## Text Smoke

| mode | status | replay/eager | outputs |
| --- | --- | --- | --- |
| prefix_off | pass/pass | 15/0 | 杭州西湖位于杭州市。<br>浙江省。<br>Blue.<br>Caching a shared prompt prefix reduces latency by avoiding redundant computation for repeated initial tokens |
| phase1_prefix_on | pass/pass | 15/0 | 杭州西湖位于杭州市。<br>浙江省。<br>Blue.<br>Caching a shared prompt prefix reduces latency by avoiding redundant computation for repeated initial tokens |
| route_b_graph | pass/pass | 15/0 | 杭州西湖位于杭州市。<br>浙江省。<br>Blue.<br>Caching a shared prompt prefix reduces latency by avoiding redundant computation for repeated initial tokens |


## Serving A/B

Full CSV/Markdown tables are in `summaries/serving_ab.*`.

| mode | mean TTFT s | mean output tok/s | hit rate | saved prefill | graph replay/eager |
| --- | --- | --- | --- | --- | --- |
| prefix_off | 1.0961 | 50.8501 | 0.0000 | 0 | 679/0 |
| phase1_prefix_on | 0.6946 | 65.4797 | 0.3359 | 65536 | 679/0 |
| route_b_graph | 0.7706 | 53.4904 | 0.3203 | 63232 | 679/0 |


Performance note: Route B recovered most of the prefix-cache work (`63232` vs `65536` saved prefill tokens, 0.9648 of phase-1) and mean TTFT was 0.0760 s above phase-1. Mean output throughput was 0.8169x phase-1 and 1.0519x prefix-off because Route B keeps decode metadata deforest guarded off; this overhead is visible in `summaries/deforest_guard_cost.md` but did not erase the prefix-cache TTFT win.

## Graph Replay

| mode | captured buckets | replay | eager | exact-bs | deforest guarded |
| --- | --- | --- | --- | --- | --- |
| prefix_off | - | 679 | 0 | see `summaries/graph_replay.md` | n/a |
| phase1_prefix_on | - | 679 | 0 | see `summaries/graph_replay.md` | n/a |
| route_b_graph | [1, 2, 4, 8, 16] | 679 | 0 | see `summaries/graph_replay.md` | yes |


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
| decision | Route_B_preferred_opt_in |
| reason | correctness/text/graph passed; saved-prefill and TTFT stayed close to phase-1, capacity recovery is meaningful, and the remaining output-throughput gap is attributable to guarded Route B decode metadata deforest rather than SWA-tail loss |
| correctness_ok | yes |
| text_ok | yes |
| graph_ok | yes |
| route_b_graph_replay | 679 |
| route_b_graph_eager | 0 |
| route_b_captured_buckets | [1, 2, 4, 8, 16] |
| phase1_saved_prefill_tokens | 65536 |
| route_b_saved_prefill_tokens | 63232 |
| route_b_saved_prefill_ratio_vs_phase1 | 0.9648 |
| exact_multiple_reuse_fraction | 0.0577 |
| theoretical_shortened_probe_tokens | 2304 |
| actual_saved_prefill_token_delta | -2304 |
| route_b_recovered_pages_rough_sum | 430 |

