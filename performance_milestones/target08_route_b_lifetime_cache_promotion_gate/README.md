# TARGET 08.28 Route B Lifetime Cache Promotion Gate

## Exact Commands And Environment

```bash
MODEL_PATH=/models/DeepSeek-V4-Flash \
NPROC=8 \
SERVING_REPEATS=3 \
PREFIX_MULTI_REPEATS=3 \
EVICTION_REPEATS=2 \
performance_milestones/target08_route_b_lifetime_cache_promotion_gate/scripts/run_lifetime_cache_promotion_gate.sh
```

All matrix runs use:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --keep-going
```

Verifier runs additionally set:

```bash
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1
```

Counter profile runs additionally set:

```bash
MINISGL_DSV4_OWNER_TIMING=1
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000
```

## Git Status

```text
M benchmark/offline/deepseek_v4_perf_matrix.py
 M benchmark/offline/deepseek_v4_text_smoke.py
 M tests/attention/test_deepseek_v4_backend_metadata.py
 M tests/benchmark/test_deepseek_v4_perf_matrix.py
 M tests/benchmark/test_deepseek_v4_text_smoke.py
?? performance_milestones/target08_route_b_lifetime_cache_promotion_gate/
```

## Correctness And Text Smoke

# Verifier Results

| check | scenario | status | verifier | graph replay/eager | output |
| --- | --- | --- | --- | --- | --- |
| text_smoke | text_smoke | pass | True |  | 杭州西湖位于杭州市。 \| 浙江省。 \| Blue. |
| verify_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | pass | True | 6/0 |  |
| verify_serving_mixed_lifetime | serving_mixed_112req_wave16 | pass | True | 441/0 |  |

## Workload Throughput

# Throughput By Workload

| group | scenario | runs | output tok/s | stdev | decode prepare s | decode forward s | graph replay/eager | saved prefill | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serving_mixed | serving_mixed_112req_wave16 | 3 | 163.7220 | 1.2721 | 1.1359 | 9.8927 | 441/0 | 0 | 0 |
| prefix_multi | prefix_multi_112req_wave16 | 3 | 105.4163 | 7.6928 | 0.2868 | 1.9164 | 49/0 | 49152 | 0 |
| prefix_eviction | prefix_eviction_pressure_96req_wave16 | 2 | 13.0260 | 0.0873 | 0.1537 | 0.1917 | 6/0 | 0 | 3 |
| decode_ladder | decode_ladder_bs16 | 1 | 98.3116 | 0.0000 | 0.1639 | 1.6786 | 63/0 | 0 | 0 |

## Phase1 / Route B / Direct C4 / Lifetime Comparison

| mode | scenario | runs | output tok/s | decode prepare s | decode forward s | graph replay/eager | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | serving_mixed_112req_wave16 | 3 | 169.7381 | 0.9403 | 9.9757 | 441/0 | 08.27/08.26 frozen summary |
| Route B graph baseline | serving_mixed_112req_wave16 | 3 | 136.2373 | 4.4798 | 10.0897 | 441/0 | 08.27/08.26 frozen summary |
| Route B direct C4 | serving_mixed_112req_wave16 | 3 | 138.1281 | 4.2067 | 10.1297 | 441/0 | 08.27/08.26 frozen summary |
| Route B direct C4 + lifetime cache | serving_mixed_112req_wave16 | 3 | 162.4726 | 1.1416 | 10.0077 | 441/0 | 08.27/08.26 frozen summary |
| 08.28 Route B direct C4 + lifetime cache | serving_mixed_112req_wave16 | 3 | 163.7220 | 1.1359 | 9.8927 | 441/0 | 08.28 current gate |

## Graph Replay / Eager

# Graph Replay

| run | scenario | captured bs | requested bs | replay/eager | verifier |
| --- | --- | --- | --- | --- | --- |
| decode_ladder_lifetime | decode_ladder_bs16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 63/0 | False |
| prefix_eviction_r01_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | False |
| prefix_eviction_r02_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | False |
| prefix_multi_r01_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| prefix_multi_r02_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| prefix_multi_r03_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| profile_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | False |
| profile_prefix_multi_lifetime | prefix_multi_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 49/0 | False |
| profile_serving_mixed_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| serving_mixed_r01_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| serving_mixed_r02_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| serving_mixed_r03_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | False |
| verify_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 6/0 | True |
| verify_serving_mixed_lifetime | serving_mixed_112req_wave16 | [16, 8, 4, 2, 1] | [1, 2, 4, 8, 16] | 441/0 | True |

## Prefix And Eviction Metrics

# Prefix And Eviction Metrics

| run | scenario | hits | saved prefill | evictions | evicted tokens | retained pages |
| --- | --- | --- | --- | --- | --- | --- |
| decode_ladder_lifetime | decode_ladder_bs16 | 0 | 0 | 0 | 0 | 0 |
| prefix_eviction_r01_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| prefix_eviction_r02_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| prefix_multi_r01_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| prefix_multi_r02_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| prefix_multi_r03_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| profile_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| profile_prefix_multi_lifetime | prefix_multi_112req_wave16 | 96 | 49152 | 0 | 0 | 16 |
| profile_serving_mixed_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| serving_mixed_r01_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| serving_mixed_r02_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| serving_mixed_r03_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |
| verify_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | 0 | 0 | 3 | 20480 | 112 |
| verify_serving_mixed_lifetime | serving_mixed_112req_wave16 | 0 | 0 | 0 | 0 | 14 |

## Component Row Dirty/Clean Counters

# Component Row Counters

| run | scenario | dirty rows | clean rows | component table ms | graph replay/eager |
| --- | --- | --- | --- | --- | --- |
| profile_prefix_eviction_lifetime | prefix_eviction_pressure_96req_wave16 | 96 | 0 | 276.7635 | 6/0 |
| profile_prefix_multi_lifetime | prefix_multi_112req_wave16 | 112 | 672 | 331.4083 | 49/0 |
| profile_serving_mixed_lifetime | serving_mixed_112req_wave16 | 112 | 2576 | 352.8349 | 441/0 |

## Small Fixes Or Tests

- Preserved `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY` across benchmark/text-smoke variant env reset.
- Added an attention metadata unit test for table-slot reuse, prefix-handle movement, and active-page growth invalidation.

## Final Decision

`promote`
