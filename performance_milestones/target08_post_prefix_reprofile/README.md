# TARGET 08.30 Post-Prefix Reprofile

## Final Configuration

Promoted prefix variant: `dsv4_sm80_a100_victory_prefix_routeb_lifetime`.

Final promoted prefix path:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--page-size 256
--num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Control variant: `dsv4_sm80_a100_victory` with page size 256, num pages 128, and graph buckets 1 2 4 8 16.

## Commands And Environment

```bash
MODEL_PATH=/models/DeepSeek-V4-Flash \
NPROC=8 \
HISTORICAL_REPEATS=3 \
SERVING_REPEATS=3 \
RUN_OWNER_TIMING=0 \
performance_milestones/target08_post_prefix_reprofile/scripts/run_post_prefix_reprofile.sh

RUN_TEXT_SMOKE=0 \
RUN_VERIFY=0 \
RUN_MACRO=0 \
RUN_OWNER_TIMING=1 \
performance_milestones/target08_post_prefix_reprofile/scripts/run_post_prefix_reprofile.sh
```

Promoted prefix matrix shape:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Non-prefix TARGET 07 control shape:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 --num-pages 128 \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Owner timing runs additionally set:

```bash
MINISGL_DSV4_OWNER_TIMING=1
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000
```

## Git Status

```text
?? performance_milestones/target08_post_prefix_reprofile/
```

## Correctness / Verifier

| check | status | verifier | graph | outputs |
| --- | --- | --- | --- | --- |
| text_smoke_promoted_verify | pass | True | 5/0 | 杭州西湖位于杭州市。 \| 浙江省。 \| Blue. |

Verifier matrix runs are included in the graph and prefix tables below.

## Workload Throughput Tables

# Workload Throughput

| variant | scenario | runs | out tok/s | stdev | CV | TTFT ms | TPOT/ITL ms | prefill fwd s | decode prep s | decode fwd s | graph | saved | evict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| promoted_prefix | decode_ladder_bs16 | 3 | 130.7605 | 0.1441 | 0.0011 | 1452.4240 | 30.5793 | 1.3516 | 0.1577 | 1.3665 | 63/0 | 0 | 0 |
| promoted_prefix | historical_4096_1024_bs4 | 3 | 137.1625 | 0.1960 | 0.0014 | 5048.6030 | 24.2519 | 4.2998 | 2.2245 | 22.2977 | 1023/0 | 0 | 2 |
| promoted_prefix | historical_4096_128_bs4 | 3 | 62.8933 | 0.1391 | 0.0022 | 5041.8038 | 24.3931 | 4.2998 | 0.2814 | 2.7684 | 127/0 | 0 | 1 |
| promoted_prefix | prefix_eviction_pressure_96req_wave16 | 3 | 13.0827 | 0.0250 | 0.0019 | 2273.9510 | 171.6111 | 10.8178 | 0.1464 | 0.1714 | 6/0 | 0 | 3 |
| promoted_prefix | prefix_multi_112req_wave16 | 3 | 110.1417 | 0.3680 | 0.0033 | 854.9569 | 43.8182 | 4.5400 | 0.2779 | 1.4750 | 49/0 | 49152 | 0 |
| promoted_prefix | serving_mixed_112req_wave16 | 3 | 163.3985 | 0.5651 | 0.0035 | 796.7467 | 31.8282 | 4.7832 | 1.0919 | 9.7624 | 441/0 | 0 | 0 |
| target07_control | decode_ladder_bs16 | 3 | 135.5353 | 3.0061 | 0.0222 | 1461.2713 | 26.9086 | 1.3874 | 0.1295 | 1.3297 | 63/0 | 0 | 0 |
| target07_control | historical_4096_1024_bs4 | 3 | 139.8415 | 0.1143 | 0.0008 | 5001.6229 | 23.7401 | 4.2955 | 1.9793 | 22.0498 | 1023/0 | 0 | 0 |
| target07_control | historical_4096_128_bs4 | 3 | 63.7732 | 0.0965 | 0.0015 | 5012.5199 | 23.7412 | 4.2913 | 0.2525 | 2.7258 | 127/0 | 0 | 0 |
| target07_control | prefix_eviction_pressure_96req_wave16 | 3 | 15.0270 | 0.0612 | 0.0041 | 2088.8871 | 40.2401 | 10.4995 | 0.0175 | 0.1453 | 6/0 | 0 | 0 |
| target07_control | prefix_multi_112req_wave16 | 3 | 51.0507 | 0.1588 | 0.0031 | 2304.1904 | 28.9489 | 13.5254 | 0.1324 | 1.1799 | 49/0 | 0 | 0 |
| target07_control | serving_mixed_112req_wave16 | 3 | 178.3004 | 0.1715 | 0.0010 | 760.6074 | 26.8178 | 4.7665 | 0.9130 | 9.2552 | 441/0 | 0 | 0 |

## Graph Replay / Eager Coverage

# Graph Coverage

| kind | run | variant | scenario | requested | captured | replay/eager | replay by padded | eager by bs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro | macro_promoted_decode_ladder_bs16_r01 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_promoted_decode_ladder_bs16_r02 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_promoted_decode_ladder_bs16_r03 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_promoted_historical_4096_1024_bs4_r01 | promoted_prefix | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_promoted_historical_4096_1024_bs4_r02 | promoted_prefix | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_promoted_historical_4096_1024_bs4_r03 | promoted_prefix | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_promoted_historical_4096_128_bs4_r01 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_promoted_historical_4096_128_bs4_r02 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_promoted_historical_4096_128_bs4_r03 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r01 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r02 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r03 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_promoted_prefix_multi_112req_wave16_r01 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_promoted_prefix_multi_112req_wave16_r02 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_promoted_prefix_multi_112req_wave16_r03 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_promoted_serving_mixed_112req_wave16_r01 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_promoted_serving_mixed_112req_wave16_r02 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_promoted_serving_mixed_112req_wave16_r03 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_control_decode_ladder_bs16_r01 | target07_control | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_control_decode_ladder_bs16_r02 | target07_control | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_control_decode_ladder_bs16_r03 | target07_control | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| macro | macro_control_historical_4096_1024_bs4_r01 | target07_control | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_control_historical_4096_1024_bs4_r02 | target07_control | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_control_historical_4096_1024_bs4_r03 | target07_control | historical_4096_1024_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 1023/0 | {"4": 1023} | {} |
| macro | macro_control_historical_4096_128_bs4_r01 | target07_control | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_control_historical_4096_128_bs4_r02 | target07_control | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_control_historical_4096_128_bs4_r03 | target07_control | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| macro | macro_control_prefix_eviction_pressure_96req_wave16_r01 | target07_control | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_control_prefix_eviction_pressure_96req_wave16_r02 | target07_control | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_control_prefix_eviction_pressure_96req_wave16_r03 | target07_control | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| macro | macro_control_prefix_multi_112req_wave16_r01 | target07_control | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_control_prefix_multi_112req_wave16_r02 | target07_control | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_control_prefix_multi_112req_wave16_r03 | target07_control | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| macro | macro_control_serving_mixed_112req_wave16_r01 | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_control_serving_mixed_112req_wave16_r02 | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| macro | macro_control_serving_mixed_112req_wave16_r03 | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| profile | profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 63/0 | {"1": 16, "16": 15, "2": 16, "4": 8, "8": 8} | {} |
| profile | profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 127/0 | {"4": 127} | {} |
| profile | profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| profile | profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 49/0 | {"16": 49} | {} |
| profile | profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| profile | profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |
| verify | verify_promoted_prefix_eviction | promoted_prefix | prefix_eviction_pressure_96req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 6/0 | {"16": 6} | {} |
| verify | verify_promoted_serving_mixed | promoted_prefix | serving_mixed_112req_wave16 | [1, 2, 4, 8, 16] | [16, 8, 4, 2, 1] | 441/0 | {"1": 112, "16": 105, "2": 112, "4": 56, "8": 56} | {} |

## Prefix Hit / Miss / Saved-Prefill / Eviction Metrics

# Prefix Metrics

| kind | run | scenario | hits | misses | saved prefill | evictions | evicted tokens | retained pages | retained MiB | SWA MiB | available comp pages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro | macro_promoted_decode_ladder_bs16_r01 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| macro | macro_promoted_decode_ladder_bs16_r02 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| macro | macro_promoted_decode_ladder_bs16_r03 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| macro | macro_promoted_historical_4096_1024_bs4_r01 | historical_4096_1024_bs4 | 0 | 4 | 0 | 2 | 9728 | 114 | 2099.7876 | 1225.5000 | 15 |
| macro | macro_promoted_historical_4096_1024_bs4_r02 | historical_4096_1024_bs4 | 0 | 4 | 0 | 2 | 9728 | 114 | 2099.7876 | 1225.5000 | 15 |
| macro | macro_promoted_historical_4096_1024_bs4_r03 | historical_4096_1024_bs4 | 0 | 4 | 0 | 2 | 9728 | 114 | 2099.7876 | 1225.5000 | 15 |
| macro | macro_promoted_historical_4096_128_bs4_r01 | historical_4096_128_bs4 | 0 | 4 | 0 | 1 | 4096 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_historical_4096_128_bs4_r02 | historical_4096_128_bs4 | 0 | 4 | 0 | 1 | 4096 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_historical_4096_128_bs4_r03 | historical_4096_128_bs4 | 0 | 4 | 0 | 1 | 4096 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r01 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r02 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_eviction_pressure_96req_wave16_r03 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| macro | macro_promoted_prefix_multi_112req_wave16_r01 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| macro | macro_promoted_prefix_multi_112req_wave16_r02 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| macro | macro_promoted_prefix_multi_112req_wave16_r03 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| macro | macro_promoted_serving_mixed_112req_wave16_r01 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| macro | macro_promoted_serving_mixed_112req_wave16_r02 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| macro | macro_promoted_serving_mixed_112req_wave16_r03 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| profile | profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | 0 | 16 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 129 |
| profile | profile_promoted_4096_128 | historical_4096_128_bs4 | 0 | 4 | 0 | 0 | 0 | 64 | 1178.8281 | 688.0000 | 65 |
| profile | profile_promoted_prefix_eviction_pressure_96req_wave16 | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| profile | profile_promoted_prefix_multi_112req_wave16 | prefix_multi_112req_wave16 | 96 | 16 | 49152 | 0 | 0 | 16 | 294.7070 | 172.0000 | 113 |
| profile | profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |
| verify | verify_promoted_prefix_eviction | prefix_eviction_pressure_96req_wave16 | 0 | 96 | 0 | 3 | 20480 | 112 | 2062.9492 | 1204.0000 | 17 |
| verify | verify_promoted_serving_mixed | serving_mixed_112req_wave16 | 0 | 112 | 0 | 0 | 0 | 14 | 257.8687 | 150.5000 | 115 |

## Memory / Capacity Ledger

# Memory And Capacity Ledger

| variant | scenario | peak alloc GiB | peak reserved GiB | KV GiB/rank | graph delta GiB | retained pages | retained tokens | retained MiB | SWA MiB | C4 MiB | C128 MiB | available comp pages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| promoted_prefix | decode_ladder_bs16 | 41.4668 | 42.1797 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 129 |
| promoted_prefix | historical_4096_1024_bs4 | 44.3071 | 45.8359 | 2.3204 | 19.0137 | 114 | 29184 | 2099.7876 | 1225.5000 | 149.6250 | 4.4531 | 15 |
| promoted_prefix | historical_4096_128_bs4 | 44.3070 | 45.8359 | 2.3204 | 19.0137 | 112 | 28672 | 2062.9492 | 1204.0000 | 147.0000 | 4.3750 | 17 |
| promoted_prefix | prefix_eviction_pressure_96req_wave16 | 42.6890 | 43.6777 | 2.3204 | 19.0137 | 112 | 28672 | 2062.9492 | 1204.0000 | 147.0000 | 4.3750 | 17 |
| promoted_prefix | prefix_multi_112req_wave16 | 42.8853 | 43.9180 | 2.3204 | 19.0137 | 16 | 4096 | 294.7070 | 172.0000 | 21.0000 | 0.6250 | 113 |
| promoted_prefix | serving_mixed_112req_wave16 | 41.5562 | 42.4648 | 2.3204 | 19.0137 | 14 | 3584 | 257.8687 | 150.5000 | 18.3750 | 0.5469 | 115 |
| target07_control | decode_ladder_bs16 | 41.4668 | 42.1543 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | historical_4096_1024_bs4 | 44.3041 | 46.4570 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | historical_4096_128_bs4 | 44.3040 | 46.4570 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | prefix_eviction_pressure_96req_wave16 | 42.6885 | 43.9805 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | prefix_multi_112req_wave16 | 42.8850 | 44.2656 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| target07_control | serving_mixed_112req_wave16 | 41.5561 | 42.4453 | 2.3204 | 19.0137 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |

## Decode Prepare vs Decode Forward

# Decode Prepare vs Forward

| variant | scenario | decode prepare s | decode forward s | prepare share | prefill forward s | prefill prepare s |
| --- | --- | --- | --- | --- | --- | --- |
| promoted_prefix | decode_ladder_bs16 | 0.1577 | 1.3665 | 0.1034 | 1.3516 | 0.0977 |
| promoted_prefix | historical_4096_1024_bs4 | 2.2245 | 22.2977 | 0.0907 | 4.2998 | 0.7325 |
| promoted_prefix | historical_4096_128_bs4 | 0.2814 | 2.7684 | 0.0923 | 4.2998 | 0.7266 |
| promoted_prefix | prefix_eviction_pressure_96req_wave16 | 0.1464 | 0.1714 | 0.4606 | 10.8178 | 2.2478 |
| promoted_prefix | prefix_multi_112req_wave16 | 0.2779 | 1.4750 | 0.1585 | 4.5400 | 0.8640 |
| promoted_prefix | serving_mixed_112req_wave16 | 1.0919 | 9.7624 | 0.1006 | 4.7832 | 0.6850 |
| target07_control | decode_ladder_bs16 | 0.1295 | 1.3297 | 0.0888 | 1.3874 | 0.0712 |
| target07_control | historical_4096_1024_bs4 | 1.9793 | 22.0498 | 0.0824 | 4.2955 | 0.7034 |
| target07_control | historical_4096_128_bs4 | 0.2525 | 2.7258 | 0.0848 | 4.2913 | 0.7184 |
| target07_control | prefix_eviction_pressure_96req_wave16 | 0.0175 | 0.1453 | 0.1075 | 10.4995 | 2.0154 |
| target07_control | prefix_multi_112req_wave16 | 0.1324 | 1.1799 | 0.1009 | 13.5254 | 2.5801 |
| target07_control | serving_mixed_112req_wave16 | 0.9130 | 9.2552 | 0.0898 | 4.7665 | 0.5424 |

## Owner Timing / Attribution

Owner timing is attribution only and is not used as final throughput evidence.

# Owner Timing

| run | variant | scenario | section | label | max-rank ms | count |
| --- | --- | --- | --- | --- | --- | --- |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3351.6042 | 5848 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3336.2684 | 5504 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3200.1859 | 5848 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3128.1046 | 5848 |
| profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3004.8750 | 3784 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2486.3387 | 3784 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 2220.1558 | 128 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 2172.3463 | 48 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | host | dsv4.prepare.decode.attention_metadata | 1976.1841 | 3528 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 1633.5723 | 5848 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 1567.6785 | 5848 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 1498.2598 | 5504 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1492.4249 | 136 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1483.0363 | 88 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1469.3135 | 136 |
| profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1374.6500 | 88 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | host | dsv4.prepare.decode.attention_metadata | 1336.5405 | 3528 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.owner.comm.dsv4.embedding_all_reduce | 1314.0829 | 136 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 1154.5177 | 48 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 1111.9479 | 56 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 909.4580 | 3584 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 889.3899 | 5504 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 876.8931 | 56 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 858.6540 | 5848 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 855.3292 | 96 |
| profile_control_serving_mixed | target07_control | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 846.0992 | 3584 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 810.6878 | 3784 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | host | dsv4.prepare.prefill.attention_metadata | 787.2578 | 8 |
| profile_promoted_decode_ladder_bs16 | promoted_prefix | decode_ladder_bs16 | cuda | dsv4.owner.moe.reduce_once_all_reduce | 728.5740 | 3784 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | host | dsv4.prepare.prefill.attention_metadata | 700.4669 | 56 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | promoted_prefix | prefix_eviction_pressure_96req_wave16 | cuda | dsv4.owner.shared_down.bf16_cache_local_total | 509.0395 | 5504 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 463.4095 | 5848 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 458.1059 | 5848 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.metadata.decode.make_c128_indices | 457.3876 | 1024 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 432.0505 | 3584 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 416.4905 | 56 |
| profile_promoted_4096_128 | promoted_prefix | historical_4096_128_bs4 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 395.2982 | 8 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.metadata.decode.make_c4_sparse_indices | 381.3584 | 56 |
| profile_promoted_prefix_multi_112req_wave16 | promoted_prefix | prefix_multi_112req_wave16 | cuda | dsv4.metadata.decode.make_c128_indices | 361.2777 | 448 |
| profile_promoted_serving_mixed_112req_wave16 | promoted_prefix | serving_mixed_112req_wave16 | cuda | dsv4.metadata.decode.make_component_page_tables | 356.1025 | 3584 |

## Component Row Counters

| run | scenario | label | count |
| --- | --- | --- | --- |
| profile_promoted_4096_128 | historical_4096_128_bs4 | dsv4.component_page_table_cache.rows/decode/dirty | 4 |
| profile_promoted_4096_128 | historical_4096_128_bs4 | dsv4.component_page_table_cache.rows/decode/clean | 504 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 16 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 224 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 64 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 32 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 32 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_decode_ladder_bs16 | decode_ladder_bs16 | dsv4.component_page_table_cache.rows/decode/clean | 16 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | prefix_eviction_pressure_96req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 96 |
| profile_promoted_prefix_eviction_pressure_96req_wave16 | prefix_eviction_pressure_96req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 0 |
| profile_promoted_prefix_multi_112req_wave16 | prefix_multi_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 112 |
| profile_promoted_prefix_multi_112req_wave16 | prefix_multi_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 672 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 112 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 1568 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 448 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 224 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 224 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/dirty | 0 |
| profile_promoted_serving_mixed_112req_wave16 | serving_mixed_112req_wave16 | dsv4.component_page_table_cache.rows/decode/clean | 112 |

## Ranked Bottleneck Table

# Ranked Bottlenecks

| rank | bucket | evidence | seconds/ms | share | interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | decode forward | serving_mixed phase total | 9.7624 | 0.5697 | dominant remaining E2E bucket; owner timing points to comm/attention work inside it |
| 2 | communication / all-reduce owners | owner timing profile, attribution only | 6489.2434 | 0.3787 | wo_b row-parallel, MoE reduce-once, and embedding all-reduce are top owner labels |
| 3 | prefill forward / TTFT base cost | serving_mixed phase total | 4.7832 | 0.2791 | not helped unless workload has real prefix hits |
| 4 | decode prepare / prefix metadata runtime | serving_mixed phase total plus owner timing | 1.0919 | 0.0637 | post-lifetime-cache tax; compare against 08.28 and owner rows |
| 5 | component page-table lifetime cache owner | owner timing profile, attribution only | 356.1025 | 0.0208 | metadata owner is now small relative to decode forward |

## Comparison To TARGET 07.79, TARGET 08.28, And vLLM

# Historical Comparison

| source | variant | scenario | out tok/s | decode prep s | decode fwd s | graph | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TARGET 07.79 non-prefix | target07_control | historical_4096_1024_bs4 | 131.7561 |  |  | 0 eager | from prompts/target.md |
| TARGET 07.79 non-prefix | target07_control | historical_4096_128_bs4 | 62.3925 |  |  | 0 eager | from prompts/target.md |
| old vLLM baseline | vLLM old serving line | 4096_1024_bs4_serving_line | 114.0700 |  |  |  | historical old serving victory line |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | serving_mixed_112req_wave16 | 163.7220 | 1.1359 | 9.8927 | 441/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | prefix_multi_112req_wave16 | 105.4163 | 0.2868 | 1.9164 | 49/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | prefix_eviction_pressure_96req_wave16 | 13.0260 | 0.1537 | 0.1917 | 6/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.28 promoted gate | route_b_lifetime_legacy_name | decode_ladder_bs16 | 98.3116 | 0.1639 | 1.6786 | 63/0 | from target08_route_b_lifetime_cache_promotion_gate |
| TARGET 08.30 current | promoted_prefix | decode_ladder_bs16 | 130.7605 | 0.1577 | 1.3665 | 63/0 | CV=0.0011 |
| TARGET 08.30 current | promoted_prefix | historical_4096_1024_bs4 | 137.1625 | 2.2245 | 22.2977 | 1023/0 | CV=0.0014 |
| TARGET 08.30 current | promoted_prefix | historical_4096_128_bs4 | 62.8933 | 0.2814 | 2.7684 | 127/0 | CV=0.0022 |
| TARGET 08.30 current | promoted_prefix | prefix_eviction_pressure_96req_wave16 | 13.0827 | 0.1464 | 0.1714 | 6/0 | CV=0.0019 |
| TARGET 08.30 current | promoted_prefix | prefix_multi_112req_wave16 | 110.1417 | 0.2779 | 1.4750 | 49/0 | CV=0.0033 |
| TARGET 08.30 current | promoted_prefix | serving_mixed_112req_wave16 | 163.3985 | 1.0919 | 9.7624 | 441/0 | CV=0.0035 |
| TARGET 08.30 current | target07_control | decode_ladder_bs16 | 135.5353 | 0.1295 | 1.3297 | 63/0 | CV=0.0222 |
| TARGET 08.30 current | target07_control | historical_4096_1024_bs4 | 139.8415 | 1.9793 | 22.0498 | 1023/0 | CV=0.0008 |
| TARGET 08.30 current | target07_control | historical_4096_128_bs4 | 63.7732 | 0.2525 | 2.7258 | 127/0 | CV=0.0015 |
| TARGET 08.30 current | target07_control | prefix_eviction_pressure_96req_wave16 | 15.0270 | 0.0175 | 0.1453 | 6/0 | CV=0.0041 |
| TARGET 08.30 current | target07_control | prefix_multi_112req_wave16 | 51.0507 | 0.1324 | 1.1799 | 49/0 | CV=0.0031 |
| TARGET 08.30 current | target07_control | serving_mixed_112req_wave16 | 178.3004 | 0.9130 | 9.2552 | 441/0 | CV=0.0010 |

## Required Questions

1. Promoted Route B lifetime prefix path improves serving TTFT/prefill? yes for shared-prefix workloads: prefix_multi saved 49152 prefill tokens; serving_mixed has no designed prefix hits, so its TTFT reflects the base serving path rather than a hit win.
2. `[1,2,4,8,16]` graph bucket still zero eager? yes.
3. SWA-tail/full-tail guard or memory retention is capacity bottleneck? not an OOM/capacity stopper in fixed --num-pages 128 runs; eviction pressure completed with 3 evictions and 112 retained pages.
4. New main bottleneck classification: decode-forward dominated, with isolated owner timing pointing most strongly to communication/all-reduce owners; prefix metadata/runtime is now secondary, and low precision/cache format remains a later candidate if the TARGET10 timeline disproves communication upside.
5. Next target recommendation: `TARGET 10`.

## Next-Target Recommendation

Status: `complete`.

Recommendation: `TARGET 10`.

Reason: decode forward dominates and isolated owner timing now points to communication/all-reduce owners; start with a narrow TARGET10 timeline before changing kernels or precision.
