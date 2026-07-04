# TARGET 08.21.4 DSV4 Route B Graph/Deforest/Serving Integration

Date: 2026-07-04

## Result

Decision: **Route B is graph-capable as a preferred opt-in candidate; prepare a
final prefix promotion gate rather than promoting by default in this target.**

Route B graph replay was restored for the selected serving buckets
`[1, 2, 4, 8, 16]` on the promoted `dsv4_sm80_a100_victory` path:

- captured buckets: `[16, 8, 4, 2, 1]`;
- text-smoke replay count: `5`;
- text-smoke eager decode count: `0`;
- compact prefix-full-hit perf replay count: `6`;
- compact prefix-full-hit perf eager decode count: `0`.

Decode metadata deforest remains guarded off under
`--enable-dsv4-component-loc-ownership`.  The current Triton deforest helper
still builds C4/C128 indices from the full-token page table and divides
`c4_full // 4` / `c128_full // 128`, which is unsafe for Route B tombstoned
full/SWA head pages.  The measured compact Route B graph run keeps graph replay
and pays only normal eager metadata construction/copy outside the graph.

The remaining SWA-tail guard is material only at exact page-multiple prompt
lengths.  For the target `page_size=256`, `prompt_len=256` has no extra Route B
loss because phase-1 already has no page-aligned hit for `input_len - 1 = 255`;
the exact-page-multiple losses are `512 -> 0` instead of `256`, and `768 -> 0`
instead of `512`.

## Artifacts

```text
performance_milestones/target08_route_b_graph_deforest_serving/
  README.md
  raw/
    text_smoke_route_b_graph.*
    text_smoke_route_b_victory_graph.*
    perf_prefix_off/
    perf_phase1_prefix_on/
    perf_route_b_graph/
    swa_tail_guard_quantification.json
  scripts/
    quantify_swa_tail_guard.py
    summarize_results.py
  summaries/
    performance_ab.csv
    performance_ab.md
    summary.json
    swa_tail_guard_table.csv
    swa_tail_guard_table.md
```

## Exact Commands

Validation:

```bash
python -m py_compile \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/engine/graph.py \
  python/minisgl/engine/engine.py \
  python/minisgl/server/args.py \
  tests/attention/test_deepseek_v4_backend_metadata.py

pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -q

ruff check \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/engine/graph.py \
  python/minisgl/engine/engine.py \
  python/minisgl/server/args.py \
  tests/attention/test_deepseek_v4_backend_metadata.py

pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/engine/test_graph_runner.py \
  tests/attention/test_deepseek_v4_backend_metadata.py

pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_cache_allocate.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_scheduler.py \
  tests/engine/test_graph_runner.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py

git diff --check
```

Route B graph text smoke:

```bash
timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 64 \
  --max-seq-len 512 \
  --max-extend-tokens 512 \
  --max-tokens 8 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_route_b_graph_deforest_serving/raw/text_smoke_route_b_victory_graph.json \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
  --prompt 'Answer in one short English sentence: what color is the sky on a clear day?' \
  > performance_milestones/target08_route_b_graph_deforest_serving/raw/text_smoke_route_b_victory_graph.log 2>&1
```

Compact performance A/B:

```bash
timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios prefix_full_hit_257_bs4 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 512 \
  --max-extend-tokens 512 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --repeats 1 \
  --warmup-repeats 0 \
  --output-dir performance_milestones/target08_route_b_graph_deforest_serving/raw/perf_prefix_off \
  --keep-going

timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios prefix_full_hit_257_bs4 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 512 \
  --max-extend-tokens 512 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --repeats 1 \
  --warmup-repeats 0 \
  --output-dir performance_milestones/target08_route_b_graph_deforest_serving/raw/perf_phase1_prefix_on \
  --keep-going

timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios prefix_full_hit_257_bs4 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 512 \
  --max-extend-tokens 512 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --repeats 1 \
  --warmup-repeats 0 \
  --output-dir performance_milestones/target08_route_b_graph_deforest_serving/raw/perf_route_b_graph \
  --keep-going
```

Milestone summaries:

```bash
python performance_milestones/target08_route_b_graph_deforest_serving/scripts/quantify_swa_tail_guard.py
python performance_milestones/target08_route_b_graph_deforest_serving/scripts/summarize_results.py

ruff check \
  performance_milestones/target08_route_b_graph_deforest_serving/scripts/quantify_swa_tail_guard.py \
  performance_milestones/target08_route_b_graph_deforest_serving/scripts/summarize_results.py
```

## Git Status Summary

This workspace already contains the B0/B1/B2 Route B changes and milestone
directories.  The B3-specific code changes are:

```text
M benchmark/offline/deepseek_v4_perf_matrix.py
M benchmark/offline/deepseek_v4_text_smoke.py
M python/minisgl/attention/deepseek_v4.py
M python/minisgl/engine/engine.py
M python/minisgl/engine/graph.py
M python/minisgl/server/args.py
M tests/attention/test_deepseek_v4_backend_metadata.py
?? performance_milestones/target08_route_b_graph_deforest_serving/
```

The broader dirty tree still includes B1/B2 files:

```text
M python/minisgl/kvcache/__init__.py
M python/minisgl/kvcache/deepseek_v4_pool.py
M python/minisgl/kvcache/radix_cache.py
M python/minisgl/scheduler/cache.py
M python/minisgl/scheduler/config.py
M python/minisgl/scheduler/scheduler.py
M tests/benchmark/test_deepseek_v4_perf_matrix.py
M tests/benchmark/test_deepseek_v4_text_smoke.py
M tests/core/test_deepseek_v4_kvcache.py
M tests/core/test_dsv4_cache_option_guards.py
?? performance_milestones/target08_component_loc_table_preflight/
?? performance_milestones/target08_independent_compressed_indexer_ownership/
?? performance_milestones/target08_compression_state_ownership/
```

## Graph Metadata Field Map

| field | phase-1 graph behavior | Route B graph behavior |
| --- | --- | --- |
| `raw_out_loc` | graph input buffer, copied by `GraphCaptureBuffer` | unchanged; remains full-token write loc |
| `positions` | graph input buffer, copied by `GraphCaptureBuffer` | unchanged |
| `page_table` | copied by replay metadata helper/fallback | still full-token table; may contain tombstones for retained heads |
| `swa_page_indices` | copied by replay metadata helper/fallback | copied from eager source; SWA still full-owned and protected by live-tail guard |
| `c4_sparse_page_indices` | phase-1 component locs derived from full locs | copied from eager source where Route B already gathered direct C4 component locs |
| `c128_page_indices` | phase-1 component locs derived from full locs | copied from eager source where Route B already gathered direct C128 component locs |
| `c4_page_table` | absent | allocated in capture metadata and copied for Route B |
| `c128_page_table` | absent | allocated in capture metadata and copied for Route B |
| `c4_indexer_page_table` | absent; indexer used full page table | allocated in capture metadata, copied, and used by `DSV4IndexerMetadata.page_table` |
| `c4_out_loc` | per-row masked locs from `raw_out_loc // 4` | compact direct C4 locs are scattered into per-row graph buffer |
| `c128_out_loc` | per-row masked locs from `raw_out_loc // 128` | compact direct C128 locs are scattered into per-row graph buffer |
| `c4_indexer_out_loc` | aliases `c4_out_loc` | separate direct C4-indexer loc graph buffer |

The in-graph helper `copy_masked_compressed_locs()` remains available for
phase-1, but is guarded under Route B because it only knows
`raw_out_loc`, `positions`, and `full_loc // ratio`.

## Deforest Decision

`decode_metadata_deforest_fallback()` stays guarded off when
`component_loc_ownership=True`.

Unsafe assumptions still present in the Triton deforest path:

- `_build_decode_metadata_indices_kernel` loads endpoint full-token locs from
  `ctx_page_table`;
- C4 sparse page indices are stored as `c4_full // 4`;
- C128 page indices are stored as `c128_full // 128`;
- no inputs exist for `c4_page_table`, `c128_page_table`, or
  `c4_indexer_page_table`.

This target did not port deforest.  The compact phase-1 versus Route B graph
run shows Route B graph replay remains usable without deforest.

## Correctness And Graph Results

| check | result | evidence |
| --- | --- | --- |
| direct component table graph replay copy | pass | `test_dsv4_component_loc_ownership_capture_replay_copies_direct_component_metadata` |
| Route B graph compressed-loc hook guard | pass | `test_dsv4_component_loc_ownership_capture_locs_graph_hook_is_guarded` |
| B1/B2 ownership, eviction, repeated hit/evict | pass | `tests/core/test_deepseek_v4_kvcache.py` and B1/B2 probes |
| graph exact-bs guard unit coverage | pass | `tests/engine/test_graph_runner.py` |
| Route B graph text smoke | pass | `raw/text_smoke_route_b_victory_graph.json` |
| graph replay buckets `[1,2,4,8,16]` | pass | captured `[16,8,4,2,1]` |
| eager fallback on graph-capable path | pass | eager decode count `0` |
| fallback variant graph attempt | expected fail-open | fallback kernels use dynamic Python/Torch operations during capture |

Text outputs from the graph-capable smoke:

| prompt style | output |
| --- | --- |
| Chinese city | `杭州西湖位于杭州市。` |
| Chinese province | `浙江省。` |
| English color | `Blue.` |

## Performance A/B

Compact one-repeat workload:
`prefix_full_hit_257_bs4`, TP8, page size `256`, graph buckets `[1,2,4,8,16]`.

```text
summaries/performance_ab.md
```

| mode | TTFT s | TPOT s | prefill tok/s | decode tok/s | output tok/s | graph replay/eager | hits/saved | live full | C4/C128/indexer | state C4/C128/indexer |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| prefix_off | 2.8736 | 0.1968 | 197.70 | 93.13 | 2.96 | 6/0 | 0/0 | 0 | 0/0/0 | 0/0/0 |
| phase1_prefix_on | 0.9904 | 0.0539 | 120.71 | 101.90 | 6.84 | 6/0 | 3/768 | 0 | 0/0/0 | 0/0/0 |
| route_b_graph | 1.0063 | 0.0655 | 120.04 | 99.54 | 6.60 | 6/0 | 3/768 | 1 | 64/2/64 | 8/128/8 |

Interpretation:

- prefix caching still gives the expected TTFT/output-throughput win versus
  prefix off;
- Route B preserves graph replay and the same hit/saved-prefill count as
  phase-1 on this workload;
- Route B is slightly slower than phase-1 in this one-repeat compact run
  (`6.60` versus `6.84` output tok/s), consistent with direct metadata copy and
  no deforest.  This is not large enough by itself to block the opt-in, but the
  final gate should rerun the full TARGET 08.10 serving suite.

## Capacity And SWA-Tail Guard

For the 257-token full-hit workload, Route B retained:

| retained logical pages | live full pages | C4 slots | C128 slots | indexer slots | C4 state | C128 state | indexer state |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 64 | 2 | 64 | 8 | 128 | 8 |

For longer retained prefixes, Route B keeps one live full/SWA tail page and
component/state pages for the retained component range.  The SWA-tail guard
quantification is in:

```text
raw/swa_tail_guard_quantification.json
summaries/swa_tail_guard_table.md
```

Important rows for the target `page_size=256`:

| prompt len | phase-1 hit | Route B hit | shortened |
| ---: | ---: | ---: | ---: |
| 256 | 0 | 0 | 0 |
| 257 | 256 | 256 | 0 |
| 512 | 256 | 0 | 256 |
| 513 | 512 | 512 | 0 |
| 768 | 512 | 0 | 512 |
| 769 | 768 | 768 | 0 |

So the real serving risk is exact page-multiple prompt lengths after a longer
prefix has been cached.  The compact workload at 257 tokens is unaffected; a
final serving gate should report prompt-length bucket counts and exact-multiple
frequency before promotion.  If exact-multiple traffic is common, open a
follow-up for SGLang-aligned independent SWA ownership rather than trying to
force promotion here.

## Remaining Gaps

- Full guarded slot-pinned logits/activation oracle was not rerun end-to-end in
  this B3 pass; prior B1/B2 CPU ownership probes and TP8 text smokes remain the
  main correctness evidence.
- The full TARGET 08.10 serving suite (`prefix_multi_112req_wave16`,
  eviction pressure, mixed hit/miss, serving distribution) was not rerun here.
  Only a compact A/B was run to verify graph replay and basic perf shape.
- Decode metadata deforest is not ported to Route B.
- SWA KV remains full-token-owned, so the live full/SWA tail guard stays.

## Final Recommendation

Keep `--enable-dsv4-component-loc-ownership` explicit, but treat Route B as the
preferred prefix-cache opt-in candidate for the next final prefix promotion
gate.  The next gate should run the full serving suite, include exact
page-multiple prompt-length distribution, and decide whether independent SWA
ownership is needed before any default promotion.
