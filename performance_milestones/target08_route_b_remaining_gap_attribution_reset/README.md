# TARGET 08.26 DSV4 Route B Remaining Gap Attribution Reset

Date: 2026-07-04

## Result

Decision: **open a focused stable-row / dirty-row metadata target next**.

The Route B direct C4 path is the best forward profiling baseline from the
08.24/08.25 family. It gives a small, repeatable unprofiled throughput win over
the Route B graph baseline, but its remaining gap to phase1 prefix-on is still
almost entirely decode prepare.

The top owner is not attention, MoE, communication, SWA-tail retention, or graph
replay coverage. It is repeated decode metadata work, especially component page
table construction for Route B. Per-request page tables, component page tables,
and per-prefix-hit C128 rows are rebuilt and replay-copied every decode step
(`441` decode replay steps in this workload), which is large enough to justify a
bounded stable-row / dirty-row follow-up.

Do **not** promote the 08.24/08.25 experimental paths as defaults from this
target. The new direct-C4 group selector is a diagnostic/profile variant.

## Exact Command

```bash
LARGE_REPEATS=3 RUN_THROUGHPUT=1 RUN_PROFILE=1 \
  bash performance_milestones/target08_route_b_remaining_gap_attribution_reset/scripts/run_remaining_gap_attribution_reset.sh
```

The script runs separate `torchrun` invocations for:

| label | variant / flags |
| --- | --- |
| phase1 prefix on | `dsv4_sm80_a100_victory --enable-dsv4-radix-prefix-cache` |
| Route B graph baseline | `dsv4_sm80_a100_victory --enable-dsv4-radix-prefix-cache --enable-dsv4-component-loc-ownership` |
| Route B direct C4 | `dsv4_sm80_a100_victory_directgraphmetadata_c4` plus Route B ownership |
| Route B direct SWA+C4+C128 | `dsv4_sm80_a100_victory_directgraphmetadata` plus Route B ownership |

Common serving args:

```text
--model-path /models/DeepSeek-V4-Flash
--scenarios serving_mixed_112req_wave16
--page-size 256 --num-pages 128
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
--keep-going
```

No nsys run was needed for this reset. Owner timing was used as a short profile
pass only, and is separated from throughput evidence.

## Git Status Summary

Captured in `raw/git_status_short.txt` after the run:

```text
 M benchmark/offline/deepseek_v4_perf_matrix.py
 M benchmark/offline/deepseek_v4_text_smoke.py
 M prompts/TARGET_08_radix_prefix_dsv4.md
 M python/minisgl/attention/deepseek_v4.py
 M python/minisgl/engine/engine.py
 M python/minisgl/kernel/deepseek_v4.py
 M python/minisgl/kernel/triton/deepseek_v4.py
 M python/minisgl/server/args.py
 M tests/benchmark/test_deepseek_v4_perf_matrix.py
 M tests/kernel/test_deepseek_v4_wrappers.py
?? performance_milestones/target08_route_b_direct_graph_metadata_buffers/
?? performance_milestones/target08_route_b_remaining_gap_attribution_reset/
?? prompts/TARGET_08.26_dsv4_sm80_route_b_remaining_gap_attribution_reset.md
```

## Throughput Evidence

Unprofiled `serving_mixed_112req_wave16` repeat runs:

| mode | runs | output tok/s mean | output tok/s stdev | decode tok/s mean | decode prepare s mean | decode forward s mean | graph replay/eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| phase1 prefix on | 3 | 169.7381 | 0.8408 | 269.4540 | 0.9403 | 9.9757 | 441/0 |
| Route B graph baseline | 3 | 136.2373 | 0.4446 | 266.4154 | 4.4798 | 10.0897 | 441/0 |
| Route B direct C4 | 3 | 138.1281 | 0.7047 | 265.3675 | 4.2067 | 10.1297 | 441/0 |
| Route B direct SWA+C4+C128 | 3 | 141.4511 | 1.2289 | 268.9379 | 3.8731 | 9.9964 | 441/0 |

Direct C4 is repeat-stable against the Route B graph baseline:

| repeat | Route B baseline output tok/s | Route B direct C4 output tok/s | delta |
| --- | ---: | ---: | ---: |
| r01 | 135.7968 | 137.3948 | +1.5980 |
| r02 | 136.6858 | 138.8002 | +2.1144 |
| r03 | 136.2293 | 138.1893 | +1.9600 |

The full direct SWA+C4+C128 control did **not** reproduce the 08.25 large-wave
throughput regression in this run. It remains a diagnostic control, not the main
line, because 08.25 already showed it can hurt output throughput and it changes
SWA/C128 surface area beyond the C4 path that has the cleanest local win.

## Prepare Versus Forward

| mode | output tok/s | decode tok/s | decode prepare s | decode forward s | prepare delta vs phase1 s | forward delta vs phase1 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 prefix on | 169.7381 | 269.4540 | 0.9403 | 9.9757 | 0.0000 | 0.0000 |
| Route B graph baseline | 136.2373 | 266.4154 | 4.4798 | 10.0897 | 3.5395 | 0.1140 |
| Route B direct C4 | 138.1281 | 265.3675 | 4.2067 | 10.1297 | 3.2664 | 0.1540 |
| Route B direct SWA+C4+C128 | 141.4511 | 268.9379 | 3.8731 | 9.9964 | 2.9328 | 0.0206 |

For the main path, Route B direct C4, the remaining phase1 gap is prepare:
`+3.2664s` decode prepare versus only `+0.1540s` decode forward.

## Profile Overhead

Owner timing was intentionally separate and reduces output tok/s by about
25-27%, so profile rows are owner-ranking evidence only:

| mode | unprofiled output tok/s mean | owner-timing output tok/s | delta |
| --- | ---: | ---: | ---: |
| phase1 prefix on | 169.7381 | 123.1169 | -27.47% |
| Route B graph baseline | 136.2373 | 102.2208 | -24.97% |
| Route B direct C4 | 138.1281 | 101.9809 | -26.17% |
| Route B direct SWA+C4+C128 | 141.4511 | 103.5551 | -26.79% |

## Decode Prepare Owners

Owner timing profile runs only; values are max-rank ms.

| mode | decode prepare s | host attention metadata ms | component tables ms | full page table ms | SWA idx ms | C4 idx ms | C128 idx ms | write locs ms | replay fused copy ms | replay comp tables ms | direct index ms | build bytes | copy bytes | direct bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 prefix on | 1.9935 | 1814.1543 | 5.5495 | 70.0731 | 90.1665 | 584.1073 | 326.8966 | 180.2352 | 16.7356 | 0 | 0.0000 | 19976768 | 19976768 | 0 |
| Route B graph baseline | 4.7805 | 4591.5918 | 3350.0037 | 69.4883 | 84.5014 | 691.7250 | 420.1730 | 245.8265 | 17.6241 | 104.0867 | 0.0000 | 20039936 | 20039936 | 0 |
| Route B direct C4 | 4.4941 | 4304.6365 | 3341.1692 | 69.8436 | 83.8588 | 409.5431 | 421.9036 | 245.3735 | 17.3471 | 103.2981 | 71.1383 | 3557120 | 3524864 | 16515072 |
| Route B direct SWA+C4+C128 | 4.1048 | 3914.8209 | 3351.5761 | 69.5198 | 1.5669 | 423.0593 | 112.0447 | 248.2373 | 16.8173 | 104.6296 | 66.9185 | 159488 | 84224 | 19955712 |

Interpretation:

- Component page-table construction is the dominant remaining prepare owner:
  direct C4 still spends `3341.1692 ms` there.
- Full `page_table` construction is small (`69.8436 ms` direct C4) and is not
  the main owner.
- Graph replay staging is visible but secondary: component page-table replay
  copies are about `103.2981 ms`, and fused replay copy is `17.3471 ms`.
- Direct C4 removes the intended C4 source/copy bytes: build bytes fall from
  about `20.04 MB` to `3.56 MB`, copy bytes from `20.04 MB` to `3.52 MB`, with
  `16.52 MB` generated directly into graph buffers.
- Host `attention_metadata` remains the prepare envelope. The data does not
  isolate a larger pure-Python owner than the component page-table work; the
  next probe should measure dirty/stable rows inside this metadata path.

## Decode Forward Owners

CUDA owner-timing labels grouped by forward compute/communication owner. Values
are max-rank ms; prepare-side metadata/replay labels are excluded here.

| mode | decode forward s | attention ms | indexer/compressor ms | MoE/shared ms | communication ms | other owner ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 prefix on | 15.0479 | 636.7226 | 119.4024 | 347.2494 | 4909.5034 | 0.0000 |
| Route B graph baseline | 14.6251 | 1142.9150 | 112.9800 | 1126.5050 | 5948.2426 | 0.0000 |
| Route B direct C4 | 14.8697 | 655.7333 | 112.5127 | 2122.0638 | 6263.7858 | 0.0000 |
| Route B direct SWA+C4+C128 | 14.4944 | 1490.7817 | 114.8328 | 1070.4746 | 6607.6030 | 0.0000 |

Forward owner timing is noisy under instrumentation and was not repeated. The
unprofiled phase totals are the stronger signal: direct C4 adds only `0.1540s`
of decode forward versus phase1 and `0.0400s` versus the Route B baseline.
Communication counters and bytes are identical across the compared unprofiled
runs (`4928` calls, `149319188480` bytes), and graph replay/eager stays `441/0`.
There is no clear direct-C4-specific attention, MoE, communication, indexer, or
graph replay forward owner that justifies TARGET 10 from this reset.

## Metadata Update Pressure

Direct C4 profile call counts show stable-candidate metadata updated every
decode replay step:

| field | class | build calls | replay-copy calls |
| --- | --- | ---: | ---: |
| `page_table` | per-request | 441 | 441 |
| `c4_page_table` | per-request | 441 | 441 |
| `c128_page_table` | per-request | 441 | 441 |
| `c4_indexer_page_table` | per-request | 441 | 441 |
| `c128_raw_indices` | per-prefix-hit | 441 | 441 |
| `c128_page_indices` | per-prefix-hit | 441 | 441 |
| `c128_full_indices` | per-prefix-hit | 441 | 441 |

This is enough pressure for a stable-row / dirty-row target, with narrow scope:
skip or reuse rows only when request-slot/page-table state is unchanged, and
measure avoided rows before any larger scheduler change.

## SWA Tail Guard

SWA-tail guard is not the current bottleneck. In this reset,
`serving_mixed_112req_wave16` has `prefix_saved_prefill_tokens=0`,
`prefix_evictions=0`, and identical graph replay/eager coverage across all four
variants. Prepare owner timing shows SWA index work (`83.8588 ms` direct C4) is
small relative to component page-table construction (`3341.1692 ms`).

Return to TARGET 08.23 only if future prefix-hit or eviction-pressure workloads
show SWA retention/capacity loss as a top owner.

## Next Target Recommendation

Recommended next step: a lightweight stable-row / dirty-row metadata target.

Suggested starting scope:

- component page tables: `c4_page_table`, `c128_page_table`,
  `c4_indexer_page_table`;
- full per-request `page_table`;
- per-prefix-hit C128 rows;
- owner counters for dirty rows, stable rows, avoided build bytes, avoided replay
  copy bytes, and output correctness/graph replay parity.

Stop that target if dirty-row reuse does not remove meaningful component
page-table work or if it requires a broad scheduler rewrite. Do not jump to
TARGET 10 or TARGET 08.23 from the present evidence. Move to TARGET 08.30 only
after this metadata-lifetime probe if no focused owner remains.

## Files

- Raw reports: `raw/`
- Run script: `scripts/run_remaining_gap_attribution_reset.sh`
- Summary script: `scripts/summarize_remaining_gap_attribution_reset.py`
- Main summaries:
  - `summaries/throughput_repeat.md`
  - `summaries/prepare_forward_attribution.md`
  - `summaries/prepare_owner_profile.md`
  - `summaries/decode_forward_owner_profile.md`
  - `summaries/metadata_update_pressure.md`
  - `summaries/profile_overhead.md`
  - `summaries/swa_tail_guard_recap.md`
