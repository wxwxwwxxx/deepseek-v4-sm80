# TARGET 08.10 Prefix Cache Serving Stability

## Recommendation

Keep DSV4 radix prefix cache as a controlled opt-in feature only.

The serving stability gate is healthy for the opt-in path: full hit, partial hit,
miss, mixed hit/miss, sustained shared-prefix traffic, and eviction pressure all
completed without runtime failures.  CUDA graph replay coverage stayed intact:
the prefix-on matrix recorded `89/0` graph replay/eager decode events, and the
long text smoke recorded `2/0`.

Do not promote it to default behavior yet.  The random-token serving matrix found
prefix-off vs prefix-on generated token id mismatches in the hit workloads.  A
long natural-language repeated-prefix smoke generated matching `OK` output and
recorded a real prefix hit, but the synthetic exactness gap is enough to block a
stronger promotion.

Final decision for this target: **controlled opt-in only; no default promotion.
Revisit after 08.18 and a focused generated-token/logit correctness follow-up.**

## Scope

Promoted exact path:

- Variant: `dsv4_sm80_a100_victory`
- Env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- Page size: `256`
- Fixed pages: `--num-pages 128`
- CUDA graph buckets: `[1,2,4,8,16]`
- Prefix cache opt-in: `--enable-dsv4-radix-prefix-cache`
- Main workload max sequence length: `1280`
- Main workload max running requests: `16`
- Main workload max extend tokens: `20000`

Inherited TARGET 08 conclusions:

- TARGET 08.05 selected `[1,2,4,8,16]` as the current serving graph bucket
  policy.
- TARGET 08.06 measured graph private-pool capture cost at about
  `19.04 GiB/rank`.
- TARGET 08.07 measured promoted BF16 cache persistent baseline at about
  `1.588 GiB/rank`.
- TARGET 08.07 ruled out BF16 caches as the main cause of the `~19 GiB` graph
  delta.

## Artifacts

| Path | Contents |
| --- | --- |
| `raw/prefix_off_control/` | Prefix disabled serving matrix reports and torchrun log. |
| `raw/prefix_on_opt_in/` | Prefix enabled serving matrix reports and torchrun log. |
| `raw/prefix_off_repeat_full_hit/` | Repeated prefix-off full-hit scenario used to check synthetic determinism. |
| `raw/text_smoke_long_prefix_off.json` | Long natural-language repeated-prefix smoke with prefix disabled. |
| `raw/text_smoke_long_prefix_on.json` | Long natural-language repeated-prefix smoke with prefix enabled. |
| `scripts/run_prefix_cache_serving_stability.sh` | Repro script for the main off/on serving matrix. |
| `scripts/summarize_prefix_cache_serving_stability.py` | Summary generator for tables and decision inputs. |
| `summaries/prefix_cache_serving_stability_summary.json` | Machine-readable summarized results. |
| `summaries/prefix_cache_serving_stability_summary.md` | Compact markdown summary. |

## Exact Commands

Main matrix:

```bash
cd /workspace/mini-sglang
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
performance_milestones/target08_prefix_cache_serving_stability/scripts/run_prefix_cache_serving_stability.sh
```

The script expands the prefix-disabled control to:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios \
    prefix_full_hit_257_bs4 \
    prefix_partial_hit_769_bs8 \
    prefix_mixed_hit_miss_bs16 \
    prefix_multi_112req_wave16 \
    prefix_eviction_pressure_96req_wave16 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1280 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --repeats 1 \
  --warmup-repeats 0 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir performance_milestones/target08_prefix_cache_serving_stability/raw/prefix_off_control \
  --keep-going
```

The prefix-enabled opt-in command is identical except for:

```bash
  --output-dir performance_milestones/target08_prefix_cache_serving_stability/raw/prefix_on_opt_in \
  --keep-going \
  --enable-dsv4-radix-prefix-cache
```

Long natural-language repeated-prefix smoke:

```bash
PROMPT="$(printf 'Please answer exactly OK. %.0s' {1..180})"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 2048 \
  --max-extend-tokens 2048 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_prefix_cache_serving_stability/raw/text_smoke_long_prefix_off.json \
  --prompt "$PROMPT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 2048 \
  --max-extend-tokens 2048 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --output performance_milestones/target08_prefix_cache_serving_stability/raw/text_smoke_long_prefix_on.json \
  --prompt "$PROMPT"
```

Summary:

```bash
python performance_milestones/target08_prefix_cache_serving_stability/scripts/summarize_prefix_cache_serving_stability.py \
  --milestone-dir performance_milestones/target08_prefix_cache_serving_stability
```

Validation:

```bash
python -m py_compile \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  performance_milestones/target08_prefix_cache_serving_stability/scripts/summarize_prefix_cache_serving_stability.py

pytest -q \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_target0810_prefix_scenarios_have_stable_prompt_shapes \
  tests/core/test_deepseek_v4_kvcache.py::test_deepseek_v4_radix_prefix_cache_tracks_full_partial_miss_and_components \
  tests/core/test_deepseek_v4_kvcache.py::test_deepseek_v4_radix_prefix_swa_window_128_boundary_is_page_safe \
  tests/core/test_deepseek_v4_kvcache.py::test_deepseek_v4_radix_prefix_repeated_hit_evict_cycle_has_no_leak
```

Validation result: `4 passed`; `py_compile` passed.

## Correctness Table

Unit coverage includes full/partial/miss accounting, retained DSV4 component
slots, repeated hit/evict leak checks, and the SWA boundary around `128` with
`page_size=256`.

| Check | Result | Notes |
| --- | --- | --- |
| Unit: prefix metrics and component retention | pass | Covers full hit, partial hit, miss, retained full/C4/C128/indexer component slots. |
| Unit: SWA boundary around 128 | pass | Confirms `page_size=256` is C128-aligned and page-safe at the SWA boundary. |
| Unit: repeated hit/evict cycle | pass | No retained-page leak after repeated hit/evict cycles. |
| Long text off/on output | pass | Both modes generated token `[11932]`, parsed text `OK`. |
| Long text prefix hit | pass | Prefix-on recorded `1` full hit, `1` miss, `768` saved prefill tokens. |
| Synthetic generated-token exactness | fail | Prefix-on and prefix-off token ids diverged in hit workloads; see table below. |

Synthetic serving matrix generated-token comparison:

| Scenario | off/on status | Outputs match | Checked requests | Prefix-on full | Prefix-on partial | Prefix-on miss | Prefix-on evict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prefix_full_hit_257_bs4` | pass/pass | no | 4 | 3 | 0 | 1 | 0 |
| `prefix_partial_hit_769_bs8` | pass/pass | no | 8 | 0 | 7 | 1 | 0 |
| `prefix_mixed_hit_miss_bs16` | pass/pass | no | 16 | 8 | 0 | 8 | 0 |
| `prefix_multi_112req_wave16` | pass/pass | no | 112 | 96 | 0 | 16 | 0 |
| `prefix_eviction_pressure_96req_wave16` | pass/pass | yes | 96 | 0 | 0 | 96 | 5 |

The synthetic mismatch is reproducible for the full-hit workload.  A repeated
prefix-off control matched the original prefix-off output, while prefix-on
differed.  The same prefix-off batch also produced different continuations for
identical prompts in different slots, so this is not a clean semantic oracle;
nevertheless it blocks default promotion.

## Serving Workload Table

| Mode | Scenario | Hit rate | Saved prefill tokens | TTFT mean s | Prefill forward s | Decode forward s | Output tok/s | Replay/eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix_off | `prefix_full_hit_257_bs4` | 0.000 | 0 | 2.569 | 5.273 | 0.135 | 2.898 | 6/0 |
| prefix_on | `prefix_full_hit_257_bs4` | 0.750 | 768 | 1.628 | 3.526 | 0.135 | 4.257 | 6/0 |
| prefix_off | `prefix_partial_hit_769_bs8` | 0.000 | 0 | 3.654 | 3.889 | 0.309 | 13.590 | 14/0 |
| prefix_on | `prefix_partial_hit_769_bs8` | 0.875 | 1792 | 3.592 | 4.113 | 0.306 | 13.819 | 14/0 |
| prefix_off | `prefix_mixed_hit_miss_bs16` | 0.000 | 0 | 4.632 | 6.424 | 0.316 | 17.589 | 14/0 |
| prefix_on | `prefix_mixed_hit_miss_bs16` | 0.500 | 6144 | 2.931 | 3.141 | 0.312 | 34.108 | 14/0 |
| prefix_off | `prefix_multi_112req_wave16` | 0.000 | 0 | 2.460 | 14.746 | 1.315 | 47.769 | 49/0 |
| prefix_on | `prefix_multi_112req_wave16` | 0.857 | 49152 | 0.976 | 6.194 | 1.309 | 107.009 | 49/0 |
| prefix_off | `prefix_eviction_pressure_96req_wave16` | 0.000 | 0 | 2.304 | 11.947 | 0.162 | 13.640 | 6/0 |
| prefix_on | `prefix_eviction_pressure_96req_wave16` | 0.000 | 0 | 1.901 | 9.508 | 0.164 | 16.467 | 6/0 |

The requested workload shapes are covered:

- disabled control vs enabled opt-in;
- full hit, partial hit, miss, repeated hit, eviction pressure;
- shared-prefix, mixed hit/miss, and multi-prefix sustained workload;
- SWA boundary around `128` and full/C4/C128/indexer retention checks;
- graph replay coverage with no prefix-cache induced eager decode.

## Eviction Pressure Result

`prefix_eviction_pressure_96req_wave16` used 96 requests in waves of 16 with
unique full-page prefixes.  Under `--num-pages 128`, prefix-on retained
`112` prefix pages and evicted `34,816` tokens across `5` eviction events.

| Metric | Prefix off | Prefix on |
| --- | ---: | ---: |
| Status | pass | pass |
| Hit rate | 0.000 | 0.000 |
| Saved prefill tokens | 0 | 0 |
| Evictions | 0 | 5 |
| Evicted tokens | 0 | 34,816 |
| Retained prefix pages | 0 | 112 |
| Retained prefix tokens | 0 | 28,672 |
| Retained memory | 0.000 GiB | 2.015 GiB |
| Graph replay/eager | 6/0 | 6/0 |

This pressure case validates eviction accounting and bounded retention with the
requested `--num-pages 128` cap.

## Memory Retention Table

| Prefix-on scenario | Retained pages | Retained tokens | Retained GiB | Full slots | C4 slots | C128 slots | Indexer slots | Evicted tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefix_full_hit_257_bs4` | 1 | 256 | 0.018 | 256 | 64 | 2 | 64 | 0 |
| `prefix_partial_hit_769_bs8` | 16 | 4,096 | 0.288 | 4,096 | 1,024 | 32 | 1,024 | 0 |
| `prefix_mixed_hit_miss_bs16` | 40 | 10,240 | 0.719 | 10,240 | 2,560 | 80 | 2,560 | 0 |
| `prefix_multi_112req_wave16` | 56 | 14,336 | 1.007 | 14,336 | 3,584 | 112 | 3,584 | 0 |
| `prefix_eviction_pressure_96req_wave16` | 112 | 28,672 | 2.015 | 28,672 | 7,168 | 224 | 7,168 | 34,816 |
| Long text smoke | 3 | 768 | 0.054 | 768 | 192 | 6 | 192 | 0 |

Component ratios match `page_size=256`: C4 slots are `tokens / 4`, C128 slots
are `tokens / 128`, and indexer slots track C4 indexing.

## Graph Replay And Memory

| Check | Value |
| --- | ---: |
| Prefix-on matrix total graph replay | 89 |
| Prefix-on matrix total eager decode | 0 |
| Long text smoke graph replay/eager | 2/0 |
| Prefix-on total saved prefill tokens | 57,856 |
| Max retained prefix pages | 112 |
| Max retained prefix memory | 2.015 GiB |
| Long text peak allocated memory | 41.315 GiB/rank |

Graph replay remained covered for every measured serving scenario.  No eager
decode count was introduced by enabling prefix cache.

## Promotion Decision

**Controlled opt-in: yes.**  The opt-in path is stable enough to keep behind
`--enable-dsv4-radix-prefix-cache` for targeted serving experiments.

**Default promotion: no.**  The synthetic generated-token mismatches mean this
target cannot claim a broad correctness promotion.

**After 08.18:** Revisit with deterministic logits/token checks and capacity
budgeting that includes the known `~19.04 GiB/rank` graph private-pool cost,
the `1.588 GiB/rank` BF16 cache baseline, and the measured prefix-retention
headroom.

This target intentionally did not implement SGLang-style independent
SWA/component retention, low-precision experiments, attention kernel
optimization, PyNCCL changes, or graph/workspace restructuring.
