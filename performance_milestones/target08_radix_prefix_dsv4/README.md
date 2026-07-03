# TARGET 08 DSV4 Radix Prefix Cache

## Result

Status: complete as an explicit opt-in path. Do not promote to the default path
yet.

The implementation adds a DeepSeek V4-aware radix prefix cache behind
`--enable-dsv4-radix-prefix-cache`. The default DeepSeek V4 path still forces
the previous naive cache behavior. Runtime DSV4 opt-in requires
`page_size % 128 == 0`; target runs used `--page-size 256 --num-pages 128`.

Source parity and design notes were written first in
`performance_milestones/target08_radix_prefix_dsv4/DESIGN.md`.

## Design Summary

Mini now reuses its existing radix tree for DSV4 instead of introducing a new
allocator. Full-token pages remain the canonical cache ownership unit. The DSV4
KV pool already derives C4, C128, C4-indexer, and compression-state ownership
from full-token page allocation/free events, so prefix retention keeps those
derived component slots live through the existing refcount path.

The vLLM/SGLang alignment points used for this target:

- vLLM-style block ownership: cached blocks/pages have a ref/lock state and can
  only be evicted when not in use.
- vLLM hybrid alignment: prefix hit length must be safe for every KV group; mini
  enforces this by requiring DSV4 radix hits to be page-aligned and 128-aligned.
- SGLang DSV4 component ownership: full/SWA/C4/C128/indexer storage is modeled
  as separate derived components. Mini phase 1 keeps full pages as the single
  owner instead of implementing SGLang's independent SWA tombstone component.

## Implementation

Main changes:

- `python/minisgl/scheduler/config.py` and `python/minisgl/server/args.py`
  expose `enable_dsv4_radix_prefix_cache`, defaulting to false.
- `python/minisgl/scheduler/scheduler.py` keeps DSV4 on the naive cache unless
  the flag is set; opt-in validates `cache_type == "radix"` and
  `page_size % 128 == 0`.
- `python/minisgl/scheduler/cache.py` records prefix cache metrics and reports
  retained/protected/evictable pages plus DSV4 component retention.
- `python/minisgl/kvcache/deepseek_v4_pool.py` estimates retained DSV4 component
  slots and retained bytes for prefix cache reporting.
- `python/minisgl/kvcache/radix_cache.py` has an integrity checker for tree
  ownership, evictable/protected accounting, and parent/child consistency.
- `benchmark/offline/deepseek_v4_text_smoke.py` and
  `benchmark/offline/deepseek_v4_perf_matrix.py` accept the opt-in flag and
  write prefix metrics.
- `benchmark/offline/deepseek_v4_perf_matrix.py` adds
  `shared_prompt_reuse_bs8`, a two-stage shared-prefix scenario that inserts a
  prefix first, then sends reuse requests in the same engine process.

## Correctness

Unit and integration tests:

```bash
pytest -q -o addopts=''
```

Result: `135 passed, 4 warnings`.

Targeted coverage includes:

- full hit, partial hit, and miss;
- eviction and repeated hit/evict cycles;
- no DSV4 refcount leaks or double-free on full/C4/C128/C4-indexer slots;
- SWA window boundary at 128 tokens with 128/256-aligned pages;
- benchmark/text-smoke CLI default false and explicit true behavior.

Text smoke used the promoted TARGET07 path
`dsv4_sm80_a100_victory` / `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1` with
`--page-size 256 --num-pages 128`.

Prefix disabled:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory \
  --page-size 256 --num-pages 128 \
  --max-seq-len 2048 --max-extend-tokens 2048 --max-tokens 8 \
  --output performance_milestones/target08_radix_prefix_dsv4/text_smoke_prefix_off.json \
  --prompt "$(printf 'Please answer exactly OK. %.0s' {1..180})"
```

Prefix enabled:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory \
  --page-size 256 --num-pages 128 \
  --max-seq-len 2048 --max-extend-tokens 2048 --max-tokens 8 \
  --enable-dsv4-radix-prefix-cache \
  --output performance_milestones/target08_radix_prefix_dsv4/text_smoke_prefix_on.json \
  --prompt "$(printf 'Please answer exactly OK. %.0s' {1..180})"
```

Smoke result:

| mode | generated text | token ids | hit requests | saved tokens | retained pages | graph |
| --- | --- | --- | ---: | ---: | ---: | --- |
| prefix off | `OK` | `[11932]` | 0 / 2 | 0 | 0 | enabled, captured `[4,2,1]`, replay 2, eager 0 |
| prefix on | `OK` | `[11932]` | 1 / 2 | 768 | 3 | enabled, captured `[4,2,1]`, replay 2, eager 0 |

No text divergence was observed after a prefix hit.

## Performance A/B

Commands:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios shared_prompt_reuse_bs8 \
  --page-size 256 --num-pages 128 \
  --max-seq-len 1280 --max-extend-tokens 2048 --max-running-req 8 \
  --repeats 1 --warmup-repeats 0 \
  --output-dir performance_milestones/target08_radix_prefix_dsv4/perf_prefix_off \
  --keep-going
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios shared_prompt_reuse_bs8 \
  --page-size 256 --num-pages 128 \
  --max-seq-len 1280 --max-extend-tokens 2048 --max-running-req 8 \
  --repeats 1 --warmup-repeats 0 \
  --enable-dsv4-radix-prefix-cache \
  --output-dir performance_milestones/target08_radix_prefix_dsv4/perf_prefix_on \
  --keep-going
```

Results:

| metric | prefix off | prefix on | change |
| --- | ---: | ---: | ---: |
| status | pass | pass | - |
| elapsed | 16.70 s | 10.43 s | -37.6% |
| mean TTFT | 4.45 s | 2.85 s | -35.8% |
| prefill input tokens | 8704 | 1536 | -82.4% |
| prefill-forward | 9.96 s | 6.23 s | -37.5% |
| prefix hit rate | 0.000 | 0.875 | +0.875 |
| saved prefill tokens | 0 | 7168 | +7168 |
| max hit length | 0 | 1024 | +1024 |
| retained prefix pages | 0 | 4 | +4 |
| retained DSV4 memory | 0 B | 77,255,680 B | +77,255,680 B |
| evictions | 0 | 0 | 0 |

The apparent `prefill_tokens_per_s` drop in the prefix-on report is expected
because the denominator becomes the actually computed suffix tokens
(`1536`) rather than the original prompt tokens (`8704`). The useful signal is
that prefill-forward time and TTFT both dropped materially.

Perf graph status stayed enabled with captured sizes `[4,2,1]`. This scenario's
second stage decodes batch size 7, so both off/on reports show 15 eager decode
steps for bs7. The bs1 text smoke above verifies the promoted TARGET07 graph
replay path itself was not disabled by the prefix opt-in.

## Metrics Recorded

The reports now expose:

- prefix match/hit/miss/full/partial counts;
- total, max, and average prefix hit length;
- saved prefill tokens and suffix prefill tokens after hit;
- inserted cached tokens;
- retained, evictable, and protected prefix pages/tokens;
- evictions and evicted pages/tokens;
- retained DSV4 full/C4/C128/C4-indexer/compression-state slots;
- estimated retained DSV4 memory bytes.

In the shared-prefix opt-in run, retained component slots were:

- full slots: 1024;
- C4 slots: 256;
- C128 slots: 8;
- C4-indexer slots: 256;
- C4 state slots: 32;
- C128 state slots: 512;
- C4-indexer state slots: 32.

## Risks And Follow-Ups

This is a conservative phase-1 implementation. It is correct for the tested
page-aligned DSV4 path, but it is less memory-efficient than SGLang's separate
SWA component/tombstone design because retaining a prefix keeps full pages as
the canonical owner.

Do not extend this target into FP8 KV cache, INT8 MoE, PyNCCL, or attention
kernel work. Those remain follow-up targets.

Recommended next step: keep the feature opt-in and promote it only as an
experimental/controlled path. Before making it default, run longer multi-request
serving tests with sustained eviction pressure and consider a TARGET09 split
for SGLang-style independent SWA component retention.
