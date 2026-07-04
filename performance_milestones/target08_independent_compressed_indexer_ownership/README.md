# TARGET 08.21.2 DSV4 Independent Compressed/Indexer Ownership

Date: 2026-07-04

## Result

Decision: **proceed to TARGET 08.21.3**.

Route B / B1 now has an explicit opt-in:

```text
--enable-dsv4-component-loc-ownership
```

With the opt-in enabled, retained C4, C128, and indexer component pages have
independent ownership/refcounts and can survive after old full/SWA head pages
are released.  Eager attention/indexer metadata consumes component-owned page
tables instead of dynamically deriving retained component reads from released
full locs.

Graph replay/deforest is intentionally guarded off for this opt-in and remains
TARGET 08.21.4 work.  Compression state ownership remains TARGET 08.21.3 work;
this target uses a conservative safe-hit guard that keeps one live full/SWA tail
page per retained branch.

## Exact Commands

Implementation validation:

```bash
python -m py_compile \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/kvcache/__init__.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/kvcache/radix_cache.py \
  python/minisgl/scheduler/cache.py \
  python/minisgl/scheduler/config.py \
  python/minisgl/scheduler/scheduler.py \
  python/minisgl/server/args.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_dsv4_cache_option_guards.py

pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_cache_allocate.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_scheduler.py \
  tests/engine/test_graph_runner.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py

ruff check \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/kvcache/__init__.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/kvcache/radix_cache.py \
  python/minisgl/scheduler/cache.py \
  python/minisgl/scheduler/config.py \
  python/minisgl/scheduler/scheduler.py \
  python/minisgl/server/args.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_dsv4_cache_option_guards.py \
  performance_milestones/target08_independent_compressed_indexer_ownership/scripts/probe_component_ownership.py

git diff --check
```

Ownership probe:

```bash
python performance_milestones/target08_independent_compressed_indexer_ownership/scripts/probe_component_ownership.py
```

TP8 text smoke:

```bash
timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants fallback \
  --page-size 256 \
  --num-pages 64 \
  --max-seq-len 512 \
  --max-extend-tokens 512 \
  --max-tokens 8 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output performance_milestones/target08_independent_compressed_indexer_ownership/raw/text_smoke_component_ownership.json \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
  --prompt 'Answer in one short English sentence: what color is the sky on a clear day?' \
  > performance_milestones/target08_independent_compressed_indexer_ownership/raw/text_smoke_component_ownership.log 2>&1
```

## Artifacts

```text
performance_milestones/target08_independent_compressed_indexer_ownership/
  README.md
  DESIGN_DELTA.md
  raw/component_ownership_probe.json
  raw/text_smoke_component_ownership.json
  raw/text_smoke_component_ownership.fallback.json
  raw/text_smoke_component_ownership.log
  scripts/probe_component_ownership.py
  summaries/component_ownership_summary.json
  summaries/component_ownership_summary.md
```

## Git Status Summary

This target modifies the DSV4 KV pool, radix cache, scheduler/cache manager,
attention metadata, engine/config/CLI plumbing, benchmark flag pass-through, and
focused tests.  The prior 08.21.1 preflight milestone remains untracked in this
workspace and was left untouched.

```text
 M benchmark/offline/deepseek_v4_perf_matrix.py
 M benchmark/offline/deepseek_v4_text_smoke.py
 M python/minisgl/attention/deepseek_v4.py
 M python/minisgl/engine/engine.py
 M python/minisgl/kvcache/__init__.py
 M python/minisgl/kvcache/deepseek_v4_pool.py
 M python/minisgl/kvcache/radix_cache.py
 M python/minisgl/scheduler/cache.py
 M python/minisgl/scheduler/config.py
 M python/minisgl/scheduler/scheduler.py
 M python/minisgl/server/args.py
 M tests/attention/test_deepseek_v4_backend_metadata.py
 M tests/benchmark/test_deepseek_v4_perf_matrix.py
 M tests/benchmark/test_deepseek_v4_text_smoke.py
 M tests/core/test_deepseek_v4_kvcache.py
 M tests/core/test_dsv4_cache_option_guards.py
?? performance_milestones/target08_component_loc_table_preflight/
?? performance_milestones/target08_independent_compressed_indexer_ownership/
```

## Runtime Design

See `DESIGN_DELTA.md` for the full design delta.

Short version:

- full/SWA pages still use the normal KV page allocator;
- C4, C128, and C4-indexer use independent component page free lists and
  refcounts under the opt-in;
- radix nodes store `DSV4ComponentPageHandles`;
- retained full/SWA head pages are released with component ownership retained;
- evicting a radix node releases its component handles and any live full tail;
- default mode keeps phase-1 `full_loc // ratio` behavior.

## Runtime Component Loc Schema

Eager attention metadata now carries direct component page tables when the
opt-in is active:

| table | consumer |
| --- | --- |
| `c4_page_table` | sparse C4 attention metadata |
| `c128_page_table` | sparse C128 attention metadata |
| `c4_indexer_page_table` | indexer logits/topk metadata |

The metadata builder joins component pages from the prefix handle with active
suffix component mappings.  C4/C128 sparse indices and indexer page tables are
then gathered from component-owned tables, not from retained full loc division.

## Correctness Results

CPU ownership probe:

```json
{
  "all_passed": true,
  "scenario_count": 7
}
```

Covered scenarios:

| scenario | result | checks |
| --- | --- | ---: |
| full hit / partial hit / miss | pass | 5/5 |
| full-page reuse and double-free guard | pass | 6/6 |
| page boundaries 255/256/257/258 | pass | 4/4 |
| C4/C128/indexer loc boundaries | pass | 6/6 |
| repeated hit/evict cycles | pass | 7/7 |
| multi-prefix branching | pass | 4/4 |
| eviction pressure | pass | 2/2 |

Focused unit coverage:

- opt-in guard requires DSV4 radix prefix cache;
- `window_size <= page_size` guard is enforced for the B1 state-safe route;
- default parser/config paths keep the opt-in disabled;
- component ownership releases old full heads without stale component reuse;
- attention/indexer metadata uses direct component tables when old full heads are
  tombstoned;
- graph runner tests confirm graph is not required for this eager route.

## Guarded Oracle Status

TARGET 08.198 says the pass/fail oracle must be slot-pinned and same-layout;
cross-slot generated-token equality is diagnostic only.  This target applies
that guard as follows:

- pass/fail CPU metadata oracle is slot-pinned and same-layout;
- TP8 text smoke is treated as human-readable output sanity, not as a broad
  cross-slot equality oracle;
- full 08.198 activation/logit harness was not extended to this new opt-in in
  this target because the existing harness is prefix-disabled.  The eager B1
  metadata path is nevertheless covered by the direct-table unit oracle and the
  ownership probe.

## Text Smoke

TP8 fallback text smoke with `--enable-dsv4-component-loc-ownership` passed.

Summary from `raw/text_smoke_component_ownership.json`:

| field | value |
| --- | --- |
| status | pass |
| model | `/models/DeepSeek-V4-Flash` |
| TP | 8 |
| page size | 256 |
| graph enabled | false |
| eager decode count | 7 |
| replacement chars in log | 0 |

Outputs were short, printable, and sane:

| prompt style | output |
| --- | --- |
| shared-prefix Chinese city question | `杭州西湖位于杭州市。` |
| shared-prefix Chinese province question | `浙江省。` |
| non-shared English color question | `The sky is blue on a clear day` |

The one-batch text smoke does not create retained prefix hits; prefix-hit
ownership is covered by the CPU probe.

## Capacity Ledger

Probe page size is 128 for compact CPU coverage.  A 260-token prompt retains
256 prefix tokens:

| mode / moment | live full pages | C4 slots | C128 slots | indexer slots | free full pages | available component pages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 0 | 0 | 0 | 0 | 8 | 8 |
| after B1 retain | 1 | 64 | 2 | 64 | 7 | 6 |
| after eviction cleanup | 0 | 0 | 0 | 0 | 8 | 8 |

Phase-1 would keep both full pages live for the same 256 retained tokens.  B1
keeps only one full/SWA tail page and retains the two component pages for each
component pool, so the old full/SWA head page is safely reclaimable.

Under eviction pressure with only 3 pages, repeated 260-token prompts triggered
evictions and ended each iteration with one live full tail page, two component
pages, and no refcount leak.

## Compression State Guard

Compression state is still phase-1/full-owned.  Safe hit length is therefore
guarded by live full/SWA tail availability:

| prompt length | phase-1 aligned hit | B1 safe hit |
| ---: | ---: | ---: |
| 255 | 128 | 128 |
| 256 | 128 | 0 |
| 257 | 256 | 256 |
| 258 | 256 | 256 |

Multi-prefix branching also obeys this guard: a shared node whose full/SWA value
is fully tombstoned is not returned as a safe hit, while full original branches
with a live tail still hit.

## Graph And Deforest

When the opt-in is enabled, DSV4 CUDA graph capture is forced off.  TP8 smoke
recorded:

```json
{
  "enabled": false,
  "captured_bs": [],
  "eager_decode_count": 7
}
```

TARGET 08.21.4 should port graph replay and deforest to copy/stage component
tables instead of deriving compressed metadata from `raw_out_loc // ratio`.

## Decision

Proceed to TARGET 08.21.3.

Reason: independent C4/C128/indexer ownership works in eager mode, old full/SWA
head pages can be released without stale component reads, double-free/leak
guards pass under reuse and eviction pressure, metadata can consume independent
component locs, phase-1 rollback remains the default, and graph is fail-closed.

The next blocker to remove is compression-state ownership so B1 no longer needs
the live-tail safe-hit guard.
