# TARGET 08.21.3 DSV4 Compression-State Ownership

Date: 2026-07-04

## Result

Decision: **proceed to TARGET 08.21.4 with the SWA-tail guard still enabled**.

Route B now has independent ownership/refcounts for retained C4 attention
state, C128 attention state, and C4 indexer state pages.  These state pages are
stored in the same radix component handle as the B1 C4/C128/indexer component
pages, survive old full-page reuse, and are released on radix eviction.

The page-boundary `prompt_len=256` hit is still guarded from `128` to `0`.
That remaining loss is not a state-owner bug: mini still stores SWA attention
KV in the full-token page namespace, so suffix prefill from a 128-token hit
would need the tombstoned first page for SWA attention.  B2 therefore removes
the state dependency on released full/SWA locs, but does not pretend to own SWA
data independently.

## Exact Commands

Implementation validation:

```bash
python -m py_compile \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/kvcache/radix_cache.py \
  python/minisgl/scheduler/cache.py \
  tests/core/test_deepseek_v4_kvcache.py

pytest -q tests/core/test_deepseek_v4_kvcache.py -q

ruff check \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/kvcache/radix_cache.py \
  python/minisgl/scheduler/cache.py \
  tests/core/test_deepseek_v4_kvcache.py

pytest -q \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_cache_allocate.py

python -m py_compile \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_cache_allocate.py
```

Milestone probe:

```bash
python performance_milestones/target08_compression_state_ownership/scripts/probe_state_ownership.py

ruff check \
  performance_milestones/target08_compression_state_ownership/scripts/probe_state_ownership.py

python -m py_compile \
  performance_milestones/target08_compression_state_ownership/scripts/probe_state_ownership.py
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
  --output performance_milestones/target08_compression_state_ownership/raw/text_smoke_state_ownership.json \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
  --prompt 'Answer in one short English sentence: what color is the sky on a clear day?' \
  > performance_milestones/target08_compression_state_ownership/raw/text_smoke_state_ownership.log 2>&1
```

## Git Status Summary

This target adds B2 state ownership on top of the existing unstaged B1 Route B
work in this workspace.  Files touched by B2:

```text
M python/minisgl/kvcache/deepseek_v4_pool.py
M python/minisgl/kvcache/radix_cache.py
M python/minisgl/scheduler/cache.py
M tests/core/test_deepseek_v4_kvcache.py
?? performance_milestones/target08_compression_state_ownership/
```

The workspace still also contains the B0/B1 milestone directories and B1 code
changes listed in the 08.21.2 README.

## Ownership Rule

Under `--enable-dsv4-component-loc-ownership`, each allocated full page now gets
independent state pages:

| state | slots per full page | owner |
| --- | ---: | --- |
| C4 attention state | 8 | Route B state allocator |
| C128 attention state | 128 | Route B state allocator |
| C4 indexer state | 8 | Route B state allocator |

The state pages are captured in `DSV4ComponentPageHandles` with the retained
C4/C128/indexer pages.  Releasing old full/SWA head pages clears active
full-to-state staging maps but does not free retained state pages.  Radix
eviction releases the state pages and detects double-free through slot
refcounts.

## Safe Fixed Point

The Route B match validator is now:

```text
matched boundary is valid iff:
  final matched node has a live full/SWA tail page
  and every matched path node has either independent state pages or a live tail
```

The final live-tail requirement remains because SWA attention KV is still stored
in full-token pages in mini.  Independent state ownership alone cannot make a
tombstoned SWA page safe for suffix prefill.

Boundary probe:

| mode | prompt len 256 hit | prompt len 257 hit |
| --- | ---: | ---: |
| phase1 | 128 | 256 |
| route_b | 0 | 256 |

## Correctness Table

| check | result | evidence |
| --- | --- | --- |
| state loc formula C4/C128/indexer | pass | `raw/state_ownership_probe.json` |
| state page retain after full head tombstone | pass | retained state pages `[0, 1]` |
| state page no stale reuse | pass | reused state pages `[3]`, stale flags false |
| page boundary 256 guard | pass | Route B hit `0`, not unsafe `128` |
| page boundary 257 hit | pass | Route B hit `256` with state handles |
| repeated eviction cleanup | pass | counts after eviction all zero |
| focused unit tests | pass | 12 KV-cache tests, 21 metadata/guard/cache tests |
| text smoke | pass | TP8 fallback smoke |

## Text Smoke

`raw/text_smoke_state_ownership.json` status: `pass`.

| field | value |
| --- | --- |
| model | `/models/DeepSeek-V4-Flash` |
| TP | 8 |
| page size | 256 |
| graph enabled | false |
| eager decode count | 7 |

Outputs:

| prompt style | output |
| --- | --- |
| shared-prefix Chinese city question | `杭州西湖位于杭州市。` |
| shared-prefix Chinese province question | `浙江省。` |
| non-shared English color question | `The sky is blue on a clear day` |

## State Memory And Capacity

The state tensors already existed in mini's DSV4 KV pool.  B2 adds ownership,
refcounts, and free-list accounting; it does not add a new persistent tensor.

CPU probe with `page_size=128`, 260-token prompt retaining 256 tokens:

| mode / moment | live full slots | C4 slots | C128 slots | indexer slots | C4 state | C128 state | indexer state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 after insert | 256 | 64 | 2 | 64 | 0 | 0 | 0 |
| Route B after retain | 128 | 64 | 2 | 64 | 16 | 256 | 16 |
| Route B after eviction | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

For the tiny CPU config, retained state bytes reported by
`estimate_prefix_retention()` were:

| state | bytes |
| --- | ---: |
| C4 attention state | 1024 |
| C128 attention state | 8192 |
| C4 indexer state | 512 |

## Remaining Limitations

- The current mini eager compressor does not read or write
  `DSV4CompressStatePool`; it computes compressed rows only from the current
  extend tensor.  B2 therefore implements the ownership boundary required by a
  future stateful/fused compressor, but does not change compression math.
- The `prompt_len=256` hit remains guarded because SWA KV is still full-owned.
- CUDA graph replay and metadata deforest stay guarded off for Route B and are
  TARGET 08.21.4 work.

## Decision

Proceed to TARGET 08.21.4 with Route B still opt-in and graph guarded off.
Compression state no longer depends on released full/SWA locs, but Route B still
needs a live full/SWA tail for SWA attention correctness.
