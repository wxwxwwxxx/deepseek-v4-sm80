# TARGET 08.18 DSV4 Prefix Cache Memory Ledger

## Result

Decision for TARGET 08.20: **GO with guardrails**.

The phase-1 DSV4 prefix cache should remain a controlled opt-in.  TARGET 08.10
found generated-token mismatches on synthetic hit workloads, so no default
promotion is justified yet.  However, the memory ledger says the SGLang-style
independent SWA/component retention target is worth doing as scoped capacity
work:

- sustained 08.10 retained state: `56 / 128` pages, `14,336` tokens,
  `1.007 GiB/rank`, or `43.8%` of the fixed KV page pool;
- eviction pressure retained state: `112 / 128` pages, `28,672` tokens,
  `2.015 GiB/rank`, or `87.5%` of the fixed KV page pool;
- that removes `2.8` and `5.6` equivalents of a `4096+1024` request from the
  fixed `--num-pages 128` logical capacity.

08.20 should not be a graph-memory, low-precision, or global allocator target.
It should solve the component-retention problems listed below, behind opt-in
gates and with correctness tests.

## Artifacts

```text
performance_milestones/target08_prefix_cache_memory_ledger/
  README.md
  raw/
    fixed_inputs.json
    ledger_cases.json
    source_measurements.json
  scripts/
    build_prefix_cache_memory_ledger.py
  summaries/
    capacity_table.md
    component_bytes_table.md
    go_no_go.json
    go_no_go.md
    ledger_cases.csv
    ledger_table.md
    sglang_savings.csv
    sglang_savings_table.md
```

Regenerate:

```bash
python performance_milestones/target08_prefix_cache_memory_ledger/scripts/build_prefix_cache_memory_ledger.py
```

## Fixed Inputs

All values are per rank and use `bytes / 2^30` for GiB.

| Item | Value | Interpretation |
| --- | ---: | --- |
| Page size | `256` tokens | TARGET 08 page policy. |
| Fixed pages | `128` | Selected capped serving policy for graph-mode tests. |
| Fixed KV pool | `2,491,495,680 B` / `2.320 GiB` | `128` pages, `32,768` token slots. |
| Marginal retained prefix page | `19,313,920 B` / `0.01799 GiB` | DSV4 component bytes from `estimate_prefix_retention()`. |
| CUDA graph private pool | `20,440,940,544 B` / `19.037 GiB` | TARGET 08.05/08.06/08.07 `[1,2,4,8,16]`; reported as `~19.04 GiB`. |
| First graph cost | `~18.83 GiB` | TARGET 08.06/08.07 single `[16]`; later buckets add only tens of MiB. |
| Promoted BF16 cache baseline | `1.588 GiB` | TARGET 08.07 model-prepare persistent baseline. |
| Device free after graph capture | `36.448 GiB` | Measured after the fixed KV pool, BF16 caches, weights, and graph private pool are resident. |

Important accounting rule: under `--num-pages 128`, prefix retention does not
create a second physical KV allocation.  It consumes logical capacity inside the
already allocated KV pool.  The tables therefore show both logical remaining KV
pages and a conservative "if prefix were extra" device-free margin only for
cross-budget intuition.

## Formulas

Model constants from `/models/DeepSeek-V4-Flash/config.json` and mini's
`DeepSeekV4KVCache`:

```text
L = 43 layers
H = 512 head_dim
Hidx = 128 index_head_dim
C4_layers = 21
C128_layers = 20
dtype_size = 2 bytes
state_dtype_size = 2 bytes
C4_state_ring = 8
C128_state_ring = 128
```

For retained page count `P` and page size `S = 256`:

```text
tokens = P * S
full_slots = tokens
SWA_slots = tokens
C4_slots = P * S / 4
C128_slots = P * S / 128
C4_indexer_slots = C4_slots
C4_state_slots = P * 8
C128_state_slots = P * 128
C4_indexer_state_slots = P * 8

SWA_bytes = L * tokens * H * 2
C4_bytes = C4_layers * C4_slots * H * 2
C128_bytes = C128_layers * C128_slots * H * 2
C4_indexer_bytes = C4_layers * C4_indexer_slots * Hidx * 2
C4_indexer_fp8_extra_bytes = C4_layers * C4_indexer_slots * (Hidx + 4)
C4_state_bytes = C4_layers * P * 8 * 4 * H * 2
C4_indexer_state_bytes = C4_layers * P * 8 * 4 * Hidx * 2
C128_state_bytes = C128_layers * P * 128 * 2 * H * 2
retained_bytes = sum(all component bytes)
```

Capacity equivalents:

```text
fixed_kv_bytes_per_page = 2,491,495,680 / 128 = 19,464,810 B
equivalent_kv_pages_by_bytes = retained_bytes / fixed_kv_bytes_per_page
equivalent_kv_tokens_by_bytes = equivalent_kv_pages_by_bytes * 256
logical_4096_prompt_equiv = retained_pages / 16
logical_4096_plus_1024_request_equiv = retained_pages / 20
remaining_pages = 128 - retained_pages
```

## Measured Data Coverage

| Workload | Source | Retention used here |
| --- | --- | --- |
| short shared prefix | TARGET 08.10 `prefix_full_hit_257_bs4` | 1 page, 256 tokens, `0.018 GiB`. |
| 1024 prefix | TARGET 08 phase-1 `shared_prompt_reuse_bs8` | 4 pages, 1024 tokens, `0.072 GiB`. |
| 4096 prefix | Formula estimate, same 16-page size as 08.10 retained row | 16 pages, 4096 tokens, `0.288 GiB`. |
| multi-prefix | TARGET 08.10 `prefix_mixed_hit_miss_bs16` final retained state | 40 pages, 10,240 tokens, `0.719 GiB`. |
| 08.10 sustained workload | TARGET 08.10 `prefix_multi_112req_wave16` final retained state | 56 pages, 14,336 tokens, `1.007 GiB`. |
| eviction pressure | TARGET 08.10 `prefix_eviction_pressure_96req_wave16` final retained state | 112 pages, 28,672 tokens, `2.015 GiB`. |

## Slot Ledger

| case | pages | tokens | full/SWA slots | C4 slots | C128 slots | indexer slots | C4 state | C128 state | idx state | GiB/rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 1 | 256 | 256 | 64 | 2 | 64 | 8 | 128 | 8 | 0.018 |
| 1024-token prefix | 4 | 1024 | 1024 | 256 | 8 | 256 | 32 | 512 | 32 | 0.072 |
| 4096-token prefix | 16 | 4096 | 4096 | 1024 | 32 | 1024 | 128 | 2048 | 128 | 0.288 |
| multi-prefix mixed | 40 | 10240 | 10240 | 2560 | 80 | 2560 | 320 | 5120 | 320 | 0.719 |
| 08.10 sustained workload | 56 | 14336 | 14336 | 3584 | 112 | 3584 | 448 | 7168 | 448 | 1.007 |
| eviction pressure | 112 | 28672 | 28672 | 7168 | 224 | 7168 | 896 | 14336 | 896 | 2.015 |

## Component Bytes

| case | SWA/full GiB | C4 GiB | C128 GiB | indexer BF16 GiB | indexer FP8 extra GiB | compress-state GiB | total GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 0.010 | 0.001 | 0.000 | 0.000 | 0.000 | 0.006 | 0.018 |
| 1024-token prefix | 0.042 | 0.005 | 0.000 | 0.001 | 0.001 | 0.023 | 0.072 |
| 4096-token prefix | 0.168 | 0.021 | 0.001 | 0.005 | 0.003 | 0.091 | 0.288 |
| multi-prefix mixed | 0.420 | 0.051 | 0.002 | 0.013 | 0.007 | 0.227 | 0.719 |
| 08.10 sustained workload | 0.588 | 0.072 | 0.002 | 0.018 | 0.009 | 0.318 | 1.007 |
| eviction pressure | 1.176 | 0.144 | 0.004 | 0.036 | 0.019 | 0.637 | 2.015 |

The largest two owners are SWA/full-token rows and compression state.  C4/C128
compressed KV and indexer rows are small by bytes, but they remain correctness
critical because suffix prefill/decode cannot reuse a prefix unless all required
components agree on a safe hit length.

## Capacity Ledger

Fixed resident capacity before prefix logical occupancy:

| Item | GiB/rank | Equivalent KV pages | Equivalent KV tokens |
| --- | ---: | ---: | ---: |
| CUDA graph private pool | 19.037 | 1050.1 | 268,838 |
| Promoted BF16 cache baseline | 1.588 | 87.6 | 22,425 |
| Fixed `--num-pages 128` KV pool | 2.320 | 128.0 | 32,768 |
| Fixed subtotal | 22.945 | 1265.7 | 324,031 |
| Device free after graph capture | 36.448 | 2010.6 | 514,713 |

Prefix logical occupancy under the fixed KV pool:

| case | KV pool pages used | 4096 prompts eq | 4096+1024 req eq | remaining KV pages | remaining 4096+1024 reqs | free GiB if extra |
| --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 0.8% | 0.06 | 0.05 | 127 | 6.35 | 36.430 |
| 1024-token prefix | 3.1% | 0.25 | 0.20 | 124 | 6.20 | 36.376 |
| 4096-token prefix | 12.5% | 1.00 | 0.80 | 112 | 5.60 | 36.160 |
| multi-prefix mixed | 31.2% | 2.50 | 2.00 | 88 | 4.40 | 35.729 |
| 08.10 sustained workload | 43.8% | 3.50 | 2.80 | 72 | 3.60 | 35.441 |
| eviction pressure | 87.5% | 7.00 | 5.60 | 16 | 0.80 | 34.434 |

The graph private pool is the dominant physical memory cost.  08.20 will not
fix that `~19.04 GiB/rank` line.  Its value is logical serving capacity: freeing
full/SWA pages from long-lived prefixes while keeping the compressed state that
makes future hits useful.

## SGLang-Style Savings Upper Bound

SGLang's SWA component can tombstone out-of-window SWA data while keeping other
components on the radix node.  With `sliding_window=128` and `page_size=256`,
the page-aligned minimum SWA tail is one page per active prefix branch.

The table below is a theoretical single-branch upper bound:

- `recoverable full pages upper` assumes all retained pages belong to one
  branch and only one SWA tail page must remain;
- `SWA-only saved` frees only old SWA/full rows;
- `SWA+state saved` additionally assumes compression state outside the retained
  boundary can be released or reconstructed safely;
- compressed C4/C128/indexer bytes remain kept, because they are the point of
  the prefix hit.

| case | recoverable full pages upper | recoverable tokens upper | SWA-only saved GiB | SWA+state saved GiB | SWA+state saved eq KV pages | compressed kept GiB |
| --- | --- | --- | --- | --- | --- | --- |
| short shared prefix | 0 | 0 | 0.000 | 0.000 | 0.0 | 0.002 |
| 1024-token prefix | 3 | 768 | 0.031 | 0.049 | 2.7 | 0.007 |
| 4096-token prefix | 15 | 3840 | 0.157 | 0.243 | 13.4 | 0.029 |
| multi-prefix mixed | 39 | 9984 | 0.409 | 0.631 | 34.8 | 0.072 |
| 08.10 sustained workload | 55 | 14080 | 0.577 | 0.890 | 49.1 | 0.101 |
| eviction pressure | 111 | 28416 | 1.165 | 1.796 | 99.1 | 0.202 |

This upper bound is optimistic for the 08.10 sustained and eviction workloads.
Those workloads contain many short two-page prefixes, so each distinct branch
needs its own page-aligned SWA tail.  Still, the retained-page pressure is high
enough that even partial recovery can buy meaningful active-request capacity.

## 08.20 Scope

If TARGET 08.20 proceeds, it should solve these component-retention issues:

1. Separate full/SWA page lifetime from compressed component lifetime.  A radix
   node should be able to keep C4/C128/indexer data without pinning all historical
   full-token pages.
2. Add page-aligned SWA window retention.  With `sliding_window=128` and
   `page_size=256`, the first target is one SWA tail page per retained branch.
3. Define compression-state ownership.  Either retain only boundary state needed
   for suffix prefill/decode, or prove a deterministic reconstruction path from
   cached components.
4. Validate match length across all components.  A prefix hit is valid only if
   full/SWA tail, C4, C128, indexer, and state availability share a safe fixed
   point.
5. Add independent metrics and leak checks for full/SWA/C4/C128/indexer/state
   slots, protected vs evictable component pages, tombstones, and recovered
   pages.
6. Preserve the current graph-bucket policy and replay coverage.  08.20 should
   not expand `[1,2,4,8,16]` or chase graph private-pool attribution.

## Risk Assessment

- Correctness risk remains the top promotion blocker.  TARGET 08.10's synthetic
  generated-token mismatches require a logits/token follow-up before default
  promotion.
- Component retention is more complex than phase-1 full-page ownership.  The
  double-free/leak surface grows because full/SWA, compressed KV, indexer, and
  state can have different lifetimes.
- Physical free memory is not the immediate blocker under `--num-pages 128`;
  the measured post-capture free margin is still `36.448 GiB/rank`.  The problem
  is logical KV capacity and eviction pressure.
- The savings table is an upper bound.  Realized savings depend on prefix branch
  count, tombstone granularity, component pool sizing, and whether compression
  state can be safely reconstructed.

## Go/No-Go

**GO for TARGET 08.20, with guardrails.**

The go criteria are met: realistic 08.10 retained states exceed the `20%-30%`
KV-pool investigation threshold, the eviction-pressure case leaves only
`16 / 128` pages free before active request KV, and the theoretical SGLang-style
upper bound can recover multiple 4096+1024 request equivalents in long-prefix
states.

Guardrails:

- keep prefix cache controlled opt-in until a correctness follow-up resolves the
  synthetic mismatch;
- keep 08.20 scoped to independent SWA/component retention and metrics;
- do not implement low-precision cache, new graph allocator policy, or graph
  private-pool attribution inside 08.20.
