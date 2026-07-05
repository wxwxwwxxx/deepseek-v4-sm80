# TARGET 09.4: DSV4 SM80 Minimal FP8 KV/Cache Slice

## Status

Conditional.  Run only if TARGET 09.3 identifies a concrete FP8 cache slice with
meaningful capacity or bandwidth upside.

## Goal

Implement and benchmark the smallest FP8 KV/cache slice that exercises the real
store/quant and gather/dequant boundary selected by TARGET 09.3:

```text
SWA packed MLA cache store + selected-row gather/dequant parity harness
```

This target should prove whether the selected FP8 cache design is practical
before attempting full E2E rollout.

The first slice is not full FP8 KV/cache.  It must not migrate C4/C128 compressed
KV, C4 indexer cache, compression state, or retained prefix pages unless the
report explicitly updates the scope and explains why the SWA slice was
insufficient.

## Selected Slice From TARGET 09.3

Use the source-aligned DeepSeek V4 packed MLA layout:

- `448` noPE dims quantized to FP8 bytes;
- `64` RoPE dims kept in BF16;
- `8` UE8M0 scale/pad bytes per token, seven scales for `448 / 64` noPE blocks
  plus one padding byte;
- page stride aligned to the SGLang/vLLM packed layout.

TARGET 09.3 estimated this SWA-only slice saves `0.576 GiB/rank` at
`page_size=256`, `num_pages=128`, which is about 86% of the full
source-aligned MLA+indexer saving.

## Required Design Constraints

- Follow SGLang/vLLM layout and scale behavior unless there is hard evidence for
  a better mini-specific design.
- Avoid standalone quant/dequant paths that cause extra HBM round trips.
- Keep prefix-cache Route B component ownership correct.
- Keep graph replay compatible.
- Keep feature opt-in and rollback simple.

## Required Work

1. Implement minimal slice

   Implement only the SWA packed MLA component from TARGET 09.3:

   - pack/store BF16 SWA rows into `448 FP8 noPE + 64 BF16 RoPE + 8 scale`
     source-aligned rows;
   - selected-row gather/dequant back to BF16;
   - parity harness against mini's current BF16 SWA cache rows.

   Do not broaden the slice without updating the report.

   Acceptable implementation forms:

   - a standalone harness that uses real mini SWA shapes, page tables, and locs;
   - a small mini opt-in path guarded by an off-by-default env flag;
   - a port of SGLang/vLLM source-aligned kernels if they can be built against
     mini's ABI.

   Non-goals for this target:

   - C4/C128 compressed KV migration;
   - C4 indexer replacement;
   - compression-state quantization;
   - retained-prefix page migration;
   - full E2E FP8 KV/cache rollout.

2. Microbench

   Compare:

   - current BF16 cache path;
   - FP8 store/quant;
   - FP8 gather/dequant;
   - combined store + decode gather/dequant at real shapes.

   Include HBM traffic estimates and scale/workspace overhead.

3. Correctness checks

   Run selected-row/page comparisons against BF16 cache:

   - direct cache value checks after store;
   - gather/dequant value checks;
   - attention input checks if practical;
   - SWA tail-heavy decode checks with real `out_loc` semantics;
   - prefix-cache hit/remap checks only for SWA rows touched by the slice.

4. Graph checks

   Verify graph capture and replay for relevant batch buckets.  Record eager
   fallbacks and graph memory delta.

5. Decision

   Decide whether TARGET 09.5 is justified.

## Gates

Pass if:

- FP8 slice is source-aligned with SGLang/vLLM or explicitly justified;
- RoPE tail remains BF16;
- microbench shows useful capacity and no unacceptable latency loss;
- correctness checks pass within documented tolerance;
- graph replay remains active;
- capacity ledger remains meaningful after scales/workspace.

Stop if:

- store/quant is slow on SM80;
- gather/dequant requires full-cache dequantization;
- standalone casts dominate decode;
- prefix-cache correctness breaks;
- graph replay breaks;
- C4/C128/indexer scope expands before the SWA slice proves parity and graph
  safety;
- macro ceiling is below `1%` and capacity savings are not compelling.

## Deliverables

Write results under:

```text
performance_milestones/target09_minimal_fp8_kv_cache_slice/
```

Include:

- `README.md` with TARGET 09.5 go/no-go;
- implementation notes and env flags;
- microbench tables;
- correctness outputs;
- graph replay/capture status;
- capacity ledger update;
- rollback instructions.
