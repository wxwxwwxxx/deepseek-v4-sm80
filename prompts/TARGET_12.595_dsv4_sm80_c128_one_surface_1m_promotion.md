# TARGET 12.595: DSV4 SM80 C128 One-Surface 1M Promotion

## Status

Execution-ready after TARGET 12.59. TARGET 12.59 classified the native helper as
`ONE_SURFACE_INTEGRATION_READY`; this target owns only the narrow eager-prefill
integration, focused contract tests, and the full-model promotion ladder.

## Proven input from TARGET 12.59

The mini-owned `c128_prefill_page_indices_one_surface` Triton helper maps the
Route-B C128 component page table and raw C128 lengths directly to the final
int32 component locations. Across 40 no-weight cases it produced zero exact
mismatches, including ragged rows and 127/128/129 token boundaries. At
`[8192,8192]` it allocated exactly the 256 MiB output, no additional CUDA-visible
temporary, no full int64 matrix, and used one kernel launch.

The concrete consumer census establishes that release eager-prefill C128 fast
attention reads only:

- `c128_page_indices`: int32 `[query_rows, aligned_c128_width]`;
- `c128_topk_lengths_clamp1`: int32 `[query_rows]`.

It does not read eager `c128_raw_indices` or `c128_full_indices`. Decode CUDA
graphs, prep/replay oracles, prefix debugging, and the final paged-MQA reference
fallback may still require raw/full and keep their existing explicit ABI.

## Integration boundary

Select one-surface construction only when all of the following hold:

1. `not batch.is_decode`;
2. at least one C128 layer exists;
3. Route-B component-location ownership is enabled;
4. the per-query-row `c128_page_table` exists;
5. the device is CUDA sm80 and the Triton helper is available;
6. the configured eager attention path supports the page-indices + lengths
   two-source contract;
7. no explicit debug/reference/oracle request requires eager raw/full matrices.

For that path, derive the logical C128 width from the existing Python
`max_seqlen_k`/request lengths:

```text
raw_width     = max(max_seqlen_k // 128, 1)
aligned_width = ceil_div(raw_width, 64) * 64
```

This must match the current attention row-stride contract and must not call
`lengths.max().item()`, `.tolist()`, or otherwise introduce a GPU-to-CPU sync.
Call the native helper with `c128_page_table`, the unclamped raw lengths,
`aligned_width`, and `c128_component_page_size`.

Publish:

```text
c128_page_indices          int32 [rows, aligned_width]  # real one surface
c128_topk_lengths_clamp1   int32 [rows]
c128_raw_indices           int32 [rows, 1], all -1      # compatibility placeholder
c128_full_indices          int32 [rows, 1], all -1      # compatibility placeholder
```

Record a runtime marker containing backend
`triton_c128_prefill_one_surface`, rows, aligned width, final surface bytes,
and raw/full placeholder bytes.

## Eager, decode, and fallback contract

- Supported release eager prefill is fail-closed: if its one-surface helper is
  unavailable or rejects the tensor contract, raise an owner-specific error.
  Never silently call the legacy all-indices constructor and recreate the OOM.
- Keep the legacy raw/page/full constructor behind a clearly named explicit
  materialization boundary for non-release configurations and debug/reference
  oracles that genuinely require those tensors.
- Leave decode behavior byte-for-byte and address-for-address unchanged:
  capture allocation widths, raw/page/full buffers, graph prep, direct replay,
  replay copies, clamping, and oracle comparisons remain on the current path.
- Preserve SWA independent lifecycle, Route-B ownership and prefix handles,
  page size 256, serving-default 8192-token chunks, BF16, MTP-off state, and the
  existing CUDA graph buckets.

## Focused validation

Add tests that prove:

- eager prefill dispatches to `triton_c128_prefill_one_surface`;
- raw/full are `[rows,1]` all-`-1` placeholders;
- page indices exactly equal the existing Torch oracle;
- ragged multi-request lengths and 127/128/129 boundaries are correct;
- prefix-hit Route-B component tables, including invalid/missing pages, map
  correctly;
- helper unavailability is explicit fail-closed behavior on the supported
  release route and does not invoke full materialization;
- non-release/debug oracle materialization remains explicit;
- decode graph metadata allocation, prep, replay copy, and oracle contracts are
  unchanged;
- the eager fast-attention consumer reads only page indices and lengths.

Run syntax, lint, and focused attention/kernel/benchmark tests before macros.

## Fresh-process TP8 promotion ladder

Each row must use a new independent `torchrun` process, 8 x A100/sm80, model
`/models/DeepSeek-V4-Flash`, true `dsv4_sm80_release_default`, page size 256,
serving-default `max_extend_tokens=8192`, and no temporary feature env that
changes the functional path:

```text
4096    / 1024 / bs4   repeat/performance guard
262144  / 1    / bs1   capacity/performance guard
524288  / 8    / bs1   long prefill-to-decode guard
1048576 / 8    / bs1   1M promotion gate
```

Run the 1M row directly with `decode_len=8`. A passing row must cover the final
prefill chunk, first-token sampling, entry into the decode CUDA graph, and later
multi-token generation. Do not first repeat the entire 1M prefill with output
length one.

For every macro record:

- exact chunk progress;
- TTFT, prefill tok/s, decode tok/s, and wall time;
- peak allocated/reserved and driver free memory for every rank;
- C128 backend marker, width, final surface bytes, and placeholder bytes;
- SWA/component/full page ledger and prefix/cache lifecycle;
- decode graph replay/eager counts;
- output token range, exact output length, and decoded-text sanity;
- on failure, the first owner, tensor shape, and requested allocation bytes.

## Promotion and stop conditions

Promote only if:

- 1M/8 completes end-to-end;
- no unbounded C128 int64 or retained eager raw/page/full family returns;
- 262k and 512k show no material performance or memory regression;
- SWA independence, Route-B ownership, prefix/cache lifecycle, and decode graph
  replay remain healthy;
- focused tests and output text/token sanity pass.

Do not tune MTP, precision, `max_extend_tokens`, CUDA graph buckets, the
streaming indexer, Marlin, communication, SWA ownership, or component ownership
in this target. Do not reduce KV pages merely to make 1M fit unless the remaining
owner is proven to be a reasonable fixed workspace reserve.

If one-surface exposes a new first-order owner, reproduce and attribute it once,
then stop and propose one focused follow-up target. If the complete 256 MiB
surface itself cannot fit, recommend query-row tiling or direct attention
consumption of component page table + lengths; do not begin a broad attention
rewrite here.

## Output

Write:

```text
performance_milestones/target12_c128_one_surface_1m_promotion/README.md
```

The report must contain the final promote/block decision, exact integration
guards, eager/decode/fallback contracts, changed files and tests, the complete
4096/262k/512k/1M ladder, C128 before/after memory accounting, 1M
prefill-to-decode evidence, and an explicit recommendation to enter TARGET
12.60 or open the focused C128 tiling/direct-attention target.
