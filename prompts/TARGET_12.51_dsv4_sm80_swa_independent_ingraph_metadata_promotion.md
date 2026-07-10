# TARGET 12.51: DSV4 SM80 SWA Independent In-Graph Metadata Promotion

## Background

TARGET 12.50 selected the Tier A release bundle as the current default
DeepSeek V4 A100/sm80 path:

```text
performance_milestones/target12_release_bundle_optin_promotion_gate/README.md
```

That bundle now includes:

- page size 256;
- radix prefix cache;
- component loc ownership;
- A100 victory bundle;
- Route-B C4 direct graph metadata;
- in-graph replay metadata prep for the non-SWA-independent path;
- BF16 MoE reduce;
- PyNCCL threshold32m;
- Marlin WNA16 prebuild/release/capacity credit;
- component-slot clear on page allocation.

TARGET 12.50 also tested SWA independent lifecycle plus direct SWA metadata.
This path is **correct and graph-replay clean**, but remains opt-in because it
regresses macro throughput by about 12-18% versus Tier A:

```text
historical_4096_128_bs4:   -12.1% output tok/s
historical_4096_1024_bs4:  -13.9% output tok/s
serving_mixed_112req_wave16: -18.2% output tok/s
prefix_multi_112req_wave16: -16.9% output tok/s
```

The main known blocker is not correctness.  It is this fail-open:

```text
prep_metadata_in_graph_requested = true
prep_metadata_in_graph = false
prep_metadata_in_graph_unsupported_reason = "swa_independent_lifecycle_not_supported"
```

SWA independent lifecycle is still a high-priority default candidate because it
dramatically improves capacity:

```text
Tier A default:       2763 pages / 707,328 tokens
SWA independent path: 6457 pages / 1,652,992 tokens
per-page KV bytes:    19,313,920 B -> 8,041,728 B
```

This target should remove the in-graph metadata compatibility blocker and try
to make SWA independent lifecycle eligible for the default bundle.

## Goal

Extend `prep_metadata_in_graph` so it supports SWA independent lifecycle, then
rerun the SWA promotion gate.

The intended outcome is:

```text
Tier A release bundle + SWA independent lifecycle + SWA direct metadata
```

passes text sanity, oracle tests, graph replay, and macro performance gates
without `prep_metadata_in_graph` fail-opening.

If this succeeds, recommend promoting SWA independent lifecycle into the
DeepSeek V4 A100/sm80 default bundle.  If it fails, identify the precise
remaining blocker and keep SWA independent opt-in only for a concrete reason.

## Key Hypothesis

The performance regression in TARGET 12.50 is mostly caused by losing the
TARGET 12.47 in-graph metadata prep fast path.  SWA independent itself is not
the likely culprit because:

- text sanity passed;
- graph capture passed;
- graph replay stayed zero-eager;
- direct SWA replay metadata kernels already have SWA-independent mapping
  support;
- the macro regression aligns with the known metadata-prep fail-open.

Prove or falsify this hypothesis.

## Current Code Surfaces

Start from these files:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/csrc/
python/minisgl/kvcache/deepseek_v4_pool.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/kernel/test_deepseek_v4_wrappers.py
```

Important current behavior:

- `python/minisgl/attention/deepseek_v4.py` currently rejects
  `prep_metadata_in_graph` when
  `kvcache.swa_independent_lifecycle_enabled == True`.
- `_run_prep_metadata_in_graph_kernel()` calls
  `dsv4_kernel.prep_decode_metadata_in_graph(...)` without passing SWA
  independent mapping tensors.
- `dsv4_kernel.direct_decode_index_metadata_for_replay(...)` already supports
  SWA independent mapping through:

```text
swa_full_to_swa_page
swa_dummy_token_start
swa_dummy_page
swa_independent=True
```

This existing direct replay path is the first implementation oracle for the new
in-graph prep behavior.

## Required Design

### 1. Define The SWA Independent Prep Contract

For SWA independent lifecycle, `swa_page_indices` must be generated from the
full-token locations through the SWA ownership mapping:

```text
full_loc = ctx_page_table[req_table_idx, logical_token]
full_page = full_loc // page_size
page_offset = full_loc % page_size
swa_page = full_to_swa_page[full_page]
swa_loc = swa_page * page_size + page_offset
```

Rules:

- invalid full loc -> `-1`;
- missing/tombstoned SWA page -> `-1`;
- dummy full token -> SWA dummy page;
- use the kvcache dummy token/page contract rather than inventing a new one;
- preserve the existing SWA window ordering and `swa_topk_lengths`;
- do not weaken SWA refcount/free-list checks.

Use:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08_radix_prefix_dsv4.md
```

as the contract references.

### 2. Extend The Prep-In-Graph Kernel Boundary

Add SWA independent inputs to the in-graph metadata prep boundary only as
needed:

```text
swa_full_to_swa_page
swa_dummy_token_start
swa_dummy_page
swa_independent
```

Potential places:

- `DSV4RawDecodeGraphMetadata`;
- capture/core metadata surfaces;
- `_run_prep_metadata_in_graph_kernel()`;
- `dsv4_kernel.prep_decode_metadata_in_graph(...)`;
- the underlying Triton/CUDA wrapper.

Do not copy large SWA tables per replay if a persistent kvcache mapping tensor
can be referenced safely.  The graph should consume stable-address tensors where
possible, matching the existing graph-input style.

### 3. Oracle First

Before macro reruns, build a focused oracle:

```text
old SWA independent metadata path
vs
new prep_metadata_in_graph SWA independent path
```

Compare at least:

- `swa_page_indices`;
- `swa_topk_lengths`;
- `seq_lens`;
- C4 sparse lengths/indices;
- C128 lengths/indices;
- `c4_out_loc`, `c128_out_loc`, `c4_indexer_out_loc`;
- `swa_out_loc` if direct/fused SWA replay metadata is enabled.

Use the existing C4 oracle pattern from TARGET 12.46/12.47 and the existing
`direct_decode_index_metadata_for_replay_swa_independent_matches_oracle` kernel
test as a reference.

### 4. Keep Graph Replay Healthy

The fixed path must keep:

```text
captured buckets: [16,8,4,2,1]
eager decode count: 0
prep_metadata_in_graph_requested: true
prep_metadata_in_graph: true
prep_metadata_in_graph_unsupported_reason: null
```

for SWA independent variants.

### 5. Performance Gate

After correctness passes, rerun the TARGET 12.50 SWA macro comparison:

```text
release default Tier A
vs
Tier A + SWA independent + direct/page-table/replay metadata
```

Scenarios:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

Expected promotion direction:

- if SWA independent is within about 0-5% of Tier A while doubling capacity,
  recommend default promotion;
- if it remains 10%+ slower but no longer fail-opens, produce a fresh owner
  attribution before deciding;
- if correctness fails, fix ownership/metadata contract first;
- if the fix requires broad graph runtime redesign, keep SWA independent opt-in
  and document the blocker precisely.

## Suggested Work Plan

1. Reproduce/confirm the fail-open on the current branch using the 12.50 Stage D
   variant.
2. Add a unit-level SWA-independent prep-in-graph oracle.
3. Extend kernel/API boundary for SWA independent mapping.
4. Enable support by removing the explicit
   `swa_independent_lifecycle_not_supported` guard only after oracle coverage is
   in place.
5. Run focused tests.
6. Run text smoke for SWA independent/direct path.
7. Run four-scenario macro comparison.
8. If performance is still poor, collect a small owner/census attribution before
   recommending default/opt-in.

## Commands

Minimum static/unit validation:

```bash
python -m py_compile \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/kernel/test_deepseek_v4_wrappers.py::test_direct_decode_index_metadata_for_replay_swa_independent_matches_oracle \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/engine/test_dsv4_release_defaults.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

Text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_swadirect_replaymetafused \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_51_swa_ingraph_text_smoke.json
```

Macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_swadirect_replaymetafused \
  --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 serving_mixed_112req_wave16 prefix_multi_112req_wave16 \
  --num-pages 0 \
  --keep-going \
  --output-dir /tmp/dsv4_target12_51_swa_ingraph_macro
```

If same-process multi-variant comparison risks CUDA/env lifecycle pollution,
split the two variants into separate fresh `torchrun` commands and summarize
them manually.

## Output

Write the report to:

```text
performance_milestones/target12_swa_independent_ingraph_metadata_promotion/README.md
```

Report must include:

- whether the fail-open was removed;
- exact code/kernel boundary changes;
- oracle coverage and results;
- text sanity result;
- graph capture and replay/eager counts;
- capacity ledger before/after;
- macro comparison against Tier A release default;
- whether SWA independent should enter the default bundle now;
- if not, the precise remaining blocker and the next smallest target.

## Stop Conditions

Stop and report if:

1. SWA independent prep-in-graph oracle cannot be made to match the old metadata
   path for a concrete contract reason.
2. The in-graph kernel cannot safely consume the SWA full-to-SWA mapping without
   broad graph runtime redesign.
3. Text sanity or graph replay fails after the fix and the failing owner is
   identified.
4. The fix passes correctness and macro performance is close enough to justify
   default promotion.
5. The fix passes correctness but remains slower; collect enough attribution to
   decide the next target.

Do not reopen TARGET 9 low precision, MTP, or broad CUDA graph bucket expansion
inside this target.
