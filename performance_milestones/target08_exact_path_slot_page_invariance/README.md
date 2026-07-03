# TARGET 08.195 DSV4 SM80 Exact Path Slot/Page Invariance

Date: 2026-07-03

## Status

Partial fix plus guard.

The probe found and fixed one real exact-path bug in the DSV4 compressor: flattened batches were pooled every `ratio` rows, so compressed C4/C128 windows could cross request boundaries when prompt lengths were not multiples of the compression ratio. A four-request batch of 257-token prompts could therefore mix the last token of one request with the first tokens of the next request.

After the fix, identical-prompt drift improved materially, but exact slot invariance is still not clean. The remaining drift appears in attention outputs and persists across the tested SM80 toggles. Prefix-cache work must not treat general multi-slot no-hit batches as a clean oracle yet.

## Changed Code

- `python/minisgl/kernel/deepseek_v4.py`
  - Compression now forms rows only from contiguous per-request position windows ending at `(position + 1) % ratio == 0`.
  - The slow fallback uses the same position-contiguous rule.
  - CUDA graph capture now conservatively returns no compressed update, avoiding variable-shape `nonzero`/Python branching during capture.
- `python/minisgl/utils/dsv4_prefix_debug.py`
  - Added opt-in activation capture controlled by:
    - `MINISGL_DSV4_PREFIX_DEBUG_ACTIVATIONS`
    - `MINISGL_DSV4_PREFIX_DEBUG_MAX_ACTIVATION_ROWS`
    - `MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_ACTIVATIONS`
- `python/minisgl/models/deepseek_v4.py`
  - Added activation checkpoints for embedding, layer input, attention input/output, compressor/indexer output, MoE input/output, final norm, and logits.
  - Compressed activation capture now selects explicit per-request compressed rows instead of assuming token-shaped row indices.

## Artifacts

- Raw compact run evidence: `raw/`
- Reproduction and summary tables: `summaries/`
- Probe and summarizer scripts: `scripts/`

The per-event `.pt` tensor dumps and per-event JSON shards were pruned after summary generation. The retained raw files keep run metadata, manifests, and rank JSONL traces; rerunning `scripts/run_all_dsv4_exact_path_invariance_probe.sh` regenerates full local raw tensors if the summaries need to be recomputed from scratch.

## Commands

Full eager + CUDA graph probe:

```bash
bash performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_all_dsv4_exact_path_invariance_probe.sh
```

Summary command used after the full raw runs and before tensor pruning:

```bash
python performance_milestones/target08_exact_path_slot_page_invariance/scripts/summarize_dsv4_exact_path_invariance_probe.py \
  --eager performance_milestones/target08_exact_path_slot_page_invariance/raw/eager \
  --graph performance_milestones/target08_exact_path_slot_page_invariance/raw/graph \
  --output-dir performance_milestones/target08_exact_path_slot_page_invariance/summaries \
  --atol 2e-2 \
  --rtol 2e-2 \
  --activation-atol 2e-2 \
  --activation-rtol 2e-2
```

Toggle bisection harness:

```bash
bash performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_toggle_bisection_dsv4_exact_path.sh
```

Syntax check:

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/utils/dsv4_prefix_debug.py \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/summarize_dsv4_exact_path_invariance_probe.py
```

## Key Results

Before the compressor fix, identical 257-token prompts in different slots drifted by:

| Probe | Prefill max_abs | Decode max_abs |
| --- | ---: | ---: |
| identical rows, pre-fix | 4.35474 to 4.75032 | 10.2854 to 11.7161 |

After the compressor fix:

| Probe | Prefill max_abs | Decode max_abs |
| --- | ---: | ---: |
| identical rows, post-fix | 0.883242 to 0.887962 | 0.780247 to 0.940769 |
| single target alone vs slot0 | 1.13529 | 0.904431 |
| single target alone vs slot1 | 0.982245 | 0.859308 |
| single target alone vs slot2 | 0.961405 | 1.34038 |
| single target alone vs slot3 | 0.913656 | 1.04161 |

Stable single-request slot/page oracle:

| Probe | Prefill | Decode |
| --- | ---: | ---: |
| table row after 2 dummy batches | pass max=0 | pass max=0 |
| table row after 3 dummy batches | pass max=0 | pass max=0 |
| physical page one-page prelude | pass max=0 | pass max=0 |
| physical page mixed-pages prelude | pass max=0 | pass max=0 |

First divergent activation checkpoint:

| Lens | Scenario | Phase | First checkpoint | Result |
| --- | --- | --- | --- | --- |
| identical rows | identical_prompts_batch | prefill | layer3.attention_output | FAIL max=0.101562 |
| single vs slots | target_in_batch_slot0 | prefill | layer0.attention_output | FAIL max=0.0625 |

CUDA graph replay:

| Scenario | Prefill eager/graph | Decode eager/graph | Real bs | Padded bs | Graph source |
| --- | --- | --- | ---: | ---: | --- |
| identical_prompts_batch | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| target_in_batch_slot0..3 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| swa_boundary_127_128_129_bs3 | pass max=0 | FAIL max=2.18697 | 3 | 4 | cuda_graph_replay |
| page_boundary_255_256_257_258 | pass max=0 | pass max=0 | 4 | 4 | cuda_graph_replay |
| c4_c128_boundary_lengths | pass max=0 | pass max=0 | 8 | 8 | cuda_graph_replay |

Toggle checks after the compressor fix did not clear the remaining identical-prompt drift:

| Disabled toggle | Prefill max_abs range | Decode max_abs range |
| --- | ---: | ---: |
| `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16` | 0.829313 to 1.26655 | 0.959872 to 1.07875 |
| `MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE` | 0.770986 to 1.00032 | 0.939438 to 1.92459 |
| `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE` | 0.883242 to 0.887962 | 0.780247 to 0.940769 |
| `MINISGL_DSV4_SM80_COMPRESS_STORE` | 0.828114 to 1.24661 | 0.676111 to 1.25284 |
| `MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT` | 0.856058 to 1.21239 | 0.782796 to 1.67699 |

## Interpretation

- The compressor boundary fix is necessary and should stay.
- Request-table row churn and physical page churn are not sufficient explanations when the target is run alone in a slot-pinned setup; those single-request probes pass exactly.
- General batch-slot invariance still fails even with prefix cache disabled.
- The bs=3 to graph bucket 4 decode path remains graph-specific unsafe for correctness assertions.
- The earliest remaining drift is in attention, not in logits-only bookkeeping.

## Guard For Follow-On Work

Do not use TARGET 08.20 general prefix-cache acceleration as an acceptance target yet.

Allowed narrow oracle:

- Prefix cache disabled.
- Single target request, slot-pinned.
- Page/table churn normalized by comparing only single-request target runs.
- Metadata must show compatible request length and table/page state for the target row.

Disallowed oracle:

- Identical prompts spread across multiple batch slots.
- Single target alone compared against the same target embedded in a multi-request batch.
- CUDA graph replay where a real bs=3 request set is padded to bucket 4 for SWA-boundary decode correctness.

TARGET 08.19 was not rerun as a pass/fail gate after this milestone because the prefix-disabled exact-path oracle is still not clean. Rerunning it before this remaining slot-invariance issue is fixed would make prefix-cache conclusions ambiguous.
