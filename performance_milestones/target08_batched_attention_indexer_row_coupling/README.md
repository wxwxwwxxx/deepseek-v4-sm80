# TARGET 08.196: DSV4 batched attention/indexer row coupling

Date: 2026-07-03

## Verdict

08.196 should not be treated as a broad correctness green light for general
multi-request DSV4 batches. The 08.195 compressor fix is preserved, and the
bs=3 graph bucket issue now has an opt-in exact-batch guard, but the main
batched eager oracle still drifts.

The earliest semantic single-target-vs-slot drift in the quick oracle is still
`layer0.q_after_q_norm_rope` at max abs `0.0351562`. Earlier projection
checkpoints are small and within the `2e-2` activation tolerance:
`layer0.wqa_output=0.000976562`, `layer0.q_wqb_output=0.00195312`.

The sparse indexer itself is not the first owner in the quick oracle:
`layer2.indexer_select.topk_raw_indices` and lengths remain stable, while
physical page/full indices differ because the target occupies different cache
locations. The summary's "first divergent" row for identical rows includes
physical index tensors, so it must not be read as a semantic first-drift row.

## Code Changes

- Added opt-in DSV4 activation checkpoints around WQA/WKV, q/k norm-rope,
  indexer query quantization, compressor input windows, indexer selection, and
  merged attention outputs.
- Added opt-in attention/indexer debug tensors under
  `MINISGL_DSV4_PREFIX_DEBUG_ATTENTION_COMPONENTS=1`.
- Added `MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1`. When enabled, decode CUDA
  graph replay requires the real batch size to have a captured graph, instead
  of padding bs=3 into bucket4.
- Added `MINISGL_DSV4_FORCE_TORCH_TOPK=1` for future top-k bisection. It was
  not used as a fix because top-k raw indices are stable and the drift appears
  earlier.
- Extended the exact-path probe with row-coupling scenarios and added the
  08.196 runner/summarizer under this milestone directory.
- Added regression tests for per-request compressor windows and graph exact-bs
  behavior.

Default serving behavior is unchanged. The new behavior is behind explicit
debug/guard environment variables.

## Compressor Preservation

The 08.195 compressor fix is preserved. `compress_forward_fallback` and the
vectorized path continue to form compression windows from each request's own
position-contiguous rows. They do not flatten/pool rows across requests.

Regression:

```bash
pytest -q tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_compress_forward_keeps_request_contiguous_windows
```

Result: pass. The test covers C4 request length 5 and C128 request length 129,
and checks that the second request's compression window is request-local.

## Quick Reproduction

Primary quick run:

```bash
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  --mode eager \
  --output-dir performance_milestones/target08_batched_attention_indexer_row_coupling/raw/quick_eager \
  --model-path /models/DeepSeek-V4-Flash \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --probe-max-tokens 2 \
  --prelude-max-tokens 1 \
  --capture-activations \
  --debug-attention-components \
  --max-activation-rows 4 \
  --scenarios single_target_alone identical_prompts_batch target_in_batch_slot0
```

Summary:

```bash
python performance_milestones/target08_batched_attention_indexer_row_coupling/scripts/summarize_target08_196_row_coupling.py \
  --eager performance_milestones/target08_batched_attention_indexer_row_coupling/raw/quick_eager \
  --output-dir performance_milestones/target08_batched_attention_indexer_row_coupling/summaries/quick_eager \
  --atol 2e-2 --rtol 2e-2 --activation-atol 2e-2 --activation-rtol 2e-2
```

| Scenario | Prompt lens | Prefill bs | Decode source | Decode padded | Target table_idx |
| --- | --- | --- | --- | --- | --- |
| `single_target_alone` | 257 | 1 | eager | 1 | 15 |
| `identical_prompts_batch` | 257,257,257,257 | 4 | eager | 4 | 15 |
| `target_in_batch_slot0` | 257,257,257,257 | 4 | eager | 4 | 12 |

Logit result:

| Comparison | Result |
| --- | --- |
| identical prefill | FAIL, max abs `0.884253 / 0.883242 / 0.887962` |
| identical decode | FAIL, max abs `0.911032 / 0.940769 / 0.780247` |
| alone vs slot0 prefill | FAIL, max abs `1.13529` |
| alone vs slot0 decode | FAIL, max abs `0.904431` |

## Checkpoint Evidence

Representative quick-eager rows:

| Checkpoint | Identical rows | Alone vs slot0 |
| --- | --- | --- |
| `layer0.attention_input` | pass max `0` | pass max `0` |
| `layer0.wqa_output` | pass max `0` | pass max `0.000976562` |
| `layer0.q_wqb_output` | pass max `0` | pass max `0.00195312` |
| `layer0.q_after_q_norm_rope` | pass max `0` | FAIL max `0.0351562` |
| `layer0.final_attention_output` | pass max `0` | FAIL max `0.0625` |
| `layer2.indexer_compressor_input_window` | pass max `0` | pass max `0.0136719` |
| `layer2.indexer_output` | pass max `0` | pass max `0.00317383` |
| `layer2.indexer_select.topk_raw_indices` | pass max `0` | pass max `0` |
| `layer2.attention_backend.merged_attention_output_before_wo` | pass max `0.000488281` | pass max `0.0175781` |
| `layer3.q_after_q_norm_rope` | FAIL max `0.101562` | FAIL max `0.1875` |
| `layer3.attention_backend.c128_attention_output` | pass max `0.00976562` | pass max `0.015625` |

Artifact:
`performance_milestones/target08_batched_attention_indexer_row_coupling/summaries/quick_eager/attention_indexer_checkpoint_diff.md`

## Toggle Bisection

All quick toggle runs used the same three scenarios and the same tolerances.

| Run | Disable env | First semantic drift | Alone vs slot0 prefill | Alone vs slot0 decode | Conclusion |
| --- | --- | --- | --- | --- | --- |
| `quick_eager` | none | `layer0.q_after_q_norm_rope`, `0.0351562` | FAIL `1.13529` | FAIL `0.904431` | baseline fails |
| `quick_disable_q_wqb` | `MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE` | same | FAIL `1.13529` | FAIL `0.904431` | q_wqb BF16 cache not owner |
| `quick_disable_q_norm_rope` | `MINISGL_DSV4_SM80_Q_NORM_ROPE` | same | FAIL `1.13529` | FAIL `0.904431` | stale bisection for this path; fused q+kv path still writes q norm/rope |
| `quick_disable_fused_q_kv_norm_rope_store` | `MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE` | same | FAIL `1.6514` | FAIL `3.23674` | not a fix; worsens logits |

Earlier 08.195 toggles for sparse attention, fused QKV store, indexer fp8,
compress store, and fused WQA/WKV also did not clear the general batched drift.

## Graph Bucket Decision

The bs=3 graph issue is handled separately from the eager row-coupling result.

Guard:

```bash
MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1
```

Graph quick command:

```bash
MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1 torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  --mode graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir performance_milestones/target08_batched_attention_indexer_row_coupling/raw/quick_graph_exact_bs_guard \
  --model-path /models/DeepSeek-V4-Flash \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --probe-max-tokens 2 \
  --prelude-max-tokens 1 \
  --scenarios swa_boundary_127_128_129_bs3
```

Observed metadata:

| Field | Value |
| --- | --- |
| captured graph buckets | `[16, 8, 4, 2, 1]` |
| `exact_bs_only` | `true` |
| bs=3 decode source | `eager` |
| decode padded size | `3` |
| `replay_count` | `0` |
| `eager_decode_count` | `1` |

Unit regression:

```bash
pytest -q tests/engine/test_graph_runner.py
```

Result: pass. Default mode still accepts a bs=3 decode padded to bucket4;
exact-bs mode rejects it unless bucket3 is captured.

## Full 08.196 Script

A broader suite is available but was not run end to end in this pass:

```bash
performance_milestones/target08_batched_attention_indexer_row_coupling/scripts/run_target08_196_row_coupling_probe.sh
```

It runs the expanded eager scenarios plus graph mode with exact-bs guard and
buckets `[1,2,4,8,16]`.

## Verification Run

Commands completed in this pass:

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/engine/graph.py \
  python/minisgl/utils/dsv4_prefix_debug.py \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  performance_milestones/target08_batched_attention_indexer_row_coupling/scripts/summarize_target08_196_row_coupling.py

pytest -q \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_compress_forward_keeps_request_contiguous_windows \
  tests/engine/test_graph_runner.py

pytest -q \
  tests/kernel/test_deepseek_v4_wrappers.py::test_topk_transform_full_reports_lens_in_torch_fallback \
  tests/kernel/test_deepseek_v4_wrappers.py::test_indexer_bf16_query_logits_and_topk_are_fallback_clean \
  tests/attention/test_deepseek_v4_backend_metadata.py

git diff --check
```

Results: all pass.

08.19 and 08.10 were not rerun. Reason: the broad multi-request exact oracle is
still failing, so those larger serving/performance confirmations would not
change the 08.196 decision. The next meaningful run is the full 08.196 script
above after a real row-coupling fix candidate exists.

## Artifacts

- `raw/quick_eager/`
- `raw/quick_disable_q_wqb/`
- `raw/quick_disable_q_norm_rope/`
- `raw/quick_disable_fused_q_kv_norm_rope_store/`
- `raw/quick_graph_exact_bs_guard/`
- `summaries/quick_eager/`
- `summaries/quick_disable_q_wqb/`
- `summaries/quick_disable_q_norm_rope/`
- `summaries/quick_disable_fused_q_kv_norm_rope_store/`
- `summaries/quick_graph_exact_bs_guard/`

The milestone directory is about 2.8 GiB because the debug runs retain raw
activation tensors.

## Git Status During Run

Before this README was added:

```text
 M performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py
 M python/minisgl/attention/deepseek_v4.py
 M python/minisgl/engine/graph.py
 M python/minisgl/kernel/deepseek_v4.py
 M python/minisgl/models/deepseek_v4.py
 M tests/kernel/test_deepseek_v4_wrappers.py
?? performance_milestones/target08_batched_attention_indexer_row_coupling/
?? tests/engine/test_graph_runner.py
```

## 08.20 Decision

Do not continue 08.20 as a broad default-serving promotion or a general
multi-request exactness claim.

It is reasonable to continue only under the narrow oracle already known to be
stable enough for follow-up work: prefix cache disabled, single-target or
slot-pinned/page-normalized comparisons, and graph exact-bs guard enabled when
graph mode is used. General batched DSV4 equality remains an open 08.196
blocker.
