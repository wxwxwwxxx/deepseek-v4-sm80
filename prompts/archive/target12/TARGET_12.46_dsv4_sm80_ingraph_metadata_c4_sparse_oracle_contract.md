# TARGET 12.46: DSV4 SM80 In-Graph Metadata C4 Sparse Oracle Contract

## Status

Active child target under:

```text
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

This target follows:

```text
performance_milestones/target12_ingraph_metadata_promotion_soak/README.md
performance_milestones/target12_sglang_in_graph_metadata_prep/README.md
```

TARGET 12.45 showed the in-graph metadata opt-in is performance-positive and
repeat-stable, but default promotion is blocked by one oracle failure:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH oracle mismatch
for c4_sparse_raw_indices over active rows=4
```

The failure appears in the long-context `historical_4096_128_bs4` oracle run,
while text smoke and text oracle pass.  This target is a focused correctness
contract fix for that boundary.

## Current Evidence

TARGET 12.45 performance evidence was good:

```text
short repeat median:
  output tok/s  +5.68%
  decode tok/s +11.37%

four-scenario soak:
  historical_4096_128_bs4   output +6.78%
  historical_4096_1024_bs4  output +16.77%
  serving_mixed_112req      output +22.22%
  prefix_multi_112req       output +25.22%
```

Invariants stayed clean:

```text
zero eager decode in supported scenarios
communication calls/bytes unchanged
wrapper counters unchanged
prefix saved tokens unchanged
capture/private-pool memory unchanged
independent SWA fallback explicit and safe
```

Promotion blocker:

```text
opt-in text smoke: pass
opt-in text smoke plus oracle: pass
opt-in short historical oracle: fail

failure path:
  DSV4AttentionBackend.validate_after_replay
    -> _compare_prep_metadata_in_graph_oracle
    -> eq_2d_active("c4_sparse_raw_indices", got.c4_sparse_topk_lengths[:rows])
```

Interpretation: this is not a performance blocker.  It is a correctness/contract
blocker around long-context C4 sparse raw-index semantics.

## Goal

Find and fix, or precisely narrow, the `c4_sparse_raw_indices` long-context
oracle mismatch.

Answer:

1. Is the mismatch caused by the in-graph Triton materialization formula?
2. Is the mismatch caused by the oracle comparing the wrong time boundary
   before/after graph replay and indexer mutation?
3. Is the old path's `c4_sparse_raw_indices` expected value actually produced
   by indexer/topk selection rather than contiguous tail raw indices?
4. Does `materialized_seq_lens` / capped sequence length differ between old and
   in-graph paths at 4096-token context?
5. Does the active-width comparator use the right length source for
   `c4_sparse_raw_indices`?
6. After the fix, does `MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1`
   pass both text smoke and `historical_4096_128_bs4`?

## Non-Goals

- Do not default-promote in-graph metadata.
- Do not optimize raw copy or residual owners.
- Do not implement TARGET 12.5 direct/fused writers.
- Do not change MoE, communication, low precision, scheduler, sampling, or MTP.
- Do not broaden independent SWA support.
- Do not weaken the oracle just to make it pass.  If the comparator is wrong,
  replace it with the correct boundary check and document why.

## Required Inputs

Read:

```text
prompts/target.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
prompts/archive/target12/TARGET_12.4_dsv4_sm80_sglang_in_graph_metadata_prep.md
prompts/archive/target12/TARGET_12.45_dsv4_sm80_ingraph_metadata_promotion_soak.md
performance_milestones/target12_ingraph_metadata_promotion_soak/README.md
performance_milestones/target12_sglang_in_graph_metadata_prep/README.md
```

Inspect:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/kernel/test_deepseek_v4_wrappers.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

Useful code anchors:

```text
DSV4AttentionBackend.validate_after_replay()
DSV4AttentionBackend._compare_prep_metadata_in_graph_oracle()
DSV4AttentionBackend._stage_prep_metadata_in_graph()
DSV4AttentionBackend._clamp_graph_replay_compressed_read_metadata()
DSV4AttentionBackend._merge_indexer_rows_in_place()
DSV4AttentionBackend._remap_indexer_topk_for_attention()
triton/deepseek_v4.py::_prep_decode_metadata_in_graph_kernel
kernel/deepseek_v4.py::prep_decode_metadata_in_graph()
```

SGLang reference if needed:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
```

## Hypotheses To Test

Test these explicitly; do not assume the first one is true.

### H1: In-Graph Kernel Formula Bug

The in-graph kernel currently derives C4 sparse raw indices from a contiguous
tail formula:

```text
c4_len = capped_seq_len // 4
c4_sparse_len = min(max(c4_len, 0), index_topk)
c4_start = max(c4_len - index_topk, 0)
c4_raw = c4_start + offsets
```

The old path may instead expect actual indexer-selected sparse raw indices after
top-k, remapping, or layer forward mutation.  If so, the in-graph kernel may be
materializing a valid initial C4 read surface but not the same
`c4_sparse_raw_indices` field the oracle compares after replay.

### H2: Oracle Boundary Bug

The oracle compares after `g.replay()`.  The captured forward may update
`core.c4_sparse_raw_indices` through indexer/top-k code during graph replay,
while `oracle_metadata` may represent an out-of-graph pre-forward state.  If the
two sides are from different time boundaries, the oracle should be split into:

```text
pre-forward metadata materialization fields
post-forward/indexer-mutated fields
```

Do not remove the check; move each field to the correct boundary.

### H3: Materialized Seq-Len / Clamp Contract Mismatch

The long-context case may differ because `materialized_seq_lens` caps the
compressed read length differently from `positions + 1`.  Compare:

```text
positions
seq_lens
materialized_seq_lens
c4_topk_lengths_raw
c4_topk_lengths_clamp1
c4_sparse_topk_lengths
expected/got c4_sparse_raw_indices active prefix
```

for the failing rows.

### H4: Active Width Comparator Bug

The oracle uses `got.c4_sparse_topk_lengths[:rows]` as active width.  Confirm
that this is the correct width for both sides after graph replay.  If expected
and got lengths can differ, report the length mismatch first and avoid comparing
raw indices under a mismatched active-width assumption.

## Required Work

### 1. Add Minimal Mismatch Diagnostics

Add a guarded diagnostic mode if needed, for example:

```text
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE_DEBUG=1
```

The diagnostic should record a tiny summary, not full tensors:

```text
rank
decode step or replay index if available
row
position
seq_len
materialized_seq_len
c4_topk_lengths_raw
c4_sparse_topk_lengths
expected first N c4_sparse_raw_indices
got first N c4_sparse_raw_indices
expected/got c4_sparse_page/full first N if relevant
whether capture was before or after graph replay
```

Keep diagnostics default-off and avoid per-token logging in normal runs.

### 2. Build A Focused Unit Or Synthetic Probe

Prefer a small test or probe that does not require a full model load, if
possible:

- call `prep_decode_metadata_in_graph()` with synthetic 4096-like rows;
- compare against the old Python/Triton metadata construction contract;
- include component page tables and materialized sequence lengths;
- cover the failing long-context shape and a small text-smoke-like shape.

If no-weight isolation is too expensive, use the short historical oracle run
as the primary repro and keep additional instrumentation minimal.

### 3. Identify The Correct Contract

Classify every compared C4 field:

```text
c4_topk_lengths_raw
c4_topk_lengths_clamp1
c4_sparse_topk_lengths
c4_sparse_raw_indices
c4_sparse_page_indices
c4_sparse_full_indices
c4_out_loc
c4_indexer_out_loc
```

as one of:

```text
raw in-graph materialization output before forward
field mutated by captured indexer/top-k during forward
field copied from prefix/component ownership state
field whose old-path oracle value must be transformed to replay contract form
```

Then fix the kernel, oracle, or staging code according to that classification.

### 4. Fix The Narrow Bug

Allowed fixes:

- correct `_prep_decode_metadata_in_graph_kernel` C4 raw-index formula;
- correct active width/length handling in oracle;
- compare `c4_sparse_raw_indices` at the correct pre/post-forward boundary;
- store a pre-forward oracle snapshot for materialized fields if needed;
- adjust expected value generation to match the replay fixed-address contract,
  as TARGET 12.4 already did for component write locs.

Do not silently skip `c4_sparse_raw_indices`.  If it is not a valid field to
compare at `validate_after_replay`, replace it with an equivalent valid check
and explain the boundary.

## Validation

Run focused correctness gates:

```bash
python -m py_compile \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
python -m pytest -q tests/kernel/test_deepseek_v4_wrappers.py
```

Run text smoke oracle:

```bash
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_46_text_smoke_ingraph_oracle.json
```

Run the failing short historical oracle:

```bash
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --output-dir /tmp/dsv4_target12_46_short_oracle_ingraph \
  --keep-going
```

After the oracle passes, run one non-oracle short pair to confirm performance
was not destroyed:

```text
historical_4096_128_bs4
baseline vs in-graph metadata
```

Do not rerun the full four-scenario soak unless the fix changes runtime
behavior beyond oracle/comparator logic or kernel C4 materialization.

Run `git diff --check`.

## Stop Conditions

Stop when one is true:

1. Text oracle and short historical oracle pass, and the report explains the
   exact contract fix.
2. The mismatch is proven to be a real in-graph kernel bug but needs a larger
   C4/indexer redesign; write the next target with exact evidence.
3. The mismatch is proven to be an oracle boundary bug and is fixed without
   weakening correctness coverage.
4. The opt-in must be narrowed or disabled for long-context C4 sparse attention;
   document the support boundary and keep default off.

Do not spend this target optimizing residual `raw_graph_copy`.

## Deliverables

Create:

```text
performance_milestones/target12_c4_sparse_oracle_contract/README.md
```

The README must include:

- git commit and dirty-state summary;
- failure reproduction command and result;
- mismatch diagnostic summary;
- hypothesis classification;
- exact contract decision for `c4_sparse_raw_indices`;
- code changes;
- correctness gate results;
- text oracle result;
- short historical oracle result;
- short non-oracle performance sanity if runtime behavior changed;
- recommendation: return to TARGET 12.45 promotion subset, proceed to TARGET
  12.5, keep opt-in with narrowed support, or no-go.
