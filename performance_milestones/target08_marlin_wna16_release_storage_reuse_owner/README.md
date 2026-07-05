# TARGET 08.37: Marlin WNA16 Release Storage Reuse Owner

## Verdict

The unsafe owner is the DSV4 KV/component-cache allocation phase that runs after immediate
`model_prepare` release.  With early physical release, the CUDA allocator immediately reuses
raw routed expert weight ranges for DSV4 KV/component pools, especially `c4_buffer`,
`c4_indexer_buffer`, `c4_indexer_fp8_paged_cache`, `c128_buffer`, and per-layer
`compress_state` / `indexer_state` buffers.  That allocator layout later produces the same
token-0 text collapse seen in TARGET 08.36.

Do not promote the original pre-KV release preset as-is.  It really releases
`18,396,217,344` bytes = `17.1328125 GiB/rank`, but the earliest passing boundary observed
is `after_kv_alloc`, not `model_prepare`.  `after_kv_alloc` still releases the full raw
expert storage before warmup/graph capture and passes eager and graph text smoke, but it
does not provide the original pre-KV capacity-planning headroom.

## Evidence At A Glance

| Run | Release timing | Graph | Status | Generated token 0 ids | Replay / eager |
| --- | --- | --- | --- | --- | --- |
| `text_smoke_release_eager_ledger` | `model_prepare` | disabled | warn | `[20, 940, 223, 0, 0, 0, 0, 0]` | 0 / 7 |
| `text_smoke_release_after_kv_ledger` | `after_kv_alloc` | disabled | pass | `[20, 940, 223, 20, 223, 15120, 223, 22]` | 0 / 7 |
| `text_smoke_release_after_kv_graph` | `after_kv_alloc` | `[1,2,4,8,16]` | pass | `[20, 940, 223, 20, 223, 15120, 223, 22]` | 7 / 0 |
| `text_smoke_release_before_warmup_graph` | `before_warmup_forward` | `[1,2,4,8,16]` | pass | `[20, 940, 223, 20, 223, 15120, 223, 22]` | 7 / 0 |
| `text_smoke_release_after_warmup_graph` | `after_warmup_forward` | `[1,2,4,8,16]` | pass | `[20, 940, 223, 20, 223, 15120, 223, 22]` | 7 / 0 |
| `text_smoke_release_after_graph_capture` | `after_graph_capture` | `[1,2,4,8,16]` | pass | `[20, 940, 223, 20, 223, 15120, 223, 22]` | 7 / 0 |
| `text_smoke_release_after_first_decode_graph` | `after_first_decode` | `[1,2,4,8,16]` | pass | `[20, 940, 223, 20, 223, 15120, 223, 22]` | 7 / 0 |

All timing runs released the same full raw expert payload when release occurred:
`18,396,217,344` bytes across `43` layers, `172` tensors per rank.

## Reuse Owner

Rank-local ledgers are consistent across all 8 ranks.

| Ledger | Freed rows/rank | Freed bytes/rank | Owner rows/rank | Overlap rows/rank |
| --- | ---: | ---: | ---: | ---: |
| Immediate release | 172 | 18,396,217,344 | 6,170 | 147 |
| Release after KV allocation | 172 | 18,396,217,344 | 6,258 | 17 |

Immediate release overlap stages:

| Stage | Overlap rows |
| --- | ---: |
| `after_kv_alloc` | 65 |
| `after_graph_runner_init` | 65 |
| `decode_bs3_padded3` | 7 |
| `decode_bs3_padded3_eager` | 7 |
| `prefill_bs3_padded3` | 2 |
| `prefill_bs3_padded3_eager` | 1 |

The decisive rows are the first `after_kv_alloc` overlaps.  Examples from rank 0:

| Owner | Tensor range | Reused freed range | Overlap bytes |
| --- | --- | --- | ---: |
| `kvcache.dsv4.c4_buffer` | `[138428775661568,138428953198592)` | layer 32 `w13_weight` | 43,319,296 |
| `kvcache.dsv4.c128_buffer` | `[138430893785088,138430899068928)` | layer 5 `w2_weight_scale_inv` | 5,283,840 |
| `kvcache.dsv4.c4_indexer_buffer` | `[138428953198592,138428997582848)` | layer 32 `w13_weight` | 44,384,256 |
| `kvcache.dsv4.c4_indexer_fp8_paged_cache` | `[138428742107136,138428764992768)` | layer 32 `w13_weight_scale_inv` | 14,497,024 |
| `kvcache.dsv4.layer2.compress_state.kv_score_buffer` | `[138432839942144,138432845185024)` | layer 19 `w2_weight_scale_inv` | 5,242,880 |

After delaying release until `after_kv_alloc`, there are no KV/component pool overlaps left.
The remaining 17 overlaps are transient forward/logits allocations that are also present in
passing runs.

## Poison And Quarantine

| Run | Pattern | Status | Meaning |
| --- | --- | --- | --- |
| Hidden ref poison | zero | pass | Keeping storage alive but overwriting raw tensors does not break text. |
| Hidden ref poison | NaN for floating tensors, zero for non-floating tensors | pass | Raw scale/weight contents are not read after prebuild in this smoke. |
| Freed-block quarantine | all released bytes, zero | pass | Preventing allocator reuse removes the failure. |
| Freed-block quarantine | 6.375 GiB/rank, zero | pass | Protecting a subset is enough for the smoke. |
| Freed-block quarantine | 3.1875 GiB/rank, zero | pass | Smaller requested pressure point also protects this smoke. |
| Freed-block quarantine | 3.1875 GiB/rank, deterministic | pass | The result depends on holding blocks, not on zero values. |

The quarantine result points at allocator storage reuse, not stale raw-weight reads.

## Layer2 Probe

Layer2 is a visible symptom boundary, not the root owner.  The direct
`layer2.indexer_select` tensors do not overlap freed expert-weight ranges in the failure
run.  The indexer logits/top-k tensors contain mask/sentinel `Inf` values even in the
passing `after_kv_alloc` run, so finite ratio alone is not the failure criterion.

What changes in the failing immediate-release run is downstream magnitude collapse:
layer2 attention output becomes huge around the third decode, and later `moe_output`,
`final_norm`, and `lm_head_logits` samples collapse to zeros.  Delaying release until
`after_kv_alloc` keeps these activation samples bounded and text sanity passes.

## Files

- `freed_expert_weight_range_ledger.md`: released tensor schema and byte totals.
- `owner_reuse_ledger.md`: owner-tagged overlap counts and examples.
- `release_timing_ladder.md`: timing ladder table and interpretation.
- `poison_quarantine.md`: poison/quarantine matrix.
- `layer2_owner_probe.md`: layer2/indexer probe details.
- `minimal_reproducer.md`: smallest deterministic repro used here.
- `code_changes_tests.md`: code changes and verification.
- `COMMANDS.md`: commands used to generate the raw artifacts.
- `raw/`: JSON, JSONL, and log artifacts.

## Final Recommendation

No-go for immediate `model_prepare` physical release as the default release preset.

The next viable path is an allocator/lifetime redesign that either reserves KV/component
owners before physical release, gives DSV4 KV/component pools their own persistent arena, or
promotes a clearly documented `after_kv_alloc` release mode as a narrowed memory-saving
policy.  The narrowed policy is correctness-clean in these smokes, but it is not the
original pre-KV capacity win.
