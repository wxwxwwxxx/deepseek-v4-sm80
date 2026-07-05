# Owner-Tagged Allocation And Reuse Ledger

## Raw Ledgers

- Immediate release:
  `raw/release_eager_ledger/marlin_wna16_owner_ledger_release_eager_ledger_rank*.jsonl`
- Release after KV allocation:
  `raw/release_after_kv_ledger/marlin_wna16_owner_ledger_release_after_kv_ledger_rank*.jsonl`

Each row records:

- `owner`
- `stage`
- tensor summary with `data_ptr`, `start`, `end`, `bytes`, `dtype`, `shape`, `stride`
- `overlaps_freed_range`
- `overlap_freed_range`
- `nearest_freed_range`
- optional integrity sample fields for layer2 probes

## Rank-Consistent Counts

| Ledger | Owner rows/rank | Overlap rows/rank |
| --- | ---: | ---: |
| Immediate release | 6,170 | 147 |
| Release after KV allocation | 6,258 | 17 |

## Immediate Release Overlap Owners

Top rank-0 overlap owners:

| Owner | Rows |
| --- | ---: |
| `dsv4.layer2_owner_probe.lm_head_logits` | 8 |
| `engine.forward.logits` | 8 |
| `kvcache.dsv4.c4_buffer` | 2 |
| `kvcache.dsv4.c128_buffer` | 2 |
| `kvcache.dsv4.c4_indexer_buffer` | 2 |
| `kvcache.dsv4.c4_indexer_fp8_paged_cache` | 2 |
| `kvcache.dsv4.layer2.compress_state.kv_score_buffer` | 2 |
| `kvcache.dsv4.layer3..layer42.compress_state.kv_score_buffer` | many, 2 each |

Top overlap stages:

| Stage | Rows |
| --- | ---: |
| `after_kv_alloc` | 65 |
| `after_graph_runner_init` | 65 |
| `decode_bs3_padded3` | 7 |
| `decode_bs3_padded3_eager` | 7 |
| `prefill_bs3_padded3` | 2 |
| `prefill_bs3_padded3_eager` | 1 |

The `after_graph_runner_init` rows are repeated observations of the same KV/component
allocations, not the first allocator owner.  The first owner is already visible at
`after_kv_alloc`.

## Exact Rank-0 Examples

| Stage | Owner | Tensor dtype/shape | Tensor range | Freed range reused | Overlap bytes |
| --- | --- | --- | --- | --- | ---: |
| `after_kv_alloc` | `kvcache.dsv4.c4_buffer` | `torch.bfloat16 [21,8256,512]` | `[138428775661568,138428953198592)` | layer 32 `w13_weight` | 43,319,296 |
| `after_kv_alloc` | `kvcache.dsv4.c128_buffer` | `torch.bfloat16 [20,258,512]` | `[138430893785088,138430899068928)` | layer 5 `w2_weight_scale_inv` | 5,283,840 |
| `after_kv_alloc` | `kvcache.dsv4.c4_indexer_buffer` | `torch.bfloat16 [21,8256,128]` | `[138428953198592,138428997582848)` | layer 32 `w13_weight` | 44,384,256 |
| `after_kv_alloc` | `kvcache.dsv4.c4_indexer_fp8_paged_cache` | `torch.uint8 [21,129,8448]` | `[138428742107136,138428764992768)` | layer 32 `w13_weight_scale_inv` | 14,497,024 |
| `after_kv_alloc` | `kvcache.dsv4.layer2.compress_state.kv_score_buffer` | `torch.bfloat16 [1280,2048]` | `[138432839942144,138432845185024)` | layer 19 `w2_weight_scale_inv` | 5,242,880 |

## After-KV Release

After delaying physical release until after KV allocation:

- `after_kv_alloc` overlap rows disappear.
- `after_graph_runner_init` overlap rows disappear.
- remaining overlaps are transient forward/logits tensors:
  - `dsv4.layer2_owner_probe.lm_head_logits`
  - `engine.forward.logits`
  - one prefill `layer2.input` buffer

Those remaining transient overlaps also occur in passing runs and are not the root allocator
owner.

## Conclusion

The root storage-reuse owner is DSV4 KV/component allocation after early release.  Layer2
and logits owners expose the bad allocator state later, but the first high-risk owner is
already present before warmup and before graph capture.
