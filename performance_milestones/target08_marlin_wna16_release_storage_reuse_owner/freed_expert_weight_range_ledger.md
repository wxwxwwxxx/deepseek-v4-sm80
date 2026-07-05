# Freed Expert-Weight Range Ledger

## Schema

Each released raw routed expert tensor is recorded before `delattr`:

| Field | Meaning |
| --- | --- |
| `rank` | TP rank writing the ledger. |
| `layer_id` | DeepSeek V4 decoder layer id. |
| `owner` | Marlin WNA16 routed expert owner label. |
| `component` | `w13_weight`, `w13_weight_scale_inv`, `w2_weight`, or `w2_weight_scale_inv`. |
| `data_ptr`, `start`, `end` | GPU address and byte range `[start,end)`. |
| `bytes` | Tensor storage bytes. |
| `dtype`, `shape`, `stride` | Tensor metadata at release time. |
| `released` | Whether the release path actually deleted the raw attribute. |

Raw ledgers:

- `raw/release_eager_ledger/marlin_wna16_freed_ranges_release_eager_ledger_rank*.jsonl`
- `raw/release_after_kv_ledger/marlin_wna16_freed_ranges_release_after_kv_ledger_rank*.jsonl`

## Totals

All ranks match.

| Ledger | Rows/rank | Layers | Tensors/layer | Bytes/rank | GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediate release | 172 | 43 | 4 | 18,396,217,344 | 17.1328125 |
| Release after KV allocation | 172 | 43 | 4 | 18,396,217,344 | 17.1328125 |

Component split:

| Component | Bytes/rank | GiB/rank |
| --- | ---: | ---: |
| `w13_weight` | 11,542,609,408 | 10.75 |
| `w13_weight_scale_inv` | 721,420,288 | 0.671875 |
| `w2_weight` | 5,771,304,704 | 5.375 |
| `w2_weight_scale_inv` | 360,710,144 | 0.3359375 |
| Total | 18,396,217,344 | 17.1328125 |

Per layer, the release is `427,819,008` bytes = `0.3984375 GiB`.

Using the TARGET 08.34 page-size 256 ledger, this byte recovery is equivalent to about
`400.10` DSV4 KV pages or `102,426` tokens per rank of theoretical headroom.

## Example Row

Rank 0, immediate-release ledger:

```json
{
  "rank": 0,
  "layer_id": 0,
  "component": "w13_weight",
  "data_ptr": 138433909489664,
  "start": 138433909489664,
  "end": 138434177925120,
  "bytes": 268435456,
  "dtype": "torch.int8",
  "shape": [256, 2, 256, 2048],
  "stride": [1048576, 524288, 2048, 1],
  "released": true
}
```

## Interpretation

The memory recovery is real and repeatable.  The correctness problem is not that release
failed to remove the raw tensors.  The problem is what the allocator does with the freed
ranges before DSV4 KV/component lifetimes have stabilized.
