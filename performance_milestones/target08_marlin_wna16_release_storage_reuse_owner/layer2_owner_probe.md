# Layer2 Indexer And Attention Owner Probe

## Why Layer2 Was Probed

TARGET 08.36 found the first visible activation/logit divergence near
`layer2.indexer_select.logits`.  This milestone added owner-address and integrity sampling
around layer2 inputs, q/kv/indexer projections, indexer select tensors, attention outputs,
MoE input/output, final norm, and final logits.

Raw probes are embedded in:

- `raw/release_eager_ledger/marlin_wna16_owner_ledger_release_eager_ledger_rank0.jsonl`
- `raw/release_after_kv_ledger/marlin_wna16_owner_ledger_release_after_kv_ledger_rank0.jsonl`

## Probe Coverage

Each prefill/decode step records these layer2 owners when enabled:

- `embedding`
- `layer2.input`
- `layer2.attention_input`
- `layer2.wqa_output`
- `layer2.wkv_shared_activation_output`
- `layer2.q_lora_after_norm`
- `layer2.q_wqb_output`
- `layer2.wkv_output`
- `layer2.q_after_q_norm_rope`
- `layer2.kv_after_kv_norm_rope`
- `layer2.indexer_query_fp8_values`
- `layer2.indexer_query_fp8_weights`
- `layer2.indexer_output`
- `indexer_select.seq_lens`
- `indexer_select.page_table`
- `indexer_select.logits`
- `indexer_select.topk_raw_indices`
- `indexer_select.topk_page_indices`
- `indexer_select.topk_full_indices`
- `indexer_select.topk_lens`
- `indexer_select.topk_scores`
- `layer2.compressor_output`
- `layer2.merged_attention_output_before_wo`
- `layer2.merged_attention_output_after_inverse_rope`
- `layer2.final_attention_output`
- `layer2.attention_output`
- `layer2.moe_input`
- `layer2.moe_output`
- `final_norm`
- `lm_head_logits`

## Immediate Release Observations

Direct layer2 indexer select tensors do not overlap freed expert-weight ranges:

| Owner | Overlaps freed range? | Notes |
| --- | --- | --- |
| `dsv4.layer2_owner_probe.indexer_select.logits` | no | Contains sentinel `Inf`/mask values. |
| `dsv4.layer2_owner_probe.indexer_select.topk_scores` | no | Contains sentinel `Inf`/mask values. |
| `dsv4.layer2_owner_probe.indexer_select.topk_*indices` | no | No direct freed-range overlap. |
| `kvcache.dsv4.layer2.indexer_state.kv_score_buffer` | no | Layer2 indexer state itself did not overlap in rank 0. |
| `kvcache.dsv4.layer2.compress_state.kv_score_buffer` | yes | Allocated at `after_kv_alloc`; overlaps layer 19 `w2_weight_scale_inv`. |

Selected integrity samples:

| Run | Owner | Stage | Finite ratio | Sample abs max | Overlap? |
| --- | --- | --- | ---: | ---: | --- |
| Immediate | `indexer_select.logits` | prefill | 0.4215 | `Inf` | no |
| Immediate | `indexer_select.logits` | early decode | 0.8750 | `Inf` | no |
| Immediate | `indexer_select.logits` | later decode | 0.7778 | `NaN` | no |
| After KV | `indexer_select.logits` | prefill | 0.4215 | `Inf` | no |
| After KV | `indexer_select.logits` | decode | 0.8750 to 0.9630 | `Inf` | no |

The `Inf` values are therefore not sufficient to explain the corruption.  They are present
in passing runs too.

## Downstream Symptom

The immediate-release run shows bounded values for the first decode steps, then large or
zeroed downstream samples:

| Owner | Immediate-release late-decode symptom | After-KV behavior |
| --- | --- | --- |
| `layer2.attention_output` | sample abs max grows to about `3.65e37` and higher | remains bounded around `5` to `7` |
| `layer2.moe_output` | later samples collapse to zero checksum/zero abs max | remains bounded and non-zero |
| `final_norm` | later samples have zero abs max with non-zero checksum changes | remains bounded and non-zero |
| `lm_head_logits` | later samples collapse to zero checksum/zero abs max | remains bounded around `46` to `56` abs max |

## Conclusion

Layer2 is where the allocator-induced corruption becomes visible, not the first owner of the
freed storage.  The direct layer2 indexer select buffers do not reuse freed expert storage,
and their sentinel non-finite values also appear in passing runs.  The root owner is earlier:
DSV4 KV/component pools allocated at `after_kv_alloc`.
