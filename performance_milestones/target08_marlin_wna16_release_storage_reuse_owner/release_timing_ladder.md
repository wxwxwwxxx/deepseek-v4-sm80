# Release Timing Ladder

All runs use TP8, page size `256`, `--num-pages 128`, `--max-tokens 8`, and
`dsv4_sm80_a100_victory_marlin_release`.

| Timing | Raw artifact | Graph | Status | Released bytes | Captured buckets | Replay / eager | Prompt 0 token ids |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| `model_prepare` | `raw/text_smoke_release_eager_ledger.dsv4_sm80_a100_victory_marlin_release.json` | disabled | warn | 18,396,217,344 | `[]` | 0 / 7 | `[20, 940, 223, 0, 0, 0, 0, 0]` |
| `after_kv_alloc` | `raw/text_smoke_release_after_kv_ledger.dsv4_sm80_a100_victory_marlin_release.json` | disabled | pass | 18,396,217,344 | `[]` | 0 / 7 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| `after_kv_alloc` | `raw/text_smoke_release_after_kv_graph.dsv4_sm80_a100_victory_marlin_release.json` | enabled | pass | 18,396,217,344 | `[16,8,4,2,1]` | 7 / 0 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| `before_warmup_forward` | `raw/text_smoke_release_before_warmup_graph.dsv4_sm80_a100_victory_marlin_release.json` | enabled | pass | 18,396,217,344 | `[16,8,4,2,1]` | 7 / 0 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| `after_warmup_forward` | `raw/text_smoke_release_after_warmup_graph.dsv4_sm80_a100_victory_marlin_release.json` | enabled | pass | 18,396,217,344 | `[16,8,4,2,1]` | 7 / 0 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| `after_graph_capture` | `raw/text_smoke_release_after_graph_capture.dsv4_sm80_a100_victory_marlin_release.json` | enabled | pass | 18,396,217,344 | `[16,8,4,2,1]` | 7 / 0 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |
| `after_first_decode` | `raw/text_smoke_release_after_first_decode_graph.dsv4_sm80_a100_victory_marlin_release.json` | enabled | pass | 18,396,217,344 | `[16,8,4,2,1]` | 7 / 0 | `[20, 940, 223, 20, 223, 15120, 223, 22]` |

## Interpretation

The ladder has a sharp boundary:

- release before KV allocation fails;
- release after KV allocation passes in eager/no-graph;
- release after KV allocation also passes with CUDA graph capture and replay;
- all later release timings pass.

This means the failure is not graph capture itself, warmup forward itself, or a required
runtime read of raw expert weights.  The unsafe interval is specifically between raw expert
storage release and DSV4 KV/component allocation.

The `before_warmup_forward` and `after_warmup_forward` graph runs show negative
`capture_memory_delta_bytes` because the delayed release happens inside graph setup and
increases free memory during capture.  That is expected for the timing probe.

## First Divergence Context

TARGET 08.36 found first token divergence at `decode_step_3`, where release produced token
`0` instead of the expected continuation and logits collapsed.  This milestone reproduces
the user-visible failure in the immediate-release text smoke: prompt 0 emits token id `0`
for the remaining generated positions after the first plausible tokens.
