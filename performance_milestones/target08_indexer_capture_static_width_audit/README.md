# TARGET 08.33 DSV4 Indexer Capture Static-Width Audit

## Result

The DSV4 C4 indexer capture width is **not incorrectly expanded by 256x** in
the measured full-model bs16 graph bucket.

The relevant indexer table is page-based. For the audited `--page-size 256`,
the indexer logits path receives `page_size=64` (`256 / 4`) and
`page_table.shape[1]=128`, so the current static width is:

```text
128 pages * 64 C4/indexer slots per page = 8192 C4 slots
```

That is the expected C4 compressed width for a 32768-token table, and matches
the SGLang `PagedIndexerMetadata.max_c4_seq_len` contract.

## Artifacts

- Commands: `COMMANDS.md`
- Probe script: `scripts/capture_width_probe.py`
- Raw capture reports:
  - `raw/capture_width_probe_current_bs16_real.json`
  - `raw/capture_width_probe_table_width_bs16_real.json`
- Indexer width JSONL:
  - `raw/indexer_capture_width_current_bs16_real_rank*.jsonl`
  - `raw/indexer_capture_width_table_width_bs16_real_rank*.jsonl`
- Graph stage ledger JSONL:
  - `raw/graph_capture_stage_current_bs16_real_rank*.jsonl`
  - `raw/graph_capture_stage_table_width_bs16_real_rank*.jsonl`
- Text smoke:
  - `raw/text_smoke_prefix_routeb_current_bs16.json`

## Why This Was Narrow

TARGET 08.32 left the first-graph `~18.8-19.0 GiB/rank` CUDA graph cost
unattributed after ruling out several synthetic owners. This target checked one
specific high-impact hypothesis: mini might pass a token-slot-wide table into
the indexer logits path, then multiply that width by `page_size` again.

That hypothesis would have produced a very large dense FP32 logits workspace.
The real captured table is not token-slot-wide.

## Page-Table Semantics

| Table | Shape in audited bs16 capture | Width means | Entries mean | Divided by page size? | Consumer page size |
| --- | ---: | --- | --- | --- | ---: |
| Global engine `ctx.page_table` | `[17, 32768]` | token slots | raw token locations in global KV/cache address space | no | global `256` |
| Decode metadata `core.page_table` | `[16, 128]` | logical pages | physical full-page ids | yes, from global raw locs / `256` | attention uses `256` |
| `c4_page_table` | `[16, 128]` | full logical pages | physical C4 component page ids | already page ids | C4 `64` slots/page |
| `c128_page_table` | `[16, 128]` | full logical pages | physical C128 component page ids | already page ids | C128 `2` slots/page |
| `c4_indexer_page_table` | `[16, 128]` | full logical pages | physical C4 indexer component page ids | already page ids | indexer `64` slots/page |
| Component page-table cache | rows = global table rows, width >= `ceil(global_width / 256)` | full logical pages | cached component page ids by request table slot | already page ids | selected into component tables |
| Capture dummy table | global dummy row `[32768]`, metadata `[16, 128]` | global: token slots; metadata: pages | dummy raw locs globally, page ids in metadata | metadata is page ids | indexer `64` |

SGLang reference: `PagedIndexerMetadata.max_seq_len =
page_table.shape[1] * page_size`, and `max_c4_seq_len =
page_table.shape[1] * c4_page_size`. Mini's captured indexer path is using the
second form.

## Real Captured Indexer Width

All ranks matched. Rank 0 representative:

| Field | Value |
| --- | ---: |
| C4/indexer calls | `21` layers (`2, 4, ..., 42`) |
| backend | `triton_fp8_paged_vllm` |
| rows | `16` |
| `q.shape` | `[16, 64, 128]` |
| `cache.shape` | `[129, 8448]` |
| `page_table.shape` | `[16, 128]` |
| indexer `page_size` | `64` |
| current static width | `8192` |
| FP32 logits per call | `16 * 8192 * 4 = 524288 bytes` (`0.5 MiB`) |
| projected repeated logits per rank | `10.5 MiB` (`0.0103 GiB`) |

The JSONL intentionally does not call `.item()` on CUDA `seq_lens` during graph
capture. Those fields are marked `not_read_during_cuda_graph_capture` to avoid
making the diagnostic itself break capture.

## Stage Memory Ledger

Rank 0, current-width bs16:

| Stage | Free delta from previous GiB | Allocated delta GiB | Reserved delta GiB |
| --- | ---: | ---: | ---: |
| after `attn_backend.init_capture_graph` | `0.0000` | `0.0002` | `0.0000` |
| after `GraphCaptureBuffer.init` | `0.0000` | `0.0077` | `0.0000` |
| after `prepare_for_capture` | `0.0000` | `0.0000` | `0.0000` |
| after `stage_capture_metadata` | `0.0000` | `0.0000` | `0.0000` |
| after warmup `model.forward()` | `18.7637` | `17.8126` | `18.4160` |
| after actual `torch.cuda.graph` capture | `-0.1777` | `0.0000` | `-0.2246` |
| after `gc` / `empty_cache` sanity | `0.0000` | `0.0000` | `0.0000` |

The large cost appears during the warmup full-model forward outside the
`torch.cuda.graph` context, not inside the measured graph context itself.

## Width A/B

Single bucket only, as required: `--cuda-graph-bs 16`, TP8, real checkpoint,
page size `256`, `num_pages=128`, max seq len `32768`, Route B component tables
enabled.

| Mode | Selected indexer width | Projected selected dense logits / rank | Graph free delta / rank |
| --- | ---: | ---: | ---: |
| `current` | `8192` | `10.5 MiB` | `18.5859 GiB` |
| `table_width` counterfactual | `128` | `0.164 MiB` | `18.5859 GiB` |

The A/B difference was `0.0000 GiB/rank` at this measurement precision. Even
the projection from eliminating nearly all dense indexer logits is only about
`10.3 MiB/rank`, roughly `0.56` DSV4 KV pages or `~143` KV tokens at the
audited page size.

`table_width` is not a safe fix: the table is page-based, so using only `128`
would truncate valid C4/indexer positions for real replay.

## Validation

Default/current path:

| Gate | Result |
| --- | --- |
| `python -m py_compile` on modified files and probe script | pass |
| capture-only current bs16 | pass |
| capture-only `table_width` counterfactual bs16 | pass for capture-only diagnostic |
| prefix Route B text smoke, current width | pass |
| graph replay zero-eager in text smoke | `replay_count=9`, `greedy_sample_replay_count=9`, `eager_decode_count=0`, padded bs `16` |

No behavior fix was promoted. The only behavior-changing mode is opt-in and
diagnostic.

## Answer

1. Is indexer capture width incorrectly enlarged?

No. The captured indexer page table width is `128` logical pages, and the
indexer page size is `64` compressed slots. The current `128 * 64 = 8192`
static width is semantically correct for C4/indexer logits in this bucket.

2. How much graph private-pool cost can it explain?

At most about `0.010 GiB/rank` by projection, and `0.000 GiB/rank` in the
single-bucket full-model A/B. It does not explain the `~18.6-19.0 GiB/rank`
first-graph cost.

3. Is there a safe fix?

No width fix should be promoted. `table_width` would be unsafe because the
incoming table is page-based, not token-slot-based. The instrumentation remains
useful and is fully opt-in.

4. If not this, what is the next minimal evidence target?

The next target should instrument the warmup `model.forward()` owner path,
because the stage ledger places the `~18.6 GiB/rank` movement there. The
smallest useful next probe is a real-module, single-bucket owner ledger around
full-model warmup forward: embedding, per-layer attention/indexer, MoE/shared
expert, projection caches/workspaces, and communication boundaries, recording
retained allocated/reserved deltas after each owner.
