# TARGET 07.41: Exact Indexer, Cache, Runtime Work

## Status

Complete.

TARGET 07.40 selected this target because the post-splitK exact profile moved
the dominant costs away from decode sparse attention.  The current best exact
variant remains:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

The reused TARGET 07.395 macro baseline is:

| Workload | Output tok/s |
| --- | ---: |
| 4096/128/bs4 | `38.9379` |
| 4096/1024/bs4 | `68.8097` |

The TARGET 07.40 rank0 node trace put the top exact-path repeat buckets at:

| Rank | Bucket | Repeat kernel s | Decode-envelope kernel s |
| ---: | --- | ---: | ---: |
| 1 | runtime/copy/cat/index kernels | `2.7523` | `1.8949` |
| 2 | legacy prefill/extend sparse attention | `2.1044` | `0.0000` |
| 3 | elementwise math graph nodes | `2.0827` | `1.4838` |
| 4 | indexer logits/topk/cache | `1.1973` | `0.2128` |
| 5 | FP8 projection GEMM | `1.1720` | `1.1720` |

Decode split-K gather/split/combine was only `0.1180 s` repeat kernel time, so
this target does not continue split-K sparse decode polish.

## Mini-vs-vLLM Comparison Checkpoint

The old vLLM Nsight repeat window is not a complete per-subgraph timing oracle.
It has incomplete child-process/repeat-window attribution, so this target uses
it only as weak supporting evidence.  vLLM macro numbers and code topology are
still useful:

- vLLM offline 4096/128/bs4: about `82.08 output tok/s`.
- vLLM offline 4096/1024/bs4: about `201.99 output tok/s`.
- vLLM sm80 uses `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, and FP8
  indexer cache.
- vLLM owns attention/indexer/cache buffers inside custom-op and V1 graph
  dispatcher boundaries.

| Mini 07.40 bucket | vLLM analogous design | Can compare time? | Adopt/adapt/reject |
| --- | --- | --- | --- |
| runtime/copy/cat/index graph nodes | persistent runner buffers, `CudagraphDispatcher`, custom-op-owned attention/indexer buffers | Usually no without a fresh complete vLLM node trace | Adapt the persistent-buffer/custom-op ownership idea to mini's exact bf16 metadata staging. |
| elementwise math graph nodes | fused attention/indexer/cache ops and compiled graph regions | Usually no | Defer. It is a broad fusion target and less isolated than replay metadata ownership. |
| legacy prefill sparse + indexer | sparse prefill/indexer path, but vLLM sm80 sparse prefill has known OOM risk in this environment | Partial/code only | Defer unless a fresh profile makes it the most actionable exact cut. |
| bf16 indexer logits/topk/cache | FP8 paged logits, persistent topk, FP8 indexer cache | Not directly; precision/layout changes dominate | Reject for this exact default unless the portable part is independent of FP8 cache/indexer layout. |

Designs that can be adapted now:

- persistent graph-owned metadata buffers;
- one custom-op/kernel boundary for replay-time metadata staging;
- bounded-shape decode replay specialization.

Designs rejected for this default exact target:

- packed `fp8_ds_mla` KV cache;
- FP8/FP4 indexer cache;
- direct vLLM `SparseAttnIndexer` adoption as the default path.

## Selected Exact Cut

Selected cut: graph-stable metadata/buffer ownership.

Mini already binds decode graph input buffers for `out_loc` and `positions`, but
`DSV4AttentionBackend.prepare_for_replay` still emits many small copy/fill
operations to stage 1D and 2D DSV4 metadata into captured graph buffers.  This
matches TARGET 07.40's largest exact bucket and does not change any attention,
indexer, cache, or precision math.

The implementation target is a single opt-in sm80 Triton replay-metadata copy
kernel:

- copies all fixed-shape decode 1D metadata;
- copies/fills the padded 2D page/index metadata;
- leaves graph-bound `raw_out_loc` and `positions` untouched when they are
  already owned by `GraphCaptureBuffer`;
- preserves the existing compressed-loc staging behavior.

Precision policy remains exact bf16 activation/cache plus Marlin WNA16 expert
weights.  No packed FP8 KV or FP8 indexer cache path is introduced.

Implemented opt-in gate:

```text
MINISGL_DSV4_SM80_REPLAY_METADATA_COPY=1
```

Registered variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Implementation summary:

- added a strict-layout Triton replay metadata copy kernel;
- added `dsv4_kernel.copy_decode_metadata_for_replay(...)` as the guarded
  wrapper;
- `DSV4AttentionBackend._copy_metadata_for_replay` tries the fused path first
  and falls back to the legacy copy/fill sequence if unsupported;
- added the `_metacopy` variant to the perf matrix and text smoke registry;
- added a focused sm80 CUDA correctness test for graph-bound and unbound
  `raw_out_loc`/`positions` modes.

## Results

### Correctness

Focused tests:

```bash
pytest -q -o addopts='' \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy \
  tests/kernel/test_deepseek_v4_wrappers.py::test_copy_decode_metadata_for_replay_matches_legacy_copy
```

Result: `12 passed in 9.42s`.

TP8 text smoke:

- artifact: `summaries/tp8_text_smoke_metacopy.json`;
- status: `pass`;
- page size: `256`;
- num pages: `128`;
- graph replay: enabled, `eager_decode_count=0`, `capture_compressed_locs_in_graph=true`;
- sample outputs were sane for the Chinese arithmetic, English sky-color, and
  Hangzhou prompts.

### Microbench

Artifact: `summaries/replay_metadata_copy_microbench.json`.

Shape approximates the bs4 4096/1024 decode graph staging boundary:

- rows: `4`;
- max seq len: `5120`;
- page size: `256`;
- graph input buffers already bound for `raw_out_loc` and `positions`.

| Path | Time / replay | Launches / replay | Delta |
| --- | ---: | ---: | ---: |
| legacy copy/fill staging | `0.2536 ms` | `18` | baseline |
| fused replay metadata copy | `0.1272 ms` | `1` | `1.99x`, `49.85%` lower |

The unbound-input check in
`summaries/replay_metadata_copy_microbench_unbound.json` was also faster:
`0.2928 ms -> 0.1269 ms`, `2.31x`.

### Macro

Baseline is the TARGET 07.40 reused TARGET 07.395 exact split-K macro for the
same TP8/page-size-256/num-pages-128 setup.

| Workload | Baseline output tok/s | Metacopy output tok/s | Delta |
| --- | ---: | ---: | ---: |
| 4096/128/bs4 | `38.9379` | `39.0028` | `+0.17%` |
| 4096/1024/bs4 | `68.8097` | `68.6314` | `-0.26%` |

Artifacts:

- `raw/dsv4_target0741_metacopy_4096x128_bs4_np128/summary.json`;
- `raw/dsv4_target0741_metacopy_4096x1024_bs4_np128/summary.json`;
- `summaries/target07_41_results_summary.json`.

No new Nsight capture was taken because macro improvement did not reach the
about-`5%` threshold.

## Next Decision

The cut clears the subgraph bar but not the macro bar.  Keep it as a validated
opt-in exact bf16 replay-staging cut, but do not promote it as the new best
exact macro variant.

Current best exact macro result remains the TARGET 07.40/07.395 baseline:

- 4096/128/bs4: `38.9379 output tok/s`;
- 4096/1024/bs4: `68.8097 output tok/s`.

Do not continue local replay-metadata copy polish unless a fresh profile shows
this bucket remains top-two after the fused copy.  The next step should be a
fresh profile/parity pass before more exact bf16 work, or TARGET 07.50 if the
new evidence points to packed FP8 KV/indexer cache/layout as the remaining
macro gap.
