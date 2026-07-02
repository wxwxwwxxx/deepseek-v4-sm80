# TARGET 07.64: DSV4 SM80 Decode Metadata Deforestation

Date: 2026-07-02

## Decision

Conclusion: **keep opt-in, do not promote**.

The implemented helper is correct, graph-replay compatible, and has no default
runtime effect unless explicitly enabled. It does not meet the TARGET 07.64
performance gate:

- `graph_runtime_copy_cat_index` in the 4096/128 decode-forward envelope moved
  from `0.846795s` to `0.834792s`, a `0.012003s` reduction.
- 4096/1024/batch4 output throughput moved from `119.4153` to `122.9414`
  output tok/s, a `2.95%` gain. The 5% gate would require `125.3861`.

Keep the helper as an opt-in ablation under
`MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`. Do not include it in the
`dsv4_sm80_a100_victory` bundle yet.

Exactly one next target: **TARGET 07.65 should be direct-copy owner attribution
only, adding finer NVTX/source attribution around graph-replay `direct_copy`
under `batch_forward` and `batch_forward_enqueue` before any implementation.**

## Artifacts

- Baseline env expansion: `raw/variant_env.json`
- Baseline metadata split from 07.63 sqlite:
  `summaries/baseline_decode_metadata_subboundary.md`
- Microbench/oracle:
  `scripts/microbench_decode_metadata_deforest.py`,
  `summaries/decode_metadata_deforest_microbench.md`
- Text smoke:
  `raw/text_smoke_dsv4_sm80_a100_victory_metadatadeforest.json`
- Macro summaries:
  `summaries/macro_4096x128_bs4_np128_summary.json`,
  `summaries/macro_4096x1024_bs4_np128_summary.json`
- New short nsys:
  `raw/nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0.sqlite`
- New metadata split:
  `summaries/nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0_decode_metadata_subboundary.md`
- New 07.63-compatible classifier:
  `summaries/nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0_classified.md`

## Baseline

Baseline variant:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

The legacy alias `target0762_woabf16bmmcache` remains historical only and is not
used as the report name.

07.63 frozen results:

| Workload | Status | Output tok/s | Decode tok/s | Replay | Eager decode | Unsupported skips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| text smoke | pass | n/a | n/a | `9` | `0` | n/a |
| 4096/128/batch4 | pass | `59.5264` | `150.2022` | `508` | `0` | `0` |
| 4096/1024/batch4 | pass | `119.4153` | `149.1220` | `4092` | `0` | `0` |

The old serving victory line is `114.07` output tok/s for 4096/1024/batch4.

## Sub-Boundary Attribution

The 07.63 profile selected `graph_runtime_copy_cat_index` at `0.846795s`,
`21.48%` of the 4096/128 decode envelope, with `248,069` kernels.

The finer split showed that the stable mini source boundary available for a
narrow implementation was the decode metadata assembly in
`DSV4AttentionBackend._build_metadata`: page table, SWA indices/lens, and
C4/C128 compressed raw/page/full indices/lens construction. The much larger
`direct_copy` portion was mostly under `batch_forward` and
`batch_forward_enqueue`, not a clean metadata-only mini boundary.

Sub-boundary comparison:

| Sub-boundary | 07.63 s | 07.64 s | Delta s |
| --- | ---: | ---: | ---: |
| `direct_copy` | `0.736769` | `0.731834` | `-0.004934` |
| `index_elementwise_kernel` | `0.005095` | `0.001985` | `-0.003110` |
| `CatArrayBatchedCopy` | `0.034110` | `0.034106` | `-0.000004` |
| `gatherTopK` | `0.074776` | `0.074760` | `-0.000016` |
| `arange_index_helper` | `0.029432` | `0.025551` | `-0.003881` |
| `topk_lens_swa_compressed_index_assembly` | `0.068879` | `0.067087` | `-0.001793` |
| `other_metadata_copy_cat_index` | `0.078360` | `0.079118` | `+0.000758` |
| adjacent selected total | `1.027421` | `1.014442` | `-0.012979` |

The source-owned `batch_prepare:decode:bs4` metadata slice dropped from
`0.019838s` to `0.005991s`, but that is too small to move the total selected
bucket enough.

## Source Parity

vLLM files reviewed:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py
```

Reviewed helpers:

- `compute_global_topk_indices_and_lens`
- `combine_topk_swa_indices`
- `flat_index_dequant_gather_blocked`

Only one mini helper was implemented. It follows the same design principle as
vLLM's fused index/lens helpers: produce padded index tensors and lengths in one
GPU helper instead of a chain of torch arange, index, fill, gather, and copy
ops. It was not a blind port:

- `compute_global_topk_indices_and_lens` already has partial mini parity through
  existing topk/global-lens code.
- `combine_topk_swa_indices` is closest in spirit, but mini's decode path has a
  different boundary, so the implementation targets decode-only page/SWA/C4/C128
  metadata construction.
- `flat_index_dequant_gather_blocked` is tied to vLLM's blocked FP8 KV cache
  path and does not map cleanly to the current mini BF16 sparse decode plus FP8
  indexer side-cache path.

Sentinel policy: the helper preserves exact `-1` padding sentinels. The
microbench observed no tolerated sentinel differences.

## Implementation

Opt-in variant:

```text
dsv4_sm80_a100_victory_metadatadeforest
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
```

Touched runtime files:

- `python/minisgl/kernel/deepseek_v4.py`
  - added `DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE`
  - added `DSV4DecodeMetadataDeforestOutput`
  - added `decode_metadata_deforest_fallback`
- `python/minisgl/kernel/triton/deepseek_v4.py`
  - added `build_decode_metadata_indices`
  - added `_build_decode_metadata_indices_kernel`
- `python/minisgl/attention/deepseek_v4.py`
  - decode-only `_build_metadata` path uses the helper when opted in
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - added `dsv4_sm80_a100_victory_metadatadeforest`
- `benchmark/offline/deepseek_v4_text_smoke.py`
  - added matching text-smoke variant

The helper emits:

- `page_table`
- `swa_page_indices`
- `swa_topk_lengths`
- `c4_topk_lengths_raw`
- `c4_topk_lengths_clamp1`
- `c4_sparse_topk_lengths`
- `c4_sparse_raw_indices`
- `c4_sparse_page_indices`
- `c4_sparse_full_indices`
- `c128_topk_lengths_clamp1`
- `c128_raw_indices`
- `c128_page_indices`
- `c128_full_indices`

## Microbench

Command:

```bash
python performance_milestones/target07_decode_metadata_deforestation/scripts/microbench_decode_metadata_deforest.py \
  --json-out performance_milestones/target07_decode_metadata_deforestation/raw/decode_metadata_deforest_microbench.json \
  --md-out performance_milestones/target07_decode_metadata_deforestation/summaries/decode_metadata_deforest_microbench.md \
  --repeats 30
```

Result: pass, `all_equal=True`, exact `-1` padding equality.

| BS | Max Seq | Old us | New us | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | `1054.14` | `153.99` | `6.85x` |
| 1 | 4096 | `1073.91` | `149.13` | `7.20x` |
| 1 | 5120 | `1059.43` | `147.97` | `7.16x` |
| 2 | 128 | `1131.67` | `149.32` | `7.58x` |
| 2 | 4096 | `1129.69` | `148.02` | `7.63x` |
| 2 | 5120 | `1135.36` | `151.06` | `7.52x` |
| 4 | 128 | `1249.14` | `148.12` | `8.43x` |
| 4 | 4096 | `1253.58` | `147.82` | `8.48x` |
| 4 | 5120 | `1246.39` | `148.99` | `8.37x` |

## Macro Results

TP8, page-size 256, num-pages 128.

| Workload | Status | Output tok/s | vs 07.63 | Decode tok/s | Replay | Eager decode | Unsupported skips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text smoke | pass | n/a | n/a | n/a | `9` | `0` | n/a |
| 4096/128/batch4 | pass | `60.3682` | `+1.41%` | `150.0042` | `508` | `0` | `0` |
| 4096/1024/batch4 | pass | `122.9414` | `+2.95%` | `149.4621` | `4092` | `0` | `0` |

The 4096/1024 result stays above the old `114.07` line, but it does not reach
the 5% target over 07.63.

Short nsys macro run:

| Workload | Status | Output tok/s | Decode tok/s | Replay | Eager decode | Unsupported skips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 nsys | pass | `48.5420` | `137.5511` | `127` | `0` | `0` |

## Profile Gate

07.63-compatible `repeat_decode_forward_envelope` bucket:

| Bucket | 07.63 s | 07.64 s | Delta s | Gate |
| --- | ---: | ---: | ---: | --- |
| `graph_runtime_copy_cat_index` | `0.846795` | `0.834792` | `-0.012003` | fail, needs at least `-0.25s` |

The total adjacent metadata split moved by `-0.012979s`, consistent with the
source-owned `batch_prepare` slice shrinking but the large graph-forward
`direct_copy` surface remaining.

## Memory And Workspace

No persistent cache, workspace manager, or KV-cache layout was added. The helper
allocates the same classes of metadata output tensors and writes them with one
Triton launch. Peak memory stayed effectively identical to 07.63:

| Workload | 07.63 peak bytes | 07.64 peak bytes | KV bytes per rank |
| --- | ---: | ---: | ---: |
| 4096/128/batch4 | `47,294,730,240` | `47,294,730,240` | `2,491,495,680` |
| 4096/1024/batch4 | `47,294,760,960` | `47,294,760,960` | `2,491,495,680` |

## Verification

- `python -m py_compile` on touched runtime, benchmark, and milestone scripts:
  pass.
- `pytest -q -o addopts='' tests/attention/test_deepseek_v4_backend_metadata.py`:
  `10 passed`.
- TP8 text smoke with `dsv4_sm80_a100_victory_metadatadeforest`: pass.
- TP8 macro 4096/128/batch4: pass.
- TP8 macro 4096/1024/batch4: pass.
- TP8 short nsys 4096/128/batch4: pass.

## Final Recommendation

Keep opt-in for future ablation and source-comparison use. Do not promote to the
victory bundle, because TARGET 07.64's required bucket reduction or 4096/1024
5% E2E gain was not achieved. Do not continue this thread into projection,
communication, MoE, precision, or broad graph cleanup work.
