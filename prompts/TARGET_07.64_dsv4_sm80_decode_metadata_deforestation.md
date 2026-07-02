# TARGET 07.64: DSV4 SM80 Decode Metadata Deforestation

Date: 2026-07-02

## Goal

Reduce the post-victory decode metadata bucket selected by TARGET 07.63:

```text
graph_runtime_copy_cat_index
```

This is a narrow implementation target for the decode metadata
gather/index/copy/cat/topk-lens boundary.  It should use vLLM's DeepSeek V4
SM80 source path as the design compass, but keep mini's current
`dsv4_sm80_a100_victory` milestone bundle and correctness gates.

Do not turn this into a generic graph/layout cleanup pass, projection/GEMM pass,
communication pass, or unified cache/workspace manager target.

## Starting Point

Fresh TARGET 07.63 confirmation:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode | Repeat spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `59.5264` | `150.2022` | `508` | `0` | `0.41%` |
| 4096/1024/batch4 | `119.4153` | `149.1220` | `4092` | `0` | `0.53%` |

The current milestone remains above the old serving line:

```text
114.07 output tok/s
```

Current variant and env:

```text
--variants dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

The old name `target0762_woabf16bmmcache` is only a compatibility alias.  Do
not use it as the primary report name for new artifacts.

TARGET 07.63 also fixed a config-path bug before promotion:

- `python/minisgl/kvcache/deepseek_v4_pool.py` now uses
  `dsv4_env_flag(MINISGL_DSV4_SM80_INDEXER_FP8_CACHE)` for the FP8 indexer
  side-cache allocation and byte estimate;
- this makes bundle expansion and KV-cache allocation agree;
- it is a correctness/config fix, not this target's optimization.

## Evidence

TARGET 07.63 rank-0 short Nsight profile:

| Bucket | Kernel s | Share | Kernel count | Auxiliary graph-node count | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `graph_runtime_copy_cat_index` | `0.846795` | `21.48%` | `248,069` | `1,895` | selected |
| `projection_gemm` | `0.812100` | `20.60%` | `100,965` | `795` | hold; too diffuse |
| `elementwise_graph_nodes` | `0.497965` | `12.63%` | `201,670` | `1,551` | secondary validation only |
| `nccl_communication` | `0.340015` | `8.62%` | `11,176` | `88` | hold |
| `moe_marlin` | `0.300516` | `7.62%` | `43,688` | `344` | hold |

Nsight caveat: this profile did not contain `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`,
so the graph-node count is an auxiliary script-derived signal.  Treat kernel
time, kernel count, NVTX source ownership, and source-boundary parity as the
primary evidence.

Top relevant kernels include:

- direct-copy kernels;
- `index_elementwise_kernel`;
- `CatArrayBatchedCopy`;
- `gatherTopK`;
- arange/index helper kernels;
- adjacent topk-lens and SWA/compressed-index assembly kernels.

## vLLM Source Parity

vLLM source root:

```text
/workspace/vllm-dsv4-docker
```

Primary reference files:

```text
vllm/model_executor/layers/deepseek_v4_attention.py
vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py
```

Named vLLM helpers to study:

| vLLM helper | Source location | Semantic target |
| --- | --- | --- |
| `compute_global_topk_indices_and_lens` | `cache_utils.py` | fuse local topk -> global KV slot mapping, valid-entry counting, and padding-token masking |
| `combine_topk_swa_indices` | `cache_utils.py` | fuse topk/SWA index concatenation, length construction, and padding/alignment |
| `flat_index_dequant_gather_blocked` | `cache_utils.py` | replace a multi-op PyTorch gather/dequant path for packed blocked FP8 cache |

Important dispatch sites in vLLM:

- `deepseek_v4_attention.py` around decode topk mapping for C4A;
- `deepseek_v4_attention.py` around sparse prefill/topk/SWA combination;
- `deepseek_v4_attention.py` around reference sparse attention gather/dequant.

vLLM timing caveat from 07.63 still applies: existing vLLM Nsight summaries in
this workspace have a repeat window with `kernel.count=0`, so source-dispatch
parity is more reliable than vLLM bucket timing until that profiler issue is
fixed.

## mini Source Boundaries

Primary mini files:

```text
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/kernel/csrc/jit/dsv4_topk_v1.cu
python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu
```

Likely current mini boundaries:

- `topk_transform_512_full_fallback` and
  `_run_local_cuda_global_topk_lens_512` already provide a local topk/global
  lens path, but 07.63 still sees a large metadata bucket;
- `dsv4_sparse_attention_two_source_splitk_bf16` expects prepared
  `swa_indices`, `swa_lengths`, optional `compressed_indices`, and optional
  `compressed_lengths`;
- `python/minisgl/kernel/triton/deepseek_v4.py::sparse_attention_splitk_bf16`
  calls `.contiguous()` on metadata tensors if strides are not ideal;
- model-side attention/indexer code can still create torch `cat`, `index`,
  gather, arange, and copy kernels before the sparse attention kernel.

The first task is to attribute which of these exact boundaries owns the 07.63
`graph_runtime_copy_cat_index` bucket.  Do not implement all three vLLM helpers
blindly.

## Scope

In scope:

- add one or two opt-in metadata fusion helpers if evidence supports them;
- port or adapt vLLM-style Triton helper logic where it matches mini's tensor
  layout;
- preserve existing exactness/precision policy for the current milestone;
- keep page size 256 and `--num-pages 128` for direct comparison;
- run focused microbench and TP8 macro benchmarks;
- reprofile the selected bucket after the implementation;
- update milestone artifacts and README.

Out of scope:

- more BF16 projection caches;
- MoE/Marlin backend changes;
- row-parallel communication/all-reduce changes;
- full FP8 KV cache or `fp8_ds_mla` E2E;
- radix/prefix-cache work;
- generic `torch.compile` or broad graph/layout cleanup;
- unified cache/workspace manager implementation.

## Candidate Cuts

Candidate A: topk global indices and lens parity

- Compare mini `_run_local_cuda_global_topk_lens_512` /
  `dsv4_topk_v1.cu` with vLLM `compute_global_topk_indices_and_lens`.
- Check whether mini still performs surrounding torch index/copy/cat work that
  vLLM fuses into the helper.
- Success shape: fewer topk-lens/index kernels and lower
  `graph_runtime_copy_cat_index` time without changing sparse attention output.

Candidate B: topk plus SWA index combination

- Locate any mini-side construction of combined SWA/compressed index matrices,
  length tensors, padding, and alignment.
- Compare to vLLM `combine_topk_swa_indices`.
- Success shape: replace torch `cat`/fill/index/copy chain with a single helper
  or with preallocated output writes that preserve graph replay addresses.

Candidate C: flat index gather/dequant blocked path

- Only pursue if the current mini profile proves an equivalent gather/dequant
  chain exists in the active `dsv4_sm80_a100_victory` decode path.
- vLLM's helper is tied to packed blocked FP8 cache.  mini currently uses a
  BF16 sparse decode path with an FP8 indexer side cache, so this may not map
  directly.
- If the layout does not match, document it and stop instead of forcing a port.

## Work Plan

1. Create milestone artifacts:

   ```text
   performance_milestones/target07_decode_metadata_deforestation/
     README.md
     raw/
     summaries/
     scripts/
   ```

2. Freeze the baseline.

   Use:

   ```text
   --variants dsv4_sm80_a100_victory
   --page-size 256
   --num-pages 128
   ```

   Record:

   - current commit and dirty state;
   - active toggles and raw env;
   - text smoke result;
   - 4096/128 and 4096/1024 macro baseline or reused 07.63 values if no code
     has changed since 07.63 except measurement scripts;
   - graph replay and eager decode counts.

3. Attribute the metadata bucket before implementation.

   Use the 07.63 SQLite profile and/or a fresh short nsys run to split
   `graph_runtime_copy_cat_index` into named source boundaries.

   Required table:

   | Sub-boundary | Kernel names | Kernel s | Count | mini source | vLLM counterpart | Decision |
   | --- | --- | ---: | ---: | --- | --- | --- |

   Do not proceed to implementation until one sub-boundary has enough measured
   cost to justify the cut.  A good first cut should plausibly remove at least
   `0.15s` from the 4096/128 decode envelope by itself, or stack cleanly with a
   second cut to reach the target gate.

4. Build a focused oracle/microbench.

   For any selected candidate helper:

   - construct representative tensors from real benchmark artifacts if
     possible;
   - compare old torch/local-CUDA path versus new helper;
   - validate exact equality for indices/lens, or document tolerated sentinel
     differences such as `-1` padding where downstream behavior is identical;
   - run token/request shapes representative of batch4 decode and, if useful,
     batch1/batch2 graph capture sizes.

5. Implement at most two narrow opt-in helpers.

   Suggested toggle:

   ```text
   MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
   ```

   Suggested benchmark/text-smoke variant suffix:

   ```text
   metadatadeforest
   ```

   If splitting candidates is clearer, use narrower toggles such as:

   ```text
   MINISGL_DSV4_SM80_FUSED_TOPK_SWA_INDICES=1
   MINISGL_DSV4_SM80_FUSED_DECODE_TOPK_LENS=1
   ```

   but still provide one combined benchmark variant on top of
   `dsv4_sm80_a100_victory`.

   Implementation constraints:

   - no decode-time large `cudaMalloc`;
   - graph replay must preserve stable buffer addresses;
   - no broad cache/workspace manager refactor;
   - no silent fallback during CUDA graph capture when the opt-in requires the
     fused helper;
   - if a helper cannot support the active shape/layout, raise clearly outside
     graph capture or disable the variant during preflight.

6. Validate correctness.

   Required:

   - Python compile check for touched Python scripts/modules;
   - focused helper/oracle tests;
   - TP8/page-size-256 text smoke with `dsv4_sm80_a100_victory` plus the new
     metadata opt-in;
   - compare generated text sanity with the 07.63 smoke.

7. Run macro benchmarks.

   Required workloads:

   - 4096/128/batch4;
   - 4096/1024/batch4.

   Compare against 07.63:

   | Workload | 07.63 baseline | New | Gain | Graph replay | Eager decode |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | 4096/128/batch4 | `59.5264` | TBD | TBD | TBD | TBD |
   | 4096/1024/batch4 | `119.4153` | TBD | TBD | TBD | TBD |

8. Reprofile.

   Capture a short 4096/128/batch4 nsys profile with the new opt-in enabled.

   Required comparison:

   | Bucket | 07.63 baseline | New | Change |
   | --- | ---: | ---: | ---: |
   | `graph_runtime_copy_cat_index` | `0.846795s` | TBD | TBD |
   | `projection_gemm` | `0.812100s` | TBD | TBD |
   | `elementwise_graph_nodes` | `0.497965s` | TBD | TBD |
   | `nccl_communication` | `0.340015s` | TBD | TBD |
   | `moe_marlin` | `0.300516s` | TBD | TBD |

   Also report top kernel count changes for direct copy, index,
   CatArrayBatchedCopy, gatherTopK, arange/index helper kernels, and any new
   fused metadata kernel.

9. Write final README.

   Include:

   - variant/toggle summary;
   - baseline and new macro results;
   - helper microbench/oracle results;
   - source parity against vLLM;
   - fresh profile bucket table;
   - whether the metadata bucket gate was met;
   - memory/workspace impact;
   - recommendation: promote, keep opt-in for ablation, or revert/stop;
   - exactly one next target decision.

## Gates

Correctness gate:

- TP8/page-size-256 text smoke must pass;
- helper oracle must match old path indices/lens semantics.

Graph gate:

- decode graph replay must remain active;
- eager decode must remain `0`;
- no unexpected unsupported kernel skips;
- fallback wrapper calls should not increase unexpectedly.

Performance gate:

- remove at least `0.25s` from 4096/128
  `graph_runtime_copy_cat_index`, or
- improve 4096/1024/batch4 E2E output throughput by at least `5%`
  over `119.4153`, or
- if the first helper removes `0.15-0.25s` and has a clean second adjacent
  sub-boundary, the final report may recommend a follow-up metadata target
  instead of declaring failure.

Regression gate:

- 4096/1024/batch4 must stay above `114.07 output tok/s` after confirming
  variant expansion, page size 256, num-pages 128, graph replay, and no eager
  decode.

## Stop Conditions

Stop and write the report when any of these happens:

- one or two scoped metadata helpers meet the gate;
- the selected sub-boundary cannot be attributed to a stable mini source
  location;
- the helper requires full cache/workspace manager refactoring;
- correctness fails after one focused fix;
- graph replay breaks or eager decode becomes nonzero;
- macro gain is below `2%` and profile bucket reduction is below `0.15s`;
- the next best idea is projection, communication, MoE, or precision work.

Do not continue into another optimization category inside this target.

