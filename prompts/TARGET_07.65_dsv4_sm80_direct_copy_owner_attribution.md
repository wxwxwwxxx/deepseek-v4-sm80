# TARGET 07.65: DSV4 SM80 Direct-Copy Owner Attribution

Date: 2026-07-02

## Goal

Attribute the remaining graph-replay `direct_copy` surface after TARGET 07.64.

This is a measurement-only target.  It should add finer NVTX/source attribution
and improve the profile classifiers enough to explain where the large
`direct_copy` bucket comes from.  It must not implement a performance
optimization.

The output should be an owner table that says which concrete mini source
boundaries create the replay direct-copy kernels under:

```text
batch_forward:decode:bs4:padded4
batch_forward_enqueue:decode:bs4:padded4
```

and exactly one evidence-backed next implementation target.

## Starting Point

Current promoted milestone:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Current confirmed promoted macro from TARGET 07.63:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `59.5264` | `150.2022` | `508` | `0` |
| 4096/1024/batch4 | `119.4153` | `149.1220` | `4092` | `0` |

TARGET 07.64 implemented an opt-in helper:

```text
dsv4_sm80_a100_victory_metadatadeforest
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
```

Keep it as an opt-in ablation only.  Do not add it to
`dsv4_sm80_a100_victory` in this target.

07.64 result:

- text smoke passed;
- 4096/1024/batch4 moved from `119.4153` to `122.9414 output tok/s`
  (`+2.95%`);
- `graph_runtime_copy_cat_index` moved only from `0.846795s` to `0.834792s`
  (`-0.012003s`);
- the target gate failed, so the helper is **keep opt-in, do not promote**.

## Evidence For This Target

07.64 sub-boundary split:

| Sub-boundary | 07.63 s | 07.64 s | Delta s |
| --- | ---: | ---: | ---: |
| `direct_copy` | `0.736769` | `0.731834` | `-0.004934` |
| `index_elementwise_kernel` | `0.005095` | `0.001985` | `-0.003110` |
| `CatArrayBatchedCopy` | `0.034110` | `0.034106` | `-0.000004` |
| `gatherTopK` | `0.074776` | `0.074760` | `-0.000016` |
| `arange_index_helper` | `0.029432` | `0.025551` | `-0.003881` |
| `topk_lens_swa_compressed_index_assembly` | `0.068879` | `0.067087` | `-0.001793` |
| `other_metadata_copy_cat_index` | `0.078360` | `0.079118` | `+0.000758` |

Owner split from 07.64:

| NVTX owner | Kernel s | Dominant sub-boundaries |
| --- | ---: | --- |
| `batch_forward:decode:bs4:padded4` | `0.653641` | direct_copy=`0.4713s`, other_metadata=`0.0536s`, gatherTopK=`0.0498s` |
| `batch_forward_enqueue:decode:bs4:padded4` | `0.354810` | direct_copy=`0.2587s`, other_metadata=`0.0255s`, gatherTopK=`0.0249s` |
| `batch_prepare:decode:bs4` | `0.005991` | topk/lens=`0.0022s`, direct_copy=`0.0018s`, arange/index=`0.0012s` |

Interpretation:

- the metadata helper from 07.64 correctly shrank `batch_prepare`;
- `batch_prepare` was too small to move the total bucket;
- the real remaining target is the `direct_copy` surface inside graph replay
  forward/enqueue boundaries;
- this `direct_copy` must be attributed before any implementation.

## Scope

In scope:

- add opt-in profiling-only NVTX ranges around suspected copy/layout/staging
  boundaries;
- improve the Nsight SQLite classifier to map `direct_copy` kernels to finer
  owners;
- capture short 4096/128/batch4 profiles for the promoted milestone and, if
  useful, the 07.64 metadata opt-in;
- produce direct-copy owner tables;
- select one next implementation target if one owner clearly dominates.

Out of scope:

- changing runtime behavior for performance;
- adding kernels;
- promoting `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST`;
- changing the `dsv4_sm80_a100_victory` bundle;
- projection/GEMM work;
- MoE/Marlin work;
- NCCL/communication work;
- precision/KV-cache work;
- radix/prefix-cache work;
- unified cache/workspace manager implementation.

## Suspected Owner Categories

The attribution pass should distinguish at least these categories:

| Category | Examples to inspect | Possible future action |
| --- | --- | --- |
| graph input staging | copying current batch tensors into static CUDA graph inputs | static input packing / reduce graph input surfaces |
| replay metadata copy | copied page tables, positions, out locs, compressed locs, sampler metadata | prebind or pack metadata buffers |
| tensor layout materialization | `.contiguous()`, dtype copies, reshape/view that triggers copy | layout contract change |
| attention boundary copies | q/kv/indexer/sparse attention staging | fuse or preallocate owner-specific buffers |
| MoE/shared-expert staging | route buffers, Marlin input/output layout, shared expert copies | MoE layout or workspace target |
| sampler/logits staging | logits all-gather/output copy, sampled token buffers | sampler/logits target |
| Python enqueue/forward bridge | tensors copied between batch preparation and graph forward call | scheduler/runner graph-input API target |

Do not assume the owner before measuring.

## Instrumentation Plan

Add profiling-only instrumentation guarded by an env flag such as:

```text
MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1
```

Suggested instrumentation rules:

- use NVTX only, no behavior changes;
- keep ranges short and owner-specific;
- include tensor name, shape, dtype, and high-level owner in range names when
  possible;
- do not add synchronization;
- do not allocate new CUDA tensors for instrumentation;
- make the code path safe when NVTX is unavailable;
- make the env flag off by default.

Candidate files to inspect/instrument:

```text
python/minisgl/engine.py
python/minisgl/scheduler/
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
benchmark/offline/deepseek_v4_perf_matrix.py
```

Use `rg` to locate existing NVTX helpers such as `_dsv4_capture_nvtx`,
`batch_forward`, `batch_forward_enqueue`, graph runner replay/copy code, and
metadata copy paths before editing.

## Work Plan

1. Create artifacts:

   ```text
   performance_milestones/target07_direct_copy_owner_attribution/
     README.md
     raw/
     summaries/
     scripts/
   ```

2. Freeze baseline context.

   Primary baseline:

   ```text
   --variants dsv4_sm80_a100_victory
   --page-size 256
   --num-pages 128
   ```

   Secondary optional ablation:

   ```text
   --variants dsv4_sm80_a100_victory_metadatadeforest
   ```

   Only use the secondary ablation to confirm direct-copy ownership stability.
   Do not treat it as promoted baseline.

3. Reuse existing profiles before adding instrumentation.

   Required inputs:

   - 07.63 SQLite:
     `performance_milestones/target07_post_victory_reprofile/raw/nsys_target0763_post_victory_4096x128_bs4_np128_rank0.sqlite`
   - 07.64 SQLite:
     `performance_milestones/target07_decode_metadata_deforestation/raw/nsys_target0764_metadatadeforest_4096x128_bs4_np128_rank0.sqlite`

   Write an initial attribution table from existing NVTX only.  This gives the
   pre-instrumentation control.

4. Add profiling-only NVTX.

   Instrument suspected copy/layout boundaries.  Prefer narrow ranges around
   code that can plausibly emit direct-copy kernels:

   - graph input copy / replay input copy;
   - batch enqueue to graph runner;
   - metadata tensor packing;
   - static graph input updates;
   - owner-specific `.copy_`, `.to`, `.contiguous`, and dtype conversion sites;
   - sampler/logits output staging if visible.

   Keep all instrumentation behind the profiling env flag.

5. Capture short nsys profiles.

   Required:

   - TP8;
   - page size 256;
   - num-pages 128;
   - 4096/128/batch4;
   - `dsv4_sm80_a100_victory`;
   - one repeat, zero warmup if that matches recent profiles;
   - CUDA/NVTX/osrt/cuBLAS/NCCL traces as supported by local nsys;
   - `MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1`.

   Optional:

   - repeat with `dsv4_sm80_a100_victory_metadatadeforest` to verify the same
     direct-copy owners remain after 07.64's helper.

6. Build the classifier.

   The classifier should produce:

   | Direct-copy owner | Kernel s | Count | Share of direct_copy | Source file/function | Evidence |
   | --- | ---: | ---: | ---: | --- | --- |

   It should also keep a residual `unattributed` bucket.  If residual is large,
   say what extra NVTX would be needed.

7. Validate instrumentation overhead.

   Because this is measurement-only, do not require full macro repeat unless
   instrumentation appears to change behavior.

   Required:

   - text smoke can be skipped if no runtime behavior changed, but record why;
   - graph replay/eager decode from the profiled run must still show replay
     active and eager decode `0`;
   - if any code path changes beyond NVTX or scripts, run text smoke.

8. Select exactly one next target.

   A valid next implementation target needs:

   - a named owner with at least `0.15s` direct-copy cost in the 4096/128
     decode envelope, or a stack of adjacent owners with a clean shared fix;
   - a concrete source boundary;
   - a plausible implementation shape;
   - expected 4096/1024 E2E upside;
   - clear stop gates.

   If no owner clears this bar, recommend another measurement target rather
   than implementation.

9. Write final README.

   Include:

   - exact commands;
   - git state;
   - profiles used;
   - instrumentation summary;
   - direct-copy owner table;
   - residual/unattributed table;
   - graph replay/eager decode state;
   - whether 07.64 metadata opt-in changes direct-copy owners;
   - next target recommendation;
   - do-not-continue notes.

## Gates

Measurement gate:

- at least `80%` of the 4096/128 `direct_copy` bucket should be assigned to
  named owners, or the report must explain why attribution cannot reach that
  level.

Next-target gate:

- select an implementation target only if one owner or clean owner group has
  at least `0.15s` measured direct-copy cost and a plausible fix.

Graph gate:

- decode graph replay remains active;
- eager decode remains `0`;
- instrumentation does not force eager fallback.

Scope gate:

- no performance optimization code lands in this target;
- no default behavior changes;
- no milestone bundle changes.

## Stop Conditions

Stop and write the report when:

- the direct-copy owner table explains the bucket well enough to choose the
  next target;
- the residual direct-copy bucket remains too large after one additional NVTX
  pass;
- instrumentation perturbs graph replay or decode behavior;
- the next idea would require implementation before attribution;
- the dominant owner points outside direct-copy into projection, communication,
  MoE, precision, or cache-manager work.

Do not continue into implementation inside TARGET 07.65.

