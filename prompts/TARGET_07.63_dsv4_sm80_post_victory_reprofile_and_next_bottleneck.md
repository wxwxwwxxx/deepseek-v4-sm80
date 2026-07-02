# TARGET 07.63: DSV4 SM80 Post-Victory Reprofile and Next Bottleneck

Date: 2026-07-02

## Goal

Freeze and reprofile the first DSV4 SM80 milestone that beats the old serving
victory line.  This target is a confirmation and evidence-reset pass, not an
implementation target.

The current milestone path is:

```text
dsv4_sm80_a100_victory
```

which expands through:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

The old variant name:

```text
target0762_woabf16bmmcache
```

is only a compatibility alias for historical artifacts and scripts.  New
reports, commands, and artifact names should use `dsv4_sm80_a100_victory`.

The output of this target should be a fresh bottleneck table and exactly one
recommended next implementation target.  Do not land performance code changes
inside this target unless a tiny measurement-script fix is required to make the
profile readable.

## Current Milestone

TARGET 07.62 completed the `attn.wo_a` BF16 grouped BMM cache and produced the
first official crossing of the old serving baseline:

| Workload | Previous best | Current milestone | Change |
| --- | ---: | ---: | ---: |
| 4096/128/batch4 | `51.2962 output tok/s` | `53.5877 output tok/s` | `+4.47%` |
| 4096/1024/batch4 | `105.7645 output tok/s` | `116.2553 output tok/s` | `+9.92%` |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- fresh vLLM offline 4096/1024/batch4 reference: about
  `202.03 output tok/s`;
- page/block size must remain `256`.

The `attn.wo_a` owner changed substantially:

| Metric | 07.61 / 07.60 baseline | 07.62 milestone |
| --- | ---: | ---: |
| `attn.wo_a` replay owner | `0.481377s` | `0.068948s` |
| graph replay | active | active |
| eager decode | `0` | `0` |

The cached BF16 projection memory ledger is now:

| Cache group | Bytes/rank | GiB/rank | Equivalent KV tokens | Equivalent pages |
| --- | ---: | ---: | ---: | ---: |
| q_wqb + wo_b + indexer.wq_b | `1.0000 GiB/rank` | `1.0000` | `14121.79` | `55.16` |
| plus wo_a BMM cache | `0.3359 GiB/rank` | `0.3359` | `4744.04` | `18.53` |
| total cached BF16 projection stack | TBD exact bytes | `1.3359` | `18865.83` | `73.69` |

Recompute the exact bytes from the fresh run if the script reports them.

## Opt-In Cleanup Context

After TARGET 07.62, the best path was cleaned up into a small milestone bundle.
This matters for all commands in this target:

- use `--variants dsv4_sm80_a100_victory`;
- do not manually stack the old long variant name unless reproducing a
  historical result;
- the compatibility alias `target0762_woabf16bmmcache` should expand to the
  same env as the new milestone path, but it should not be used in new reports;
- the milestone bundle intentionally does not enable the stale
  `Q_WQB/WO_B/INDEXER_WQB_FP8_GEMM` opt-ins because the BF16 projection-cache
  paths supersede them in the current best stack;
- keep individual cache toggles available for ablation because they have real
  VRAM tradeoffs;
- keep recording the expanded active toggle list in every artifact so future
  threads can tell exactly what the bundle meant at this milestone.

Expected high-level bundle contents:

- Marlin WNA16 MoE backend;
- DSV4 decode CUDA graph replay with greedy sample capture;
- FP8 indexer cache backend;
- split-K BF16 sparse decode;
- replay metadata copy path;
- HC/RMSNorm/fused WQA-WKV/Q-KV norm-rope-store helpers;
- gate fp32 weight cache and indexer-store norm fp32 weight cache;
- four BF16 projection caches:
  `q_wqb`, `wo_b`, `indexer.wq_b`, and `wo_a` grouped BMM cache.

## Scope

In scope:

- TP8/page-size-256 text smoke on the milestone variant;
- macro reruns for 4096/128/batch4 and 4096/1024/batch4;
- decode-throughput rerun if supported by the existing benchmark scenarios;
- fresh mini owner/profile attribution;
- short Nsight Systems profile for the 4096/128/batch4 shape;
- fallback counter and graph replay/eager decode audit;
- memory ledger for all persistent caches introduced by the milestone stack;
- source and profile comparison against vLLM for the top remaining mini
  buckets;
- final next-target recommendation.

Out of scope:

- implementing a new kernel;
- adding another cached-weight path by inertia;
- generic graph/layout cleanup;
- full FP8 KV cache or `fp8_ds_mla` E2E;
- radix/prefix-cache work;
- broad refactors of cache/workspace management.

## Required Artifacts

Create:

```text
performance_milestones/target07_post_victory_reprofile/
  README.md
  raw/
  summaries/
  scripts/
```

Large `.nsys-rep`, `.sqlite`, and other profiler files should be symlinked if
they already live outside the milestone directory.  Small JSON, JSONL, log, and
summary files can be copied.

## Work Plan

1. Record repository state.

   Include:

   - current branch;
   - current commit;
   - the milestone git tag if available;
   - dirty files, if any;
   - active CUDA devices;
   - PyTorch/CUDA/NCCL versions if already reported by the benchmark harness.

2. Confirm variant expansion.

   Use the benchmark/text-smoke variant configuration path to record:

   - `raw_dsv4_sm80_env`;
   - `active_dsv4_toggles`;
   - selected MoE expert backend;
   - whether the stale `Q_WQB/WO_B/INDEXER_WQB_FP8_GEMM` opt-ins are inactive.

   The report should explicitly say whether:

   ```text
   dsv4_sm80_a100_victory == target0762_woabf16bmmcache
   ```

   at the env-expansion level.

3. Run correctness smoke.

   Required:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_text_smoke.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory \
     --page-size 256 \
     --output /tmp/dsv4_target0763_text_smoke.json
   ```

   The smoke must pass without乱码/garbage/repetition failure before any
   performance result is promoted.

4. Run fair macro confirmation.

   Required workload set:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_perf_matrix.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory \
     --scenarios decode_throughput_bs8 \
     --prompt-len 4096 \
     --decode-len 128 \
     --batch-size 4 \
     --repeats 3 \
     --warmup-repeats 1 \
     --page-size 256 \
     --output-dir /tmp/dsv4_target0763_4096x128_bs4 \
     --keep-going
   ```

   and the same shape with:

   ```text
   --decode-len 1024
   --output-dir /tmp/dsv4_target0763_4096x1024_bs4
   ```

   If the benchmark harness prefers named scenarios over overridden shape
   arguments, preserve the same logical shape and record the exact command used.

5. Capture a short Nsight Systems profile.

   Use the existing mini nsys script pattern under
   `performance_milestones/vllm/scripts/` or prior TARGET 07 scripts as a
   template.  Capture the 4096/128/batch4 milestone variant only.

   Required properties:

   - profile a short repeat window, not all warmup and model load;
   - include CUDA, NVTX, OS runtime, cuBLAS, and NCCL traces if supported by the
     local `nsys` version;
   - do not use `-t nccl` if this `nsys` build rejects it;
   - write or symlink the `.nsys-rep` into `raw/`;
   - export SQLite or another script-readable summary if existing tools expect
     it.

6. Run mini owner/profile attribution.

   Use the current project tooling from recent TARGET 07 milestones.  The
   summary must at least bucket:

   - projection/GEMM owners;
   - MoE/Marlin/shared experts;
   - sparse attention decode;
   - indexer/cache/topk;
   - graph/runtime/copy/cat/index;
   - elementwise graph nodes;
   - NCCL communication;
   - sampler/logits if visible;
   - prefill-specific sparse/reference paths if they appear in the measured
     workload.

7. Compare against vLLM at the evidence level.

   For the top two or three remaining mini buckets, write a parity table:

   | Bucket | mini current behavior | vLLM source/profile evidence | Difference | Plausible E2E upside | Next action |
   | --- | --- | --- | --- | ---: | --- |

   Use vLLM source under:

   ```text
   /workspace/vllm-dsv4-docker
   ```

   Important reference files:

   - `vllm/model_executor/models/deepseek_v4.py`;
   - `vllm/model_executor/layers/deepseek_v4_attention.py`;
   - `vllm/v1/attention/ops/deepseek_v4_ops/`;
   - `vllm/model_executor/layers/fused_moe/`.

   If vLLM profile timing is still incomplete or unusable, say so directly and
   rely on mini owner timing plus vLLM source-dispatch parity.

8. Recompute the memory ledger.

   Include:

   - cache/workspace owner;
   - shape;
   - dtype;
   - bytes/rank;
   - GiB/rank;
   - lifecycle;
   - equivalent KV tokens/rank and pages/rank at page size 256;
   - whether the allocation is prebuilt before graph replay or can happen
     during decode.

   Highlight any cache that still uses ad hoc ownership and should eventually
   move into a unified cache/workspace manager.  Do not implement that manager
   in this target.

9. Select exactly one next implementation target.

   The recommendation must be evidence-ranked.  Candidate categories include:

   - shared experts or other remaining MoE path;
   - row-parallel/NCCL communication and overlap;
   - a specific graph/runtime/layout boundary with concentrated cost;
   - another vLLM-aligned attention/indexer backend;
   - precision/cache path only if the profile clearly shows a precision-format
     bottleneck.

   Avoid selecting a target that only has vague potential.  The next target
   should have:

   - a named owner or subgraph;
   - measured current cost;
   - vLLM parity evidence or a clear local microbench reason;
   - expected E2E upside;
   - stop conditions.

10. Write `README.md`.

    The README must include:

    - milestone variant and expanded env;
    - correctness result;
    - macro table;
    - graph replay/eager decode table;
    - fresh top-bucket table;
    - vLLM parity table for top candidates;
    - memory ledger;
    - next target recommendation;
    - explicit do-not-continue notes.

## Gates

Correctness gate:

- TP8/page-size-256 text smoke must pass.

Milestone confirmation gate:

- 4096/1024/batch4 should remain above `114.07 output tok/s`;
- if it regresses below `114.07`, first confirm the variant expansion and graph
  replay/eager decode state before treating it as a real performance failure.

Graph gate:

- decode graph replay should remain active;
- eager decode should remain `0` for the measured decode loops.

Profile gate:

- the final report must explain at least `70%` of the measured decode-envelope
  time by bucket, or state clearly why the available traces cannot do so.

Next-target gate:

- the selected next target should plausibly offer at least `5%` E2E gain on
  4096/1024/batch4 or remove a top-two bottleneck that blocks later work;
- if no single candidate clears that bar, recommend a new measurement/parity
  target instead of an implementation target.

## Stop Conditions

Stop after producing the final report and one next-target recommendation.

Hard stops:

- correctness smoke fails after one focused diagnosis;
- graph replay is not active or eager decode is nonzero and cannot be explained
  by command/config error;
- two fresh macro repeats disagree by more than `5%` and no stable baseline can
  be established;
- the profile is missing CUDA kernels or NVTX attribution, and a short rerun
  does not fix it;
- the next proposed optimization is a generic cleanup rather than a named
  measured bottleneck.

Do not continue into implementation work in this thread.  Open a new target for
the selected next bottleneck.

