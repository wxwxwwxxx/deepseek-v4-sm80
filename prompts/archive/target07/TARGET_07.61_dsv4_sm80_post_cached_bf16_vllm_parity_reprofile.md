# TARGET 07.61: Post-Cached-BF16 vLLM Parity Reprofile

Date: 2026-07-02

## Goal

After TARGET 07.60, the three large mini `_quantized_linear_fp8_kernel`
owners have opt-in cached BF16 dequantized-weight backends:

- `attn.q_wqb`;
- `attn.wo_b`;
- `indexer.wq_b`.

This target must freeze that new mini baseline, compare it against the vLLM
DeepSeek V4 sm80 path, and choose the next implementation target from
evidence.  Do not start another broad optimization before the parity table is
complete.

## Current Mini Baseline

Use this opt-in variant unless the codebase has renamed it:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Required toggles are represented by the variant, including:

- `MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE=1`;
- `MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE=1`;
- `MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE=1`.

Latest recorded macro:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `51.2962` | `132.3013` | `127` | `0` |
| 4096/1024/batch4 | `105.7645` | `132.5127` | `1023` | `0` |

Memory cost of the three cached BF16 owners:

| Cached owners | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | ---: | ---: |
| `q_wqb + wo_b + indexer.wq_b` | `1.0000` | `14121.79` | `55.16` |

Reference lines:

- old serving victory line: `114.07 output tok/s` at 4096/1024/batch4;
- prior vLLM offline reference: about `82.28 output tok/s` at
  4096/128/batch4 and about `202.03 output tok/s` at 4096/1024/batch4.

The vLLM path may use different precision/cache choices.  Treat vLLM as the
performance and backend-behavior reference, not as an automatic precision
contract for mini.

## Inputs To Read

Primary mini records:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`;
- `prompts/TARGET_07.57_dsv4_sm80_projection_gemm_backend_parity.md`;
- `prompts/TARGET_07.58_dsv4_sm80_cached_bf16_projection_backend.md`;
- `prompts/TARGET_07.59_dsv4_sm80_cached_bf16_wo_b_projection_backend.md`;
- `prompts/TARGET_07.60_dsv4_sm80_cached_bf16_indexer_wq_b_projection_backend.md`;
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/README.md`;
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/summaries/nsys_target0760_qwqb_wob_idxwqb_post_sequence_buckets.md`;
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/summaries/nsys_target0760_qwqb_wob_idxwqb_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`;
- `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/summaries/qwqb_wob_idxwqb_memory_ledger.md`.

vLLM reference locations:

- source root: `/workspace/vllm-dsv4-docker`;
- virtualenv: `/workspace/venvs/vllm-dsv4`;
- DSV4 model:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`;
- DSV4 attention:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`;
- DSV4 attention ops:
  `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`;
- quantization:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`;
- fused MoE:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`;
- prior profile artifacts, if present:
  `performance_milestones/vllm/raw/`.

## Scope

In scope:

- run or reuse a fresh mini profile for the 07.60 three-owner cached BF16
  stack;
- run or reuse a same-shape vLLM profile when feasible;
- compare mini and vLLM at macro, bucket, owner, graph-boundary, and source
  dispatch levels;
- identify the next highest-confidence mini implementation target.

Out of scope:

- landing a large optimization PoC;
- expanding cached BF16 weights to new owners without fresh owner evidence;
- changing mini default precision;
- enabling full `fp8_ds_mla` KV cache E2E;
- radix/prefix-cache work;
- broad refactors of cache/workspace ownership.

Small instrumentation changes, profile scripts, and report-generation scripts
are allowed.

## Work Plan

1. Create `performance_milestones/target07_post_cached_bf16_vllm_parity_reprofile/`.
   Put raw profiles under `raw/`, derived tables under `summaries/`, helper
   scripts under `scripts/`, and the final conclusion in `README.md`.

2. Freeze the mini baseline.
   Record the exact git status, variant string, environment toggles, page size,
   `num_pages`, TP size, model path, and macro result.  If rerunning mini is
   affordable, run both 4096/128/batch4 and 4096/1024/batch4; otherwise reuse
   TARGET 07.60 artifacts and clearly label them as reused.

3. Collect vLLM evidence.
   Prefer existing vLLM artifacts if they are already same-shape and trustworthy.
   If a fresh vLLM profile is run, keep it short first, ideally 4096/128/batch4
   with one repeat and no warmup.  vLLM has previously shown OOM/hang behavior
   on some sparse prefill paths in this container, so do not spend the whole
   target debugging vLLM execution.  If vLLM cannot run cleanly, fall back to
   source-level dispatch parity and mark runtime parity as incomplete.

4. Produce a macro parity table.
   Include at least:

   | Framework | Workload | Output tok/s | Decode tok/s | TTFT/prefill note | Precision/cache note |
   | --- | --- | ---: | ---: | --- | --- |

   The table must make clear whether vLLM uses packed FP8/indexer/cache paths
   that mini does not use by default.

5. Produce a bucket parity table.
   Start from the 07.60 mini decode-envelope buckets:

   | Mini bucket | Mini time s | Mini share | vLLM comparable bucket | vLLM time s | Decision note |
   | --- | ---: | ---: | --- | ---: | --- |
   | graph/runtime/copy/cat/index | `1.141006` | `25.88%` | TBD | TBD | largest broad mini cluster |
   | projection/GEMM | `0.805080` | `18.26%` | TBD | TBD | after cached BF16 owners |
   | elementwise graph nodes | `0.639409` | `14.50%` | TBD | TBD | includes layout-adjacent work |
   | NCCL communication | `0.346671` | `7.86%` | TBD | TBD | includes `wo_b` all-reduce |
   | MoE/Marlin | `0.316854` | `7.19%` | TBD | TBD | not currently primary |
   | FP8 indexer cache/logits/topk | `0.131028` | `2.97%` | TBD | TBD | mini has vLLM-aligned indexer backend |
   | sparse attention decode | `0.118408` | `2.69%` | TBD | TBD | split-K path mostly solved |

6. Produce an owner/boundary parity table.
   At minimum compare these mini owners with their vLLM source dispatch and,
   where possible, vLLM profile timing:

   | Owner/boundary | Mini 07.60 evidence | vLLM evidence to find | Next-action implication |
   | --- | --- | --- | --- |
   | `attn.wo_a` | replay kernel `0.481377s`; copy/layout `0.290148s`; elementwise `0.137695s` | vLLM `wo_a` layout, BMM/einsum/custom op boundary | possible narrow graph/layout target |
   | shared experts gate/up/down | gate/up copy/layout `0.169527s`; down copy/layout `0.129588s`; combined owner replay about `0.469s` | vLLM shared-expert projection/fusion path | possible shared-expert boundary target |
   | `attn.wo_b` all-reduce | owner communication about `0.159649s`; total NCCL `0.346671s` | vLLM communication count, dtype, overlap, graph boundary | possible comm/overlap target only if vLLM is clearly better |
   | graph/runtime/copy/cat/index | broad mini bucket `1.141006s` | vLLM graph node count and input/output layout behavior | broad target only if sub-boundary is identified |
   | remaining projection/GEMM | owner-attributed `0.469287s`, unattributed `0.335793s` | vLLM projection backend coverage after FP8/cached paths | avoid blind cached-weight expansion |
   | MoE/Marlin | `0.316854s` bucket | vLLM MoE backend timing | not primary unless vLLM shows a large gap |

7. Choose exactly one next target.
   Use the following gate:

   - choose `wo_a`/attention projection boundary adaptation if vLLM avoids
     mini's `wo_a` copy/layout chain and the expected macro gain is at least
     `5%`;
   - choose shared-expert projection/layout if vLLM has a fused or lower-copy
     boundary and expected macro gain is at least `5%`;
   - choose communication/overlap only if vLLM has materially lower NCCL cost
     or a clearly better overlap schedule;
   - choose graph/layout deforestation only after naming the concrete copied,
     concatenated, indexed, or reshaped boundary, not as a generic bucket;
   - choose precision/cache only if vLLM's advantage is tied to a specific
     precision backend and the target can state its quality risk.

## Required README Output

The milestone README must include:

- mini baseline table;
- vLLM baseline table, or a clear explanation of why runtime vLLM evidence is
  unavailable;
- mini-vs-vLLM macro parity table;
- bucket parity table;
- owner/boundary parity table;
- source-dispatch notes with file paths;
- next target recommendation;
- do-not-continue conditions.

## Stop Rules

Stop this target once a next target is selected from evidence.  Do not spend
time implementing speculative fixes after the parity report is good enough.

Hard stops:

- vLLM runtime collection is blocked after one focused environment/profiling
  fix attempt; continue with source parity and mark the gap;
- no candidate has at least `5%` expected E2E gain or a top-two bucket owner;
- the next change would require a precision-policy decision not covered by
  this target;
- mini text smoke fails and is not fixed by one focused correctness attempt.

The final decision should say one of:

- proceed to a named `wo_a`/attention boundary target;
- proceed to a named shared-expert projection/layout target;
- proceed to a named communication/overlap target;
- proceed to a named precision/cache target;
- rerun broader profiling because parity evidence is insufficient.
