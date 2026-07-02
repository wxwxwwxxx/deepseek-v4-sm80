# TARGET 07.62: DSV4 SM80 `wo_a` Attention Boundary Parity

Date: 2026-07-02

## Goal

Adapt mini's `attn.wo_a` attention projection boundary toward the vLLM SM80
design, using an opt-in BF16 per-group BMM/cache path first.  The goal is to
remove the current decode-time `wo_a` dequant/layout/einsum chain without
changing mini's default precision policy.

This is a narrow owner-boundary target.  It is not a generic graph/layout pass
and it is not a full FP8/`fp8_ds_mla` precision target.

## Evidence From TARGET 07.61

Current mini best stack is the TARGET 07.60 three-owner cached BF16 variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Current macro:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `51.2962` | `132.3013` | `127` | `0` |
| 4096/1024/batch4 | `105.7645` | `132.5127` | `1023` | `0` |

Reference lines:

- old serving victory line: `114.07 output tok/s`;
- vLLM 4096/128/batch4 reference: about `82.28 output tok/s`;
- vLLM 4096/1024/batch4 reference: about `202.03 output tok/s`.

TARGET 07.61 did not recover reliable vLLM per-bucket timing because the fresh
vLLM Nsight repeat window contained no CUDA kernels.  The next decision is
therefore based on mini owner timing plus vLLM source-boundary parity.

Mini 07.60 owner evidence:

| Owner/boundary | Time/evidence |
| --- | ---: |
| `attn.wo_a` replay owner | `0.481377s` |
| `attn.wo_a` copy/layout | `0.290148s` |
| `attn.wo_a` elementwise | `0.137695s` |
| `attn.wo_a` intrinsic/GEMM | `0.053534s` |
| `attn.wo_a` share of decode envelope | `10.92%` |

If about `60%` of the non-GEMM `wo_a` chain can be removed, the expected
primary 4096/1024 E2E gain is roughly `5.3%`.  This may not alone beat
`114.07 output tok/s`, but it is the strongest single owner-boundary cut now
available.

## Source Parity Target

Mini current path:

- `python/minisgl/models/deepseek_v4.py`: `DeepSeekV4Attention` calls
  `dsv4_kernel.wo_a_grouped_projection_fallback`;
- `python/minisgl/kernel/deepseek_v4.py`: `wo_a_grouped_projection_fallback`
  does `dequant_fp8_weight(weight, scale)`, reshapes the result, and runs
  `torch.einsum("tgd,grd->tgr", ...)` every forward.

vLLM SM80 reference path:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`:
  `wo_a.is_bmm = True` and `wo_a.bmm_batch_size = n_local_groups`;
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`:
  `_ensure_wo_a_bmm_weight()` builds a per-group BF16 BMM weight cache, and
  `_apply_wo_a_bmm()` runs `torch.bmm`;
- the vLLM non-reference path also has fused inverse RoPE plus
  `deepseek_v4_fp8_einsum`, but this target should not jump to that precision
  lane before the BF16 BMM/cache path is tested.

The first implementation hypothesis is:

```text
decode-time dequant_fp8_weight + view + einsum
    -> graph/capture-prepared BF16 grouped BMM weight cache + torch.bmm
```

Expected mini cache layout:

```text
weight_bf16_bmm: [num_local_groups, d_per_group, o_lora_rank]
input o:         [tokens, num_local_groups, d_per_group]
output:          [tokens, num_local_groups * o_lora_rank]
```

The exact shape, bytes/rank, and KV-token/page cost must be derived from the
loaded model and recorded in the milestone.

## Scope

In scope:

- implement an opt-in `wo_a` BF16 grouped BMM weight cache;
- prebuild/cache it before CUDA graph replay, preferably during the existing
  DSV4 graph preparation path;
- avoid decode-time cache rebuilds or large allocations;
- preserve the current BF16 activation/cache precision policy;
- compare against vLLM source boundaries and the current mini owner profile;
- run focused microbench, text smoke, macro, memory ledger, and owner profile.

Out of scope:

- default-on precision changes;
- full `fp8_ds_mla` KV cache;
- adapting vLLM `deepseek_v4_fp8_einsum` as the first step;
- broad graph/runtime deforestation;
- shared-expert layout/overlap;
- communication/all-reduce changes;
- radix/prefix-cache work.

If BF16 BMM/cache fails the gates, the final report may recommend a separate
future target for fused inverse RoPE plus FP8 `wo_a` einsum, but it should not
silently expand this target into that implementation.

## Work Plan

1. Create milestone artifacts under:

   ```text
   performance_milestones/target07_wo_a_attention_boundary_parity/
   ```

   Use `raw/`, `summaries/`, and `scripts/` subdirectories.

2. Re-read the source boundaries.
   Confirm mini's current `wo_a` shape, grouping, dtype, and scale semantics.
   Confirm the vLLM `_ensure_wo_a_bmm_weight()` and `_apply_wo_a_bmm()` logic,
   but prefer direct mini weight dequantization over vLLM's identity-matrix
   recovery if mini has direct access to the unpacked FP8 weight and scale.

3. Add an opt-in flag and benchmark/text-smoke variant.
   Suggested flag:

   ```text
   MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE=1
   ```

   Suggested variant suffix:

   ```text
   woabf16bmmcache
   ```

   Keep default behavior unchanged.

4. Implement cache preparation.
   The cache should be built before graph replay and should not rebuild inside
   decode/capture/replay.  A missing or stale cache on the opt-in path should
   raise clearly instead of silently falling back during graph replay.

   Preferred implementation shape:

   - dequantize `self.wo_a.weight` once to BF16;
   - reshape to `[num_local_groups, o_lora_rank, d_per_group]`;
   - transpose/contiguous to `[num_local_groups, d_per_group, o_lora_rank]`;
   - store it as a persistent per-layer cache;
   - in forward, reshape `o` to `[tokens, num_local_groups, d_per_group]`,
     transpose to `[num_local_groups, tokens, d_per_group]`, run `torch.bmm`,
     transpose back, and flatten to `[tokens, num_local_groups * o_lora_rank]`.

5. Add focused microbench coverage.
   Use real loaded `wo_a` weights if feasible.  Compare current fallback vs
   cached BF16 BMM for representative token counts, at least:

   ```text
   M = 1, 4, 8, 16
   ```

   Record current total ms, cached BMM total ms, cache-build time, max abs/rel
   error, and whether outputs match existing tolerances.

6. Add memory ledger.
   Report:

   - cached layers;
   - shape per layer/rank;
   - bytes/rank;
   - GiB/rank;
   - equivalent KV tokens/rank;
   - equivalent KV pages/rank;
   - peak allocated/reserved delta if measurable.

7. Validate correctness and graph behavior.
   Required:

   - `py_compile` touched Python files and helper scripts;
   - focused microbench correctness;
   - TP8/page-size-256 text smoke with simple prompts;
   - graph replay count preserved and eager decode count remains `0`.

8. Run macro benchmarks.
   Required workloads:

   - 4096/128/batch4;
   - 4096/1024/batch4.

   Compare against the 07.60 baseline:

   | Workload | Baseline output tok/s | New output tok/s | Gain | Graph replay | Eager decode |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | 4096/128/batch4 | `51.2962` | TBD | TBD | TBD | TBD |
   | 4096/1024/batch4 | `105.7645` | TBD | TBD | TBD | TBD |

9. Run a fresh owner/profile pass if the macro or microbench indicates a win.
   The report must show what happened to `attn.wo_a`:

   | Metric | 07.60 baseline | New path | Change |
   | --- | ---: | ---: | ---: |
   | replay owner | `0.481377s` | TBD | TBD |
   | copy/layout | `0.290148s` | TBD | TBD |
   | elementwise | `0.137695s` | TBD | TBD |
   | intrinsic/GEMM/BMM | `0.053534s` | TBD | TBD |

10. Write the final README.
    Include current best, implementation summary, source parity notes, memory
    ledger, microbench table, macro table, owner-profile table, decision, and
    do-not-continue conditions.

## Success Gates

Correctness gates:

- text smoke passes;
- graph replay is preserved;
- eager decode stays `0`;
- output parity vs current fallback is within an explicitly reported tolerance;
- no decode-time cache rebuild or large allocation is observed on the opt-in
  path.

Performance gates:

- focused `wo_a` microbench improves total projection time by at least `30%`;
- fresh owner profile reduces `attn.wo_a` replay owner time by at least `15%`,
  or 4096/1024 output throughput improves by at least `5%`;
- no 4096/128 or 4096/1024 macro regression greater than `1%`;
- promotion is strongest if 4096/1024 output throughput reaches or exceeds
  `114.07 output tok/s`.

If the microbench wins but macro does not move, stop after one fresh profile and
explain which broader bucket absorbed the gain.

## Decision Rules

Promote the opt-in path if correctness passes and either:

- 4096/1024 output throughput improves by at least `5%`; or
- `attn.wo_a` owner time drops by at least `15%` and macro has no meaningful
  regression, with a clear follow-up target explaining the remaining gap.

Do not continue in this target if:

- graph replay breaks or eager decode becomes nonzero after one focused fix;
- text smoke fails after one focused fix;
- the implementation requires changing default precision policy;
- the BF16 BMM/cache path does not beat the focused microbench gate;
- the next useful change is fused inverse RoPE plus FP8 `wo_a` einsum;
- the next useful change is shared-expert layout, communication, or broad graph
  deforestation.

The final decision should say one of:

- promote `wo_a` BF16 BMM/cache and reprofile remaining gap;
- keep it as opt-in only and move to the next evidence-backed target;
- stop `wo_a` BF16 BMM/cache and open a separate precision-boundary target;
- stop because the expected owner gain is not real.
