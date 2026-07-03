# TARGET 07.74: DSV4 SM80 vLLM FP8 Marlin Dense Projection Runtime Opt-In

Date: 2026-07-02

## Goal

Integrate the TARGET 07.73 passing standalone backend into mini as a bounded
runtime opt-in:

```text
vLLM FP8 Marlin W8A16 block linear
```

The target is a runtime proof, not a broad precision rewrite.  It should test
whether the standalone Marlin gains survive mini's actual TP8 model path,
CUDA graph capture, owner scheduling, and text correctness gates.

Primary candidate toggle:

```text
MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1
```

Suggested candidate variant:

```text
dsv4_sm80_a100_victory_fp8marlinproj
```

Do not add this path to `dsv4_sm80_a100_victory` unless all promotion gates
pass.

## Baseline

Use the promoted path as the baseline:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Known promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

Always compare baseline and candidate in the same run.  Do not use inactive
opt-ins as the baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
```

## Starting Evidence From TARGET 07.73

TARGET 07.73 tested standalone quantized-linear backends under the vLLM venv.
Lane A passed:

```text
vllm_fp8_marlin_w8a16_block
```

Representative M=`4` rows:

| Owner | Promoted cached BF16 ms | vLLM block Marlin ms | Speedup |
| --- | ---: | ---: | ---: |
| WQA/WKV/compress | `0.100167` | `0.077816` | `22.31%` |
| `q_wqb` | `0.092440` | `0.066223` | `28.36%` |
| `wo_b` local | `0.091272` | `0.064005` | `29.87%` |
| shared experts gate/up | `0.098032` | `0.079082` | `19.33%` |
| shared experts down | `0.092495` | `0.064094` | `30.70%` |
| `wo_a` grouped two-launch | `0.057180` | `0.189748` | reject |

All-M gate passed for:

- `attn.q_wqb`;
- `attn.wo_b` local projection;
- shared experts down.

Lane A is preferred because it preserves the native DeepSeek V4 block FP8
`weight_scale_inv` contract.  FBGEMM-derived Marlin also passed local timing,
but it requires load-time block-FP8 -> BF16 -> per-channel-FP8 conversion and
should not be the first runtime integration route.

## vLLM Contract To Preserve

The runtime path should align with vLLM's SM80 Marlin behavior:

- use native checkpoint FP8 weights and block `weight_scale_inv`;
- run vLLM-style block processing, Marlin repack, scale expansion/permutation,
  and FP8 exponent-bias fusion once before CUDA graph capture;
- keep activations BF16 for the Marlin GEMM;
- do not insert replay-time activation FP8 quantization;
- do not do replay-time weight dequantization, scale conversion, repacking, or
  workspace allocation;
- call the Marlin GEMM steady-state path during decode replay.

Relevant vLLM source:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/kernels/linear/scaled_mm/marlin.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py
```

## Runtime Backend Availability Gate

TARGET 07.73 ran standalone under:

```text
/workspace/venvs/vllm-dsv4/bin/python
```

Mini runtime may not be able to import vLLM or `vllm._custom_ops` in the normal
mini environment.  Therefore 07.74 must start with a runtime availability
probe.

Required checks:

- can the mini torchrun environment import or call the needed Marlin custom
  ops?
- can the vLLM venv run mini's benchmark scripts with `PYTHONPATH` pointing at
  this repository?
- does the project already have a mini-owned Marlin custom-op bridge that can
  be reused or extended?
- if a full vLLM dependency is required, is it acceptable only for this opt-in,
  or should the target stop and propose a smaller bridge target?

Stop early if no safe runtime bridge exists.  The final README should then
record the exact missing op/package and propose the bridge work as the next
target.

## Owner Scope

Phase A, first integration subset:

| Owner | Why |
| --- | --- |
| `attn.q_wqb` | Passed all-M standalone gate; no all-reduce in local projection. |
| `attn.wo_b` local projection | Passed all-M standalone gate; measure local compute separately from row-parallel all-reduce. |
| shared experts down | Passed all-M standalone gate; small but clean dense owner. |

Phase B, only if Phase A passes profile and correctness gates:

| Owner | Why |
| --- | --- |
| WQA/WKV/compress | Standalone speedup exists, but this owner may have more fused-boundary complexity. |
| shared experts gate/up | Standalone speedup exists, but activation/error behavior should be checked after shared-down first. |

Explicitly out of scope:

- `wo_a` grouped two-launch Marlin route;
- FBGEMM-derived per-channel conversion route;
- INT8 W8A8 projection;
- full FP8 KV-cache / `fp8_ds_mla` E2E;
- HC/router precision changes;
- MoE routed expert backend changes.

## Weight And Memory Lifecycle

This target must avoid accidentally doubling memory.

The promoted path currently uses cached BF16 dequantized projection weights for
several owners.  The Marlin candidate should replace those owner-specific
cached BF16 weights, not sit beside them permanently.

Rules:

- For an owner switched to Marlin, do not allocate its cached BF16 projection
  weight unless needed for an explicit debug/fallback mode.
- Build Marlin-packed weights/scales before CUDA graph capture.
- If the original checkpoint FP8 tensor or old cached BF16 tensor is not used
  after successful Marlin packing, release its Python/module reference.
- If fallback requires keeping the original tensor, record that as a deliberate
  memory cost and do not promote the path until the cost is justified.
- Take memory snapshots before and after packing, after releasing superseded
  tensors, and after graph capture.
- The README must report persistent bytes/rank, workspace bytes/rank, and
  equivalent KV-cache token/page cost.

Implementation note:

```text
Deleting a tensor reference is not enough if another module attribute still
points to the same storage.  Audit module attributes and cache registries.
For measurement, it is fine to run gc.collect() and torch.cuda.empty_cache()
after deleting superseded references, but never put allocator cleanup in the
decode replay path.
```

Promotion-quality memory behavior:

- no duplicate cached BF16 + Marlin packed weights for the same promoted owner;
- no decode-time allocation or repack;
- memory ledger shows Marlin replacement saves or at least does not materially
  increase persistent bytes versus the current promoted BF16 cache for the
  switched owners.

## Implementation Plan

1. Create milestone artifacts:

```text
performance_milestones/target07_vllm_fp8_marlin_dense_projection_runtime/
```

Use:

```text
scripts/
raw/
summaries/
```

2. Run backend bridge probes.

Suggested artifact:

```text
raw/runtime_marlin_backend_availability.json
summaries/runtime_marlin_backend_bridge.md
```

If the bridge cannot be made safe in this target, stop and write the README.

3. Add a default-off runtime opt-in.

Suggested toggle:

```text
MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1
```

Suggested variant:

```text
dsv4_sm80_a100_victory_fp8marlinproj
```

The variant should include the promoted victory bundle plus this new opt-in.

4. Implement Phase A owner replacement.

Required behavior:

- preserve existing promoted path when the toggle is off;
- load or retain the native FP8 checkpoint weights/scales needed for Marlin;
- prepare Marlin weights/scales before CUDA graph capture;
- skip or release superseded cached BF16 weights for switched owners;
- call Marlin only for the selected local projection compute;
- preserve row-parallel all-reduce around `wo_b`;
- preserve graph replay and eager decode count `0`.

5. Add focused owner microbench and quality checks.

Compare Phase A owners against the promoted cached BF16 path:

- M=`1,4,8,16`;
- max/mean/p99 abs error;
- cosine;
- latency with no replay-time repack or allocation;
- memory bytes for both paths.

6. Run TP8 correctness smoke before macro.

Use page size `256` and the same model path:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_fp8marlinproj \
  --output /tmp/dsv4_target0774_text_smoke.json
```

If the variant name is not wired into the benchmark harness yet, add it there
or document the equivalent env invocation.

7. Run same-run macro.

Minimum:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_fp8marlinproj \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --output-dir /tmp/dsv4_target0774_marlin_4096x128_bs4 \
  --keep-going
```

Also run 4096/1024/batch4 before any promotion claim.

8. Capture a fresh 4096/128 profile.

The README must compare:

- projection/GEMM bucket;
- Phase A owners;
- row-parallel `wo_b` all-reduce;
- graph replay count;
- eager decode count;
- any new copy/layout/allocator events caused by Marlin packing.

## Gates

Backend bridge gate:

- mini runtime can call the Marlin backend, or the target stops with a bridge
  plan;
- no full-vLLM dependency is silently added to the promoted path;
- opt-in import failure must degrade cleanly with a clear error.

Correctness gate:

- focused owner output comparisons are acceptable versus promoted cached BF16;
- TP8 text smoke passes for page size `256`;
- graph replay remains active;
- eager decode remains `0`.

Performance gate for keeping the opt-in:

- Phase A focused owner latency retains most of the 07.73 standalone signal;
- 4096/128 profile shows at least `0.03s` projection/GEMM reduction, or the
  README explains why the owner timing moved elsewhere;
- 4096/1024 same-run macro does not regress.

Promotion gate:

- Phase A profile reduction is at least `0.04s` on 4096/128 rank0;
- 4096/1024 same-run output tok/s improves by at least `1.5%`;
- quality and text smoke pass;
- persistent memory does not increase for switched owners after releasing
  superseded cached BF16/original tensors;
- no new decode-time allocations, repacks, or scale conversions appear.

Expansion gate:

- only try Phase B if Phase A passes correctness and shows a real profile
  reduction;
- after Phase B, require at least `0.07s` projection/GEMM reduction before any
  broader promotion discussion.

## Required Final README

Write:

```text
performance_milestones/target07_vllm_fp8_marlin_dense_projection_runtime/README.md
```

It must include:

- backend bridge status and interpreter/env used;
- exact toggle and variant;
- owners implemented;
- weight/scale preparation description;
- memory lifecycle ledger, including what was released;
- focused owner latency and quality tables;
- TP8 text smoke result;
- 4096/128 and 4096/1024 same-run macro tables;
- fresh profile owner/bucket comparison;
- graph replay/eager decode counts;
- decision: promote, keep opt-in, expand to Phase B, stop, or open bridge
  target;
- do-not-continue condition.

## Stop Conditions

Stop without broadening the target if:

- Marlin backend cannot be safely called from mini runtime;
- Phase A fails text smoke or hidden/logit quality checks;
- graph replay breaks or eager decode becomes nonzero;
- Phase A profile reduction is below `0.03s` and macro is flat/regressed;
- implementation requires keeping both cached BF16 and Marlin-packed weights
  for the same owners without a clear memory plan;
- the next idea is `wo_a`, INT8, FBGEMM-derived conversion, full FP8 KV cache,
  or HC/router precision.

Those should be separate targets with separate quality and memory gates.
