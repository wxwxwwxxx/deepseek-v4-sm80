# Target 07.57: DSV4 SM80 Projection/GEMM Backend Parity

Date: 2026-07-02

## Result

This target completed the requested owner-level projection/GEMM attribution first. No large kernel PoC was implemented or promoted.

The strongest evidence is a backend contract, not a single isolated owner:

| Gate | Status | Evidence |
| --- | --- | --- |
| At least 0.50s attributed projection/GEMM owner or backend contract | PASS | `_quantized_linear_fp8_kernel` across `attn.q_wqb`, `attn.wo_b`, and `indexer.wq_b` is `1.172645s` inside the decode replay envelope. |
| Focused projection/GEMM reduction >=15% or 4096/128 output tok/s +5% | NOT CLAIMED | Attribution-only target; no big kernel or backend replacement was landed. |
| CUDA graph replay preserved | PASS | Nsight run had `127` replayed decode steps; text smoke had `9` graph replays. |
| Eager decode count remains 0 | PASS | Nsight summary and text smoke both report eager decode `0`. |

Decision: pivot to a focused projection backend-contract target for `_quantized_linear_fp8_kernel` on SM80 small-M decode, starting with `attn.q_wqb`, then `attn.wo_b`, then `indexer.wq_b`. This should be a vLLM-boundary/retuned-small-M-kernel experiment, not another generic graph/layout pass and not an activation-quant-only fusion.

## Baseline

Active 07.54/07.55 variant:

`v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`

| Case | Baseline output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `43.0685` | `104.2028` | `127` | `0` |
| 4096/1024/batch4 | `87.0831` | n/a | `1023` | `0` |

07.56 static scale cache only reached `43.2194` output tok/s on 4096/128/batch4, or `+0.35%`, so this target did not continue broad graph/layout or scale-cache micro-work.

## Attribution Run

Command:

```bash
performance_milestones/target07_projection_gemm_backend_parity/scripts/nsys_projection_owner_4096x128_bs4.sh
```

Key artifacts:

- `raw/nsys_target0757_projection_owner_4096x128_bs4_np128_rank0.nsys-rep`
- `raw/nsys_target0757_projection_owner_4096x128_bs4_np128_rank0.sqlite`
- `summaries/nsys_target0757_projection_owner_4096x128_bs4_np128_rank0_projection_owner.md`
- `summaries/nsys_target0757_projection_owner_4096x128_bs4_np128_rank0_projection_owner.json`
- `summaries/mini_projection_owner_4096x128_bs4_np128_nsys_summary.json`

The profiling run preserved graph replay but should not be compared directly with the non-Nsight baseline:

| Metric | Value |
| --- | ---: |
| Output tok/s under Nsight | `42.56588382063167` |
| Decode tok/s under Nsight | `97.64098863887658` |
| Decode graph replay count | `127` |
| Eager decode count | `0` |
| Decode envelope | `5.436976s` |
| Projection/GEMM intrinsic bucket by kernel names | `1.796818s` |
| Owner-attributed projection/GEMM intrinsic | `1.462901s` |
| Unattributed projection/GEMM intrinsic | `0.333917s` |

## Owner Table

| Owner | Intrinsic GEMM s | Runtime/copy/layout s | Notes |
| --- | ---: | ---: | --- |
| `attn.q_wqb` | `0.404178` | `0.029766` | `_quantized_linear_fp8_kernel`; lifted query projection. |
| `attn.wo_b` | `0.403710` | `0.028501` | `_quantized_linear_fp8_kernel` plus row-parallel all-reduce (`0.169172s`). |
| `indexer.wq_b` | `0.364756` | `0.018079` | `_quantized_linear_fp8_kernel`; indexer query projection. |
| `attn.q_proj_wqa_wkv` | `0.089755` | `0.029276` | Active `fwqakvcache` path is already much smaller. |
| `attn.wo_a` | `0.053440` | `0.427810` | Total owner work is `0.481250s`, but intrinsic GEMM is tiny; mostly staging/layout and elementwise. |
| `shared_experts.gate_up_proj` | `0.045668` | `0.228309` | Not the first projection/GEMM owner. |
| `shared_experts.down_proj` | `0.029229` | `0.166127` | Not the first projection/GEMM owner. |
| `indexer.compressor` | `0.026728` | `0.062478` | Context only. |
| `lm_head` | `0.026654` | `0.044760` | Context only. |
| `indexer.weights_proj` | `0.018783` | `0.000000` | Context only. |

Backend-contract total:

```text
attn.q_wqb _quantized_linear_fp8_kernel   0.404178s
attn.wo_b  _quantized_linear_fp8_kernel   0.403710s
indexer.wq_b _quantized_linear_fp8_kernel 0.364756s
----------------------------------------------------
contract total                            1.172645s
```

This is the only evidence-strong first PoC candidate. `attn.wo_a` is tempting because the owner total is close to 0.50s, but it does not explain the 1.7968s projection/GEMM bucket: its intrinsic GEMM is only `0.053440s`.

## vLLM Boundary Check

Source-level comparison used these vLLM files:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`

Findings:

- vLLM keeps `wq_b` and `wo_a` behind `ColumnParallelLinear`, `wo_b` behind `RowParallelLinear`, and applies quantization through `quant_method.apply`.
- `QuantFP8` uses the `scaled_fp8_quant` custom-op boundary instead of mini's current activation-quant wrapper path.
- `Fp8LinearMethod` can select a Marlin FP8 scaled-mm path on SM80 or a batch-invariant dequantized BF16 GEMM boundary.
- vLLM's DeepSeek-V4 attention treats `wo_a` specially through the SM80 BMM/reference branch or the `deepseek_v4_fp8_einsum` branch, but mini's attribution says this is not the first owner to port.
- Mini's active `q_proj/wqa+wkv` path is already small under `fwqakvcache`; it should not be the first parity target.

Runtime vLLM probing was blocked by a local PyTorch/vLLM ABI mismatch:

```text
ImportError: /workspace/vllm-dsv4-docker/vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

The source-level comparison is still sufficient for owner selection because mini's Nsight attribution identifies the backend contract directly.

## Real-Weight Microbench

Command:

```bash
python performance_milestones/target07_projection_gemm_backend_parity/scripts/microbench_real_fp8_linear_contract.py --warmup 5 --iters 20
```

Artifacts:

- `raw/real_fp8_linear_microbench.json`
- `summaries/real_fp8_linear_microbench.md`

Summary over M values `[1, 4, 8, 16]`:

| Owner | Shape | Avg wrapper ms | Avg intrinsic cached-scale ms | Intrinsic share | Cached-dequant BF16 F.linear ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb.layer0` | K=1024, N=32768 | `0.4120` | `0.3973` | `96.4%` | about `0.053` |
| `attn.wo_b.layer0` | K=8192, N=4096 | `0.6603` | `0.6446` | `97.6%` | about `0.052` |
| `indexer.wq_b.layer2` | K=1024, N=8192 | `0.1677` | `0.1551` | `92.4%` | about `0.019` |

All microbench correctness checks passed. The wrapper is dominated by the intrinsic `_quantized_linear_fp8_kernel`, while activation quant and static-scale handling are too small to justify another scale-cache/layout-only change. A future PoC should therefore change the linear backend contract itself: retune/replace the SM80 small-M FP8 kernel, or test an opt-in vLLM-like packed/dequantized-weight boundary with explicit memory accounting.

## Code Changes

- Added debug-only NVTX owner ranges in `python/minisgl/models/deepseek_v4.py`, guarded by the existing `MINISGL_DSV4_GRAPH_CAPTURE_NVTX` / `_dsv4_capture_nvtx` path.
- Added `scripts/summarize_projection_owner_nsys.py` to map capture-time owner NVTX ranges to replay graph nodes through `CUDA_GRAPH_NODE_EVENTS.originalGraphNodeId`.
- Added `scripts/nsys_projection_owner_4096x128_bs4.sh` as the reproducible owner-attribution runner.
- Added `scripts/microbench_real_fp8_linear_contract.py` for real-weight FP8 linear backend-contract timing.

## Validation

```bash
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  performance_milestones/target07_projection_gemm_backend_parity/scripts/summarize_projection_owner_nsys.py \
  performance_milestones/target07_projection_gemm_backend_parity/scripts/microbench_real_fp8_linear_contract.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --page-size 256 \
  --output performance_milestones/target07_projection_gemm_backend_parity/raw/text_smoke.json
```

Passed with graph capture sizes `[4, 2, 1]`, replay count `9`, greedy sample replay count `9`, and eager decode count `0`.

## Next PoC

Use one owner/backend contract only:

```text
PoC target: DSV4Linear FP8 small-M backend contract for attn.q_wqb
Evidence: 0.404178s owner intrinsic, shares the same backend contract with a 1.172645s total across q_wqb/wo_b/indexer.wq_b
Success gate: focused projection/GEMM contract time -15% or 4096/128/batch4 output tok/s +5%, graph replay preserved, eager decode 0
```

Then apply the same backend to `attn.wo_b` and `indexer.wq_b` if the first owner passes.
