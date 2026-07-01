# TARGET 07.51: vLLM FP8 Backend Parity

## Status

Completed on 2026-07-01 on A100 sm80.

Decision: port/adapt vLLM FP8 indexer backend next.

This does not revive the current mini-owned software FP8 indexer logits kernel.
TARGET 07.50 remains a failed slice: quality passed, but same-run 4096/128/bs4
macro dropped from `37.9237` to `29.6691` output tok/s.  The new evidence is
that vLLM's native indexer backend is much faster than mini's FP8 kernel and
faster than mini bf16 on the representative large indexer shape.

No 4096/1024 macro was run in this target.  The target was deliberately kept to
isolated backend microbench.

## Inputs Preserved From 07.50

07.50 mini microbench, A100 sm80:

| Case | Mini bf16 logits ms | Mini FP8 logits ms | Mini bf16 select ms | Mini FP8 select ms | Mini FP8 Q quant ms | Mini FP8 K store ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch1, history1024 | `0.1271` | `0.1573` | `0.3093` | `0.3182` | `0.2317` | `0.1715` |
| batch4, history2048 | `0.1246` | `0.2655` | `0.3037` | `0.3163` | `0.2292` | `0.1714` |
| batch16, history4096 | `0.3076` | `1.3072` | `0.3586` | `1.7368` | `0.2308` | `0.2941` |

07.50 macro:

| Variant | 4096/128/bs4 output tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| exact control | `37.9237` | `79.8574` | `4.3114` |
| mini FP8 indexer cache/logits | `29.6691` | `81.5617` | `6.7446` |

Conclusion preserved: stop optimizing the current mini-owned FP8 logits kernel
unless vLLM native backend evidence clears the microbench threshold.

## vLLM Checkout Hygiene

vLLM root: `/workspace/vllm-dsv4-docker`.

Branch during measurement: `minisgl_docker`.

The checkout already had the two TARGET 07.43 env-gated ablation edits:

- `vllm/model_executor/layers/deepseek_v4_attention.py`
- `vllm/model_executor/layers/sparse_attn_indexer.py`

The ablation envs were unset during this run:

- `VLLM_DSV4_ABLATE_AUX_STREAM=None`
- `VLLM_DSV4_ABLATE_PERSISTENT_TOPK=None`

No vLLM source was edited for TARGET 07.51.  The pre-existing diff snapshot is
recorded at:

- `summaries/vllm_preexisting_0743_ablation_patch_snapshot.diff`

The untracked vLLM ncu report directories were left untouched:

- `/workspace/vllm-dsv4-docker/benchmarks/kernels/mqa_logits_triton_variants/ncu_reports/`
- `/workspace/vllm-dsv4-docker/benchmarks/kernels/ncu_reports/`

## Artifacts

Script:

- `scripts/vllm_fp8_backend_microbench.py`

Raw:

- `raw/vllm_fp8_backend_microbench.json`
- `raw/vllm_fp8_backend_microbench_quick.json`

Summaries:

- `summaries/vllm_fp8_backend_microbench_summary.json`
- `summaries/vllm_fp8_backend_microbench_quick_summary.json`

Command:

```bash
source /workspace/venvs/vllm-dsv4/bin/activate
source /workspace/mini-sglang/performance_milestones/vllm/scripts/vllm_env.sh
setup_vllm_runtime_env
python /workspace/mini-sglang/performance_milestones/target07_vllm_fp8_backend_parity/scripts/vllm_fp8_backend_microbench.py
```

Validation:

```bash
python -m py_compile performance_milestones/target07_vllm_fp8_backend_parity/scripts/vllm_fp8_backend_microbench.py
python -m black --check performance_milestones/target07_vllm_fp8_backend_parity/scripts/vllm_fp8_backend_microbench.py
```

## Isolated Backend Pieces

| Backend piece | Isolation result | Source path |
| --- | --- | --- |
| `fused_indexer_q_rope_quant` FP8 Q path | Isolated and timed directly.  Q bytes exactly matched the vLLM torch reference in the synthetic probe. | `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` |
| FP8 indexer K/cache quant/store | Isolated via `vllm._custom_ops.indexer_k_quant_and_cache`; `cp_gather_indexer_k_quant_cache` also isolated.  This measures K rows already produced by the compressor, not the full model compressor. | `/workspace/vllm-dsv4-docker/csrc/cache_kernels.cu`, `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py` |
| `fp8_paged_mqa_logits_triton` decode logits | Isolated and timed directly over vLLM's packed indexer cache layout. | `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/mqa_logits_triton.py` |
| `fp8_mqa_logits_triton` prefill/gathered logits | Isolated and timed directly. | same as above |
| `gather_dequant_two_scopes_with_mask` | Isolated over reference-packed `fp8_ds_mla` cache with SWA topk128 plus C4 topk512. | `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` |
| `dequantize_and_gather_k_cache` | Isolated and timed for full per-request gather. | same as above |
| standalone `quantize_and_insert_k_cache` | Blocked on SM80 as a standalone op: it compiles `tl.float8e4nv`, unsupported on A100.  vLLM's model path uses fused compressor kernels with a separate SM80 software-FP8 route, so this blocker applies only to the standalone probe. | same as above |

## Indexer Timing

| Case | vLLM Q quant ms | vLLM K store ms | vLLM K gather ms | vLLM decode logits ms | vLLM logits+topk ms | Mini bf16 logits ms | Mini FP8 logits ms | Mini bf16 select ms | Mini FP8 select ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch1, history1024 | `0.0876` | `0.0258` | `0.0201` | `0.1474` | `0.1772` | `0.1271` | `0.1573` | `0.3093` | `0.3182` |
| batch4, history2048 | `0.0845` | `0.0295` | `0.0191` | `0.1425` | `0.1761` | `0.1246` | `0.2655` | `0.3037` | `0.3163` |
| batch16, history4096 | `0.0839` | `0.0964` | `0.0195` | `0.1529` | `0.1804` | `0.3076` | `1.3072` | `0.3586` | `1.7368` |

Interpretation:

- vLLM Q quant and K store are faster than the mini 07.50 FP8 helpers.
- vLLM decode logits is not faster than mini bf16 at the two small shapes, but
  at batch16/history4096 it is `2.01x` faster than mini bf16 logits and `8.55x`
  faster than mini FP8 logits.
- vLLM logits+topk is `1.99x` faster than mini bf16 select at
  batch16/history4096.  The measured topk backend was `vllm_persistent_topk`,
  but TARGET 07.43 already showed persistent topk is not a standalone macro
  factor, so the next action should focus on the paged FP8 logits/cache backend
  first.
- vLLM prefill/gathered `fp8_mqa_logits_triton` was `0.1225`, `0.1188`,
  `0.4035` ms.  The large prefill-like gathered shape is not a win over the
  mini bf16 decode-style logits line, so the first port should target decode
  paged logits, not prefill.

## Indexer Quality

| Case | Logits mean abs vs bf16 | Logits max abs vs bf16 | Top-k overlap mean | K dequant mean abs | K dequant max abs | Q ref |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| batch1, history1024 | `0.02026` | `0.08299` | `0.9883` | `0.01791` | `0.25` | byte-exact |
| batch4, history2048 | `0.02107` | `0.11764` | `0.9810` | `0.01795` | `0.25` | byte-exact |
| batch16, history4096 | `0.02025` | `0.12885` | `0.9730` | `0.01797` | `0.25` | byte-exact |

The quality profile is comparable to 07.50's mini FP8 indexer quality and is
good enough for an opt-in backend port prototype.

## `fp8_ds_mla` Gather/Dequant Probe

The cache layout used for the gather probe is the vLLM DeepSeek V4 packed token
layout:

- `584` bytes/token total;
- `448` FP8 NoPE bytes;
- `128` bf16 RoPE bytes;
- `8` scale bytes with `7` real UE8M0 scale bytes plus pad.

The standalone vLLM `quantize_and_insert_k_cache` wrapper could not be used on
SM80 because it compiles a native Triton FP8 cast unsupported on A100.  To keep
the gather/dequant probe isolated, the script packed the cache with a PyTorch
reference and then called vLLM's gather/dequant kernels.

| Case | `gather_dequant_two_scopes_with_mask` ms | `dequantize_and_gather_k_cache` full gather ms | NoPE mean abs | RoPE max abs | Invalid mask |
| --- | ---: | ---: | ---: | ---: | --- |
| batch1, history1024 | `0.1110` | `0.0632` | `0.01789` | `0.0` | exact |
| batch4, history2048 | `0.1116` | `0.0877` | `0.01793` | `0.0` | exact |
| batch16, history4096 | `0.1173` | `0.3963` | `0.01796` | `0.0` | exact |

Comparison point from TARGET 07.395:

- mini exact bf16 sparse-only decode, T=4/H=4096: `0.2284 ms`;
- mini globaltopk+indexer+sparse decode, T=4/H=4096: `0.4350 ms`.

The vLLM two-scope FP8 gather/dequant kernel is promising as a later packed
KV-cache slice, but the standalone quant/insert blocker and the fact that mini
exact split-K sparse decode is already close mean it should not preempt the
clearer indexer backend port.

## Decision Table

| Backend piece | vLLM isolated time | mini bf16 time | mini FP8 time | Quality/error | Portability | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| FP8 Q path | `0.0839 ms` at batch16 | n/a | mini FP8 Q `0.2308 ms` | Q bytes exact vs vLLM torch ref | Direct Python/Triton wrapper semantics are portable; mini needs local wrapper or vendored kernel | `adapt-vllm-indexer` |
| FP8 indexer K store | `0.0964 ms` at batch16/history4096 | n/a | mini FP8 store `0.2941 ms` | dequant mean abs `0.01797`, max `0.25` | C++ custom op is not directly vendored; algorithm/layout are clear | `adapt-vllm-indexer` |
| FP8 paged indexer decode logits | `0.1529 ms` at batch16/history4096 | mini bf16 logits `0.3076 ms` | mini FP8 logits `1.3072 ms` | top-k overlap `0.9730`, mean abs `0.02025` | Triton wrapper is portable enough to port/adapt; primary win evidence | `adapt-vllm-indexer` |
| FP8 prefill/gathered logits | `0.4035 ms` at batch16/history4096 | mini bf16 decode-style logits `0.3076 ms` | mini FP8 logits `1.3072 ms` | same error profile | Not first slice | `defer` |
| `fp8_ds_mla` two-scope gather/dequant | `0.1173 ms` at batch16/history4096 topk128+512 | mini sparse-only decode boundary `0.2284 ms` includes attention | n/a | NoPE mean abs `0.01796`, RoPE exact | Gather kernel portable; standalone quant insert blocked on SM80 unless using fused compressor path | `probe-fp8-ds-mla` later |

## Go/No-Go

Go for a narrow vLLM FP8 indexer backend port/adaptation.

The next slice should be:

1. Replace mini's current FP8 indexer logits kernel with a vLLM-style
   `fp8_paged_mqa_logits_triton` backend or a close mini-owned port.
2. Keep the 07.50 opt-in guard; do not change the exact bf16 default.
3. Reuse vLLM-style Q scale folding and indexer cache layout.
4. Measure only the backend slice first, then rerun 4096/128/bs4 macro if the
   microbench remains close to the vLLM isolated line.

Do not port full `fp8_ds_mla` KV cache yet.  Keep it as the next fallback if
the indexer backend port loses integration performance or cannot be made
graph-safe.

Final decision: port/adapt vLLM FP8 indexer backend next.
