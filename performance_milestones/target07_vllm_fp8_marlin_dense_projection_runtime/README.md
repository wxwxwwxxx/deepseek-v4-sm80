# TARGET 07.74: vLLM FP8 Marlin Dense Projection Runtime Opt-In

Status: implemented as a default-off opt-in, but not promoted.

## Toggle And Variant

- Toggle: `MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1`
- Variant: `dsv4_sm80_a100_victory_fp8marlinproj`
- Variant env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1` plus the Marlin toggle

Implemented Phase A owners only:

- `attn.q_wqb`
- `attn.wo_b` local projection, preserving row-parallel all-reduce
- `shared_experts.down_proj`

Out of scope remains unchanged: `wo_a`, FBGEMM-derived conversion, INT8, full
FP8 KV cache, router/HC precision, and routed MoE backend changes.

## Implementation

The opt-in adds `python/minisgl/kernel/vllm_fp8_marlin.py`, a lazy vLLM bridge
around:

- `process_fp8_weight_block_strategy`
- `prepare_fp8_layer_for_marlin`
- `apply_fp8_marlin_linear`

Preparation happens in `prepare_for_cuda_graph_capture()`. Forward only reads
the prepared packed weight/scale/workspace. Under the Marlin toggle, mini skips
the owner-specific BF16 dequant cache for q_wqb, wo_b, and shared-down. After a
successful Marlin pack, those owners delete their original FP8 `weight` and
`weight_scale_inv` attributes so the opt-in does not retain original FP8 plus
BF16 plus Marlin copies.

The promoted path is unchanged when the toggle is off.

## Backend Bridge

See `summaries/runtime_marlin_backend_bridge.md`.

Key result: focused vLLM Marlin calls work from `/workspace/venvs/vllm-dsv4`, but
full mini TP8 runtime is not safely bridged.

- Default mini Python: no `vllm`; adding the vLLM source path hits a vLLM `_C`
  torch ABI mismatch.
- vLLM venv: can import mini and vLLM Marlin ops.
- vLLM venv TP8 mini smoke: failed first due inherited bad `sgl_kernel`; after
  optional package probing was made non-fatal, the promoted baseline hung in
  CUDA graph capture before candidate execution. Interrupt/fail-open then
  exposed a mini-owned Marlin WNA16 extension ABI mismatch in the same venv.

## Focused Owner Results

Artifact: `summaries/focused_marlin_projection_runtime_microbench.md`.

M=`4` summary:

| Owner | Baseline ms | Marlin ms | Speedup | Mean abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `0.255962` | `0.064580` | `74.77%` | `0.015490` | `0.99967110` |
| `attn.wo_b` local | `0.260682` | `0.064178` | `75.38%` | `0.018019` | `0.99965119` |
| shared experts down | `0.256383` | `0.063991` | `75.04%` | `0.005111` | `0.99960858` |

This preserves the standalone signal in focused CUDA code. It does not clear
the full mini runtime gate.

## Memory Ledger

Artifact: `summaries/marlin_projection_memory_ledger.md`.

Per rank, across 43 layers:

| Metric | Bytes | GiB |
| --- | ---: | ---: |
| Promoted BF16 cache incremental for switched owners | `811597824` | `0.7559` |
| Marlin total after release | `412195248` | `0.3839` |
| Delta vs BF16 cache incremental | `-399402576` | `-0.3720` |
| Delta vs promoted owner total if original retained | `-805226256` | `-0.7499` |

At page size 256, the Marlin replacement saves about `20.68` KV pages/rank
versus the switched BF16 caches alone, or about `41.69` pages/rank versus
original FP8 plus BF16 cache retained.

## TP8 Smoke, Profile, Macro

Required TP8 same-run text smoke did not pass. See
`raw/tp8_text_smoke_attempts.json`.

Because the promoted baseline could not complete under the only interpreter that
can call vLLM Marlin ops, the 4096/128 profile and 4096/1024 macro were not run.
Running them would not satisfy the target gates without a passing TP8 graph/text
smoke.

## Decision

Do not promote. Do not expand to Phase B.

Keep the opt-in code path for continued bridge work, but treat this target as
stopped at the runtime bridge gate. The next target should first establish a
single coherent runtime:

- either port the dense FP8 Marlin custom op surface into mini and build it
  against default mini torch;
- or repair the vLLM venv so `sgl_kernel`, vLLM custom ops, and mini CUDA graph
  capture all work together on A100/sm80 before revisiting macro/profile gates.
