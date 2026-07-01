# TARGET 07.39: DSV4 SM80 Marlin Custom-Op Bridge Feasibility

## Final conclusion

The bridge feasibility probe is positive. The locally installed vLLM Marlin
custom ops can be imported and called on an A100 SM80 process, the vLLM MXFP4
weight transform accepts a DSV4-like packed-weight layout, the Marlin WNA16 MoE
call runs for both decode-like `T=4` and prefill-like `T=4096`, and the synthetic
routed-MoE timings are clearly faster than mini's current grouped FP4 path.

This target should therefore open a follow-up to vendor or reimplement a narrow
mini-owned Marlin WNA16 custom-op surface. It should not promote vLLM as a mini
runtime dependency, and it should not move to TARGET 07.4 precision lanes.

## Artifacts

- Probe script:
  `performance_milestones/target07_marlin_custom_op_bridge/scripts/probe_vllm_marlin_bridge.py`
- Canonical raw report:
  `performance_milestones/target07_marlin_custom_op_bridge/raw/vllm_marlin_bridge_probe.json`
- Summary:
  `performance_milestones/target07_marlin_custom_op_bridge/summaries/bridge_probe_summary.json`

Canonical command:

```bash
PYTHONPATH=/workspace/mini-sglang/python /workspace/venvs/vllm-dsv4/bin/python \
  performance_milestones/target07_marlin_custom_op_bridge/scripts/probe_vllm_marlin_bridge.py \
  --tokens 4 4096 \
  --warmup 1 \
  --iters 3 \
  --pretty \
  --output performance_milestones/target07_marlin_custom_op_bridge/raw/vllm_marlin_bridge_probe.json
```

## Import and op status

Environment:

- Python: `/workspace/venvs/vllm-dsv4/bin/python`
- torch: `2.11.0+cu128`
- vLLM: `0.1.dev3+gc9f425bef.d20260630`
- GPU: `NVIDIA A100-SXM4-80GB`
- CUDA capability: `sm80`

Import/registration probe:

- `vllm._custom_ops.gptq_marlin_repack`: pass
- `vllm._custom_ops.moe_wna16_marlin_gemm`: pass
- `torch.ops._C.gptq_marlin_repack`: pass
- `torch.ops._moe_C.moe_wna16_marlin_gemm`: pass

Direct custom-op probes:

- `gptq_marlin_repack`: pass for the synthetic W13 expert payload.
- `moe_wna16_marlin_gemm`: pass for the synthetic W13 route GEMM.

## Layout findings

The probe uses DSV4-like synthetic shape:

- hidden: `4096`
- local intermediate: `256`
- experts: `256`
- top-k: `6`
- activations: `bf16`
- weights: MXFP4
- scales: E8M0 byte representation

Raw layout compatibility:

- mini W13 is byte-compatible with `[experts, 2, local_intermediate, hidden/2]`
  int8 MXFP4. The vLLM helper expects `[experts, 2*local_intermediate, hidden/2]`
  uint8. The bridge probe reshapes/views the same packed bytes.
- mini W2 is byte-compatible with `[experts, hidden, local_intermediate/2]`
  int8 MXFP4. vLLM expects the same logical shape as uint8.
- scale bytes are uint8 viewed as `torch.float8_e8m0fnu`; the synthetic probe
  used byte `127`, which represents scale `1.0`.

Marlin transform output shapes:

- W13 qweight: `[256, 256, 1024]`
- W13 scale: `[256, 128, 512]`
- W2 qweight: `[256, 16, 8192]`
- W2 scale: `[256, 8, 4096]`

Route metadata:

- T=4 uses `block_size_m=8`; vLLM and mini both report
  `num_tokens_post_padded=192`, and the expert-id prefix matches.
- T=4096 uses `block_size_m=64`; vLLM and mini both report
  `num_tokens_post_padded=32768`, and the expert-id prefix matches.
- For T=4096, vLLM allocates larger metadata buffers
  (`sorted_token_ids` shape `[40704]`, `expert_ids` shape `[636]`) while mini's
  route plan stores `[32768]` sorted route ids and `[512]` expert ids. A future
  adapter must honor vLLM's allocation and sentinel contract, not only the live
  padded-token count.

Top-k weighting:

- vLLM `fused_marlin_moe` was called with `apply_router_weight_on_input=False`,
  so route weights are multiplied in the W2 GEMM.
- mini grouped FP4 applies route weights before W2 input. For per-route scalar
  weights this is mathematically equivalent, but the future port should preserve
  vLLM's exact placement to minimize drift.

## Synthetic performance

All timings below are the canonical `--warmup 1 --iters 3` run.

| case | direct W13 Marlin GEMM | fused Marlin MoE | mini grouped FP4 | speedup |
| --- | ---: | ---: | ---: | ---: |
| T=4 | `0.052224 ms` | `0.441344 ms` | `2.008064 ms` | `4.55x` |
| T=4096 | `1.035264 ms` | `2.343936 ms` | `46.994090 ms` | `20.05x` |

The numeric comparison against mini grouped FP4 used random synthetic packed
MXFP4 bytes and scale bytes, so it is only a rough layout/perf sanity check.
It is not a real model correctness proof. The probe records finite outputs for
both paths, with mean absolute differences around `0.105`.

## Mini integration status

The mini runtime now recognizes an explicit experimental marker:

```bash
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=vllm_marlin_bridge
```

This marker intentionally raises `NotImplementedError` through
`require_supported_moe_expert_backend()`. It exists so benchmark matrices can
record the bridge candidate without silently falling back to grouped FP4 and
without making vLLM a default runtime dependency.

The existing `marlin_mxfp4_w4a16` marker from TARGET 07.38 remains blocked for
the same reason: mini does not yet own the required custom-op surface.

## Validation

Passed:

```bash
python -m compileall \
  python/minisgl/kernel/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  performance_milestones/target07_marlin_custom_op_bridge/scripts/probe_vllm_marlin_bridge.py

python -m json.tool \
  performance_milestones/target07_marlin_custom_op_bridge/raw/vllm_marlin_bridge_probe.json

pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_moe_expert_backend_selector_blocks_marlin \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_records_vllm_marlin_bridge_backend \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_records_marlin_candidate_backend

pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py
```

The broader touched-surface pytest pass reported `41 passed, 4 warnings`.

The bridge is external/probe-only, so this target did not run TP8 text smoke,
4096/1024 macro, or Nsight. Those should move to the mini-owned csrc-port target
after the custom-op surface is available inside mini.

## Next target

Open a vendor/narrow csrc port target for the minimum Marlin WNA16 surface:

- `gptq_marlin_repack` or equivalent repack support;
- `moe_wna16_marlin_gemm` or equivalent W4A16 expert GEMM;
- MXFP4 E8M0 scale transform compatible with
  `prepare_moe_mxfp4_layer_for_marlin`;
- route metadata adapter for the vLLM Marlin contract;
- cached transformed weights so runtime does not repack per request;
- explicit unsupported errors for shape/layout misses.

Do not move to TARGET 07.4 unless a future implementation proves that the win
depends on activation quantization or another precision-lane semantic.
