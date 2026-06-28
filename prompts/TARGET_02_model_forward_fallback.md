# TARGET 02: DSV4 Model Structure and Fallback Forward

## Goal

Build the DSV4 model structure in mini-sglang and make a short prefill forward pass work through a correctness-first fallback path.

This target is complete when a small input can run through the DSV4 model and produce logits that can be compared against the official oracle or old branch tests.

## Primary References

- Main implementation reference: `/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py`
- Official oracle: `/models/DeepSeek-V4-Flash/inference/model.py`
- Official kernels for semantics only: `/models/DeepSeek-V4-Flash/inference/kernel.py`
- Current local layer patterns:
  - `python/minisgl/layers/linear.py`
  - `python/minisgl/layers/moe.py`
  - `python/minisgl/layers/norm.py`
  - `python/minisgl/layers/rotary.py`
  - `python/minisgl/models/qwen3_moe.py`

## Old dsv4 Branch References

Use these for oracle tests and shape expectations:

- `git show dsv4:python/minisgl/models/deepseek_v4.py`
- `git show dsv4:tests/models/test_deepseek_v4.py`
- `git show dsv4:benchmark/offline/deepseek_v4_oracle_matrix.py`
- `git show dsv4:benchmark/offline/deepseek_v4_correctness_matrix.py`

Treat old model code as a reference implementation, not as the final architecture.

## Plan

1. Add the DSV4 model file.
   - Implement the model class registered in TARGET 01.
   - Follow sglang-main's module boundaries where possible.
   - Keep module names weight-loader-friendly.

2. Implement DSV4 decoder components.
   - `MQALayer` with:
     - q LoRA path
     - q norm
     - q rope/nope split
     - kv latent projection
     - kv norm
     - rope application
     - grouped output projection `wo_a` / `wo_b`
   - DSV4 MLP/MoE block with:
     - routed experts
     - shared expert
     - sqrtsoftplus scoring
     - top-k routing
     - routed scaling factor
   - Final norm and lm head.

3. Use correctness-first fallback math.
   - Use torch operations when no sm80 kernel is ready.
   - Keep tensor layouts compatible with later high-performance kernel wrappers.
   - Avoid optimizing before oracle alignment is available.

4. Connect to DSV4 attention backend interface.
   - Call the DSV4 backend introduced in TARGET 04.
   - If TARGET 04 is not complete yet, use a minimal temporary attention shim that has the same expected API.
   - Do not force DSV4 through the normal MHA attention API if the metadata does not fit.

5. Add oracle comparison tests.
   - Use very short prompts and a small number of layers if a reduced config is available.
   - Compare logits or selected intermediate tensors.
   - Keep tolerances explicit for bf16/fp8/fallback differences.

## Done Criteria

- A DSV4 model can be instantiated and called with a short token batch.
- The forward path reaches logits without missing module or missing tensor errors.
- MoE routing and attention calls have correctness-first behavior.
- At least one oracle comparison test exists.
- Unsupported fp4/fused paths fail loudly or use an explicitly named fallback.

## Non-Goals

- Final throughput optimization.
- CUDA graph capture.
- Radix prefix cache.
- Full fp4 expert GEMM performance.
