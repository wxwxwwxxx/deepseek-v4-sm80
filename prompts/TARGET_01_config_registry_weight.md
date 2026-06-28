# TARGET 01: DSV4 Config, Registry, and Weight Loading

## Goal

Make mini-sglang recognize `/models/DeepSeek-V4-Flash`, parse its DeepSeek-V4-specific config, register the correct model class, and load the major checkpoint tensors into a DSV4 model skeleton.

This target is complete when the project can construct a DSV4 model from the local model directory and validate important weight shapes without running full generation.

## Primary References

- Main implementation reference: `/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py`
- Config reference: `/workspace/sglang-main/python/sglang/srt/configs/deepseek_v4.py`
- Local model config: `/models/DeepSeek-V4-Flash/config.json`
- Local model inference config: `/models/DeepSeek-V4-Flash/inference/config.json`
- Current mini-sglang entry points:
  - `python/minisgl/models/config.py`
  - `python/minisgl/models/register.py`
  - `python/minisgl/models/weight.py`
  - `python/minisgl/models/base.py`

## Old dsv4 Branch References

Use the old branch only as a shape/name oracle, not as the main design source.

- `git show dsv4:python/minisgl/models/deepseek_v4.py`
- `git show dsv4:tests/models/test_deepseek_v4.py`
- `git show dsv4:benchmark/offline/deepseek_v4_runtime_env.py`

Do not copy the old raw-model-centered architecture wholesale. The new branch should follow sglang-main's DSV4 structure.

## Plan

1. Add a DSV4 config path.
   - Parse all fields needed by sglang's DSV4 model, including:
     - `q_lora_rank`
     - `o_lora_rank`
     - `qk_nope_head_dim`
     - `qk_rope_head_dim`
     - `v_head_dim`
     - `window_size`
     - `compress_ratios`
     - `index_topk`
     - `index_head_dim`
     - `index_n_heads`
     - `n_routed_experts`
     - `num_experts_per_tok`
     - `n_shared_experts`
     - `scoring_func`
     - `expert_dtype`
     - `routed_scaling_factor`
     - `hc_mult`
     - `hc_sinkhorn_iters`
     - `hc_eps`
   - Preserve generic config behavior for existing models.

2. Register the DSV4 model.
   - Add `DeepseekV4ForCausalLM` or equivalent naming consistent with local style.
   - Wire architecture-name detection so `/models/DeepSeek-V4-Flash` selects DSV4 automatically.
   - Keep existing Llama/Qwen/Mistral/Qwen3-MoE paths unchanged.

3. Implement weight-name mapping.
   - Start from sglang-main naming.
   - Add a local remap layer only when checkpoint names differ from mini-sglang module names.
   - Validate key tensor groups:
     - token embedding
     - final norm and lm head
     - q LoRA projections
     - kv latent projection and norm
     - output grouped projection
     - compressor/indexer weights
     - MoE gate, routed experts, shared experts

4. Add quantization-aware loading stubs.
   - bf16/fp8 checkpoint tensors should load or fail with a precise unsupported message.
   - fp4 expert tensors may remain as explicit TODO in this target.
   - Avoid silently converting unsupported formats in a way that hides correctness problems.

5. Add smoke tests.
   - Config parse test using `/models/DeepSeek-V4-Flash/config.json`.
   - Registry test that confirms DSV4 architecture selection.
   - Weight-shape test that loads metadata or a small subset of tensors and verifies expected shapes.

## Done Criteria

- `ModelConfig` or the DSV4 config object exposes all DSV4-specific fields needed by later targets.
- Model registry resolves DeepSeek-V4-Flash to the DSV4 class.
- Weight loading reaches DSV4 modules instead of failing on missing config/model type.
- Tests cover config parse, registry selection, and important weight shape mapping.
- No runtime dependency on old `dsv4` branch code.

## Non-Goals

- Full forward correctness.
- High-performance fp4/fp8 kernels.
- DSV4 KV cache implementation.
- Radix prefix cache.
