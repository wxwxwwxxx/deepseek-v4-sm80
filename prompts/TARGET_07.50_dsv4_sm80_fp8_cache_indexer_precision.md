# TARGET 07.50: DSV4 SM80 Opt-In FP8 Cache And Indexer Precision Lane

## Goal

Evaluate an opt-in precision/cache lane for DeepSeek V4 sm80 after the exact
bf16 path has matched the comparable decode sparse boundary and still remains
well behind vLLM at macro level.

This target must not change mini's exact default.  It should produce evidence
for whether packed FP8 KV/indexer cache is worth adopting as an explicit
variant.

## Start Condition

Start this target only if one of these is true:

- TARGET 07.40 selects precision/cache as the next best target;
- TARGET 07.41 fails to clear exact-path stop conditions;
- a fresh mini/vLLM comparison shows the remaining gap is dominated by
  cache/indexer precision/layout rather than exact bf16 kernel boundaries.

## Required Inputs

Read first:

- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`
- `performance_milestones/target07_post_splitk_reprofile/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`

vLLM references:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## Scope

In scope:

- identify the minimal vLLM-compatible cache/indexer precision surface:
  - packed `fp8_ds_mla` KV cache;
  - FP8 indexer cache;
  - cache insert/compressor/indexer store behavior;
  - gather/dequant compatibility with split-K sparse decode;
- implement only opt-in variants with explicit flags and variant names;
- compare against the best exact bf16 stack from TARGET 07.395 or later;
- run quality checks before interpreting performance as useful.

Out of scope:

- making FP8 cache/indexer the default;
- hiding precision changes behind an exact variant name;
- INT8 Tensor Core MoE;
- changing MoE expert backend unless the precision lane explicitly requires a
  fair comparison against Marlin WNA16.

## Quality Gates

At minimum:

- TP8 text smoke, page size 256;
- logits/top-k or sampled-token agreement probes against the exact bf16 stack
  on short deterministic prompts;
- malformed-output checks for Chinese and English prompts;
- explicit note of expected precision differences.

Do not promote this lane beyond opt-in unless a later target defines stronger
quality gates.

## Performance Gates

Useful result requires:

- microbench evidence for cache insert/indexer/gather/dequant;
- 4096/128/batch4 macro;
- 4096/1024/batch4 macro if the short macro improves;
- comparison against both:
  - best exact mini stack;
  - fresh or recorded vLLM `deepseek_v4_fp8` line.

## Expected Output

Create:

- `performance_milestones/target07_fp8_cache_indexer_precision/README.md`
- `scripts/`, `raw/`, and `summaries/` under that directory.

The README must answer:

- which precision/cache pieces were implemented or blocked;
- whether the speedup comes from memory bandwidth, cache layout, indexer cache,
  gather/dequant, or graph/runtime side effects;
- what quality changed relative to exact bf16;
- whether this lane should continue, stop, or remain a research-only opt-in.
