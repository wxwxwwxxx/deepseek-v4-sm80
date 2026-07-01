# TARGET 07.51: DSV4 SM80 vLLM FP8 Backend Parity And Go/No-Go

## Goal

Decide whether mini-sglang should continue the FP8 cache/indexer lane by
measuring vLLM's actual DeepSeek V4 FP8 backend pieces in isolation.

TARGET 07.50 implemented a narrow mini-owned FP8 indexer cache/logits slice.
It was graph-safe and passed text smoke, but failed performance:

- exact control 4096/128/bs4: `37.9237 output tok/s`;
- mini FP8 indexer cache/logits: `29.6691 output tok/s`;
- FP8 indexer logits microbench was slower than bf16 logits, especially at
  larger history shapes.

This target must not continue optimizing the current mini software FP8 indexer
kernel blindly.  It should first answer:

1. Is vLLM's real FP8 indexer backend faster than mini's bf16 and mini's
   current FP8 implementation on comparable shapes?
2. Is vLLM's `fp8_ds_mla` gather/dequant backend faster enough to justify a
   minimal packed KV-cache slice?
3. If vLLM's backend is fast, which exact kernel/op should be ported or
   adapted next?
4. If vLLM's backend is not fast or cannot be isolated, should the FP8 lane
   stop and the project return to exact graph/runtime/prefill work?

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.50_dsv4_sm80_fp8_cache_indexer_precision.md`
- `performance_milestones/target07_fp8_cache_indexer_precision/README.md`
- `performance_milestones/target07_fp8_cache_indexer_precision/summaries/target0750_indexer_fp8_summary.json`
- `performance_milestones/target07_vllm_ablation_before_precision/README.md`
- `performance_milestones/target07_vllm_metadata_runtime_parity/README.md`
- `performance_milestones/vllm/scripts/vllm_env.sh`

Relevant mini code:

- `performance_milestones/target07_fp8_cache_indexer_precision/scripts/mini_indexer_fp8_microbench.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`

Relevant vLLM code:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`

## vLLM Checkout Hygiene

The vLLM checkout at `/workspace/vllm-dsv4-docker` may already be on the
project branch `minisgl_docker`.  It may contain unrelated untracked ncu report
directories.  Do not clean or modify those reports.

It is OK to use normal git hygiene in the vLLM checkout:

- create a temporary microbench branch if useful;
- use `git stash` to move between experiments;
- save exact diffs with `git diff`;
- use non-interactive git commands where possible.

All vLLM edits for this target must be minimal, reversible, and recorded under
the milestone directory.  Prefer standalone scripts that import vLLM ops over
editing vLLM source.  If source edits are needed, gate them with environment
variables.

## Work Plan

### 1. Preserve 07.50 As The Failed Mini Baseline

Do not change the 07.50 implementation before collecting parity numbers.

Record in the new README:

- 07.50 mini bf16 microbench results;
- 07.50 mini FP8 microbench results;
- 07.50 exact-control and FP8 macro results;
- the conclusion that the current mini-owned FP8 indexer slice is stopped
  unless a vLLM backend port proves worth it.

### 2. Build A vLLM-Side FP8 Indexer Microbench

Create a script under:

```text
performance_milestones/target07_vllm_fp8_backend_parity/scripts/
```

The script should run from the vLLM virtualenv:

```bash
source /workspace/venvs/vllm-dsv4/bin/activate
source /workspace/mini-sglang/performance_milestones/vllm/scripts/vllm_env.sh
setup_vllm_runtime_env
```

Benchmark comparable indexer cases to 07.50:

- batch `1`, history `1024`;
- batch `4`, history `2048`;
- batch `16`, history `4096`;
- indexer heads `64`;
- head dim `128`;
- topk width `512`;
- sm80/A100.

Try to isolate:

- `fused_indexer_q_rope_quant` FP8 Q path;
- vLLM FP8 indexer K/cache quant/store if callable without full engine;
- `fp8_paged_mqa_logits_triton` for decode-like paged logits;
- `fp8_mqa_logits_triton` or prefill-like gathered logits if available;
- topk only as a secondary measurement, since TARGET 07.43 showed persistent
  topk is not a standalone macro factor.

Compare against:

- mini bf16 logits/select from 07.50;
- mini current FP8 logits/select from 07.50;
- vLLM FP8 logits/select where isolated.

If vLLM ops cannot be imported or called cleanly without full engine metadata,
record the blocker precisely.  Do not spend the whole target rebuilding vLLM's
engine around the microbench.

### 3. Build A Minimal `fp8_ds_mla` Gather/Dequant Probe

If indexer backend parity is inconclusive or negative, probe the broader
cache/layout hypothesis before writing more mini E2E code.

Try to isolate:

- `gather_dequant_two_scopes_with_mask`;
- `dequantize_and_gather_k_cache`;
- the shape/layout assumptions for `fp8_ds_mla` token storage
  (`584` bytes/token, `448` FP8 NoPE bytes, `128` bf16 RoPE bytes, `8` scale
  bytes with `7` ue8m0 scales plus pad);
- compatibility with mini's existing bf16 split-K sparse decode input.

Compare the vLLM gather/dequant cost to mini's bf16 gather/split-K boundary
from 07.395/07.40.  Do not implement a full mini `fp8_ds_mla` cache yet unless
the isolated backend looks promising.

### 4. Classify The Result

Produce a decision table:

| Backend piece | vLLM isolated time | mini bf16 time | mini FP8 time | Quality/error | Portability | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |

Allowed decisions:

- `port-vllm-indexer`: vLLM FP8 indexer backend is clearly faster and portable;
- `adapt-vllm-indexer`: vLLM backend is faster but needs a mini-owned wrapper;
- `stop-fp8-indexer`: vLLM indexer backend is not faster or cannot be isolated;
- `probe-fp8-ds-mla`: indexer is negative, but KV gather/dequant looks promising;
- `port-fp8-ds-mla-gather`: vLLM gather/dequant is clearly faster and portable;
- `stop-fp8-lane`: neither indexer nor KV gather/dequant has evidence;
- `return-exact-runtime`: evidence says the next work should be exact
  graph/runtime/prefill instead of precision.

## Performance Thresholds

Use microbench thresholds before any macro work:

- vLLM FP8 indexer logits/select should beat mini bf16 by at least `20%` on a
  representative shape, or show a convincing path to at least `5%` E2E gain.
- vLLM `fp8_ds_mla` gather/dequant should beat the comparable mini bf16
  gather/decode boundary by at least `20%`, or reduce enough memory traffic to
  justify a minimal mini slice.
- If isolated vLLM backend timings are within noise or slower, do not port.
- Do not run 4096/1024 macro in this target unless a port/adapt prototype first
  beats the microbench threshold.

## Quality Checks

For every isolated backend, report:

- max/mean absolute error vs bf16 reference when applicable;
- top-k overlap for indexer logits when applicable;
- dtype/layout assumptions;
- whether scales are stored or folded exactly as in vLLM;
- any known mismatch from 07.50's precision-boundary map.

## Out Of Scope

- further optimizing the current mini software FP8 indexer logits kernel
  without vLLM backend evidence;
- full mini E2E `fp8_ds_mla` cache implementation before gather/dequant proof;
- changing the default exact bf16 path;
- MoE/Marlin work;
- aux-stream or persistent-topk adaptation;
- long vLLM node-trace debugging;
- full 4096/1024 macro unless a new backend slice first wins microbench.

## Expected Output

Create:

- `performance_milestones/target07_vllm_fp8_backend_parity/README.md`
- `scripts/`, `raw/`, and `summaries/` under that directory.

The README must include:

- what vLLM ops were successfully isolated;
- what could not be isolated and why;
- side-by-side vLLM vs mini bf16 vs mini FP8 microbench table;
- quality/error table;
- exact source paths and patch/diff paths if vLLM source was edited;
- final go/no-go decision;
- next target recommendation.

## Final Decision Template

End with one of:

- `Decision: port/adapt vLLM FP8 indexer backend next`;
- `Decision: stop FP8 indexer and probe fp8_ds_mla gather/dequant next`;
- `Decision: port/adapt vLLM fp8_ds_mla gather/dequant next`;
- `Decision: stop FP8 precision lane and return to exact runtime/prefill`;
- `Decision: blocked because vLLM backend cannot be isolated; required external package or engine fixture is ...`.
