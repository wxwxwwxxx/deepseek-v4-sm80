# TARGET 07.50: DSV4 SM80 vLLM-Aligned FP8 Cache/Indexer Precision Lane

## Goal

Implement and evaluate an opt-in FP8 cache/indexer precision lane for DeepSeek
V4 on A100/sm80, using vLLM's current DeepSeek V4 fast path as the primary
behavioral reference.

This target exists because the exact bf16 path has already matched the
comparable decode sparse boundary, while the macro gap remains large:

- best mini exact 4096/128/bs4: `38.9379 output tok/s`;
- best mini exact 4096/1024/bs4: `68.8097 output tok/s`;
- vLLM 4096/128/bs4 control: `82.2825 output tok/s`;
- vLLM 4096/1024/bs4 control: `202.0342 output tok/s`.

This target must not change mini's default exact bf16 path.  Every precision
change must be opt-in, named explicitly, and quality-gated.

## Start Condition

Start this target now.  The required evidence exists:

- TARGET 07.40 showed decode split-K is no longer the primary bottleneck.
- TARGET 07.41 optimized replay metadata copy but macro did not improve.
- TARGET 07.42 found no justified exact runtime PoC and identified vLLM's
  packed FP8 cache/indexer lane as the strongest next hypothesis.
- TARGET 07.43 showed vLLM aux-stream and persistent-topk ablations are below
  the `5%` decision bar, while vLLM eager-vs-graph confirms graph is mandatory
  but not a new mini action item by itself.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.30_dsv4_sm80_attention_history.md`
- `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`
- `prompts/TARGET_07.41_dsv4_sm80_indexer_cache_runtime_exact.md`
- `prompts/TARGET_07.42_dsv4_sm80_vllm_metadata_runtime_parity.md`
- `prompts/TARGET_07.43_dsv4_sm80_vllm_ablation_before_precision.md`
- `performance_milestones/target07_post_splitk_reprofile/README.md`
- `performance_milestones/target07_indexer_cache_runtime_exact/README.md`
- `performance_milestones/target07_vllm_metadata_runtime_parity/README.md`
- `performance_milestones/target07_vllm_ablation_before_precision/README.md`
- `performance_milestones/target07_attention_indexer_cache_runtime/summaries/dispatch_backend_report.md`
- `performance_milestones/target07_bf16_sparse_decode_splitk/README.md`

vLLM reference code:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`

mini reference code:

- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

## vLLM Alignment Policy

For this target, vLLM is the default precision/algorithm oracle.  Do not invent
a different quantization or cache algorithm unless there is a written reason
and a focused experiment showing it is better or safer.

Before coding, write a precision-boundary note in the milestone README:

| Boundary | vLLM behavior | mini exact behavior | Planned 07.50 behavior | Deviation from vLLM? |
| --- | --- | --- | --- | --- |

The note must cover at least:

- where activations remain bf16/fp32;
- where Q is quantized for the indexer;
- whether Q scale is stored or folded into weights;
- how indexer K/cache is quantized and stored;
- how indexer logits are computed;
- how top-k indices are stored/reused;
- how MLA/SWA KV cache is packed;
- how gather/dequant feeds sparse decode;
- whether dequantization happens inside the kernel, in a gather step, or on
  load/store.

If the vLLM path moves precision inside a fused/custom op, the mini experiment
should match that boundary when possible.  If mini moves precision outside the
operator for implementation convenience, record that difference and measure the
cost.

## vLLM Facts To Preserve

Use these as starting facts, but verify them in code before implementation:

- vLLM DeepSeek V4 selects `deepseek_v4_fp8`.
- vLLM canonicalizes FP8 KV cache to `fp8_ds_mla`.
- The MLA compressed cache spec uses `dtype=torch.uint8`,
  `cache_dtype_str="fp8_ds_mla"`, `alignment=576`, and
  `model_version="deepseek_v4"`.
- vLLM's FP8 indexer cache stores `128` FP8 bytes plus `4` bytes of scale per
  head, i.e. `132` bytes for indexer head dim `128`.
- vLLM's indexer uses `scale_fmt="ue8m0"` and quant block size `128`.
- In the FP8 indexer Q path, `fused_indexer_q_rope_quant` folds the per-token Q
  scale into `weights`; Q scale is not separately passed to the FP8 logits
  kernel.
- On sm80, vLLM's FP8 indexer decode path uses
  `fp8_paged_mqa_logits_triton`, not DeepGEMM.
- vLLM sparse decode path is:
  `compute_global_topk_indices_and_lens -> gather_dequant_two_scopes_with_mask
  -> _dsv4_sm80_sparse_attn_decode_triton`.
- vLLM aux-stream overlap and persistent topk are not proven standalone
  sources of the current macro gap by TARGET 07.43.

## Work Plan

### 1. Build The Precision Boundary Map

Create the README skeleton first.  Fill in:

- exact vLLM source lines/functions;
- mini exact source lines/functions;
- tensors, shapes, dtype, scale format, and storage layout;
- which pieces are safe to port directly and which require mini-owned kernels.

Do not start implementation until the map identifies a minimal first slice.

### 2. Prefer A Narrow Indexer-First Slice

The first implementation slice should be the narrowest opt-in FP8 indexer lane
that can be measured:

- FP8 indexer K/cache store;
- FP8 indexer Q RoPE/quant with vLLM-style weight-fold semantics;
- FP8 paged MQA logits or a mini-owned equivalent matching vLLM's math;
- existing top-k transform if persistent topk is not needed;
- quality comparison against bf16 indexer selection.

Reason: the indexer/cache mismatch is isolated, measurable, and less invasive
than replacing the whole MLA cache layout.

If this slice cannot be implemented cleanly, record why and move to the
smallest viable `fp8_ds_mla` cache/gather slice instead.

### 3. Then Consider Packed `fp8_ds_mla` KV Cache

Only after the indexer-first slice is measured or blocked, attempt the broader
MLA/SWA cache lane:

- packed `fp8_ds_mla`-style KV/cache storage;
- cache insert/compressor/store path;
- gather/dequant compatible with mini's existing bf16 split-K sparse decode;
- exact comparison against the current bf16 flat cache gather path.

Do not replace the split-K sparse decode kernel unless the gather/dequant
boundary proves it is necessary.  The prior bf16 split-K sparse decode boundary
is already close to the vLLM sparse decode probe.

### 4. Use Graph Node Profiles Only As Explanatory Evidence

Do not block this target on another graph-node investigation.

Use existing TARGET 07.40 mini node-trace data as the pre-07.50 baseline.  If an
FP8/indexer/cache slice improves macro throughput by at least `5%`, capture a
fresh mini node-trace profile to explain which buckets moved:

- runtime/copy/cat/index graph nodes;
- elementwise graph nodes;
- indexer logits/topk/cache;
- cache store/gather/dequant;
- sparse decode gather/split/combine.

Do not spend the thread trying to force a vLLM node trace.  TARGET 07.42 already
showed that this is fragile in the current environment.

## Scope

In scope:

- opt-in FP8 indexer cache and paged-logits lane;
- opt-in packed `fp8_ds_mla`-style KV/cache lane;
- mini-owned kernels that match vLLM's algorithmic and precision boundary;
- microbenches for indexer store/logits/topk and KV gather/dequant;
- macro comparison against best exact mini and vLLM control lines;
- quality/correctness probes for precision changes.

Out of scope:

- making FP8 cache/indexer the default;
- hiding precision changes behind exact variant names;
- INT8 Tensor Core MoE;
- further MoE/Marlin work;
- aux-stream/custom-op adaptation before FP8 evidence;
- persistent-topk/indexer adaptation unless the FP8 lane identifies it as a new
  bottleneck;
- broad graph boundary cleanup before FP8/cache evidence.

## Quality Gates

At minimum:

- focused synthetic correctness tests for quantize/dequantize/store/gather;
- indexer top-k overlap/agreement probes vs exact bf16 on deterministic short
  prompts;
- logits or selected-index similarity summaries before using macro as success;
- TP8 text smoke, page size `256`, with Chinese and English prompts;
- malformed-output and repeated-gibberish checks;
- explicit note of expected precision differences.

For indexer selection, report at least:

- top-k exact match rate where applicable;
- top-k overlap rate;
- max/mean indexer-logit error on small controlled cases;
- sampled text sanity.

For KV/cache attention, report at least:

- gather/dequant max/mean error vs bf16 cache;
- sparse attention output error vs bf16 split-K path;
- text smoke result.

Do not promote this lane beyond opt-in unless a later target defines stronger
quality gates.

## Performance Gates

Useful result requires:

- microbench evidence for the implemented slice;
- 4096/128/batch4 macro;
- 4096/1024/batch4 macro if 4096/128 improves by at least `5%`;
- comparison against both:
  - best exact mini stack;
  - vLLM `deepseek_v4_fp8` control from TARGET 07.43.

Stop early if:

- the slice cannot pass correctness/text smoke after one focused fix attempt;
- the implemented slice improves neither microbench nor macro;
- the design drifts away from vLLM's precision boundary without evidence;
- implementation requires changing the default exact path.

## Expected Output

Create:

- `performance_milestones/target07_fp8_cache_indexer_precision/README.md`
- `scripts/`, `raw/`, and `summaries/` under that directory.

The README must answer:

- which vLLM precision/cache behavior was matched;
- which precision/cache behavior was intentionally not matched and why;
- which pieces were implemented, blocked, or deferred;
- whether speedup comes from memory bandwidth, cache layout, indexer logits,
  gather/dequant, graph-node reduction, or another source;
- what quality changed relative to exact bf16;
- current best exact result;
- current best opt-in FP8 result;
- whether this lane should continue, stop, or remain research-only.

## Final Decision Template

End with one of:

- `Decision: continue FP8 indexer/cache lane`;
- `Decision: expand to packed fp8_ds_mla KV cache`;
- `Decision: stop FP8 lane because quality/performance failed`;
- `Decision: keep FP8 lane research-only and return to exact graph/runtime`;
- `Decision: promote only as opt-in benchmark variant, not default`.
