# TARGET 07.54: Graph/Layout Replay Deforestation

Date: 2026-07-02

## Result

This target implemented one focused graph/layout PoC: replace the repeated
PyTorch FP8 activation fake-quant graph with an opt-in Triton helper.

Decision: continue graph/layout replay deforestation for one more focused cut,
not a broad rewrite.

Why: the PoC hit the TARGET 07.54 primary gate by reducing the 4096/128/bs4
decode-envelope `graph_runtime_copy_cat_index + elementwise_graph_nodes`
cluster from `2.9752 s` to `1.8271 s`, a `38.59%` reduction.  Even if the new
fused `_fp8_activation_quantize_kernel` is conservatively charged back to the
same cluster, the reduction is still `36.04%`.  The 4096/128 output-throughput
gate did not independently pass (`43.07` vs `41.66`, only `+3.38%`), but the
target's primary condition is cluster reduction or short macro throughput.  The
required long follow-up did pass: 4096/1024/bs4 improved from `73.67` to
`87.08` output tok/s (`+18.21%`).

This still does not reach the old serving victory line (`114.07`) or vLLM
(`~202` at 4096/1024/bs4).  Projection/GEMM is now co-dominant and must be the
next pivot if another graph/layout cut fails to translate.

## 07.53 Baseline Preserved

Current best opt-in FP8-indexer lines before this target:

| Shape | Output tok/s | Decode tok/s | Graph replay | Eager decode | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| 4096/128/bs4 | `41.6606` | `85.7568` | `127` | `0` | TARGET 07.53 fresh macro |
| 4096/1024/bs4 | `73.6706` | `86.2124` | `1023` | `0` | TARGET 07.52/07.53 retained line |

07.53 rank0 decode-envelope profile:

| Bucket | Kernel s | Wall share | Count | Graph nodes |
| --- | ---: | ---: | ---: | ---: |
| projection/GEMM | `1.7973` | `27.49%` | `100965` | `795` |
| graph/runtime/copy/cat/index | `1.6170` | `24.73%` | `466128` | `3612` |
| elementwise graph nodes | `1.3583` | `20.77%` | `561461` | `4384` |
| NCCL/communication | `0.3313` | `5.07%` | `11176` | `88` |
| MoE/Marlin | `0.3193` | `4.88%` | `43688` | `344` |
| sampling/logits | `0.1826` | `2.79%` | `43815` | `345` |
| FP8 indexer | `0.1301` | `1.99%` | `20828` | `164` |
| sparse attention decode | `0.1180` | `1.80%` | `21590` | `170` |
| KV/compressor/cache store | `0.0281` | `0.43%` | `8128` | `64` |

Target graph/layout cluster:

```text
1.616982515 + 1.358253289 = 2.975235804 s
```

## Source Attribution

Attribution started from the 07.53 top repeated graph-layout kernels:
direct-copy, bf16 copy, float8 copy, index/gather, CatArrayBatchedCopy, and the
clamp/log2/reduce/pow chain.

| Candidate | Kernel evidence | Suspected mini boundary | vLLM analogous boundary | PoC idea | Expected gain |
| --- | ---: | --- | --- | --- | ---: |
| FP8 activation fake-quant chain | 07.53 top kernels include direct-copy `1.2244 s`, bf16 copy `0.2335 s`, clamp `0.1917 s`, reduce max `0.1395 s`, pow scalar `0.1299 s`, mean reduce `0.1176 s`, abs `0.0944 s`, float8 copy `0.0928 s`, pow tensor `0.0923 s`, log2 `0.0699 s`; the scale/quant nodes recur as `279` graph nodes. | [quantize_fp8_activation_ref](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:928), called by [quantized_linear_ref](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:1146) and projection wrappers such as [DSV4Linear.forward](/workspace/mini-sglang/python/minisgl/models/deepseek_v4.py:237), attention WQA/WKV/QWQB/KV/WO paths at [deepseek_v4.py](/workspace/mini-sglang/python/minisgl/models/deepseek_v4.py:507). | vLLM wraps FP8 quantization behind `QuantFP8` and `_C.*scaled_fp8_quant` custom ops: `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py:29`, `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py:2002`; compile matchers know these quant ops at `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/matcher_utils.py:33`. | Fuse group absmax, log2/ceil/exp2 scale, clamp, E4M3 encode/decode, and output cast into one opt-in Triton helper. | High, because it removes a repeated 279-node elementwise/copy chain and aligns with vLLM's custom-op boundary. |
| Replay metadata copy | Direct-copy and copy/cat/index bucket remains large; graph replay copies input ids/out loc/positions and attention metadata every decode. | [GraphReplayBuffer.copy_from](/workspace/mini-sglang/python/minisgl/engine/graph.py:56), [GraphRunner._replay_to_buffer](/workspace/mini-sglang/python/minisgl/engine/graph.py:256), [copy_decode_metadata_for_replay](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:3322), [DSV4AttentionBackend._copy_metadata_for_replay](/workspace/mini-sglang/python/minisgl/attention/deepseek_v4.py:962). | vLLM keeps more of the attention layout under custom op and compile graph boundaries, especially `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:657`. | Further prepack replay metadata or fuse remaining copy kernels. | Medium, but 07.52 already fused part of this path and attribution is spread across many tensors. |
| Index/gather page-table and sparse top-k layout | 07.53 index kernel around `0.2801 s`, `gatherTopK` around `0.0593 s`, plus CatArrayBatchedCopy `0.1565 s`/`0.0406 s`. | Page table and compressed index assembly in [DSV4AttentionBackend._build_metadata](/workspace/mini-sglang/python/minisgl/attention/deepseek_v4.py:520), [_gather_full_locs](/workspace/mini-sglang/python/minisgl/attention/deepseek_v4.py:694), [topk_transform_512_full_fallback](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:3037), indexer merge rows at [deepseek_v4.py](/workspace/mini-sglang/python/minisgl/attention/deepseek_v4.py:415). | vLLM attention/indexer paths are custom-op and compile-bounded rather than exposing this much PyTorch indexing in replay. | Replace selected gather/scatter assembly with preallocated buffers or a fused kernel. | Medium, but the profile evidence is less concentrated than FP8 activation quant. |
| CatArray/cat fallback candidates | CatArrayBatchedCopy appears in top kernels; attention fallback also cats candidate lists. | [DSV4AttentionBackend._decode_candidates_for_row](/workspace/mini-sglang/python/minisgl/attention/deepseek_v4.py:884), top-k/index row merges, and metadata padding helpers. | vLLM sparse attention hides candidate assembly behind attention custom ops. | Avoid cat in decode candidate assembly or keep persistent candidate buffers. | Low to medium; much of this is fallback or spread across attention metadata. |
| Static layout transforms around projection/GEMM | direct-copy and bf16 copy still surround `_quantized_linear_fp8_kernel`; projection/GEMM bucket remains `1.7973 s`. | Projection wrappers in [DSV4Linear.forward](/workspace/mini-sglang/python/minisgl/models/deepseek_v4.py:237) and attention projection blocks [deepseek_v4.py](/workspace/mini-sglang/python/minisgl/models/deepseek_v4.py:563). | vLLM lifts selected projections around custom ops and uses FP8 quant/GEMM boundaries, for example `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:643`. | Deeper projection/GEMM parity target. | High but out of scope for this target. |

Selected PoC: FP8 activation fake-quant fusion.  It had the clearest source
attribution, the strongest vLLM boundary match, and a plausible 10% cluster
cut without rewriting projection/GEMM.

Rejected for this target: replay metadata copy, index/gather/top-k layout, and
CatArray work.  They remain real, but their source attribution was more
distributed and risked turning this target into a broad layout rewrite.

## Implementation

New opt-in guard:

```text
MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON=1
```

Files changed:

| File | Change |
| --- | --- |
| [python/minisgl/kernel/deepseek_v4.py](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:42) | Added the opt-in toggle and routed [quantize_fp8_activation_ref](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:928) to Triton when enabled. |
| [python/minisgl/kernel/triton/deepseek_v4.py](/workspace/mini-sglang/python/minisgl/kernel/triton/deepseek_v4.py:1287) | Added `_fp8_activation_quantize_kernel`; wrapper is [fp8_activation_quantize](/workspace/mini-sglang/python/minisgl/kernel/triton/deepseek_v4.py:3563). |
| [benchmark/offline/deepseek_v4_perf_matrix.py](/workspace/mini-sglang/benchmark/offline/deepseek_v4_perf_matrix.py:53) | Added the `idxfp8cache_actqtriton` graph variant. |
| [benchmark/offline/deepseek_v4_text_smoke.py](/workspace/mini-sglang/benchmark/offline/deepseek_v4_text_smoke.py:59) | Added the same text-smoke variant. |
| [tests/kernel/test_deepseek_v4_wrappers.py](/workspace/mini-sglang/tests/kernel/test_deepseek_v4_wrappers.py:926) | Added a CUDA SM80 comparison against the previous PyTorch reference. |
| [scripts/summarize_graph_layout_nsys.py](/workspace/mini-sglang/performance_milestones/target07_graph_layout_replay_deforestation/scripts/summarize_graph_layout_nsys.py:1) | Added a 07.54 classifier with an explicit `fp8_activation_quant_poc` bucket. |

The helper fuses:

```text
contiguous/view/float -> abs -> amax -> clamp_min -> log2 -> ceil -> pow/exp2
-> divide -> clamp -> E4M3 encode -> E4M3 decode -> scale multiply -> cast/store
```

The exact BF16 default path remains unchanged unless the new env var is set.

## Verification

Compilation:

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py
```

Unit tests:

```bash
python -m pytest -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_fp8_activation_quant_triton_matches_torch_reference -q

python -m pytest -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_quantized_linear_fp8_per_call_gemm_matches_fallback \
  tests/kernel/test_deepseek_v4_wrappers.py::test_quantized_linear_fp8_pair_shared_activation_matches_fallback \
  tests/kernel/test_deepseek_v4_wrappers.py::test_fused_wqa_wkv_cached_weight_matches_shared_activation -q
```

Text smoke:

| Artifact | Status | Graph |
| --- | --- | --- |
| `raw/text_smoke_actqtriton.json` | pass | captured `[4, 2, 1]`, replay `9`, greedy replay `9`, eager decode `0` |

## Macro Results

Variant:

```text
v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

| Shape | Baseline output tok/s | New output tok/s | Delta | Baseline decode tok/s | New decode tok/s | Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4096/128/bs4 | `41.6606` | `43.0685` | `+3.38%` | `85.7568` | `104.2028` | replay `127`, eager `0` |
| 4096/1024/bs4 | `73.6706` | `87.0831` | `+18.21%` | `86.2124` | `104.3427` | replay `1023`, eager `0` |

4096/128 did not clear the standalone `+5%` output-throughput threshold
(`43.7436` would be required), but decode throughput improved `+21.51%`.
Because the profile cluster gate passed, the 4096/1024 secondary run was
required and passed.

Artifacts:

| Path | Contents |
| --- | --- |
| `raw/macro_4096x128_bs4_np128_actqtriton` | 4096/128 non-Nsight macro. |
| `raw/macro_4096x1024_bs4_np128_actqtriton` | 4096/1024 non-Nsight macro. |
| `raw/dsv4_target0754_graph_layout_4096x128_bs4_np128_actqtriton_nsys` | Symlink to the Nsight macro output. |
| `raw/nsys_target0754_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0.*` | Rank0 Nsight report and SQLite. |
| `summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.*` | 07.54 classified profile. |

## Nsight Reprofile

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OUTPUT_DIR=/tmp/dsv4_target0754_graph_layout_4096x128_bs4_np128_actqtriton_nsys \
NSYS_BASE=/tmp/nsys_target0754_graph_layout_node_4096x128_bs4_np128_actqtriton \
performance_milestones/target07_graph_layout_replay_deforestation/scripts/nsys_graph_layout_4096x128_bs4.sh
```

Decode-envelope comparison:

| Bucket | 07.53 kernel s | 07.54 kernel s | Delta | 07.54 graph nodes |
| --- | ---: | ---: | ---: | ---: |
| graph/runtime/copy/cat/index | `1.6170` | `1.1875` | `-26.56%` | `2217` |
| elementwise graph nodes | `1.3583` | `0.6396` | `-52.91%` | `1594` |
| graph/layout cluster | `2.9752` | `1.8271` | `-38.59%` | `3811` |
| fp8 activation quant PoC | n/a | `0.0759` | n/a | `279` |
| projection/GEMM | `1.7973` | `1.7968` | `-0.03%` | `795` |
| FP8 indexer | `0.1301` | `0.1311` | `+0.75%` | `164` |
| sparse attention decode | `0.1180` | `0.1179` | `-0.02%` | `170` |
| NCCL/communication | `0.3313` | `0.3428` | `+3.49%` | `88` |
| MoE/Marlin | `0.3193` | `0.3170` | `-0.72%` | `344` |
| sampling/logits | `0.1826` | `0.1838` | `+0.68%` | `345` |

Other profile facts:

| Metric | 07.53 | 07.54 |
| --- | ---: | ---: |
| Decode envelope wall | `6.5390 s` | `5.4741 s` |
| Decode-envelope kernel total | `5.9054 s` | `4.8441 s` |
| Decode graph-layout nodes | `7996` | `3811` |
| Decode graph-layout node reduction | n/a | `52.34%` |

Interpretation:

- The targeted FP8 activation quant chain is now one `_fp8_activation_quantize_kernel`
  bucket (`0.0759 s` in decode envelope) instead of many PyTorch elementwise,
  copy, and float8 conversion graph nodes.
- Conservatively counting the new fused helper as graph/layout gives
  `1.9030 s` post-PoC cluster time, still a `36.04%` reduction from 07.53.
- The graph/layout cluster remains large at `1.8271 s`; it is now effectively
  tied with projection/GEMM (`1.7968 s`).  This justifies one more focused
  graph/layout cut, but not indefinite work in this direction.

## vLLM Boundary Comparison

| Axis | Mini before 07.54 | Mini after 07.54 | vLLM source boundary | Assessment |
| --- | --- | --- | --- | --- |
| FP8 activation quantization | PyTorch graph chain in [quantize_fp8_activation_ref](/workspace/mini-sglang/python/minisgl/kernel/deepseek_v4.py:928): `float`, `abs`, `amax`, `clamp`, `log2`, `pow`, `to(float8)`, `float`, multiply. | Opt-in Triton helper in [fp8_activation_quantize](/workspace/mini-sglang/python/minisgl/kernel/triton/deepseek_v4.py:3563). | `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py:29` registers `QuantFP8`; CUDA path calls `ops.scaled_fp8_quant` at line `123`; `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py:2056` dispatches to `_C.dynamic_*` or `_C.static_scaled_fp8_quant`. | Mini is closer: the quant boundary is no longer a large exposed PyTorch subgraph. |
| Compile/layout cleanup | Reshape/copy/cast nodes remained graph-visible around quant. | Fewer quant-related elementwise and copy graph nodes; classifier separates the fused PoC bucket. | vLLM matcher lists scaled FP8 quant ops as recognized quant ops at `/workspace/vllm-dsv4-docker/vllm/compilation/passes/fusion/matcher_utils.py:33`; noop elimination explicitly removes reshape around `static_scaled_fp8_quant` at `/workspace/vllm-dsv4-docker/vllm/compilation/passes/utility/noop_elimination.py:42`. | Mini still lacks vLLM's compile pass, but this PoC manually creates the same kind of hard quant boundary. |
| Attention/custom op boundary | Mini DSV4 attention still exposes many metadata/index/copy pieces around replay. | Unchanged in this PoC. | vLLM calls `torch.ops.vllm.deepseek_v4_attention` at `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:657` and uses fused inverse-RoPE FP8 quant plus `deepseek_v4_fp8_einsum` at lines `684-714`. | Remaining mini-vs-vLLM topology gap is now attention/layout plus projection/GEMM parity, not this FP8 quant chain. |

## Commands Used

Text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --page-size 256 \
  --output performance_milestones/target07_graph_layout_replay_deforestation/raw/text_smoke_actqtriton.json
```

4096/128 macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_graph_layout_replay_deforestation/raw/macro_4096x128_bs4_np128_actqtriton \
  --keep-going
```

4096/1024 macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_graph_layout_replay_deforestation/raw/macro_4096x1024_bs4_np128_actqtriton \
  --keep-going
```

Nsight classification:

```bash
python performance_milestones/target07_graph_layout_replay_deforestation/scripts/summarize_graph_layout_nsys.py \
  performance_milestones/target07_graph_layout_replay_deforestation/raw/nsys_target0754_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0.sqlite \
  --repeat-nvtx repeat:decode_throughput_bs8:0 \
  --json-out performance_milestones/target07_graph_layout_replay_deforestation/summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.json \
  --md-out performance_milestones/target07_graph_layout_replay_deforestation/summaries/nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.md \
  --top 40
```

## Final Decision

Decision: continue graph/layout replay deforestation for exactly one more
focused target before pivoting.

Rationale:

- Primary 07.54 profile gate passed: graph/layout cluster reduced `38.59%`.
- Secondary long-decode gate passed: 4096/1024 output tok/s improved `18.21%`.
- Graph replay remains active and graph-correct: 4096/128 replay `127`, eager
  `0`; 4096/1024 replay `1023`, eager `0`.
- Remaining graph/layout cluster (`1.8271 s`) is still top-two and effectively
  tied with projection/GEMM (`1.7968 s`).
- 4096/128 output gain did not reach `5%`; do not treat this as permission for
  broad layout churn.  The next graph/layout cut must show macro translation or
  stop.

## Do-Not-Continue Condition

Do not continue graph/layout replay deforestation after the next focused cut if
any of the following is true:

- it cannot remove at least `10%` of the fresh graph/layout cluster
  (`graph_runtime_copy_cat_index + elementwise_graph_nodes`) in a comparable
  4096/128/bs4 decode-envelope profile;
- it cannot improve 4096/128/bs4 output throughput by at least `5%` or explain
  the miss with a passing 4096/1024/bs4 long-decode gain;
- graph/layout is no longer top-two after a fresh profile;
- projection/GEMM remains at or above the graph/layout cluster after one more
  cut.

If any stop condition is hit, pivot to projection/GEMM backend parity against
vLLM.  The first gate for that pivot should compare mini `_quantized_linear_fp8_kernel`
and related BF16/CUTLASS/Marlin projection paths against vLLM's source-level
`QuantFP8`, scaled FP8 quant, `deepseek_v4_fp8_einsum`, and projection custom-op
boundaries.

Also do not resume old mini-owned FP8 indexer work, standalone
`quantize_and_insert_k_cache`, split-K sparse decode polishing, MoE/Marlin
revisit, or communication/NCCL work unless a fresh profile makes that bucket
top-two.
