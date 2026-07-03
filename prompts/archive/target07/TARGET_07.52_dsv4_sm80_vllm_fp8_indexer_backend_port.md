# TARGET 07.52: DSV4 SM80 vLLM FP8 Indexer Backend Port

## Goal

Port or closely adapt vLLM's proven DeepSeek V4 SM80 FP8 indexer backend into
mini-sglang as an opt-in experimental path.

This target exists because TARGET 07.50 proved that mini's first mini-owned
FP8 indexer slice was the wrong implementation, while TARGET 07.51 proved that
vLLM's real FP8 indexer backend is fast enough to justify a focused port.

The first win condition is backend parity, not a broad precision rewrite:

- keep mini's exact bf16 default unchanged;
- keep `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE=1` or a clearly named opt-in guard;
- match vLLM's indexer Q scale folding, FP8 indexer cache layout, and
  `fp8_paged_mqa_logits_triton` decode semantics;
- prove the new mini backend in microbench before running long macro;
- stop before full `fp8_ds_mla` KV cache E2E unless this target's evidence
  explicitly says the indexer path is no longer the right next step.

## Evidence From Prior Targets

TARGET 07.50 failed as a mini-owned FP8 indexer slice:

- exact control 4096/128/bs4: `37.9237 output tok/s`;
- mini FP8 indexer cache/logits: `29.6691 output tok/s`;
- batch16/history4096 mini bf16 logits: `0.3076 ms`;
- batch16/history4096 mini FP8 logits: `1.3072 ms`;
- batch16/history4096 mini FP8 select: `1.7368 ms`.

TARGET 07.51 isolated vLLM's real backend on A100/sm80:

| Piece | vLLM time at batch16/history4096 | 07.50 mini comparison |
| --- | ---: | ---: |
| FP8 Q path | `0.0839 ms` | mini FP8 Q `0.2308 ms` |
| FP8 indexer K store | `0.0964 ms` | mini FP8 store `0.2941 ms` |
| FP8 K gather | `0.0195 ms` | no equivalent isolated win |
| FP8 paged decode logits | `0.1529 ms` | mini bf16 logits `0.3076 ms`, mini FP8 logits `1.3072 ms` |
| FP8 logits plus topk | `0.1804 ms` | mini bf16 select `0.3586 ms`, mini FP8 select `1.7368 ms` |

Quality from TARGET 07.51 was acceptable for an opt-in prototype:

- Q path byte-exact vs vLLM torch reference;
- logits mean abs about `0.020`;
- top-k overlap about `0.973` at batch16/history4096;
- K dequant mean abs about `0.018`, max `0.25`.

Important constraint: do not port the standalone vLLM
`quantize_and_insert_k_cache` wrapper for SM80.  TARGET 07.51 showed that this
standalone probe compiles `tl.float8e4nv` and fails on A100.  vLLM's real model
path uses fused compressor/insert kernels and software-FP8 branches on SM80.
That belongs to a later packed KV-cache target, not this indexer target.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.50_dsv4_sm80_fp8_cache_indexer_precision.md`
- `prompts/TARGET_07.51_dsv4_sm80_vllm_fp8_backend_parity.md`
- `performance_milestones/target07_fp8_cache_indexer_precision/README.md`
- `performance_milestones/target07_vllm_fp8_backend_parity/README.md`
- `performance_milestones/target07_vllm_fp8_backend_parity/scripts/vllm_fp8_backend_microbench.py`

Mini code to inspect:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM code to inspect:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/mqa_logits_triton.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`
- `/workspace/vllm-dsv4-docker/vllm/_custom_ops.py`
- `/workspace/vllm-dsv4-docker/csrc/cache_kernels.cu`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_compressor.py`

Use the vLLM virtualenv when probing vLLM:

```bash
source /workspace/venvs/vllm-dsv4/bin/activate
source /workspace/mini-sglang/performance_milestones/vllm/scripts/vllm_env.sh
setup_vllm_runtime_env
```

## Work Plan

### 1. Rebuild The Exact Backend Contract

Before editing kernels, write down the exact indexer tensor contract used by
mini and vLLM:

- Q shape, dtype, head count, head dim;
- K/indexer cache value layout and scale layout;
- page table/block table layout and page size;
- context length semantics under decode graph replay;
- whether Q scale is stored separately or folded into weights;
- how top-k receives logits and valid-length masks.

The expected vLLM FP8 indexer semantics are:

- Q is quantized to E4M3 FP8;
- per-token/head Q scale is folded into `weights`;
- K is stored as FP8 values plus scales;
- decode logits use a paged kernel equivalent to
  `fp8_paged_mqa_logits_triton`;
- the logits kernel loads FP8 bytes as `uint8` and decodes through a BF16 LUT on
  SM80 instead of relying on native Triton FP8 casts.

Record this contract in the milestone README before implementation.

### 2. Replace The Slow Mini FP8 Backend With A vLLM-Aligned Backend

Implement the smallest opt-in path that can reproduce the fast vLLM indexer
microbench pieces.

Prioritize in this order:

1. Port or closely adapt vLLM's `fp8_paged_mqa_logits_triton` decode logits
   kernel into `python/minisgl/kernel/triton/deepseek_v4.py`.
2. Match vLLM's FP8 indexer cache layout and scale handling in
   `python/minisgl/kvcache/deepseek_v4_pool.py` and the store/select wrappers.
3. Match vLLM's Q scale folding semantics.  The logits call should receive
   folded FP32 weights rather than a separate Q scale tensor.
4. Replace mini's 07.50 FP8 Q/store helpers only as much as needed for the
   backend microbench and decode integration to match vLLM's isolated behavior.

Temporary bridge rule:

- A local proof may call installed vLLM ops behind a very explicit experimental
  guard if that is the fastest way to confirm semantics.
- A promoted mini path should not require importing vLLM at runtime.  If a
  vLLM external bridge is used, the README must say exactly which source should
  be vendored or reimplemented next.

Do not port full `fp8_ds_mla` KV cache in this target.

### 3. Backend Microbench Gate

Create or extend a mini-side script under:

```text
performance_milestones/target07_vllm_fp8_indexer_backend_port/scripts/
```

Benchmark the same representative shapes as TARGET 07.51:

- batch `1`, history `1024`;
- batch `4`, history `2048`;
- batch `16`, history `4096`;
- indexer heads `64`;
- head dim `128`;
- topk width `512`;
- page size `256`.

Measure at least:

- Q quant plus folded weights;
- K store/quant/cache;
- paged FP8 logits only;
- logits plus topk/select;
- end-to-end indexer select boundary.

Compare against:

- TARGET 07.50 mini bf16 and mini FP8 numbers;
- TARGET 07.51 vLLM isolated numbers;
- the new mini vLLM-aligned FP8 backend.

Microbench success threshold:

- at batch16/history4096, new mini paged FP8 logits should be within `1.25x`
  of vLLM's isolated `0.1529 ms`, or beat mini bf16 logits by at least `20%`;
- new mini logits plus topk/select should beat mini bf16 select by at least
  `15%` on the large representative shape;
- Q/store helpers should not be more than `1.5x` slower than vLLM after one
  focused fix attempt.

If these thresholds fail, stop before macro and write the reason.

### 4. Quality Gate

For every backend shape, compare against the existing bf16 path:

- logits mean/max absolute error;
- top-k overlap;
- K dequant mean/max absolute error;
- whether invalid/padded positions stay masked;
- text smoke with TP8, page size 256.

Suggested smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe \
  --page-size 256 \
  --output /tmp/dsv4_target0752_text_smoke.json
```

Use the actual promoted variant name if it differs from `v1_moe`.

### 5. Macro Gate

Only run macro if the backend microbench gate passes.

First run the short macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --output-dir /tmp/dsv4_target0752_fp8_indexer_4096x128_bs4 \
  --keep-going
```

Run 4096/1024/batch4 only if 4096/128 improves at least `5%` over the current
same-run exact control and text smoke passes.

Official comparison points:

- current best exact 4096/128/batch4: `38.94 output tok/s`;
- current best exact 4096/1024/batch4: `68.81 output tok/s`;
- old serving victory line: `114.07 output tok/s`;
- fresh vLLM 4096/1024/batch4 reference: about `202 output tok/s`.

## Explicitly Out Of Scope

- full `fp8_ds_mla` KV cache E2E;
- porting standalone `quantize_and_insert_k_cache`;
- changing mini's exact bf16 default;
- MoE/Marlin work;
- aux-stream ablations;
- persistent-topk-only tuning;
- local micro-optimizations after the backend gate fails;
- 4096/1024 macro without a passing 4096/128 and microbench gate.

## Follow-Up Decision Rules

End the README with exactly one next decision:

- `Decision: promote opt-in vLLM-aligned FP8 indexer and reprofile macro`;
- `Decision: indexer backend microbench passes but macro integration loses; next target is graph/layout integration`;
- `Decision: stop FP8 indexer port and probe fused fp8_ds_mla KV cache store/gather`;
- `Decision: stop precision lane and return to exact runtime/prefill`;
- `Decision: blocked by missing vLLM package/op ...`.

If the next decision is `probe fused fp8_ds_mla KV cache store/gather`, state
that the correct vLLM reference is the fused compressor/insert model path with
SM80 software-FP8 handling, not standalone `quantize_and_insert_k_cache`.

## Expected Output

Create:

- `performance_milestones/target07_vllm_fp8_indexer_backend_port/README.md`
- `performance_milestones/target07_vllm_fp8_indexer_backend_port/scripts/`
- `performance_milestones/target07_vllm_fp8_indexer_backend_port/raw/`
- `performance_milestones/target07_vllm_fp8_indexer_backend_port/summaries/`

The README must include:

- backend contract table;
- mini-vs-vLLM source mapping;
- implementation summary and env flags;
- microbench table against 07.50 and 07.51;
- quality table;
- macro results if run;
- final decision and do-not-continue condition.

