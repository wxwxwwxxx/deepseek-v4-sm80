# TARGET 12.49: DSV4 SM80 Release Long-Context And Large-Batch Soak

## Background

TARGET 12.47 promoted SGLang-style in-graph replay metadata prep for the
supported A100/sm80 DSV4 Route-B prefix baseline. TARGET 12.48 then folds that
recipe into the DeepSeek V4 release defaults so a normal Engine/LLM
construction can use the optimized path without manually setting page size,
prefix/cache ownership, graph buckets, or kernel env toggles.

Mature serving frameworks generally avoid making users hand-author every CUDA
graph batch bucket.  They expose a maximum capture size and derive a bucket
list with dense coverage for small batches and coarser spacing for larger
batches.  For example, the local vLLM reference in
`/workspace/vllm-dsv4-docker/vllm/config/vllm.py` derives capture sizes roughly
as `[1,2,4]`, then step 8 below 256, then step 16 up to a capped maximum such as
`min(max_num_seqs * decode_query_len * 2, max_num_batched_tokens, 512)`.
Mini-sglang already has an older helper in `python/minisgl/engine/graph.py`
that can derive `[1,2,4] + range(8, max+1, 8)` from `cuda_graph_max_bs`; TARGET
12.49 should evaluate whether release defaults should keep an explicit small
bucket list or move to a vLLM/SGLang-style `max_bs -> generated buckets` policy.

Release default intent:

```text
LLM("/models/DeepSeek-V4-Flash", ...)

page_size defaults from 1 to 256 for DSV4
radix prefix cache enabled
component loc ownership enabled
attention_backend="dsv4"
cuda_graph_bs=[1,2,4,8,16]
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
PyNCCL threshold32m remains the default TP communication policy on sm80
```

Fallback/oracle paths are still available through benchmark variants or
`MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1`. Do not use that env for normal
serving unless intentionally testing old/fallback behavior.

## Goal

Prove the release-default path is stable beyond the historical
`4096/1024/bs4` line:

1. Long-context sanity for progressively longer prefills.
2. Larger active decode batches and serving-style waves.
3. CUDA graph bucket behavior and private-pool memory cost for larger buckets.
4. Kernel/backend behavior when batch/token dimension `M` grows.
5. Clear go/no-go guidance for whether release defaults should stay at the
   conservative explicit bucket list, adopt a generated `cuda_graph_max_bs`
   policy, or gain only a few additional tested buckets.

Do not default-promote `cuda_graph_max_bs=256` or `2048` in this target unless
the memory, correctness, and throughput evidence is clean. The expected first
answer may be "keep release default at `[1,2,4,8,16]`; implement an automatic
bucket policy later; larger batches run eager or need a graph-memory target."

## Required Setup

Use the current branch and system Python for mini-sglang. Use one variant per
fresh `torchrun` process when checking release defaults, because CUDA graph
capture and env-dependent graph init should not be compared in a multi-variant
same-process run.

Use the release-default path first, not the historical fully spelled-out env
recipe. That means omit manual page/prefix/graph/env flags where possible and
record what Engine resolved.

For TP8:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --scenarios historical_4096_128_bs4 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_49_release_smoke \
  --keep-going
```

If the benchmark harness still requires explicit variant-level prefix flags,
record that as a harness cleanup issue. The Engine path itself should default
page size/prefix/component ownership.

## Phase 1: Release-Default Smoke

Run a text smoke with as few manual flags as the harness permits:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_49_release_text_smoke.json
```

Record:

- resolved page size;
- resolved `num_pages` and token capacity;
- active DSV4 env/toggles;
- graph buckets captured;
- replay/eager counts;
- text sanity status.

## Phase 2: Long-Context Ladder

Use synthetic prompts first so the test is reproducible and does not need a
dataset. Run a progressive ladder and stop early if one rung fails:

```text
prompt/decode/batch:
8192 / 16 / 1
32768 / 16 / 1
65536 / 8 / 1
131072 / 4 / 1
```

For each rung:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --scenarios historical_4096_128_bs4 \
  --prompt-len <PROMPT> \
  --decode-len <DECODE> \
  --batch-size <BATCH> \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_49_long_<PROMPT> \
  --keep-going
```

If a long rung hits memory pressure, record planned token capacity and actual
failure mode. Do not hide OOM by shrinking the model path or disabling release
features unless doing an explicit A/B.

## Phase 3: Large-Batch Decode Ladder

Start from prompt length 128 and moderate decode length so the test isolates
decode/batch behavior:

```text
batch sizes: 16, 32, 64, 128, 256
prompt_len=128
decode_len=64
```

Run first with release-default graph buckets `[1,2,4,8,16]`; larger batches may
run eager. Record whether eager fallback is expected and whether throughput
still scales.

Before proposing a new default, do a short source-parity check against vLLM and
SGLang if full SGLang source is available locally:

- What public knob do they expose (`max_cudagraph_capture_size`,
  `cudagraph_capture_sizes`, `cuda_graph_max_bs`, etc.)?
- What bucket spacing do they derive from the max?
- What hard cap or scheduler-derived cap do they apply to avoid startup time
  and private-pool memory blowups?
- How do they handle runtime batch sizes between buckets and above the largest
  bucket?

Treat vLLM's local implementation as the first concrete reference and state
which conclusions are source-derived.

Then try generated-policy candidates as separate processes. If the current
benchmark CLI can pass only explicit `--cuda-graph-bs`, synthesize the candidate
lists and record that exposing `--cuda-graph-max-bs` in the harness is a cleanup
item. Candidate policies:

```text
current release explicit: [1,2,4,8,16]
mini legacy max32:        [1,2,4] + range(8, 33, 8)
mini legacy max64:        [1,2,4] + range(8, 65, 8)
mini legacy max128:       [1,2,4] + range(8, 129, 8)
mini legacy max256:       [1,2,4] + range(8, 257, 8)
vLLM-style max256:        [1,2,4] + range(8, 256, 8) + [256]
vLLM-style max512:        [1,2,4] + range(8, 256, 8) + range(256, 513, 16)
```

For compatibility with the existing harness, the staged explicit bucket
expansions are:

```text
--cuda-graph-bs 1 2 4 8 16 32
--cuda-graph-bs 1 2 4 8 16 32 64
--cuda-graph-bs 1 2 4 8 16 32 64 128
--cuda-graph-bs 1 2 4 8 16 32 64 128 256
```

Do not jump directly to 2048. If 256 is clean and useful, write a follow-up
target for 512/1024/2048 with explicit graph private-pool and capture-time
memory accounting.

Record:

- capture success/failure per bucket;
- per-bucket private-pool memory delta;
- graph replay/eager counts;
- output/decode tok/s;
- top kernel/backend changes if `M` grows;
- any new Triton/CUDA shape limitations.
- whether release should expose `cuda_graph_max_bs` and auto-generate buckets,
  rather than requiring users to pass explicit lists.

## Phase 4: Kernel/Backend M-Growth Check

If large batch shows a cliff, use small no-weight or one-layer harnesses before
full-model bisection. Check at least:

- sparse attention C4/C128 kernels;
- MoE Marlin WNA16 route/gemm path;
- projection/cache kernels;
- in-graph metadata prep kernel;
- sampler/graph output buffer sizing.

Prefer microbench scripts and targeted tests over full model loops until a
specific backend is implicated.

## Validation

Minimum correctness:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/kernel/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy
```

Minimum release smoke:

- text smoke passes with no garbled output;
- `historical_4096_128_bs4` replay/eager remains zero-eager for captured
  buckets;
- active toggles include in-graph metadata prep;
- automatic `num_pages` reports a plausible capacity.

## Stop Conditions

Stop and report when:

1. Release-default smoke fails correctness or text sanity.
2. A long-context rung fails and the failure mode is identified.
3. Large-batch graph expansion shows memory cost that outweighs throughput.
4. Larger `M` exposes a clear backend limitation that deserves its own target.
5. 16/32/64 batch serving-style workloads are stable and no top bottleneck is
   newly exposed.
6. The source-parity bucket policy and empirical bucket cost are sufficient to
   recommend one of: keep explicit small buckets, adopt generated max-bs policy,
   or defer larger graph capture to a separate memory-policy target.

Do not spend the whole target polishing a single large-batch kernel unless the
soak proves it is the dominant release blocker.
