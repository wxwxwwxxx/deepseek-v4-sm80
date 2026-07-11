# TARGET 12.49: DSV4 SM80 Release Long-Context And Large-Batch Soak

## Background

This target is now a **rerun after TARGET 12.52**, not the older pre-SWA
release-default check.

The current true no-env release default is the TARGET 12.52 bundle:

```text
performance_milestones/target12_swa_independent_release_default_cleanup/README.md
```

That gate promoted SWA independent lifecycle and SWA direct/page-table/replay
metadata into `dsv4_sm80_release_default`.

Current release-default behavior:

```text
LLM("/models/DeepSeek-V4-Flash", ...)

page_size defaults from 1 to 256 for DSV4
attention_backend=dsv4
radix prefix cache enabled
component loc ownership enabled
cuda_graph_bs=[1,2,4,8,16]

MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
```

TARGET 12.52 true no-env smoke passed with:

```text
captured buckets: [16,8,4,2,1]
replay/eager: 9 / 0
prep_metadata_in_graph=true
planned capacity: 6604 pages / 1,690,624 tokens
```

The 12.52 four-scenario macro also passed with zero eager decode fallback:

```text
historical_4096_128_bs4:      52.611 output tok/s
historical_4096_1024_bs4:     141.863 output tok/s
serving_mixed_112req_wave16:  171.736 output tok/s
prefix_multi_112req_wave16:   113.486 output tok/s
```

TARGET 12.49 should now answer whether this release default remains sane for
longer contexts, larger active decode batches, and larger CUDA graph bucket
policies.

## Goal

Prove or bound the current release-default serving envelope:

1. Long-context sanity for progressively longer prefills.
2. Larger active decode batches and serving-style waves.
3. CUDA graph bucket behavior and private-pool memory cost for larger buckets.
4. Backend/kernel behavior as the decode batch dimension grows.
5. Whether mini should keep explicit small graph buckets `[1,2,4,8,16]`,
   expose a generated `cuda_graph_max_bs` policy, or add only a small tested
   set of extra buckets.

Do not default-promote `cuda_graph_max_bs=256`, `512`, or larger in this target
unless correctness, memory, and throughput evidence are clean.  A valid result
is: "the release default stays conservative; larger buckets need a separate
graph-memory or scheduler-policy target."

## Required Setup

Use the current branch and system Python for mini-sglang.

Run release-default checks in fresh `torchrun` processes.  Do not compare
multiple graph/env-dependent variants inside one Python process.

For the default path, use:

```text
--variants dsv4_sm80_release_default
--num-pages 0
```

Do **not** manually set DSV4 release env variables for normal release-default
runs.  The whole point is to verify the true no-env default.

Fallback/oracle remains:

```bash
MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1
```

Only use fallback/oracle when explicitly comparing against old behavior.

## Phase 0: Static And Unit Sanity

Run:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/engine/test_marlin_wna16_release_credit.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/kernel/test_deepseek_v4_wrappers.py::test_direct_decode_index_metadata_for_replay_swa_independent_matches_oracle \
  tests/kernel/test_deepseek_v4_wrappers.py::test_prep_decode_metadata_in_graph_swa_independent_matches_direct_oracle
```

If this is too slow for an initial run, at least run the release-default,
benchmark, text-smoke, and SWA metadata oracle subsets, then record the skipped
coverage explicitly.

## Phase 1: True Release-Default Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_49_release_default_text_smoke.json
```

Required signals:

```text
text sanity: pass, no garble
captured buckets: [16,8,4,2,1]
decode replay/eager: replay > 0, eager = 0
prep_metadata_in_graph_requested=true
prep_metadata_in_graph=true
prep_metadata_in_graph_unsupported_reason=null
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
planned capacity: around the TARGET 12.52 SWA scale, about 1.6M tokens on TP8 A100
```

Record resolved page size, `num_pages`, capacity, graph private-pool delta,
active DSV4 toggles, and text outputs.

## Phase 2: Long-Context Ladder

Use synthetic prompts first so the test is reproducible and does not require a
dataset.  Run progressively and stop early if one rung fails:

```text
prompt/decode/batch:
8192   / 16 / 1
32768  / 16 / 1
65536  / 8  / 1
131072 / 4  / 1
262144 / 2  / 1   optional stretch if earlier rungs are clean
```

For each rung:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
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

Record:

- pass/fail and text/sanity warnings if available;
- planned pages/tokens and used tokens;
- prefill latency and throughput;
- decode replay/eager counts;
- graph private-pool delta;
- peak/resolved memory and failure mode if any.

If a long rung hits memory pressure, record the exact capacity ledger and
actual exception.  Do not hide OOM by disabling SWA, Marlin release, radix, or
graph unless doing a named A/B.

## Phase 3: Large-Batch Decode Ladder

Start with short prompt length and moderate decode length so the test isolates
decode/batch behavior:

```text
batch sizes: 16, 32, 64, 128, 256
prompt_len=128
decode_len=64
```

Run first with the current release-default graph buckets `[1,2,4,8,16]`.
Larger batches may run eager.  That is acceptable for the first pass; record
whether throughput still scales and whether eager fallback is expected.

For each batch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 128 \
  --decode-len 64 \
  --batch-size <BATCH> \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_49_large_bs_<BATCH> \
  --keep-going
```

Record output/decode tok/s, replay/eager counts, peak memory, graph-private
pool, and whether any kernel/backend shape limitation appears.

## Phase 4: CUDA Graph Bucket Policy Probe

Before proposing a new default, do a short source-parity check against vLLM and
SGLang:

- What public knob do they expose (`max_cudagraph_capture_size`,
  `cudagraph_capture_sizes`, `cuda_graph_max_bs`, etc.)?
- What bucket spacing do they derive from the max?
- What hard cap or scheduler-derived cap avoids startup time and private-pool
  memory blowups?
- How do they handle runtime batch sizes between buckets and above the largest
  bucket?

Local references:

```text
/workspace/vllm-dsv4-docker
/workspace/venvs/vllm-dsv4
/workspace/sglang-main
python/minisgl/engine/graph.py
```

Treat vLLM/SGLang conclusions as source-derived unless measured in mini.

Then try staged explicit bucket expansions in fresh processes:

```text
current release explicit: 1 2 4 8 16
small extension:          1 2 4 8 16 32
medium extension:         1 2 4 8 16 32 64
large extension:          1 2 4 8 16 32 64 128
stretch extension:        1 2 4 8 16 32 64 128 256
```

Use the shortest useful large-batch probe first, for example batch 32/64 with
`prompt_len=128`, `decode_len=64`.  If capture cost looks high, stop before
128/256 and write a graph-memory target.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 128 \
  --decode-len 64 \
  --batch-size 64 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --cuda-graph-bs 1 2 4 8 16 32 64 \
  --output-dir /tmp/dsv4_target12_49_graph_bs64 \
  --keep-going
```

Record:

- capture success/failure per bucket list;
- graph private-pool memory delta;
- graph replay/eager counts;
- output/decode tok/s;
- startup/capture time if available;
- whether `cuda_graph_max_bs` should be exposed and auto-generate buckets.

Do not jump to 512/1024/2048 inside this target.  If 128 or 256 is useful and
safe, write a follow-up target for larger max-bs policies.

## Phase 5: Kernel/Backend M-Growth Check

If large batch shows a cliff, use small no-weight or one-layer harnesses before
full-model bisection.  Check likely owners:

- sparse attention C4/C128 kernels;
- MoE Marlin WNA16 route/GEMM path;
- in-graph metadata prep kernel;
- direct SWA/C4 graph metadata writers;
- sampler/graph output buffer sizing;
- communication and final all-gather owners if batch increases token traffic.

Prefer microbench scripts and targeted tests over full model loops until a
specific backend is implicated.

## Output

Write the report to:

```text
performance_milestones/target12_release_long_context_large_batch_soak/README.md
```

The report must include:

- git commit and dirty-state summary;
- static/unit test result;
- true release-default text smoke result;
- active DSV4 env/toggles observed at runtime;
- long-context ladder table;
- large-batch ladder table;
- CUDA graph bucket policy/source-parity summary;
- graph private-pool memory and capture behavior;
- capacity ledger in pages/tokens/bytes;
- recommended next step:
  - keep current `[1,2,4,8,16]`;
  - promote a small extra bucket set;
  - implement generated `cuda_graph_max_bs`;
  - open a graph-memory target;
  - open a backend/kernel M-growth target.

## Stop Conditions

Stop and report when:

1. True release-default smoke fails correctness or text sanity.
2. A long-context rung fails and the failure mode is identified.
3. Large-batch graph expansion shows memory cost that outweighs throughput.
4. Larger `M` exposes a clear backend limitation that deserves its own target.
5. Batch 16/32/64 serving-style workloads are stable and no top bottleneck is
   newly exposed.
6. Source-parity and empirical bucket cost are sufficient to recommend one of:
   keep explicit small buckets, add a small bucket set, adopt generated
   max-bs policy, or defer larger graph capture to a separate target.

Do not spend the whole target polishing a single large-batch kernel unless the
soak proves it is the dominant release blocker.
