# TARGET 12.56: DSV4 SM80 Chunked Prefill Long-Context Path

## Background

TARGET 12.54 and TARGET 12.55 established the current long-context boundary:

```text
performance_milestones/target12_post_hc_release_envelope_rerun/README.md
performance_milestones/target12_graph_activation_memory_accounting/README.md
```

The post-HC release default now passes:

```text
32768 / 16 / 1
```

but full single-forward prefill still fails at:

```text
65536 / 8 / 1
```

TARGET 12.55 proved this is not a small missing KV reserve.  Lowering
`memory_ratio` from `0.90` to `0.85` freed about `3.92 GiB` of KV capacity
(`524 pages`, `134,144 tokens`) and still failed.  The failure owner moved as
more headroom was made available:

```text
0.90:  attention wo_a BF16 BMM, 128 MiB
0.885: MoE gate fallback, 1024 MiB
0.875: Marlin WNA16 MoE route_out, 3 GiB
0.85:  Marlin WNA16 MoE route_out, 3 GiB
```

The Marlin WNA16 owner is expected for a full 64k prefill with the current
wrapper shape:

```text
route_out = [tokens * topk, hidden] bf16
```

For `tokens=65536`, `topk=8`, and `hidden ~= 3072`, this is about `3 GiB`.
This is a full-prefill activation/workspace scaling problem.  The correct
first release route is to bound the maximum prefill forward size by chunking.

Important existing mini code:

```text
python/minisgl/scheduler/prefill.py
python/minisgl/scheduler/config.py
python/minisgl/scheduler/scheduler.py
```

Mini already has a `ChunkedReq` / `PrefillManager.token_budget` skeleton.
`SchedulerConfig.max_extend_tokens` defaults to `8192`, but the offline perf
matrix currently sets:

```text
max_extend_tokens = args.max_extend_tokens or _max_extend_tokens(scenarios)
```

so TARGET 12.54 and 12.55 intentionally ran full single-forward prefill unless
`--max-extend-tokens` was explicitly passed.  Do not assume chunked prefill is
absent; first validate and harden the existing mechanism.

## Goal

Implement or promote a conservative DSV4 release chunked-prefill path that:

1. keeps decode CUDA graph behavior unchanged;
2. keeps prefill/chunked prefill eager for the first version;
3. bounds per-forward prefill tokens with a configurable chunk token budget;
4. preserves DSV4 radix prefix cache, component loc ownership, SWA independent
   lifecycle, Marlin WNA16 release, and in-graph decode metadata;
5. makes `65536 / 8 / 1` pass without a large global KV capacity reserve;
6. provides a clear path toward `131k`, `262k`, `512k`, and eventually `1M`
   single-request context.

This target should answer:

```text
What is the largest safe DSV4 A100/sm80 release prefill chunk size?
Can existing ChunkedReq scheduling be promoted, or does it need correctness or
lifecycle fixes first?
```

Do not implement prefill CUDA graph in this target.  Do not expand default
decode CUDA graph buckets in this target.  Do not revisit MTP or low precision.

## Conceptual Contract

Use these meanings consistently:

```text
decode CUDA graph bs:
  Number of active decode request rows.  Usually one generated token per row.

prefill chunk token budget:
  Maximum total extend tokens processed by one prefill forward.

prefill batch token count:
  sum(req.extend_len for req in batch.reqs) for the current chunk.
```

`cuda_graph_max_bs` / decode graph buckets are not the right knob for 64k
prefill memory.  The right first knob is `max_extend_tokens` or a DSV4 release
alias for it.

First-version behavior:

```text
prefill/chunked prefill: eager forward, chunked by token budget
decode: existing CUDA graph replay for captured decode buckets
```

Correctness invariants:

- A `ChunkedReq` must not be sampled or returned to the user.
- After each chunk, the request's committed/cache-visible prefix must advance
  exactly by the chunk length.
- Page table, token pool, KV/component writes, SWA independent mapping, and
  radix prefix ownership must remain valid across chunks.
- Final non-chunk prefill for the request may sample the first output token and
  then enter decode.
- Prefix cache insertion should only expose complete and valid component/SWA
  state, not half-published chunk state.
- Decode graph replay counts should remain zero-eager for captured decode
  buckets after chunked prefill finishes.

## References To Read First

Mini:

```text
performance_milestones/target12_graph_activation_memory_accounting/README.md
performance_milestones/target12_post_hc_release_envelope_rerun/README.md
python/minisgl/scheduler/prefill.py
python/minisgl/scheduler/scheduler.py
python/minisgl/scheduler/cache.py
python/minisgl/scheduler/config.py
python/minisgl/core.py
python/minisgl/engine/engine.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

SGLang/vLLM source references:

```text
/workspace/sglang-main/python/sglang/srt/managers/scheduler.py
/workspace/sglang-main/python/sglang/srt/managers/schedule_batch.py
/workspace/sglang-main/python/sglang/srt/managers/scheduler_components/
/workspace/sglang-main/python/sglang/srt/mem_cache/
/workspace/vllm-dsv4-docker/vllm/engine/arg_utils.py
/workspace/vllm-dsv4-docker/vllm/v1/core/scheduler.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py
```

Use these frameworks as design references for chunk budget, scheduling policy,
and memory planning.  Do not require vLLM runtime for this target.

## Phase 0: Static And Unit Baseline

Run:

```bash
python -m py_compile \
  python/minisgl/core.py \
  python/minisgl/scheduler/config.py \
  python/minisgl/scheduler/prefill.py \
  python/minisgl/scheduler/scheduler.py \
  python/minisgl/scheduler/cache.py \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

If scheduler/cache tests exist for prefix/SWA lifecycle, run the relevant
subset too.  If they do not exist, add focused unit tests when changing
chunked-prefill behavior.

## Phase 1: Source Parity And Mini Contract

Before changing code, write a short source-derived contract in the report:

- how mini currently schedules `ChunkedReq`;
- how `cached_len`, `device_len`, `extend_len`, and `can_decode` evolve across
  chunks;
- when page table entries are populated;
- when KV/component/SWA cache entries are written;
- when radix prefix cache insertion happens;
- how SGLang/vLLM bound prefill tokens or batched tokens;
- which behavior mini should match for first release.

Pay special attention to this existing mini behavior:

```text
SchedulerConfig.max_extend_tokens default: 8192
perf matrix default: max_extend_tokens = scenario max unless explicitly passed
```

The target must distinguish serving defaults from benchmark defaults.

## Phase 2: No-Code Existing-Chunk Sweep

First test the existing chunking mechanism by passing `--max-extend-tokens`.
Use true no-env release default:

```text
--variants dsv4_sm80_release_default
--num-pages 0
```

Run `65536 / 8 / 1` with these chunk budgets:

```text
max_extend_tokens: 32768, 24576, 16384, 8192
```

Command template:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 65536 \
  --decode-len 8 \
  --batch-size 1 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --max-extend-tokens <CHUNK_TOKENS> \
  --output-dir /tmp/dsv4_target12_56_chunk_<CHUNK_TOKENS>_65536 \
  --keep-going
```

For each row record:

- pass/fail and owner;
- number of prefill forwards;
- max observed prefill `input_tokens`;
- max observed `extend_len`;
- prefill forward time and TTFT;
- planned pages/tokens;
- fixed SWA cache bytes;
- graph private-pool delta;
- peak allocated/reserved memory;
- replay/eager decode counts after prefill;
- generated text sanity if available.

Interpretation:

- If `32768` passes, prefer it as the first release candidate because it keeps
  common `4096 x bs4` prefill single-forward.
- If `32768` fails but `16384` passes, use `16384` as the first candidate.
- If only `8192` passes, measure macro regressions carefully before promotion.
- If all chunk sizes fail, identify whether the existing ChunkedReq mechanism is
  incorrect or whether a backend still allocates full-request workspace.

## Phase 3: Correctness And Lifecycle Probes

Chunking changes cache publication and request lifecycle.  Run correctness
checks before performance promotion.

### Same-Length Oracle

Use a shape that passes unchunked and chunked:

```text
32768 / 16 / 1
```

Compare:

```text
unchunked: --max-extend-tokens 32768
chunked:   --max-extend-tokens 16384 or selected candidate
```

Record output tokens/text, logits or token equality if the harness supports it,
prefix-cache metrics, and DSV4 cache integrity reports.

### Long Text Smoke

Run a long synthetic prompt or repeated natural prompt with the selected chunk
budget and `--fail-on-warning`.  It is acceptable to add a small benchmark/text
smoke helper if current scripts cannot express the shape cleanly.

### Prefix/SWA Safety

If possible, run a shared-prefix case where the prefix is chunked on first
insert and reused by later requests.  Verify:

- no stale prefix handle;
- no invalid SWA page mapping;
- component page ownership remains valid;
- replay/eager decode remains expected.

If a correctness issue appears, stop broad performance work and fix the
lifecycle bug first.

## Phase 4: Implement Or Promote Chunk Budget

If Phase 2 proves that existing chunking works only when explicitly passing
`--max-extend-tokens`, decide whether code changes are needed.

Likely acceptable outcomes:

```text
NO_CODE_PROMOTION:
  Existing serving default already chunks; update benchmark/release docs and
  release soak commands to use an explicit DSV4 prefill chunk budget.

BENCHMARK_FIX:
  perf_matrix currently disables chunking by default for long scenarios; add a
  release-long-context mode or clearer CLI alias so future release soak does not
  accidentally test only monolithic prefill.

RUNTIME_PROMOTION:
  Add a DSV4 A100/sm80 release-default prefill chunk budget if actual LLM()
  serving does not already use one.

LIFECYCLE_FIX:
  Existing ChunkedReq scheduling is present but has DSV4 cache/SWA/prefix bugs;
  fix those before promotion.
```

Potential knobs, only if needed:

```text
MINISGL_DSV4_SM80_PREFILL_CHUNK_TOKENS
MINISGL_DSV4_SM80_PREFILL_CHUNK_DISABLE=1
```

Prefer config/CLI clarity over hidden env behavior.  If the existing
`max_extend_tokens` config is enough, do not add redundant env knobs.

Selection criteria for default chunk budget:

- Largest chunk that passes `65536/8/1`.
- Does not split the historical `4096/1024/bs4` prefill if avoidable.
- Keeps fixed SWA cache and activation peaks within safe memory headroom.
- Does not introduce text sanity or prefix-cache correctness failures.

## Phase 5: Long-Context Ladder

After selecting a candidate chunk budget, run a ladder:

```text
65536  / 8 / 1
131072 / 4 / 1
262144 / 2 / 1
524288 / 1 / 1 optional stretch if time allows
1048576 / 1 / 1 optional design probe only; do not chase performance in this target
```

Use the selected chunk budget.  Keep each rung as a fresh `torchrun`.

Record:

- pass/fail;
- TTFT and prefill tokens/s;
- number of prefill chunks;
- peak memory;
- planned pages/tokens;
- fixed SWA bytes;
- decode replay/eager;
- failure owner if any.

Stop the ladder when a new nontrivial owner appears.  Do not spend this target
optimizing a new backend owner beyond diagnosis.

## Phase 6: Macro And Large-Batch Guard

Run enough guards to decide whether the candidate is safe as a release default:

```text
historical_4096_128_bs4
historical_4096_1024_bs4, if runtime allows
128 / 64 / 128
128 / 64 / 256
```

Important: if the selected chunk budget is smaller than common batched prefill
sizes, report the throughput/TTFT cost explicitly.  Do not promote a tiny chunk
size just because it passes 64k if it badly regresses normal serving.

## Phase 7: Decision

End with one of:

```text
PROMOTE_CHUNKED_PREFILL:
  Selected chunk budget passes correctness, 64k, and guard workloads with
  acceptable capacity/performance tradeoff.

EXPERIMENTAL_CHUNKED_PREFILL:
  Chunking works and enables long context, but macro/serving cost or correctness
  coverage is not yet release-default quality.

LIFECYCLE_FIX_REQUIRED:
  Existing ChunkedReq path has DSV4 cache/SWA/prefix correctness bugs.

BACKEND_FIX_REQUIRED:
  A backend still allocates full-request workspace even under chunking.

SCHEDULER_REFACTOR_REQUIRED:
  Current scheduler cannot express serving-grade chunked prefill cleanly.
```

If `PROMOTE_CHUNKED_PREFILL`, update the TARGET 12 route and recommend the next
target: graph/private-pool accounting for larger decode buckets, fallback
census, or serving-style workload soak.  If long-context still fails, specify
whether the next step is lifecycle repair or backend workspace optimization.

## Output

Write:

```text
performance_milestones/target12_chunked_prefill_long_context/README.md
```

The report must include:

- git commit and dirty-state summary;
- source parity and mini chunked-prefill contract;
- no-code chunk sweep table;
- selected chunk budget and rationale;
- correctness/lifecycle probe results;
- long-context ladder;
- macro/large-batch guard;
- capacity and memory ledger before/after chunking;
- decision and next target recommendation.

## Stop Conditions

Stop and report when:

1. existing chunking is proven correct and a default candidate is identified;
2. a DSV4 lifecycle correctness bug appears;
3. chunking fails to reduce the relevant workspace owner;
4. 64k passes but macro guard regresses enough that default promotion is not
   justified;
5. the work starts drifting into prefill CUDA graph, low precision, MTP, or
   broad backend rewrites.
