# TARGET 12.58: DSV4 SM80 Post-Indexer Long-Context Release Envelope

## Status

Run after TARGET 12.57.

TARGET 12.57 fixed the first unbounded long-context indexer memory surface by
adapting the vLLM-style bounded query-row contract:

```text
FP8 paged indexer logits workspace <= 512 MiB
each bounded logits slice is consumed immediately by native top-k
only the final [rows, 512] int32 selection surface survives
Route-B component/full remap is fused in Triton
```

The former 2.25 GiB indexer-logits failures now pass:

```text
131072 / max_extend_tokens 24576: fail -> pass
262144 / max_extend_tokens 16384: fail -> pass
```

The true serving default remains:

```text
max_extend_tokens=8192
page_size=256
SWA independent lifecycle enabled
dsv4_sm80_release_default
```

TARGET 12.57 did not complete the 512k/1M ladder.  Its only new OOM owner was a
manual `65536 / max_extend_tokens=32768` experiment, where Marlin WNA16 needed
a 1.50 GiB routed output:

```text
32768 tokens * topk 6 * hidden 4096 * bf16 2 bytes = 1.50 GiB
```

This is not yet a release-default blocker.  Existing measurements also do not
show that 16384/24576-token chunks are consistently faster than 8192-token
chunks.  Do not start by rewriting Marlin.

## Purpose

Establish a repeat-stable promotion gate for TARGET 12.57, then measure the
actual single-request long-context envelope of the true release default at
512k and 1M context.

The target must answer:

1. Is the bounded indexer/remap path performance-neutral enough to promote?
2. Does the default 8192-token chunk policy reach 512k and 1M without OOM,
   scheduler spin, malformed output, or SWA/component lifecycle failure?
3. What is the first measured owner if the ladder fails or scales poorly?
4. Is the next implementation target Marlin routed-output lifetime, streaming
   indexer logits+top-k, cache capacity/lifecycle, or something else?

This is an attribution and release-envelope target.  Implement only a small,
obvious correctness/instrumentation fix if it is required to obtain reliable
evidence.  Otherwise stop after naming and quantifying the next owner.

## Non-Goals

- Do not re-enable or modify MTP.
- Do not change model precision or add FP8/INT8 experiments.
- Do not increase the release `max_extend_tokens` default above 8192.
- Do not add prefill CUDA graph capture.
- Do not expand decode CUDA graph buckets.
- Do not write a new Marlin kernel in this target.
- Do not write a streaming indexer/top-k kernel in this target.
- Do not tune unrelated decode metadata or communication paths.
- Do not claim 512k/1M throughput from a failed or one-off partial run.

## References To Read First

Evidence and plans:

```text
performance_milestones/target12_release_fallback_census_native_backend_gate/README.md
performance_milestones/target12_chunked_prefill_long_context/README.md
performance_milestones/target12_graph_activation_memory_accounting/README.md
prompts/TARGET_12.56_dsv4_sm80_chunked_prefill_long_context.md
prompts/TARGET_12.57_dsv4_sm80_release_fallback_census_native_backend_gate.md
```

Mini implementation:

```text
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/marlin_wna16.py
python/minisgl/models/deepseek_v4.py
python/minisgl/scheduler/prefill.py
python/minisgl/scheduler/cache.py
python/minisgl/kvcache/deepseek_v4_pool.py
benchmark/offline/deepseek_v4_perf_matrix.py
```

Relevant mature reference behavior:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

In particular, note that vLLM reuses `intermediate_cache13` between W13 output
and W2 routed output.  Mini currently allocates `w13_out`, `activated`, and
`route_out` as separate live tensors.  This is a strong follow-up candidate,
but it should be implemented only after this target proves Marlin is the next
release-relevant owner.

## Work Items

### 1. Record The Exact Release State

Record:

```text
git commit
dirty worktree summary
GPU model and count
model/tokenizer max context
page_size
requested/effective max_extend_tokens
effective release-default features
CUDA graph decode buckets
SWA/component/full page capacities
```

Use the system Python environment for mini-sglang.  Run each macro repeat in a
fresh `torchrun` process so CUDA allocator, graph capture, and engine lifecycle
state cannot leak across variants or repeats.

Do not inject feature env vars when testing the true release default.  If
instrumentation env vars are necessary, list every one and show that they do
not change backend selection or allocation policy.

### 2. Repeat-Stable TARGET 12.57 Promotion Gate

Run these true-default shapes:

```text
4096 / 1024 / bs4: at least 3 independent repeats
65536 / 8 / bs1: at least 2 independent repeats
262144 / 1 / bs1: at least 2 independent repeats
```

Use `--use-serving-max-extend-tokens`, not an explicit larger chunk budget.

Record per repeat and summarize median/range:

```text
TTFT
prefill tok/s
decode tok/s where applicable
wall time
peak allocated/reserved memory per rank
effective chunk lengths and chunk count
bounded indexer slice count/backend
decode graph replay/eager count
output token range and text sanity
```

Compare against TARGET 12.56 and TARGET 12.57, but do not treat two unrelated
single runs as a regression.  Classify the promotion result as:

```text
PROMOTE
PROMOTE_WITH_RECORDED_OVERHEAD
BLOCKED_BY_REPEATABLE_REGRESSION
```

A repeatable regression larger than about 5% on `4096/1024/bs4`, or a larger
than about 5% long-context TTFT regression without a corresponding capacity or
memory benefit, requires attribution before promotion.  Smaller variation may
be recorded as neutral/noise when the repeat distributions overlap.

### 3. Default-8192 Long-Context Ladder

After the promotion gate is healthy, run one request at a time:

```text
524288 / 1 / bs1 / serving default 8192
1048576 / 1 / bs1 / serving default 8192
```

The model and tokenizer advertise a 1M-token maximum.  Treat these as release
smoke/capacity probes, not throughput promises.

Run 512k first.  Run 1M only if 512k either passes or fails in a way that still
leaves a meaningful, safe 1M admission probe.  Do not repeatedly rerun an
expensive known OOM without changing the hypothesis.

For each rung record:

```text
admission result and planned cache capacity
number and exact length of prefill chunks
time to first output token
per-chunk or chunk-range latency progression
prefill tok/s
peak allocated/reserved and free memory per rank
KV/component/SWA pages allocated, retained, and evicted
bounded indexer slice count and maximum temporary size
first failing allocation/owner, if any
scheduler progress or spin evidence
output token range and basic text sanity
```

If practical, collect low-overhead owner counters.  Do not run a full Nsight
trace unless owner counters and the failure stack cannot distinguish the next
owner.

### 4. Attribute Scaling, Not Just The Final OOM

Use the 65k, 262k, 512k, and 1M rows to identify scaling behavior:

```text
indexer logits/select time versus context length
number of bounded query slices per layer/chunk
Marlin W13/activation/route_out peak bytes
attention and indexer temporary peaks
cache capacity consumed per committed token
SWA retained-token/page plateau
chunk lifecycle CPU overhead
```

Distinguish:

```text
expected model compute scaling
expected cache-capacity scaling
bounded but expensive workspace
unbounded or accidentally retained temporary
fragmentation/allocation-lifetime failure
correctness/lifecycle failure
```

Do not call ordinary 1M compute cost a memory leak.  Conversely, do not accept
a tensor whose live size scales as the full query-by-context product when only
top-k output is required.

### 5. Decide The Next Owner With Explicit Gates

#### Marlin routed-output follow-up

Open a Marlin lifetime/tiling target only if at least one is true:

- Marlin `route_out` blocks the default 8192-token 512k/1M path;
- a repeat-stable chunk sweep proves a larger chunk materially improves TTFT;
- Marlin workspace is a dominant peak even at the release default.

The follow-up should try, in order:

1. vLLM-aligned W13/W2 `intermediate_cache13` storage reuse;
2. a reusable preallocated MoE workspace with capacity included in KV planning;
3. bounded token-row MoE execution so route output has a fixed upper bound;
4. a kernel ABI change only if the first three cannot meet the gate.

#### Streaming indexer follow-up

Open a fused/streaming paged-logits+top-k target only if:

- bounded indexer slicing is a material fraction of TTFT;
- its launch count or time grows enough to dominate 512k/1M prefill; or
- the 512 MiB workspace itself blocks capacity.

The next kernel must preserve top-k set/tie semantics and compare against the
current bounded Triton+CUDA implementation as the baseline/oracle.

#### Cache/lifecycle follow-up

Open a cache target if admission, page ownership, SWA eviction, prefix state,
or component/full capacity fails before a compute owner.  Keep fixes aligned
with the existing SWA independent lifecycle contract.

### 6. Correctness And Validation

At minimum run the focused suite from TARGET 12.57:

```bash
PYTHONPATH=python python -m pytest -q --no-cov \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/engine/test_dsv4_release_defaults.py \
  tests/core/test_chunked_prefill_lifecycle.py
```

Also confirm:

- bounded and unbounded shorter indexer oracles retain top-k set parity;
- chunked versus unchunked token IDs still match on a feasible shorter oracle;
- SWA independent remains enabled and its retained tail does not grow with the
  full prompt;
- no decode eager fallback appears in the 4096/1024 and 65536/8 rows;
- no malformed or out-of-vocabulary output is produced.

Do not weaken correctness checks to make a long-context rung pass.

## Termination Conditions

Stop when one of these is true:

1. TARGET 12.57 is repeat-stable and both 512k and 1M pass under the true
   release default; document the supported envelope and next performance owner.
2. A 512k/1M rung fails and the first owner is reproduced and classified well
   enough to write one focused implementation target.
3. A repeatable short/262k regression blocks 12.57 promotion and its owner is
   narrowed to one subsystem.
4. The remaining work requires a Marlin ABI change, streaming indexer kernel,
   or cache-contract redesign; write the follow-up target instead of beginning
   that large implementation here.

Do not optimize multiple owners in this target.  Do not spend time polishing
manual 32768-token chunks unless evidence shows they matter to the release
default or materially improve TTFT.

## Output

Write:

```text
performance_milestones/target12_post_indexer_long_context_envelope/README.md
```

Required report sections:

- decision and next owner;
- commit, dirty state, hardware, and exact release settings;
- repeat-stable promotion table with median/range;
- 65k/262k/512k/1M long-context ladder;
- per-rank memory and cache-capacity ledger;
- bounded-indexer slice/time scaling;
- Marlin/indexer/cache next-owner gate;
- correctness and focused-test results;
- one explicit recommendation:

```text
promote 12.57 and close long-context envelope
open Marlin routed-output lifetime target
open streaming indexer/top-k target
open cache/lifecycle target
block promotion on a measured regression
```
