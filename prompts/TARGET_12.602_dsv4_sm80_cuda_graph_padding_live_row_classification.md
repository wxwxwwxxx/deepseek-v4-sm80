# TARGET 12.602: DSV4 SM80 CUDA Graph Padding Live-Row Classification

## Status

Current after TARGET 12.60.

TARGET 12.60 found that upward-padded CUDA graphs can change live-request token
IDs relative to exact-M graphs. The existing synthetic workload proves a
batch-shape-dependent difference, but it does not distinguish harmless BF16
numerical non-invariance from dummy-row/state contamination, and it does not
contain a meaningful natural-language text sanity gate.

## Purpose

Classify the padding behavior before generated graph buckets are promoted:

```text
ACCEPTABLE_BATCH_SHAPE_NUMERICAL_NON_INVARIANCE
BLOCKING_DUMMY_ROW_OR_STATE_CONTAMINATION
BLOCKING_LARGE_LIVE_ROW_NUMERICAL_DRIFT
INCONCLUSIVE_NEEDS_NARROW_OWNER_PROBE
```

The first classification may be accepted for the first release. The two
blocking classifications must be fixed before upward padding becomes default.

## References

```text
performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md
performance_milestones/target12_cuda_graph_bucket_policy_preflight/raw/
python/minisgl/engine/graph.py
python/minisgl/engine/engine.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner_utils/buffers.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py
```

Inspect SGLang/vLLM padding initialization and live-row masking first. Adapt an
existing safety convention where possible.

## Required Work

### 1. Reproduce The Candidate Policy, Not A Power-Of-Two Substitute

Use candidate generated buckets through 64:

```text
[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
```

At minimum compare exact and padded execution for:

```text
17 -> 24
33 -> 40
57 -> 64
```

Keep `65 -> 72` or a larger boundary only as a non-release diagnostic. Do not
describe M=65 as covered by a max64 policy.

### 2. Dummy-Row Poison Invariance

For the same live rows and the same padded graph shape, run at least two valid
dummy-row initializations. Change dummy token IDs, positions, hidden/input
values, metadata fillers, and isolated dummy cache contents where safe. Keep
all addresses and indices valid; do not manufacture undefined behavior.

Compare only live rows. Determine whether changing dummy content changes:

- per-layer or selected-boundary hidden states;
- MoE routing counts, offsets, expert assignments, or workspace ownership;
- attention outputs and cache writes;
- final logits, top-k IDs, or sampled tokens;
- live cache/component state after the step.

If dummy poison changes live results beyond the agreed tolerance, locate the
first producer boundary. Prefer a no-weight/one-layer or captured-subgraph
harness before a full-model binary search.

### 3. Shape-Only Numerical Drift

When dummy poison is inert, compare exact-M and padded-M live rows with the
same inputs. Record:

```text
max_abs / mean_abs / RMS logit error
hidden-state error at selected boundaries
top1-top2 margin at every flipped token
whether the flip is explained by error relative to that margin
NaN/Inf counts
repeat-to-repeat stability for the same shape
```

Do not require bit exactness merely because M changes. Conversely, do not call
a large or stateful difference BF16 noise without margin and boundary evidence.

### 4. Natural-Language Text Sanity

Run a small, diverse set of real prompts through exact, padded, and eager
paths. Include English, Chinese, code, arithmetic, and short instruction
following. Check:

- valid decoding with no replacement-character/garbled-token pathology;
- no abnormal repetition or immediate degeneration attributable to padding;
- finite logits and legal token IDs;
- comparable answer intent for the short smoke;
- no cache ownership or lifecycle errors.

This is a sanity gate, not a large quality benchmark.

### 5. Decision Contract

Padding may be accepted as documented numerical non-invariance only if:

1. dummy poison is inert for live rows and persistent state;
2. no NaN/Inf, invalid location, routing pollution, or cache corruption occurs;
3. flips are concentrated at small top-token margins and logit drift is
   compatible with BF16/shape-dependent backend selection;
4. repeated execution at the same shape is stable;
5. natural-language text sanity passes.

If accepted, add a focused regression test and document that mini does not
promise batch-shape token invariance. If any condition fails, identify the
first owner and provide a narrowly scoped repair target. Do not patch multiple
downstream symptoms.

## Non-Goals

- Do not implement graph-memory reservation here.
- Do not promote larger default buckets.
- Do not tune M>512 or change model precision.
- Do not require exact graph and eager to be bit-identical if the numerical
  acceptance contract is satisfied.
- Do not start a broad all-layer census unless poison or boundary evidence
  requires it.

## Stop Conditions

Stop after the classification is supported by poison, margin, state, repeat,
and text evidence. If three focused probes fail to move the first divergent
boundary, write the narrowest remaining owner and stop.

## Output

```text
performance_milestones/target12_cuda_graph_padding_live_row_classification/README.md
```
