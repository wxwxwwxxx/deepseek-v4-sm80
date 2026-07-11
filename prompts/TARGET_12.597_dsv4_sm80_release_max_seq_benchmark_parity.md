# TARGET 12.597: DSV4 SM80 Release Max-Sequence And Benchmark Parity

## Status

Run after TARGET 12.595 and before TARGET 12.60. This is the current immediate
target.

TARGET 12.595 promoted the C128 eager-prefill one-surface path and completed a
TP8 `1048576 / 8 / bs1` run with all 128 prefill chunks plus seven decode CUDA
graph replays. The C128 result is valid: the old raw/page/full and full-matrix
int64 allocation family is gone.

However, the benchmark did not use the model-configured maximum sequence
length. `Scenario.max_seq_len` returns `prompt_len + decode_len`, and
`_init_llm` currently passes that value unconditionally as
`max_seq_len_override`:

```text
prompt_len:                 1048576
decode_len:                       8
implicit max_seq_len_override: 1048584
model max_position_embeddings: 1048576
```

For short scenarios, the same behavior shrinks the engine maximum to only the
scenario length. That changes RoPE cache size, request page-table width, DSV4
capture metadata, CUDA graph memory, automatic KV capacity, and potentially
the backend envelope. A benchmark may use this diagnostic mode, but it must
not call it the true serving/release default.

TARGET 12.60 will study large CUDA graph buckets. Its memory evidence is not
trustworthy until max-sequence configuration is aligned with ordinary:

```python
LLM("/models/DeepSeek-V4-Flash")
```

## Purpose

Define and enforce the release max-sequence contract, make benchmark engine
configuration explicit and observable, then validate the C128 promotion with
a legal total-sequence 1M workload under the real model-configured limit.

The target must answer:

1. Is `max_seq_len` the maximum prompt or maximum total sequence?
2. When does the benchmark use model default, explicit override, or
   scenario-sized diagnostic mode?
3. What are the model-configured, requested, and effective engine limits?
4. Does the scheduler reject or clamp requests before any RoPE/cache position
   can exceed the effective limit?
5. Does the release engine still complete a legal 1M total-sequence workload?
6. How much do real model-max RoPE/page-table/graph surfaces change the short
   release baseline used by TARGET 12.60?

## Contract

For the release/serving path:

```text
max_seq_len = maximum total sequence length
              = prompt tokens + generated tokens retained by the request

valid model positions satisfy:
  0 <= position < effective_engine_max_seq_len

request admission satisfies:
  input_len < effective_engine_max_seq_len
  requested_output <= effective_engine_max_seq_len - input_len
```

An input whose length equals the maximum cannot request another token. The
scheduler may reject it with a clear result/error; it must not silently run
decode beyond the RoPE cache.

An explicit research override larger than the model-configured limit may remain
available only when the user requests it deliberately. It must be recorded as
an override/non-default mode and must not be used for release claims.

## Non-Goals

- Do not change the TARGET 12.595 C128 one-surface algorithm.
- Do not optimize FP8 indexer, C128 attention, Marlin, communication, or MoE.
- Do not expand CUDA graph buckets in this target.
- Do not add prefill CUDA graph capture.
- Do not change page size 256, chunk size 8192, SWA independent lifecycle, or
  Route-B component ownership.
- Do not re-enable MTP or change precision.
- Do not rerun the invalid `1048576 + 8` release claim after the contract is
  established.

## References To Read First

Evidence:

```text
performance_milestones/target12_c128_one_surface_1m_promotion/README.md
performance_milestones/target12_c128_prefill_metadata_contract_native_micro/README.md
prompts/TARGET_12.595_dsv4_sm80_c128_one_surface_1m_promotion.md
prompts/TARGET_12.60_dsv4_sm80_cuda_graph_bucket_policy_preflight.md
```

Mini source:

```text
benchmark/offline/deepseek_v4_perf_matrix.py
python/minisgl/engine/config.py
python/minisgl/engine/engine.py
python/minisgl/layers/rotary.py
python/minisgl/scheduler/scheduler.py
python/minisgl/scheduler/prefill.py
python/minisgl/llm/llm.py
python/minisgl/server/args.py
```

Relevant tests:

```text
tests/benchmark/test_deepseek_v4_perf_matrix.py
tests/engine/test_dsv4_release_defaults.py
tests/core/test_chunked_prefill_lifecycle.py
tests/benchmark/test_deepseek_v4_text_smoke.py
```

## Work Items

### 1. Audit Max-Sequence Ownership End To End

Trace and report:

```text
HF max_position_embeddings
ModelConfig.rotary_config.max_position
EngineConfig.max_seq_len_override
EngineConfig.max_seq_len
Engine.max_seq_len after KV-capacity clamp
RoPE cache length
context page-table width
attention capture max_seq_len
scheduler admission/clamp
benchmark scenario max
server/LLM public arguments
```

For each value identify:

```text
source
unit/meaning
whether it is user-visible
whether it may exceed model config
whether it controls allocation, admission, or both
```

Do not assume `args.max_seq_len is None` means model default. Record the value
actually passed to `LLM` and the effective value observed after construction.

### 2. Separate Benchmark Modes Explicitly

The desired behavior is:

```text
model_default / serving_default:
  do not pass max_seq_len_override
  EngineConfig derives the limit from model config and actual capacity

explicit_override:
  pass only the user-provided --max-seq-len
  record that this is not model default when values differ

scenario_sized_diagnostic:
  retain prompt+decode-sized engines only behind an explicit CLI option
  label reports as diagnostic/non-serving
```

Choose clear CLI names consistent with existing
`--use-serving-max-extend-tokens`. Preserve useful historical behavior through
an explicit flag rather than an invisible default.

Requirements:

- explicit override and scenario-sized mode are mutually exclusive;
- the true release-default command omits the override entirely;
- command/config summaries expose the selected mode;
- old reports remain interpretable but are not silently mixed with the new
  model-default baseline;
- benchmark tests cover argument parsing and `_init_llm` kwargs directly;
- do not load model weights merely to test mode selection.

### 3. Record Requested And Effective Limits

Every rank report, aggregate report, and `run_config.json` must record at
minimum:

```text
model_config_max_seq_len
max_seq_len_mode
requested_max_seq_len_override
scenario_required_total_seq_len
effective_engine_max_seq_len
effective_rope_cache_len, if available without expensive instrumentation
prompt_len
requested_decode_len
admitted_decode_len
```

Also record whether:

```text
scenario_required_total_seq_len <= effective_engine_max_seq_len
all observed positions are within the effective limit
```

If the benchmark cannot satisfy a scenario, fail before the expensive model
run with an owner-specific configuration error. Do not hang waiting for a
request that the scheduler dropped.

### 4. Harden Scheduler And RoPE Boundary Tests

Preserve the existing scheduler policy where appropriate, but make its
contract testable and observable:

```text
input + output < max: accepted unchanged
input + output == max: accepted unchanged
input + requested output > max: output is explicitly clamped or rejected
input == max and output > 0: rejected with a visible completion/error
input > max: rejected
```

Add no-weight or mocked tests proving no scheduled model position can reach or
exceed the RoPE cache length. Test small synthetic limits such as 8/16/32; do
not use the full model for boundary unit tests.

If the current scheduler warning-only drop can leave offline callers waiting,
repair the result/error propagation narrowly. Do not redesign the scheduler.

### 5. Quantify Scenario-Sized Versus Model-Default Memory

Before TARGET 12.60, run a paired short TP8 probe in independent processes:

```text
same DSV4 release workload and graph buckets
A: explicit scenario-sized diagnostic engine
B: model-default max-sequence engine with no override
```

A representative workload may be `4096/128/bs4` or another short gate that
finishes quickly. Record:

```text
RoPE cache bytes/length
context page-table shape/bytes
DSV4 capture metadata shapes/bytes
CUDA graph private-pool/capture delta
planned KV pages/tokens/bytes
peak allocated/reserved/free
startup/capture time
graph replay/eager count
output/decode throughput
```

This comparison is diagnostic. The model-default row becomes the baseline for
TARGET 12.60. Do not tune graph buckets here.

### 6. Run A Legal 1M Total-Sequence Gate

After unit and short TP8 gates pass, run exactly one expensive legal gate:

```text
model configured max: 1048576
prompt_len:           1048568
decode_len:                 8
total:                1048576
batch_size:                 1
page_size:                 256
max_extend_tokens:        8192 serving default
```

Requirements:

- use model-default max-sequence mode;
- do not pass an explicit override;
- record effective engine and RoPE limits;
- complete all chunked prefill segments;
- produce eight output tokens or an explicitly documented EOS completion;
- enter bs1 decode CUDA graph and replay without eager fallback;
- preserve one-surface C128 backend, placeholders, and memory behavior;
- record maximum observed model position and prove it is below the effective
  limit;
- preserve component/SWA/prefix ownership and zero unexpected eviction.

Do not rerun the full 1M gate repeatedly. If it fails, capture the first owner
and stop unless one small benchmark-contract fix clearly resolves it.

### 7. Promotion And Documentation

If the gate passes:

- keep C128 one-surface promoted;
- make model-default mode the named release benchmark baseline;
- label scenario-sized mode diagnostic;
- update TARGET 12.60 references so graph preflight uses model-default mode;
- document that the supported 1M claim refers to total sequence length;
- preserve explicit override for deliberate research use, with clear status.

Do not claim that the 39-minute 1M TTFT is practical serving performance. This
target establishes configuration and functional correctness only.

## Validation

At minimum run:

```text
py_compile / ruff / git diff --check
benchmark CLI/config unit tests
scheduler max-length boundary tests
release-default config tests
chunked-prefill lifecycle tests
C128 one-surface focused tests
text-smoke tests
paired short TP8 memory/graph probe
one legal 1M-total TP8 promotion gate
```

Run every TP8 variant/repeat in a fresh `torchrun` process. Preserve the dirty
worktree and do not revert unrelated target changes.

## Termination Conditions

Stop when one of these is true:

1. Benchmark modes are explicit, scheduler/RoPE bounds are proven, the paired
   short probe establishes the real model-default graph-memory baseline, and
   legal 1M-total completes.
2. Model-default short initialization/capture exposes a new memory owner that
   must be fixed before graph bucket work; reproduce and classify it once.
3. Legal 1M-total fails at a new first-order owner; record the exact boundary
   and open one focused target.
4. Correct release semantics require a scheduler/API redesign larger than this
   target; document the contract and defer implementation.

Do not expand graph buckets or optimize long-context kernels after the parity
decision is complete.

## Output

Write:

```text
performance_milestones/target12_release_max_seq_benchmark_parity/README.md
```

Required report sections:

- decision and TARGET 12.60 readiness;
- source ownership/meaning table;
- benchmark mode contract and CLI changes;
- scheduler/RoPE boundary behavior;
- requested/model/effective max-sequence ledger;
- scenario-sized versus model-default short TP8 memory/graph table;
- legal 1M-total promotion result;
- C128/component/SWA/decode-graph regression status;
- exact next recommendation.

