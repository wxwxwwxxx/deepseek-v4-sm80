# TARGET 12.55: DSV4 SM80 Graph And Activation Memory Accounting

## Background

TARGET 12.54 reran the true no-env post-HC release default and stopped at a
new memory boundary:

```text
performance_milestones/target12_post_hc_release_envelope_rerun/README.md
```

Important facts from that report:

```text
32768 / 16 / 1: pass
65536 / 8  / 1: fail during first prefill forward
failure owner: attention wo_a BF16 BMM projection
failed allocation: 128 MiB
free memory at failure: about 45 MiB
planned KV: about 49.7 GiB
fixed SWA cache at 65536: about 2.72 GiB
graph private-pool delta: about 0.96 GiB
planned token capacity: about 1.6M tokens
```

This is not the old HC 2 GiB temporary.  The request has enough nominal KV token
capacity, but the runtime has almost no remaining activation/workspace margin.
The current auto KV planner uses `memory_ratio=0.9` and Marlin release capacity
credit, but it does not make graph private-pool, activation peak, allocator
slack, or operator workspace first-class reserves.

One DSV4 SWA-independent KV page is about:

```text
8,041,728 bytes/page
256 tokens/page
1 GiB reserve ~= 133 pages ~= 34k tokens
memory_ratio 0.90 -> 0.875 frees about 2 GiB on A100 80G, or about 67k tokens
```

Therefore, the first question is not "can chunked prefill fix this?"  Chunked
prefill likely will be needed for 131k/262k/512k/1M.  The first question is:

```text
Does the current release default simply over-allocate KV pages by 1-3 GiB, and
should release KV planning reserve graph/activation/workspace headroom by
default?
```

## Goal

Build a concrete graph/activation memory ledger and, if the evidence is clear,
implement a release-default KV planning reserve that lets 64k single-request
prefill pass without sacrificing too much serving capacity.

This target should answer:

1. How much memory headroom does `65536/8/1` need after graph capture?
2. Does reducing KV pages via `--memory-ratio` make it pass?
3. If yes, what explicit reserve should the DSV4 A100/sm80 release default use?
4. Does that reserve preserve text sanity, 32768, large-batch, and normal macro
   performance?
5. Should the next target be chunked prefill, larger CUDA graph buckets, or
   fallback/native backend cleanup?

Do not implement chunked prefill in this target.  Do not expand default CUDA
graph buckets in this target.  Do not change low-precision or MTP behavior.

## References To Read First

Mini:

```text
performance_milestones/target12_post_hc_release_envelope_rerun/README.md
performance_milestones/target12_hc_prenorm_temp_elimination/README.md
performance_milestones/target12_release_long_context_large_batch_soak/README.md
python/minisgl/engine/engine.py
python/minisgl/engine/config.py
python/minisgl/engine/graph.py
python/minisgl/models/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

Reference frameworks:

```text
/workspace/sglang-main/python/sglang/srt/managers/scheduler.py
/workspace/sglang-main/python/sglang/srt/mem_cache/
/workspace/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_worker.py
/workspace/vllm-dsv4-docker/vllm/v1/executor/abstract.py
/workspace/vllm-dsv4-docker/vllm/engine/arg_utils.py
/workspace/vllm-dsv4-docker/vllm/config/cache.py
```

Use SGLang/vLLM source only to understand memory-budget structure and naming.
Do not require vLLM runtime for this target.

## Phase 0: Static And Unit Baseline

Run:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/engine/config.py \
  python/minisgl/engine/graph.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

If you touch kernel wrapper behavior, also run the relevant subset of:

```bash
python -m pytest -q tests/kernel/test_deepseek_v4_wrappers.py
```

## Phase 1: No-Code Memory-Ratio Sweep

Use fresh `torchrun` processes.  Normal route:

```text
--variants dsv4_sm80_release_default
--num-pages 0
```

The known failing shape is:

```text
prompt_len=65536
decode_len=8
batch_size=1
```

Run a small ratio sweep.  The `0.90` row may use TARGET 12.54 as historical
evidence, but rerun it if the environment changed.

```text
memory_ratio: 0.90, 0.885, 0.875, 0.85
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
  --memory-ratio <RATIO> \
  --output-dir /tmp/dsv4_target12_55_ratio_<RATIO> \
  --keep-going
```

For each row, record:

- pass/fail and exact failure owner;
- planned pages/tokens;
- planned KV bytes and fixed SWA bytes;
- graph private-pool delta and capture free before/after;
- peak allocated/reserved memory;
- free memory at failure or after success;
- prefill tok/s and TTFT if successful;
- whether PyTorch reports large reserved-unallocated memory.

Interpretation:

- If a ratio near `0.875` passes, the primary issue is KV planning headroom.
- If even `0.85` fails at the same `wo_a` owner, there may be an activation
  algorithm/backend problem or chunked prefill may be mandatory before 64k.
- If failure owner changes, report the new owner and stop broad tuning.

## Phase 2: Explicit Reserve Design

If the ratio sweep proves that releasing 1-3 GiB of KV memory fixes 64k, design
an explicit reserve instead of permanently relying on a lower naked
`memory_ratio`.

Preferred release-default design:

```text
available_for_kv =
  memory_ratio * startup_free
  - model_memory
  + Marlin_release_credit
  - fixed_swa_cache
  - graph_private_pool_reserve
  - prefill_activation_reserve
  - allocator_slack_reserve
```

Keep the first implementation conservative and auditable:

- Add named report fields to `kv_capacity_plan_report`.
- Keep `memory_ratio` visible and unchanged unless the evidence says otherwise.
- Prefer a DSV4 A100/sm80 release-only default reserve, not a global default for
  every model.
- Provide env override knobs for bisecting and rollback.
- Do not reduce capacity when the user explicitly passes `--num-pages`.

Suggested knobs and semantics, unless the codebase already has better local
names:

```text
MINISGL_DSV4_SM80_RELEASE_KV_RESERVE_BYTES
  Explicit bytes subtracted from auto KV planning for the release default.

MINISGL_DSV4_SM80_RELEASE_KV_RESERVE_DISABLE=1
  Disable the release reserve for A/B and rollback.

MINISGL_DSV4_SM80_RELEASE_KV_RESERVE_REPORT_ONLY=1
  Report the computed reserve without applying it, if useful for debugging.
```

Default reserve candidate:

```text
2 GiB, if memory_ratio=0.875 is the first clearly passing sweep row.
3 GiB, if 0.875 is marginal but 0.85 is clean.
```

The report must translate the chosen reserve into page/token cost:

```text
reserve_bytes / 8,041,728 ~= pages lost
pages_lost * 256 ~= tokens lost
```

## Phase 3: Implement Only If Evidence Is Clear

Implement planner changes only after Phase 1 demonstrates that the failure is
headroom-sensitive.

Likely files:

```text
python/minisgl/engine/engine.py
python/minisgl/engine/config.py
tests/engine/test_dsv4_release_defaults.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

Implementation requirements:

- Reserve must be applied before KV allocation and reflected in
  `kv_capacity_plan_report`.
- Reserve must be active only for true DSV4 A100/sm80 release defaults unless
  explicitly requested by env.
- `--num-pages` / `num_page_override` must remain exact and should not silently
  lose pages.
- Existing Marlin release capacity credit must remain visible separately from
  the reserve.
- Add tests for:
  - default reserve injection/reporting;
  - disable override;
  - explicit `--num-pages` behavior;
  - no accidental fallback/oracle path changes.

## Phase 4: Promotion Gate

After implementing the reserve, rerun these rows with true no-env release
default:

### Text Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_55_release_default_text_smoke.json
```

### Key Long-Context Rows

```text
32768 / 16 / 1
65536 / 8  / 1
131072 / 4 / 1 optional diagnostic only if 65536 is clean
```

Use the same command template as Phase 1, without manual `--memory-ratio` if
the reserve has been defaulted.

### Large-Batch Rows

Rerun the shapes skipped by TARGET 12.54:

```text
128 / 64 / 128
128 / 64 / 256
```

### Normal Macro Guard

Run at least:

```text
historical_4096_128_bs4
historical_4096_1024_bs4, if runtime allows
```

Record output tok/s, decode tok/s, TTFT, graph replay/eager counts, planned
capacity, and peak memory.

## Phase 5: Decision

End with one of these decisions:

```text
PROMOTE_RESERVE:
  65536 passes; text/macro/large-batch are clean; capacity loss is acceptable.

RATIO_ONLY_EVIDENCE:
  ratio sweep proves reserve would help, but implementation is deferred with a
  precise recommended reserve.

CHUNKED_PREFILL_REQUIRED:
  headroom reserve does not fix the 64k owner or capacity loss is too high.

BACKEND_FIX_REQUIRED:
  failure points to an avoidable operator temporary/backend issue rather than
  general prefill activation headroom.
```

If `PROMOTE_RESERVE`, update the TARGET 12 route and recommend TARGET 12.56
chunked prefill as the next long-context step.  If `CHUNKED_PREFILL_REQUIRED`,
do not keep tuning the reserve; move directly to TARGET 12.56.

## Output

Write:

```text
performance_milestones/target12_graph_activation_memory_accounting/README.md
```

The report must include:

- git commit and dirty-state summary;
- static/unit results;
- ratio sweep table;
- explicit reserve design, if implemented;
- before/after capacity ledger in pages, tokens, bytes, and GiB;
- graph private-pool ledger;
- long-context and large-batch promotion gate;
- capacity cost versus 12.54 default;
- recommendation for 12.56 chunked prefill and whether larger graph buckets
  should remain deferred.

## Stop Conditions

Stop and report when:

1. ratio sweep proves the required reserve and implementation is safe;
2. ratio sweep disproves reserve as a useful fix;
3. a new non-memory owner appears;
4. text sanity fails;
5. capacity loss exceeds the target's acceptable range without clear 64k gain;
6. the work starts drifting into chunked prefill or native backend cleanup.
