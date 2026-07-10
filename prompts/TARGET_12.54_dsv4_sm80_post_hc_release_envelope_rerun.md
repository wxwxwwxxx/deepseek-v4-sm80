# TARGET 12.54: DSV4 SM80 Post-HC Release Envelope Rerun

## Background

TARGET 12.53 fixed and promoted the HC prenorm temporary-elimination path:

```text
performance_milestones/target12_hc_prenorm_temp_elimination/README.md
```

The promoted release-default pair is:

```text
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_LINEAR_BF16_FP32=1
```

It fixed the two TARGET 12.49 OOM shapes:

```text
prompt_len=32768, decode_len=16, batch=1
prompt_len=128,   decode_len=64, batch=256
```

and improved the four historical macro scenarios by about 5-9% in the 12.53
repeat gate.  The next step is not to continue micro-optimizing HC.  The next
step is to rerun the release serving envelope after the new HC path is part of
the true no-env default.

## Goal

Establish the post-HC release envelope for:

1. long-context single-request prefill;
2. large active decode batch;
3. CUDA graph bucket probes after the HC temporary is gone;
4. the next exposed owner or memory-planning gap.

This target should answer:

```text
After HC_GRAPH_CLEANUP + LINEAR_BF16_FP32 are defaulted, what is the next
release blocker before 512k/1M context or larger default graph buckets?
```

Do not implement chunked prefill, global workspace management, new low-precision
paths, or broad fallback cleanup in this target.  This is a measurement and
promotion-confirmation target.

## Required Setup

Use the current branch and system Python.  Use fresh `torchrun` processes for
each meaningful graph/env configuration.

Default route:

```text
--variants dsv4_sm80_release_default
--num-pages 0
```

Do not manually set DSV4 release env for normal release-default runs.  The
target must verify true no-env behavior.

Fallback/oracle remains:

```bash
MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1
```

Only use fallback/oracle for named A/B comparisons.

## Phase 0: Static And Unit

Run:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/kernel/test_deepseek_v4_wrappers.py
```

If the full wrapper test is too slow, run the HC, release-default, and benchmark
subsets first and record skipped coverage.

## Phase 1: True No-Env Text Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_54_release_default_text_smoke.json
```

Required signals:

```text
text sanity: pass, no garble
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_LINEAR_BF16_FP32=1
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
prep_metadata_in_graph=true
decode eager fallback: 0 for captured buckets
planned capacity: SWA-scale, about 1.6M tokens on TP8 A100
```

Record graph private-pool delta, capture free before/after, planned pages and
tokens, active toggles, and generated text.

## Phase 2: Long-Context Key Rerun

Rerun the previous failing shape and then step upward:

```text
prompt/decode/batch:
32768  / 16 / 1
65536  / 8  / 1
131072 / 4  / 1
262144 / 2  / 1   optional stretch only if 131072 is clean
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
  --batch-size 1 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_54_long_<PROMPT> \
  --keep-going
```

Record:

- pass/fail and exact failure owner;
- prefill tok/s, TTFT, elapsed;
- planned tokens/pages and used tokens;
- graph capture and replay/eager counts;
- peak allocated/reserved memory;
- remaining free memory after graph capture if available;
- whether failure is token-capacity, activation/workspace, graph pool, or a
  kernel shape issue.

## Phase 3: Large-Batch Key Rerun

Rerun the previous failing large-batch shape:

```text
prompt_len=128
decode_len=64
batch sizes: 128, 256
```

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
  --output-dir /tmp/dsv4_target12_54_large_bs_<BATCH> \
  --keep-going
```

Batch 256 may still run eager decode under the default bucket list.  That is
acceptable.  Do not expand default graph buckets just to avoid eager in this
phase.  Record whether the prefill OOM is gone and whether a new owner appears.

## Phase 4: Graph Bucket Spot Check

Run only the useful bucket probes from TARGET 12.49:

```text
--cuda-graph-bs 1 2 4 8 16 32
--cuda-graph-bs 1 2 4 8 16 32 64
--cuda-graph-bs 1 2 4 8 16 32 64 128
```

Use short prefill and target the matching batch:

```text
batch 32 for max32
batch 64 for max64
batch 128 for max128
prompt_len=128
decode_len=64
```

Record graph private-pool delta, capture success, replay/eager, output tok/s,
and post-capture free memory.  Do not run max256 unless batch 256 default
prefill is clean and max128 has comfortable memory headroom.

## Phase 5: Fallback/Backend Snapshot

Use existing kernel counter output from the perf matrix.  Do not build new
instrumentation unless counters are missing.

For the main successful rows, record top remaining wrapper/backend counters:

- `fallback_wrapper_calls`;
- optional kernel `None` skips;
- unsupported kernel skips;
- owner timing if enabled;
- any backend that silently returned to torch fallback.

This is not the full fallback cleanup target.  It is a lightweight snapshot so
the next target can decide whether graph/memory planning or fallback cleanup is
more urgent.

## Output

Write the report to:

```text
performance_milestones/target12_post_hc_release_envelope_rerun/README.md
```

The report must include:

- git commit and dirty-state summary;
- static/unit result;
- true no-env text smoke;
- active release env/toggles, especially HC and SWA toggles;
- long-context key rerun table;
- large-batch key rerun table;
- graph bucket spot-check table;
- capacity ledger in pages/tokens/bytes;
- graph private-pool and post-capture free-memory ledger;
- new blocker or next owner;
- recommendation for next target:
  - graph/activation memory accounting;
  - chunked prefill;
  - CUDA graph max-bs policy;
  - fallback census/native backend cleanup.

## Stop Conditions

Stop and report when:

1. true no-env release default does not inject the new HC toggles;
2. text sanity fails;
3. `32768/16/1` fails again and the owner is identified;
4. `128/64/256` fails again and the owner is identified;
5. 65536 or 131072 exposes a clear new memory owner;
6. graph max128 has unacceptable private-pool/free-memory cost;
7. the rerun provides enough evidence to choose the next route.

Do not spend this target optimizing a newly found kernel.  Identify the owner,
write the report, and recommend the next focused target.
