# TARGET 12.52: DSV4 SM80 SWA Independent Release Default Cleanup

## Background

TARGET 12.51 fixed the main blocker that kept SWA independent lifecycle out of
the DeepSeek V4 A100/sm80 release bundle:

```text
performance_milestones/target12_swa_independent_ingraph_metadata_promotion/README.md
```

The old TARGET 12.50 blocker was:

```text
prep_metadata_in_graph_requested=true
prep_metadata_in_graph=false
prep_metadata_in_graph_unsupported_reason="swa_independent_lifecycle_not_supported"
```

TARGET 12.51 removed that fail-open by extending the in-graph metadata prep
kernel/API to consume the SWA independent full-page to SWA-page mapping.

The SWA candidate now passed:

- unit/oracle gates: `134 passed`;
- focused CUDA oracle: `2 passed`;
- SWA graph-version guard tests: `3 passed`;
- TP8 text smoke with no garbled output;
- CUDA graph replay with zero eager decode fallback;
- four-scenario paired macro gate.

The paired macro result was positive versus same-run release default:

```text
historical_4096_128_bs4:      +9.52% output tok/s
historical_4096_1024_bs4:     +1.43%
serving_mixed_112req_wave16:  +1.15%
prefix_multi_112req_wave16:   +4.79%
```

The capacity win is still the main reason to default this path:

```text
Tier A default:       2763 pages / 707,328 tokens
SWA independent path: 6457 pages / 1,652,992 tokens
per-page KV bytes:    19,313,920 B -> 8,041,728 B
```

However, TARGET 12.51 only proved the opt-in candidate.  The Engine release
defaults still need to be updated so the true no-env `dsv4_sm80_release_default`
path uses SWA independent lifecycle.

## Goal

Promote the TARGET 12.51 SWA independent path into the DeepSeek V4 A100/sm80
release default bundle.

After this target, a user should be able to run:

```python
from minisgl import LLM

llm = LLM("/models/DeepSeek-V4-Flash")
```

or the benchmark variant:

```text
dsv4_sm80_release_default
```

and get the optimized release path without manually setting SWA lifecycle
environment variables.

## Required Default Bundle

Update the release default environment in:

```text
python/minisgl/engine/engine.py
```

The default bundle should include the existing Tier A settings:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
```

and add the SWA independent/direct metadata settings:

```text
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
```

Do not set a fixed `MINISGL_DSV4_SWA_INDEPENDENT_NUM_PAGES` by default unless a
fresh capacity test proves the automatic planner is wrong.  The intended
release behavior is automatic KV/SWA capacity planning.

## Required Cleanup

1. Update release-default tests so they assert the SWA defaults are injected.
2. Update benchmark/text-smoke release-default expectations so active DSV4
   toggles show SWA independent and SWA direct replay metadata.
3. Update any user-facing release-default summary string that currently lists
   the old Tier A bundle without SWA.
4. Preserve fallback/oracle behavior:

```bash
MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1
```

5. Keep non-SWA explicit benchmark variants available as comparison/oracle
   paths; do not delete them in this cleanup target.

## Required Validation

### Static And Unit

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

### True No-Env Text Smoke

Run `dsv4_sm80_release_default` in a fresh process with no manual DSV4 release
env overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_52_release_default_swa_text_smoke.json
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
planned capacity: roughly the TARGET 12.51 SWA scale, about 1.6M tokens on TP8 A100
```

### Macro Sanity

Run the true release default as a fresh process:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 serving_mixed_112req_wave16 prefix_multi_112req_wave16 \
  --num-pages 0 \
  --keep-going \
  --output-dir /tmp/dsv4_target12_52_release_default_swa_macro
```

Compare against TARGET 12.51 SWA candidate numbers:

```text
historical_4096_128_bs4:      53.210 output tok/s
historical_4096_1024_bs4:     142.169 output tok/s
serving_mixed_112req_wave16:  172.066 output tok/s
prefix_multi_112req_wave16:   112.713 output tok/s
```

Use normal run-to-run noise tolerance.  This target should fail only for a clear
correctness issue, graph replay/eager regression, lost SWA capacity, lost
`prep_metadata_in_graph`, or a material repeatable performance regression.

## Output

Write the report to:

```text
performance_milestones/target12_swa_independent_release_default_cleanup/README.md
```

The report must include:

- exact default env changes;
- unit/static test results;
- true no-env release-default text sanity result;
- active toggles observed from the runtime report;
- graph capture and replay/eager counts;
- `prep_metadata_in_graph` requested/actual/unsupported reason;
- capacity ledger in pages/tokens/bytes;
- four-scenario macro result;
- decision: SWA independent default promoted, blocked, or promoted with a
  clearly named caveat.

## Stop Conditions

Stop and report if:

1. The true release-default path does not inject the SWA defaults.
2. SWA default injection breaks fallback/oracle behavior.
3. Text sanity fails or produces garbled output.
4. CUDA graph replay falls back to eager decode.
5. `prep_metadata_in_graph` fail-opens again.
6. The automatic capacity plan does not reflect the SWA independent capacity
   win.
7. Macro performance has a material repeatable regression versus the TARGET
   12.51 SWA candidate.

Do not expand CUDA graph bucket policy, low precision, MTP, or long-context
soak inside this cleanup target.  If this target passes, the next target should
rerun TARGET 12.49 using the new true release default.
