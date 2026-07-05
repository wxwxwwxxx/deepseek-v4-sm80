# Code Changes And Tests

## Instrumentation Added

`python/minisgl/utils/dsv4_memory_debug.py`

- Added freed-range registration and JSONL ledgers.
- Added owner allocation ledgers with nearest/overlap freed-range attribution.
- Added optional tensor integrity sampling.
- Added debug envs:
  - `MINISGL_DSV4_MARLIN_WNA16_RELEASE_LEDGER_DEBUG`
  - `MINISGL_DSV4_MARLIN_WNA16_OWNER_LEDGER_INTEGRITY`
  - `MINISGL_DSV4_MARLIN_WNA16_LAYER2_OWNER_PROBE`

`python/minisgl/models/deepseek_v4.py`

- Records raw expert tensor ranges before release.
- Adds deferred release timings:
  - `model_prepare`
  - `after_kv_alloc`
  - `before_warmup_forward`
  - `after_warmup_forward`
  - `after_graph_capture`
  - `after_first_decode`
- Adds hidden-ref poison and freed-block quarantine debug modes.
- Records Marlin cache/quarantine owners.
- Adds layer2 activation owner probes.

`python/minisgl/engine/engine.py`

- Calls delayed release at `after_kv_alloc`, `after_graph_capture`, and
  `after_first_decode`.
- Records owner rows around model prepare, KV allocation, page-table allocation, graph
  runner init, forward logits, sampler args, and sampled tokens.
- Preserves the older `MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_AFTER_GRAPH_CAPTURE`
  attribution alias.

`python/minisgl/engine/graph.py`

- Calls delayed release before/after warmup forward.
- Records graph capture buffer owners.

`python/minisgl/kvcache/deepseek_v4_pool.py`

- Records DSV4 KV/component pool owners, refcount/mapping tensors, compress state pools,
  and indexer state pools.

`python/minisgl/attention/deepseek_v4.py`

- Records attention metadata/component page-table cache owners.
- Adds layer2 indexer select owner and integrity probes.

## Verification Commands

Compile check:

```bash
python -m py_compile \
  python/minisgl/utils/dsv4_memory_debug.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/engine/graph.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py
```

Result: pass.

Focused tests:

```bash
pytest -q \
  tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

Result: `76 passed, 4 warnings in 16.10s`.

The warnings are third-party FlashInfer/CUTLASS deprecation warnings already outside this
change.

## Runtime Verification

Passing full-model TP8 smokes:

- baseline graph
- Marlin prebuild-only graph
- release after KV allocation, eager/no-graph
- release after KV allocation, graph buckets `[1,2,4,8,16]`
- release before warmup forward, graph buckets `[1,2,4,8,16]`
- release after warmup forward, graph buckets `[1,2,4,8,16]`
- release after graph capture, graph buckets `[1,2,4,8,16]`
- release after first decode, graph buckets `[1,2,4,8,16]`
- hidden-ref poison zero
- hidden-ref poison NaN
- freed-block quarantine all bytes zero
- freed-block quarantine 6.375 GiB zero
- freed-block quarantine 3.1875 GiB zero
- freed-block quarantine 3.1875 GiB deterministic

Failing/warning full-model TP8 smoke:

- immediate `model_prepare` release, eager/no-graph, with owner ledger:
  token id `0` flood and text sanity warning.

## Default Behavior

The default release preset still releases at `model_prepare`.  The diagnostic
`after_kv_alloc` mode is available through:

```bash
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_kv_alloc
```

This milestone does not promote that mode as the default preset.  It is a narrowed safe
boundary and attribution aid.
