# TARGET 08.34: DSV4 SM80 MoE Marlin WNA16 Cache Lifecycle

## Status

Active TARGET 08 capacity follow-up after TARGET 08.33.

Run this before any broader CUDA graph workspace-manager redesign.  TARGET
08.33 proved the C4 indexer capture-width hypothesis is false, but it also
found a stronger clue: almost all of the `~18.6-19.0 GiB/rank` movement appears
after the warmup `model.forward()` outside the `torch.cuda.graph` context, not
inside the graph capture block itself.

## Goal

Attribute and, if confirmed, fix the large warmup-forward memory jump by
auditing the MoE Marlin WNA16 expert-weight cache lifecycle.

Primary hypothesis:

```text
The first real decode forward lazily repacks and retains Marlin WNA16 routed
expert weights for all 43 layers.  This cache is currently absent from model
prepare / KV capacity planning, so it appears as a huge "graph capture" memory
cost even though it is really lazy MoE backend state.
```

If confirmed, implement the smallest safe lifecycle fix:

- prebuild the Marlin WNA16 MoE cache before CUDA graph capture and before KV
  capacity decisions when the backend is selected;
- report its persistent bytes and equivalent KV-token/page cost;
- decide whether original FP4 expert weights can be released after repack under
  an opt-in or promoted backend contract;
- keep correctness and graph replay unchanged.

## Why This Target Exists

TARGET 08.32 ruled out generic CUDA graph overhead and many synthetic
subgraphs.  TARGET 08.33 ruled out the specific indexer-width bug:

- real C4 indexer table is page-based;
- indexer static width is `128 * 64 = 8192`, which is expected;
- repeated dense indexer logits explain only about `0.010 GiB/rank`.

But TARGET 08.33's stage ledger showed:

```text
after warmup model.forward():        about +17.8 GiB allocated/rank
after actual torch.cuda.graph block: about +0.0 GiB allocated/rank
```

That means the large cost is probably a live tensor/cache produced by the first
real forward, not pure graph-private allocator state.

The current A100 victory bundle selects the MoE backend `marlin_wna16` by
default.  The code path stores a per-layer cache in `self._marlin_wna16_weights`
and creates it lazily on first routed-expert forward.  Rough memory math is
already suspicious:

```text
per TP rank, per layer:
  w13 raw packed FP4 bytes ~= 0.250 GiB
  w2 raw packed FP4 bytes  ~= 0.125 GiB
  scales                  ~= 0.023 GiB
  total raw+scale class   ~= 0.398 GiB

0.398 GiB/layer * 43 layers ~= 17.1 GiB/rank
```

This is close enough to the measured `~17.8 GiB/rank` warmup jump that it must
be audited before investigating lower-probability graph-pool mechanisms.

## Non-Goals

- Do not continue indexer-width experiments in this target.
- Do not redesign all CUDA graph memory management.
- Do not implement INT8 MoE, FP8 KV cache, or other low-precision research.
- Do not change MoE numerical behavior unless the change is strictly lifecycle
  / cache placement.
- Do not release original FP4 expert weights by default until correctness,
  fallback behavior, and rollback semantics are clear.
- Do not spend time on sub-`1 GiB/rank` memory effects.

## Source References

Mini source:

- `python/minisgl/models/deepseek_v4.py`
  - `DSV4FusedRoutedExperts.__init__`
  - `DSV4FusedRoutedExperts.forward`
  - `DeepseekV4Model.prepare_for_cuda_graph_capture`
- `python/minisgl/kernel/deepseek_v4.py`
  - `dsv4_moe_expert_backend`
  - `moe_route_dispatch_bf16_marlin_wna16`
  - `DSV4_SM80_MOE_EXPERT_BACKEND_ENV`
- `python/minisgl/kernel/marlin_wna16.py`
  - `prepare_moe_mxfp4_weights`
  - `run_moe`
- `python/minisgl/engine/engine.py`
  - model prepare and KV capacity ordering
- `python/minisgl/engine/graph.py`
  - warmup forward and graph capture stage ledger
- `benchmark/offline/deepseek_v4_perf_matrix.py`
  - A100 victory bundle expansion and MoE backend variants

Milestone evidence:

- `performance_milestones/target08_indexer_capture_static_width_audit/README.md`
- `performance_milestones/target08_cuda_graph_private_pool_micro_attribution/README.md`
- `performance_milestones/target07_post_victory_reprofile/summaries/cache_workspace_memory_ledger.md`

Reference frameworks, if lifecycle comparison helps:

- `/workspace/sglang-main`
- `/workspace/vllm-dsv4-docker`
- `/workspace/venvs/vllm-dsv4`

## Required Approach

### 1. Confirm The Backend And Memory Math

Start by recording the exact runtime backend used by the active preset:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND
```

For the loaded model config and TP size, compute per-rank theoretical bytes:

- raw packed `w13_weight`;
- raw packed `w2_weight`;
- `w13_weight_scale_inv`;
- `w2_weight_scale_inv`;
- repacked Marlin `w13`;
- repacked Marlin `w2`;
- repacked Marlin scales;
- total per layer and total across 43 layers.

The final report must compare theory with measured memory deltas.

### 2. Add Focused Opt-In Instrumentation

Add diagnostics behind explicit environment flags, for example:

```text
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1
```

The Marlin cache debug should record per layer:

- layer id;
- whether `_marlin_wna16_weights` was already present;
- before/after free memory;
- before/after PyTorch allocated/reserved memory;
- source tensor shapes/dtypes/bytes;
- repacked tensor shapes/dtypes/bytes;
- scale tensor shapes/dtypes/bytes;
- elapsed time for `prepare_moe_mxfp4_weights`;
- whether the cache matches the source signature.

The warmup-forward debug should record owner-scoped memory points around real
full-model warmup:

- model embed / HC expand;
- each decoder layer before and after;
- inside each layer: attention, MoE, and layer output;
- inside MoE: gate, route plan, routed experts, shared experts, reduce;
- lm head.

Prefer low-volume JSONL under this target's milestone directory.  Do not enable
activation dumps or full tensor saves.

### 3. Run Minimal Full-Model A/B

Use one single-bucket graph-capture run first, not a broad matrix.

Required comparisons:

1. current victory backend, expected `marlin_wna16`;
2. forced grouped FP4 backend:

```text
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=grouped_fp4
```

Use the same `--page-size 256`, `--num-pages 128`, TP8, and
`--cuda-graph-bs 16` shape used by TARGET 08.33 unless there is a concrete
reason to choose a smaller bucket.

Decision rule:

```text
If the warmup-forward memory jump disappears or drops by >=10 GiB/rank when
Marlin WNA16 is disabled, the hypothesis is confirmed.
```

If grouped FP4 is too slow, use capture-only or very short decode.  This target
is about memory attribution, not final throughput.

### 4. Prebuild Cache PoC If Confirmed

If Marlin WNA16 lazy repack is confirmed, implement an opt-in PoC that prebuilds
the cache before graph capture.  Prefer wiring it through:

```text
DeepseekV4Model.prepare_for_cuda_graph_capture()
```

or an equivalent model-prepare phase that runs before KV capacity planning in
`Engine.__init__`.

The PoC should:

- build all per-layer Marlin WNA16 caches exactly once;
- report persistent bytes in `model_prepare_report`;
- make the later warmup `model.forward()` show no large lazy-cache jump;
- preserve graph replay;
- preserve text smoke.

Important capacity check:

```text
If the cache is prebuilt before _determine_num_pages(), automatic KV page
planning should see the real remaining memory.  This may reduce auto num_pages
but makes capacity accounting honest.
```

### 5. Evaluate Original FP4 Weight Release

After cache prebuild is working, evaluate whether original routed-expert FP4
weights/scales can be released when `marlin_wna16` is the selected backend.

This must be opt-in unless the backend contract is made fail-closed:

```text
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
```

Required checks:

- no fallback path silently needs the released raw weights;
- checkpoint reload is not required after release;
- `cache.matches(...)` does not require live source tensors, or the signature is
  captured before release;
- error messages are clear if a user switches backend after release;
- text smoke still passes;
- graph capture and replay still pass;
- memory saved is reported separately from prebuild movement.

Do not promote release-by-default unless correctness and rollback semantics are
obvious.

### 6. Compare Lifecycle With SGLang/vLLM

Do a source-level comparison before final recommendation:

- does SGLang or vLLM prepack/repack MoE weights during model load/prepare?
- do they release original packed weights after backend-specific packing?
- how do they account for packed expert memory in KV capacity planning?
- is there an existing mature design we should mirror?

If exact runtime comparison is expensive, source-derived lifecycle comparison is
acceptable, but label it as static inference.

## Required Analysis

The final README must include:

- recap of TARGET 08.32/08.33 and why the investigation moved to warmup
  forward;
- backend confirmation table;
- theoretical Marlin WNA16 cache memory ledger;
- measured per-layer Marlin cache memory table;
- warmup-forward owner memory table;
- current versus grouped-FP4 memory A/B;
- prebuild PoC result, if attempted;
- original-weight release feasibility and measured savings, if attempted;
- effect on KV capacity:
  - GiB/rank;
  - equivalent DSV4 KV pages;
  - equivalent KV tokens at page size `256`;
- correctness and graph replay gates;
- comparison with SGLang/vLLM lifecycle;
- final recommendation:
  - promote prebuild;
  - keep release as opt-in;
  - open a separate release/promotion target;
  - or return to real-module warmup owner attribution if hypothesis is false.

## Gates

Pass this target if it produces one of:

1. proof that Marlin WNA16 lazy repack explains at least `10 GiB/rank` of the
   warmup-forward memory jump, plus a prebuild/cache-accounting PoC;
2. proof that Marlin WNA16 explains the memory jump but original-weight release
   needs a later dedicated safety target;
3. proof that Marlin WNA16 is not the owner, plus a ranked warmup-forward owner
   table that identifies the next concrete `>=2 GiB/rank` suspect.

Stop early if:

- the current preset is not using `marlin_wna16`;
- grouped-FP4 A/B does not materially reduce warmup-forward memory and the
  per-layer Marlin cache logs stay below `1 GiB/rank` total;
- instrumentation itself saves full activations or causes OOM;
- a proposed fix changes MoE math or silently falls back to a different backend.

## Deliverables

Write results under:

```text
performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/
```

Include:

- `README.md`;
- commands used for each A/B;
- JSON/JSONL memory ledgers;
- any scripts used to summarize memory;
- raw logs or symlinks;
- code changes for opt-in instrumentation and any PoC;
- final go/no-go recommendation for prebuild and original-weight release.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read `prompts/target.md`,
`prompts/TARGET_08_radix_prefix_dsv4.md`, this file, and the TARGET 08.33
report.  Start by confirming the active MoE backend and adding opt-in memory
instrumentation around `prepare_moe_mxfp4_weights()` and real warmup
`model.forward()`.  Run only a single-bucket current-versus-grouped-FP4 A/B
until the Marlin WNA16 cache hypothesis is confirmed or rejected.
