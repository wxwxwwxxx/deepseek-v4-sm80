# TARGET 08.39: DSV4 SM80 Marlin WNA16 Old-Address Root Cause

## Status

Active TARGET 08 release-route root-cause target after TARGET 08.38.

TARGET 08.38 proved that a `before_kv_alloc` Marlin WNA16 raw-expert release
can be made useful with a `3.1875 GiB/rank` guard arena: text smokes, graph
replay, and historical 4096x128 / 4096x1024 macro runs passed, while auto KV
planning improved from `1,826` to `2,602` pages.

However, this is still a guarded diagnostic candidate, not the desired final
fix.  The old unguarded immediate release remains unsafe, and the guard does
not explain which stale owner, uninitialized read, stream-lifetime issue, or
kernel boundary makes released expert-weight addresses dangerous when
KV/component tensors reuse them.

## Goal

Find and fix the root cause that makes unguarded early release unsafe.

The preferred end state is:

```text
Prebuild Marlin WNA16 caches.
Release original routed expert weights before KV/component allocation.
Let KV/component buffers safely reuse the formerly raw expert-weight address
ranges.
Recover the full raw-expert release capacity, minus only principled allocator
metadata/safety costs.
```

Do not spend this target primarily on shrinking the guard size.  Guard-size
sweeps are allowed only as diagnostics.  The main objective is to identify the
first owner or kernel boundary where old expert addresses become semantically
unsafe, then remove that lifecycle bug.

## Background

Required starting reports:

```text
performance_milestones/target08_marlin_wna16_release_correctness_attribution/README.md
performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/README.md
performance_milestones/target08_marlin_wna16_safe_release_arena_capacity/README.md
```

Important prior evidence:

- TARGET 08.36: baseline and prebuild-only pass; release fails in both graph
  and eager/no-graph, so CUDA graph replay is not the primary cause.
- TARGET 08.36: `force-prepacked raw-present`, `keep-hidden-ref`, and
  `release-after-capture` pass; `weights-only` release fails and `scales-only`
  release passes.
- TARGET 08.36: partial release threshold landed between about `3.1875` and
  `6.375 GiB/rank`.
- TARGET 08.37: immediate release becomes unsafe when DSV4 KV/component
  allocations reuse freed raw-expert ranges.
- TARGET 08.37: releasing after KV allocation is correctness-clean but
  capacity-neutral.
- TARGET 08.37: freed-block quarantine passes, so the failure follows address
  ownership/reuse rather than a simple need to retain raw weight contents.
- TARGET 08.38: `before_kv_alloc` release plus a `3.1875 GiB/rank` guard arena
  passes short text smokes, graph replay, and historical macros.
- TARGET 08.38: owner ledgers prove that KV/component/logit allocations still
  reuse non-guarded released ranges; safe-arena is not merely avoiding all
  allocator reuse.
- TARGET 08.38: old-address guard mutation records were zero, but guard checks
  used a live deterministic guard and therefore did not prove why unguarded
  reuse corrupts text.

The observed `3.1875 GiB/rank` guard is currently tied to the release-ledger
order, not to a proven semantic model boundary.  In the TARGET 08.38 rank-0
safe-arena records it is exactly the first 32 released items:

```text
layers 0-7, each with:
  w13_weight             268,435,456 bytes
  w13_weight_scale_inv    16,777,216 bytes
  w2_weight              134,217,728 bytes
  w2_weight_scale_inv      8,388,608 bytes
per layer total          427,819,008 bytes
8 layers total         3,422,552,064 bytes = 3.1875 GiB
```

This relation must be verified across ranks/runs.  Do not assume layer 0-7 is
the real broken semantic region until a targeted probe proves it.

## Working Hypotheses

Keep these hypotheses separate in the report:

1. **Stale old-address read/write**
   - some Python attribute, cached pointer, Marlin/MoE state, graph buffer, or
     custom kernel still touches raw expert-weight addresses after release;
   - if it writes, live guards should mutate;
   - if it reads, hidden-ref poison or poison-then-free should expose it.
2. **Uninitialized KV/component read**
   - `torch.empty` KV/component tensors may reuse old expert storage whose
     contents are arbitrary;
   - the failing path may read a C4/C128/indexer/compress-state/logit region
     before the legitimate writer initializes it;
   - a live guard would pass because it prevents that tensor from occupying the
     dangerous old-content range.
3. **Stream/lifetime reuse bug**
   - release/reuse occurs before all kernels using the old storage have
     completed, or storage was recorded on the wrong stream;
   - explicit sync or `record_stream` changes may make unguarded release pass.
4. **Allocator/block-order dependent alias**
   - the guard protects a block shape/order, not a semantic layer;
   - a different release order, layer filter, or allocator warmup may move the
     failure threshold.
5. **True kernel out-of-bounds write**
   - a custom attention/indexer/MoE/cache kernel writes outside its intended
     output and corrupts whichever tensor occupies a nearby released block.

## Required Investigation

### 1. Guard-Range Census

Build a small analyzer for TARGET 08.38 raw ledgers and new runs:

- map each guard tensor to `source_released_item`;
- summarize by rank, layer, component, bytes, original address, guard address,
  and allocator reuse order;
- verify whether `3.1875 GiB` is always layers `0-7` or only rank/run/order
  dependent;
- compare failing unguarded owner overlaps against guarded owner overlaps.

Deliverable:

```text
performance_milestones/target08_marlin_wna16_old_address_root_cause/guard_range_census.md
```

### 2. Old Expert Region Trap Modes

Implement explicit trap modes.  These may be debug env flags only; they should
not become default runtime behavior.

Required modes:

1. **Live guard mutation trap**
   - allocate guard tensors over released ranges;
   - fill with deterministic bytes and, where dtype/interpretation allows,
     NaN-like bit patterns;
   - check after every important stage;
   - this extends TARGET 08.38 and should confirm whether any kernel writes
     into guarded old-address ranges.
2. **Poison-then-free trap**
   - allocate over the old expert ranges;
   - fill with strong NaN / signaling byte patterns;
   - free those poison tensors before KV/component allocation;
   - let KV/component tensors reuse those same addresses;
   - determine whether corruption becomes deterministic or earlier.
   - This probe is especially important for the uninitialized-read hypothesis.
3. **KV-as-sentinel trap**
   - after KV/component allocation, identify tensors overlapping freed expert
     ranges;
   - fill selected overlapping owners with poison before the earliest expected
     legal writer;
   - checkpoint after append/indexer/compress/attention stages;
   - if a tensor is read before being overwritten, logits or internal checks
     should reveal the owner.
4. **Owner inversion trap**
   - deliberately protect different layer windows, for example layers `0-7`,
     `8-15`, `16-23`, `24-31`, `32-40`, while keeping the same guard byte
     budget when possible;
   - if only a semantic window protects correctness, focus there;
   - if any same-size window protects correctness, the issue is likely block
     order/content/lifetime rather than layer semantics.

### 3. Stage And Layer Bisection

Add enough instrumentation to locate the first failure boundary.

Required checkpoints:

- after release;
- after guard or poison allocation;
- after KV/component allocation;
- after page-table allocation;
- after attention backend init;
- after graph runner init;
- before and after warmup forward;
- before and after graph capture;
- before and after decode steps 1, 2, and 3;
- per layer around the first divergent token when feasible:
  - after input norm / attention;
  - after C4/C128/indexer metadata use;
  - after routed MoE;
  - after shared expert;
  - after residual/post blocks;
  - after logits.

Use rank-scoped logging and keep reports compact.  If full TP8 per-layer logs
are too noisy, add a short single-rank or limited-layer repro that preserves
the failing boundary.

### 4. Stream/Lifetime Controls

Run a small matrix before deeper kernel work:

- immediate release with explicit `torch.cuda.synchronize()` before release;
- explicit sync after release and before KV allocation;
- `CUDA_LAUNCH_BLOCKING=1` debug run;
- if relevant tensors can be identified, try `record_stream` or equivalent
  lifetime ownership on the stream where Marlin prebuild kernels use them;
- compare default stream and any auxiliary stream usage around Marlin prebuild,
  attention/indexer, and graph warmup.

If sync/lifetime controls fix the unguarded path, prioritize a principled
stream-lifetime fix over guard promotion.

### 5. Narrow Kernel Attribution

Only after the above narrows the boundary, use heavier tools:

- NVTX ranges around candidate layer/operator blocks;
- rank-scoped Nsight Systems trace for the smallest failing smoke;
- compute-sanitizer/memcheck only on a reduced repro, if runtime is manageable;
- focused microbench for the suspected kernel with poisoned inputs/outputs.

Candidate owner classes from previous ledgers:

- `kvcache.dsv4.c4_buffer`;
- `kvcache.dsv4.c128_buffer`;
- `kvcache.dsv4.c4_indexer_buffer`;
- `kvcache.dsv4.c4_indexer_fp8_paged_cache`;
- `kvcache.dsv4.layer*.compress_state.kv_score_buffer`;
- `kvcache.dsv4.layer*.indexer_state.kv_score_buffer`;
- `graph.capture_buffer.logits`;
- `engine.forward.logits`;
- layer-2 indexer/attention path, because TARGET 08.36 observed early symptom
  near `layer2.indexer_select.logits`.

## Fix Strategy

Implement the smallest fix justified by evidence:

- If the root is an uninitialized KV/component read, initialize exactly the
  affected tensor/row/window or fix the writer coverage.  Avoid blanket
  multi-GiB zeroing unless the cost is measured and accepted.
- If the root is stale raw expert access, remove the stale pointer path or copy
  the required metadata into the Marlin packed-cache owner before release.
- If the root is stream/lifetime, add the correct synchronization or
  `record_stream` ownership at the producer/consumer boundary.
- If the root is out-of-bounds kernel behavior, fix the kernel/indexing and add
  a micro test that fails without the fix.
- If the root cannot be found after bounded evidence gathering, keep the
  TARGET 08.38 safe-arena path as a fallback candidate, but explicitly label it
  as empirical and preserve all root-cause evidence for the next pass.

## Validation

Minimum validation for any proposed root fix:

```text
pytest -q tests/engine/test_marlin_wna16_release_credit.py
pytest -q tests/engine/test_deepseek_v4_text_smoke.py  # or local equivalent
```

Then run TP8 smokes with page size `256`:

- prebuild-only baseline;
- old unsafe unguarded release control, expected to fail before the fix and
  pass after the fix if this target succeeds;
- fixed unguarded release;
- guarded safe-arena control from TARGET 08.38.

Required behavioral gates:

- text smoke passes in eager/no-graph and CUDA graph mode;
- no token-0 flood or obvious corrupted text;
- graph replay remains zero-eager for captured buckets;
- logit sanity does not show NaN/Inf explosion;
- Marlin WNA16 packed-cache integrity remains stable;
- owner ledger proves KV/component tensors may overlap formerly raw expert
  ranges in the fixed unguarded path.

Required capacity gates:

- report raw expert bytes released/rank;
- report guard bytes/rank, ideally zero for the final root fix;
- report net capacity credit/rank;
- report planned pages/tokens at page size `256`;
- compare against TARGET 08.38 safe-arena (`2,602` pages) and prebuild-only
  lower bound (`1,826` pages).

Run historical macros only after short correctness passes:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants <fixed_release_variant> \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --output-dir /tmp/dsv4_target0839_4096x128 \
  --keep-going
```

Optionally run 4096x1024 after 4096x128 passes.

## Stop Conditions

Stop and report if any of these happens:

- a concrete root owner is found and fixed, and unguarded release passes the
  required gates;
- a concrete root owner is found but the fix is too large/risky for this
  target, with a next target proposed;
- all trap modes show no mutation/stale access but poison-then-free proves an
  uninitialized-read class, with the suspected buffer and first read boundary
  documented;
- after two focused probe/fix iterations, no new evidence is produced.  In that
  case, do not keep sweeping guard size; summarize the evidence and recommend
  whether to temporarily promote TARGET 08.38 safe-arena or defer release.

## Deliverables

Write all outputs under:

```text
performance_milestones/target08_marlin_wna16_old_address_root_cause/
```

Required files:

- `README.md` with verdict, exact root hypothesis status, and next action;
- `guard_range_census.md`;
- `trap_mode_results.md`;
- `stage_layer_bisection.md`;
- `stream_lifetime_matrix.md`;
- `fix_summary.md` if code changes are made;
- `capacity_ledger.md`;
- raw logs / JSONL under `raw/`.

The README must answer:

1. Does `3.1875 GiB` correspond to a semantic model region or only to release
   order / allocator block behavior?
2. Which owner first makes old expert addresses unsafe?
3. Is the failure a stale old-address access, uninitialized KV/component read,
   stream-lifetime bug, allocator/block-order artifact, or kernel OOB bug?
4. Can unguarded release now pass while KV/component uses the recovered
   `~17.13 GiB/rank`?
5. If not, what is the shortest evidence-based next target?
