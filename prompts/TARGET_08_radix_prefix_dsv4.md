# TARGET 08: DSV4 Radix Prefix Cache And SWA Lifecycle Roadmap

## Status

Closed as the DSV4 prefix-cache and SWA-lifecycle history line.

Active work should not start from TARGET 08 child prompts by default.  The
fine-grained prompts are archived under:

```text
prompts/archive/target08/
```

Use this root file as the main reference for new Codex threads.  Open archived
prompts only when a specific historical stop rule, command, or design detail is
needed.

## Why This Target Existed

TARGET 07 beat the old vLLM baseline on the fixed no-prefix benchmark.  TARGET
08 turned that speed-focused path into a more serving-like system by adding:

- DSV4 radix prefix cache;
- Route-B component ownership so compressed/indexer/component state can outlive
  or differ from full token locations;
- SGLang-inspired metadata lifetime rules;
- SWA independent lifecycle and high-capacity experiments;
- Marlin WNA16 original-weight release and safe capacity reuse;
- prefix/SWA metadata deforestation and graph replay cleanup.

The rule throughout TARGET 08 was: prefer SGLang's mature ownership/lifetime
design when available, and use mini-specific changes only when they preserve the
same invariants with less machinery.

## Promoted Prefix Baseline

Milestone tag:

```text
dsv4-sm80-prefix-routeb-lifetime-baseline
```

Promoted prefix preset:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
```

Runtime shape:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--page-size 256
--num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Prefix cache remains an explicit feature/preset rather than an unconditional
default because no-hit serving workloads still pay overhead.  For future work
that touches scheduler, cache metadata, graph buffers, eviction, SWA, or DSV4
component ownership, this prefix route should still be considered an important
baseline.

## Main Results

TARGET 08.30 closed the first prefix-cache baseline:

- text smoke/verifier passed;
- CUDA graph replay stayed zero-eager for the measured buckets;
- `prefix_multi_112req_wave16` improved from `51.0507` to `110.1417` output
  tok/s and saved `49152` prefill tokens;
- no-hit `4096/1024/bs4` stayed close to TARGET 07 control:
  `137.1625` versus `139.8415` output tok/s;
- no-hit `serving_mixed_112req_wave16` still paid opt-in overhead:
  `163.3985` versus `178.3004` output tok/s.

TARGET 08.31-08.48 then reopened the route for SWA independent lifecycle and
Marlin release capacity work:

- SWA independent lifecycle was implemented and contract-audited.
- Large-capacity serving bugs were traced to concrete ownership/addressing
  issues rather than treated as random CUDA failures.
- The Engine/KV dummy-token contract was fixed so dummy rows map to the SWA
  dummy page instead of real negative SWA locations.
- Stale prefix-handle tombstones and same-Engine Marlin-release/SWA address
  reuse bugs were fixed.
- The SWA lifecycle contract is documented in
  `prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md`.

TARGET 08.34-08.40 resolved Marlin WNA16 release as a capacity feature:

- Marlin WNA16 prebuild/release can recover about `17 GiB/rank` of original
  routed expert weight storage.
- The unsafe reuse symptom was eventually attributed to uninitialized DSV4
  component-cache reads after allocator reuse of old raw expert ranges.
- Clearing component slots on page allocation made unguarded release pass.
- The preferred long-term model is: prebuild owned backend caches, clear newly
  allocated component slots, then allow KV/component arenas to reuse released
  weight capacity safely.

TARGET 08.49-08.55 reduced the remaining SWA/prefix metadata overhead:

- `MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1` cut
  `dsv4.prepare.decode.attention_metadata` on `serving_mixed_112req_wave16`
  from `2968.232 ms` to `1381.333 ms`.
- `MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1` bypassed decode full SWA
  page-table materialization structurally, with small macro gains.
- Later attribution showed graph staging/copy bytes were not the main owner;
  captured decode forward and small extra kernels mattered more.
- Direct replay metadata fusion cleaned the primary SWA direct replay metadata
  gap to a reasonable level.  Do not keep shaving tiny metadata kernels unless a
  fresh profile makes them top bottlenecks again.

## Important Correctness Decisions

- `page_size=256` remains the DSV4 benchmark/default assumption.
- Radix prefix cache requires `page_size % 128 == 0` for the current DSV4
  compressed/SWA assumptions.
- Batch-slot invariance is not guaranteed as a system contract for now; token
  output sanity and request-local correctness are the relevant smoke gates.
- SWA independent lifecycle must follow the contract in
  `prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md`.
- Prefix/SWA/component ownership bugs should be fixed at the ownership metadata
  layer first, not by weakening duplicate-free or address-safety guards.

## Archive Map

Useful archived milestones:

- `prompts/archive/target08/TARGET_08.10_dsv4_sm80_prefix_cache_serving_stability_promotion_gate.md`
- `prompts/archive/target08/TARGET_08.18_dsv4_sm80_prefix_cache_memory_ledger_go_nogo.md`
- `prompts/archive/target08/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md`
- `prompts/archive/target08/TARGET_08.27_dsv4_sm80_sglang_aligned_route_b_metadata_lifetime.md`
- `prompts/archive/target08/TARGET_08.30_dsv4_sm80_post_prefix_reprofile_next_bottleneck.md`
- `prompts/archive/target08/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md`
- `prompts/archive/target08/TARGET_08.40_dsv4_sm80_marlin_wna16_release_component_clear_promotion.md`
- `prompts/archive/target08/TARGET_08.45_dsv4_sm80_swa_independent_lifecycle_contract.md`
- `prompts/archive/target08/TARGET_08.48_dsv4_sm80_marlin_swa_auto_cross_case_lifecycle_fix.md`
- `prompts/archive/target08/TARGET_08.55_dsv4_sm80_compressed_metadata_boundary_replay_cleanup.md`

The archive is historical source material, not the active todo list.

## Current Recommendation

TARGET 08 should remain closed unless a future feature changes prefix/SWA/cache
ownership.  TARGET 11 MTP was investigated after this route and is now paused
for release; it should not be treated as the active continuation of TARGET 08.

After MTP cleanup, validate this prefix/SWA baseline with a short non-MTP soak:
text sanity, graph replay health, `serving_mixed_112req_wave16`, and
`prefix_multi_112req_wave16` on the promoted TARGET 10 preset.

If future profiling reopens TARGET 08, start from a fresh attribution on the
current promoted baseline rather than replaying old metadata-cleanup prompts.
